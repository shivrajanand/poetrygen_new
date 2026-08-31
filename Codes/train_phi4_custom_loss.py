import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import unsloth
from unsloth import FastLanguageModel
import torch
import torch.nn as nn
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from trl import SFTConfig, SFTTrainer
import json
import random
from collections import Counter

import torch.utils.checkpoint as ckpt
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

HYPERPARAMS = {
    "MODEL_NAME": "unsloth/phi-4",
    "MAX_LEN": 1536, #=3×512,  # Max len for prompt, hi, sa in gitapress final is 1529 
    "LOAD_IN_4BIT": True,
    "BATCH_SIZE": 16,
    "GRAD_ACC": 4,
    "EPOCHS": 5,
    "LR": 2.4e-4,
    "LOG_STEPS": 25,
    "SAVE_STEPS": 365, #Once per epoch
    "SAVE_LIMIT": 5, #Saves checkpoint for all 5 epcohs
    "EVAL_STEPS": 365, #Evaluates every epoch
    "WEIGHT_DECAY": 0.00465,
    "WARMUP_RATIO": 0.05,
    "MAX_GRAD_NORM": 1.0,

    "LORA_R": 32,
    "LORA_ALPHA": 64,
    "LORA_DROPOUT": 0.1,

    "ES_THRESHOLD": 0.001,
    "ES_PATIENCE": 5,

    "DATA_FILE_PATH": "Files/v3_gitapress_final.csv",
    "OUTPUT_DIR": "Trained_Models/Phi4-14B-customLoss",
}

os.makedirs(HYPERPARAMS["OUTPUT_DIR"], exist_ok=True)

METER_TO_ID = {
    "Anuṣṭubh": 0,
    "Vasantatilakā": 1,
    "Śārdūlavikrīḍita": 2,
    "Indravajrā": 3,
    "Sragdharā": 4,
    "Vaṃśastha": 5,
    "Śikhariṇī": 6,
    "Upendravajrā": 7,
    "Mālinī": 8,
    "Śālinī": 9,
}
ID_TO_METER = {v: k for k, v in METER_TO_ID.items()}
NUM_METERS = len(METER_TO_ID)  # M = 10

print(ID_TO_METER)
print(NUM_METERS)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=HYPERPARAMS["MODEL_NAME"],
    max_seq_length=HYPERPARAMS["MAX_LEN"],
    load_in_4bit=HYPERPARAMS["LOAD_IN_4BIT"]
)

model = FastLanguageModel.get_peft_model(
    model,
    r=HYPERPARAMS["LORA_R"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=HYPERPARAMS["LORA_ALPHA"],
    lora_dropout=HYPERPARAMS["LORA_DROPOUT"],
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

if tokenizer.pad_token_id is None:
    # Needed because we now pad per-batch ourselves (packing is disabled, see below)
    tokenizer.pad_token = tokenizer.eos_token

tokenizer = get_chat_template(tokenizer, chat_template="phi-4")

ds = load_dataset('csv', data_files=HYPERPARAMS["DATA_FILE_PATH"])["train"]

train_ds = ds.filter(lambda x: x["split"] == "train")
val_ds = ds.filter(lambda x: x["split"] == "val")
print(f"Train: {len(train_ds)}")
print(f"Val: {len(val_ds)}")

unique_meters_train = sorted(set(train_ds["meter_cd"]))
print("--------------------------------------------------------------------------------------")
print("Unique meter_cd values found in TRAIN split:")
for m in unique_meters_train:
    print(f"  '{m}'")
print("--------------------------------------------------------------------------------------")

unknown_meters = [m for m in unique_meters_train if m not in METER_TO_ID]
if unknown_meters:
    raise ValueError(
        f"Found meter_cd values in the training split that are not in METER_TO_ID: {unknown_meters}. "
        f"Fix METER_TO_ID (exact string match, including diacritics) before proceeding."
    )

# Also verify val doesn't contain meters we have no id for (would break meter_id lookup)
unknown_val_meters = [m for m in set(val_ds["meter_cd"]) if m not in METER_TO_ID]
if unknown_val_meters:
    raise ValueError(f"Found meter_cd values in the val split not in METER_TO_ID: {unknown_val_meters}")

meter_counts = Counter(train_ds["meter_cd"])  # computed from actual data, not hardcoded

weights_raw = {}
for meter_name, meter_id in METER_TO_ID.items():
    n_m = meter_counts.get(meter_name, 0)
    if n_m <= 0:
        raise ValueError(
            f"Meter '{meter_name}' has zero examples in the training split; "
            f"1/sqrt(N_m) is undefined. Check METER_TO_ID / the dataset."
        )
    weights_raw[meter_name] = 1.0 / (n_m ** 0.5)

mean_raw = sum(weights_raw.values()) / NUM_METERS  # (1/M) * sum_j w_j^raw

meter_weights = {m: (weights_raw[m] / mean_raw) for m in METER_TO_ID}

print("--------------------------------------------------------------------------------------")
print(f"{'Meter':<20}{'Count':<10}{'Weight':<10}")
print("-" * 50)
for meter_name, meter_id in sorted(METER_TO_ID.items(), key=lambda kv: kv[1]):
    print(f"{meter_name:<20}{meter_counts.get(meter_name, 0):<10}{meter_weights[meter_name]:<10.4f}")
print("-" * 50)
print(f"Mean weight (should be 1.0): {sum(meter_weights.values()) / NUM_METERS:.6f}")
print("--------------------------------------------------------------------------------------")

# Ordered tensor, index == meter_id
meter_weight_tensor = torch.zeros(NUM_METERS, dtype=torch.float32)
for meter_name, meter_id in METER_TO_ID.items():
    meter_weight_tensor[meter_id] = meter_weights[meter_name]
    
    
def format_and_tokenize(example):
    system_msg = {"role": "system", "content": example["prompt"]}
    user_msg = {"role": "user", "content": f"Meaning:\n{example['hi']}\n\nGenerate the Sanskrit verse.\n"}
    assistant_msg = {"role": "assistant", "content": example["sa"]}

    full_convo = [system_msg, user_msg, assistant_msg]
    prompt_only_convo = [system_msg, user_msg]

    full_text = tokenizer.apply_chat_template(
        full_convo, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_only_convo, tokenize=False, add_generation_prompt=True
    )

    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Sanity check that the prompt is a true prefix of the full sequence.
    prefix_len = len(prompt_ids)
    if input_ids[:prefix_len] != prompt_ids:
        # Fall back defensively: mask nothing extra, but flag it loudly.
        raise ValueError(
            "Tokenized prompt is not a prefix of the tokenized full conversation. "
            "Chat template / tokenizer mismatch — inspect this example before training."
        )

    if len(input_ids) > HYPERPARAMS["MAX_LEN"]:
        input_ids = input_ids[:HYPERPARAMS["MAX_LEN"]]

    labels = list(input_ids)
    mask_len = min(prefix_len, len(labels))
    for i in range(mask_len):
        labels[i] = -100

    attention_mask = [1] * len(input_ids)
    meter_id = METER_TO_ID[example["meter_cd"]]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "meter_id": meter_id,
    }


keep_cols = ["input_ids", "attention_mask", "labels", "meter_id"]

train_ds_tok = train_ds.map(format_and_tokenize, batched=False)
val_ds_tok = val_ds.map(format_and_tokenize, batched=False)

train_ds_tok = train_ds_tok.remove_columns(
    [c for c in train_ds_tok.column_names if c not in keep_cols]
)
val_ds_tok = val_ds_tok.remove_columns(
    [c for c in val_ds_tok.column_names if c not in keep_cols]
)

print("--------------------------------------------------------------------------------------")
print("Sample tokenized/labeled/meter-tagged training examples:")
sample_idx = random.sample(range(len(train_ds)), k=min(3, len(train_ds)))
for idx in sample_idx:
    row = train_ds[idx]
    print(f"meter_cd: {row['meter_cd']}")
    print(f"meter_id: {METER_TO_ID[row['meter_cd']]}")
    print(f"prompt/message (hi): {row['hi']}")
    print(f"target (sa): {row['sa']}")
    print()
print("--------------------------------------------------------------------------------------")

class MeterWeightedCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_meter_ids = []

        for f in features:
            ids = f["input_ids"]
            mask = f["attention_mask"]
            labels = f["labels"]
            pad_n = max_len - len(ids)

            batch_input_ids.append(ids + [self.pad_id] * pad_n)
            batch_attention_mask.append(mask + [0] * pad_n)
            batch_labels.append(labels + [-100] * pad_n)
            batch_meter_ids.append(f["meter_id"])

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "meter_ids": torch.tensor(batch_meter_ids, dtype=torch.long),
        }
        

base_model = model.base_model.model

class MeterWeightedSFTTrainer(SFTTrainer):
    def __init__(self, *args, meter_weight_tensor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.meter_weight_tensor = meter_weight_tensor

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        meter_ids = inputs.pop("meter_ids")
        labels = inputs["labels"]
        
        outputs = model.base_model.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
        logits = outputs.logits

        # Standard causal-LM shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())  # [B, T-1]

        valid_mask = (shift_labels != -100).float()  # [B, T-1]

        # L_i^CE = (1/T_i) * sum_t CE_{i,t}
        sequence_loss = (token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)

        # w_{m_i}
        weights = self.meter_weight_tensor.to(sequence_loss.device, sequence_loss.dtype)[meter_ids]

        # L_i = w_{m_i} * L_i^CE ; L_batch = mean_i L_i
        weighted_sequence_loss = sequence_loss * weights
        loss = weighted_sequence_loss.mean()

        return (loss, outputs) if return_outputs else loss
    
    
trainer = MeterWeightedSFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds_tok,
    eval_dataset=val_ds_tok,
    data_collator=MeterWeightedCollator(tokenizer),
    meter_weight_tensor=meter_weight_tensor,
    args=SFTConfig(
        output_dir=HYPERPARAMS["OUTPUT_DIR"],

        per_device_train_batch_size=HYPERPARAMS["BATCH_SIZE"],
        gradient_accumulation_steps=HYPERPARAMS["GRAD_ACC"],
        num_train_epochs=HYPERPARAMS["EPOCHS"],

        learning_rate=HYPERPARAMS["LR"],
        lr_scheduler_type="cosine",
        warmup_ratio=HYPERPARAMS["WARMUP_RATIO"],
        weight_decay=HYPERPARAMS["WEIGHT_DECAY"],
        max_grad_norm=HYPERPARAMS["MAX_GRAD_NORM"],

        logging_steps=HYPERPARAMS["LOG_STEPS"],
        logging_strategy="steps",
        logging_dir=HYPERPARAMS["OUTPUT_DIR"] + "/runs",
        report_to="tensorboard",

        save_steps=HYPERPARAMS["SAVE_STEPS"],
        save_total_limit=HYPERPARAMS["SAVE_LIMIT"],

        eval_strategy="steps",
        eval_steps=HYPERPARAMS["EVAL_STEPS"],

        fp16=False,
        bf16=True,
        
        # max_steps=30,

        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        optim="adamw_8bit",
        seed=3407,

        # Required so the `meter_id` metadata column survives into the collator.
        remove_unused_columns=False,
        # We already tokenized/labeled the dataset ourselves above.
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    ),
)

trainer.train()

trainer.save_model(HYPERPARAMS["OUTPUT_DIR"] + "/final_model")
tokenizer.save_pretrained(HYPERPARAMS["OUTPUT_DIR"] + "/final_model")

print("BEST MODEL STATS")
print(trainer.state.best_model_checkpoint)
print(trainer.state.best_metric)

essential_config = {
    "HYPER-PARAMETERS": HYPERPARAMS,
    "TRAIN_DATASET_LEN": len(train_ds_tok),
    "VAL_DATASET_LEN": len(val_ds_tok),
    "METER_TO_ID": METER_TO_ID,
    "METER_COUNTS_TRAIN": {m: meter_counts.get(m, 0) for m in METER_TO_ID},
    "METER_WEIGHTS": meter_weights,
    "best_model": {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_model_metric": trainer.state.best_metric,
    },
}

with open(HYPERPARAMS["OUTPUT_DIR"] + "/essential_config.json", "w", encoding="utf-8") as f:
    json.dump(essential_config, f, indent=4, default=str, ensure_ascii=False)