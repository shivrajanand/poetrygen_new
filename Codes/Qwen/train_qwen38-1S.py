# This script has used logic from unsloth official notebook for Qwen 3.8 27B model
# LINK: https://www.kaggle.com/notebooks/welcome?src=https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3.8_(27B)-Conversational.ipynb&accelerator=nvidiaTeslaT4

import unsloth
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only
from trl import SFTConfig, SFTTrainer
import json
from transformers import EarlyStoppingCallback
import random

import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

HYPERPARAMS = {
    "MODEL_NAME": "unsloth/Qwen3.8-27B",
    "MAX_LEN": 1200, #based on token length analysis
    "LOAD_IN_4BIT": True,
    "BATCH_SIZE":16,
    "GRAD_ACC": 2,
    "EPOCHS": 5,
    "LR": 2.4e-4,
    "LOG_STEPS": 10,
    "SAVE_STEPS": 100,
    "SAVE_LIMIT": 3,
    "EVAL_STEPS": 100,
    "WEIGHT_DECAY": 0.00465,
    "WARMUP_RATIO": 0.05,
    "MAX_GRAD_NORM": 1.0,

    "LORA_R": 32,
    "LORA_ALPHA": 64,
    "LORA_DROPOUT": 0.1,
    
    "ES_THRESHOLD": 0.001,
    "ES_PATIENCE": 5,
    
    "DATA_FILE_PATH": "Files/v3_gitapress_final_1shot_prompts.csv",
    "OUTPUT_DIR": "Trained_Models/Qwen/Qwen38-27B-1shotTrained",

}

os.makedirs(HYPERPARAMS["OUTPUT_DIR"], exist_ok=True)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=HYPERPARAMS["MODEL_NAME"],
    max_seq_length=HYPERPARAMS["MAX_LEN"],
    load_in_4bit=HYPERPARAMS["LOAD_IN_4BIT"],
    full_finetuning = False
)

model = FastLanguageModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Reaches the linear_attn (gated delta) modules too
    finetune_mlp_modules       = True,  # Should leave on always!
    
    r=HYPERPARAMS["LORA_R"],
    lora_alpha=HYPERPARAMS["LORA_ALPHA"],
    lora_dropout=HYPERPARAMS["LORA_DROPOUT"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

ds = load_dataset('csv', data_files=HYPERPARAMS["DATA_FILE_PATH"])["train"]

train_ds = ds.filter(lambda x: x["split"] == "train")
val_ds = ds.filter(lambda x: x["split"] == "val")
print(f"Train: {len(train_ds)}")
print(f"Val: {len(val_ds)}")


from collections import Counter
train_dist = Counter(train_ds["meter_cd"])
val_dist = Counter(val_ds["meter_cd"])
print("Train distribution:")
for meter, count in sorted(train_dist.items()):
    print(f"{meter}: {count}")
print("\nValidation distribution:")
for meter, count in sorted(val_dist.items()):
    print(f"{meter}: {count}")


def format_and_tokenize(batch):
    texts = []

    for prompt, hi, sa in zip(
        batch["prompt"], batch["hi"], batch["sa"]
    ):
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Meaning:\n{hi}\n\nGenerate the Sanskrit verse.",
            },
            {"role": "assistant", "content": sa},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
            preserve_thinking=False,
        )

        text = text.replace("<think>\n\n</think>\n\n", "")

        texts.append(text)

    return {"text": texts}


train_ds = train_ds.map(format_and_tokenize, batched=True)
val_ds = val_ds.map(format_and_tokenize, batched=True)

print("--------------------------------------------------------------------------------------")
for key in train_ds[0].keys():
    print(key)
    print(train_ds[0][key])
    print("--------------------------------------------------------------------------------------")
    print()
print("--------------------------------------------------------------------------------------")


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    dataset_text_field="text",
    max_seq_length=HYPERPARAMS["MAX_LEN"],
    packing=True,
    dataset_num_proc=1,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=HYPERPARAMS["ES_PATIENCE"], early_stopping_threshold = HYPERPARAMS["ES_THRESHOLD"])],
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

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # max_steps=10,
        optim="adamw_8bit",
        seed=3407,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
    )

trainer.train()

trainer.save_model(HYPERPARAMS["OUTPUT_DIR"] + "/final_model")
tokenizer.save_pretrained(HYPERPARAMS["OUTPUT_DIR"] + "/final_model")

print("BEST MODEL STATS")
print(trainer.state.best_model_checkpoint)
print(trainer.state.best_metric)

essential_config = {
    "HYPER-PARAMETERS": HYPERPARAMS,
    "TRAIN_DATASET_LEN": len(train_ds),
    "VAL_DATASET_LEN": len(val_ds),
    "best_model": {"best_model_checkpoint": trainer.state.best_model_checkpoint,
    "best_model_metric": trainer.state.best_metric}
}

with open(HYPERPARAMS["OUTPUT_DIR"]+"/essential_config.json", "w", encoding="utf-8") as f:
    json.dump(essential_config, f, indent=4, default=str)