import os
import json
import random

import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback

import warnings

warnings.filterwarnings("ignore")

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


HYPERPARAMS = {
"MODEL_NAME": "unsloth/gemma-4-31B-it",
"MAX_LEN": 768,
"LOAD_IN_4BIT": True,

"BATCH_SIZE": 8,
"GRAD_ACC": 12,
"EPOCHS": 10,

"LR": 1.0385e-4,            

"LOG_STEPS": 50,
"SAVE_STEPS": 100,
"SAVE_LIMIT": 3,
"EVAL_STEPS": 100,

"WEIGHT_DECAY": 0.01,
"WARMUP_RATIO": 0.03,
"MAX_GRAD_NORM": 1.0,

"LORA_R": 16,
"LORA_ALPHA": 32,
"LORA_DROPOUT": 0.0312,

"ES_THRESHOLD": 0.001,
"ES_PATIENCE": 8,

"DATA_FILE_PATH": "Files/v3_gitapress_final.csv",
"OUTPUT_DIR": "Trained_Models/Gemma4-31B-0shot",
}

os.makedirs(HYPERPARAMS["OUTPUT_DIR"], exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load model + LoRA (text-only -> FastModel, not FastVisionModel)
# ---------------------------------------------------------------------------
model, tokenizer = FastModel.from_pretrained(
    model_name=HYPERPARAMS["MODEL_NAME"],
    max_seq_length=HYPERPARAMS["MAX_LEN"],
    load_in_4bit=HYPERPARAMS["LOAD_IN_4BIT"],
    full_finetuning=False,
)

model = FastModel.get_peft_model(
    model,
    r=HYPERPARAMS["LORA_R"],
    lora_alpha=HYPERPARAMS["LORA_ALPHA"],
    lora_dropout=HYPERPARAMS["LORA_DROPOUT"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
    loftq_config=None,

    finetune_vision_layers=False,      # text-only task
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    target_modules="all-linear",
)

tokenizer = get_chat_template(tokenizer, "gemma-4")


# ---------------------------------------------------------------------------
# 2. Load + split data
# ---------------------------------------------------------------------------
ds = load_dataset("csv", data_files=HYPERPARAMS["DATA_FILE_PATH"])["train"]

train_ds = ds.filter(lambda x: x["split"] == "train")
val_ds = ds.filter(lambda x: x["split"] == "val")

print(f"Train: {len(train_ds)}")
print(f"Val: {len(val_ds)}")


# ---------------------------------------------------------------------------
# 3. Convert each ROW (not batch) into conversation format
# ---------------------------------------------------------------------------
def convert_to_text(sample):
    messages = [
        {"role": "system", "content": sample["prompt"]},
        {
            "role": "user",
            "content": f"Meaning:\n{sample['hi']}\n\nGenerate the Sanskrit verse.\n",
        },
        {"role": "assistant", "content": sample["sa"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ).removeprefix("<bos>")  # processor/tokenizer adds <bos> again at train time

    return {"text": text}


train_dataset = train_ds.map(
    convert_to_text,
    remove_columns=train_ds.column_names,
    num_proc=os.cpu_count(),
)

val_dataset = val_ds.map(
    convert_to_text,
    remove_columns=val_ds.column_names,
    num_proc=os.cpu_count(),
)

print("--------------------------------------------------------------------------------------")
print("0th data point in train set")
print("--------------------------------------------------------------------------------------")
print(train_dataset[0])
print("--------------------------------------------------------------------------------------")


INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"


# ---------------------------------------------------------------------------
# 5. Trainer
# ---------------------------------------------------------------------------
FastModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=HYPERPARAMS["ES_PATIENCE"],
            early_stopping_threshold=HYPERPARAMS["ES_THRESHOLD"],
        )
    ],
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

        save_strategy="steps",
        save_steps=HYPERPARAMS["SAVE_STEPS"],
        save_total_limit=HYPERPARAMS["SAVE_LIMIT"],

        eval_strategy="steps",
        eval_steps=HYPERPARAMS["EVAL_STEPS"],

        fp16=False,
        bf16=True,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        optim="adamw_8bit",
        seed=3407,
        remove_unused_columns=True,
        dataset_text_field="text",
        max_length=HYPERPARAMS["MAX_LEN"],
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part=INSTRUCTION_PART,
    response_part=RESPONSE_PART,
    num_proc=1,
)

trainer_stats = trainer.train(resume_from_checkpoint="/home/shivraj-pg/poetrygen_new/Trained_Models/Gemma4-31B-0shot-Run1-Epochs1TO5/checkpoint-1220")
trainer.save_state()

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
final_dir = HYPERPARAMS["OUTPUT_DIR"] + "/final_model"
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)

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

with open(HYPERPARAMS["OUTPUT_DIR"] + "/essential_config.json", "w", encoding="utf-8") as f:
    json.dump(essential_config, f, indent=4, default=str)