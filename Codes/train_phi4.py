import unsloth
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only
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
    "MODEL_NAME": "unsloth/phi-4",
    "MAX_LEN": 3750, #based on token length analysis
    "LOAD_IN_4BIT": True,
    "BATCH_SIZE":12,
    "GRAD_ACC": 8,
    "EPOCHS": 5,
    "LR": 3.4e-4,
    "LOG_STEPS": 200,
    "SAVE_STEPS": 200,
    "SAVE_LIMIT": 3,
    "EVAL_STEPS": 200,
    "WEIGHT_DECAY": 0.03,
    "WARMUP_RATIO": 0.04,

    "LORA_R": 64,
    "LORA_ALPHA": 64,
    "LORA_DROPOUT": 0.05,
    
    "ES_THRESHOLD": 0.001,
    "ES_PATIENCE": 5,
    
    "DATA_FILE_PATH": "Files/v3_gitapress_final_5shot_prompts.csv",
    "OUTPUT_DIR": "Trained_Models/Phi4-14B-DEV-5SHOT",

}


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=HYPERPARAMS["MODEL_NAME"],
    max_seq_length=HYPERPARAMS["MAX_LEN"],
    load_in_4bit=HYPERPARAMS["LOAD_IN_4BIT"]
)

model = FastLanguageModel.get_peft_model(
    model,
    r=HYPERPARAMS["LORA_R"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=HYPERPARAMS["LORA_ALPHA"],
    lora_dropout=HYPERPARAMS["LORA_DROPOUT"],
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

ds = load_dataset('csv', data_files=HYPERPARAMS["DATA_FILE_PATH"])["train"]
train_ds = ds.filter(lambda x: x["split"] == "train")
val_ds = ds.filter(lambda x: x["split"] == "val")
print(f"Train: {len(train_ds)}")
print(f"Val: {len(val_ds)}")

def format_and_tokenize(batch):
    texts = []
    
    for prompt, hi, sa in zip(batch["prompt"], batch["hi"], batch["sa"]):
        convo = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Meaning:\n{hi}\n\nGenerate the Sanskrit verse.\n"},
            {"role": "assistant", "content": sa},
        ]
        
        text = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        
        texts.append(text)
    
    return {"text": texts}

tokenizer = get_chat_template(tokenizer, chat_template="phi-4")

train_ds = train_ds.map(format_and_tokenize, batched=True)
val_ds = val_ds.map(format_and_tokenize, batched=True)

print("--------------------------------------------------------------------------------------")
for key in train_ds[0].keys():
    print(key)
    print(train_ds[0][key])
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

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # max_steps=30,
        optim="adamw_8bit",
        seed=3407,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user<|im_sep|>",
    response_part="<|im_start|>assistant<|im_sep|>",
)

trainer.train()

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