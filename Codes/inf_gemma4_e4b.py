import torch
import pandas as pd
from unsloth import FastVisionModel
from unsloth.chat_templates import get_chat_template
from tqdm import tqdm
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
# =========================
# PATHS
# =========================
BASE_MODEL = "unsloth/gemma-4-E4B-it"
INPUT_CSV = "Files/v3_gitapress_final_anustubh.csv"

MAX_LEN = 768          # must match training's MAX_LEN
LOAD_IN_4BIT = True    # must match training's LOAD_IN_4BIT
MAX_NEW_TOKENS = 128
SAVE_FREQUENCY = 1
BATCH_SIZE = 20
LORA_PATH = "Trained_Models/Gemma4-E4B-Anustubh/checkpoint-1000"
OUTPUT_CSV = "Outputs/unsloth_gemma4-e4b_FT_anustubh-c1000.csv"

print("Test File in use: ", INPUT_CSV)

if 'UT' in OUTPUT_CSV:
    SET_LORA = False
elif 'FT' in OUTPUT_CSV:
    SET_LORA = True
else:
    print("Mention FT: Finetuned or UT: Untrained in OUTPUT CSV")
    sys.exit(0)

if SET_LORA:
    LORA_PATH = LORA_PATH
else:
    LORA_PATH = ""

# =========================
# Existing file warning
# =========================

if os.path.exists(OUTPUT_CSV):
    choice = input(
        f"{OUTPUT_CSV} already exists.\n"
        "Do you want to overwrite it? (y/n): "
    ).strip().lower()

    if choice not in ("y", "yes"):
        print("Exiting without overwriting.")
        sys.exit(0)

model_name = LORA_PATH if SET_LORA else BASE_MODEL

model, processor = FastVisionModel.from_pretrained(
    model_name=model_name,
    max_seq_length=MAX_LEN,
    load_in_4bit=LOAD_IN_4BIT,
)

processor = get_chat_template(processor, "gemma-4")

if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

processor.tokenizer.padding_side = "left"

print(f"Model used is {BASE_MODEL}", end="")
if SET_LORA:
    print(f" with LoRA checkpoint = {LORA_PATH}", end="")
print()

FastVisionModel.for_inference(model)

# =========================
# GENERATION
# =========================
def generate_batch(batch_df):

    per_example_inputs = []

    for _, row in batch_df.iterrows():
        msg = [
            {
                "role": "system",
                "content": [{"type": "text", "text": row["prompt"]}],
            },
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"Meaning:\n{row['hi']}\n\nGenerate the Sanskrit verse.\n",
                }],
            },
        ]

        example_inputs = processor.apply_chat_template(
            msg,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
        per_example_inputs.append(example_inputs)

    input_ids_list = [ex["input_ids"][0] for ex in per_example_inputs]
    attention_mask_list = [ex["attention_mask"][0] for ex in per_example_inputs]

    max_len = max(ids.shape[0] for ids in input_ids_list)
    pad_id = processor.tokenizer.pad_token_id

    padded_input_ids = []
    padded_attention_mask = []
    for ids, mask in zip(input_ids_list, attention_mask_list):
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            pad_ids = torch.full((pad_len,), pad_id, dtype=ids.dtype)
            pad_mask = torch.zeros((pad_len,), dtype=mask.dtype)
            ids = torch.cat([pad_ids, ids], dim=0)
            mask = torch.cat([pad_mask, mask], dim=0)
        padded_input_ids.append(ids)
        padded_attention_mask.append(mask)

    inputs = {
        "input_ids": torch.stack(padded_input_ids, dim=0),
        "attention_mask": torch.stack(padded_attention_mask, dim=0),
    }
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    preds = []

    for i in range(len(per_example_inputs)):
        input_len = inputs["input_ids"].shape[1]   # same for every row - it's the padded length
        generated = outputs[i][input_len:]

        response = processor.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        preds.append(response)

    return preds

# =========================
# RUN
# =========================
if os.path.exists(OUTPUT_CSV):
    print(f"Resuming from {OUTPUT_CSV}")
    df = pd.read_csv(OUTPUT_CSV)
else:
    df = pd.read_csv(INPUT_CSV)
    df = df[df["split"] == "test"].reset_index(drop=True)
    df["model_out"] = ""

df["model_out"] = df["model_out"].astype("object")
size = len(df)

# Find rows that still need inference
pending_mask = (
    df["model_out"].isna()
    | (df["model_out"].astype(str).str.strip() == "")
)

pending_indices = df.index[pending_mask].tolist()

print(f"Total Samples     = {size}")
print(f"Remaining Samples = {len(pending_indices)}")

for start in tqdm(range(0, len(pending_indices), BATCH_SIZE)):
    batch_indices = pending_indices[start:start + BATCH_SIZE]

    batch = df.loc[batch_indices]

    preds = generate_batch(batch)

    df.loc[batch_indices, "model_out"] = preds

    # Save periodically
    if ((start // BATCH_SIZE) + 1) % SAVE_FREQUENCY == 0:
        df.to_csv(OUTPUT_CSV, index=False)

# Final save
df.to_csv(OUTPUT_CSV, index=False)

print(f"Output file saved to {OUTPUT_CSV}")