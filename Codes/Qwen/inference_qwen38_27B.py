import sys

if len(sys.argv) < 4:
    print("ERROR:\nProgram Usage: python3 inf.py <MODEL_PATH> <DATA_FILE_PATH> <OUTPUT_CSV>")
    sys.exit(1)

import unsloth
from unsloth import FastLanguageModel
import torch
import pandas as pd
from tqdm import tqdm
import os

MODEL_PATH = sys.argv[1]
DATA_FILE_PATH = sys.argv[2]
OUTPUT_CSV = sys.argv[3]

SPLIT = "test"
MAX_LEN = 918
BATCH_SIZE = 52
MAX_NEW_TOKENS = 256
SAVE_FREQUENCY = 1


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_LEN,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

tok = getattr(tokenizer, "tokenizer", tokenizer)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def build_prompt(system_prompt, hi):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Meaning:\n{hi}\n\nGenerate the Sanskrit verse."},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        preserve_thinking=False,
    )
    return text.replace("<think>\n\n</think>\n\n", "")




if os.path.exists(OUTPUT_CSV):
    print(f"Resuming from {OUTPUT_CSV}")
    df = pd.read_csv(OUTPUT_CSV)
else:
    df = pd.read_csv(DATA_FILE_PATH)
    df = df[df["split"] == SPLIT].copy().reset_index(drop=True)
    df["model_out"] = ""

# Ensure output contains only the test split
df = df[df["split"] == SPLIT].copy().reset_index(drop=True)

if "model_out" not in df.columns:
    df["model_out"] = ""

df["model_out"] = df["model_out"].astype("object")

pending_mask = (
    df["model_out"].isna()
    | (df["model_out"].astype(str).str.strip() == "")
)

pending_indices = df.index[pending_mask].tolist()

size = len(df)

print(f"Total Samples     = {size}")
print(f"Already Completed = {size - len(pending_indices)}")
print(f"Remaining Samples = {len(pending_indices)}")


for step, start in enumerate(
    tqdm(range(0, len(pending_indices), BATCH_SIZE)), start=1
):
    batch_idx = pending_indices[start:start + BATCH_SIZE]

    prompts = [
        build_prompt(df.at[idx, "prompt"], df.at[idx, "hi"])
        for idx in batch_idx
    ]

    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )

    new_tokens = out[:, enc["input_ids"].shape[1]:]
    batch_predictions = [
        p.strip()
        for p in tok.batch_decode(
            new_tokens, skip_special_tokens=True
        )
    ]

    df.loc[batch_idx, "model_out"] = batch_predictions

    if step % SAVE_FREQUENCY == 0:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"Saved output to {OUTPUT_CSV}")
print(f"Completed: {df['model_out'].notna().sum()}/{len(df)}")