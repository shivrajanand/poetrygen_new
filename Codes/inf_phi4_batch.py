import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from peft import PeftModel
import json
import os
import sys

# =========================
# PATHS
# =========================
BASE_MODEL = "unsloth/phi-4"
INPUT_CSV = "Files/v3_gitapress_final_1shot_prompts.csv"
OUTPUT_CSV = "Outputs/unsloth_phi4_UT_1S.csv"
MAX_NEW_TOKENS = 256
SAVE_FREQUENCY = 1
BATCH_SIZE = 24
CONFIG_FILE = "Trained_Models/Phi4-14B-DEV-1SHOT/essential_config.json"

print("Test File in use: ", INPUT_CSV)

if 'UT' in OUTPUT_CSV:
    SET_LORA = False
elif 'FT' in OUTPUT_CSV:
    SET_LORA = True
else:
    print("Mention FT: Finetuned or UT: Untrained in OUTPUT CSV")
    sys.exit(0)

if SET_LORA:
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
    LORA_PATH = config["best_model"]["best_model_checkpoint"]
    
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

# =========================
# LOAD MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"


base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
print(f"Model used is {BASE_MODEL}", end="")

model=base_model
if SET_LORA:
    print(f" with PEFT LORA path = {LORA_PATH}", end="")
    model = PeftModel.from_pretrained(
        model,
        LORA_PATH,
    )
    
    model = model.merge_and_unload()

print()


model.eval()

model.to(device)
model = torch.compile(model)
# =========================
# GENERATION
# =========================
def generate_batch(batch_df):
    messages = []

    for _, row in batch_df.iterrows():
        msg = [
            {"role": "system", "content": row["prompt"]},
            {
                "role": "user",
                "content": f"Meaning:\n{row['hi']}\n\nGenerate the Sanskrit verse.\n",
            },
        ]

        messages.append(
            tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    inputs = tokenizer(
        messages,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    preds = []

    for i in range(len(messages)):
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[i][input_len:]
        preds.append(
            tokenizer.decode(
                generated,
                skip_special_tokens=True,
            ).strip()
        )

    return preds




# # =========================
# # RUN
# # =========================
# df = pd.read_csv(INPUT_CSV)
# df = df[df["split"] == "test"].reset_index(drop=True)
# size = df.shape[0]
# print("Total Samples = ", size)


# df["model_out"] = ""

# for start in tqdm(range(0, size, BATCH_SIZE)):
#     end = min(start + BATCH_SIZE, size)

#     batch = df.iloc[start:end]

#     preds = generate_batch(batch)

#     df.loc[start:end-1, "model_out"] = preds

#     if end % SAVE_FREQUENCY == 0:
#         df.to_csv(OUTPUT_CSV, index=False)


# df.to_csv(OUTPUT_CSV, index=False)

# print(f"Output file saved to {OUTPUT_CSV}")



# =========================
# RUN
# =========================
import os

if os.path.exists(OUTPUT_CSV):
    print(f"Resuming from {OUTPUT_CSV}")
    df = pd.read_csv(OUTPUT_CSV)
else:
    df = pd.read_csv(INPUT_CSV)
    df = df[df["split"] == "test"].reset_index(drop=True)
    df["model_out"] = ""

size = len(df)

# Find rows that still need inference
pending_mask = df["model_out"].isna() | (df["model_out"].astype(str).str.strip() == "")
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