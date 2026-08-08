import unsloth
import torch
import pandas as pd
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from tqdm import tqdm
import os
import sys
import warnings

warnings.filterwarnings("ignore")

if len(sys.argv) < 3:
    print(
        "ERROR: Incorrect Usage\n"
        "Command must be like: python myscript.py <config_file> <output_csv>"
    )
    sys.exit(1)
    
# =========================
# PATHS
# =========================
BASE_MODEL = "unsloth/gemma-4-31B-it"          # must match training's MODEL_NAME
INPUT_CSV = "Files/v3_gitapress_final_anustubh.csv"

MAX_LEN = 768          # must match training's MAX_LEN
LOAD_IN_4BIT = True    # must match training's LOAD_IN_4BIT
MAX_NEW_TOKENS = 128
SAVE_FREQUENCY = 1
BATCH_SIZE = 16      

CONFIG_FILE = sys.argv[1]  
OUTPUT_CSV = sys.argv[2] 

print("Test File in use: ", INPUT_CSV)

if 'UT' in OUTPUT_CSV:
    SET_LORA = False
elif 'FT' in OUTPUT_CSV:
    SET_LORA = True
else:
    print(
        "ERROR: OUTPUT_CSV must contain either 'FT' (Finetuned) or 'UT' (Untrained).\n"
        "FT = Finetuned model for inference.\n"
        "UT = Base model for inference."
    )
    sys.exit(1)

LORA_PATH = CONFIG_FILE if SET_LORA else ""

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

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

# ---------------------------------------------------------------------------
# Load model + tokenizer (text-only -> FastModel, matches training script)
# ---------------------------------------------------------------------------
model, tokenizer = FastModel.from_pretrained(
    model_name=model_name,
    max_seq_length=MAX_LEN,
    load_in_4bit=LOAD_IN_4BIT,
)

tokenizer = get_chat_template(tokenizer, "gemma-4")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

print(f"Model used is {BASE_MODEL}", end="")
if SET_LORA:
    print(f" with LoRA checkpoint = {LORA_PATH}", end="")
print()

FastModel.for_inference(model)

# =========================
# GENERATION
# =========================
def generate_batch(batch_df):
    input_ids_list = []
    attention_mask_list = []

    for _, row in batch_df.iterrows():
        msg = [
            {"role": "system", "content": row["prompt"]},
            {
                "role": "user",
                "content": f"Meaning:\n{row['hi']}\n\nGenerate the Sanskrit verse.\n",
            },
        ]

        # Render with tokenize=False + plain string content: this matches
        # training exactly and avoids the processor's visuals-scanning code
        # path (which requires content to be a list of {"type": ...} dicts
        # and breaks on plain strings when tokenize=True).
        text = tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True,
        )

        # NOTE: must pass text as a KEYWORD arg. unsloth's patched processor
        # __call__ signature is (images=None, text=None, videos=None, ...) —
        # a positional call binds our string to `images` instead of `text`,
        # leaving `text=None` and crashing on `text[0]`.
        enc = tokenizer(text=text, return_tensors="pt", add_special_tokens=False)
        input_ids_list.append(enc["input_ids"][0])
        attention_mask_list.append(enc["attention_mask"][0])

    max_len = max(ids.shape[0] for ids in input_ids_list)
    pad_id = tokenizer.pad_token_id

    padded_input_ids = []
    padded_attention_mask = []
    for ids, mask in zip(input_ids_list, attention_mask_list):
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            pad_ids = torch.full((pad_len,), pad_id, dtype=ids.dtype)
            pad_mask = torch.zeros((pad_len,), dtype=mask.dtype)
            ids = torch.cat([pad_ids, ids], dim=0)   # left-pad
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
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[106, tokenizer.eos_token_id], #EOT_TOKEN_ID = 106
        )

    input_len = inputs["input_ids"].shape[1]  # same for every row - left-padded
    preds = []
    for i in range(len(input_ids_list)):
        generated = outputs[i][input_len:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
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

    if ((start // BATCH_SIZE) + 1) % SAVE_FREQUENCY == 0:
        df.to_csv(OUTPUT_CSV, index=False)

df.to_csv(OUTPUT_CSV, index=False)
print(f"Output file saved to {OUTPUT_CSV}")