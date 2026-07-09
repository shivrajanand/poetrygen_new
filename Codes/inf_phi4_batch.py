import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# =========================
# PATHS
# =========================
BASE_MODEL = "microsoft/phi-4"
INPUT_CSV = "Files/v3_gitapress_final.csv"
OUTPUT_CSV = "Outputs/phi4_ut_zs.csv"
MAX_NEW_TOKENS = 256
SAVE_FREQUENCY = 1
BATCH_SIZE = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"


model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
)

model.eval()

# model.to(device)
model.eval()
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




# =========================
# RUN
# =========================
df = pd.read_csv(INPUT_CSV)
df = df[df["split"] == "test"].reset_index(drop=True)
size = df.shape[0]
print("Total Samples = ", size)


df["model_out"] = ""

for start in tqdm(range(0, size, BATCH_SIZE)):
    end = min(start + BATCH_SIZE, size)

    batch = df.iloc[start:end]

    preds = generate_batch(batch)

    df.loc[start:end-1, "model_out"] = preds

    if end % SAVE_FREQUENCY == 0:
        df.to_csv(OUTPUT_CSV, index=False)


df.to_csv(OUTPUT_CSV, index=False)

print(f"Output file saved to {OUTPUT_CSV}")