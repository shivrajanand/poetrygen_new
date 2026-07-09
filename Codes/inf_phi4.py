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
SAVE_FREQUENCY = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
)

model.to(device)
model.eval()

# =========================
# GENERATION
# =========================
def generate(row):
    messages = [
        {"role": "system", "content": row['prompt']},
        {"role": "user", "content": f"Meaning:\n{row['hi']}\n\nGenerate the Sanskrit verse.\n"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
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


    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()




# =========================
# RUN
# =========================
df = pd.read_csv(INPUT_CSV)
df = df[df["split"] == "test"].reset_index(drop=True)
size = df.shape[0]
print("Total Samples = ", size)


df["model_out"] = ""
for i in tqdm(range(size)):
    pred = generate(df.iloc[i])
    df.at[i, "model_out"] = pred


    if (i + 1) % SAVE_FREQUENCY == 0:
        df.to_csv(OUTPUT_CSV, index=False)


df.to_csv(OUTPUT_CSV, index=False)

print(f"Output file saved to {OUTPUT_CSV}")