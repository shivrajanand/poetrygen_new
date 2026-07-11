import torch
import pandas as pd
from transformers import AutoProcessor, AutoModelForMultimodalLM
from tqdm import tqdm

# =========================
# PATHS
# =========================
BASE_MODEL = "google/gemma-4-12B-it"
INPUT_CSV = "Files/v3_gitapress_final.csv"
OUTPUT_CSV = "Outputs/phi4_ut_zs.csv"
MAX_NEW_TOKENS = 256
SAVE_FREQUENCY = 1
BATCH_SIZE = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL
# =========================
processor = AutoProcessor.from_pretrained(BASE_MODEL)

model = AutoModelForMultimodalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
)

model.to(device)
model.eval()

# =========================
# INSTRUCTION
# =========================
SYSTEM_INSTRUCTION = """You are an expert assistant for AyurParam, a multiple-choice question answering benchmark containing government-exam questions on Ayurveda, Ayurvedic medicines, pharmacology, Indian medical systems, and related subjects.

Given a question and exactly four answer options, select the single best answer.

Rules:
1. Use domain knowledge in Ayurveda and relevant modern biomedical science where applicable.
2. Carefully distinguish classical Ayurvedic terminology, formulations, dravyaguna, pharmacognosy, and clinical concepts.
3. Return only the option label: A, B, C, or D.
4. Do not provide explanations, reasoning, confidence scores, or any additional text.
5. If the question is ambiguous, choose the most likely answer according to standard government-exam conventions.
6. Stricly output only 1 character of A, B, C, D. Nothing else"""

# =========================
# GENERATION
# =========================
def generate(row):

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": f"""Question:
{row['question']}

Options:
A. {row['option_a']}
B. {row['option_b']}
C. {row['option_c']}
D. {row['option_d']}

Answer:""",
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    input_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    generated_tokens = outputs[0][input_length:]

    response = processor.decode(
        generated_tokens,
        skip_special_tokens=False,
    )

    parsed = processor.parse_response(response)

    return parsed["content"].strip()

# =========================
# RUN
# =========================
df = pd.read_csv(INPUT_CSV)
size = len(df)

print("Total Samples =", size)

df["model_out"] = ""

for i in tqdm(range(size)):
    pred = generate(df.iloc[i])
    df.loc[i, "model_out"] = pred

    if (i + 1) % SAVE_FREQUENCY == 0:
        df.to_csv(OUTPUT_CSV, index=False)

df.to_csv(OUTPUT_CSV, index=False)

print(f"Output file saved to {OUTPUT_CSV}")