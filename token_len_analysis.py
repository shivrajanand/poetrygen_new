import pandas as pd
from transformers import AutoTokenizer

# ======================================================
# Hyperparameters
# ======================================================
MODEL_NAME = "microsoft/phi-4"
CSV_FILE = "Files/v3_gitapress_final.csv"
TEXT_COLUMNS = ["sa"]   
SPLIT = "train"            

# ======================================================
# Load data
# ======================================================
df = pd.read_csv(CSV_FILE)

if SPLIT is not None:
    df = df[df["split"] == SPLIT].copy()

# ======================================================
# Load tokenizer
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ======================================================
# Concatenate columns and compute token counts
# ======================================================
text = (
    df[TEXT_COLUMNS]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
)

token_counts = text.apply(
    lambda x: len(tokenizer.encode(x, add_special_tokens=False))
)

# ======================================================
# Statistics
# ======================================================
stats = {
    "Samples": len(token_counts),
    "Min": int(token_counts.min()),
    "Median": int(token_counts.median()),
    "Mean": round(token_counts.mean(), 2),
    "90p": int(token_counts.quantile(0.90)),
    "95p": int(token_counts.quantile(0.95)),
    "99p": int(token_counts.quantile(0.99)),
    "Max": int(token_counts.max()),
}

# ======================================================
# Statistics as a one-row table
# ======================================================
summary_df = pd.DataFrame([{
    "Model": MODEL_NAME,
    "Dataset": CSV_FILE,
    "Split": SPLIT if SPLIT else "All",
    "Columns": ", ".join(TEXT_COLUMNS),
    "Samples": len(token_counts),
    "Min": int(token_counts.min()),
    "Median": int(token_counts.median()),
    "Mean": round(token_counts.mean(), 2),
    "90p": int(token_counts.quantile(0.90)),
    "95p": int(token_counts.quantile(0.95)),
    "99p": int(token_counts.quantile(0.99)),
    "Max": int(token_counts.max()),
}])

print(summary_df.to_markdown(index=False))