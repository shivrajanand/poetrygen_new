import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path("chandas-detector").resolve()))
import chandas_detector
from chandas_detector import detect_meter, format_result

from sentence_transformers import SentenceTransformer
from skrutable.meter_identification import MeterIdentifier

FILEPATH = sys.argv[1]

INPUT_COL = "hi"                    # source text fed to the model
GROUND_TRUTH = "meter_cd"           # ground-truth meter
GROUND_TRUTH_SYLLABLES = "syllable_count"   # ground-truth syllable count
PRED_COL = "model_out"              # generated Sanskrit verse
PRED_METER = "out_meter"            # meter detected from model_out
PRED_SYLLABLES = "pred_syllable_count"      # syllable count detected from model_out

df = pd.read_csv(FILEPATH)

########################################
####### Markdown level prints
########################################

print("## FILE DETAILS")
print("------------------------------")
print(f"- FILEPATH: {FILEPATH}")
print(f"- INPUT_COL: {INPUT_COL}")
print(f"- GROUND_TRUTH: {GROUND_TRUTH}")
print(f"- GROUND_TRUTH_SYLLABLES: {GROUND_TRUTH_SYLLABLES}")
print(f"- PRED_COL: {PRED_COL}")
print(f"- PRED_METER: {PRED_METER}")
print(f"- PRED_SYLLABLES: {PRED_SYLLABLES}")

########################################
####### Check for alphanum problems
########################################
df[PRED_METER] = None
df[PRED_SYLLABLES] = pd.NA

mask_problem = df[PRED_COL].str.contains(r"[A-Za-z0-46-9]", na=False)

df_with_alnum = df[mask_problem]

if df_with_alnum.empty:
    print("\nOutputs are clean")
else:
    problem_file = FILEPATH.replace(".csv", "_problem_cols.csv")
    problem_file = FILEPATH.replace("Vasantatalika-Overfit", "Vasantatalika-Overfit/Problems")
    df_with_alnum.to_csv(problem_file, index=False)

    # Mark problematic rows instead of removing them
    df.loc[mask_problem, PRED_METER] = "problem"

    print(f"\nProblematic rows saved to {problem_file}.")
    print("Letter '5' is ignored because models sometimes use it for avagraha (ऽ).")
    print(f"Marked {len(df_with_alnum)} rows as 'problem' in '{PRED_METER}'.")

########################################
####### skrutable syllable counter
########################################

MI = MeterIdentifier()


def get_pred_syllable_count(verse_text):
    """
    Count syllables in a generated verse using skrutable's MeterIdentifier.

    Configuration per spec:
      from_scheme="DEV", resplit_option="resplit_lite", resplit_keep_midpoint=True

    skrutable.scansion.Verse exposes `syllable_weights`: a string of 'l'/'g'
    characters (one per syllable), with pādas joined by '\n', e.g.
      "gggglggg\nllgllglg\nllgglggl\nllgllgll"
    Stripping whitespace/newlines and taking the length gives the syllable
    count directly (confirmed against the installed skrutable package).
    """
    result = MI.identify_meter(
        verse_text,
        from_scheme="DEV",
        resplit_option="resplit_lite",
        resplit_keep_midpoint=True,
    )

    weights = (result.syllable_weights or "").replace("\n", "").replace(" ", "")
    return len(weights) if weights else None


########################################
####### Getting output meters + syllable counts
########################################

for idx in tqdm(df.index, total=len(df), desc="Detecting meters"):
    if df.at[idx, PRED_METER] == "problem":
        continue

    verse = df.at[idx, PRED_COL].strip()
    result = detect_meter(verse)

    df.at[idx, PRED_METER] = (
        result.meter if result.confidence == "exact" else None
    )

    try:
        df.at[idx, PRED_SYLLABLES] = get_pred_syllable_count(verse)
    except Exception:
        df.at[idx, PRED_SYLLABLES] = None

########################################
####### Row-wise metrics: half_acc / full_acc
########################################

eval_df = df[df[PRED_METER] != "problem"].copy()
eval_df[PRED_METER] = eval_df[PRED_METER].fillna("UNKNOWN")

# Cast to a nullable numeric dtype so e.g. int64 vs object/NA comparisons
# don't silently evaluate to False.
eval_df[GROUND_TRUTH_SYLLABLES] = pd.to_numeric(
    eval_df[GROUND_TRUTH_SYLLABLES], errors="coerce"
).astype("Int64")
eval_df[PRED_SYLLABLES] = pd.to_numeric(
    eval_df[PRED_SYLLABLES], errors="coerce"
).astype("Int64")

eval_df["half_acc"] = (
    eval_df[GROUND_TRUTH_SYLLABLES] == eval_df[PRED_SYLLABLES]
).astype(int)

eval_df["full_acc"] = (
    (eval_df[GROUND_TRUTH] == eval_df[PRED_METER])
    & (eval_df[GROUND_TRUTH_SYLLABLES] == eval_df[PRED_SYLLABLES])
).astype(int)

# sanity check: half accuracy should always be >= full accuracy
assert (eval_df["half_acc"] >= eval_df["full_acc"]).all(), (
    "Invariant violated: half_acc must be >= full_acc for every row"
)

df["half_acc"] = pd.NA
df["full_acc"] = pd.NA
df.loc[eval_df.index, "half_acc"] = eval_df["half_acc"]
df.loc[eval_df.index, "full_acc"] = eval_df["full_acc"]

########################################
####### Metric: Semantic Similarity
########################################

sim_df = df.dropna(subset=[INPUT_COL, PRED_COL]).copy()
sim_df = sim_df[sim_df[PRED_METER] != "problem"]

semantic_model = SentenceTransformer('sanganaka/bge-m3-sanskritFT')

inputs = sim_df[INPUT_COL].astype(str).tolist()
poetry_outputs = sim_df[PRED_COL].astype(str).tolist()

in_embs = semantic_model.encode(inputs, convert_to_tensor=True)
out_embs = semantic_model.encode(poetry_outputs, convert_to_tensor=True)

sims = semantic_model.similarity_pairwise(in_embs, out_embs)
sims = sims.cpu().numpy()

df["semsim"] = pd.NA
df.loc[sim_df.index, "semsim"] = sims

# bring semsim into eval_df for aggregate reporting (only for non-problem rows)
eval_df["semsim"] = df.loc[eval_df.index, "semsim"]

########################################
####### Overall Evaluation (Markdown)
########################################

half_acc_overall = eval_df["half_acc"].mean()
full_acc_overall = eval_df["full_acc"].mean()
semsim_overall = eval_df["semsim"].astype(float).mean()

print("\n## Overall Evaluation\n")
print("| Metric | Value |")
print("|--------|------:|")
print(f"| Half Accuracy | {half_acc_overall:.2%} |")
print(f"| Full Accuracy | {full_acc_overall:.2%} |")
print(f"| Mean Semantic Similarity | {semsim_overall:.4f} |")

print("\n(supporting detail)")
print(f"- Total samples      : {len(eval_df)}")
print(f"- Problem rows       : {(df[PRED_METER] == 'problem').sum()}")
print(f"- Null meters        : {(eval_df[PRED_METER] == 'UNKNOWN').sum()}")

########################################
####### Meter-wise Evaluation (Markdown)
########################################

meters = sorted(eval_df[GROUND_TRUTH].dropna().unique())

results = []

for meter in meters:
    temp_df = eval_df[eval_df[GROUND_TRUTH] == meter]

    samples = len(temp_df)
    half_acc = temp_df["half_acc"].mean() if samples else 0
    full_acc = temp_df["full_acc"].mean() if samples else 0
    mean_semsim = temp_df["semsim"].astype(float).mean() if samples else float("nan")

    results.append({
        "Meter": meter,
        "Samples": samples,
        "Half Accuracy": half_acc,
        "Full Accuracy": full_acc,
        "Mean Semantic Similarity": mean_semsim,
    })

results_df = pd.DataFrame(results)

print("\n## Meter-wise Evaluation\n")
print("| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |")
print("|-------|--------:|--------------:|--------------:|-------------------------:|")
for _, row in results_df.iterrows():
    print(
        f"| {row['Meter']} | {row['Samples']} | "
        f"{row['Half Accuracy']:.2%} | {row['Full Accuracy']:.2%} | "
        f"{row['Mean Semantic Similarity']:.4f} |"
    )

########################################
####### Save
########################################

df.to_csv(FILEPATH, index=False)
print(f"\nAll score/semsim updates saved back to {FILEPATH}")