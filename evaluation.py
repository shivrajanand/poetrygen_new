import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path("chandas-detector").resolve()))
import chandas_detector
from chandas_detector import detect_meter, format_result
from sentence_transformers import SentenceTransformer

FILEPATH = sys.argv[1]

INPUT_COL = "hi"           # source text fed to the model
GROUND_TRUTH = "meter_cd"  # ground-truth meter
PRED_COL = "model_out"     # generated Sanskrit verse
PRED_METER = "out_meter"   # meter detected from model_out

df = pd.read_csv(FILEPATH)

########################################
####### Markdown level prints
########################################

print("## FILE DETAILS")
print("------------------------------")
print(f"- FILEPATH: {FILEPATH}")
print(f"- INPUT_COL: {INPUT_COL}")
print(f"- GROUND_TRUTH: {GROUND_TRUTH}")
print(f"- PRED_COL: {PRED_COL}")
print(f"- PRED_METER: {PRED_METER}")

########################################
####### Check for alphanum problems
########################################
df[PRED_METER] = None

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
####### Getting output meters
########################################

for idx in tqdm(df.index, total=len(df), desc="Detecting meters"):
    if df.at[idx, PRED_METER] == "problem":
        continue

    verse = df.at[idx, PRED_COL].strip()
    result = detect_meter(verse)

    df.at[idx, PRED_METER] = (
        result.meter if result.confidence == "exact" else None
    )

########################################
####### Metric 1: Overall Accuracy
########################################

eval_df = df[df[PRED_METER] != "problem"].copy()
eval_df[PRED_METER] = eval_df[PRED_METER].fillna("UNKNOWN")

eval_df["score"] = (eval_df[GROUND_TRUTH] == eval_df[PRED_METER]).astype(int)

df["score"] = pd.NA
df.loc[eval_df.index, "score"] = eval_df["score"]

overall_acc = eval_df["score"].mean()

print("\n## Metric 1: Overall Accuracy (meter_cd vs out_meter)")
print("------------------")
print(f"- Total samples      : {len(eval_df)}")
print(f"- Correct predictions: {eval_df['score'].sum()}")
print(f"- Accuracy           : {overall_acc:.2%}")
print(f"- Null meters        : {(eval_df[PRED_METER] == 'UNKNOWN').sum()}")
print(f"- Problem rows       : {(df[PRED_METER] == 'problem').sum()}")

########################################
####### Metric 2: Semantic Similarity
########################################

print("\n## Metric 2: Semantic Similarity (input vs model_out)")
print("------------------")

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

print(f"- Total samples          : {len(sim_df)}")
print(f"- Mean semantic similarity: {sims.mean():.4f}")
print(f"- Std semantic similarity : {sims.std():.4f}")
print(f"- Min / Max               : {sims.min():.4f} / {sims.max():.4f}")

########################################
####### Metric 3: Meter-wise Accuracy
########################################

meters = sorted(eval_df[GROUND_TRUTH].dropna().unique())

results = []

for meter in meters:
    total = (eval_df[GROUND_TRUTH] == meter).sum()

    correct = (
        (eval_df[GROUND_TRUTH] == meter) &
        (eval_df[PRED_METER] == meter)
    ).sum()

    accuracy = 100 * correct / total if total else 0

    null_preds = (
        (eval_df[GROUND_TRUTH] == meter) &
        (eval_df[PRED_METER] == "UNKNOWN")
    ).sum()

    results.append({
        "Meter": meter,
        "Total": total,
        "Correct": correct,
        "Accuracy (%)": round(accuracy, 2),
        "Null": null_preds,
    })

results_df = pd.DataFrame(results)

print("\n## Metric 3: Meter-wise Accuracy\n")
print(results_df.to_markdown(index=False))

########################################
####### Save
########################################

df.to_csv(FILEPATH, index=False)
print(f"\nAll score/semsim updates saved back to {FILEPATH}")

# ########################################
# ####### Creating master csv
# ########################################

# master_csv = "master-score.csv"

# ########################################
# ####### Update master csv
# ########################################

# from pathlib import Path
# import os

# summary = {
#     "experiment": Path(FILEPATH).parent.name,
#     "Checkpoint": Path(FILEPATH).stem,
#     "overall_acc": round(overall_acc * 100, 2),
#     "sem-sim": round(float(sims.mean()), 4),
#     "total": len(eval_df),
#     "problem-rows": int((df[PRED_METER] == "problem").sum()),
#     "correct": int(eval_df["score"].sum()),
#     "null": int((eval_df[PRED_METER] == "UNKNOWN").sum()),
# }

# summary_df = pd.DataFrame([summary])

# if os.path.exists(master_csv):
#     master = pd.read_csv(master_csv)

#     # Remove existing entry for same checkpoint (if rerunning)
#     master = master[
#         ~(
#             (master["experiment"] == summary["experiment"]) &
#             (master["Checkpoint"] == summary["Checkpoint"])
#         )
#     ]

#     master = pd.concat([master, summary_df], ignore_index=True)

# else:
#     master = summary_df

# # Sort checkpoints numerically within each experiment
# def checkpoint_number(x):
#     try:
#         return int(str(x).split("-")[-1].split("c")[-1])
#     except:
#         return 999999

# master = master.sort_values(
#     by=["experiment", "Checkpoint"],
#     key=lambda s: s.map(checkpoint_number) if s.name == "Checkpoint" else s
# ).reset_index(drop=True)

# master.to_csv(master_csv, index=False)

# print(f"Updated master csv: {master_csv}")

