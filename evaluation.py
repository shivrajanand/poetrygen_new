import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path("chandas-detector").resolve()))
import chandas_detector
from chandas_detector import detect_meter, format_result
from sklearn.metrics import precision_recall_fscore_support

FILEPATH = sys.argv[1]
GROUND_TRUTH = "meter_cd"
PRED_COL = "model_out"
PRED_METER = "out_meter"

df = pd.read_csv(FILEPATH)

########################################
####### Markdown level prints
########################################

print("## FILE DETAILS")
print("------------------------------")
print(f"- FILEPATH: {FILEPATH}")
print(f"- GROUND_TRUTH: {GROUND_TRUTH}")
print(f"- PRED_COL: {PRED_COL}")
print(f"- PRED_METER: {PRED_METER}")

########################################
####### Check for alphanum
########################################

########################################
####### Check for alphanum
########################################
df[PRED_METER] = None

mask_problem = df[PRED_COL].str.contains(r"[A-Za-z0-46-9]", na=False)

df_with_alnum = df[mask_problem]

if df_with_alnum.empty:
    print("Outputs are clean")
else:
    problem_file = FILEPATH.replace(".csv", "_problem_cols.csv")
    df_with_alnum.to_csv(problem_file, index=False)

    # Mark problematic rows instead of removing them
    df.loc[mask_problem, PRED_METER] = "problem"

    print(f"Problematic rows saved to {problem_file}.")
    print("Letter '5' is ignored because models sometimes use it for avagraha (ऽ).")
    print(f"Marked {len(df_with_alnum)} rows as 'problem' in '{PRED_METER}'.")
    
########################################
####### Getting output meters 
########################################

for idx in tqdm(df.index, total=len(df)):
    if df.at[idx, PRED_METER] == "problem":
        continue

    verse = df.at[idx, PRED_COL].strip()
    result = detect_meter(verse)

    df.at[idx, PRED_METER] = (
        result.meter if result.confidence == "exact" else None
    )

########################################
####### Scores 
########################################

eval_df = df[df[PRED_METER] != "problem"].copy()
eval_df[PRED_METER] = eval_df[PRED_METER].fillna("UNKNOWN")

eval_df["score"] = (eval_df[GROUND_TRUTH] == eval_df[PRED_METER]).astype(int)

df["score"] = pd.NA

df.loc[eval_df.index, "score"] = eval_df["score"]

########################################
####### Results Overall
########################################

overall_acc = eval_df["score"].mean()

print("\n## Overall Evaluation")
print("------------------")
print(f"- Total samples      : {len(eval_df)}")
print(f"- Correct predictions: {eval_df['score'].sum()}")
print(f"- Accuracy           : {overall_acc:.2%}")
print(f"- Null meters        : {eval_df[PRED_METER].isna().sum()}")
print(f"- Problem rows       : {(df[PRED_METER] == 'problem').sum()}")

########################################
####### Macro Analysis
########################################

precision, recall, f1, _ = precision_recall_fscore_support(
    eval_df[GROUND_TRUTH],
    eval_df[PRED_METER],
    average="macro",
    zero_division=0,
)

print("\n## Macro Report")
print("------------------")
print(f"- Precision : {precision:.3f}")
print(f"- Recall    : {recall:.3f}")
print(f"- F1 Score  : {f1:.3f}")

########################################
####### Meter wise analysis
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
        (
            eval_df[PRED_METER].isna() |
            (eval_df[PRED_METER].astype(str).str.strip() == "")
        )
    ).sum()

    y_true = (eval_df[GROUND_TRUTH] == meter).astype(int)
    y_pred = (eval_df[PRED_METER] == meter).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    results.append({
        "Meter": meter,
        "Total": total,
        "Correct": correct,
        "Accuracy (%)": round(accuracy, 2),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3),
        "Null": null_preds,
    })

results_df = pd.DataFrame(results)

print("## Meter-wise Evaluation\n")
print(results_df.to_markdown(index=False))

df.to_csv(FILEPATH, index=False)
print(f"All score updates saved back to {FILEPATH}")