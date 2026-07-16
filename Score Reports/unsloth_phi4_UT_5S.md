## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_phi4_UT_5S.csv
- GROUND_TRUTH: meter_cd
- PRED_COL: model_out
- PRED_METER: out_meter
Problematic rows saved to Outputs/unsloth_phi4_UT_5S_problem_cols.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 24 rows as 'problem' in 'out_meter'.


## Overall Evaluation
------------------
- Total samples      : 2895
- Correct predictions: 246
- Accuracy           : 8.50%
- Null meters        : 0
- Problem rows       : 24

## Macro Report
------------------
- Precision : 0.091
- Recall    : 0.008
- F1 Score  : 0.015
## Meter-wise Evaluation

| Meter            |   Total |   Correct |   Accuracy (%) |   Precision |   Recall |    F1 |   Null |
|:-----------------|--------:|----------:|---------------:|------------:|---------:|------:|-------:|
| Anuṣṭubh         |    2701 |       246 |           9.11 |           1 |    0.091 | 0.167 |      0 |
| Indravajrā       |      22 |         0 |           0    |           0 |    0     | 0     |      0 |
| Mālinī           |       9 |         0 |           0    |           0 |    0     | 0     |      0 |
| Sragdharā        |      19 |         0 |           0    |           0 |    0     | 0     |      0 |
| Upendravajrā     |      12 |         0 |           0    |           0 |    0     | 0     |      0 |
| Vasantatilakā    |      65 |         0 |           0    |           0 |    0     | 0     |      0 |
| Vaṃśastha        |      18 |         0 |           0    |           0 |    0     | 0     |      0 |
| Śikhariṇī        |      15 |         0 |           0    |           0 |    0     | 0     |      0 |
| Śālinī           |       7 |         0 |           0    |           0 |    0     | 0     |      0 |
| Śārdūlavikrīḍita |      27 |         0 |           0    |           0 |    0     | 0     |      0 |
All score updates saved back to Outputs/unsloth_phi4_UT_5S.csv