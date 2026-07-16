## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_phi4_UT_1S.csv
- GROUND_TRUTH: meter_cd
- PRED_COL: model_out
- PRED_METER: out_meter
Problematic rows saved to Outputs/unsloth_phi4_UT_1S_problem_cols.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 11 rows as 'problem' in 'out_meter'.


## Overall Evaluation
------------------
- Total samples      : 2908
- Correct predictions: 70
- Accuracy           : 2.41%
- Null meters        : 0
- Problem rows       : 11

## Macro Report
------------------
- Precision : 0.091
- Recall    : 0.002
- F1 Score  : 0.005
## Meter-wise Evaluation

| Meter            |   Total |   Correct |   Accuracy (%) |   Precision |   Recall |   F1 |   Null |
|:-----------------|--------:|----------:|---------------:|------------:|---------:|-----:|-------:|
| Anuṣṭubh         |    2710 |        70 |           2.58 |           1 |    0.026 | 0.05 |      0 |
| Indravajrā       |      22 |         0 |           0    |           0 |    0     | 0    |      0 |
| Mālinī           |       9 |         0 |           0    |           0 |    0     | 0    |      0 |
| Sragdharā        |      21 |         0 |           0    |           0 |    0     | 0    |      0 |
| Upendravajrā     |      12 |         0 |           0    |           0 |    0     | 0    |      0 |
| Vasantatilakā    |      66 |         0 |           0    |           0 |    0     | 0    |      0 |
| Vaṃśastha        |      18 |         0 |           0    |           0 |    0     | 0    |      0 |
| Śikhariṇī        |      16 |         0 |           0    |           0 |    0     | 0    |      0 |
| Śālinī           |       7 |         0 |           0    |           0 |    0     | 0    |      0 |
| Śārdūlavikrīḍita |      27 |         0 |           0    |           0 |    0     | 0    |      0 |
All score updates saved back to Outputs/unsloth_phi4_UT_1S.csv