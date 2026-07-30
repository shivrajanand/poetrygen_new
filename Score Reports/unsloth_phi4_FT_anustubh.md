(stableenv) shivraj-pg@Cyclops:~/v3_poetrygen_new$ python3 evaluation.py Outputs/unsloth_phi4_FT_anustubh1S.csv 
## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_phi4_FT_anustubh.csv
- GROUND_TRUTH: meter_cd
- PRED_COL: model_out
- PRED_METER: out_meter
Outputs are clean

## Overall Evaluation
------------------
- Total samples      : 2721
- Correct predictions: 1629
- Accuracy           : 59.87%
- Null meters        : 0
- Problem rows       : 0

## Macro Report
------------------
- Precision : 0.500
- Recall    : 0.299
- F1 Score  : 0.374
## Meter-wise Evaluation

| Meter    |   Total |   Correct |   Accuracy (%) |   Precision |   Recall |    F1 |   Null |
|:---------|--------:|----------:|---------------:|------------:|---------:|------:|-------:|
| Anuṣṭubh |    2721 |      1629 |          59.87 |           1 |    0.599 | 0.749 |      0 |
All score updates saved back to Outputs/unsloth_phi4_FT_anustubh1S.csv