## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_gemma4-e4b_FT_anustubh.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- PRED_COL: model_out
- PRED_METER: out_meter

Problematic rows saved to Outputs/unsloth_gemma4-e4b_FT_anustubh.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 7 rows as 'problem' in 'out_meter'.
Detecting meters: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2721/2721 [00:00<00:00, 4047.94it/s]

## Metric 1: Overall Accuracy (meter_cd vs out_meter)
------------------
- Total samples      : 2714
- Correct predictions: 5
- Accuracy           : 0.18%
- Null meters        : 2709
- Problem rows       : 7

## Metric 2: Semantic Similarity (input vs model_out)
------------------
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 8072.48it/s]
- Total samples          : 2714
- Mean semantic similarity: 0.6806
- Std semantic similarity : 0.1231
- Min / Max               : 0.0683 / 0.9507

## Metric 3: Meter-wise Accuracy

| Meter    |   Total |   Correct |   Accuracy (%) |   Null |
|:---------|--------:|----------:|---------------:|-------:|
| Anuṣṭubh |    2714 |         5 |           0.18 |   2709 |

All score/semsim updates saved back to Outputs/unsloth_gemma4-e4b_FT_anustubh.csv