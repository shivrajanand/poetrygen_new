## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_phi4_FT_anustubh.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- PRED_COL: model_out
- PRED_METER: out_meter

Outputs are clean
Detecting meters: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2721/2721 [00:00<00:00, 4289.86it/s]

## Metric 1: Overall Accuracy (meter_cd vs out_meter)
------------------
- Total samples      : 2721
- Correct predictions: 1629
- Accuracy           : 59.87%
- Null meters        : 1092
- Problem rows       : 0

## Metric 2: Semantic Similarity (input vs model_out)
------------------
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 7709.97it/s]
- Total samples          : 2721
- Mean semantic similarity: 0.6546
- Std semantic similarity : 0.1281
- Min / Max               : 0.0773 / 0.9645

## Metric 3: Meter-wise Accuracy

| Meter    |   Total |   Correct |   Accuracy (%) |   Null |
|:---------|--------:|----------:|---------------:|-------:|
| Anuṣṭubh |    2721 |      1629 |          59.87 |   1092 |

All score/semsim updates saved back to Outputs/unsloth_phi4_FT_anustubh.csv