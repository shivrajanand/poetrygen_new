## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_phi4_FT1S_Eval0S.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- PRED_COL: model_out
- PRED_METER: out_meter

Outputs are clean
Detecting meters: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2919/2919 [00:00<00:00, 3616.19it/s]

## Metric 1: Overall Accuracy (meter_cd vs out_meter)
------------------
- Total samples      : 2919
- Correct predictions: 744
- Accuracy           : 25.49%
- Null meters        : 2150
- Problem rows       : 0

## Metric 2: Semantic Similarity (input vs model_out)
------------------
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 7786.15it/s]
- Total samples          : 2919
- Mean semantic similarity: 0.6704
- Std semantic similarity : 0.1366
- Min / Max               : -0.0961 / 0.9539

## Metric 3: Meter-wise Accuracy

| Meter            |   Total |   Correct |   Accuracy (%) |   Null |
|:-----------------|--------:|----------:|---------------:|-------:|
| Anuṣṭubh         |    2721 |       744 |          27.34 |   1977 |
| Indravajrā       |      22 |         0 |           0    |     18 |
| Mālinī           |       9 |         0 |           0    |      9 |
| Sragdharā        |      21 |         0 |           0    |     20 |
| Upendravajrā     |      12 |         0 |           0    |     12 |
| Vasantatilakā    |      66 |         0 |           0    |     58 |
| Vaṃśastha        |      18 |         0 |           0    |     14 |
| Śikhariṇī        |      16 |         0 |           0    |     15 |
| Śālinī           |       7 |         0 |           0    |      5 |
| Śārdūlavikrīḍita |      27 |         0 |           0    |     22 |

All score/semsim updates saved back to Outputs/unsloth_phi4_FT1S_Eval0S.csv