## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_phi4_FT_ZS.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- PRED_COL: model_out
- PRED_METER: out_meter

Outputs are clean
Detecting meters: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2919/2919 [00:00<00:00, 4152.20it/s]

## Metric 1: Overall Accuracy (meter_cd vs out_meter)
------------------
- Total samples      : 2919
- Correct predictions: 1598
- Accuracy           : 54.74%
- Null meters        : 1320
- Problem rows       : 0

## Metric 2: Semantic Similarity (input vs model_out)
------------------
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 7905.08it/s]
- Total samples          : 2919
- Mean semantic similarity: 0.6647
- Std semantic similarity : 0.1293
- Min / Max               : 0.0999 / 0.9537

## Metric 3: Meter-wise Accuracy

| Meter            |   Total |   Correct |   Accuracy (%) |   Null |
|:-----------------|--------:|----------:|---------------:|-------:|
| Anuṣṭubh         |    2721 |      1598 |          58.73 |   1123 |
| Indravajrā       |      22 |         0 |           0    |     22 |
| Mālinī           |       9 |         0 |           0    |      8 |
| Sragdharā        |      21 |         0 |           0    |     21 |
| Upendravajrā     |      12 |         0 |           0    |     12 |
| Vasantatilakā    |      66 |         0 |           0    |     66 |
| Vaṃśastha        |      18 |         0 |           0    |     18 |
| Śikhariṇī        |      16 |         0 |           0    |     16 |
| Śālinī           |       7 |         0 |           0    |      7 |
| Śārdūlavikrīḍita |      27 |         0 |           0    |     27 |

All score/semsim updates saved back to Outputs/unsloth_phi4_FT_ZS.csv