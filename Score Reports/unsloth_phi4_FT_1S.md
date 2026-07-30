## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_phi4_FT_1S.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- PRED_COL: model_out
- PRED_METER: out_meter

Outputs are clean
Detecting meters: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2919/2919 [00:00<00:00, 4160.16it/s]

## Metric 1: Overall Accuracy (meter_cd vs out_meter)
------------------
- Total samples      : 2919
- Correct predictions: 1701
- Accuracy           : 58.27%
- Null meters        : 1218
- Problem rows       : 0

## Metric 2: Semantic Similarity (input vs model_out)
------------------
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 7807.65it/s]
- Total samples          : 2919
- Mean semantic similarity: 0.6529
- Std semantic similarity : 0.1299
- Min / Max               : 0.0669 / 1.0000

## Metric 3: Meter-wise Accuracy

| Meter            |   Total |   Correct |   Accuracy (%) |   Null |
|:-----------------|--------:|----------:|---------------:|-------:|
| Anuṣṭubh         |    2721 |      1692 |          62.18 |   1029 |
| Indravajrā       |      22 |         2 |           9.09 |     20 |
| Mālinī           |       9 |         0 |           0    |      9 |
| Sragdharā        |      21 |         0 |           0    |     21 |
| Upendravajrā     |      12 |         2 |          16.67 |     10 |
| Vasantatilakā    |      66 |         4 |           6.06 |     62 |
| Vaṃśastha        |      18 |         1 |           5.56 |     17 |
| Śikhariṇī        |      16 |         0 |           0    |     16 |
| Śālinī           |       7 |         0 |           0    |      7 |
| Śārdūlavikrīḍita |      27 |         0 |           0    |     27 |

All score/semsim updates saved back to Outputs/unsloth_phi4_FT_1S.csv