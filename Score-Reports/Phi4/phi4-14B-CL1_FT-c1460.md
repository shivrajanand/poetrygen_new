## FILE DETAILS
------------------------------
- FILEPATH: Outputs/phi4-14B-customLoss_FT-c1460.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/phi4-14B-customLoss_FT-c1460.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 1 rows as 'problem' in 'out_meter'.

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 57.11% |
| Full Accuracy | 42.21% |
| Mean Semantic Similarity | 0.6592 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 1
- Null meters        : 1685

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 60.75% | 45.17% | 0.6610 |
| Indravajrā | 22 | 9.09% | 4.55% | 0.6535 |
| Mālinī | 9 | 11.11% | 11.11% | 0.6593 |
| Sragdharā | 21 | 4.76% | 0.00% | 0.5663 |
| Upendravajrā | 12 | 8.33% | 0.00% | 0.6314 |
| Vasantatilakā | 66 | 10.61% | 1.52% | 0.6798 |
| Vaṃśastha | 18 | 0.00% | 0.00% | 0.5970 |
| Śikhariṇī | 16 | 0.00% | 0.00% | 0.5865 |
| Śālinī | 7 | 28.57% | 0.00% | 0.7106 |
| Śārdūlavikrīḍita | 27 | 0.00% | 0.00% | 0.5858 |

All score/semsim updates saved back to Outputs/phi4-14B-customLoss_FT-c1460.csv
