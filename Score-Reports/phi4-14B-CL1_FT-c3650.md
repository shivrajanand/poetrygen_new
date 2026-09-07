## FILE DETAILS
------------------------------
- FILEPATH: Outputs/phi4-14B-customLoss_FT-c3650.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/phi4-14B-customLoss_FT-c3650.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 2 rows as 'problem' in 'out_meter'.

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 69.75% |
| Full Accuracy | 58.27% |
| Mean Semantic Similarity | 0.6481 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 2
- Null meters        : 1216

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 73.61% | 61.96% | 0.6476 |
| Indravajrā | 22 | 36.36% | 31.82% | 0.6483 |
| Mālinī | 9 | 22.22% | 22.22% | 0.6936 |
| Sragdharā | 21 | 4.76% | 0.00% | 0.6511 |
| Upendravajrā | 12 | 50.00% | 16.67% | 0.6515 |
| Vasantatilakā | 66 | 12.12% | 1.52% | 0.6234 |
| Vaṃśastha | 18 | 11.11% | 5.56% | 0.6575 |
| Śikhariṇī | 16 | 18.75% | 0.00% | 0.6712 |
| Śālinī | 7 | 0.00% | 0.00% | 0.7775 |
| Śārdūlavikrīḍita | 27 | 11.11% | 7.41% | 0.6852 |

All score/semsim updates saved back to Outputs/phi4-14B-customLoss_FT-c3650.csv
