## FILE DETAILS
------------------------------
- FILEPATH: Outputs/phi4-14B-customLoss_FT-c4380.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/phi4-14B-customLoss_FT-c4380.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 3 rows as 'problem' in 'out_meter'.

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 67.90% |
| Full Accuracy | 55.94% |
| Mean Semantic Similarity | 0.6469 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 3
- Null meters        : 1282

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 71.55% | 59.39% | 0.6461 |
| Indravajrā | 22 | 27.27% | 22.73% | 0.6466 |
| Mālinī | 9 | 33.33% | 22.22% | 0.6780 |
| Sragdharā | 21 | 9.52% | 0.00% | 0.6416 |
| Upendravajrā | 12 | 33.33% | 16.67% | 0.6546 |
| Vasantatilakā | 66 | 18.18% | 7.58% | 0.6511 |
| Vaṃśastha | 18 | 11.11% | 5.56% | 0.6536 |
| Śikhariṇī | 16 | 12.50% | 0.00% | 0.6748 |
| Śālinī | 7 | 14.29% | 0.00% | 0.7124 |
| Śārdūlavikrīḍita | 27 | 11.11% | 7.41% | 0.6728 |

All score/semsim updates saved back to Outputs/phi4-14B-customLoss_FT-c4380.csv
