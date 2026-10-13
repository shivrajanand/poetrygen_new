## FILE DETAILS
------------------------------
- FILEPATH: Outputs/phi4-14B-grpo-FT-c770.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/phi4-14B-grpo-FT-c770.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 2 rows as 'problem' in 'out_meter'.

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 91.23% |
| Full Accuracy | 88.25% |
| Mean Semantic Similarity | 0.6579 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 2
- Null meters        : 340

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 95.70% | 93.38% | 0.6600 |
| Indravajrā | 22 | 45.45% | 36.36% | 0.6387 |
| Mālinī | 9 | 33.33% | 11.11% | 0.6779 |
| Sragdharā | 21 | 14.29% | 0.00% | 0.6088 |
| Upendravajrā | 12 | 25.00% | 16.67% | 0.6248 |
| Vasantatilakā | 66 | 46.97% | 27.27% | 0.6119 |
| Vaṃśastha | 18 | 16.67% | 16.67% | 0.5961 |
| Śikhariṇī | 16 | 0.00% | 0.00% | 0.6644 |
| Śālinī | 7 | 28.57% | 0.00% | 0.7010 |
| Śārdūlavikrīḍita | 27 | 14.81% | 11.11% | 0.6506 |

All score/semsim updates saved back to Outputs/phi4-14B-grpo-FT-c770.csv
