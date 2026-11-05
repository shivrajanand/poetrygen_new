## FILE DETAILS
------------------------------
- FILEPATH: /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_AnuOnly_c1800.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_AnuOnly_c1800.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 1 rows as 'problem' in 'out_meter'.

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 69.00% |
| Full Accuracy | 49.91% |
| Mean Semantic Similarity | 0.6650 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 1
- Null meters        : 1371

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 74.02% | 53.55% | 0.6764 |
| Indravajrā | 22 | 0.00% | 0.00% | 0.5607 |
| Mālinī | 9 | 0.00% | 0.00% | 0.5198 |
| Sragdharā | 21 | 0.00% | 0.00% | 0.3640 |
| Upendravajrā | 12 | 0.00% | 0.00% | 0.5656 |
| Vasantatilakā | 66 | 0.00% | 0.00% | 0.5495 |
| Vaṃśastha | 18 | 0.00% | 0.00% | 0.5618 |
| Śikhariṇī | 16 | 0.00% | 0.00% | 0.4266 |
| Śālinī | 7 | 0.00% | 0.00% | 0.6506 |
| Śārdūlavikrīḍita | 27 | 0.00% | 0.00% | 0.4254 |

All score/semsim updates saved back to /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_AnuOnly_c1800.csv
