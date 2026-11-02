## FILE DETAILS
------------------------------
- FILEPATH: /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_1S_inf1S_c1400.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_1S_inf1S_c1400.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 1 rows as 'problem' in 'out_meter'.

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 69.27% |
| Full Accuracy | 45.19% |
| Mean Semantic Similarity | 0.6847 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 1
- Null meters        : 1599

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 72.18% | 47.96% | 0.6866 |
| Indravajrā | 22 | 50.00% | 13.64% | 0.6433 |
| Mālinī | 9 | 22.22% | 0.00% | 0.6959 |
| Sragdharā | 21 | 9.52% | 0.00% | 0.6664 |
| Upendravajrā | 12 | 41.67% | 0.00% | 0.6290 |
| Vasantatilakā | 66 | 25.76% | 10.61% | 0.6709 |
| Vaṃśastha | 18 | 50.00% | 16.67% | 0.5998 |
| Śikhariṇī | 16 | 25.00% | 0.00% | 0.6125 |
| Śālinī | 7 | 42.86% | 14.29% | 0.7293 |
| Śārdūlavikrīḍita | 27 | 18.52% | 0.00% | 0.6906 |

All score/semsim updates saved back to /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_1S_inf1S_c1400.csv
