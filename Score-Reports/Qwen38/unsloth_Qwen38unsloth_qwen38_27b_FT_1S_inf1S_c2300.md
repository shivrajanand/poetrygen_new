## FILE DETAILS
------------------------------
- FILEPATH: /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_1S_inf1S_c2300.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_1S_inf1S_c2300.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 1 rows as 'problem' in 'out_meter'.

## Overall Evaluation
FT
| Metric | Value |
|--------|------:|
| Half Accuracy | 74.34% |
| Full Accuracy | 56.01% |
| Mean Semantic Similarity | 0.6710 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 1
- Null meters        : 1283

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 77.25% | 59.39% | 0.6728 |
| Indravajrā | 22 | 45.45% | 9.09% | 0.6056 |
| Mālinī | 9 | 22.22% | 11.11% | 0.6151 |
| Sragdharā | 21 | 4.76% | 0.00% | 0.6792 |
| Upendravajrā | 12 | 41.67% | 8.33% | 0.6496 |
| Vasantatilakā | 66 | 48.48% | 18.18% | 0.6609 |
| Vaṃśastha | 18 | 33.33% | 5.56% | 0.5785 |
| Śikhariṇī | 16 | 18.75% | 0.00% | 0.6289 |
| Śālinī | 7 | 14.29% | 0.00% | 0.7333 |
| Śārdūlavikrīḍita | 27 | 29.63% | 7.41% | 0.6619 |

All score/semsim updates saved back to /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_1S_inf1S_c2300.csv
