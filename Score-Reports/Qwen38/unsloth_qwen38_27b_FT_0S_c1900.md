## FILE DETAILS
------------------------------
- FILEPATH: Outputs/Qwen3_8/unsloth_qwen38_27b_FT_0S_c1900.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Outputs are clean

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 72.39% |
| Full Accuracy | 54.68% |
| Mean Semantic Similarity | 0.6746 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 0
- Null meters        : 1323

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 76.07% | 58.29% | 0.6752 |
| Indravajrā | 22 | 22.73% | 0.00% | 0.6598 |
| Mālinī | 9 | 22.22% | 11.11% | 0.6902 |
| Sragdharā | 21 | 0.00% | 0.00% | 0.6711 |
| Upendravajrā | 12 | 25.00% | 16.67% | 0.6631 |
| Vasantatilakā | 66 | 31.82% | 10.61% | 0.6619 |
| Vaṃśastha | 18 | 16.67% | 0.00% | 0.6072 |
| Śikhariṇī | 16 | 12.50% | 0.00% | 0.7008 |
| Śālinī | 7 | 14.29% | 0.00% | 0.7700 |
| Śārdūlavikrīḍita | 27 | 22.22% | 0.00% | 0.6698 |

All score/semsim updates saved back to Outputs/Qwen3_8/unsloth_qwen38_27b_FT_0S_c1900.csv
