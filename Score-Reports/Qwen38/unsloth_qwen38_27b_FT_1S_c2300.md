## FILE DETAILS
------------------------------
- FILEPATH: Outputs/Qwen3_8/unsloth_qwen38_27b_FT_1S_c2300.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/Qwen3_8/unsloth_qwen38_27b_FT_1S_c2300.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 3 rows as 'problem' in 'out_meter'.

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 34.40% |
| Full Accuracy | 20.25% |
| Mean Semantic Similarity | 0.6943 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 3
- Null meters        : 2325

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 36.57% | 21.72% | 0.6954 |
| Indravajrā | 22 | 4.55% | 0.00% | 0.6452 |
| Mālinī | 9 | 11.11% | 0.00% | 0.6726 |
| Sragdharā | 21 | 4.76% | 0.00% | 0.5971 |
| Upendravajrā | 12 | 0.00% | 0.00% | 0.6991 |
| Vasantatilakā | 66 | 7.58% | 0.00% | 0.7051 |
| Vaṃśastha | 18 | 5.56% | 0.00% | 0.7260 |
| Śikhariṇī | 16 | 0.00% | 0.00% | 0.6789 |
| Śālinī | 7 | 0.00% | 0.00% | 0.7412 |
| Śārdūlavikrīḍita | 27 | 0.00% | 0.00% | 0.6587 |

All score/semsim updates saved back to Outputs/Qwen3_8/unsloth_qwen38_27b_FT_1S_c2300.csv
