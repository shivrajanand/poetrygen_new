## FILE DETAILS
------------------------------
- FILEPATH: Outputs/Qwen3_8/unsloth_qwen38_27b_FT_0S_c1400.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/Qwen3_8/unsloth_qwen38_27b_FT_0S_c1400.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 1 rows as 'problem' in 'out_meter'.

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 69.06% |
| Full Accuracy | 48.82% |
| Mean Semantic Similarity | 0.6825 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 1
- Null meters        : 1493

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 73.02% | 52.30% | 0.6826 |
| Indravajrā | 22 | 27.27% | 4.55% | 0.6978 |
| Mālinī | 9 | 0.00% | 0.00% | 0.7267 |
| Sragdharā | 21 | 0.00% | 0.00% | 0.6910 |
| Upendravajrā | 12 | 25.00% | 0.00% | 0.6693 |
| Vasantatilakā | 66 | 22.73% | 1.52% | 0.6911 |
| Vaṃśastha | 18 | 0.00% | 0.00% | 0.6677 |
| Śikhariṇī | 16 | 18.75% | 0.00% | 0.6120 |
| Śālinī | 7 | 14.29% | 0.00% | 0.7401 |
| Śārdūlavikrīḍita | 27 | 3.70% | 0.00% | 0.6561 |

All score/semsim updates saved back to Outputs/Qwen3_8/unsloth_qwen38_27b_FT_0S_c1400.csv