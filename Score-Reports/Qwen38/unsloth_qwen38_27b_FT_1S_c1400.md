## FILE DETAILS
------------------------------
- FILEPATH: Outputs/Qwen3_8/unsloth_qwen38_27b_FT_1S_c1400.csv
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
| Half Accuracy | 32.82% |
| Full Accuracy | 15.59% |
| Mean Semantic Similarity | 0.7038 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 0
- Null meters        : 2464

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 34.99% | 16.72% | 0.7059 |
| Indravajrā | 22 | 9.09% | 0.00% | 0.6685 |
| Mālinī | 9 | 11.11% | 0.00% | 0.6551 |
| Sragdharā | 21 | 0.00% | 0.00% | 0.6505 |
| Upendravajrā | 12 | 0.00% | 0.00% | 0.7173 |
| Vasantatilakā | 66 | 3.03% | 0.00% | 0.7133 |
| Vaṃśastha | 18 | 5.56% | 0.00% | 0.6679 |
| Śikhariṇī | 16 | 0.00% | 0.00% | 0.6053 |
| Śālinī | 7 | 0.00% | 0.00% | 0.7729 |
| Śārdūlavikrīḍita | 27 | 0.00% | 0.00% | 0.6120 |

All score/semsim updates saved back to Outputs/Qwen3_8/unsloth_qwen38_27b_FT_1S_c1400.csv
