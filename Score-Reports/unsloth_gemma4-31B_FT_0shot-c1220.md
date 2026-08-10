## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_gemma4-31B_FT_0shot-c1220.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/unsloth_gemma4-31B_FT_0shot-c1220.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 225 rows as 'problem' in 'out_meter'.


## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 54.03% |
| Full Accuracy | 38.81% |
| Mean Semantic Similarity | 0.6348 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 225
- Null meters        : 1561

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 57.30% | 41.57% | 0.6339 |
| Indravajrā | 22 | 18.18% | 0.00% | 0.6466 |
| Mālinī | 9 | 11.11% | 11.11% | 0.6279 |
| Sragdharā | 21 | 4.76% | 0.00% | 0.6302 |
| Upendravajrā | 12 | 25.00% | 0.00% | 0.6839 |
| Vasantatilakā | 66 | 6.06% | 0.00% | 0.6537 |
| Vaṃśastha | 18 | 11.11% | 0.00% | 0.5772 |
| Śikhariṇī | 16 | 0.00% | 0.00% | 0.6672 |
| Śālinī | 7 | 0.00% | 0.00% | 0.7175 |
| Śārdūlavikrīḍita | 27 | 11.11% | 3.70% | 0.6560 |

All score/semsim updates saved back to Outputs/unsloth_gemma4-31B_FT_0shot-c1220.csv
