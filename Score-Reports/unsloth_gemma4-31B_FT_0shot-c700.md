## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_gemma4-31B_FT_0shot-c700.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/unsloth_gemma4-31B_FT_0shot-c700.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 1 rows as 'problem' in 'out_meter'.

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 63.38% |
| Full Accuracy | 44.26% |
| Mean Semantic Similarity | 0.6754 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 1
- Null meters        : 1626

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 67.03% | 47.41% | 0.6738 |
| Indravajrā | 22 | 40.91% | 0.00% | 0.6832 |
| Mālinī | 9 | 11.11% | 11.11% | 0.6920 |
| Sragdharā | 21 | 4.76% | 0.00% | 0.6834 |
| Upendravajrā | 12 | 8.33% | 0.00% | 0.7373 |
| Vasantatilakā | 66 | 10.61% | 0.00% | 0.7138 |
| Vaṃśastha | 18 | 33.33% | 0.00% | 0.6138 |
| Śikhariṇī | 16 | 0.00% | 0.00% | 0.6944 |
| Śālinī | 7 | 0.00% | 0.00% | 0.7421 |
| Śārdūlavikrīḍita | 27 | 3.70% | 3.70% | 0.7047 |

All score/semsim updates saved back to Outputs/unsloth_gemma4-31B_FT_0shot-c700.csv
