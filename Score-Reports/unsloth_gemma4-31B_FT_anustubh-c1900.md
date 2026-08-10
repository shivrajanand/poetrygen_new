## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_gemma4-31B_FT_anustubh-c1900.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/unsloth_gemma4-31B_FT_anustubh-c1900.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 415 rows as 'problem' in 'out_meter'.


## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 51.86% |
| Full Accuracy | 35.69% |
| Mean Semantic Similarity | 0.6603 |

(supporting detail)
- Total samples      : 2721
- Problem rows       : 415
- Null meters        : 1335

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 51.86% | 35.69% | 0.6603 |

All score/semsim updates saved back to Outputs/unsloth_gemma4-31B_FT_anustubh-c1900.csv
