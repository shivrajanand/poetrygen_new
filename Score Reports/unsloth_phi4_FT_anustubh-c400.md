## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_gemma4-31B_FT_anustubh-c400.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/unsloth_gemma4-31B_FT_anustubh-c400.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 2 rows as 'problem' in 'out_meter'.
Detecting meters: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2721/2721 [00:05<00:00, 508.08it/s]
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 7434.72it/s]

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 62.92% |
| Full Accuracy | 41.57% |
| Mean Semantic Similarity | 0.6868 |

(supporting detail)
- Total samples      : 2721
- Problem rows       : 2
- Null meters        : 1588

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 62.92% | 41.57% | 0.6868 |

All score/semsim updates saved back to Outputs/unsloth_gemma4-31B_FT_anustubh-c400.csv