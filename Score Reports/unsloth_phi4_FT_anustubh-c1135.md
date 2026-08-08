## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_gemma4-31B_FT_anustubh-c1135.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/unsloth_gemma4-31B_FT_anustubh-c1135.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 377 rows as 'problem' in 'out_meter'.
Detecting meters: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2721/2721 [00:04<00:00, 595.73it/s]
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 7874.91it/s]

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 54.32% |
| Full Accuracy | 37.38% |
| Mean Semantic Similarity | 0.6595 |

(supporting detail)
- Total samples      : 2721
- Problem rows       : 377
- Null meters        : 1327

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 54.32% | 37.38% | 0.6595 |

All score/semsim updates saved back to Outputs/unsloth_gemma4-31B_FT_anustubh-c1135.csv