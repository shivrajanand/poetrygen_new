## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_phi4_FT_anustubh.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Outputs are clean
Detecting meters: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2721/2721 [00:03<00:00, 697.08it/s]
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 8136.68it/s]

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 76.30% |
| Full Accuracy | 59.87% |
| Mean Semantic Similarity | 0.6546 |

(supporting detail)
- Total samples      : 2721
- Problem rows       : 0
- Null meters        : 1092

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 76.30% | 59.87% | 0.6546 |

All score/semsim updates saved back to Outputs/unsloth_phi4_FT_anustubh.csv