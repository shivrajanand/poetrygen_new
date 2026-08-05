## FILE DETAILS
------------------------------
- FILEPATH: Outputs/Vasantatalika-only-Phi4/unsloth_phi4_FT_vasantatalika-c70.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/Vasantatalika-only-Phi4/unsloth_phi4_FT_vasantatalika-c70.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 1 rows as 'problem' in 'out_meter'.
Detecting meters: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 66/66 [00:00<00:00, 128.04it/s]
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 3710.04it/s]

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 9.23% |
| Full Accuracy | 3.08% |
| Mean Semantic Similarity | 0.6545 |

(supporting detail)
- Total samples      : 65
- Problem rows       : 1
- Null meters        : 63

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Vasantatilakā | 65 | 9.23% | 3.08% | 0.6545 |

All score/semsim updates saved back to Outputs/Vasantatalika-only-Phi4/unsloth_phi4_FT_vasantatalika-c70.csv