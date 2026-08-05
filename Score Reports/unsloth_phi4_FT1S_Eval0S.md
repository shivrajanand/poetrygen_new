## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_phi4_FT1S_Eval0S.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Outputs are clean
Detecting meters: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2919/2919 [00:09<00:00, 311.67it/s]
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 4729.79it/s]

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 37.86% |
| Full Accuracy | 25.49% |
| Mean Semantic Similarity | 0.6704 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 0
- Null meters        : 2150

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 40.46% | 27.34% | 0.6785 |
| Indravajrā | 22 | 0.00% | 0.00% | 0.6161 |
| Mālinī | 9 | 0.00% | 0.00% | 0.6203 |
| Sragdharā | 21 | 0.00% | 0.00% | 0.4229 |
| Upendravajrā | 12 | 0.00% | 0.00% | 0.6298 |
| Vasantatilakā | 66 | 4.55% | 0.00% | 0.5889 |
| Vaṃśastha | 18 | 5.56% | 0.00% | 0.5686 |
| Śikhariṇī | 16 | 0.00% | 0.00% | 0.5408 |
| Śālinī | 7 | 0.00% | 0.00% | 0.6552 |
| Śārdūlavikrīḍita | 27 | 0.00% | 0.00% | 0.4764 |

All score/semsim updates saved back to Outputs/unsloth_phi4_FT1S_Eval0S.csv