## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_phi4_FT_1S.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Outputs are clean
Detecting meters: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2919/2919 [00:05<00:00, 508.11it/s]
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 8312.29it/s]

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 73.52% |
| Full Accuracy | 58.27% |
| Mean Semantic Similarity | 0.6529 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 0
- Null meters        : 1218

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 76.88% | 62.18% | 0.6560 |
| Indravajrā | 22 | 27.27% | 9.09% | 0.6312 |
| Mālinī | 9 | 0.00% | 0.00% | 0.6756 |
| Sragdharā | 21 | 4.76% | 0.00% | 0.5597 |
| Upendravajrā | 12 | 41.67% | 16.67% | 0.5693 |
| Vasantatilakā | 66 | 43.94% | 6.06% | 0.6328 |
| Vaṃśastha | 18 | 38.89% | 5.56% | 0.5387 |
| Śikhariṇī | 16 | 6.25% | 0.00% | 0.5707 |
| Śālinī | 7 | 57.14% | 0.00% | 0.7456 |
| Śārdūlavikrīḍita | 27 | 3.70% | 0.00% | 0.6061 |

All score/semsim updates saved back to Outputs/unsloth_phi4_FT_1S.csv