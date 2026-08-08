## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_phi4_FT_ZS.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Outputs are clean
Detecting meters: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2919/2919 [00:06<00:00, 443.32it/s]
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 5664.12it/s]

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 66.84% |
| Full Accuracy | 54.74% |
| Mean Semantic Similarity | 0.6647 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 0
- Null meters        : 1320

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 71.30% | 58.73% | 0.6651 |
| Indravajrā | 22 | 4.55% | 0.00% | 0.6512 |
| Mālinī | 9 | 11.11% | 0.00% | 0.7023 |
| Sragdharā | 21 | 0.00% | 0.00% | 0.6162 |
| Upendravajrā | 12 | 0.00% | 0.00% | 0.6848 |
| Vasantatilakā | 66 | 9.09% | 0.00% | 0.6542 |
| Vaṃśastha | 18 | 16.67% | 0.00% | 0.6290 |
| Śikhariṇī | 16 | 0.00% | 0.00% | 0.6865 |
| Śālinī | 7 | 0.00% | 0.00% | 0.7741 |
| Śārdūlavikrīḍita | 27 | 0.00% | 0.00% | 0.6609 |

All score/semsim updates saved back to Outputs/unsloth_phi4_FT_ZS.csv