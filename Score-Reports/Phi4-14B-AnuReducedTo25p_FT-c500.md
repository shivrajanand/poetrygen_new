## FILE DETAILS
------------------------------
- FILEPATH: Outputs/Phi4-14B-AnuReducedTo25p_FT-c500.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Outputs are clean

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 50.43% |
| Full Accuracy | 36.55% |
| Mean Semantic Similarity | 0.6536 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 0
- Null meters        : 1852

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 53.47% | 39.18% | 0.6539 |
| Indravajrā | 22 | 9.09% | 0.00% | 0.6349 |
| Mālinī | 9 | 0.00% | 0.00% | 0.7075 |
| Sragdharā | 21 | 9.52% | 0.00% | 0.6541 |
| Upendravajrā | 12 | 0.00% | 0.00% | 0.6336 |
| Vasantatilakā | 66 | 13.64% | 1.52% | 0.6639 |
| Vaṃśastha | 18 | 5.56% | 0.00% | 0.6399 |
| Śikhariṇī | 16 | 0.00% | 0.00% | 0.6380 |
| Śālinī | 7 | 14.29% | 0.00% | 0.6833 |
| Śārdūlavikrīḍita | 27 | 7.41% | 0.00% | 0.6167 |

All score/semsim updates saved back to Outputs/Phi4-14B-AnuReducedTo25p_FT-c500.csv
