## FILE DETAILS
------------------------------
- FILEPATH: Outputs/phi4-14B-customLoss-V2_FT-c4380.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/phi4-14B-customLoss-V2_FT-c4380.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 4 rows as 'problem' in 'out_meter'.


## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 65.33% |
| Full Accuracy | 56.15% |
| Mean Semantic Similarity | 0.6576 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 4
- Null meters        : 1274

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 69.31% | 59.83% | 0.6570 |
| Indravajrā | 22 | 18.18% | 13.64% | 0.6862 |
| Mālinī | 9 | 33.33% | 22.22% | 0.6777 |
| Sragdharā | 21 | 4.76% | 0.00% | 0.6369 |
| Upendravajrā | 12 | 25.00% | 25.00% | 0.6540 |
| Vasantatilakā | 66 | 6.06% | 3.03% | 0.6680 |
| Vaṃśastha | 18 | 22.22% | 5.56% | 0.6240 |
| Śikhariṇī | 16 | 6.25% | 0.00% | 0.6859 |
| Śālinī | 7 | 0.00% | 0.00% | 0.7500 |
| Śārdūlavikrīḍita | 27 | 3.70% | 0.00% | 0.6667 |

All score/semsim updates saved back to Outputs/phi4-14B-customLoss-V2_FT-c4380.csv
