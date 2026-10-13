## FILE DETAILS
------------------------------
- FILEPATH: Outputs/phi4-14B-customLoss-V2_FT-c7300.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/phi4-14B-customLoss-V2_FT-c7300.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 10 rows as 'problem' in 'out_meter'.

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 66.46% |
| Full Accuracy | 56.70% |
| Mean Semantic Similarity | 0.6575 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 10
- Null meters        : 1254

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 70.64% | 60.42% | 0.6567 |
| Indravajrā | 22 | 13.64% | 13.64% | 0.6964 |
| Mālinī | 9 | 55.56% | 22.22% | 0.7031 |
| Sragdharā | 21 | 4.76% | 0.00% | 0.6168 |
| Upendravajrā | 12 | 16.67% | 16.67% | 0.6785 |
| Vasantatilakā | 66 | 6.06% | 4.55% | 0.6640 |
| Vaṃśastha | 18 | 5.56% | 5.56% | 0.6144 |
| Śikhariṇī | 16 | 6.25% | 0.00% | 0.6780 |
| Śālinī | 7 | 0.00% | 0.00% | 0.8014 |
| Śārdūlavikrīḍita | 27 | 3.70% | 0.00% | 0.6710 |

All score/semsim updates saved back to Outputs/phi4-14B-customLoss-V2_FT-c7300.csv
