## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_phi4_FT_0S_notAnustubh-c250.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/unsloth_phi4_FT_0S_notAnustubh-c250.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 35 rows as 'problem' in 'out_meter'.

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 23.40% |
| Full Accuracy | 14.90% |
| Mean Semantic Similarity | 0.5782 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 35
- Null meters        : 2449

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 24.22% | 15.51% | 0.5798 |
| Indravajrā | 22 | 22.73% | 22.73% | 0.6028 |
| Mālinī | 9 | 22.22% | 11.11% | 0.6480 |
| Sragdharā | 21 | 4.76% | 0.00% | 0.5215 |
| Upendravajrā | 12 | 25.00% | 25.00% | 0.4817 |
| Vasantatilakā | 66 | 13.64% | 6.06% | 0.5569 |
| Vaṃśastha | 18 | 5.56% | 0.00% | 0.5450 |
| Śikhariṇī | 16 | 6.25% | 0.00% | 0.5661 |
| Śālinī | 7 | 0.00% | 0.00% | 0.6264 |
| Śārdūlavikrīḍita | 27 | 7.41% | 0.00% | 0.5272 |

All score/semsim updates saved back to Outputs/unsloth_phi4_FT_0S_notAnustubh-c250.csv
