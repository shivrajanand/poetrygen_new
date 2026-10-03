## FILE DETAILS
------------------------------
- FILEPATH: Outputs/phi4-14B-grpo-FT-c1540.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 93.22% |
| Full Accuracy | 91.88% |
| Mean Semantic Similarity | 0.6815 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 0
- Null meters        : 231

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 97.28% | 96.69% | 0.6860 |
| Indravajrā | 22 | 50.00% | 40.91% | 0.6237 |
| Mālinī | 9 | 22.22% | 11.11% | 0.6929 |
| Sragdharā | 21 | 23.81% | 4.76% | 0.6178 |
| Upendravajrā | 12 | 66.67% | 50.00% | 0.6417 |
| Vasantatilakā | 66 | 57.58% | 40.91% | 0.6018 |
| Vaṃśastha | 18 | 22.22% | 16.67% | 0.5728 |
| Śikhariṇī | 16 | 6.25% | 0.00% | 0.6233 |
| Śālinī | 7 | 0.00% | 0.00% | 0.7137 |
| Śārdūlavikrīḍita | 27 | 18.52% | 14.81% | 0.6269 |

All score/semsim updates saved back to Outputs/phi4-14B-grpo-FT-c1540.csv
