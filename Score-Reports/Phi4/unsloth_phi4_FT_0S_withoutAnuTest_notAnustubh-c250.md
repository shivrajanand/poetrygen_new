## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_phi4_FT_0S_withoutAnuTest_notAnustubh-c250.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/unsloth_phi4_FT_0S_withoutAnuTest_notAnustubh-c250.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 7 rows as 'problem' in 'out_meter'.

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 14.14% |
| Full Accuracy | 8.59% |
| Mean Semantic Similarity | 0.5585 |

(supporting detail)
- Total samples      : 198
- Problem rows       : 7
- Null meters        : 174

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Indravajrā | 22 | 31.82% | 27.27% | 0.5939 |
| Mālinī | 9 | 11.11% | 11.11% | 0.5783 |
| Sragdharā | 21 | 4.76% | 0.00% | 0.5135 |
| Upendravajrā | 12 | 25.00% | 16.67% | 0.5391 |
| Vasantatilakā | 66 | 16.67% | 12.12% | 0.5701 |
| Vaṃśastha | 18 | 0.00% | 0.00% | 0.5520 |
| Śikhariṇī | 16 | 0.00% | 0.00% | 0.5440 |
| Śālinī | 7 | 0.00% | 0.00% | 0.6554 |
| Śārdūlavikrīḍita | 27 | 18.52% | 0.00% | 0.5265 |

All score/semsim updates saved back to Outputs/unsloth_phi4_FT_0S_withoutAnuTest_notAnustubh-c250.csv
