## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_phi4_FT_3S-c700.csv
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
| Half Accuracy | 70.37% |
| Full Accuracy | 51.28% |
| Mean Semantic Similarity | 0.6673 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 0
- Null meters        : 1422

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 73.25% | 54.39% | 0.6696 |
| Indravajrā | 22 | 45.45% | 9.09% | 0.6637 |
| Mālinī | 9 | 22.22% | 0.00% | 0.6820 |
| Sragdharā | 21 | 4.76% | 0.00% | 0.6163 |
| Upendravajrā | 12 | 41.67% | 16.67% | 0.6109 |
| Vasantatilakā | 66 | 46.97% | 18.18% | 0.6302 |
| Vaṃśastha | 18 | 33.33% | 5.56% | 0.6248 |
| Śikhariṇī | 16 | 12.50% | 0.00% | 0.6013 |
| Śālinī | 7 | 14.29% | 0.00% | 0.7170 |
| Śārdūlavikrīḍita | 27 | 11.11% | 0.00% | 0.6420 |

All score/semsim updates saved back to Outputs/unsloth_phi4_FT_3S-c700.csv
