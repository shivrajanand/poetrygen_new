## FILE DETAILS
------------------------------
- FILEPATH: Outputs/phi4-14B-customLoss-V2_FT-c2190.csv
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
| Half Accuracy | 60.57% |
| Full Accuracy | 50.63% |
| Mean Semantic Similarity | 0.6673 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 0
- Null meters        : 1439

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 64.42% | 54.21% | 0.6681 |
| Indravajrā | 22 | 9.09% | 0.00% | 0.6848 |
| Mālinī | 9 | 0.00% | 0.00% | 0.6532 |
| Sragdharā | 21 | 0.00% | 0.00% | 0.6210 |
| Upendravajrā | 12 | 8.33% | 0.00% | 0.6472 |
| Vasantatilakā | 66 | 9.09% | 3.03% | 0.6691 |
| Vaṃśastha | 18 | 16.67% | 5.56% | 0.6372 |
| Śikhariṇī | 16 | 0.00% | 0.00% | 0.6696 |
| Śālinī | 7 | 0.00% | 0.00% | 0.6876 |
| Śārdūlavikrīḍita | 27 | 11.11% | 0.00% | 0.6316 |

All score/semsim updates saved back to Outputs/phi4-14B-customLoss-V2_FT-c2190.csv
