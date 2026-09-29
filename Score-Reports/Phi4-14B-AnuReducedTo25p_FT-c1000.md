## FILE DETAILS
------------------------------
- FILEPATH: Outputs/Phi4-14B-AnuReducedTo25p_FT-c1000.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/Phi4-14B-AnuReducedTo25p_FT-c1000.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 4 rows as 'problem' in 'out_meter'.

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 59.34% |
| Full Accuracy | 45.84% |
| Mean Semantic Similarity | 0.6353 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 4
- Null meters        : 1577

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 62.95% | 48.92% | 0.6350 |
| Indravajrā | 22 | 18.18% | 9.09% | 0.6427 |
| Mālinī | 9 | 22.22% | 11.11% | 0.7000 |
| Sragdharā | 21 | 9.52% | 0.00% | 0.6309 |
| Upendravajrā | 12 | 0.00% | 0.00% | 0.5878 |
| Vasantatilakā | 66 | 9.09% | 4.55% | 0.6357 |
| Vaṃśastha | 18 | 5.56% | 0.00% | 0.5988 |
| Śikhariṇī | 16 | 6.25% | 0.00% | 0.6441 |
| Śālinī | 7 | 0.00% | 0.00% | 0.6913 |
| Śārdūlavikrīḍita | 27 | 11.11% | 3.70% | 0.6654 |

All score/semsim updates saved back to Outputs/Phi4-14B-AnuReducedTo25p_FT-c1000.csv
