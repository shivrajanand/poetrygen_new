## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_phi4_FT_0S_notAnustubh.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to Outputs/unsloth_phi4_FT_0S_notAnustubh.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 1 rows as 'problem' in 'out_meter'.

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 28.30% |
| Full Accuracy | 19.32% |
| Mean Semantic Similarity | 0.6309 |

(supporting detail)
- Total samples      : 2919
- Problem rows       : 1
- Null meters        : 2353

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Anuṣṭubh | 2721 | 29.29% | 20.29% | 0.6317 |
| Indravajrā | 22 | 27.27% | 9.09% | 0.6331 |
| Mālinī | 9 | 22.22% | 11.11% | 0.6662 |
| Sragdharā | 21 | 9.52% | 0.00% | 0.5950 |
| Upendravajrā | 12 | 25.00% | 16.67% | 0.6191 |
| Vasantatilakā | 66 | 18.18% | 10.61% | 0.6189 |
| Vaṃśastha | 18 | 0.00% | 0.00% | 0.5931 |
| Śikhariṇī | 16 | 0.00% | 0.00% | 0.6041 |
| Śālinī | 7 | 0.00% | 0.00% | 0.7114 |
| Śārdūlavikrīḍita | 27 | 14.81% | 0.00% | 0.6257 |

All score/semsim updates saved back to Outputs/unsloth_phi4_FT_0S_notAnustubh.csv
