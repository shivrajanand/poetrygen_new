## FILE DETAILS
------------------------------
- FILEPATH: /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_VasantOnly_c50.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Outputs are clean
Detecting meters:   0%|          | 0/66 [00:00<?, ?it/s]Detecting meters:  23%|██▎       | 15/66 [00:00<00:00, 147.31it/s]Detecting meters:  50%|█████     | 33/66 [00:00<00:00, 161.03it/s]Detecting meters:  77%|███████▋  | 51/66 [00:00<00:00, 165.85it/s]Detecting meters: 100%|██████████| 66/66 [00:00<00:00, 162.80it/s]
Loading weights:   0%|          | 0/391 [00:00<?, ?it/s]Loading weights: 100%|██████████| 391/391 [00:00<00:00, 11100.70it/s]

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 15.15% |
| Full Accuracy | 3.03% |
| Mean Semantic Similarity | 0.6085 |

(supporting detail)
- Total samples      : 66
- Problem rows       : 0
- Null meters        : 64

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Vasantatilakā | 66 | 15.15% | 3.03% | 0.6085 |

All score/semsim updates saved back to /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_VasantOnly_c50.csv
