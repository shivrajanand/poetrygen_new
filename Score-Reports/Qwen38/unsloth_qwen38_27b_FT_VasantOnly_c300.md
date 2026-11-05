## FILE DETAILS
------------------------------
- FILEPATH: /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_VasantOnly_c300.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_VasantOnly_c300.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 1 rows as 'problem' in 'out_meter'.
Detecting meters:   0%|          | 0/66 [00:00<?, ?it/s]Detecting meters:  20%|█▉        | 13/66 [00:00<00:00, 128.38it/s]Detecting meters:  45%|████▌     | 30/66 [00:00<00:00, 150.08it/s]Detecting meters:  71%|███████   | 47/66 [00:00<00:00, 157.56it/s]Detecting meters:  97%|█████████▋| 64/66 [00:00<00:00, 159.89it/s]Detecting meters: 100%|██████████| 66/66 [00:00<00:00, 155.62it/s]
Loading weights:   0%|          | 0/391 [00:00<?, ?it/s]Loading weights: 100%|██████████| 391/391 [00:00<00:00, 11021.69it/s]

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 25.76% |
| Full Accuracy | 1.52% |
| Mean Semantic Similarity | 0.6014 |

(supporting detail)
- Total samples      : 66
- Problem rows       : 1
- Null meters        : 64

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Vasantatilakā | 66 | 25.76% | 1.52% | 0.6014 |

All score/semsim updates saved back to /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_VasantOnly_c300.csv
