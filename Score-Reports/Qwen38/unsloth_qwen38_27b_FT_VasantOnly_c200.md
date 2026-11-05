## FILE DETAILS
------------------------------
- FILEPATH: /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_VasantOnly_c200.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Problematic rows saved to /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_VasantOnly_c200.csv.
Letter '5' is ignored because models sometimes use it for avagraha (ऽ).
Marked 1 rows as 'problem' in 'out_meter'.
Detecting meters:   0%|          | 0/66 [00:00<?, ?it/s]Detecting meters:  20%|█▉        | 13/66 [00:00<00:00, 127.58it/s]Detecting meters:  39%|███▉      | 26/66 [00:00<00:00, 108.96it/s]Detecting meters:  67%|██████▋   | 44/66 [00:00<00:00, 135.36it/s]Detecting meters:  97%|█████████▋| 64/66 [00:00<00:00, 158.34it/s]Detecting meters: 100%|██████████| 66/66 [00:00<00:00, 146.21it/s]
Loading weights:   0%|          | 0/391 [00:00<?, ?it/s]Loading weights: 100%|██████████| 391/391 [00:00<00:00, 11244.32it/s]

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 18.18% |
| Full Accuracy | 3.03% |
| Mean Semantic Similarity | 0.5981 |

(supporting detail)
- Total samples      : 66
- Problem rows       : 1
- Null meters        : 63

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Vasantatilakā | 66 | 18.18% | 3.03% | 0.5981 |

All score/semsim updates saved back to /home/shivraj-pg/poetrygen_new/Outputs/Qwen3_8/unsloth_qwen38_27b_FT_VasantOnly_c200.csv
