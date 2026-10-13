## FILE DETAILS
------------------------------
- FILEPATH: Outputs/unsloth_phi4_FT_0S_withoutAnuTest_notAnustubh-c100.csv
- INPUT_COL: hi
- GROUND_TRUTH: meter_cd
- GROUND_TRUTH_SYLLABLES: syllable_count
- PRED_COL: model_out
- PRED_METER: out_meter
- PRED_SYLLABLES: pred_syllable_count

Outputs are clean
Detecting meters:   0%|          | 0/198 [00:00<?, ?it/s]Detecting meters:   5%|▌         | 10/198 [00:00<00:01, 97.76it/s]Detecting meters:  10%|█         | 20/198 [00:00<00:01, 89.98it/s]Detecting meters:  16%|█▌        | 31/198 [00:00<00:01, 97.80it/s]Detecting meters:  22%|██▏       | 43/198 [00:00<00:01, 100.96it/s]Detecting meters:  27%|██▋       | 54/198 [00:00<00:01, 98.58it/s] Detecting meters:  32%|███▏      | 64/198 [00:00<00:01, 93.00it/s]Detecting meters:  37%|███▋      | 74/198 [00:00<00:01, 93.80it/s]Detecting meters:  43%|████▎     | 85/198 [00:00<00:01, 97.60it/s]Detecting meters:  49%|████▉     | 98/198 [00:00<00:00, 104.60it/s]Detecting meters:  55%|█████▌    | 109/198 [00:01<00:00, 103.09it/s]Detecting meters:  61%|██████    | 121/198 [00:01<00:00, 105.77it/s]Detecting meters:  67%|██████▋   | 132/198 [00:01<00:00, 99.07it/s] Detecting meters:  72%|███████▏  | 143/198 [00:01<00:00, 98.48it/s]Detecting meters:  79%|███████▉  | 156/198 [00:01<00:00, 104.10it/s]Detecting meters:  84%|████████▍ | 167/198 [00:01<00:00, 101.08it/s]Detecting meters:  90%|█████████ | 179/198 [00:01<00:00, 105.97it/s]Detecting meters:  96%|█████████▋| 191/198 [00:01<00:00, 107.56it/s]Detecting meters: 100%|██████████| 198/198 [00:01<00:00, 100.65it/s]
Loading weights:   0%|          | 0/391 [00:00<?, ?it/s]Loading weights: 100%|██████████| 391/391 [00:00<00:00, 6910.85it/s]

## Overall Evaluation

| Metric | Value |
|--------|------:|
| Half Accuracy | 15.66% |
| Full Accuracy | 6.57% |
| Mean Semantic Similarity | 0.6129 |

(supporting detail)
- Total samples      : 198
- Problem rows       : 0
- Null meters        : 185

## Meter-wise Evaluation

| Meter | Samples | Half Accuracy | Full Accuracy | Mean Semantic Similarity |
|-------|--------:|--------------:|--------------:|-------------------------:|
| Indravajrā | 22 | 22.73% | 9.09% | 0.6295 |
| Mālinī | 9 | 22.22% | 11.11% | 0.6627 |
| Sragdharā | 21 | 4.76% | 0.00% | 0.5966 |
| Upendravajrā | 12 | 33.33% | 16.67% | 0.6078 |
| Vasantatilakā | 66 | 24.24% | 12.12% | 0.6204 |
| Vaṃśastha | 18 | 5.56% | 0.00% | 0.6059 |
| Śikhariṇī | 16 | 6.25% | 0.00% | 0.5875 |
| Śālinī | 7 | 0.00% | 0.00% | 0.6358 |
| Śārdūlavikrīḍita | 27 | 3.70% | 0.00% | 0.5932 |

All score/semsim updates saved back to Outputs/unsloth_phi4_FT_0S_withoutAnuTest_notAnustubh-c100.csv
