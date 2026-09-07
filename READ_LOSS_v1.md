# Loss Function — Experiment 1 (Length-Normalized + Meter-Weighted)

**Script:** `train_meter_weighted.py` (original)
**Output dir:** `Trained_Models/Phi4-14B-customLoss`

## 1. What it does

For every training row `(prompt, hi, sa)`, the model is trained to generate the Sanskrit verse `sa` given `prompt + hi` as context. Loss is computed **only on the response tokens** (`sa`) — the prompt/instruction tokens are masked out with `label = -100` and contribute nothing to the loss.

On top of that base setup, this version applies two corrections to the per-example loss before averaging over the batch:

1. **Length normalization** — divide the summed token loss by the number of response tokens, so a long verse and a short verse contribute comparably to the loss instead of the long one dominating just because it has more tokens.
2. **Meter-frequency weighting** — multiply that per-token-averaged loss by a weight tied to how rare the row's meter (`meter_cd`) is in the training data, so rare meters (e.g. Śālinī, Mālinī) push the gradient roughly as hard as the extremely common Anuṣṭubh.

## 2. Step-by-step

**Step A — token-level cross-entropy**
Standard causal-LM next-token cross-entropy is computed for every response token `t` in example `i`:

```
CE_{i,t} = CrossEntropy(logits_{i,t-1}, label_{i,t})
```

Prompt tokens are excluded via the `-100` label mask (`valid_mask`).

**Step B — length normalization (this is the part removed in v2)**
The per-example loss is the *mean* CE over its `T_i` response tokens:

```
L_i^CE = (1 / T_i) * sum_t CE_{i,t}
```

So a 5-token verse and a 40-token verse each produce one loss value on a comparable scale, regardless of how many tokens they have.

**Step C — meter weight `w_i`**
Computed once, before training, directly from the training-split meter counts:

```
w_raw(m) = 1 / sqrt(N_m)              # N_m = number of training rows with meter m
w(m)     = w_raw(m) / mean_j(w_raw(j))  # normalize so the 10 weights average to 1.0
```

The `1/sqrt(N_m)` shape means the correction is gentle: a meter with 100x fewer rows gets ~10x the weight, not ~100x. The final `/ mean` rescaling keeps the *average* weight at 1.0, so the weighting redistributes emphasis across meters without changing the overall loss magnitude.

**Step D — combine**

```
L_i     = w(meter_i) * L_i^CE
L_batch = mean_i(L_i)
```

## 3. Code (from `compute_loss`)

```python
valid_mask = (shift_labels != -100).float()

# Step B: per-token average
sequence_loss = (token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)

# Step C: meter weight lookup
weights = self.meter_weight_tensor.to(sequence_loss.device, sequence_loss.dtype)[meter_ids]

# Step D: combine + batch mean
weighted_sequence_loss = sequence_loss * weights
loss = weighted_sequence_loss.mean()
```

## 4. Computed weights (from actual train-split counts)

| Meter | Count (N_m) | 1/√N_m (raw) | Final weight w(m) |
|---|---:|---:|---:|
| Anuṣṭubh | 21,766 | 0.006778 | **0.0852** |
| Vasantatilakā | 529 | 0.043478 | **0.5467** |
| Śārdūlavikrīḍita | 212 | 0.068680 | **0.8637** |
| Indravajrā | 174 | 0.075809 | **0.9532** |
| Sragdharā | 166 | 0.077616 | **0.9760** |
| Vaṃśastha | 139 | 0.084819 | **1.0665** |
| Śikhariṇī | 131 | 0.087374 | **1.0986** |
| Upendravajrā | 99 | 0.100504 | **1.2640** |
| Mālinī | 75 | 0.115470 | **1.4521** |
| Śālinī | 55 | 0.134840 | **1.6953** |

(Weights average to 1.0 across all 10 meters by construction.)

## 5. What this means in practice

- Every example's contribution to the loss depends on **two independent factors**: how "hard" it is per-token (length-normalized CE) and how rare its meter is (weight table above).
- Because loss is length-normalized first, `w(m)` is the *only* thing that changes an example's relative pull on gradients due to meter — verse length itself is neutralized.
- Anuṣṭubh rows (96.5% of the dataset) are down-weighted to ~8.5% the influence of an "average" meter; Śālinī rows (0.24% of the dataset) get ~1.7x the influence.
- This directly counteracts class imbalance in meter representation without needing to under-sample the dominant Anuṣṭubh class or over-sample the rare ones.

## 6. Known interaction to be aware of

Because `L_i^CE` is already length-normalized before `w(m)` is applied, this loss is **blind to verse length** — a 3-token and 30-token verse of the same meter contribute equally per training step. If verse length carries useful difficulty signal (e.g. longer meters like Śārdūlavikrīḍita/Sragdharā are inherently harder to get metrically correct), that signal is suppressed here. This is exactly the behavior Experiment 2 (v2) removes — see `README_loss_v2.md`.


## Code for loss 

```python 
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    meter_ids = inputs.pop("meter_ids")
    labels = inputs["labels"]
    
    outputs = model.base_model.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
    logits = outputs.logits

    # Standard causal-LM shift
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())  # [B, T-1]

    valid_mask = (shift_labels != -100).float()  # [B, T-1]

    # L_i^CE = (1/T_i) * sum_t CE_{i,t}
    sequence_loss = (token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)

    # w_{m_i}
    weights = self.meter_weight_tensor.to(sequence_loss.device, sequence_loss.dtype)[meter_ids]

    # L_i = w_{m_i} * L_i^CE ; L_batch = mean_i L_i
    weighted_sequence_loss = sequence_loss * weights
    loss = weighted_sequence_loss.mean()

    return (loss, outputs) if return_outputs else loss

```