# Loss Function — Experiment 2 (Meter-Weighted Only, No Length Normalization)

**Script:** `train_meter_weighted_v2.py`
**Output dir:** `Trained_Models/Phi4-14B-customLoss-v2`

## 1. What changed vs. v1

Everything is identical to v1 — response-only masking, meter-weight computation, batch mean — **except one step is removed**: the per-example loss is no longer divided by its number of response tokens.

| | v1 | v2 |
|---|---|---|
| Response-only loss masking | ✅ | ✅ |
| Divide by response token count (`/ T_i`) | ✅ | ❌ **removed** |
| Multiply by meter weight `w(m)` | ✅ | ✅ |
| Batch mean over examples | ✅ | ✅ |

## 2. Step-by-step

**Step A — token-level cross-entropy** *(unchanged from v1)*

```
CE_{i,t} = CrossEntropy(logits_{i,t-1}, label_{i,t})
```
computed only over response tokens (prompt tokens masked with `-100`).

**Step B — raw summed loss (no length normalization)**
Instead of averaging over `T_i` tokens, the per-example loss is just the **sum**:

```
L_i^CE = sum_t CE_{i,t}
```

This means an example's raw loss now scales with how many response tokens it has. A verse with twice as many tokens as another (all else equal) contributes roughly twice the raw loss.

**Step C — meter weight `w(m)`** *(identical formula and identical values to v1 — same table below)*

```
w_raw(m) = 1 / sqrt(N_m)
w(m)     = w_raw(m) / mean_j(w_raw(j))
```

**Step D — combine** *(same structure as v1, applied to the un-normalized loss)*

```
L_i     = w(meter_i) * L_i^CE
L_batch = mean_i(L_i)
```

## 3. Code (from `compute_loss`)

```python
valid_mask = (shift_labels != -100).float()

# Step B: raw sum, NOT divided by token count
sequence_loss = (token_loss * valid_mask).sum(dim=1)

# Step C: meter weight lookup (same table as v1)
weights = self.meter_weight_tensor.to(sequence_loss.device, sequence_loss.dtype)[meter_ids]

# Step D: combine + batch mean
weighted_sequence_loss = sequence_loss * weights
loss = weighted_sequence_loss.mean()
```

## 4. Meter weights (unchanged from v1 — computed from actual train-split counts)

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

## 5. What this means in practice

- An example's pull on the gradient now depends on **three interacting factors**: number of response tokens (unnormalized), per-token difficulty (summed CE), and meter rarity (`w(m)`).
- Longer verses get more total loss signal — the model is pushed harder to get every token of a long verse right, rather than treating a 5-token verse and a 40-token verse as equally important training signal.
- Meter weighting still corrects for class imbalance the same way it did in v1 (Anuṣṭubh down to ~8.5% influence, Śālinī up to ~1.7x), but this correction is now stacked on top of length, not applied to a length-neutral quantity.

## 6. Known interaction to be aware of

**Meter length and meter weight are no longer independent.** If some meters are systematically longer/shorter than others (Sanskrit meters have fixed or near-fixed syllable counts, so this is likely — e.g. Śārdūlavikrīḍita and Sragdharā are long meters; Anuṣṭubh is short), then removing length normalization means:

- A meter's *effective* training influence = `w(m) × (its typical token length)`, not `w(m)` alone.
- A long, already-weighted-up rare meter (e.g. Śārdūlavikrīḍita, `w = 0.86`) could end up dominating the gradient even more than the weight table alone suggests, simply because each of its examples is longer and now contributes more raw loss.
- Conversely, a rare but short meter would get less of a boost than the weight table implies.

**Recommendation:** before drawing conclusions from this experiment, it's worth checking the per-meter average response-token length (`sa` length in tokens) alongside the weight table above, so you can separate "the weighting worked as intended" from "length happened to help/hurt a particular meter."