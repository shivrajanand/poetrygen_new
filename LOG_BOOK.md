# Project Description

## Objective: given a hindi prose generate a sanskrit poetry in a given specific style, following the poetry meter rules while maintaining semantically similar with the hindi input. 

- Experiment focuses on 10 styles of sanskrit poetry mentioned as follows with their counts in data. 
    - Anuṣṭubh  27208
    - Vasantatilakā  661
    - Śārdūlavikrīḍita  265
    - Indravajrā  218
    - Sragdharā  207
    - Vaṃśastha  174
    - Śikhariṇī  164
    - Upendravajrā  124
    - Mālinī  93
    - Śālinī  69

- We have 1 prompt for each type of poetry meter. 
- Major previous work done is chandomitra with benchmark scores:


- Stage 1: single meter IFT Experiment (mainly Anustubh), combined meters 0 shot, 3 shot and 5 shot IFT Experiments. 

- Stage2: introduction of custom loss function.

## Evaluation Pipeline

This script evaluates generated Sanskrit verses (`model_out`) against
ground-truth meter and syllable-count labels, and reports how semantically
faithful the generations are to their source input (`hi`).

It is run as:

```bash
python evaluate.py <path_to_csv>
```

The input CSV must contain the following columns:

| Column            | Meaning                                              |
|-------------------|-------------------------------------------------------|
| `hi`              | Source text fed to the model                          |
| `meter_cd`        | Ground-truth meter                                     |
| `syllable_count`  | Ground-truth syllable count                            |
| `model_out`       | Model-generated Sanskrit verse                         |

The script adds `out_meter`, `pred_syllable_count`, `half_acc`, `full_acc`,
and `semsim` columns, and overwrites the CSV in place with these results.

### 1. Problem-row detection

Before scoring, each `model_out` value is checked for stray Latin
alphanumeric characters (e.g. leftover English tokens or malformed output).
The digit `5` is excluded from this check, since models sometimes use it to
represent the avagraha (ऽ). Rows that fail this check are:

- Marked `"problem"` in `out_meter`
- Saved separately to a `..._problem_cols.csv` file for inspection
- Skipped by meter/syllable detection, but **still included** in scoring
  (see below) rather than dropped from the dataset

### 2. Meter and syllable detection

For all non-problem rows, two independent tools are run on `model_out`:

- **`chandas_detector`** — detects the meter of the verse. A meter is only
  recorded if detection confidence is `"exact"`; otherwise the row is
  treated as `UNKNOWN`.
- **`skrutable.MeterIdentifier`** — scans the verse (`from_scheme="DEV"`,
  `resplit_option="resplit_lite"`, `resplit_keep_midpoint=True`) and counts
  syllables from the resulting light/heavy (`l`/`g`) weight string.

### 3. Row-wise metrics

Two binary metrics are computed per row:

- **`half_acc`** — 1 if the predicted syllable count matches the ground
  truth, else 0.
- **`full_acc`** — 1 if *both* the predicted meter and predicted syllable
  count match ground truth, else 0.

**Problem rows are included in this scoring and are always scored `0` on
both metrics** — a mixed-script or malformed output cannot be considered a
valid, correctly-metered verse, so it counts as a failure rather than being
excluded from the denominator. This keeps `half_acc`/`full_acc` honest as a
measure of "did the model produce a well-formed, correctly-metered verse,"
rather than only measuring accuracy on the subset of outputs that happened
to be clean.

A sanity-check assertion enforces the invariant `half_acc >= full_acc` for
every row, since full accuracy is a strictly stronger condition.

### 4. Semantic similarity

Using the `sanganaka/bge-m3-sanskritFT` sentence embedding model, each
`model_out` is embedded and compared against its corresponding `hi` input
via cosine similarity (`semsim`).

This step is computed for **all rows with non-null input/output text,
including problem rows** — metrical correctness and semantic fidelity are
treated as independent questions. A garbled or mixed-script output can
still be topically on-target, and this is useful diagnostic signal
(distinguishing "broken formatting but on-topic" from "broken formatting
and off-topic" failures) that would otherwise be lost.

### 5. Reporting

The script prints two Markdown tables:

- **Overall Evaluation** — mean `half_acc`, `full_acc`, and `semsim` across
  all rows, plus supporting counts (total samples, problem rows, unknown
  meters).
- **Meter-wise Evaluation** — the same metrics broken down per ground-truth
  meter, so per-meter model performance can be compared.

Finally, the enriched dataframe (including all per-row predictions and
scores) is written back to the original CSV path.



### Chandomitra Benchamark: 
---

Best metrical result: NLLB-dist-1.3B + constrained decoding + fine-tuning
--
- Full: 99.86%
- Partial: 99.86%
- Semantic similarity: 64.91

Best instruction-finetuned model for the paper's trade-off: Phi-4-14B
--
- Full: 57.42%
- Partial: 75.01%
- Semantic: 67.29


# Experiment Results (current project)
- Results in [Excel Sheet](https://docs.google.com/spreadsheets/d/1VW-hheN1PJbYdRWDJGcP0qAq-Ed19z3GWYy9Ltc0d4A/edit?gid=0#gid=0)
- Phi4 anustubh only, 0 shot and 1 shot experiment done. 
- Tried gemma4-12B but it requires latest transformers with Unified Architecture class which is not supported by our environment so shifting to gemma4-31B. 

- `Interesting Observation`: When performed 0shot training on Phi4-14B we noticed that overall accuracy was around 54% which was majorly contributed by Anustubh Samples only. Rest of the poetry forms were not being predicted by the trained models. But as we increase the number of examples inside the prompt during training itself, it increases the metric wise results as more types of meters get produced. So for this we move to `n-shot trainings` with n=1,3,5. 

- `n-shot trainings` means during the training feed the model data like (**prompt** + **n-shot-examples** + **input** -> **output**)
- n-shot-training where n>1 poses a different problem. The input length rapidly increase with added examples which requires more compute for training. 

- `Interesting Observation`: We did a vasantatalika only finetuning of Phi4-14B. We have two usable checkpoints, the **best model** which has lowest eval loss and **last checkpoint** which is the last checkpoint obtained before early stop hit. Interstingly best model inference gives a 0% full-acc and a 6% half-acc whereas the last checkpoint inference gives a 3% full-acc and 9% half-acc. To check if this is an outlier observation or a trend we tested across some other finetuned checkpoints but it was not following the discussed trend. We can hypothetically blame the low size of Vasantatalika samples for this strange observation.