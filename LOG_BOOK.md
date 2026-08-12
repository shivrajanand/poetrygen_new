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
- Major previous work done is chandomitra with benchmark scores:


- Stage 1: single meter IFT Experiment (mainly Anustubh), combined meters 0 shot, 3 shot and 5 shot IFT Experiments. 

- Stage2: introduction of custom loss function.

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


# Experiment Results
- Results in [Excel Sheet](https://docs.google.com/spreadsheets/d/1VW-hheN1PJbYdRWDJGcP0qAq-Ed19z3GWYy9Ltc0d4A/edit?gid=0#gid=0)
- Phi4 anustubh only, 0 shot and 1 shot experiment done. 
- Tried gemma4-12B but it requires latest transformers with Unified Architecture class which is not supported by our environment so shifting to gemma4-31B. 

- `Interesting Observation`: When performed 0shot training on Phi4-14B we noticed that overall accuracy was around 54% which was majorly contributed by Anustubh Samples only. Rest of the poetry forms were not being predicted by the trained models. But as we increase the number of examples inside the prompt during training itself, it increases the metric wise results as more types of meters get produced. So for this we move to `n-shot trainings` with n=1,3,5. 

- n-shot-training where n>1 poses a different problem. The input length rapidly increase with added examples which requires more compute for training. 

- `Interesting Observation`: We did a vasantatalika only finetuning of Phi4-14B. We have two usable checkpoints, the **best model** which has lowest eval loss and **last checkpoint** which is the last checkpoint obtained before early stop hit. Interstingly best model inference gives a 0% full-acc and a 6% half-acc whereas the last checkpoint inference gives a 3% full-acc and 9% half-acc. To check if this is an outlier observation or a trend we tested across some other finetuned checkpoints but it was not following the discussed trend. We can hypothetically blame the low size of Vasantatalika samples for this strange observation.