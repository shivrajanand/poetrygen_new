import os
import warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"

import unsloth
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import json
import random
import re
from collections import Counter

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

DEBUG = False

HYPERPARAMS = {
    "MODEL_NAME": "unsloth/phi-4",
    "SFT_ADAPTER_PATH": "Trained_Models/Phi4-14B-customLoss/checkpoint-3650", #Best scores till now

    "MAX_PROMPT_LEN": 1310,        # 99p=860; max=1305
    "MAX_COMPLETION_LEN": 280,     # 99p=224; max=276
    "LOAD_IN_4BIT": True,

    # ---- GRPO specific ----
    "NUM_GENERATIONS": 4,      
    "TEMPERATURE": 0.7,         
    "TOP_P": 0.9,           
    "BETA": 0.0,                  # KL coefficient. 0.0 = no reference model (saves memory).
    "LOSS_TYPE": "dapo",          # Supported in sanskrit server: grpo, bnpo, dr_grpo, dapo
    "NUM_ITERATIONS": 1,       
    "USE_VLLM": False,          

    # ---- optimisation ----
    "BATCH_SIZE": 4,              # Make sure BATCH_SIZE % NUM_GENERATIONS == 0  per device;
    "GRAD_ACC": 4,
    "EPOCHS": 3,             
    "LR": 5e-5,               
    "LOG_STEPS": 1,
    "SAVE_STEPS": 100,
    "SAVE_LIMIT": 3,
    "EVAL_STEPS": 100,
    "WEIGHT_DECAY": 0.0,
    "WARMUP_RATIO": 0.05,
    "MAX_GRAD_NORM": 0.1,      

    "LORA_R": 32,
    "LORA_ALPHA": 64,
    "LORA_DROPOUT": 0.0,          # dropout adds noise to policy log-probs; 0 is common for RL

    "REWARD_WEIGHTS": {
        "validity": 0.10,
        "half_acc": 0.30,
        "full_acc": 0.40,
        "semantic": 0.20,
    },

    "DATA_FILE_PATH": "Files/v3_gitapress_final.csv",
    "OUTPUT_DIR": "Trained_Models/Phi4-14B-grpo",
}

os.makedirs(HYPERPARAMS["OUTPUT_DIR"], exist_ok=True)

# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=HYPERPARAMS["MODEL_NAME"],
    max_seq_length=HYPERPARAMS["MAX_PROMPT_LEN"] + HYPERPARAMS["MAX_COMPLETION_LEN"],
    load_in_4bit=HYPERPARAMS["LOAD_IN_4BIT"],
    # fast_inference=HYPERPARAMS["USE_VLLM"],
    # max_lora_rank=HYPERPARAMS["LORA_R"],
)

if HYPERPARAMS["SFT_ADAPTER_PATH"]:
    model.load_adapter(
        HYPERPARAMS["SFT_ADAPTER_PATH"],
        adapter_name="default",
        is_trainable=True,
    )
else:
    model = FastLanguageModel.get_peft_model(
        model,
        r=HYPERPARAMS["LORA_R"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=HYPERPARAMS["LORA_ALPHA"],
        lora_dropout=HYPERPARAMS["LORA_DROPOUT"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="phi-4")

# --------------------------------------------------------------------------------------
# Data
# GRPO needs a "prompt" column only. Other columns (sa, meter_cd, ...) are forwarded to the
# reward functions as keyword arguments, so we keep them.
# --------------------------------------------------------------------------------------
ds = load_dataset("csv", data_files=HYPERPARAMS["DATA_FILE_PATH"])["train"]

train_ds = ds.filter(lambda x: x["split"] == "train")
val_ds = ds.filter(lambda x: x["split"] == "val")
print(f"Train: {len(train_ds)}")
print(f"Val: {len(val_ds)}")

print("Train distribution:", sorted(Counter(train_ds["meter_cd"]).items()))
print("Val distribution:", sorted(Counter(val_ds["meter_cd"]).items()))


def build_prompt(example):
    return {
        "prompt": [
            {"role": "system", "content": example["prompt"]},
            {"role": "user", "content": f"Meaning:\n{example['hi']}\n\nGenerate the Sanskrit verse.\n"},
        ],
        "reference": example["sa"],  
    }

train_ds = train_ds.map(build_prompt)
val_ds = val_ds.map(build_prompt)

# --------------------------------------------------------------------------------------
# Reward functions
# R = 0.1*V + 0.3*H + 0.4*F + 0.2*S
# Implemented as four separate reward functions combined by reward_weights, so TensorBoard
# logs each component (rewards/validity_reward/mean, ...) separately.
# Definitions mirror the evaluation script exactly.
# Requires dataset columns: hi, meter_cd, syllable_count (all kept through build_prompt's map).
# --------------------------------------------------------------------------------------
import sys
from pathlib import Path
import time

sys.path.append(str(Path("chandas-detector").resolve()))
from chandas_detector import detect_meter
from skrutable.meter_identification import MeterIdentifier
from sentence_transformers import SentenceTransformer

MI = MeterIdentifier()

# TODO: this sits on the GPU next to the 14B model. Set device="cpu" if you run out of memory
# (slower: it embeds G * batch completions every step).
SEMANTIC_MODEL = SentenceTransformer("sanganaka/bge-m3-sanskritFT")

INVALID_CHARS_RE = re.compile(r"[A-Za-z0-46-9]")   # same as eval; '5' allowed (avagraha)
DEVANAGARI_ANY_RE = re.compile(r"[\u0900-\u097F]")


def _text(completion):
    if isinstance(completion, list):
        return completion[0]["content"]
    return completion


def _is_valid(t):
    """V: non-empty, contains Devanagari, no Latin letters / digits (except 5)."""
    return bool(t) and bool(DEVANAGARI_ANY_RE.search(t)) and not INVALID_CHARS_RE.search(t)


def _pred_syllable_count(verse):
    result = MI.identify_meter(
        verse,
        from_scheme="DEV",
        resplit_option="resplit_lite",
        resplit_keep_midpoint=True,
    )
    weights = (result.syllable_weights or "").replace("\n", "").replace(" ", "")
    return len(weights) if weights else None


def _to_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


# The meter/syllable checkers are CPU-heavy, and V/H/F share the same computation,
# so cache per (text, meter, syllables).
_STRUCT_CACHE = {}


def _structure_scores(text, meter_cd, gt_syll):
    """Returns (V, H, F) as floats."""
    key = (text, meter_cd, gt_syll)
    if key in _STRUCT_CACHE:
        return _STRUCT_CACHE[key]

    t = text.strip()
    V = H = F = 0.0
    if _is_valid(t):
        V = 1.0
        try:
            pred_syll = _pred_syllable_count(t)
        except Exception:
            pred_syll = None
        H = float(pred_syll is not None and gt_syll is not None and pred_syll == gt_syll)

        if H == 1.0:  # F requires H, so skip detect_meter otherwise
            try:
                r = detect_meter(t)
                pred_meter = r.meter if r.confidence == "exact" else None
            except Exception:
                pred_meter = None
            F = float(pred_meter == meter_cd)

    if len(_STRUCT_CACHE) > 50_000:   # crude memory cap
        _STRUCT_CACHE.clear()
    _STRUCT_CACHE[key] = (V, H, F)
    return V, H, F


def _batch_structure(completions, meter_cd, syllable_count):
    if DEBUG: start = time.perf_counter()
    
    result = [
        _structure_scores(_text(c), m, _to_int(s))
        for c, m, s in zip(completions, meter_cd, syllable_count)
    ]
    
    if DEBUG: elapsed = time.perf_counter() - start
    
    if DEBUG: print(f"[TIMING] Structure rewards: {elapsed:.3f}s | {len(completions)} completions | {elapsed / max(len(completions), 1):.3f}s/completion")

    return result


def validity_reward(completions, meter_cd, syllable_count, **kwargs):
    return [v for v, _, _ in _batch_structure(completions, meter_cd, syllable_count)]


def half_acc_reward(completions, meter_cd, syllable_count, **kwargs):
    return [h for _, h, _ in _batch_structure(completions, meter_cd, syllable_count)]


def full_acc_reward(completions, meter_cd, syllable_count, **kwargs):
    return [f for _, _, f in _batch_structure(completions, meter_cd, syllable_count)]


def semantic_reward(completions, hi, **kwargs):
    """S: cosine similarity between Hindi meaning and generated verse, clipped to [0, 1].
    Computed for every non-empty completion (not gated by V/H/F), per the reward design."""
    
    if DEBUG: start = time.perf_counter()
    
    texts = [_text(c).strip() for c in completions]
    scores = [0.0] * len(texts)
    idx = [i for i, t in enumerate(texts) if t]
    
    if not idx:
        if DEBUG: print("[TIMING] Semantic reward: 0.000s | no non-empty completions")
        return scores

    if DEBUG: encode_start = time.perf_counter()
    
    with torch.no_grad():
        in_embs = SEMANTIC_MODEL.encode([str(hi[i]) for i in idx], convert_to_tensor=True, batch_size=32)
        out_embs = SEMANTIC_MODEL.encode([texts[i] for i in idx], convert_to_tensor=True, batch_size=32)
        sims = SEMANTIC_MODEL.similarity_pairwise(in_embs, out_embs).cpu().tolist()
        
    if DEBUG: encode_elapsed = time.perf_counter() - encode_start

    for i, s in zip(idx, sims):
        scores[i] = float(min(1.0, max(0.0, s)))
        
    if DEBUG: elapsed = time.perf_counter() - start
    
    if DEBUG: print(f"[TIMING] Semantic reward: | {elapsed:.3f}s | (encode/similarity: {encode_elapsed:.3f}s) | {len(idx)} completions")
    return scores


reward_funcs = [validity_reward, half_acc_reward, full_acc_reward, semantic_reward]
_w = HYPERPARAMS["REWARD_WEIGHTS"]
reward_weights = [_w["validity"], _w["half_acc"], _w["full_acc"], _w["semantic"]]



# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------
training_args = GRPOConfig(
    output_dir=HYPERPARAMS["OUTPUT_DIR"],

    num_generations=HYPERPARAMS["NUM_GENERATIONS"],
    max_prompt_length=HYPERPARAMS["MAX_PROMPT_LEN"],
    max_completion_length=HYPERPARAMS["MAX_COMPLETION_LEN"],
    temperature=HYPERPARAMS["TEMPERATURE"],
    top_p=HYPERPARAMS["TOP_P"],
    beta=HYPERPARAMS["BETA"],
    loss_type=HYPERPARAMS["LOSS_TYPE"],
    num_iterations=HYPERPARAMS["NUM_ITERATIONS"],
    reward_weights=reward_weights,
    use_vllm=HYPERPARAMS["USE_VLLM"],
    mask_truncated_completions = True,


    per_device_train_batch_size=HYPERPARAMS["BATCH_SIZE"],
    gradient_accumulation_steps=HYPERPARAMS["GRAD_ACC"],
    num_train_epochs=HYPERPARAMS["EPOCHS"],
    learning_rate=HYPERPARAMS["LR"],
    lr_scheduler_type="cosine",
    warmup_ratio=HYPERPARAMS["WARMUP_RATIO"],
    weight_decay=HYPERPARAMS["WEIGHT_DECAY"],
    max_grad_norm=HYPERPARAMS["MAX_GRAD_NORM"],
    optim="adamw_8bit",

    # logging / saving
    logging_steps=HYPERPARAMS["LOG_STEPS"],
    logging_dir=HYPERPARAMS["OUTPUT_DIR"] + "/runs",
    report_to="tensorboard",
    save_steps=HYPERPARAMS["SAVE_STEPS"],
    save_total_limit=HYPERPARAMS["SAVE_LIMIT"],
    # eval_strategy="steps", 
    # eval_steps=HYPERPARAMS["EVAL_STEPS"],

    fp16=False,
    bf16=True,
    seed=3407,
    # max_steps=30,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=train_ds,
    # eval_dataset=val_ds, 
)

trainer.train()

trainer.save_model(HYPERPARAMS["OUTPUT_DIR"] + "/final_model")
tokenizer.save_pretrained(HYPERPARAMS["OUTPUT_DIR"] + "/final_model")

essential_config = {
    "HYPER-PARAMETERS": HYPERPARAMS,
    "TRAIN_DATASET_LEN": len(train_ds),
    "VAL_DATASET_LEN": len(val_ds),
}

with open(HYPERPARAMS["OUTPUT_DIR"] + "/essential_config.json", "w", encoding="utf-8") as f:
    json.dump(essential_config, f, indent=4, default=str)