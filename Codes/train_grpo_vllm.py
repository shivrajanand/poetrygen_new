import os
import json
import random
import re
import sys
from pathlib import Path
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

HP = {
    "MODEL_NAME": "Trained_Models/phi4_sft_merged_bf16",

    "VLLM_HOST": "127.0.0.1",
    "VLLM_PORT": 8000,

    "MAX_PROMPT_LEN": 1310,
    "MAX_COMPLETION_LEN": 280,

    "NUM_GENERATIONS": 8,
    "TEMPERATURE": 0.7,
    "TOP_P": 1.0,
    "BETA": 0.0,
    "LOSS_TYPE": "dapo",
    "EPSILON_HIGH": 0.28,
    "NUM_ITERATIONS": 1,

    "BATCH_SIZE": 4,
    "GRAD_ACC": 8,
    "MAX_STEPS": 0,    
    "EPOCHS": 1,           
    "LR": 5e-5,
    "LOG_STEPS": 1,
    "SAVE_STEPS": 100,
    "SAVE_LIMIT": 3,
    "WEIGHT_DECAY": 0.0,
    "WARMUP_RATIO": 0.05,
    "MAX_GRAD_NORM": 0.1,

    "LORA_R": 32,
    "LORA_ALPHA": 64,
    "LORA_DROPOUT": 0.0,

    "REWARD_WEIGHTS": {"validity": 0.10, "half_acc": 0.30, "full_acc": 0.40, "semantic": 0.20},

    "SEMANTIC_DEVICE": "cpu", 

    "DATA_FILE_PATH": "Files/v3_gitapress_grpo_train_balanced.csv",
    "OUTPUT_DIR": "Trained_Models/Phi4-14B-grpo-vllm-balanced",
}

os.makedirs(HP["OUTPUT_DIR"], exist_ok=True)
os.environ["TENSORBOARD_LOGGING_DIR"] = HP["OUTPUT_DIR"] + "/runs"


tokenizer = AutoTokenizer.from_pretrained(HP["MODEL_NAME"])

model = AutoModelForCausalLM.from_pretrained(
    HP["MODEL_NAME"],
    dtype=torch.bfloat16,            
    attn_implementation="sdpa",  
)

peft_config = LoraConfig(
    r=HP["LORA_R"],
    lora_alpha=HP["LORA_ALPHA"],
    lora_dropout=HP["LORA_DROPOUT"],
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)


ds = load_dataset("csv", data_files=HP["DATA_FILE_PATH"])["train"]
train_ds = ds.filter(lambda x: x["split"] == "train")
print("Train distribution:", sorted(Counter(train_ds["meter_cd"]).items()))


def build_prompt(example):
    return {
        "prompt": [
            {"role": "system", "content": example["prompt"]},
            {"role": "user", "content": f"Meaning:\n{example['hi']}\n\nGenerate the Sanskrit verse.\n"},
        ],
    }


train_ds = train_ds.map(build_prompt)

# from datasets import concatenate_datasets
# anu   = train_ds.filter(lambda x: x["meter_cd"] == "Anuṣṭubh").shuffle(seed=42).select(range(1500))
# other = train_ds.filter(lambda x: x["meter_cd"] != "Anuṣṭubh")
# train_ds = concatenate_datasets([anu, other]).shuffle(seed=42)


# TODO (sanity check, once): the SFT run used Unsloth's "phi-4" chat template. Print the rendered prompt and
# compare it with what you trained on (<|im_start|>system<|im_sep|> ... <|im_end|> ... assistant<|im_sep|>):
# print(tokenizer.apply_chat_template(train_ds[0]["prompt"], tokenize=False, add_generation_prompt=True))

# --------------------------------------------------------------------------------------
# Rewards: R = 0.1*V + 0.3*H + 0.4*F + 0.2*S (same definitions as your eval script)
# Needs dataset columns: hi, meter_cd, syllable_count
# --------------------------------------------------------------------------------------
sys.path.append(str(Path("chandas-detector").resolve()))
from chandas_detector import detect_meter
from skrutable.meter_identification import MeterIdentifier
from sentence_transformers import SentenceTransformer

MI = MeterIdentifier()
SEMANTIC_MODEL = SentenceTransformer("sanganaka/bge-m3-sanskritFT", device=HP["SEMANTIC_DEVICE"])

INVALID_CHARS_RE = re.compile(r"[A-Za-z0-46-9]")   # '5' allowed (avagraha)
DEVANAGARI_ANY_RE = re.compile(r"[\u0900-\u097F]")


def _text(completion):
    return completion[0]["content"] if isinstance(completion, list) else completion


def _is_valid(t):
    return bool(t) and bool(DEVANAGARI_ANY_RE.search(t)) and not INVALID_CHARS_RE.search(t)


def _pred_syllable_count(verse):
    result = MI.identify_meter(verse, from_scheme="DEV", resplit_option="resplit_lite", resplit_keep_midpoint=True)
    weights = (result.syllable_weights or "").replace("\n", "").replace(" ", "")
    return len(weights) if weights else None


def _to_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


_STRUCT_CACHE = {}


def _structure_scores(text, meter_cd, gt_syll):
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
        if H == 1.0:
            try:
                r = detect_meter(t)
                pred_meter = r.meter if r.confidence == "exact" else None
            except Exception:
                pred_meter = None
            F = float(pred_meter == meter_cd)
    if len(_STRUCT_CACHE) > 50_000:
        _STRUCT_CACHE.clear()
    _STRUCT_CACHE[key] = (V, H, F)
    return V, H, F


def _batch_structure(completions, meter_cd, syllable_count):
    return [_structure_scores(_text(c), m, _to_int(s)) for c, m, s in zip(completions, meter_cd, syllable_count)]


def validity_reward(completions, meter_cd, syllable_count, **kwargs):
    return [v for v, _, _ in _batch_structure(completions, meter_cd, syllable_count)]


def half_acc_reward(completions, meter_cd, syllable_count, **kwargs):
    return [h for _, h, _ in _batch_structure(completions, meter_cd, syllable_count)]


def full_acc_reward(completions, meter_cd, syllable_count, **kwargs):
    return [f for _, _, f in _batch_structure(completions, meter_cd, syllable_count)]


def semantic_reward(completions, hi, **kwargs):
    texts = [_text(c).strip() for c in completions]
    scores = [0.0] * len(texts)
    idx = [i for i, t in enumerate(texts) if t]
    if not idx:
        return scores
    with torch.no_grad():
        in_embs = SEMANTIC_MODEL.encode([str(hi[i]) for i in idx], convert_to_tensor=True, batch_size=32)
        out_embs = SEMANTIC_MODEL.encode([texts[i] for i in idx], convert_to_tensor=True, batch_size=32)
        sims = SEMANTIC_MODEL.similarity_pairwise(in_embs, out_embs).cpu().tolist()
    for i, s in zip(idx, sims):
        scores[i] = float(min(1.0, max(0.0, s)))
    return scores


reward_funcs = [validity_reward, half_acc_reward, full_acc_reward, semantic_reward]

import time, functools
def timed(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        t = time.perf_counter(); out = fn(*a, **k)
        print(f"[TIMING] {fn.__name__}: {time.perf_counter()-t:.1f}s", flush=True); return out
    return w
reward_funcs = [timed(f) for f in reward_funcs]

_w = HP["REWARD_WEIGHTS"]
reward_weights = [_w["validity"], _w["half_acc"], _w["full_acc"], _w["semantic"]]

# --------------------------------------------------------------------------------------
# Trainer
# TODO: argument names below follow TRL ~0.24+. Check them against GRPOConfig of the TRL version pip gives you
# (vllm_mode, vllm_server_host, vllm_server_port, max_prompt_length, epsilon_high, steps_per_generation).
# --------------------------------------------------------------------------------------
training_args = GRPOConfig(
    output_dir=HP["OUTPUT_DIR"],

    # vLLM (server mode: generation on GPU 1, weights synced from this process every step)
    use_vllm=True,
    vllm_mode="server",
    vllm_server_host=HP["VLLM_HOST"],
    vllm_server_port=HP["VLLM_PORT"],

    num_generations=HP["NUM_GENERATIONS"],
    max_completion_length=HP["MAX_COMPLETION_LEN"],
    temperature=HP["TEMPERATURE"],
    top_p=HP["TOP_P"],
    beta=HP["BETA"],
    loss_type=HP["LOSS_TYPE"],
    epsilon_high=HP["EPSILON_HIGH"],
    num_iterations=HP["NUM_ITERATIONS"],
    reward_weights=reward_weights,
    mask_truncated_completions=True,

    per_device_train_batch_size=HP["BATCH_SIZE"],
    gradient_accumulation_steps=HP["GRAD_ACC"],
    # max_steps=HP["MAX_STEPS"],
    num_train_epochs=HP["EPOCHS"],
    learning_rate=HP["LR"],
    lr_scheduler_type="cosine",
    warmup_steps=50,
    weight_decay=HP["WEIGHT_DECAY"],
    max_grad_norm=HP["MAX_GRAD_NORM"],
    optim="adamw_torch",

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    fp16=False,
    seed=3407,

    logging_steps=HP["LOG_STEPS"],
    report_to="tensorboard",
    save_steps=HP["SAVE_STEPS"],
    save_total_limit=HP["SAVE_LIMIT"],
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=train_ds,
    peft_config=peft_config,
)

trainer.train()

trainer.save_model(HP["OUTPUT_DIR"] + "/final_model")   # saves the LoRA adapter
tokenizer.save_pretrained(HP["OUTPUT_DIR"] + "/final_model")

with open(HP["OUTPUT_DIR"] + "/essential_config.json", "w", encoding="utf-8") as f:
    json.dump({"HYPER-PARAMETERS": HP, "TRAIN_DATASET_LEN": len(train_ds)},
              f, indent=4, default=str) 
    
    
print("Training Finished")