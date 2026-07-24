import unsloth
from unsloth import FastLanguageModel
import torch
import gc
import json
import random

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from datasets import load_dataset
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ---------------------------------------------------------------------------
# Fixed settings (systems/throughput choices -- not being searched)
# ---------------------------------------------------------------------------
MODEL_NAME = "unsloth/phi-4"
MAX_LEN = 2700                                              # CHANGED for 3-shot
LOAD_IN_4BIT = True
DATA_FILE_PATH = "Files/v3_gitapress_final_3shot_prompts.csv"  # CHANGED for 3-shot

# Same as the proven-working 5-shot run -- untouched.
SEARCH_BATCH_SIZE = 8
SEARCH_GRAD_ACC = 4          # effective batch = 32 for the search phase
GRAD_CHECKPOINTING = True

SEARCH_TRAIN_SUBSET = 8000     # rows of train data used per trial
SEARCH_VAL_SUBSET = 800        # rows of val data used per trial for eval
SEARCH_MAX_STEPS = 120          # optimizer steps per trial (proxy run, not full training)
SEARCH_EVAL_STEPS = 20         # eval every N steps -> enables pruning signal
N_TRIALS = 20
# STUDY_TIMEOUT_HOURS = 9        # hard safety cap so the study stops before morning

STUDY_DB = "sqlite:///hpo_study_3shot.db"   # CHANGED for 3-shot -- separate from the 5-shot study's db
STUDY_NAME = "phi4_sanskrit_hpo_3shot"      # CHANGED for 3-shot -- separate study name

OUTPUT_DIR = "hpo_runs_3shot"               # CHANGED for 3-shot


def build_formatter(tokenizer):
    def fn(batch):
        texts = []
        for prompt, hi, sa in zip(batch["prompt"], batch["hi"], batch["sa"]):
            convo = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Meaning:\n{hi}\n\nGenerate the Sanskrit verse.\n"},
                {"role": "assistant", "content": sa},
            ]
            texts.append(tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False))
        return {"text": texts}
    return fn


def load_raw_data():
    ds = load_dataset('csv', data_files=DATA_FILE_PATH)["train"]
    train_ds = ds.filter(lambda x: x["split"] == "train").shuffle(seed=42)
    val_ds = ds.filter(lambda x: x["split"] == "val").shuffle(seed=42)
    train_ds = train_ds.select(range(min(len(train_ds), SEARCH_TRAIN_SUBSET)))
    val_ds = val_ds.select(range(min(len(val_ds), SEARCH_VAL_SUBSET)))
    print(f"Search-phase data: train={len(train_ds)}, val={len(val_ds)}")
    return train_ds, val_ds


class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial):
        self.trial = trial
        self.eval_count = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or "eval_loss" not in metrics:
            return control

        self.trial.report(metrics["eval_loss"], state.global_step)

        if self.trial.should_prune():
            raise optuna.TrialPruned()

        return control


def objective(trial, train_ds_raw, val_ds_raw):
    lr = trial.suggest_float("lr", 1e-4, 8e-4, log=True)
    lora_r = trial.suggest_categorical("lora_r", [16,32,64])
    lora_alpha = trial.suggest_categorical("lora_alpha",[16, 32, 64, 128])
    lora_dropout = trial.suggest_categorical("lora_dropout", [0.0, 0.02, 0.05, 0.1])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.03)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.02, 0.15)
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3,0.5,1.0])
    scheduler = trial.suggest_categorical("scheduler",["cosine","linear","cosine_with_restarts",])
    packing = trial.suggest_categorical("packing",[True, False])
    label = f"trial{trial.number}_lr{lr:.2e}_r{lora_r}_a{lora_alpha}_do{lora_dropout}"
    print(f"\n{'='*80}\n{label}\n{'='*80}")

    model = None
    tokenizer = None
    trainer = None
    eval_loss = float("inf")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_LEN,
            load_in_4bit=LOAD_IN_4BIT,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing=GRAD_CHECKPOINTING,
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        tokenizer = get_chat_template(tokenizer, chat_template="phi-4")
        fmt = build_formatter(tokenizer)
        train_ds = train_ds_raw.map(fmt, batched=True)
        val_ds = val_ds_raw.map(fmt, batched=True)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            dataset_text_field="text",
            max_seq_length=MAX_LEN,
            packing=packing,
            dataset_num_proc=1,
            callbacks=[OptunaPruningCallback(trial)],
            args=SFTConfig(
                output_dir=f"/tmp/{OUTPUT_DIR}/{label}",
                per_device_train_batch_size=SEARCH_BATCH_SIZE,
                gradient_accumulation_steps=SEARCH_GRAD_ACC,
                max_steps=SEARCH_MAX_STEPS,
                learning_rate=lr,
                lr_scheduler_type=scheduler,
                max_grad_norm=max_grad_norm,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                logging_steps=10,
                eval_strategy="steps",
                eval_steps=SEARCH_EVAL_STEPS,
                save_strategy="no",
                report_to="none",
                fp16=False,
                bf16=True,
                optim="adamw_8bit",
                seed=3407,
            ),
        )

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user<|im_sep|>",
            response_part="<|im_start|>assistant<|im_sep|>",
        )

        trainer.train()
        metrics = trainer.evaluate()
        eval_loss = metrics["eval_loss"]
        print(f"{label} -> eval_loss={eval_loss:.4f}")

    except torch.cuda.OutOfMemoryError:
        print(f"{label} -> OOM, pruning trial")
        raise optuna.TrialPruned()

    finally:
        del trainer, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    return eval_loss


if __name__ == "__main__":
    train_ds_raw, val_ds_raw = load_raw_data()

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=STUDY_NAME,
        storage=STUDY_DB,
        load_if_exists=True,
    )

    n_already_done = len(study.trials)
    if n_already_done > 0:
        print(f"Resuming study: {n_already_done} trial(s) already recorded in {STUDY_DB}")

    study.optimize(
        lambda t: objective(t, train_ds_raw, val_ds_raw),
        n_trials=max(0, N_TRIALS - n_already_done),
        # timeout=STUDY_TIMEOUT_HOURS * 3600,
        gc_after_trial=True,
    )

    print("\n\n" + "=" * 80)
    print("STUDY SUMMARY")
    print("=" * 80)
    all_trials_df = study.trials_dataframe()
    print(all_trials_df['state'].value_counts() if 'state' in all_trials_df.columns else "No trials recorded")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("\nNo COMPLETE trials in this study.")
    else:
        print(f"\nBEST TRIAL (of {len(completed)} completed)")
        print(f"eval_loss: {study.best_value:.4f}")
        print("params:")
        for k, v in study.best_trial.params.items():
            print(f"  {k}: {v}")

    study.trials_dataframe().to_csv("hpo_results-3shot.csv", index=False)
    with open("hpo_best_params-3shot.json", "w") as f:
        json.dump({
            "best_eval_loss": study.best_value if completed else None,
            "best_params": study.best_trial.params if completed else None,
            "n_trials_completed": len(completed),
            "n_trials_total": len(study.trials),
        }, f, indent=2)

    print("\nFull results: hpo_results-3shot.csv, hpo_best_params-3shot.json")