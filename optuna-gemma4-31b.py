import os
import gc
import json
import random

import torch
import optuna
from optuna.trial import TrialState

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback

import warnings

warnings.filterwarnings("ignore")

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# Fixed constants (not searched).
#
# BATCH_SIZE / GRAD_ACC are kept identical to what final training will use
# (8 / 12, effective batch 96) -- this is the config that already utilizes
# the L40 reasonably, and per_device_bs affects how the other hyperparameters
# behave, so search and final training must share it to make the search
# results valid.
# ---------------------------------------------------------------------------
CONSTANTS = {
    "MODEL_NAME": "unsloth/gemma-4-31B-it",
    "MAX_LEN": 768,
    "LOAD_IN_4BIT": True,
    "TARGET_MODULES": "all-linear",
    "OPTIM": "adamw_8bit",

    "BATCH_SIZE": 8,
    "GRAD_ACC": 12,             # Effective batch = 96
    "WEIGHT_DECAY": 0.01,
    "WARMUP_RATIO": 0.03,
    "MAX_GRAD_NORM": 1.0,

    "LOG_STEPS": 10,

    "DATA_FILE_PATH": "Files/v3_gitapress_final.csv",
    "STUDY_OUTPUT_DIR": "Trained_Models/Gemma4-31B-optuna",
    "N_TRIALS": 15,             # 4-param search space -- 15 trials is plenty for TPE
    "STUDY_NAME": "gemma4_lora_search",
    "STORAGE": "sqlite:///Trained_Models/Gemma4-31B-optuna/optuna_study.db",

    # Small subset, purely to RANK configs against each other.
    # Full set is 23346 train / 2918 val.
    "HPO_TRAIN_SUBSET": 800,
    "HPO_VAL_SUBSET": 150,

    # Proxy-run budget per trial: a handful of optimizer steps, not full
    # epochs. At ~134s/step on an L40 for this 31B model, 60 steps is
    # ~2.2h worst case per trial, and the pruner kills bad configs well
    # before that.
    "SEARCH_MAX_STEPS": 60,
    "SEARCH_EVAL_STEPS": 15,    # eval every N steps -> gives the pruner a signal
}

os.makedirs(CONSTANTS["STUDY_OUTPUT_DIR"], exist_ok=True)

INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"


# ---------------------------------------------------------------------------
# Build the text datasets ONCE (independent of LoRA hyperparameters).
# We load a throwaway model just to get the chat-templated tokenizer, then
# free it before entering the Optuna loop.
# ---------------------------------------------------------------------------
def build_datasets():
    tmp_model, tmp_tokenizer = FastModel.from_pretrained(
        model_name=CONSTANTS["MODEL_NAME"],
        max_seq_length=CONSTANTS["MAX_LEN"],
        load_in_4bit=CONSTANTS["LOAD_IN_4BIT"],
        full_finetuning=False,
    )
    tmp_tokenizer = get_chat_template(tmp_tokenizer, "gemma-4")

    ds = load_dataset("csv", data_files=CONSTANTS["DATA_FILE_PATH"])["train"]
    train_ds = ds.filter(lambda x: x["split"] == "train")
    val_ds = ds.filter(lambda x: x["split"] == "val")

    print(f"Full train: {len(train_ds)}")
    print(f"Full val:   {len(val_ds)}")

    # ---- subsample BEFORE mapping, so we never tokenize/format rows we
    # won't actually use for the search. Shuffled once with a fixed seed
    # so every trial in the study sees the exact same subset.
    train_subset_n = min(CONSTANTS["HPO_TRAIN_SUBSET"], len(train_ds))
    val_subset_n = min(CONSTANTS["HPO_VAL_SUBSET"], len(val_ds))

    train_ds = train_ds.shuffle(seed=SEED).select(range(train_subset_n))
    val_ds = val_ds.shuffle(seed=SEED).select(range(val_subset_n))

    print(f"HPO train subset: {len(train_ds)}")
    print(f"HPO val subset:   {len(val_ds)}")

    def convert_to_text(sample):
        messages = [
            {"role": "system", "content": sample["prompt"]},
            {
                "role": "user",
                "content": f"Meaning:\n{sample['hi']}\n\nGenerate the Sanskrit verse.\n",
            },
            {"role": "assistant", "content": sample["sa"]},
        ]
        text = tmp_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ).removeprefix("<bos>")
        return {"text": text}

    train_dataset = train_ds.map(
        convert_to_text, remove_columns=train_ds.column_names, num_proc=os.cpu_count()
    )
    val_dataset = val_ds.map(
        convert_to_text, remove_columns=val_ds.column_names, num_proc=os.cpu_count()
    )

    print("--------------------------------------------------------------------------------------")
    print("0th data point in train set")
    print(train_dataset[0])
    print("--------------------------------------------------------------------------------------")

    # free the throwaway model before real trials start
    del tmp_model, tmp_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Optuna <-> Trainer bridge: reports eval_loss after every evaluation and
# lets Optuna decide whether to prune the trial.
# ---------------------------------------------------------------------------
class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial):
        self.trial = trial
        self.step_count = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or "eval_loss" not in metrics:
            return control
        self.step_count += 1
        self.trial.report(metrics["eval_loss"], step=self.step_count)
        if self.trial.should_prune():
            control.should_training_stop = True
            raise optuna.TrialPruned()
        return control


def free_memory(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def make_objective(train_dataset, val_dataset):
    def objective(trial: optuna.Trial) -> float:
        # ---- search space -------------------------------------------------
        # Only the LoRA-side knobs are searched. Everything training-dynamics
        # related (batch size, grad accum, weight decay, warmup, grad norm)
        # is fixed in CONSTANTS and matches what final training will use.
        lr = trial.suggest_float("LR", 1e-4, 3e-4, log=True)
        lora_r = trial.suggest_categorical("LORA_R", [16, 32, 64])
        lora_dropout = trial.suggest_float("LORA_DROPOUT", 0.0, 0.1)
        use_rslora = trial.suggest_categorical("use_rslora", [False, True])
        lora_alpha = 2 * lora_r  # derived, not searched independently

        label = (
            f"trial{trial.number}_lr{lr:.2e}_r{lora_r}_a{lora_alpha}"
            f"_do{lora_dropout:.3f}_rslora{use_rslora}"
        )
        trial_dir = os.path.join(CONSTANTS["STUDY_OUTPUT_DIR"], label)
        os.makedirs(trial_dir, exist_ok=True)

        print("=" * 90)
        print(label)
        print("=" * 90)

        model = None
        tokenizer = None
        trainer = None
        eval_loss = float("inf")

        try:
            # ---- fresh base model + LoRA for this trial -------------------
            model, tokenizer = FastModel.from_pretrained(
                model_name=CONSTANTS["MODEL_NAME"],
                max_seq_length=CONSTANTS["MAX_LEN"],
                load_in_4bit=CONSTANTS["LOAD_IN_4BIT"],
                full_finetuning=False,
            )

            model = FastModel.get_peft_model(
                model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=use_rslora,
                loftq_config=None,

                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                target_modules=CONSTANTS["TARGET_MODULES"],
            )

            tokenizer = get_chat_template(tokenizer, "gemma-4")
            FastModel.for_training(model)

            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks=[OptunaPruningCallback(trial)],
                args=SFTConfig(
                    output_dir=trial_dir,

                    per_device_train_batch_size=CONSTANTS["BATCH_SIZE"],
                    gradient_accumulation_steps=CONSTANTS["GRAD_ACC"],
                    max_steps=CONSTANTS["SEARCH_MAX_STEPS"],  # proxy run, not full training

                    learning_rate=lr,
                    weight_decay=CONSTANTS["WEIGHT_DECAY"],
                    warmup_ratio=CONSTANTS["WARMUP_RATIO"],
                    max_grad_norm=CONSTANTS["MAX_GRAD_NORM"],

                    logging_steps=CONSTANTS["LOG_STEPS"],
                    logging_strategy="steps",
                    report_to="none",

                    save_strategy="no",   # throwaway proxy runs -- nothing worth checkpointing

                    eval_strategy="steps",
                    eval_steps=CONSTANTS["SEARCH_EVAL_STEPS"],

                    fp16=False,
                    bf16=True,

                    optim=CONSTANTS["OPTIM"],
                    seed=3407,
                    remove_unused_columns=True,
                    dataset_text_field="text",
                    max_length=CONSTANTS["MAX_LEN"],
                ),
            )

            trainer = train_on_responses_only(
                trainer,
                instruction_part=INSTRUCTION_PART,
                response_part=RESPONSE_PART,
                num_proc=1,
            )

            trainer.train()
            metrics = trainer.evaluate()
            eval_loss = metrics["eval_loss"]
            print(f"{label} -> eval_loss={eval_loss:.4f}")

            # persist per-trial config for traceability
            with open(os.path.join(trial_dir, "trial_config.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "trial_number": trial.number,
                        "params": {
                            "LR": lr,
                            "LORA_R": lora_r,
                            "LORA_ALPHA": lora_alpha,
                            "LORA_DROPOUT": lora_dropout,
                            "use_rslora": use_rslora,
                        },
                        "eval_loss": eval_loss,
                    },
                    f,
                    indent=4,
                    default=str,
                )

        except torch.cuda.OutOfMemoryError:
            print(f"{label} -> OOM, pruning trial")
            raise optuna.TrialPruned()

        except optuna.TrialPruned:
            raise

        finally:
            free_memory(trainer, model, tokenizer)

        return eval_loss

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_dataset, val_dataset = build_datasets()

    study = optuna.create_study(
        study_name=CONSTANTS["STUDY_NAME"],
        storage=CONSTANTS["STORAGE"],
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        # n_warmup_steps=1 -> a trial can be pruned right after its FIRST
        # eval (step 15/60) instead of waiting for its second. This is what
        # actually saves wall-clock time: bad configs die in ~30-35 min
        # instead of running to completion.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
        load_if_exists=True,
    )

    remaining_trials = max(0, CONSTANTS["N_TRIALS"] - len(study.trials))

    print(f"Existing trials: {len(study.trials)}")
    print(f"Running {remaining_trials} more trials")

    study.optimize(
        make_objective(train_dataset, val_dataset),
        n_trials=remaining_trials,
        gc_after_trial=True,
    )

    # ---- report -------------------------------------------------------
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]

    print(f"Finished trials: {len(study.trials)}")
    print(f"Pruned trials:   {len(pruned)}")
    print(f"Complete trials: {len(complete)}")

    print("Best trial:")
    best = study.best_trial
    print(f"  Value (eval_loss): {best.value}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    summary = {
        "best_value": best.value,
        "best_params": best.params,
        "best_derived_lora_alpha": 2 * best.params["LORA_R"],
        "n_trials": len(study.trials),
        "n_pruned": len(pruned),
        "n_complete": len(complete),
        "note": (
            "eval_loss values come from short SEARCH_MAX_STEPS proxy runs on "
            "a small subset -- use this to RANK LoRA configs, then run "
            "train_final.py (reads best_params from this file) to do the "
            "real, full-dataset / full-epoch training with those winning "
            "LoRA params."
        ),
        "constants": CONSTANTS,
    }

    summary_path = os.path.join(CONSTANTS["STUDY_OUTPUT_DIR"], "study_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, default=str)
    print(f"Wrote {summary_path} -- run train_final.py next.")

    # optional: full trials dataframe for later analysis
    try:
        df = study.trials_dataframe()
        df.to_csv(os.path.join(CONSTANTS["STUDY_OUTPUT_DIR"], "study_trials.csv"), index=False)
    except Exception as e:
        print(f"Could not export trials dataframe: {e}")