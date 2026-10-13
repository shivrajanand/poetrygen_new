import unsloth
from unsloth import FastLanguageModel
import os
import gc
import json
import random

import torch
import optuna
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import get_chat_template, train_on_responses_only

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# ----------------------------------------------------------------------------
# Fixed settings (not searched)
# ----------------------------------------------------------------------------
FIXED = {
    "MODEL_NAME": "unsloth/phi-4",
    "MAX_LEN": 1536,
    "LOAD_IN_4BIT": True,
    "BATCH_SIZE": 16,
    "GRAD_ACC": 2,  # effective batch size = 16 x 2 = 32
    "HPO_MAX_STEPS": 100,
    "FINAL_EPOCHS": 5,
    "EVAL_STEPS": 25,
    "SEED": 3407,
    "DATA_FILE_PATH": "Files/v3_gitapress_final_not_anustubh.csv",
    "OUTPUT_DIR": "Trained_Models/Phi4-14B-not-anustubh-optuna",
}

# Study settings
N_TRIALS = 10
STUDY_NAME = "phi4_sanskrit_lora_epochs5"  # new name: search space changed
STORAGE = f"sqlite:///{FIXED['OUTPUT_DIR']}/optuna_study_epochs5.db"  # resumable

os.makedirs(FIXED["OUTPUT_DIR"], exist_ok=True)

# ----------------------------------------------------------------------------
# Data (loaded once, formatted per trial because the tokenizer is reloaded)
# ----------------------------------------------------------------------------
raw_ds = load_dataset("csv", data_files=FIXED["DATA_FILE_PATH"])["train"]
raw_train = raw_ds.filter(lambda x: x["split"] == "train")
raw_val = raw_ds.filter(lambda x: x["split"] == "val")
print(f"Train: {len(raw_train)} | Val: {len(raw_val)}")


def make_formatter(tokenizer):
    def format_and_tokenize(batch):
        texts = []
        for prompt, hi, sa in zip(batch["prompt"], batch["hi"], batch["sa"]):
            convo = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Meaning:\n{hi}\n\nGenerate the Sanskrit verse.\n"},
                {"role": "assistant", "content": sa},
            ]
            texts.append(
                tokenizer.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False
                )
            )
        return {"text": texts}

    return format_and_tokenize


# ----------------------------------------------------------------------------
# Optuna pruning callback: reports eval_loss at every evaluation
# ----------------------------------------------------------------------------
class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return
        loss = metrics["eval_loss"]
        if loss != loss or loss == float("inf"):  # NaN / inf
            raise optuna.TrialPruned("eval_loss is NaN/inf")
        self.trial.report(loss, step=state.global_step)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ----------------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    # ---- Search space (epochs fixed at FIXED['EPOCHS']) ----
    lr = trial.suggest_float("lr", 3e-5, 3e-4, log=True)
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32, 64])
    alpha_ratio = trial.suggest_categorical("alpha_ratio", [1, 2, 4])
    lora_alpha = lora_r * alpha_ratio
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.2)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.02)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2.0)

    trial.set_user_attr("lora_alpha", lora_alpha)
    print(f"\n=== Trial {trial.number} | r={lora_r} alpha={lora_alpha} lr={lr:.2e} "
          f"dropout={lora_dropout:.3f} warmup={warmup_ratio:.3f} "
          f"wd={weight_decay:.4f} gn={max_grad_norm:.2f} ===")

    # Same seeds every trial so differences come from hyperparameters
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = tokenizer = trainer = None
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=FIXED["MODEL_NAME"],
            max_seq_length=FIXED["MAX_LEN"],
            load_in_4bit=FIXED["LOAD_IN_4BIT"],
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=FIXED["SEED"],
            use_rslora=False,
            loftq_config=None,
        )

        tokenizer = get_chat_template(tokenizer, chat_template="phi-4")
        fmt = make_formatter(tokenizer)
        train_ds = raw_train.map(fmt, batched=True)
        val_ds = raw_val.map(fmt, batched=True)

        trial_dir = os.path.join(FIXED["OUTPUT_DIR"], f"trial_{trial.number}")

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            dataset_text_field="text",
            max_seq_length=FIXED["MAX_LEN"],
            packing=True,
            dataset_num_proc=1,
            # No EarlyStoppingCallback here: Optuna's pruner is the only thing
            # that stops a trial early. Add early stopping back for final training.
            callbacks=[OptunaPruningCallback(trial)],
            args=SFTConfig(
                output_dir=trial_dir,
                per_device_train_batch_size=FIXED["BATCH_SIZE"],
                gradient_accumulation_steps=FIXED["GRAD_ACC"],
                max_steps=FIXED["HPO_MAX_STEPS"],
                learning_rate=lr,
                lr_scheduler_type="cosine",
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                logging_steps=10,
                logging_strategy="steps",
                report_to="none",
                save_strategy="no",  # no checkpoints during the search (saves disk)
                eval_strategy="steps",
                eval_steps=FIXED["EVAL_STEPS"],
                fp16=False,
                bf16=True,
                load_best_model_at_end=False,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                optim="adamw_8bit",
                seed=FIXED["SEED"],
            ),
        )

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user<|im_sep|>",
            response_part="<|im_start|>assistant<|im_sep|>",
        )

        trainer.train()

        eval_losses = [
            log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log
        ]
        if not eval_losses:
            raise optuna.TrialPruned("no eval_loss recorded")

        best_loss = min(eval_losses)
        trial.set_user_attr("final_eval_loss", eval_losses[-1])
        trial.set_user_attr("n_evals", len(eval_losses))
        return best_loss

    except torch.cuda.OutOfMemoryError:
        print(f"Trial {trial.number}: CUDA OOM -> pruned")
        raise optuna.TrialPruned("CUDA OOM")

    finally:
        del trainer, model, tokenizer
        cleanup()


# ----------------------------------------------------------------------------
# Run study
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2,
            n_warmup_steps=25,
            interval_steps=1,
        ),
    )

    # Seed the search with your current config as the first trial (baseline)
    if len(study.trials) == 0:
        study.enqueue_trial({
            "lr": 2.4e-4,
            "lora_r": 32,
            "alpha_ratio": 2,
            "lora_dropout": 0.1,
            "warmup_ratio": 0.05,
            "weight_decay": 0.00465,
            "max_grad_norm": 1.0,
        })

    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

    # ---- Report ----
    best = study.best_trial
    best_params = dict(best.params)
    best_params["lora_alpha"] = best.user_attrs["lora_alpha"]

    print("\n================ BEST TRIAL ================")
    print(f"Trial #{best.number} | best eval_loss = {best.value:.5f}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    with open(os.path.join(FIXED["OUTPUT_DIR"], "best_params.json"), "w") as f:
        json.dump({"best_value": best.value, "params": best_params}, f, indent=4)

    study.trials_dataframe().to_csv(
        os.path.join(FIXED["OUTPUT_DIR"], "all_trials.csv"), index=False
    )