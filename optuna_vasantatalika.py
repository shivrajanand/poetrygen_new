import unsloth
import gc
import json
import random

import optuna
import torch
from datasets import load_dataset
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*AttentionMaskConverter.*",
    category=FutureWarning,
)

warnings.filterwarnings(
    "ignore",
    message=".*use_return_dict.*",
    category=FutureWarning,
)

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ------------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------------
MODEL_NAME = "unsloth/phi-4"
MAX_LEN = 1408
LOAD_IN_4BIT = True

DATA_FILE_PATH = "Files/v3_gitapress_final_vasantatilaka.csv"   # <-- CHANGE IF NEEDED

OUTPUT_DIR = "hpo_runs_vasantatilaka"
STUDY_NAME = "phi4_vasantatilaka_hpo"
STUDY_DB = "sqlite:///vasantatilaka_hpo.db"

N_TRIALS = 15
NUM_EPOCHS = 15
EVAL_STEPS = 5

GRAD_CHECKPOINTING = True

FINAL_JSON = "vasantatilaka_best_params.json"
FINAL_CSV = "vasantatilaka_hpo_results.csv"
def build_formatter(tokenizer):
    def fn(batch):
        texts = []
        for prompt, hi, sa in zip(batch["prompt"], batch["hi"], batch["sa"]):
            convo = [
                {"role": "system", "content": prompt},
                {"role": "user",
                 "content": f"Meaning:\n{hi}\n\nGenerate the Sanskrit verse.\n"},
                {"role": "assistant", "content": sa},
            ]
            texts.append(
                tokenizer.apply_chat_template(
                    convo,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text": texts}
    return fn


def load_raw_data():
    ds = load_dataset("csv", data_files=DATA_FILE_PATH)["train"]

    train_ds = ds.filter(lambda x: x["split"] == "train").shuffle(seed=42)
    val_ds = ds.filter(lambda x: x["split"] == "val").shuffle(seed=42)

    print(f"Train: {len(train_ds)}")
    print(f"Validation: {len(val_ds)}")

    return train_ds, val_ds


class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control

        if "eval_loss" not in metrics:
            return control

        self.trial.report(metrics["eval_loss"], state.global_step)

        if self.trial.should_prune():
            raise optuna.TrialPruned()

        return control


def objective(trial, train_raw, val_raw):

    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32, 64])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64, 128])
    lora_dropout = trial.suggest_categorical(
        "lora_dropout",
        [0.0, 0.05, 0.1],
    )
    weight_decay = trial.suggest_float(
        "weight_decay",
        0.0,
        0.1,
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [4, 8, 12, 16],
    )

    label = (
        f"trial{trial.number}"
        f"_bs{batch_size}"
        f"_lr{lr:.2e}"
        f"_r{lora_r}"
    )
    
    effective_bs = batch_size * 4

    print("\n" + "=" * 80)
    print(f"Trial {trial.number}")
    print(f"Batch Size        : {batch_size}")
    print(f"Effective Batch   : {effective_bs}")
    print(f"Learning Rate     : {lr:.2e}")
    print(f"LoRA Rank         : {lora_r}")
    print(f"LoRA Alpha        : {lora_alpha}")
    print(f"LoRA Dropout      : {lora_dropout}")
    print(f"Weight Decay      : {weight_decay:.4f}")
    print("=" * 80)

    trainer = None
    model = None
    tokenizer = None

    try:

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_LEN,
            load_in_4bit=LOAD_IN_4BIT,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing=GRAD_CHECKPOINTING,
            random_state=3407,
        )

        tokenizer = get_chat_template(
            tokenizer,
            chat_template="phi-4",
        )

        formatter = build_formatter(tokenizer)

        train_ds = train_raw.map(formatter, batched=True)
        val_ds = val_raw.map(formatter, batched=True)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            dataset_text_field="text",
            max_seq_length=MAX_LEN,
            packing=True,
            dataset_num_proc=1,
            callbacks=[
                OptunaPruningCallback(trial),
            ],
            args=SFTConfig(
                output_dir=f"{OUTPUT_DIR}/trial_{trial.number}",

                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                num_train_epochs=NUM_EPOCHS,

                learning_rate=lr,
                lr_scheduler_type="cosine",
                warmup_ratio=0.05,
                weight_decay=weight_decay,
                max_grad_norm=1.0,

                logging_steps=5,
                logging_strategy="steps",
                logging_dir=f"{OUTPUT_DIR}/trial_{trial.number}/runs",
                report_to="none",

                eval_strategy="steps",
                eval_steps=EVAL_STEPS,

                save_strategy="no",

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

        print(f"{label} : {metrics['eval_loss']:.4f}")

        return metrics["eval_loss"]

  
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("OOM -> Trial pruned")
            raise optuna.TrialPruned()
        raise

    finally:
        del trainer
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def main():

    train_raw, val_raw = load_raw_data()

    sampler = TPESampler(seed=42)

    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=2,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=STUDY_NAME,
        storage=STUDY_DB,
        load_if_exists=True,
    )

    completed = len(study.trials)

    if completed:
        print(f"Resuming study with {completed} existing trials.")

    study.optimize(
        lambda t: objective(t, train_raw, val_raw),
        n_trials=max(0, N_TRIALS - completed),
        gc_after_trial=True,
    )

    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    study.trials_dataframe().to_csv(
        FINAL_CSV,
        index=False,
    )

    if len(completed_trials) == 0:
        print("\nNo completed trials.")
    else:
        print("\nBest validation loss:", study.best_value)
        print("\nBest parameters:")

        for k, v in study.best_trial.params.items():
            print(f"{k}: {v}")

        with open(FINAL_JSON, "w") as f:
            json.dump(
                {
                    "best_eval_loss": study.best_value,
                    "best_params": study.best_trial.params,
                    "n_trials_completed": len(completed_trials),
                    "n_trials_total": len(study.trials),
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()