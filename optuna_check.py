import optuna

study = optuna.load_study(
    study_name="gemma4_lora_search",
    storage="sqlite:///Trained_Models/Gemma4-31B-optuna/optuna_study.db",
)

for t in study.trials:
    print(
        f"Trial {t.number:2d} | {t.state.name:8s} | value={t.value} | params={t.params}"
    )