from pathlib import Path

import pandas as pd

from src.tune_optuna import OptunaTuningConfig, run_optuna_tuning


def test_run_optuna_tuning_writes_best_model_and_study_outputs(tmp_path: Path) -> None:
    config = OptunaTuningConfig(
        data_path=Path("data/raw/dataset_practica_final.csv"),
        output_dir=tmp_path / "outputs",
        models_dir=tmp_path / "models",
        storage=f"sqlite:///{tmp_path / 'study.db'}",
        study_name="smoke_optuna",
        feature_set="victor",
        n_trials=1,
        timeout=60,
        sample_size=500,
        cv_splits=2,
        random_state=42,
        create_plots=False,
        model_jobs=1,
    )

    result = run_optuna_tuning(config)

    assert result.best_model_path.exists()
    assert result.best_params_path.exists()
    assert result.trials_path.exists()
    assert result.metrics_path.exists()
    assert result.best_value >= 0

    metrics = pd.read_csv(result.metrics_path)
    assert metrics.loc[0, "model"] == "lightgbm_optuna"
    assert metrics.loc[0, "f1"] >= 0
