from pathlib import Path

from src.tune_optuna_native import NativeOptunaConfig, run_native_optuna


def test_run_native_optuna_writes_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    models_dir = tmp_path / "models"
    storage = f"sqlite:///{(output_dir / 'study.db').as_posix()}"

    config = NativeOptunaConfig(
        data_path=Path("data/raw/dataset_practica_final.csv"),
        output_dir=output_dir,
        models_dir=models_dir,
        storage=storage,
        study_name="native_smoke",
        feature_set="binary",
        n_trials=1,
        timeout=120,
        sample_size=800,
        cv_splits=2,
        model_jobs=1,
        study_jobs=1,
    )

    result = run_native_optuna(config)

    assert result.best_model_path.exists()
    assert result.best_params_path.exists()
    assert result.trials_path.exists()
    assert result.metrics_path.exists()
