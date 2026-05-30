from pathlib import Path

import pandas as pd

from src.train import TrainingConfig, run_training


def test_run_training_writes_metrics_and_best_model(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    models_dir = tmp_path / "models"

    config = TrainingConfig(
        data_path=Path("data/raw/dataset_practica_final.csv"),
        output_dir=output_dir,
        models_dir=models_dir,
        processed_data_path=tmp_path / "processed_alejandro.csv",
        feature_set="alejandro",
        models=("logistic_regression", "decision_tree"),
        sample_size=800,
        create_plots=False,
        tune=False,
        neural_epochs=1,
        random_state=42,
    )

    result = run_training(config)

    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "best_model_summary.json"

    assert metrics_path.exists()
    assert summary_path.exists()
    assert result.best_model_name in {"logistic_regression", "decision_tree"}
    assert result.best_model_path.exists()
    assert config.processed_data_path.exists()

    metrics = pd.read_csv(metrics_path)
    processed = pd.read_csv(config.processed_data_path)
    assert set(metrics["model"]) == {"logistic_regression", "decision_tree"}
    assert metrics["f1"].between(0, 1).all()
    assert metrics["roc_auc"].between(0, 1).all()
    assert "assigned_room_type" not in processed.columns
    assert "total_nights" in processed.columns
