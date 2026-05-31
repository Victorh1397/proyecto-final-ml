from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

from src.data import FEATURE_SET_VICTOR, FEATURE_SETS, build_modeling_dataset, load_dataset, split_features_target
from src.evaluate import (
    EvaluationResult,
    evaluate_model,
    predict_scores,
    save_best_summary,
    save_confusion_matrix_plot,
    save_feature_importance,
    save_metrics,
    save_model_artifact,
    save_roc_comparison,
    save_roc_plot,
)
from src.models import get_model_definitions


@dataclass(frozen=True)
class TrainingConfig:
    data_path: Path = Path("data/raw/dataset_practica_final.csv")
    output_dir: Path = Path("outputs")
    models_dir: Path = Path("models")
    processed_data_path: Path | None = None
    feature_set: str = FEATURE_SET_VICTOR
    models: tuple[str, ...] = ("all",)
    metric: str = "f1"
    test_size: float = 0.2
    sample_size: int | None = None
    random_state: int = 42
    create_plots: bool = True
    tune: bool = False
    neural_epochs: int = 10


@dataclass(frozen=True)
class TrainingResult:
    metrics: pd.DataFrame
    best_model_name: str
    best_model_path: Path


def _stratified_sample(data: pd.DataFrame, sample_size: int | None, random_state: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(data):
        return data
    sample, _ = train_test_split(
        data,
        train_size=sample_size,
        stratify=data["is_canceled"],
        random_state=random_state,
    )
    return sample.reset_index(drop=True)


def _fit_estimator(definition, estimator, x_train, y_train, config: TrainingConfig):
    if config.tune and definition.param_grid:
        search = GridSearchCV(
            estimator=estimator,
            param_grid=definition.param_grid,
            cv=3,
            scoring=config.metric,
            n_jobs=-1,
        )
        search.fit(x_train, y_train)
        return search.best_estimator_
    estimator.fit(x_train, y_train)
    return estimator


def _validate_metric(metric: str) -> None:
    allowed_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    if metric not in allowed_metrics:
        raise ValueError(f"La metrica principal debe ser una de: {', '.join(sorted(allowed_metrics))}.")


def run_training(config: TrainingConfig) -> TrainingResult:
    """Ejecuta el pipeline completo de entrenamiento y evaluacion."""
    _validate_metric(config.metric)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_dataset(config.data_path)
    modeled_data = build_modeling_dataset(raw_data, feature_set=config.feature_set)
    if config.processed_data_path is not None:
        config.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        modeled_data.to_csv(config.processed_data_path, index=False)

    modeled_data = _stratified_sample(modeled_data, config.sample_size, config.random_state)

    x_data, y_data = split_features_target(modeled_data)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=config.test_size,
        stratify=y_data,
        random_state=config.random_state,
    )

    results: list[EvaluationResult] = []
    fitted_estimators = {}
    roc_curves = {}

    for definition in get_model_definitions(config.models, config.random_state, config.neural_epochs):
        estimator = _fit_estimator(definition, definition.builder(), x_train, y_train, config)
        model_path = save_model_artifact(estimator, config.models_dir / f"{definition.name}.joblib")
        result = evaluate_model(
            name=definition.name,
            estimator=estimator,
            x_test=x_test,
            y_test=y_test,
            description=definition.description,
            model_path=model_path,
        )
        results.append(result)
        fitted_estimators[definition.name] = estimator

        if config.create_plots:
            save_confusion_matrix_plot(definition.name, estimator, x_test, y_test, config.output_dir)
            save_roc_plot(definition.name, estimator, x_test, y_test, config.output_dir)
            save_feature_importance(definition.name, estimator, config.output_dir)
            scores = predict_scores(estimator, x_test)
            from sklearn.metrics import roc_auc_score, roc_curve

            fpr, tpr, _ = roc_curve(y_test, scores)
            roc_curves[definition.name] = (fpr, tpr, roc_auc_score(y_test, scores))

    metrics = save_metrics(results, config.output_dir / "metrics.csv")
    best_row = metrics.sort_values([config.metric, "roc_auc"], ascending=False).iloc[0]
    best_model_name = str(best_row["model"])
    best_model_path = save_model_artifact(
        fitted_estimators[best_model_name],
        config.models_dir / "best_model.joblib",
    )

    metrics.loc[metrics["model"] == best_model_name, "model_path"] = str(best_model_path)
    metrics.to_csv(config.output_dir / "metrics.csv", index=False)
    best_row = metrics[metrics["model"] == best_model_name].iloc[0]
    save_best_summary(best_row, config.output_dir / "best_model_summary.json", config.metric)

    if config.create_plots and roc_curves:
        save_roc_comparison(roc_curves, config.output_dir)

    return TrainingResult(
        metrics=metrics,
        best_model_name=best_model_name,
        best_model_path=best_model_path,
    )


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Entrena y evalua modelos de cancelacion hotelera.")
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/dataset_practica_final.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--processed-data-path", type=Path, default=None)
    parser.add_argument("--feature-set", default=FEATURE_SET_VICTOR, choices=FEATURE_SETS)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--metric", default="f1", choices=["accuracy", "precision", "recall", "f1", "roc_auc"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--neural-epochs", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    return TrainingConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        processed_data_path=args.processed_data_path,
        feature_set=args.feature_set,
        models=tuple(args.models),
        metric=args.metric,
        test_size=args.test_size,
        sample_size=args.sample_size,
        random_state=args.random_state,
        create_plots=not args.no_plots,
        tune=args.tune,
        neural_epochs=args.neural_epochs,
    )


def main() -> None:
    result = run_training(parse_args())
    print(f"Mejor modelo: {result.best_model_name}")
    print(f"Artefacto: {result.best_model_path}")
    print(result.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
