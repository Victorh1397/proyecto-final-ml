from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.data import FEATURE_SET_VICTOR, FEATURE_SETS, build_modeling_dataset, load_dataset, split_features_target
from src.evaluate import (
    EvaluationResult,
    evaluate_model,
    save_best_summary,
    save_confusion_matrix_plot,
    save_feature_importance,
    save_metrics,
    save_model_artifact,
    save_roc_plot,
)
from src.models import build_pipeline
from src.train import _stratified_sample


@dataclass(frozen=True)
class OptunaTuningConfig:
    data_path: Path = Path("data/raw/dataset_practica_final.csv")
    output_dir: Path = Path("outputs/optuna_lightgbm")
    models_dir: Path = Path("models/optuna_lightgbm")
    storage: str = "sqlite:///outputs/optuna_lightgbm/optuna_study.db"
    study_name: str = "hotel_lightgbm_f1"
    feature_set: str = FEATURE_SET_VICTOR
    metric: str = "f1"
    timeout: int = 28_800
    n_trials: int | None = None
    test_size: float = 0.2
    sample_size: int | None = None
    cv_splits: int = 3
    random_state: int = 42
    create_plots: bool = True
    model_jobs: int = -1
    study_jobs: int = 1


@dataclass(frozen=True)
class OptunaTuningResult:
    best_value: float
    best_model_path: Path
    best_params_path: Path
    trials_path: Path
    metrics_path: Path


def _suggest_lightgbm_params(trial: optuna.Trial, random_state: int, model_jobs: int) -> dict[str, object]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256, step=8),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10, 12, 14, 16]),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 180, step=10),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "random_state": random_state,
        "n_jobs": model_jobs,
        "verbose": -1,
    }


def _build_lightgbm_pipeline(params: dict[str, object]):
    return build_pipeline(LGBMClassifier(**params))


def _validate_metric(metric: str) -> None:
    allowed_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    if metric not in allowed_metrics:
        raise ValueError(f"La metrica principal debe ser una de: {', '.join(sorted(allowed_metrics))}.")


def _save_best_params(study: optuna.Study, config: OptunaTuningConfig, path: Path) -> Path:
    payload = {
        "study_name": config.study_name,
        "feature_set": config.feature_set,
        "primary_metric": config.metric,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _save_trials(study: optuna.Study, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    trials = study.trials_dataframe()
    trials.to_csv(path, index=False)
    return path


def run_optuna_tuning(config: OptunaTuningConfig) -> OptunaTuningResult:
    _validate_metric(config.metric)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_dataset(config.data_path)
    modeled_data = build_modeling_dataset(raw_data, feature_set=config.feature_set)
    modeled_data = _stratified_sample(modeled_data, config.sample_size, config.random_state)
    x_data, y_data = split_features_target(modeled_data)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=config.test_size,
        stratify=y_data,
        random_state=config.random_state,
    )

    cv = StratifiedKFold(n_splits=config.cv_splits, shuffle=True, random_state=config.random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_lightgbm_params(trial, config.random_state, config.model_jobs)
        estimator = _build_lightgbm_pipeline(params)
        scores = cross_validate(
            estimator,
            x_train,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )
        for metric_name in scoring:
            trial.set_user_attr(metric_name, float(scores[f"test_{metric_name}"].mean()))
        return float(scores[f"test_{config.metric}"].mean())

    sampler = optuna.samplers.TPESampler(seed=config.random_state)
    study = optuna.create_study(
        study_name=config.study_name,
        storage=config.storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout,
        n_jobs=config.study_jobs,
        show_progress_bar=True,
    )

    best_params = _suggest_lightgbm_params(study.best_trial, config.random_state, config.model_jobs)
    best_estimator = _build_lightgbm_pipeline(best_params)
    best_estimator.fit(x_train, y_train)
    model_path = save_model_artifact(best_estimator, config.models_dir / "best_model.joblib")

    evaluation = evaluate_model(
        name="lightgbm_optuna",
        estimator=best_estimator,
        x_test=x_test,
        y_test=y_test,
        description="LightGBM optimizado con Optuna",
        model_path=model_path,
    )
    metrics_path = config.output_dir / "metrics.csv"
    metrics = save_metrics([evaluation], metrics_path)
    save_best_summary(pd.Series(metrics.iloc[0]), config.output_dir / "best_model_summary.json", config.metric)

    if config.create_plots:
        save_confusion_matrix_plot("lightgbm_optuna", best_estimator, x_test, y_test, config.output_dir)
        save_roc_plot("lightgbm_optuna", best_estimator, x_test, y_test, config.output_dir)
        save_feature_importance("lightgbm_optuna", best_estimator, config.output_dir)

    best_params_path = _save_best_params(study, config, config.output_dir / "best_params.json")
    trials_path = _save_trials(study, config.output_dir / "trials.csv")

    return OptunaTuningResult(
        best_value=float(study.best_value),
        best_model_path=model_path,
        best_params_path=best_params_path,
        trials_path=trials_path,
        metrics_path=metrics_path,
    )


def parse_args() -> OptunaTuningConfig:
    parser = argparse.ArgumentParser(description="Optimiza LightGBM con Optuna para cancelaciones hoteleras.")
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/dataset_practica_final.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/optuna_lightgbm"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/optuna_lightgbm"))
    parser.add_argument("--storage", default="sqlite:///outputs/optuna_lightgbm/optuna_study.db")
    parser.add_argument("--study-name", default="hotel_lightgbm_f1")
    parser.add_argument("--feature-set", default=FEATURE_SET_VICTOR, choices=FEATURE_SETS)
    parser.add_argument("--metric", default="f1", choices=["accuracy", "precision", "recall", "f1", "roc_auc"])
    parser.add_argument("--timeout", type=int, default=28_800)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-jobs", type=int, default=-1)
    parser.add_argument("--study-jobs", type=int, default=1)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    return OptunaTuningConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        storage=args.storage,
        study_name=args.study_name,
        feature_set=args.feature_set,
        metric=args.metric,
        timeout=args.timeout,
        n_trials=args.n_trials,
        test_size=args.test_size,
        sample_size=args.sample_size,
        cv_splits=args.cv_splits,
        random_state=args.random_state,
        create_plots=not args.no_plots,
        model_jobs=args.model_jobs,
        study_jobs=args.study_jobs,
    )


def main() -> None:
    result = run_optuna_tuning(parse_args())
    print(f"Mejor valor Optuna: {result.best_value:.6f}")
    print(f"Modelo: {result.best_model_path}")
    print(f"Metricas: {result.metrics_path}")
    print(f"Parametros: {result.best_params_path}")
    print(f"Trials: {result.trials_path}")


if __name__ == "__main__":
    main()
