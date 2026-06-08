from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.data import FEATURE_SET_BINARY, FEATURE_SETS, build_modeling_dataset, load_dataset, split_features_target
from src.evaluate import EvaluationResult, save_best_summary, save_metrics, save_model_artifact
from src.train import _stratified_sample


@dataclass(frozen=True)
class NativeOptunaConfig:
    data_path: Path = Path("data/raw/dataset_practica_final.csv")
    output_dir: Path = Path("outputs/optuna_binary_native_lgbm")
    models_dir: Path = Path("models/optuna_binary_native_lgbm")
    storage: str = "sqlite:///outputs/optuna_binary_native_lgbm/optuna_study.db"
    study_name: str = "hotel_binary_native_lgbm_f1"
    feature_set: str = FEATURE_SET_BINARY
    metric: str = "f1"
    timeout: int = 86_400
    n_trials: int | None = None
    test_size: float = 0.2
    sample_size: int | None = None
    cv_splits: int = 3
    random_state: int = 42
    model_jobs: int = -1
    study_jobs: int = 1


@dataclass(frozen=True)
class NativeOptunaResult:
    best_value: float
    best_model_path: Path
    best_params_path: Path
    trials_path: Path
    metrics_path: Path


class NativeCategoricalLGBMClassifier(ClassifierMixin, BaseEstimator):
    """LightGBM wrapper that preserves pandas categorical metadata for inference."""

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self.params = params or {}

    def fit(self, x_data: pd.DataFrame, y_data: pd.Series):
        x_native = x_data.copy()
        self.categorical_columns_ = x_native.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        self.categories_ = {}
        for column in self.categorical_columns_:
            x_native[column] = x_native[column].astype("category")
            self.categories_[column] = list(x_native[column].cat.categories)

        self.model_ = LGBMClassifier(**self.params)
        self.model_.fit(x_native, y_data, categorical_feature=self.categorical_columns_)
        self.classes_ = self.model_.classes_
        return self

    def _prepare(self, x_data: pd.DataFrame) -> pd.DataFrame:
        x_native = x_data.copy()
        for column in self.categorical_columns_:
            x_native[column] = pd.Categorical(x_native[column], categories=self.categories_[column])
        return x_native

    def predict(self, x_data: pd.DataFrame):
        return self.model_.predict(self._prepare(x_data))

    def predict_proba(self, x_data: pd.DataFrame):
        return self.model_.predict_proba(self._prepare(x_data))


def _suggest_params(trial: optuna.Trial, random_state: int, model_jobs: int) -> dict[str, object]:
    return {
        "objective": "binary",
        "n_estimators": trial.suggest_int("n_estimators", 180, 1400, step=20),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.09, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 63, 255, step=8),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 8, 10, 12, 14, 16]),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 120, step=10),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.65, 0.95),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.65, 0.95),
        "bagging_freq": 1,
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.15),
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": model_jobs,
        "verbose": -1,
    }


def _validate_metric(metric: str) -> None:
    allowed_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    if metric not in allowed_metrics:
        raise ValueError(f"La metrica principal debe ser una de: {', '.join(sorted(allowed_metrics))}.")


def _save_best_params(study: optuna.Study, config: NativeOptunaConfig, path: Path) -> Path:
    payload = {
        "study_name": config.study_name,
        "feature_set": config.feature_set,
        "primary_metric": config.metric,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
        "fixed_params": {
            "objective": "binary",
            "class_weight": "balanced",
            "native_categorical": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _save_trials(study: optuna.Study, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(path, index=False)
    return path


def _evaluate(name: str, estimator, x_test, y_test, model_path: Path) -> EvaluationResult:
    predictions = estimator.predict(x_test)
    scores = estimator.predict_proba(x_test)[:, 1]
    return EvaluationResult(
        model=name,
        estimator_description="LightGBM con categoricas nativas optimizado con Optuna",
        accuracy=accuracy_score(y_test, predictions),
        precision=precision_score(y_test, predictions, zero_division=0),
        recall=recall_score(y_test, predictions, zero_division=0),
        f1=f1_score(y_test, predictions, zero_division=0),
        roc_auc=roc_auc_score(y_test, scores),
        model_path=str(model_path),
    )


def run_native_optuna(config: NativeOptunaConfig) -> NativeOptunaResult:
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
        params = _suggest_params(trial, config.random_state, config.model_jobs)
        estimator = NativeCategoricalLGBMClassifier(params)
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

    best_params = _suggest_params(study.best_trial, config.random_state, config.model_jobs)
    best_estimator = NativeCategoricalLGBMClassifier(best_params).fit(x_train, y_train)
    model_path = save_model_artifact(best_estimator, config.models_dir / "best_model.joblib")

    evaluation = _evaluate("lightgbm_native_optuna", best_estimator, x_test, y_test, model_path)
    metrics_path = config.output_dir / "metrics.csv"
    metrics = save_metrics([evaluation], metrics_path)
    save_best_summary(pd.Series(metrics.iloc[0]), config.output_dir / "best_model_summary.json", config.metric)

    best_params_path = _save_best_params(study, config, config.output_dir / "best_params.json")
    trials_path = _save_trials(study, config.output_dir / "trials.csv")
    joblib.dump(best_estimator, model_path, compress=3)

    return NativeOptunaResult(
        best_value=float(study.best_value),
        best_model_path=model_path,
        best_params_path=best_params_path,
        trials_path=trials_path,
        metrics_path=metrics_path,
    )


def parse_args() -> NativeOptunaConfig:
    parser = argparse.ArgumentParser(description="Optimiza LightGBM nativo categorico con Optuna.")
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/dataset_practica_final.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/optuna_binary_native_lgbm"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/optuna_binary_native_lgbm"))
    parser.add_argument("--storage", default="sqlite:///outputs/optuna_binary_native_lgbm/optuna_study.db")
    parser.add_argument("--study-name", default="hotel_binary_native_lgbm_f1")
    parser.add_argument("--feature-set", default=FEATURE_SET_BINARY, choices=FEATURE_SETS)
    parser.add_argument("--metric", default="f1", choices=["accuracy", "precision", "recall", "f1", "roc_auc"])
    parser.add_argument("--timeout", type=int, default=86_400)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-jobs", type=int, default=-1)
    parser.add_argument("--study-jobs", type=int, default=1)
    args = parser.parse_args()
    return NativeOptunaConfig(
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
        model_jobs=args.model_jobs,
        study_jobs=args.study_jobs,
    )


def main() -> None:
    result = run_native_optuna(parse_args())
    print(f"Mejor valor Optuna: {result.best_value:.6f}")
    print(f"Modelo: {result.best_model_path}")
    print(f"Metricas: {result.metrics_path}")
    print(f"Parametros: {result.best_params_path}")
    print(f"Trials: {result.trials_path}")


if __name__ == "__main__":
    main()
