from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg", force=True)


@dataclass
class EvaluationResult:
    model: str
    estimator_description: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    model_path: str


def predict_scores(estimator, x_data) -> np.ndarray:
    """Devuelve una puntuacion continua para ROC-AUC."""
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x_data)
        return np.asarray(probabilities)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(x_data)
        scores = np.asarray(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())
    return estimator.predict(x_data)


def evaluate_model(
    name: str,
    estimator,
    x_test,
    y_test,
    description: str,
    model_path: Path,
) -> EvaluationResult:
    predictions = estimator.predict(x_test)
    scores = predict_scores(estimator, x_test)
    return EvaluationResult(
        model=name,
        estimator_description=description,
        accuracy=accuracy_score(y_test, predictions),
        precision=precision_score(y_test, predictions, zero_division=0),
        recall=recall_score(y_test, predictions, zero_division=0),
        f1=f1_score(y_test, predictions, zero_division=0),
        roc_auc=roc_auc_score(y_test, scores),
        model_path=str(model_path),
    )


def save_model_artifact(estimator, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path, compress=3)
    return path


def save_metrics(results: list[EvaluationResult], path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([asdict(result) for result in results])
    metrics = metrics.sort_values(["f1", "roc_auc"], ascending=False).reset_index(drop=True)
    metrics.to_csv(path, index=False)
    return metrics


def save_best_summary(best_row: pd.Series, path: Path, primary_metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "primary_metric": primary_metric,
        "selected_model": best_row["model"],
        "selected_score": float(best_row[primary_metric]),
        "model_path": best_row["model_path"],
        "metrics": {
            "accuracy": float(best_row["accuracy"]),
            "precision": float(best_row["precision"]),
            "recall": float(best_row["recall"]),
            "f1": float(best_row["f1"]),
            "roc_auc": float(best_row["roc_auc"]),
        },
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def save_confusion_matrix_plot(name: str, estimator, x_test, y_test, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    predictions = estimator.predict(x_test)
    matrix = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(matrix, display_labels=["No cancela", "Cancela"])
    display.plot(values_format="d", cmap="Blues")
    plt.title(f"Matriz de confusion - {name}")
    path = output_dir / f"confusion_matrix_{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def save_roc_plot(name: str, estimator, x_test, y_test, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    scores = predict_scores(estimator, x_test)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, scores)
    auc_score = roc_auc_score(y_test, scores)

    plt.figure(figsize=(7, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Curva ROC - {name}")
    plt.legend(loc="lower right")
    path = output_dir / f"roc_curve_{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def save_roc_comparison(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    for name, (false_positive_rate, true_positive_rate, auc_score) in curves.items():
        plt.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Comparativa de curvas ROC")
    plt.legend(loc="lower right")
    path = output_dir / "roc_curves_comparison.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def save_feature_importance(name: str, estimator, output_dir: Path, max_features: int = 20) -> Path | None:
    import matplotlib.pyplot as plt

    if not hasattr(estimator, "named_steps") or "model" not in estimator.named_steps:
        return None

    model = estimator.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
    else:
        return None

    preprocessor = estimator.named_steps["prep"]
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = np.array([f"feature_{index}" for index in range(len(values))])

    importance = (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(max_features)
        .sort_values("importance")
    )
    if importance.empty:
        return None

    plt.figure(figsize=(9, 6))
    plt.barh(importance["feature"], importance["importance"], color="#2f6f73")
    plt.xlabel("Importancia")
    plt.title(f"Importancia de variables - {name}")
    path = output_dir / f"feature_importance_{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return path
