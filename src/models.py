from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    builder: Callable[[], Pipeline]
    description: str
    param_grid: dict[str, list[object]] = field(default_factory=dict)
    uses_optional_dependency: bool = False


class KerasBinaryClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper minimo para entrenar una red Keras dentro de un Pipeline."""

    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 256,
        random_state: int = 42,
        verbose: int = 0,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, x_data, y_data):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.random_state)
        x_array = np.asarray(x_data).astype("float32")
        y_array = np.asarray(y_data).astype("float32")

        self.classes_ = np.array([0, 1])
        self.model_ = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(x_array.shape[1],)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model_.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        self.model_.fit(
            x_array,
            y_array,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=self.verbose,
        )
        return self

    def predict_proba(self, x_data):
        x_array = np.asarray(x_data).astype("float32")
        probabilities = self.model_.predict(x_array, verbose=0).reshape(-1)
        return np.column_stack([1 - probabilities, probabilities])

    def predict(self, x_data):
        return (self.predict_proba(x_data)[:, 1] >= 0.5).astype(int)


def build_preprocessor() -> ColumnTransformer:
    """Crea el preprocesador compartido por los modelos."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                make_column_selector(dtype_include=["object", "category", "string"]),
            ),
            ("num", StandardScaler(), make_column_selector(dtype_include=["number"])),
        ],
        remainder="drop",
    )


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(steps=[("prep", build_preprocessor()), ("model", estimator)])


def _gradient_boosting_pipeline(random_state: int) -> tuple[Pipeline, str, bool]:
    try:
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            n_estimators=140,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        return build_pipeline(estimator), "XGBoostClassifier", True
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(
            n_estimators=140,
            max_depth=-1,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        return build_pipeline(estimator), "LightGBMClassifier", True
    except Exception:
        pass

    estimator = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.08,
        random_state=random_state,
    )
    return build_pipeline(estimator), "HistGradientBoostingClassifier fallback", False


def _neural_network_pipeline(random_state: int, neural_epochs: int) -> tuple[Pipeline, str, bool]:
    try:
        import tensorflow  # noqa: F401

        estimator = KerasBinaryClassifier(
            epochs=neural_epochs,
            batch_size=256,
            random_state=random_state,
            verbose=0,
        )
        return build_pipeline(estimator), "TensorFlow/Keras MLP", True
    except Exception:
        estimator = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.0005,
            early_stopping=True,
            max_iter=max(50, neural_epochs * 20),
            random_state=random_state,
        )
        return build_pipeline(estimator), "sklearn MLPClassifier fallback", False


def _lightgbm_pipeline(random_state: int) -> tuple[Pipeline, str, bool]:
    try:
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(
            n_estimators=220,
            learning_rate=0.06,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        return build_pipeline(estimator), "LightGBMClassifier", True
    except Exception:
        estimator = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.06,
            random_state=random_state,
        )
        return build_pipeline(estimator), "HistGradientBoostingClassifier fallback for LightGBM", False


def _catboost_pipeline(random_state: int) -> tuple[Pipeline, str, bool]:
    try:
        from catboost import CatBoostClassifier

        estimator = CatBoostClassifier(
            iterations=220,
            depth=6,
            learning_rate=0.06,
            loss_function="Logloss",
            eval_metric="F1",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        return build_pipeline(estimator), "CatBoostClassifier", True
    except Exception:
        estimator = GradientBoostingClassifier(
            n_estimators=140,
            learning_rate=0.06,
            max_depth=3,
            random_state=random_state,
        )
        return build_pipeline(estimator), "GradientBoostingClassifier fallback for CatBoost", False


def get_model_definitions(
    selected_models: tuple[str, ...] | list[str] | None = None,
    random_state: int = 42,
    neural_epochs: int = 10,
) -> list[ModelDefinition]:
    """Devuelve la lista de modelos disponibles para entrenamiento."""
    selected = tuple(selected_models or ("all",))
    include_all = "all" in selected

    gradient_pipeline, gradient_description, gradient_optional = _gradient_boosting_pipeline(random_state)
    neural_pipeline, neural_description, neural_optional = _neural_network_pipeline(random_state, neural_epochs)
    lightgbm_pipeline, lightgbm_description, lightgbm_optional = _lightgbm_pipeline(random_state)
    catboost_pipeline, catboost_description, catboost_optional = _catboost_pipeline(random_state)

    definitions = [
        ModelDefinition(
            name="logistic_regression",
            builder=lambda: build_pipeline(
                LogisticRegression(
                    max_iter=800,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=random_state,
                )
            ),
            description="Regresion logistica con class_weight='balanced'",
            param_grid={"model__C": [0.1, 1.0, 10.0]},
        ),
        ModelDefinition(
            name="decision_tree",
            builder=lambda: build_pipeline(
                DecisionTreeClassifier(
                    max_depth=10,
                    min_samples_leaf=8,
                    class_weight="balanced",
                    random_state=random_state,
                )
            ),
            description="Arbol de decision balanceado",
            param_grid={
                "model__max_depth": [6, 10, 14],
                "model__min_samples_leaf": [4, 8, 16],
            },
        ),
        ModelDefinition(
            name="random_forest",
            builder=lambda: build_pipeline(
                RandomForestClassifier(
                    n_estimators=160,
                    max_depth=18,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                )
            ),
            description="Random Forest con class_weight='balanced_subsample'",
            param_grid={
                "model__n_estimators": [120, 180],
                "model__max_depth": [14, 20],
                "model__min_samples_leaf": [3, 8],
            },
        ),
        ModelDefinition(
            name="gradient_boosting",
            builder=lambda pipeline=gradient_pipeline: pipeline,
            description=gradient_description,
            param_grid={},
            uses_optional_dependency=gradient_optional,
        ),
        ModelDefinition(
            name="neural_network",
            builder=lambda pipeline=neural_pipeline: pipeline,
            description=neural_description,
            param_grid={},
            uses_optional_dependency=neural_optional,
        ),
        ModelDefinition(
            name="extra_trees",
            builder=lambda: build_pipeline(
                ExtraTreesClassifier(
                    n_estimators=220,
                    max_depth=None,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                )
            ),
            description="Extra Trees con class_weight='balanced'",
            param_grid={},
        ),
        ModelDefinition(
            name="ada_boost",
            builder=lambda: build_pipeline(
                AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=3, random_state=random_state),
                    n_estimators=140,
                    learning_rate=0.08,
                    random_state=random_state,
                )
            ),
            description="AdaBoost con arbol base poco profundo",
            param_grid={},
        ),
        ModelDefinition(
            name="bagging",
            builder=lambda: build_pipeline(
                BaggingClassifier(
                    estimator=DecisionTreeClassifier(
                        max_depth=12,
                        min_samples_leaf=6,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                    n_estimators=80,
                    max_samples=0.85,
                    max_features=0.85,
                    random_state=random_state,
                    n_jobs=-1,
                )
            ),
            description="Bagging de arboles de decision",
            param_grid={},
        ),
        ModelDefinition(
            name="knn",
            builder=lambda: build_pipeline(
                KNeighborsClassifier(
                    n_neighbors=25,
                    weights="distance",
                    n_jobs=-1,
                )
            ),
            description="K-Nearest Neighbors con pesos por distancia",
            param_grid={},
        ),
        ModelDefinition(
            name="svm_rbf",
            builder=lambda: build_pipeline(
                SVC(
                    kernel="rbf",
                    C=2.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=random_state,
                )
            ),
            description="Support Vector Machine con kernel RBF",
            param_grid={},
        ),
        ModelDefinition(
            name="linear_svm",
            builder=lambda: build_pipeline(
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    dual="auto",
                    max_iter=5000,
                    random_state=random_state,
                )
            ),
            description="Linear SVM con pesos balanceados",
            param_grid={},
        ),
        ModelDefinition(
            name="gaussian_nb",
            builder=lambda: build_pipeline(GaussianNB()),
            description="Gaussian Naive Bayes",
            param_grid={},
        ),
        ModelDefinition(
            name="ridge_classifier",
            builder=lambda: build_pipeline(
                RidgeClassifier(
                    alpha=1.0,
                    class_weight="balanced",
                    random_state=random_state,
                )
            ),
            description="RidgeClassifier con pesos balanceados",
            param_grid={},
        ),
        ModelDefinition(
            name="lightgbm",
            builder=lambda pipeline=lightgbm_pipeline: pipeline,
            description=lightgbm_description,
            param_grid={},
            uses_optional_dependency=lightgbm_optional,
        ),
        ModelDefinition(
            name="catboost",
            builder=lambda pipeline=catboost_pipeline: pipeline,
            description=catboost_description,
            param_grid={},
            uses_optional_dependency=catboost_optional,
        ),
    ]

    if include_all:
        return definitions

    by_name = {definition.name: definition for definition in definitions}
    unknown = sorted(set(selected) - set(by_name))
    if unknown:
        raise ValueError(f"Modelos no reconocidos: {', '.join(unknown)}")
    return [by_name[name] for name in selected]
