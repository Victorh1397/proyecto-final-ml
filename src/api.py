from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import (
    FEATURE_SET_ALEJANDRO,
    FEATURE_SET_BINARY,
    FEATURE_SET_POST_ASSIGNMENT,
    FEATURE_SET_VICTOR,
    TARGET_COLUMN,
    build_modeling_dataset,
)


DEFAULT_MODELS_DIR = Path("models/finalists_full")
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class PredictionRequest(BaseModel):
    feature_set: Literal[
        FEATURE_SET_VICTOR,
        FEATURE_SET_ALEJANDRO,
        FEATURE_SET_BINARY,
        FEATURE_SET_POST_ASSIGNMENT,
    ] = FEATURE_SET_VICTOR
    records: list[dict[str, Any]] = Field(min_length=1)


class PredictionResponse(BaseModel):
    model_name: str
    feature_set: str
    predictions: list[int]
    probabilities: list[float]


class ModelListResponse(BaseModel):
    models: list[str]


def _normalize_model_name(model_name: str) -> str:
    normalized = model_name.removesuffix(".joblib")
    if not MODEL_NAME_PATTERN.fullmatch(normalized):
        raise HTTPException(status_code=422, detail="Nombre de modelo no valido.")
    return normalized


def _resolve_model_path(models_dir: Path, model_name: str) -> Path:
    normalized = _normalize_model_name(model_name)
    model_path = (models_dir / f"{normalized}.joblib").resolve()
    models_root = models_dir.resolve()
    if not model_path.is_relative_to(models_root):
        raise HTTPException(status_code=422, detail="Ruta de modelo no valida.")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"No existe el modelo '{normalized}'.")
    return model_path


@lru_cache(maxsize=16)
def _load_model_cached(path: str, modified_time: int):
    return joblib.load(path)


def load_model(model_path: Path):
    return _load_model_cached(str(model_path), model_path.stat().st_mtime_ns)


def list_model_names(models_dir: Path) -> list[str]:
    if not models_dir.exists():
        return []
    return sorted(path.stem for path in models_dir.glob("*.joblib"))


def prepare_prediction_frame(records: list[dict[str, Any]], feature_set: str) -> pd.DataFrame:
    raw_records = pd.DataFrame(records)
    modeled_records = build_modeling_dataset(raw_records, feature_set=feature_set)
    if TARGET_COLUMN in modeled_records.columns:
        modeled_records = modeled_records.drop(columns=[TARGET_COLUMN])
    if modeled_records.empty:
        raise ValueError("No hay registros validos para predecir despues del preprocesamiento.")
    return modeled_records


def predict_probabilities(estimator, records: pd.DataFrame) -> list[float]:
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(records))
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1].astype(float).tolist()
        return probabilities.ravel().astype(float).tolist()

    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(records), dtype=float).ravel()
        score_min = scores.min()
        score_max = scores.max()
        if score_max == score_min:
            return [0.5 for _ in scores]
        return ((scores - score_min) / (score_max - score_min)).astype(float).tolist()

    return np.asarray(estimator.predict(records), dtype=float).ravel().tolist()


def create_app(models_dir: str | Path | None = None) -> FastAPI:
    selected_models_dir = Path(models_dir or os.getenv("HOTEL_MODELS_DIR", DEFAULT_MODELS_DIR))
    app = FastAPI(
        title="Hotel Cancellation Prediction API",
        description="API para predecir cancelaciones hoteleras con un modelo entrenado seleccionado.",
        version="1.0.0",
    )
    app.state.models_dir = selected_models_dir

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/models", response_model=ModelListResponse)
    def models() -> ModelListResponse:
        return ModelListResponse(models=list_model_names(app.state.models_dir))

    @app.post("/predict/{model_name}", response_model=PredictionResponse)
    def predict(model_name: str, request: PredictionRequest) -> PredictionResponse:
        normalized_model_name = _normalize_model_name(model_name)
        model_path = _resolve_model_path(app.state.models_dir, normalized_model_name)
        estimator = load_model(model_path)

        try:
            prediction_frame = prepare_prediction_frame(request.records, request.feature_set)
            predictions = np.asarray(estimator.predict(prediction_frame)).astype(int).ravel().tolist()
            probabilities = predict_probabilities(estimator, prediction_frame)
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        return PredictionResponse(
            model_name=normalized_model_name,
            feature_set=request.feature_set,
            predictions=predictions,
            probabilities=probabilities,
        )

    return app


app = create_app()
