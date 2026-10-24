"""Model candidates, pipeline construction and persistence."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import sklearn
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

import config
from data_preprocessing import build_preprocessor
from utils import (MLProjectError, ModelIncompatibleError, ModelNotFoundError,
                   ensure_dirs, get_logger, load_json, save_json)

log = get_logger(__name__)


def get_model_candidates(random_state: int = config.RANDOM_STATE, quick: bool = False) -> dict[str, BaseEstimator]:
    return {
        config.BASELINE_NAME: DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=30 if quick else 300, min_samples_leaf=3, n_jobs=-1, random_state=random_state),
    }


def build_pipeline(estimator: BaseEstimator, features: list[str] | None = None) -> Pipeline:
    """Preprocessing and estimator travel together so prediction == training logic."""
    return Pipeline([("preprocess", build_preprocessor(features)), ("model", estimator)])


def _paths(models_dir: str | Path | None) -> tuple[Path, Path]:
    d = Path(models_dir) if models_dir else config.MODELS_DIR
    return d / config.MODEL_FILENAME, d / config.METADATA_FILENAME


def model_exists(models_dir: str | Path | None = None) -> bool:
    m, meta = _paths(models_dir)
    return m.exists() and meta.exists()


def save_artifacts(pipe: Pipeline, metadata: dict, models_dir: str | Path | None = None) -> None:
    model_path, meta_path = _paths(models_dir)
    ensure_dirs(model_path.parent)
    tmp = model_path.with_suffix(".tmp")
    try:
        joblib.dump(pipe, tmp)
        os.replace(tmp, model_path)
    except PermissionError as exc:
        raise MLProjectError(f"Cannot write model to {model_path}: permission denied.") from exc
    save_json(metadata, meta_path)
    log.info("Saved model -> %s", model_path)


def load_metadata(models_dir: str | Path | None = None) -> dict:
    _, meta_path = _paths(models_dir)
    if not meta_path.exists():
        raise ModelNotFoundError("No trained model found. Run `python train.py` first (or use the Train button in the app).")
    try:
        return load_json(meta_path)
    except ValueError as exc:
        raise ModelIncompatibleError(f"{meta_path} is corrupted: {exc}. Retrain the model.") from exc


def load_pipeline(models_dir: str | Path | None = None) -> Pipeline:
    model_path, _ = _paths(models_dir)
    if not model_path.exists():
        raise ModelNotFoundError("No trained model found. Run `python train.py` first (or use the Train button in the app).")
    try:
        pipe = joblib.load(model_path)
    except Exception as exc:  # joblib/pickle raise many different types for bad or incompatible files
        raise ModelIncompatibleError(
            f"Could not load {model_path} ({type(exc).__name__}: {exc}). "
            "It may be corrupted or built with another scikit-learn version - retrain with `python train.py`.") from exc
    try:
        saved = load_metadata(models_dir).get("library_versions", {}).get("scikit-learn")
        if saved and saved != sklearn.__version__:
            log.warning("Model was trained with scikit-learn %s but %s is installed.", saved, sklearn.__version__)
    except MLProjectError:
        pass
    return pipe
