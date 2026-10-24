"""Metrics, time-aware cross-validation, residuals and importance."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import config
from utils import InsufficientDataError, TrainingError, get_logger

log = get_logger(__name__)


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_validate_candidates(pipelines: dict, X: pd.DataFrame, y: pd.Series,
                              n_splits: int = config.CV_SPLITS, progress=None) -> pd.DataFrame:
    """Expanding-window CV: every validation fold lies strictly AFTER its training fold."""
    if len(X) < (n_splits + 1) * 5:
        raise InsufficientDataError(f"{len(X)} training rows are too few for {n_splits}-fold time-series CV.")
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for i, (name, pipe) in enumerate(pipelines.items()):
        start = time.perf_counter()
        fold_scores, error = [], None
        try:
            for tr, va in splitter.split(X):
                model = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
                fold_scores.append(regression_metrics(y.iloc[va], model.predict(X.iloc[va])))
        except Exception as exc:  # isolate one failing model so the others can still be compared
            error = f"{type(exc).__name__}: {exc}"
            log.exception("Model %s failed during cross-validation", name)
        row = {"model": name, "status": "ok" if error is None else "failed", "error": error,
               "fit_seconds": round(time.perf_counter() - start, 2)}
        if error is None:
            for m in ("mae", "rmse", "r2"):
                vals = [f[m] for f in fold_scores]
                row[f"cv_{m}_mean"] = float(np.mean(vals))
                row[f"cv_{m}_std"] = float(np.std(vals))
        rows.append(row)
        if progress:
            progress((i + 1) / len(pipelines), f"Cross-validated {name}")
    return pd.DataFrame(rows)


def select_best_model(cv: pd.DataFrame, metric: str = config.SELECTION_METRIC) -> str:
    col = f"cv_{metric}_mean"
    if col not in cv.columns:
        raise TrainingError("All models failed during cross-validation; see the log above.")
    ok = cv[(cv["status"] == "ok") & (cv["model"] != config.BASELINE_NAME)]
    if ok.empty:
        raise TrainingError("No candidate model trained successfully; see the log above.")
    return str(ok.sort_values(col).iloc[0]["model"])


def residual_frame(datetimes, y_true, y_pred) -> pd.DataFrame:
    out = pd.DataFrame({"datetime": datetimes, "actual": np.asarray(y_true, dtype=float),
                        "predicted": np.asarray(y_pred, dtype=float)})
    out["residual"] = out["actual"] - out["predicted"]
    return out


def importance_table(pipe, X: pd.DataFrame, y: pd.Series, random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Permutation importance (model-agnostic): increase in MAE when a feature is shuffled."""
    res = permutation_importance(pipe, X, y, scoring="neg_mean_absolute_error",
                                 n_repeats=10, random_state=random_state, n_jobs=1)
    return (pd.DataFrame({"feature": X.columns, "importance_mae_increase": res.importances_mean,
                          "std": res.importances_std})
            .sort_values("importance_mae_increase", ascending=False).reset_index(drop=True))
