"""End-to-end training: raw data -> cleaning -> chronological split -> CV model
comparison -> final fit -> one-time test evaluation -> saved artefacts.

Usage:  python train.py [--data PATH] [--quick] [--no-eda]
"""
from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone

import config
from data_preprocessing import (clean_dataset, get_xy, load_raw_data,
                                split_chronological)
from eda import generate_eda
from evaluation import (cross_validate_candidates, importance_table,
                        regression_metrics, residual_frame, select_best_model)
from pipeline import build_pipeline, get_model_candidates, save_artifacts
from utils import MLProjectError, TrainingError, ensure_dirs, get_logger, save_json

log = get_logger("train")

METHODOLOGY = (
    "Rows are ordered by time. The first 80% of the timeline is the training set and the last 20% is a "
    "held-out test set that is evaluated exactly once, for the already-selected model and the baseline. "
    "Models are compared with expanding-window TimeSeriesSplit cross-validation inside the training set "
    "(each validation fold is strictly later than its training fold). Imputation and scaling are fitted "
    "on training folds only, inside the saved Pipeline."
)


def _feature_ranges(X: pd.DataFrame) -> dict:
    out = {}
    for c in X.columns:
        s = X[c].dropna()
        out[c] = {"min": float(s.min()), "max": float(s.max()), "median": float(s.median()),
                  "mean": float(s.mean()), "p01": float(s.quantile(.01)), "p99": float(s.quantile(.99))}
    return out


def _simple_params(estimator) -> dict:
    return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in estimator.get_params().items()}


def run_training(raw_path=None, models_dir=None, reports_dir=None, processed_dir=None,
                 quick: bool = False, make_eda: bool = True, progress=None) -> dict:
    """Run the whole workflow and return the metadata dict that was saved."""
    report = progress or (lambda frac, msg: None)
    models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
    reports_dir = Path(reports_dir) if reports_dir else config.REPORTS_DIR
    processed_dir = Path(processed_dir) if processed_dir else config.DATA_PROCESSED_DIR
    figures_dir, metrics_dir = reports_dir / "figures", reports_dir / "metrics"
    ensure_dirs(models_dir, figures_dir, metrics_dir, processed_dir)
    features, target = list(config.FEATURE_COLUMNS), config.TARGET_COLUMN

    report(0.02, "Loading raw dataset")
    raw = load_raw_data(raw_path)
    log.info("Loaded raw data: %s rows x %s columns", *raw.shape)

    report(0.08, "Validating and cleaning")
    clean, summary = clean_dataset(raw, features, target)
    clean.to_csv(processed_dir / "cleaned.csv", index=False)
    save_json(summary, metrics_dir / "cleaning_summary.json")
    log.info("Cleaning: %d -> %d rows (%.2f%% removed)", summary["rows_raw"], summary["rows_final"], summary["pct_removed"])

    report(0.14, "Chronological train/test split")
    train_df, test_df = split_chronological(clean)
    X_train, y_train = get_xy(train_df, features, target)
    X_test, y_test = get_xy(test_df, features, target)
    log.info("Train rows=%d (%s -> %s) | Test rows=%d (%s -> %s)", len(train_df), train_df.datetime.min(),
             train_df.datetime.max(), len(test_df), test_df.datetime.min(), test_df.datetime.max())

    if make_eda:
        report(0.18, "Generating EDA figures (training portion only)")
        generate_eda(train_df, summary, figures_dir, metrics_dir, features, target)

    candidates = get_model_candidates(config.RANDOM_STATE, quick)
    pipelines = {name: build_pipeline(est, features) for name, est in candidates.items()}
    n_splits = 3 if quick else config.CV_SPLITS

    report(0.25, "Cross-validating models (time-aware)")
    cv = cross_validate_candidates(
        pipelines, X_train, y_train, n_splits,
        progress=lambda f, m: report(0.25 + 0.5 * f, m))
    cv.to_csv(metrics_dir / "cv_results.csv", index=False)
    best_name = select_best_model(cv, config.SELECTION_METRIC)
    log.info("Selected model by CV %s: %s", config.SELECTION_METRIC.upper(), best_name)

    report(0.80, f"Fitting final model: {best_name}")
    try:
        final = clone(pipelines[best_name]).fit(X_train, y_train)
        baseline = clone(pipelines[config.BASELINE_NAME]).fit(X_train, y_train)
    except Exception as exc:
        raise TrainingError(f"Final model fit failed: {type(exc).__name__}: {exc}") from exc

    report(0.84, "Sensitivity check without dominant sensor column(s) (cross-validation only)")
    ablation = None
    reduced = [f for f in features if f not in config.ABLATION_FEATURES]
    if reduced and len(reduced) < len(features):
        abl_pipe = build_pipeline(clone(candidates[best_name]), reduced)
        abl = cross_validate_candidates({best_name: abl_pipe}, X_train[reduced], y_train, n_splits)
        if abl.iloc[0]["status"] == "ok":
            ablation = {"dropped_features": config.ABLATION_FEATURES, "model": best_name,
                        **{k: float(abl.iloc[0][k]) for k in ("cv_mae_mean", "cv_rmse_mean", "cv_r2_mean")}}
            log.info("Without %s: CV MAE=%.3f RMSE=%.3f R2=%.3f", config.ABLATION_FEATURES,
                     ablation["cv_mae_mean"], ablation["cv_rmse_mean"], ablation["cv_r2_mean"])

    report(0.88, "Evaluating once on the held-out test set")
    pred = final.predict(X_test)
    test_metrics = regression_metrics(y_test, pred)
    baseline_metrics = regression_metrics(y_test, baseline.predict(X_test))
    beats = test_metrics["mae"] < baseline_metrics["mae"]
    if not beats:
        log.warning("Selected model does NOT beat the mean baseline on the test set.")
    save_json({"selected_model": best_name, "test": test_metrics, "baseline_test": baseline_metrics,
               "beats_baseline_mae": beats}, metrics_dir / "test_metrics.json")
    residual_frame(test_df[config.DATETIME_COLUMN], y_test, pred).to_csv(metrics_dir / "test_predictions.csv", index=False)
    imp = importance_table(final, X_test, y_test)
    imp.to_csv(metrics_dir / "feature_importance.csv", index=False)

    report(0.95, "Saving model")
    cv_row = cv[cv["model"] == best_name].iloc[0].to_dict()
    metadata = {
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "regression", "target": target, "features": features,
        "feature_descriptions": config.FEATURE_DESCRIPTIONS,
        "excluded_columns": config.EXCLUDED_COLUMNS,
        "feature_ranges": _feature_ranges(X_train),
        "target_range_train": {"min": float(y_train.min()), "max": float(y_train.max()),
                               "median": float(y_train.median()), "p99": float(y_train.quantile(.99))},
        "dataset": {"source_file": str(raw_path or config.RAW_DATA_PATH), "rows_raw": summary["rows_raw"],
                    "rows_clean": summary["rows_final"], "train_rows": len(train_df), "test_rows": len(test_df),
                    "train_period": [train_df.datetime.min().isoformat(), train_df.datetime.max().isoformat()],
                    "test_period": [test_df.datetime.min().isoformat(), test_df.datetime.max().isoformat()]},
        "selection_metric": f"cv_{config.SELECTION_METRIC}_mean", "cv_splits": n_splits,
        "cv_selected": {k: v for k, v in cv_row.items() if k.startswith("cv_")},
        "test_metrics": test_metrics, "baseline_test_metrics": baseline_metrics, "beats_baseline_mae": beats,
        "sensitivity_without_dominant_sensor": ablation,
        "data_caveat": ("In this file the target is an almost deterministic function of PT08.S2(NMHC), so scores "
                        "with that column are optimistic for real-world estimation. See the sensitivity check."),
        "hyperparameters": _simple_params(final.named_steps["model"]),
        "evaluation_methodology": METHODOLOGY, "quick_mode": quick, "random_state": config.RANDOM_STATE,
        "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                             "joblib": joblib.__version__},
    }
    save_artifacts(final, metadata, models_dir)
    report(1.0, "Done")
    log.info("TEST  MAE=%.3f  RMSE=%.3f  R2=%.3f | baseline MAE=%.3f", test_metrics["mae"], test_metrics["rmse"],
             test_metrics["r2"], baseline_metrics["mae"])
    return metadata


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the benzene (C6H6) regression pipeline.")
    ap.add_argument("--data", help="Path to the raw CSV (default: data/raw/training.csv)")
    ap.add_argument("--quick", action="store_true", help="Fewer trees/folds for smoke tests (not for reporting)")
    ap.add_argument("--no-eda", action="store_true", help="Skip EDA figure generation")
    args = ap.parse_args(argv)
    try:
        run_training(raw_path=args.data, quick=args.quick, make_eda=not args.no_eda)
    except MLProjectError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


