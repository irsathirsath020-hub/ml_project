"""Prediction helpers (used by the dashboard and the command line).

Usage:  python predict.py --input new_readings.csv --output predictions.csv
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data_preprocessing import sniff_and_read_text
from pipeline import load_metadata, load_pipeline
from utils import (DataLoadError, MLProjectError, ModelIncompatibleError,
                   SchemaError, get_logger)

log = get_logger(__name__)
PRED_COL = f"Predicted_{config.TARGET_COLUMN}"


def read_input_file(source, name: str | None = None) -> pd.DataFrame:
    """Read a CSV/TXT/XLSX from a path or an uploaded file-like object."""
    if hasattr(source, "read"):
        raw = source.read()
        name = name or getattr(source, "name", "upload.csv")
    else:
        p = Path(source)
        if not p.exists():
            raise DataLoadError(f"Input file not found: {p}")
        raw, name = p.read_bytes(), p.name
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not raw.strip():
        raise DataLoadError(f"{name} is empty.")
    suffix = Path(name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(io.BytesIO(raw))
        except ImportError as exc:
            raise DataLoadError("Reading Excel files needs 'openpyxl' (pip install openpyxl).") from exc
        except Exception as exc:  # corrupted workbook types vary by engine
            raise DataLoadError(f"Could not read Excel file {name}: {exc}") from exc
    if suffix not in config.SUPPORTED_SUFFIXES:
        raise DataLoadError(f"Unsupported file type '{suffix}'. Use .csv, .txt or .xlsx.")
    for enc in config.ENCODINGS:
        try:
            return sniff_and_read_text(raw.decode(enc), source=name)
        except UnicodeDecodeError:
            continue
    raise DataLoadError(f"Could not decode {name}.")


def prepare_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Validate columns/types and convert to the numeric schema used in training."""
    if df is None or df.empty:
        raise SchemaError("The input contains no rows.")
    df = df.rename(columns=lambda c: str(c).strip())
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required column(s): {missing}. Expected exactly these inputs: {features}")
    X = pd.DataFrame(index=df.index)
    for c in features:
        conv = pd.to_numeric(df[c], errors="coerce")
        bad = conv.isna() & df[c].notna()
        if bad.any():
            rows = [int(i) for i in df.index[bad][:5]]
            raise SchemaError(f"Column '{c}' has non-numeric values (e.g. at rows {rows}).")
        X[c] = conv
    X = X.mask(X == config.MISSING_SENTINEL)          # same missing-value convention as training
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise SchemaError("Input contains infinite values.")
    return X.astype(float)


def predict_dataframe(df: pd.DataFrame, pipeline=None, metadata: dict | None = None,
                      require_complete: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Return (input + prediction columns, list of human-readable warnings)."""
    pipeline = pipeline if pipeline is not None else load_pipeline()
    metadata = metadata if metadata is not None else load_metadata()
    features = metadata["features"]
    X = prepare_features(df, features)

    n_missing = X.isna().sum(axis=1)
    warnings: list[str] = []
    if require_complete and (n_missing > 0).any():
        raise SchemaError("All input values are required for a single prediction.")
    if (n_missing == len(features)).any():
        raise SchemaError("At least one row has no usable sensor values at all.")
    if (n_missing > 0).any():
        warnings.append(f"{int((n_missing > 0).sum())} row(s) have missing values; they were filled with training medians, "
                        "so those predictions are less reliable.")

    outside = pd.Series(False, index=X.index)
    for c in features:
        r = metadata["feature_ranges"][c]
        outside |= (X[c] < r["min"]) | (X[c] > r["max"])
    if outside.any():
        warnings.append(f"{int(outside.sum())} row(s) contain values outside the range seen in training; "
                        "the model is extrapolating there.")
    try:
        pred = pipeline.predict(X)
    except Exception as exc:  # sklearn raises assorted errors for incompatible pipelines
        raise ModelIncompatibleError(f"The saved model rejected the input: {type(exc).__name__}: {exc}") from exc
    if (pred < 0).any():
        warnings.append("Some predictions are negative (linear models can do this); benzene concentration cannot be negative.")

    out = df.copy()
    out[PRED_COL] = pred
    out["n_missing_features"] = n_missing.to_numpy()
    out["outside_training_range"] = outside.to_numpy()
    return out, warnings


def predict_single(values: dict, pipeline=None, metadata: dict | None = None) -> dict:
    metadata = metadata if metadata is not None else load_metadata()
    result, warnings = predict_dataframe(pd.DataFrame([values]), pipeline, metadata, require_complete=True)
    return {"prediction": float(result[PRED_COL].iloc[0]), "warnings": warnings,
            "outside_training_range": bool(result["outside_training_range"].iloc[0])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Predict benzene concentration for a file of sensor readings.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args(argv)
    try:
        result, warnings = predict_dataframe(read_input_file(args.input))
        result.to_csv(args.output, index=False)
    except MLProjectError as exc:
        log.error("%s", exc)
        return 1
    for w in warnings:
        log.warning(w)
    log.info("Wrote %d predictions to %s", len(result), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
