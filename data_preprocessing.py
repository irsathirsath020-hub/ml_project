"""Loading, validating, cleaning and chronologically splitting the dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from utils import (DataLoadError, InsufficientDataError, SchemaError,
                   get_logger)

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def sniff_and_read_text(text: str, source: str = "input") -> pd.DataFrame:
    """Parse CSV text; auto-detect ';' (decimal ',') versus ',' (decimal '.')."""
    header = text.splitlines()[0] if text.strip() else ""
    sep = ";" if header.count(";") > header.count(",") else ","
    decimal = "," if sep == ";" else "."
    from io import StringIO
    try:
        df = pd.read_csv(StringIO(text), sep=sep, decimal=decimal)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"{source} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"{source} is not a valid delimited file: {exc}") from exc
    return _tidy_raw(df, source)


def _tidy_raw(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # UCI's semicolon file carries trailing empty "Unnamed" columns/rows.
    empty_unnamed = [c for c in df.columns if c.startswith("Unnamed") and df[c].isna().all()]
    df = df.drop(columns=empty_unnamed).dropna(how="all").reset_index(drop=True)
    if df.empty or df.shape[1] == 0:
        raise DataLoadError(f"{source} contains no data rows.")
    return df


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw dataset. The file on disk is never modified."""
    path = Path(path) if path else config.RAW_DATA_PATH
    if not path.exists():
        raise DataLoadError(
            f"Dataset not found: {path}\n"
            "Place the CSV in data/raw/ (default name training.csv), edit RAW_DATA_PATH in "
            "config.py, or run: python train.py --data <path-to-file>")
    if not path.is_file():
        raise DataLoadError(f"{path} is not a file.")
    if path.suffix.lower() not in config.SUPPORTED_SUFFIXES:
        raise DataLoadError(f"Unsupported file type '{path.suffix}'. Use one of {sorted(config.SUPPORTED_SUFFIXES)}.")
    if path.stat().st_size == 0:
        raise DataLoadError(f"{path} is empty (0 bytes).")

    last_error: Exception | None = None
    for enc in config.ENCODINGS:
        try:
            text = path.read_text(encoding=enc)
            return sniff_and_read_text(text, source=str(path.name))
        except UnicodeDecodeError as exc:
            last_error = exc
        except PermissionError as exc:
            raise DataLoadError(f"Permission denied reading {path}.") from exc
    raise DataLoadError(f"Could not decode {path} with {config.ENCODINGS}: {last_error}")


# --------------------------------------------------------------------------
# Validation + cleaning
# --------------------------------------------------------------------------
def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Missing required column(s): {missing}.\nColumns found: {list(df.columns)}\n"
            "If your column names differ, update TARGET_COLUMN / FEATURE_COLUMNS in config.py.")


def clean_dataset(raw: pd.DataFrame, features: list[str] | None = None,
                  target: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (cleaned chronologically sorted frame, summary of every decision).

    Nothing learned from data happens here (no imputation/scaling), so there is
    no leakage; learned steps live in the sklearn Pipeline and see only training rows.
    """
    features = list(features or config.FEATURE_COLUMNS)
    target = target or config.TARGET_COLUMN
    validate_columns(raw, [target, *features, config.DATE_COLUMN, config.TIME_COLUMN])

    df = raw.copy()
    summary: dict = {"rows_raw": int(len(df)), "columns_raw": int(df.shape[1]), "removed": {}, "decisions": []}

    # 1) timestamps -> chronological order is required for leakage-free splitting
    stamp = (df[config.DATE_COLUMN].astype(str).str.strip() + " "
             + df[config.TIME_COLUMN].astype(str).str.strip().str.replace(".", ":", regex=False))
    ts = pd.to_datetime(stamp, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    n_bad_ts = int(ts.isna().sum())
    df[config.DATETIME_COLUMN] = ts
    df = df.loc[ts.notna()].copy()
    summary["removed"]["unparseable_timestamp"] = n_bad_ts
    summary["decisions"].append("Parsed Date (dd/mm/yyyy) + Time (HH.MM.SS) into a datetime; rows that fail are dropped.")

    # 2) numeric coercion for model columns
    coerced = {}
    for c in [target, *features]:
        before_na = df[c].isna()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        coerced[c] = int((df[c].isna() & ~before_na).sum())
    summary["non_numeric_coerced_to_nan"] = coerced

    # 3) -200 sentinel -> NaN for all numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    sentinel_counts = {c: int((df[c] == config.MISSING_SENTINEL).sum()) for c in numeric_cols}
    df[numeric_cols] = df[numeric_cols].mask(df[numeric_cols] == config.MISSING_SENTINEL)
    summary["sentinel_counts"] = sentinel_counts
    summary["decisions"].append(f"Replaced the {config.MISSING_SENTINEL} missing-value sentinel with NaN (it is not a real measurement).")

    # 4) physically impossible values -> NaN
    invalid = {}
    for col, (lo, hi) in config.PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= df[col] < lo
        if hi is not None:
            bad |= df[col] > hi
        invalid[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    summary["physically_invalid_set_nan"] = invalid

    # 5) rows with no target cannot be used for supervised learning
    n_before = len(df)
    df = df.loc[df[target].notna()].copy()
    summary["removed"]["missing_target"] = int(n_before - len(df))
    summary["decisions"].append("Dropped rows whose target is missing. Missing FEATURE values are kept and imputed inside the pipeline (training-set medians).")

    # 6) duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=[config.DATETIME_COLUMN], keep="first")
    summary["removed"]["duplicate_timestamp"] = int(n_before - len(df))

    # 7) chronological order, no shuffling anywhere
    df = df.sort_values(config.DATETIME_COLUMN, kind="stable").reset_index(drop=True)

    df = df.drop(columns=[config.DATE_COLUMN, config.TIME_COLUMN])
    ordered = [config.DATETIME_COLUMN] + [c for c in df.columns if c != config.DATETIME_COLUMN]
    df = df[ordered]

    summary["rows_final"] = int(len(df))
    summary["rows_removed_total"] = int(summary["rows_raw"] - len(df))
    summary["pct_removed"] = round(100 * summary["rows_removed_total"] / max(summary["rows_raw"], 1), 2)
    summary["remaining_missing_features"] = {c: int(df[c].isna().sum()) for c in features}

    if len(df) < config.MIN_ROWS_AFTER_CLEANING:
        raise InsufficientDataError(
            f"Only {len(df)} usable rows remain after cleaning (need >= {config.MIN_ROWS_AFTER_CLEANING}).")
    if df[target].nunique() < 2:
        raise InsufficientDataError(f"Target '{target}' is constant after cleaning; nothing to learn.")
    if summary["pct_removed"] > 10:
        log.warning("%.1f%% of rows were removed during cleaning - please review the cleaning summary.", summary["pct_removed"])

    # info only: temporal gaps and IQR outliers (outliers are NOT removed)
    gaps = df[config.DATETIME_COLUMN].diff().dropna()
    summary["time_range"] = [df[config.DATETIME_COLUMN].min().isoformat(), df[config.DATETIME_COLUMN].max().isoformat()]
    summary["gaps_over_1h"] = int((gaps > pd.Timedelta(hours=1)).sum())
    summary["largest_gap_hours"] = float(gaps.max() / pd.Timedelta(hours=1)) if len(gaps) else 0.0
    outliers = {}
    for c in [target, *features]:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers[c] = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    summary["iqr_outliers_kept"] = outliers
    summary["decisions"].append("IQR outliers are reported but NOT removed (extreme pollution episodes are legitimate).")
    summary["excluded_columns"] = config.EXCLUDED_COLUMNS
    return df, summary


# --------------------------------------------------------------------------
# Splitting, matrices, preprocessing
# --------------------------------------------------------------------------
def split_chronological(df: pd.DataFrame, test_fraction: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Past -> train, future -> test. No shuffling, so no look-ahead leakage."""
    test_fraction = config.TEST_FRACTION if test_fraction is None else test_fraction
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if not df[config.DATETIME_COLUMN].is_monotonic_increasing:
        df = df.sort_values(config.DATETIME_COLUMN, kind="stable")
    n = len(df)
    n_test = int(round(n * test_fraction))
    n_train = n - n_test
    if n_test < 10 or n_train < 50:
        raise InsufficientDataError(f"Cannot split {n} rows reliably (train={n_train}, test={n_test}).")
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def get_xy(df: pd.DataFrame, features: list[str] | None = None, target: str | None = None):
    features = list(features or config.FEATURE_COLUMNS)
    target = target or config.TARGET_COLUMN
    validate_columns(df, [target, *features])
    return df[features].astype(float), df[target].astype(float)


def build_preprocessor(features: list[str] | None = None) -> ColumnTransformer:
    """Median imputation + standardisation. Fitted only on training data inside the Pipeline."""
    features = list(features or config.FEATURE_COLUMNS)
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([("num", numeric, features)], remainder="drop", verbose_feature_names_out=False)
