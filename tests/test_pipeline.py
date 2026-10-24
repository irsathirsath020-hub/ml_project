"""Automated tests. Run from the project root:  python -m pytest -v"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from data_preprocessing import (build_preprocessor, clean_dataset, get_xy,  # noqa: E402
                                load_raw_data, split_chronological)
from pipeline import build_pipeline, load_pipeline, model_exists, save_artifacts  # noqa: E402
from predict import predict_dataframe, predict_single, read_input_file  # noqa: E402
from train import run_training  # noqa: E402
from utils import (DataLoadError, InsufficientDataError, ModelNotFoundError,  # noqa: E402
                   SchemaError)
from sklearn.ensemble import RandomForestRegressor  # noqa: E402


def make_raw(n=600, seed=0) -> pd.DataFrame:
    """Synthetic frame in the UCI layout (-200 sentinel, dd/mm/yyyy, HH.MM.SS)."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2004-03-10 18:00", periods=n, freq="h")
    s2 = rng.uniform(500, 1500, n)
    df = pd.DataFrame({
        "Date": t.strftime("%d/%m/%Y"), "Time": t.strftime("%H.%M.%S"),
        "CO(GT)": rng.uniform(0, 5, n), "PT08.S1(CO)": rng.uniform(900, 1500, n),
        "NMHC(GT)": -200.0, "C6H6(GT)": 0.02 * s2 - 5 + rng.normal(0, 0.3, n),
        "PT08.S2(NMHC)": s2, "NOx(GT)": rng.uniform(0, 400, n), "PT08.S3(NOx)": rng.uniform(600, 1300, n),
        "NO2(GT)": rng.uniform(10, 200, n), "PT08.S4(NO2)": rng.uniform(1000, 2000, n),
        "PT08.S5(O3)": rng.uniform(500, 1800, n), "T": rng.uniform(0, 35, n), "RH": rng.uniform(10, 90, n),
        "AH": rng.uniform(0.3, 2, n)})
    df.loc[10:14, df.columns[2:]] = -200.0          # whole-row outage, like the real data
    df.loc[30, "PT08.S1(CO)"] = -200.0              # feature-only gap
    return df


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    csv = d / "raw.csv"
    make_raw().to_csv(csv, index=False)
    meta = run_training(raw_path=csv, models_dir=d / "models", reports_dir=d / "reports",
                        processed_dir=d / "processed", quick=True, make_eda=True)
    return d, meta


# ---- loading ------------------------------------------------------------
def test_load_comma_file(tmp_path):
    p = tmp_path / "a.csv"
    make_raw(50).to_csv(p, index=False)
    assert load_raw_data(p).shape[0] == 50


def test_load_semicolon_decimal_comma(tmp_path):
    p = tmp_path / "b.csv"
    make_raw(50).to_csv(p, index=False, sep=";", decimal=",")
    df = load_raw_data(p)
    assert pd.api.types.is_float_dtype(df["T"]) and df.shape[0] == 50


def test_load_errors(tmp_path):
    with pytest.raises(DataLoadError):
        load_raw_data(tmp_path / "missing.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(DataLoadError):
        load_raw_data(empty)
    bad = tmp_path / "x.json"
    bad.write_text("{}")
    with pytest.raises(DataLoadError):
        load_raw_data(bad)


def test_real_dataset_loads_if_present():
    if not config.RAW_DATA_PATH.exists():
        pytest.skip("raw dataset not present")
    df = load_raw_data()
    assert config.TARGET_COLUMN in df.columns and len(df) > 1000


# ---- cleaning -----------------------------------------------------------
def test_sentinel_and_missing_target_handling():
    clean, s = clean_dataset(make_raw())
    assert (clean[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]] == config.MISSING_SENTINEL).sum().sum() == 0
    assert s["removed"]["missing_target"] == 5
    assert clean[config.TARGET_COLUMN].notna().all()
    assert clean["PT08.S1(CO)"].isna().sum() == 1          # kept, to be imputed in the pipeline
    assert clean[config.DATETIME_COLUMN].is_monotonic_increasing


def test_missing_target_column_and_too_few_rows():
    with pytest.raises(SchemaError):
        clean_dataset(make_raw().drop(columns=["C6H6(GT)"]))
    with pytest.raises(InsufficientDataError):
        clean_dataset(make_raw(50))


def test_duplicate_timestamps_removed():
    raw = make_raw()
    dup = pd.concat([raw, raw.iloc[100:103]], ignore_index=True)
    clean, s = clean_dataset(dup)
    assert s["removed"]["duplicate_timestamp"] == 3


# ---- splitting / leakage -------------------------------------------------
def test_chronological_split_has_no_lookahead():
    clean, _ = clean_dataset(make_raw())
    tr, te = split_chronological(clean)
    assert tr[config.DATETIME_COLUMN].max() < te[config.DATETIME_COLUMN].min()
    assert len(tr) + len(te) == len(clean)


def test_imputer_uses_training_statistics_only():
    X = pd.DataFrame({c: [1.0, 2.0, 3.0, np.nan] for c in config.FEATURE_COLUMNS})
    pre = build_preprocessor().fit(X.iloc[:3])
    out = pre.transform(pd.DataFrame({c: [np.nan] for c in config.FEATURE_COLUMNS}))
    assert np.isfinite(out).all()                     # NaN filled with the training median (2.0)
    assert np.allclose(out, 0.0)                      # median 2.0 == training mean -> standardised to 0


# ---- training / persistence / prediction --------------------------------
def test_training_creates_artifacts(trained):
    d, meta = trained
    assert model_exists(d / "models")
    assert (d / "reports" / "metrics" / "cv_results.csv").exists()
    assert meta["features"] == config.FEATURE_COLUMNS
    assert meta["model_name"] != config.BASELINE_NAME
    assert meta["test_metrics"]["mae"] < meta["baseline_test_metrics"]["mae"]


def test_save_and_load_roundtrip(trained, tmp_path):
    d, meta = trained
    pipe = load_pipeline(d / "models")
    save_artifacts(pipe, meta, tmp_path)
    reloaded = load_pipeline(tmp_path)
    X = pd.DataFrame([{c: meta["feature_ranges"][c]["median"] for c in meta["features"]}])
    assert np.allclose(pipe.predict(X), reloaded.predict(X))


def test_feature_columns_consistent(trained):
    d, meta = trained
    pipe = load_pipeline(d / "models")
    assert list(pipe.named_steps["preprocess"].transformers_[0][2]) == meta["features"]


def test_prediction_shape_and_flags(trained):
    d, meta = trained
    pipe = load_pipeline(d / "models")
    clean, _ = clean_dataset(make_raw())
    X, _ = get_xy(clean.head(20))
    X.iloc[0, 0] = np.nan
    res, warns = predict_dataframe(X, pipe, meta)
    assert len(res) == 20 and res["n_missing_features"].iloc[0] == 1
    assert any("missing" in w for w in warns)


def test_invalid_inputs_rejected(trained):
    d, meta = trained
    pipe = load_pipeline(d / "models")
    good = {c: meta["feature_ranges"][c]["median"] for c in meta["features"]}
    with pytest.raises(SchemaError):
        predict_dataframe(pd.DataFrame([{k: v for k, v in good.items() if k != "T"}]), pipe, meta)
    with pytest.raises(SchemaError):
        predict_dataframe(pd.DataFrame([{**good, "T": "hot"}]), pipe, meta)
    with pytest.raises(SchemaError):
        predict_single({**good, "RH": None}, pipe, meta)
    with pytest.raises(SchemaError):
        predict_dataframe(pd.DataFrame(), pipe, meta)


def test_out_of_range_is_flagged(trained):
    d, meta = trained
    pipe = load_pipeline(d / "models")
    vals = {c: meta["feature_ranges"][c]["median"] for c in meta["features"]}
    vals["T"] = 500.0
    out = predict_single(vals, pipe, meta)
    assert out["outside_training_range"] and isinstance(out["prediction"], float)


def test_no_model_message(tmp_path):
    assert not model_exists(tmp_path)
    with pytest.raises(ModelNotFoundError):
        load_pipeline(tmp_path)


def test_read_input_file_variants(tmp_path):
    p = tmp_path / "in.csv"
    pd.DataFrame({"T": [1.0], "RH": [50.0]}).to_csv(p, index=False)
    assert read_input_file(p).shape == (1, 2)
    with pytest.raises(DataLoadError):
        read_input_file(tmp_path / "nope.csv")


def test_pipeline_factory_keeps_preprocessing_inside():
    pipe = build_pipeline(RandomForestRegressor(n_estimators=5, random_state=0))
    assert [n for n, _ in pipe.steps] == ["preprocess", "model"]
