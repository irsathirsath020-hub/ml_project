"""Streamlit dashboard. Run with:  python -m streamlit run app.py

The dashboard never trains on open. It loads the saved pipeline created by `python train.py`
(or by the Train button on the evaluation page) and shows only measured numbers.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from data_preprocessing import clean_dataset, load_raw_data
from pipeline import load_metadata, load_pipeline, model_exists
from predict import PRED_COL, predict_dataframe, predict_single, read_input_file
from train import run_training
from utils import MLProjectError

st.set_page_config(page_title="Benzene Estimator", page_icon="🌫️", layout="wide")

CYAN, BLUE, AMBER, RED = "#22d3ee", "#3b82f6", "#f59e0b", "#f43f5e"

CSS = """
<style>
.stApp {background: radial-gradient(1200px 600px at 10% -10%, #0b2447 0%, #050b18 55%, #02050d 100%);}
[data-testid="stSidebar"] {background: rgba(8,16,34,.85); border-right: 1px solid rgba(34,211,238,.15);}
h1,h2,h3 {letter-spacing:.2px;}
.hero {padding:18px 22px; border-radius:18px; border:1px solid rgba(34,211,238,.25);
  background: linear-gradient(135deg, rgba(34,211,238,.10), rgba(59,130,246,.08));
  backdrop-filter: blur(10px); margin-bottom:14px;}
.hero h1 {margin:0; font-size:1.7rem; background:linear-gradient(90deg,#22d3ee,#60a5fa);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.hero p {margin:4px 0 0 0; color:#9fb3d1;}
.card {padding:14px 16px; border-radius:16px; background: rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.08); backdrop-filter: blur(8px);
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease; height:100%;}
.card:hover {transform: translateY(-2px); border-color: rgba(34,211,238,.55);
  box-shadow: 0 6px 22px rgba(34,211,238,.12);}
.card .lab {font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; color:#8aa0c2;}
.card .val {font-size:1.55rem; font-weight:700; color:#e6f1ff; margin-top:2px;}
.card .sub {font-size:.78rem; color:#7f93b3; margin-top:2px;}
.pill {display:inline-block; padding:4px 12px; border-radius:999px; font-size:.78rem; font-weight:600;}
.pill.ok {background:rgba(34,211,238,.14); color:#67e8f9; border:1px solid rgba(34,211,238,.4);}
.pill.warn {background:rgba(245,158,11,.14); color:#fbbf24; border:1px solid rgba(245,158,11,.4);}
.note {padding:12px 14px; border-radius:12px; border-left:3px solid #f59e0b;
  background:rgba(245,158,11,.08); color:#f5d68a; font-size:.9rem;}
@media (max-width: 700px) {.card .val {font-size:1.25rem;} .hero h1 {font-size:1.3rem;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
def card(label: str, value: str, sub: str = "") -> str:
    return f'<div class="card"><div class="lab">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'


def hero(title: str, text: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{text}</p></div>', unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=50, b=10),
                      colorway=[CYAN, BLUE, AMBER, RED, "#a78bfa"])
    return fig


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def f2(x, d=3) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


@st.cache_data(show_spinner=False)
def cached_dataset(path_str: str, stamp: float):
    raw = load_raw_data(path_str)
    clean, summary = clean_dataset(raw)
    return raw, clean, summary


@st.cache_data(show_spinner=False)
def cached_metadata(stamp: float):
    return load_metadata()


@st.cache_resource(show_spinner=False)
def cached_pipeline(stamp: float):
    return load_pipeline()


@st.cache_data(show_spinner=False)
def cached_csv(path_str: str, stamp: float, parse_dates: tuple = ()):
    return pd.read_csv(path_str, parse_dates=list(parse_dates))


def get_dataset():
    try:
        return cached_dataset(str(config.RAW_DATA_PATH), mtime(config.RAW_DATA_PATH))
    except MLProjectError as exc:
        st.error(str(exc))
        return None


def get_model():
    """Return (pipeline, metadata) or (None, None) with a visible message."""
    if not model_exists():
        return None, None
    try:
        stamp = mtime(config.MODELS_DIR / config.MODEL_FILENAME)
        return cached_pipeline(stamp), cached_metadata(mtime(config.MODELS_DIR / config.METADATA_FILENAME))
    except MLProjectError as exc:
        st.error(str(exc))
        return None, None


def no_model_notice() -> None:
    st.markdown('<div class="note"><b>No trained model found.</b> Nothing is predicted or scored until a real model exists. '
                'Run <code>python train.py</code> in the project folder, or open <b>Model Evaluation</b> and press '
                '<b>Run training</b>.</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- sidebar
PAGES = ["Overview", "Data Explorer", "Model Evaluation", "Prediction Studio", "Model Information"]
pipe, meta = get_model()
with st.sidebar:
    st.markdown("### 🌫️ Benzene Estimator")
    st.caption("Low-cost sensors → C6H6 (µg/m³)")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    if meta:
        st.markdown('<span class="pill ok">● TRAINED MODEL ACTIVE</span>', unsafe_allow_html=True)
        st.caption(f"{meta['model_name']} · trained {meta['trained_at'][:16].replace('T', ' ')} UTC")
    else:
        st.markdown('<span class="pill warn">● NO TRAINED MODEL</span>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- pages
def page_overview() -> None:
    hero("Air Quality · Benzene Estimation", "Regression from metal-oxide sensors, temperature and humidity — chronological, leakage-aware evaluation.")
    if meta:
        st.markdown('<span class="pill ok">● Using the trained pipeline from disk (not demo data)</span>', unsafe_allow_html=True)
    else:
        no_model_notice()
    data = get_dataset()
    if data is None:
        return
    raw, clean, summary = data
    c = st.columns(4)
    c[0].markdown(card("Raw rows", f"{summary['rows_raw']:,}", f"{summary['columns_raw']} columns"), unsafe_allow_html=True)
    c[1].markdown(card("Usable rows", f"{summary['rows_final']:,}", f"{summary['pct_removed']}% removed (missing target)"), unsafe_allow_html=True)
    c[2].markdown(card("Features", str(len(config.FEATURE_COLUMNS)), "sensors + T + RH"), unsafe_allow_html=True)
    c[3].markdown(card("Model", meta["model_name"] if meta else "—", "selected by time-series CV" if meta else "not trained"), unsafe_allow_html=True)
    if meta:
        st.write("")
        t, b = meta["test_metrics"], meta["baseline_test_metrics"]
        m = st.columns(4)
        m[0].markdown(card("Test MAE", f2(t["mae"]), f"mean-baseline {f2(b['mae'])} µg/m³"), unsafe_allow_html=True)
        m[1].markdown(card("Test RMSE", f2(t["rmse"]), f"mean-baseline {f2(b['rmse'])} µg/m³"), unsafe_allow_html=True)
        m[2].markdown(card("Test R²", f2(t["r2"], 4), "held-out final 20% of timeline"), unsafe_allow_html=True)
        s = meta.get("sensitivity_without_dominant_sensor")
        m[3].markdown(card("CV MAE w/o S2", f2(s["cv_mae_mean"]) if s else "n/a", "sensitivity check"), unsafe_allow_html=True)
        st.write("")
        st.markdown(f'<div class="note"><b>Read this before trusting the score:</b> {meta.get("data_caveat", "")}</div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Missing readings (−200 sentinel)")
        s = pd.Series(summary["sentinel_counts"]).rename("count").reset_index()
        s.columns = ["column", "count"]
        s = s[s["count"] > 0].sort_values("count")
        st.plotly_chart(style(px.bar(s, x="count", y="column", orientation="h", title="Rows flagged missing in raw file"), 360))
    with right:
        st.subheader("Cleaning decisions")
        for d in summary["decisions"]:
            st.markdown(f"- {d}")
        st.caption(f"Time range {summary['time_range'][0][:10]} → {summary['time_range'][1][:10]} · "
                   f"{summary['gaps_over_1h']} gaps > 1 h (largest {summary['largest_gap_hours']:.0f} h)")
    st.subheader("Dataset preview (cleaned)")
    st.dataframe(clean.head(100))


def page_explorer() -> None:
    hero("Data Explorer", "Descriptive views of the cleaned data. Nothing here is used to fit the model.")
    data = get_dataset()
    if data is None:
        return
    raw, clean, summary = data
    cols = [config.TARGET_COLUMN, *config.FEATURE_COLUMNS]
    t1, t2, t3, t4, t5 = st.tabs(["Columns & statistics", "Missing values", "Distributions", "Time series", "Correlation"])
    with t1:
        st.dataframe(pd.DataFrame({"dtype": clean.dtypes.astype(str), "missing": clean.isna().sum(),
                                   "unique": clean.nunique()}))
        st.dataframe(clean[cols].describe().T)
    with t2:
        miss = pd.DataFrame({"column": list(summary["sentinel_counts"]), "raw −200 count": list(summary["sentinel_counts"].values())})
        miss["% of raw rows"] = (100 * miss["raw −200 count"] / summary["rows_raw"]).round(2)
        st.plotly_chart(style(px.bar(miss.sort_values("% of raw rows"), x="% of raw rows", y="column", orientation="h",
                                     title="Share of raw rows flagged missing")))
        st.caption("NMHC(GT) is ~90% missing and is excluded. Rows with a missing benzene value are dropped; they coincide with sensor outages.")
        st.dataframe(pd.DataFrame({"IQR outliers (kept)": summary["iqr_outliers_kept"]}))
    with t3:
        col = st.selectbox("Variable", cols, key="dist_col")
        a, b = st.columns(2)
        a.plotly_chart(style(px.histogram(clean, x=col, nbins=50, title=f"Distribution of {col}")))
        b.plotly_chart(style(px.box(clean, y=col, title=f"Box plot of {col}")))
    with t4:
        col = st.selectbox("Variable", cols, key="ts_col")
        roll = st.slider("Rolling mean window (hours)", 1, 168, 24)
        d = clean[[config.DATETIME_COLUMN, col]].copy()
        d["rolling"] = d[col].rolling(roll, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d[config.DATETIME_COLUMN], y=d[col], mode="lines", name="hourly", line=dict(width=1, color=BLUE), opacity=.5))
        fig.add_trace(go.Scatter(x=d[config.DATETIME_COLUMN], y=d["rolling"], mode="lines", name=f"{roll} h mean", line=dict(color=CYAN)))
        split_at = clean[config.DATETIME_COLUMN].iloc[int(len(clean) * (1 - config.TEST_FRACTION))]
        fig.add_vline(x=split_at.to_pydatetime(), line_dash="dash", line_color=AMBER)
        fig.add_annotation(x=split_at.to_pydatetime(), y=1, yref="paper", text="train | test", showarrow=False, font=dict(color=AMBER))
        st.plotly_chart(style(fig.update_layout(title=f"{col} over time"), 420))
    with t5:
        corr = clean[cols].corr()
        st.plotly_chart(style(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                                        title="Correlation matrix"), 520))
        feat = st.selectbox("Scatter vs target", config.FEATURE_COLUMNS)
        st.plotly_chart(style(px.scatter(clean, x=feat, y=config.TARGET_COLUMN, opacity=.5, title=f"{config.TARGET_COLUMN} vs {feat}")))


def page_evaluation() -> None:
    hero("Model Training & Evaluation", "Time-aware cross-validation for model choice; a single final evaluation on the later 20% of the timeline.")
    if st.session_state.get("train_msg"):
        st.success(st.session_state.pop("train_msg"))
    with st.expander("Train / retrain the model", expanded=meta is None):
        quick = st.checkbox("Quick mode (30 trees, 3 CV folds) — smoke test only, do not report these numbers", value=False)
        if st.button("Run training", type="primary"):
            bar = st.progress(0.0, text="Starting…")
            try:
                new_meta = run_training(quick=quick, progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m))
            except MLProjectError as exc:
                bar.empty()
                st.error(f"Training failed: {exc}")
            else:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.session_state["train_msg"] = f"Training finished. Selected {new_meta['model_name']}; test MAE {new_meta['test_metrics']['mae']:.3f} µg/m³."
                st.rerun()
    if not meta:
        no_model_notice()
        return
    metrics_dir = config.METRICS_DIR
    stamp = mtime(config.MODELS_DIR / config.METADATA_FILENAME)

    st.subheader("Model comparison (cross-validation on the training period)")
    cv = cached_csv(str(metrics_dir / "cv_results.csv"), stamp)
    show = cv.drop(columns=[c for c in ("error",) if c in cv.columns])
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if c.startswith("cv_")}))
    ok = cv[cv["status"] == "ok"]
    fig = go.Figure(go.Bar(x=ok["model"], y=ok["cv_mae_mean"], error_y=dict(type="data", array=ok["cv_mae_std"]), marker_color=CYAN))
    st.plotly_chart(style(fig.update_layout(title="Cross-validated MAE (µg/m³) — lower is better", yaxis_title="MAE"), 340))

    st.subheader("Held-out test performance")
    t, b = meta["test_metrics"], meta["baseline_test_metrics"]
    st.dataframe(pd.DataFrame({"MAE": [t["mae"], b["mae"]], "RMSE": [t["rmse"], b["rmse"]], "R²": [t["r2"], b["r2"]]},
                              index=[meta["model_name"], config.BASELINE_NAME]).style.format("{:.4f}"))
    s = meta.get("sensitivity_without_dominant_sensor")
    if s:
        st.markdown(f'<div class="note"><b>Sensitivity check:</b> {meta.get("data_caveat", "")} Cross-validated {s["model"]} '
                    f'without {", ".join(s["dropped_features"])}: MAE {s["cv_mae_mean"]:.3f}, RMSE {s["cv_rmse_mean"]:.3f}, '
                    f'R² {s["cv_r2_mean"]:.3f} (training period only).</div>', unsafe_allow_html=True)

    preds = cached_csv(str(metrics_dir / "test_predictions.csv"), stamp, ("datetime",))
    a, c = st.columns(2)
    lim = [min(preds.actual.min(), preds.predicted.min()), max(preds.actual.max(), preds.predicted.max())]
    f1 = px.scatter(preds, x="actual", y="predicted", opacity=.5, title="Actual vs predicted (test)")
    f1.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="perfect", line=dict(color=AMBER, dash="dash")))
    a.plotly_chart(style(f1))
    c.plotly_chart(style(px.scatter(preds, x="predicted", y="residual", opacity=.5, title="Residuals vs predicted")))
    a2, c2 = st.columns(2)
    a2.plotly_chart(style(px.histogram(preds, x="residual", nbins=60, title="Residual distribution")))
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=preds.datetime, y=preds.actual, name="actual", line=dict(color=BLUE, width=1)))
    f3.add_trace(go.Scatter(x=preds.datetime, y=preds.predicted, name="predicted", line=dict(color=CYAN, width=1)))
    c2.plotly_chart(style(f3.update_layout(title="Test period over time")))

    imp = cached_csv(str(metrics_dir / "feature_importance.csv"), stamp)
    st.plotly_chart(style(px.bar(imp.sort_values("importance_mae_increase"), x="importance_mae_increase", y="feature",
                                 orientation="h", error_x="std", title="Permutation importance (MAE increase when shuffled, test set)")))


def page_prediction() -> None:
    hero("Prediction Studio", "Enter hourly sensor readings or upload a file. Predictions come from the saved pipeline.")
    if not meta:
        no_model_notice()
        return
    features = meta["features"]
    tab1, tab2 = st.tabs(["Single reading", "Batch file"])
    with tab1:
        with st.form("single"):
            cols = st.columns(3)
            values = {}
            for i, f in enumerate(features):
                r = meta["feature_ranges"][f]
                step = max((r["max"] - r["min"]) / 200, 0.01)
                values[f] = cols[i % 3].number_input(f, value=float(round(r["median"], 2)), step=float(round(step, 2)),
                                                     help=f"{meta['feature_descriptions'].get(f, '')}. Training range {r['min']:.1f}–{r['max']:.1f}")
            go_btn = st.form_submit_button("Predict benzene", type="primary")
        if go_btn:
            try:
                res = predict_single(values, pipe, meta)
            except MLProjectError as exc:
                st.error(str(exc))
            else:
                pred = res["prediction"]
                for w in res["warnings"]:
                    st.warning(w)
                k1, k2 = st.columns([1, 2])
                k1.markdown(card("Estimated C6H6", f"{pred:.2f} µg/m³", f"expected typical error ≈ ±{meta['test_metrics']['mae']:.2f} (test MAE)"), unsafe_allow_html=True)
                gmax = max(meta["target_range_train"]["p99"] * 1.2, pred * 1.2, 1)
                gauge = go.Figure(go.Indicator(mode="gauge+number", value=pred, number=dict(suffix=" µg/m³"),
                                               gauge=dict(axis=dict(range=[0, gmax]), bar=dict(color=CYAN))))
                k2.plotly_chart(style(gauge, 260))
                data = get_dataset()
                if data is not None:
                    hist = px.histogram(data[1], x=config.TARGET_COLUMN, nbins=60, title="Where this estimate sits among historical hourly values")
                    hist.add_vline(x=pred, line_color=AMBER, line_dash="dash")
                    st.plotly_chart(style(hist, 320))
                out = pd.DataFrame([{**values, PRED_COL: pred}])
                st.download_button("Download result (CSV)", out.to_csv(index=False).encode("utf-8"), "prediction.csv", "text/csv")
    with tab2:
        st.caption(f"Required columns: {', '.join(features)}. CSV/TXT (comma or semicolon) or XLSX. Missing values may be left empty or as −200.")
        st.download_button("Download blank template", pd.DataFrame(columns=features).to_csv(index=False).encode("utf-8"),
                           "template.csv", "text/csv")
        up = st.file_uploader("Upload sensor readings", type=["csv", "txt", "xlsx"])
        if up is not None:
            try:
                df = read_input_file(up, up.name)
                result, warns = predict_dataframe(df, pipe, meta)
            except MLProjectError as exc:
                st.error(str(exc))
            else:
                for w in warns:
                    st.warning(w)
                st.success(f"Predicted {len(result):,} rows.")
                st.dataframe(result.head(200))
                st.plotly_chart(style(px.histogram(result, x=PRED_COL, nbins=50, title="Distribution of predictions"), 320))
                st.download_button("Download predictions (CSV)", result.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")


def page_info() -> None:
    hero("Model Information", "What was trained, on what data, and how it was evaluated.")
    if not meta:
        no_model_notice()
        return
    c = st.columns(3)
    c[0].markdown(card("Model", meta["model_name"], f"trained {meta['trained_at'][:16].replace('T', ' ')} UTC"), unsafe_allow_html=True)
    c[1].markdown(card("Train rows", f"{meta['dataset']['train_rows']:,}", f"{meta['dataset']['train_period'][0][:10]} → {meta['dataset']['train_period'][1][:10]}"), unsafe_allow_html=True)
    c[2].markdown(card("Test rows", f"{meta['dataset']['test_rows']:,}", f"{meta['dataset']['test_period'][0][:10]} → {meta['dataset']['test_period'][1][:10]}"), unsafe_allow_html=True)
    st.subheader("Evaluation methodology")
    st.write(meta["evaluation_methodology"])
    st.subheader("Hyperparameters")
    st.json(meta["hyperparameters"])
    st.subheader("Features")
    st.dataframe(pd.DataFrame({"feature": meta["features"], "description": [meta["feature_descriptions"].get(f, "") for f in meta["features"]]}))
    st.subheader("Columns intentionally not used")
    st.dataframe(pd.DataFrame({"column": list(meta["excluded_columns"]), "reason": list(meta["excluded_columns"].values())}))
    st.subheader("Important limitations")
    st.markdown(
        f"- {meta.get('data_caveat', '')}\n"
        "- One monitoring site and about one year (Mar 2004 – Apr 2005); metal-oxide sensors drift and respond to season, so performance on other sites or later years is unknown.\n"
        "- The test period is later in time but still the same site and instruments.\n"
        "- Inputs outside the training range trigger a warning because tree models cannot extrapolate.\n"
        "- Missing sensor values are filled with training medians, which lowers reliability.\n"
        "- The model is trained on the earlier 80% only; it is not refitted on the test period.")
    st.subheader("Library versions")
    st.json(meta["library_versions"])


{"Overview": page_overview, "Data Explorer": page_explorer, "Model Evaluation": page_evaluation,
 "Prediction Studio": page_prediction, "Model Information": page_info}[page]()
