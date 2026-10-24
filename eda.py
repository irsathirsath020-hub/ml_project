"""Exploratory plots saved as PNG files (matplotlib only)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from utils import ensure_dirs, get_logger

log = get_logger(__name__)


def generate_eda(df: pd.DataFrame, summary: dict, figures_dir: Path | None = None,
                 metrics_dir: Path | None = None, features: list[str] | None = None, target: str | None = None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir) if figures_dir else config.FIGURES_DIR
    metrics_dir = Path(metrics_dir) if metrics_dir else config.METRICS_DIR
    ensure_dirs(figures_dir, metrics_dir)
    features = list(features or config.FEATURE_COLUMNS)
    target = target or config.TARGET_COLUMN
    saved: list[Path] = []

    # 1) target through time
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(df[config.DATETIME_COLUMN], df[target], lw=0.6, color="#22d3ee")
    ax.set(title=f"{target} over time", ylabel="µg/m³")
    p = figures_dir / "target_timeseries.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); saved.append(p)

    # 2) distributions
    cols = [target, *features]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, c in zip(axes.ravel(), cols):
        ax.hist(df[c].dropna(), bins=40, color="#3b82f6")
        ax.set_title(c, fontsize=9)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    p = figures_dir / "distributions.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); saved.append(p)

    # 3) correlation heatmap
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    fig.colorbar(im, ax=ax); ax.set_title("Correlation (training portion)")
    p = figures_dir / "correlation.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); saved.append(p)

    # 4) missing-value sentinel counts
    counts = pd.Series(summary.get("sentinel_counts", {}))
    counts = counts[counts > 0].sort_values()
    if not counts.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(counts.index, counts.values, color="#f59e0b")
        ax.set_title(f"Readings flagged {config.MISSING_SENTINEL} (missing) in raw data")
        p = figures_dir / "missing_sentinels.png"
        fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); saved.append(p)

    df[cols].describe().T.to_csv(metrics_dir / "summary_statistics.csv")
    log.info("Saved %d EDA figures to %s", len(saved), figures_dir)
    return saved
