"""Save EDA charts from the cleaned synthetic employee-cycle dataset.

Run from the project root:

    python -m src.eda_visualizations

Charts are descriptive only. They are not causal evidence and do not use
employee_id as a plotting signal.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.eda import BAND_ORDER, CLEAN_PATH, NUMERIC_FEATURE_COLUMNS
from src.generate_data import JOB_LEVELS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

PALETTE = {
    "Low": "#c44e52",
    "Medium": "#4c72b0",
    "High": "#55a868",
}

DIST_COLUMNS = [
    "on_time_delivery_rate",
    "avg_task_delay_days",
    "bug_count",
    "code_review_score",
    "story_points_completed",
    "billable_hours_ratio",
    "training_hours",
    "peer_feedback_score",
    "previous_manager_score",
    "kudos_count",
    "sick_days",
    "avg_login_hours",
]

SIGNAL_COLUMNS = [
    "on_time_delivery_rate",
    "avg_task_delay_days",
    "bug_count",
    "training_hours",
    "peer_feedback_score",
    "previous_manager_score",
]


def _prepare_frame(path: Path = CLEAN_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {path}. Run `python -m src.eda` first."
        )
    frame = pd.read_csv(path, dtype={"employee_id": "string"})
    frame["perf_band_next"] = pd.Categorical(
        frame["perf_band_next"], categories=BAND_ORDER, ordered=True
    )
    return frame


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def plot_target_distribution(frame: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "target_distribution.png"
    counts = frame["perf_band_next"].value_counts().reindex(BAND_ORDER)
    perc = (counts / counts.sum() * 100).round(2)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.barplot(
        x=[str(x) for x in counts.index],
        y=counts.values,
        hue=[str(x) for x in counts.index],
        palette=PALETTE,
        dodge=False,
        ax=ax,
        legend=False,
    )
    ax.set_title("Next-cycle performance band (synthetic)")
    ax.set_xlabel("perf_band_next")
    ax.set_ylabel("Row count")
    for i, (n, p) in enumerate(zip(counts.values, perc.values)):
        ax.text(i, n + 8, f"{int(n)}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(counts.values) * 1.18)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_numeric_distributions(frame: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "numeric_feature_distributions.png"
    cols = [c for c in DIST_COLUMNS if c in frame.columns]
    n = len(cols)
    nrows = 3
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 9))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(frame[col], bins=24, kde=True, color="#4c72b0", ax=axes[i])
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("count", fontsize=8)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Numeric feature distributions (synthetic cycles)", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _stacked_percent(frame: pd.DataFrame, group_col: str, order: list[str] | None = None) -> pd.DataFrame:
    counts = pd.crosstab(frame[group_col], frame["perf_band_next"])
    for band in BAND_ORDER:
        if band not in counts.columns:
            counts[band] = 0
    counts = counts[BAND_ORDER]
    if order is not None:
        counts = counts.reindex([x for x in order if x in counts.index])
    return counts.div(counts.sum(axis=1), axis=0) * 100


def plot_performance_by_department(frame: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "performance_by_department.png"
    pct = _stacked_percent(frame, "department")
    pct = pct.loc[pct.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pct.plot(
        kind="bar",
        stacked=True,
        color=[PALETTE[b] for b in BAND_ORDER],
        ax=ax,
        width=0.82,
    )
    ax.set_title("Performance-band mix by department (row %, synthetic)")
    ax.set_xlabel("department")
    ax.set_ylabel("Percent of department rows")
    ax.legend(title="perf_band_next", loc="upper right")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=35)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_performance_by_job_level(frame: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "performance_by_job_level.png"
    pct = _stacked_percent(frame, "job_level", order=list(JOB_LEVELS))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    pct.plot(
        kind="bar",
        stacked=True,
        color=[PALETTE[b] for b in BAND_ORDER],
        ax=ax,
        width=0.8,
    )
    ax.set_title("Performance-band mix by job level (row %, synthetic)")
    ax.set_xlabel("job_level")
    ax.set_ylabel("Percent of job-level rows")
    ax.legend(title="perf_band_next")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_correlation_heatmap(frame: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "feature_correlation_heatmap.png"
    cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in frame.columns]
    corr = frame[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
    )
    ax.set_title("Pearson correlation among numeric columns (synthetic)")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_review_cycle_comparison(frame: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "review_cycle_comparison.png"
    cycle_order = sorted(frame["review_cycle"].unique())
    pct = _stacked_percent(frame, "review_cycle", order=list(cycle_order))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.countplot(
        data=frame,
        x="review_cycle",
        order=cycle_order,
        color="#8da0cb",
        ax=axes[0],
    )
    axes[0].set_title("Row counts by review cycle")
    axes[0].set_xlabel("review_cycle")
    axes[0].set_ylabel("Row count")
    axes[0].tick_params(axis="x", rotation=30)

    pct.plot(
        kind="bar",
        stacked=True,
        color=[PALETTE[b] for b in BAND_ORDER],
        ax=axes[1],
        width=0.8,
    )
    axes[1].set_title("Band mix by review cycle (row %)")
    axes[1].set_xlabel("review_cycle")
    axes[1].set_ylabel("Percent")
    axes[1].set_ylim(0, 100)
    axes[1].legend(title="perf_band_next", loc="upper right")
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle("Review-cycle comparison (synthetic)", y=1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_performance_signal_comparisons(frame: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "performance_signal_comparisons.png"
    cols = [c for c in SIGNAL_COLUMNS if c in frame.columns]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.boxplot(
            data=frame,
            x="perf_band_next",
            y=col,
            hue="perf_band_next",
            order=BAND_ORDER,
            palette=PALETTE,
            legend=False,
            ax=axes[i],
        )
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("perf_band_next")
    fig.suptitle(
        "Selected work/feedback signals by next-cycle band (synthetic; overlap is expected)",
        y=1.01,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_all_figures(path: Path = CLEAN_PATH, out_dir: Path = FIGURE_DIR) -> list[Path]:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _prepare_frame(path)
    writers = [
        plot_target_distribution,
        plot_numeric_distributions,
        plot_performance_by_department,
        plot_performance_by_job_level,
        plot_correlation_heatmap,
        plot_review_cycle_comparison,
        plot_performance_signal_comparisons,
    ]
    saved = [fn(frame, out_dir) for fn in writers]
    return saved


def main() -> None:
    saved = generate_all_figures()
    print("Saved figures:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
