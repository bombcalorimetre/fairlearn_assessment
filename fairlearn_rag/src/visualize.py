"""
visualize.py
------------
Visualization helpers for fairness results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fairlearn.metrics import MetricFrame


# ─────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────
PALETTE = {
    "female":  "#E07B91",
    "male":    "#5B9BD5",
    "older":   "#F4A460",
    "younger": "#8FBC8F",
    "good":    "#4CAF50",
    "poor":    "#F44336",
}


def plot_scores_by_group(results_df: pd.DataFrame, sensitive_col: str, ax=None) -> plt.Axes:
    """Bar chart of mean keyword score per group."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    grouped = results_df.groupby(sensitive_col)["keyword_score"].mean().reset_index()
    colors = [PALETTE.get(g, "#888888") for g in grouped[sensitive_col]]
    bars = ax.bar(grouped[sensitive_col], grouped["keyword_score"], color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(y=results_df["keyword_score"].mean(), color="#333", linestyle="--", linewidth=1.2, label="Overall mean")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean Keyword Score", fontsize=11)
    ax.set_title(f"Answer Quality by '{sensitive_col}'", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, grouped["keyword_score"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_composite_label_distribution(results_df: pd.DataFrame, sensitive_col: str, ax=None) -> plt.Axes:
    """Stacked bar chart of good vs poor answers per group."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    groups = results_df[sensitive_col].unique()
    good_counts = []
    poor_counts = []
    for g in groups:
        sub = results_df[results_df[sensitive_col] == g]
        good_counts.append((sub["composite_label"] == 1).sum())
        poor_counts.append((sub["composite_label"] == 0).sum())
    x = np.arange(len(groups))
    width = 0.45
    ax.bar(x, good_counts, width, label="Good answer", color=PALETTE["good"], edgecolor="white")
    ax.bar(x, poor_counts, width, bottom=good_counts, label="Poor answer", color=PALETTE["poor"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("Number of Queries", fontsize=11)
    ax.set_title(f"Answer Quality Distribution by '{sensitive_col}'", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_metric_frame_heatmap(metric_frame: MetricFrame, ax=None) -> plt.Axes:
    """Heatmap of all metrics across groups from a MetricFrame."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    data = metric_frame.by_group
    im = ax.imshow(data.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, fontsize=11)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=11)
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            ax.text(j, i, f"{data.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, color="black")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("Metrics by Group (Heatmap)", fontsize=13, fontweight="bold")
    return ax


def plot_disparity_summary(report_gender: dict, report_age: dict, ax=None) -> plt.Axes:
    """Side-by-side disparity bars for gender vs age group."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    metrics = list(report_gender["disparity"].index)
    gender_vals = report_gender["disparity"].values
    age_vals = report_age["disparity"].values
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, gender_vals, width, label="Gender disparity", color=PALETTE["female"], alpha=0.85)
    ax.bar(x + width/2, age_vals, width, label="Age group disparity", color=PALETTE["older"], alpha=0.85)
    ax.axhline(y=0.1, color="red", linestyle="--", linewidth=1.2, label="Fairness threshold (0.1)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Disparity (max gap)", fontsize=11)
    ax.set_title("Fairness Disparity by Sensitive Feature", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def full_dashboard(results_df: pd.DataFrame, report_gender: dict, report_age: dict) -> plt.Figure:
    """Render a 2×2 dashboard summarising all fairness results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RAG Fairness Dashboard — Qwen2.5 3B + Fairlearn", fontsize=16, fontweight="bold", y=1.01)

    plot_scores_by_group(results_df, "gender", ax=axes[0, 0])
    plot_scores_by_group(results_df, "age_group", ax=axes[0, 1])
    plot_composite_label_distribution(results_df, "gender", ax=axes[1, 0])
    plot_disparity_summary(report_gender, report_age, ax=axes[1, 1])

    plt.tight_layout()
    return fig
