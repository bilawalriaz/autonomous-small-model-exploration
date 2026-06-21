"""Plotting utilities for experiment results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from .utils import load_config, save_json, PROJECT_ROOT


def get_plot_config() -> dict:
    """Load plotting configuration."""
    return load_config("plotting")


def setup_style() -> None:
    """Set up matplotlib style from config."""
    cfg = get_plot_config()
    style = cfg.get("style", {})
    plt.rcParams.update({
        "figure.dpi": style.get("figure_dpi", 150),
        "font.size": style.get("font_size", 11),
    })
    try:
        import seaborn as sns
        sns.set_style(style.get("seaborn_style", "whitegrid"))
    except ImportError:
        pass


def save_plot(fig: plt.Figure, name: str, save_data: dict | None = None) -> Path:
    """Save a plot and optionally its data table."""
    cfg = get_plot_config()
    out_dir = PROJECT_ROOT / cfg.get("output_dir", "experiments/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = cfg.get("style", {}).get("save_format", "png")
    path = out_dir / f"{name}.{fmt}"
    fig.savefig(path, dpi=cfg.get("style", {}).get("figure_dpi", 150), bbox_inches="tight")
    plt.close(fig)

    # Save data table if provided
    if save_data and cfg.get("save_data_tables", True):
        table_dir = PROJECT_ROOT / cfg.get("data_table_dir", "experiments/tables")
        table_dir.mkdir(parents=True, exist_ok=True)
        save_json(save_data, table_dir / f"{name}.json")

    return path


def plot_task_scores(
    scores: dict[str, float],
    title: str = "Baseline Task Scores",
    ylabel: str = "Score",
) -> Path:
    """Bar chart of task family scores."""
    setup_style()
    cfg = get_plot_config().get("bar_plots", {})

    fig, ax = plt.subplots(figsize=get_plot_config().get("style", {}).get("figsize_default", [10, 6]))

    families = list(scores.keys())
    values = list(scores.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(families)))

    bars = ax.bar(families, values, color=colors,
                  edgecolor=cfg.get("edge_color", "black"),
                  alpha=cfg.get("alpha", 0.8))

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(max(values) * 1.15, 1.0))
    plt.xticks(rotation=45, ha="right")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    return save_plot(fig, "baseline_task_scores", scores)


def plot_ablation_heatmap(
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str = "Layer Ablation Heatmap",
    xlabel: str = "Task Family",
    ylabel: str = "Layer",
    name: str = "layer_ablation_heatmap",
) -> Path:
    """Heatmap of ablation effects."""
    setup_style()
    cfg = get_plot_config().get("heatmaps", {})

    fig, ax = plt.subplots(figsize=cfg.get("figsize", [12, 8]))

    im = ax.imshow(data, cmap=get_plot_config().get("style", {}).get("colormap", "RdYlBu_r"),
                   aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.colorbar(im, ax=ax, shrink=cfg.get("cbar_kws", {}).get("shrink", 0.8))

    # Annotate cells
    if cfg.get("annot", True):
        fmt = cfg.get("fmt", ".2f")
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(data[i, j]) > data.max() / 2 else "black")

    return save_plot(fig, name, {
        "data": data.tolist(),
        "row_labels": row_labels,
        "col_labels": col_labels,
    })


def plot_line(
    x: list | np.ndarray,
    y: list | np.ndarray,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    name: str = "line_plot",
    label: str | None = None,
) -> Path:
    """Simple line plot."""
    setup_style()
    cfg = get_plot_config().get("line_plots", {})

    fig, ax = plt.subplots(figsize=get_plot_config().get("style", {}).get("figsize_default", [10, 6]))
    ax.plot(x, y, linewidth=cfg.get("linewidth", 2), marker="o",
            markersize=cfg.get("marker_size", 6), label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if cfg.get("grid", True):
        ax.grid(True, alpha=0.3)
    if label:
        ax.legend()

    return save_plot(fig, name, {"x": list(x) if not isinstance(x, list) else x, "y": list(y) if not isinstance(y, list) else y})


def plot_multi_line(
    data: dict[str, tuple[list, list]],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    name: str = "multi_line",
) -> Path:
    """Multi-series line plot."""
    setup_style()

    fig, ax = plt.subplots(figsize=get_plot_config().get("style", {}).get("figsize_default", [10, 6]))
    for label, (x, y) in data.items():
        ax.plot(x, y, marker="o", label=label, linewidth=2, markersize=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_data = {label: {"x": x, "y": y} for label, (x, y) in data.items()}
    return save_plot(fig, name, save_data)
