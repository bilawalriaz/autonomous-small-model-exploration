"""Generate efficiency experiment plots."""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mi_atlas.utils import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"
TABLES_DIR = PROJECT_ROOT / "experiments" / "tables"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "font.family": "serif",
    "axes.grid": True, "grid.alpha": 0.3, "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load_result(name):
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_plot(fig, name, save_data=None):
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if save_data:
        with open(TABLES_DIR / f"{name}.json", "w") as f:
            json.dump(save_data, f, indent=2)
    print(f"    Saved: {path}")


def plot_efficiency():
    data = load_result("layer_skipping_early_exit")
    if not data:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Layer skipping KL
    ax1 = axes[0]
    skip_results = data["part1_layer_skipping"]
    names = [r["config"].replace("skip_", "").replace("_", " ") for r in skip_results]
    kls = [r["mean_kl"] for r in skip_results]
    n_skip = [r["n_skipped"] for r in skip_results]

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
    bars = ax1.barh(range(len(names)), kls, color=colors, edgecolor="black", alpha=0.8)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels([f"{n} ({s}L)" for n, s in zip(names, n_skip)], fontsize=8)
    ax1.set_xlabel("Mean KL Divergence (nats)", fontsize=10)
    ax1.set_title("Layer Skipping: Output Degradation\n(0% top-5 overlap in ALL configs)", fontsize=11, fontweight="bold")
    ax1.axvline(x=0, color="black", linewidth=0.5)

    for bar, val in zip(bars, kls):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}", va="center", fontsize=8)

    # Panel 2: Early exit
    ax2 = axes[1]
    exit_results = data["part2_early_exit"]
    exit_layers = [e["exit_layer"] for e in exit_results]
    exit_kls = [e["mean_kl"] for e in exit_results]
    exit_argmax = [e["mean_argmax_match"] * 100 for e in exit_results]
    exit_speedup = [e["theoretical_speedup"] for e in exit_results]

    ax2_twin = ax2.twinx()

    bars = ax2.bar(exit_layers, exit_kls, color="coral", edgecolor="black", alpha=0.7, label="KL Divergence")
    ax2.plot(exit_layers, exit_argmax, "bs-", linewidth=2, markersize=6, label="Argmax Match %")
    ax2_twin.plot(exit_layers, exit_speedup, "g^--", linewidth=2, markersize=6, label="Theoretical Speedup")

    ax2.set_xlabel("Exit Layer", fontsize=10)
    ax2.set_ylabel("KL Divergence / Argmax Match %", fontsize=10, color="coral")
    ax2_twin.set_ylabel("Theoretical Speedup (x)", fontsize=10, color="green")
    ax2.set_title("Early Exit: Quality vs Speed\n(L22 exit = 0% argmax match)", fontsize=11, fontweight="bold")
    ax2.set_xticks(exit_layers)
    ax2.set_ylim(-5, 110)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    # Panel 3: Task-aware selective computation
    ax3 = axes[2]
    task_results = data["part3_selective_computation"]
    task_names = [r["config"].replace("_", " ") for r in task_results]
    task_kls = [r["mean_kl"] for r in task_results]
    task_argmax = [r["mean_argmax_match"] * 100 for r in task_results]

    x = range(len(task_names))
    width = 0.35
    ax3.bar([i - width/2 for i in x], task_kls, width, color="coral", alpha=0.8, label="KL Divergence")
    ax3.bar([i + width/2 for i in x], task_argmax, width, color="steelblue", alpha=0.8, label="Argmax Match %")
    ax3.set_xticks(x)
    ax3.set_xticklabels(task_names, fontsize=8, rotation=20, ha="right")
    ax3.set_ylabel("Value", fontsize=10)
    ax3.set_title("Task-Aware Selective Computation\n(All configs: 0% argmax match)", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.axhline(y=100, color="green", linestyle="--", alpha=0.3)

    fig.suptitle("Efficiency Experiments: Layer Skipping & Early Exit\nKey finding: Naive layer removal destroys output — all layers are necessary",
                 fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    save_plot(fig, "pub_efficiency_experiments", {
        "layer_skip_max_kl": max(kls),
        "layer_skip_min_kl": min(kls),
        "all_top5_overlap_zero": True,
        "early_exit_l22_argmax": exit_argmax[1],
        "conclusion": "All layers necessary. Naive skipping fails. Atlas value is in targeting training, not inference skipping.",
    })


def main():
    print("Generating efficiency plots...")
    plot_efficiency()
    print("Done.")


if __name__ == "__main__":
    main()
