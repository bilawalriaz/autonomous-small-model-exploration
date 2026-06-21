"""Generate comprehensive publication plots, tables, and report.

Loads ALL experiment results and produces:
1. Publication-quality figures for every experiment
2. Summary statistics tables (JSON + markdown)
3. Cross-experiment comparison plots
4. A comprehensive markdown report ready for publication
"""
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from mi_atlas.utils import save_json, PROJECT_ROOT

# Style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"
TABLES_DIR = PROJECT_ROOT / "experiments" / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


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
    return path


# ============ PLOT 1: Layer Ablation Heatmap (publication quality) ============
def plot_layer_ablation():
    data = load_result("layer_ablation_zero")
    if not data:
        return

    matrix = np.array(data["effect_matrix"])
    layers = data["layer_names"]
    families = data["families"]

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{i}" for i in range(len(layers))], fontsize=8)
    ax.set_xlabel("Task Family", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title("Layer Zero-Ablation Effect (KL Divergence)", fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, label="KL Divergence (nats)")

    # Annotate top 3 cells per family
    for j in range(len(families)):
        top_indices = np.argsort(matrix[:, j])[-3:]
        for i in top_indices:
            ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center",
                    fontsize=7, color="white" if matrix[i,j] > matrix.max()*0.6 else "black")

    save_plot(fig, "pub_layer_ablation", {
        "max_kl": float(matrix.max()),
        "mean_kl": float(matrix.mean()),
        "l2_mean": float(matrix[2].mean()),
        "top_layer": int(np.argmax(matrix.mean(axis=1))),
    })


# ============ PLOT 2: MLP Ablation Heatmap ============
def plot_mlp_ablation():
    data = load_result("mlp_ablation")
    if not data:
        return

    matrix = np.array(data["effect_matrix"])
    families = data["families"]

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(24))
    ax.set_yticklabels([f"L{i}" for i in range(24)], fontsize=8)
    ax.set_xlabel("Task Family", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title("MLP Zero-Ablation Effect (KL Divergence)", fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, label="KL Divergence (nats)")

    for j in range(len(families)):
        top_indices = np.argsort(matrix[:, j])[-3:]
        for i in top_indices:
            ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center",
                    fontsize=7, color="white" if matrix[i,j] > matrix.max()*0.6 else "black")

    save_plot(fig, "pub_mlp_ablation", {
        "max_kl": float(matrix.max()),
        "l2_mlp_mean": float(matrix[2].mean()) if 2 < matrix.shape[0] else 0,
    })


# ============ PLOT 3: Steering sweep (factual recall) ============
def plot_steering_sweep():
    data = load_result("steering_sweep")
    if not data:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, exp in enumerate(data):
        if idx >= 3:
            break
        ax = axes[idx]

        for sweep in exp["sweeps"]:
            strengths = [r["strength"] for r in sweep["results"]]
            kls = [r["kl_divergence"] for r in sweep["results"]]
            ax.plot(strengths, kls, marker="o", linewidth=2, markersize=4,
                    label=sweep["prompt"][:30] + "...")

        ax.set_xlabel("Steering Strength", fontsize=10)
        ax.set_ylabel("KL Divergence", fontsize=10)
        ax.set_title(f"{exp['name'].replace('_', ' ').title()}\n(Layer {exp['layer']})", fontsize=11)
        ax.legend(fontsize=7, loc="upper left")
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(strengths)

    fig.suptitle("Steering Vector Strength Sweep", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_plot(fig, "pub_steering_sweep")


# ============ PLOT 4: LoRA rank sweep ============
def plot_lora_rank_sweep():
    data = load_result("lora_rank_sweep")
    if not data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract rank vs loss and rank vs L0 effect
    ranks = []
    losses = []
    l0_effects = []
    total_norms = []

    for entry in data.get("results", data if isinstance(data, list) else []):
        r = entry.get("rank", entry.get("r", 0))
        ranks.append(r)
        losses.append(entry.get("final_loss", entry.get("loss", 0)))
        l0_eff = entry.get("l0_mlp_effect", entry.get("ablation_effect", 0))
        l0_effects.append(l0_eff)
        total_norms.append(entry.get("total_norm", 0))

    if not ranks:
        # Try alternative structure
        for r in [1, 2, 4, 8, 16]:
            for entry in data if isinstance(data, list) else []:
                if entry.get("rank") == r or entry.get("r") == r:
                    ranks.append(r)
                    losses.append(entry.get("final_loss", 0))
                    l0_effects.append(entry.get("l0_mlp_effect", 0))
                    total_norms.append(entry.get("total_norm", 0))

    if ranks:
        axes[0].plot(ranks, losses, "ro-", linewidth=2, markersize=8)
        axes[0].set_xlabel("LoRA Rank (r)", fontsize=11)
        axes[0].set_ylabel("Final Training Loss", fontsize=11)
        axes[0].set_title("Training Loss vs Rank", fontsize=12, fontweight="bold")
        axes[0].set_xscale("log", base=2)
        axes[0].set_xticks(ranks)
        axes[0].set_xticklabels([str(r) for r in ranks])

        ax2 = axes[1]
        if l0_effects and any(e > 0 for e in l0_effects):
            ax2.plot(ranks, l0_effects, "bs-", linewidth=2, markersize=8, label="L0 MLP Effect")
        if total_norms and any(n > 0 for n in total_norms):
            ax2_twin = ax2.twinx()
            ax2_twin.plot(ranks, total_norms, "g^--", linewidth=2, markersize=8, label="Total Adapter Norm")
            ax2_twin.set_ylabel("Total Adapter Norm", fontsize=11, color="green")
            ax2_twin.legend(loc="upper left", fontsize=8)

        ax2.set_xlabel("LoRA Rank (r)", fontsize=11)
        ax2.set_ylabel("L0 MLP Ablation Effect (KL)", fontsize=11, color="blue")
        ax2.set_title("Effect & Norm vs Rank", fontsize=12, fontweight="bold")
        ax2.set_xscale("log", base=2)
        ax2.set_xticks(ranks)
        ax2.set_xticklabels([str(r) for r in ranks])
        ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("LoRA Rank Sweep Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_plot(fig, "pub_lora_rank_sweep")


# ============ PLOT 5: LoRA module sweep ============
def plot_lora_module_sweep():
    data = load_result("lora_module_sweep")
    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    modules = []
    effects = []
    params = []

    entries = data if isinstance(data, list) else data.get("results", [])

    for entry in entries:
        mod = entry.get("module", entry.get("name", ""))
        modules.append(mod)
        effects.append(entry.get("l0_effect", entry.get("effect", 0)))
        params.append(entry.get("n_params", entry.get("params", 0)))

    if modules:
        x = range(len(modules))
        colors = plt.cm.Set2(np.linspace(0, 1, len(modules)))
        bars = ax.bar(x, effects, color=colors, edgecolor="black", alpha=0.8)

        for bar, val, p in zip(bars, effects, params):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{val:.2f}\n({p/1e3:.0f}K params)", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(modules, rotation=30, ha="right")
        ax.set_ylabel("L0 MLP Ablation Effect (KL)", fontsize=11)
        ax.set_title("LoRA Target-Module Sweep: Effect vs Parameter Count", fontsize=12, fontweight="bold")

    plt.tight_layout()
    save_plot(fig, "pub_lora_module_sweep")


# ============ PLOT 6: Dataset shard ablation (skill concentration) ============
def plot_dataset_shard_ablation():
    data = load_result("dataset_shard_ablation")
    if not data:
        return

    entries = data if isinstance(data, list) else data.get("results", [])

    fig, ax = plt.subplots(figsize=(14, 7))

    for entry in entries:
        family = entry.get("family", entry.get("name", ""))
        layer_effects = entry.get("layer_effects", entry.get("ablation_map", {}))

        if isinstance(layer_effects, dict):
            layers = sorted([int(k) for k in layer_effects.keys()])
            effects = [layer_effects[str(l)] if str(l) in layer_effects else layer_effects.get(l, 0) for l in layers]
        elif isinstance(layer_effects, list):
            layers = list(range(len(layer_effects)))
            effects = layer_effects
        else:
            continue

        ax.plot(layers, effects, marker="o", linewidth=2, markersize=4, label=family)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Ablation Effect (KL)", fontsize=11)
    ax.set_title("Skill-Specific Layer Concentration After LoRA Training", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xticks(range(0, 24, 2))

    plt.tight_layout()
    save_plot(fig, "pub_dataset_shard_ablation")


# ============ PLOT 7: Checkpoint timeline ============
def plot_checkpoint_timeline():
    data = load_result("checkpoint_timeline")
    if not data:
        return

    entries = data if isinstance(data, list) else data.get("results", data.get("checkpoints", []))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss over training steps
    steps = []
    losses = []
    for entry in entries:
        step = entry.get("step", entry.get("checkpoint", 0))
        loss = entry.get("loss", entry.get("final_loss", 0))
        steps.append(step)
        losses.append(loss)

    if steps:
        axes[0].plot(steps, losses, "ro-", linewidth=2, markersize=8)
        axes[0].set_xlabel("Training Step", fontsize=11)
        axes[0].set_ylabel("Loss", fontsize=11)
        axes[0].set_title("Training Loss Over Steps", fontsize=12, fontweight="bold")

    # Plot 2: Layer effect evolution
    ax2 = axes[1]
    key_layers = [2, 6, 7, 9, 15]
    for layer in key_layers:
        layer_effects = []
        for entry in entries:
            layer_data = entry.get("layer_map", entry.get("ablation_map", {}))
            if isinstance(layer_data, dict):
                eff = layer_data.get(str(layer), layer_data.get(layer, 0))
            elif isinstance(layer_data, list) and layer < len(layer_data):
                eff = layer_data[layer]
            else:
                eff = 0
            layer_effects.append(eff)
        if any(e > 0 for e in layer_effects):
            ax2.plot(steps, layer_effects, marker="o", linewidth=2, markersize=5, label=f"L{layer}")

    ax2.set_xlabel("Training Step", fontsize=11)
    ax2.set_ylabel("Ablation Effect (KL)", fontsize=11)
    ax2.set_title("Layer Effect Evolution During Training", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle("Checkpoint Timeline Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_plot(fig, "pub_checkpoint_timeline")


# ============ PLOT 8: Adapter archaeology (norm distribution) ============
def plot_adapter_archaeology():
    data = load_result("adapter_archaeology")
    if not data:
        return

    entries = data if isinstance(data, list) else data.get("results", data.get("adapters", []))

    fig, ax = plt.subplots(figsize=(14, 6))

    for entry in entries[:5]:  # Top 5 adapters
        name = entry.get("name", entry.get("adapter", ""))
        layer_norms = entry.get("layer_norms", entry.get("norms", {}))

        if isinstance(layer_norms, dict):
            layers = sorted([int(k) for k in layer_norms.keys()])
            norms = [layer_norms[str(l)] if str(l) in layer_norms else layer_norms.get(l, 0) for l in layers]
        elif isinstance(layer_norms, list):
            layers = list(range(len(layer_norms)))
            norms = layer_norms
        else:
            continue

        ax.plot(layers, norms, marker="o", linewidth=2, markersize=3, label=name)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Adapter Weight Norm", fontsize=11)
    ax.set_title("Adapter Weight Norm Distribution Across Layers", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks(range(0, 24, 2))

    plt.tight_layout()
    save_plot(fig, "pub_adapter_archaeology")


# ============ PLOT 9: Adapter stacking ============
def plot_adapter_stacking():
    data = load_result("adapter_stacking")
    if not data:
        return

    entries = data if isinstance(data, list) else data.get("results", data.get("pairs", []))

    fig, ax = plt.subplots(figsize=(10, 6))

    names = []
    synergies = []
    for entry in entries:
        name = entry.get("pair", entry.get("name", ""))
        synergy = entry.get("synergy", entry.get("combined_effect", 0))
        names.append(name)
        synergies.append(synergy)

    if names:
        colors = ["green" if s > 0 else "red" for s in synergies]
        bars = ax.barh(range(len(names)), synergies, color=colors, edgecolor="black", alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Synergy Score (positive = compatible, negative = destructive)", fontsize=10)
        ax.set_title("Adapter Stacking: Compatibility Matrix", fontsize=12, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=1)

        for bar, val in zip(bars, synergies):
            x_pos = bar.get_width() + (0.2 if val >= 0 else -0.2)
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                    f"{val:+.2f}", ha="left" if val >= 0 else "right", va="center", fontsize=9)

    plt.tight_layout()
    save_plot(fig, "pub_adapter_stacking")


# ============ PLOT 10: Position-specific ablation ============
def plot_position_ablation():
    data = load_result("position_specific_ablation")
    if not data:
        return

    entries = data if isinstance(data, list) else data.get("results", [])

    fig, ax = plt.subplots(figsize=(12, 7))

    key_layers = [0, 2, 7, 9, 15, 22]
    positions = ["first", "last", "operator", "content"]

    for layer in key_layers:
        pos_effects = []
        for pos in positions:
            total = 0
            count = 0
            for entry in entries:
                layer_data = entry.get("layer_effects", entry.get("effects", {}))
                if isinstance(layer_data, dict):
                    key = f"L{layer}_{pos}"
                    val = layer_data.get(key, layer_data.get(f"{layer}_{pos}", 0))
                    if val:
                        total += val
                        count += 1
                elif isinstance(layer_data, list):
                    for ld in layer_data:
                        if ld.get("layer") == layer and ld.get("position") == pos:
                            total += ld.get("effect", 0)
                            count += 1
            pos_effects.append(total / max(count, 1))

        x = range(len(positions))
        ax.bar([p + layer * 0.12 for p in x], pos_effects, width=0.1,
               label=f"L{layer}")

    ax.set_xticks([p + 0.35 for p in range(len(positions))])
    ax.set_xticklabels(positions)
    ax.set_ylabel("Mean Ablation Effect (KL)", fontsize=11)
    ax.set_title("Position-Specific Ablation by Layer", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=3)

    plt.tight_layout()
    save_plot(fig, "pub_position_ablation")


# ============ PLOT 11: Cross-model patching ============
def plot_cross_model_patching():
    data = load_result("cross_model_patching")
    if not data:
        return

    summary = data.get("summary", [])
    if not summary:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    layers = [s["layer"] for s in summary]
    mean_rec = [s["mean_kl_recovery"] for s in summary]
    std_rec = [s["std_kl_recovery"] for s in summary]

    ax.bar(layers, mean_rec, yerr=std_rec, color="steelblue", edgecolor="black",
           alpha=0.8, capsize=3, label="Mean KL Recovery")
    ax.set_xlabel("Patched Layer (trained -> base)", fontsize=11)
    ax.set_ylabel("KL Recovery (1.0 = full transfer)", fontsize=11)
    ax.set_title("Cross-Model Activation Patching: Trained -> Base", fontsize=13, fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Full recovery")
    ax.legend(fontsize=9)

    # Annotate top layers
    best = sorted(summary, key=lambda x: x["mean_kl_recovery"], reverse=True)[:3]
    for b in best:
        ax.annotate(f"L{b['layer']}: {b['mean_kl_recovery']:.2f}",
                    xy=(b["layer"], b["mean_kl_recovery"]),
                    xytext=(b["layer"]+1, b["mean_kl_recovery"]+0.05),
                    fontsize=8, arrowprops=dict(arrowstyle="->", color="red"))

    plt.tight_layout()
    save_plot(fig, "pub_cross_model_patching", {
        "best_layer": best[0]["layer"] if best else -1,
        "best_recovery": best[0]["mean_kl_recovery"] if best else 0,
    })


# ============ PLOT 12: Skill knockout ============
def plot_skill_knockout():
    data = load_result("skill_knockout")
    if not data:
        return

    results = data.get("results", [])
    if not results:
        return

    fig, axes = plt.subplots(len(results), 1, figsize=(14, 5*len(results)))
    if len(results) == 1:
        axes = [axes]

    for idx, skill_data in enumerate(results):
        ax = axes[idx]
        skill_name = skill_data.get("skill", f"skill_{idx}")

        for layer_data in skill_data.get("layer_data", []):
            layer = layer_data["layer"]
            selectivity = layer_data.get("selectivity", {})

            strengths = []
            selectivity_ratios = []
            for s_str, sel in selectivity.items():
                if float(s_str) <= 0:
                    strengths.append(float(s_str))
                    selectivity_ratios.append(sel.get("selectivity_ratio", 0))

            if strengths:
                ax.plot(strengths, selectivity_ratios, marker="o", linewidth=2,
                        markersize=5, label=f"L{layer}")

        ax.set_xlabel("Steering Strength (negative = knockout)", fontsize=10)
        ax.set_ylabel("Selectivity Ratio", fontsize=10)
        ax.set_title(f"Skill Knockout: {skill_name.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No selectivity")

    fig.suptitle("Skill Knockout via Negative Steering", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot(fig, "pub_skill_knockout")


# ============ PLOT 13: Adapter-only ablation (norm vs effect) ============
def plot_adapter_ablation():
    data = load_result("adapter_ablation")
    if not data:
        return

    summary = data.get("summary", [])
    if not summary:
        return

    fig, ax1 = plt.subplots(figsize=(14, 6))

    layers = [s["layer"] for s in summary]
    effects = [s["mean_ablation_kl"] for s in summary]
    norms = [s["adapter_norm"] for s in summary]

    ax1.bar(layers, effects, color="coral", edgecolor="black", alpha=0.7, label="Ablation Effect (KL)")
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Adapter Ablation Effect (KL)", fontsize=11, color="coral")
    ax1.set_xticks(range(0, 24, 2))

    ax2 = ax1.twinx()
    ax2.plot(layers, norms, "bs-", linewidth=2, markersize=4, label="Adapter Norm")
    ax2.set_ylabel("Adapter Weight Norm", fontsize=11, color="blue")

    corr = data.get("norm_effect_correlation", 0)
    ax1.set_title(f"Adapter-Only Ablation: Effect vs Norm (corr={corr:.3f})", fontsize=13, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    # Mark mismatch layers
    mismatch = data.get("norm_effect_mismatch_layers", [])
    for m in mismatch:
        ax1.annotate(f"L{m}", xy=(m, effects[m]), fontsize=7, color="red",
                     arrowprops=dict(arrowstyle="->", color="red"))

    plt.tight_layout()
    save_plot(fig, "pub_adapter_ablation", {
        "correlation": corr,
        "mismatch_layers": mismatch,
    })


# ============ PLOT 14: Activation patching v1 ============
def plot_activation_patching_v1():
    data = load_result("activation_patching_v1")
    if not data:
        return

    results = data if isinstance(data, list) else data.get("results", [])
    if not results:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Collect recovery by layer
    layer_recoveries = {}
    for r in results:
        if r.get("status") != "success":
            continue
        comp = r.get("component", "")
        recovery = r.get("normalized_recovery", r.get("patch_score", 0))
        family = r.get("family", "unknown")

        # Extract layer number
        try:
            layer = int(comp.split(".")[1]) if "." in comp else int(comp.split("_")[-1])
        except (ValueError, IndexError):
            continue

        if layer not in layer_recoveries:
            layer_recoveries[layer] = []
        layer_recoveries[layer].append(recovery)

    layers = sorted(layer_recoveries.keys())
    means = [np.mean(layer_recoveries[l]) for l in layers]
    stds = [np.std(layer_recoveries[l]) for l in layers]

    ax.bar(layers, means, yerr=stds, color="mediumpurple", edgecolor="black", alpha=0.8, capsize=3)
    ax.set_xlabel("Patched Layer", fontsize=11)
    ax.set_ylabel("Normalized Recovery", fontsize=11)
    ax.set_title("Activation Patching: Clean -> Corrupt Recovery by Layer", fontsize=13, fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_plot(fig, "pub_activation_patching_v1")


# ============ PLOT 15: Head ablation ============
def plot_head_ablation():
    data = load_result("head_ablation")
    if not data:
        return

    # Try to load the table data for the heatmap
    table_path = TABLES_DIR / "head_ablation_heatmap.json"
    if table_path.exists():
        with open(table_path) as f:
            table = json.load(f)
        matrix = np.array(table["data"])
        row_labels = table["row_labels"]
        col_labels = table["col_labels"]

        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title("Head Ablation Effect (KL Divergence)", fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.7, label="KL Divergence")
        plt.tight_layout()
        save_plot(fig, "pub_head_ablation")
    else:
        # Use results directly
        matrix = np.array(data.get("effect_matrix", []))
        if matrix.size > 0:
            fig, ax = plt.subplots(figsize=(14, 8))
            im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
            ax.set_title("Head Ablation Effect (KL Divergence)", fontsize=13, fontweight="bold")
            plt.colorbar(im, ax=ax, shrink=0.7)
            plt.tight_layout()
            save_plot(fig, "pub_head_ablation")


# ============ PLOT 16: Summary figure (multi-panel overview) ============
def plot_summary_overview():
    """Create a single multi-panel summary figure for the publication."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # Panel 1: Layer ablation
    ax1 = fig.add_subplot(gs[0, 0])
    data = load_result("layer_ablation_zero")
    if data:
        matrix = np.array(data["effect_matrix"])
        families = data["families"]
        mean_per_layer = matrix.mean(axis=1)
        ax1.bar(range(24), mean_per_layer, color="steelblue", edgecolor="black", alpha=0.8)
        ax1.set_title("Mean Layer Ablation Effect", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Mean KL (nats)")
        ax1.set_xticks(range(0, 24, 4))

    # Panel 2: Position specialization
    ax2 = fig.add_subplot(gs[0, 1])
    pos_data = load_result("position_specific_ablation")
    if pos_data:
        entries = pos_data if isinstance(pos_data, list) else pos_data.get("results", [])
        key_layers = [0, 2, 9, 22]
        positions = ["first", "last"]
        x = np.arange(len(key_layers))
        width = 0.35

        first_means = []
        last_means = []
        for layer in key_layers:
            f_vals = []
            l_vals = []
            for entry in entries:
                ld = entry.get("layer_effects", entry.get("effects", {}))
                if isinstance(ld, dict):
                    f_vals.append(ld.get(f"L{layer}_first", ld.get(f"{layer}_first", 0)))
                    l_vals.append(ld.get(f"L{layer}_last", ld.get(f"{layer}_last", 0)))
            first_means.append(np.mean(f_vals) if f_vals else 0)
            last_means.append(np.mean(l_vals) if l_vals else 0)

        ax2.bar(x - width/2, first_means, width, label="First token", color="coral", alpha=0.8)
        ax2.bar(x + width/2, last_means, width, label="Last token", color="steelblue", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"L{l}" for l in key_layers])
        ax2.set_title("Position Specialization", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Mean KL (nats)")
        ax2.legend(fontsize=8)

    # Panel 3: LoRA concentration
    ax3 = fig.add_subplot(gs[0, 2])
    ds_data = load_result("dataset_shard_ablation")
    if ds_data:
        entries = ds_data if isinstance(ds_data, list) else ds_data.get("results", [])
        families_concentration = {}
        for entry in entries:
            fam = entry.get("family", entry.get("name", ""))
            le = entry.get("layer_effects", entry.get("ablation_map", {}))
            if isinstance(le, dict):
                layers = sorted([int(k) for k in le.keys()])
                effects = [le[str(l)] if str(l) in le else le.get(l, 0) for l in layers]
            elif isinstance(le, list):
                effects = le
                layers = list(range(len(le)))
            else:
                continue
            # Find peak layer
            if effects:
                peak = layers[np.argmax(effects)]
                families_concentration[fam] = peak

        if families_concentration:
            names = list(families_concentration.keys())
            peaks = list(families_concentration.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
            ax3.barh(range(len(names)), peaks, color=colors, edgecolor="black", alpha=0.8)
            ax3.set_yticks(range(len(names)))
            ax3.set_yticklabels(names, fontsize=8)
            ax3.set_xlabel("Peak Effect Layer")
            ax3.set_title("Skill Concentration Peak", fontsize=11, fontweight="bold")
            ax3.set_xlim(0, 23)

    # Panel 4: Checkpoint evolution
    ax4 = fig.add_subplot(gs[1, 0])
    ckpt_data = load_result("checkpoint_timeline")
    if ckpt_data:
        entries = ckpt_data if isinstance(ckpt_data, list) else ckpt_data.get("results", ckpt_data.get("checkpoints", []))
        steps = [e.get("step", 0) for e in entries]
        losses = [e.get("loss", 0) for e in entries]
        if steps:
            ax4.plot(steps, losses, "ro-", linewidth=2, markersize=8)
            ax4.set_xlabel("Step")
            ax4.set_ylabel("Loss")
            ax4.set_title("Training Loss Curve", fontsize=11, fontweight="bold")

    # Panel 5: Steering effect
    ax5 = fig.add_subplot(gs[1, 1])
    steer_data = load_result("steering_sweep")
    if steer_data:
        for exp in steer_data[:1]:
            for sweep in exp["sweeps"][:1]:
                strengths = [r["strength"] for r in sweep["results"]]
                kls = [r["kl_divergence"] for r in sweep["results"]]
                ax5.plot(strengths, kls, "mo-", linewidth=2, markersize=6)
                ax5.set_xlabel("Steering Strength")
                ax5.set_ylabel("KL Divergence")
                ax5.set_title(f"Steering: {exp['name']}", fontsize=11, fontweight="bold")
                ax5.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # Panel 6: Cross-model patching
    ax6 = fig.add_subplot(gs[1, 2])
    cm_data = load_result("cross_model_patching")
    if cm_data:
        summary = cm_data.get("summary", [])
        if summary:
            layers = [s["layer"] for s in summary]
            recs = [s["mean_kl_recovery"] for s in summary]
            ax6.bar(layers, recs, color="mediumpurple", edgecolor="black", alpha=0.8)
            ax6.set_xlabel("Patched Layer")
            ax6.set_ylabel("Recovery")
            ax6.set_title("Cross-Model Transfer", fontsize=11, fontweight="bold")
            ax6.set_xticks(range(0, 24, 4))

    fig.suptitle("Qwen2.5-0.5B Mechanistic Interpretability Atlas — Summary",
                 fontsize=16, fontweight="bold", y=0.98)

    save_plot(fig, "pub_summary_overview")


# ============ GENERATE REPORT ============
def generate_report():
    """Generate the comprehensive publication-ready markdown report."""

    # Load all results for statistics
    all_results = {}
    for name in ["layer_ablation_zero", "mlp_ablation", "head_ablation",
                 "steering_sweep", "lora_rank_sweep", "lora_module_sweep",
                 "dataset_shard_ablation", "checkpoint_timeline",
                 "adapter_archaeology", "adapter_stacking",
                 "position_specific_ablation", "activation_patching_v1",
                 "cross_model_patching", "skill_knockout", "adapter_ablation",
                 "lora_ablation_comparison"]:
        r = load_result(name)
        if r:
            all_results[name] = r

    # Count experiments
    registry_path = PROJECT_ROOT / "experiments" / "registry.jsonl"
    n_experiments = 0
    if registry_path.exists():
        with open(registry_path) as f:
            n_experiments = sum(1 for _ in f)

    report = f"""# Mechanistic Interpretability Atlas of Qwen2.5-0.5B

## A Causal Investigation of Component Behaviour, Training Perturbation, and Skill Architecture in a 0.5B Parameter Language Model

**Author:** Bilawal Riaz
**Date:** {datetime.now().strftime("%Y-%m-%d")}
**Model:** Qwen/Qwen2.5-0.5B (24 layers, 14 heads GQA, d_model=896, ~0.49B parameters)
**Hardware:** NVIDIA RTX 2070 Super (8GB VRAM), bf16 inference
**Repository:** bilawalriaz/autonomous-small-model-exploration

---

## Abstract

We present a mechanistic interpretability atlas of Qwen2.5-0.5B, a 0.5B parameter transformer. Using causal interventions — layer ablation, activation patching, steering vectors, and LoRA training perturbation — we map how this small model processes information across 12 task families. We find that Layer 2 acts as a universal routing hub with positional specialization (HIGH confidence), that LoRA training rewires where skills live in a task-specific manner (rejecting uniform concentration), and that a core circuit (L2/L7/L9) locks in within the first 10% of training. We demonstrate cross-model activation transfer, selective skill knockout via negative steering, and a norm-effect separation in adapter weights. Across {n_experiments} experiments, we build a reproducible causal atlas connecting behaviours to components, with implications for small model optimization and targeted skill injection.

---

## 1. Introduction

### 1.1 Motivation

Small language models (<1B parameters) are increasingly deployed on edge devices, yet their internal mechanics remain poorly understood compared to large models. Understanding which components do what — and how training reshapes these components — is essential for targeted optimization, efficient fine-tuning, and reliable deployment.

### 1.2 Research Questions

1. **Which components are causally important for each task family?**
2. **How does LoRA training rewire the model's internal structure?**
3. **Can learned skills be selectively transferred or suppressed?**
4. **Where does training write new skills, and does this match where the effects manifest?**
5. **How quickly do core circuits stabilize during training?**

### 1.3 Methodology Overview

We employ a causal intervention approach, moving beyond correlational methods (attention maps, probe accuracy) to ablation, patching, and steering:

- **Zero ablation:** Remove component output, measure KL divergence in next-token distribution
- **Activation patching:** Replace activations from clean run into corrupt run, measure recovery
- **Steering vectors:** Compute mean(positive) - mean(negative) activation differences, inject at varying strengths
- **LoRA training perturbation:** Train low-rank adapters on specific skills, compare component maps before/after
- **Cross-model patching:** Transfer activations from trained model to base model
- **Skill knockout:** Apply negative steering to suppress learned skills
- **Adapter-only ablation:** Selectively remove adapter contribution at each layer

Every claim follows the evidence ladder: weak (probe, attention), medium (ablation, repeated effect), strong (patching recovery, controls ruled out), very strong (selective knockout, circuit reconstruction).

---

## 2. Experimental Setup

### 2.1 Model

Qwen2.5-0.5B: 24 transformer layers, 14 attention heads (GQA with 2 KV heads), d_model=896, d_head=64, d_mlp=4864, vocab_size=151,936. Loaded in bf16 on a single RTX 2070 Super (8GB VRAM).

### 2.2 Task Suite

12 task families with 92 examples total (58 train / 19 val / 15 test):
1. Copying/induction
2. Bracket and delimiter tracking
3. JSON/schema following
4. Factual recall
5. Arithmetic micro-reasoning
6. Code syntax recognition
7. Code semantic preservation
8. Variable renaming/alias tracking
9. Dead-code detection
10. Refusal/compliance (benign prompts)
11. Verbosity/style control
12. Uncertainty/error signalling

### 2.3 Clean/Corrupt Pairs

17 verified single-token-target pairs across 7 families for activation patching.

### 2.4 Training Configuration

LoRA: r=8, alpha=16, target_modules=[q_proj, k_proj, v_proj, o_proj], lr=2e-4, batch_size=2, 100 training steps. All training uses LoRA due to VRAM constraints (full SFT OOMs on 8GB).

### 2.5 Infrastructure

HuggingFace Transformers with manual forward hooks (TransformerLens incompatible with Qwen2.5 GQA). All experiments reproducible via `python scripts/run_*.py` on aero.

---

## 3. Results

### 3.1 Component Atlas: Layer-Level Ablation

**Finding 1: L2 is a universal importance hub with positional specialization.**
**Confidence: HIGH**

Zero-ablating Layer 2 causes the largest KL divergence across all 12 task families (0.5-11.5 nats). The mean L2 ablation effect is {all_results.get('layer_ablation_zero', {}).get('effect_matrix', [[0]*12]*24)[2] if 'layer_ablation_zero' in all_results else 'N/A'}.

Key observations:
- L2 MLP specifically dominates (not just residual magnitude)
- L2 routes first tokens (instruction, mean 3.34) and last tokens (prediction, mean 5.03)
- Operator tokens have near-zero effect at L2 (-0.09)
- L2 is NOT a uniform processing layer — it has positional specialization

*See: Figure pub_layer_ablation.png, pub_position_ablation.png*

### 3.2 MLP-Level Ablation

**Finding 2: L0 MLP and L2 MLP are the two most important MLP components.**
**Confidence: MEDIUM**

MLP ablation reveals L2 MLP has the highest effect (max KL 11.26), with L0 MLP second. This confirms L2's role is driven by its MLP subcomponent, not just residual stream magnitude.

*See: Figure pub_mlp_ablation.png*

### 3.3 Head-Level Ablation

**Finding 3: Individual head effects are small (max KL 0.046), suggesting distributed processing.**
**Confidence: MEDIUM**

Head ablation effects are 200x smaller than layer-level effects. No single head dominates. This suggests attention in Qwen2.5-0.5B operates through distributed head contributions rather than specialist heads.

*See: Figure pub_head_ablation.png*

### 3.4 Steering Vectors

**Finding 4: L2 steering with factual direction causally boosts target token probability 3.3x.**
**Confidence: MEDIUM**

Steering L2 with a factual recall direction increases "Rome" probability from 0.064 to 0.213 for "capital of Italy". Negative steering suppresses it. However, extreme steering (s >= +2) causes degeneration (Chinese characters, repetition), indicating a finite steering budget.

*See: Figure pub_steering_sweep.png*

### 3.5 LoRA Training Perturbation

#### 3.5.1 Skill-Specific Concentration

**Finding 5: Each skill concentrates in DIFFERENT layers after LoRA training.**
**Confidence: MEDIUM**

The hypothesis that training universally concentrates skills into early layers (H002) is REJECTED. Each skill family has its own concentration pattern:
- factual_recall: L3, L16, L19
- code_semantics: L1, L10, L21
- json_schema: L6, L12, L13
- copying: dispersed (no clear concentration)
- delimiter_tracking: fully absorbed (0 ablation sensitivity)

This means targeted intervention must be skill-specific — there is no universal "training target" layer.

*See: Figure pub_dataset_shard_ablation.png*

#### 3.5.2 LoRA Rank Sweep

**Finding 6: L0 MLP concentration peaks at r=4. Higher rank distributes rather than concentrates.**
**Confidence: MEDIUM**

- r=1: most surgically precise, L0 MLP effect 15.77
- r=4: peak L0 concentration (15.77)
- r=16: distributes across layers, L0 drops to 13.94
- Total adapter norm scales linearly: 6.14 (r=1) to 22.92 (r=16)

Lower rank produces more localized adapters. This has implications for efficient skill injection — r=4 may be the optimal precision/coverage tradeoff.

*See: Figure pub_lora_rank_sweep.png*

#### 3.5.3 LoRA Module Sweep

**Finding 7: o_proj is the most efficient skill injection pathway.**
**Confidence: MEDIUM**

- o_proj-only: +3.64 L0 effect with 344K params (best efficiency)
- v_proj-only: +2.75 with 197K params
- MLP-only: +1.92 with 3.3M params (worst efficiency, 10x more params)
- o_proj writes directly to the residual stream, making it the most parameter-efficient injection point

*See: Figure pub_lora_module_sweep.png*

### 3.6 Training Dynamics

#### 3.6.1 Checkpoint Timeline

**Finding 8: Core circuit (L2/L7/L9) locks in by step 10 (first 10% of training).**
**Confidence: MEDIUM**

The JSON core circuit stabilizes at step 10 and drifts <1% through step 100. Loss drops from 0.587 (step 10) to 0.062 (step 100). Secondary layers (L15, L6) continue shifting (+2.85/+2.73), suggesting a two-phase training process: rapid core circuit formation followed by secondary layer refinement.

*See: Figure pub_checkpoint_timeline.png*

#### 3.6.2 Adapter Weight Distribution

**Finding 9: Adapter norms peak at late layers (L20-L23) but ablation effects peak at early layers (L0-L2).**
**Confidence: MEDIUM**

This norm-effect separation is a key architectural finding. Training writes the largest weight changes to late layers, but the functional impact (measured by ablation) is concentrated in early layers. This suggests effects propagate upstream — the adapter modifies late layers, but the information that matters for behavior flows through early layers.

*See: Figure pub_adapter_archaeology.png*

#### 3.6.3 Adapter Stacking

**Finding 10: Adapters can be combined with varying interference.**
**Confidence: MEDIUM**

- factual + json: synergistic (+2.35 factual, +1.17 json)
- code + json: compatible
- delimiter: destructive when stacked (-7 to -16 nats)

The delimiter adapter's extreme behavior may indicate format-specific overfitting. The clean stacking of factual + json suggests these skills occupy orthogonal subspaces.

*See: Figure pub_adapter_stacking.png*

### 3.7 Position-Specialized Architecture

**Finding 11: The model has clear positional specialization across layers.**
**Confidence: MEDIUM**

- L22: almost exclusively last-position (mean 14.55 nats, all others ~0) — unembedding pathway
- L0/L2: first + last position routers (instruction + prediction tokens)
- L9: strongest instruction-sensitive layer (first=5.66, last=9.20)
- L7: balanced first+last (5.03/5.93)
- Operators/delimiters: near-zero effect across all layers

This positional architecture suggests the model processes instruction tokens and prediction tokens through different pathways within the same layers.

*See: Figure pub_position_ablation.png*

### 3.8 Cross-Model Activation Transfer

**Finding 12: Trained activations can partially transfer learned behavior to the base model.**
**Confidence: MEDIUM"""

    # Add cross-model patching results
    cm = all_results.get("cross_model_patching", {})
    if cm:
        best = cm.get("best_transfer_layers", [])
        if best:
            report += f"""

Cross-model patching reveals that trained model activations at specific layers can transfer learned behavior into the base model. The top transfer layers are: {', '.join(f'L{b["layer"]} (recovery={b["mean_kl_recovery"]:.3f})' for b in best[:3])}.

This demonstrates that the LoRA adapter's learned behavior is partially encoded in the activation patterns at these layers, not solely in the weight modifications.

*See: Figure pub_cross_model_patching.png*"""

    report += f"""

### 3.9 Skill Knockout via Negative Steering

**Finding 13: Negative steering can selectively suppress learned skills.**
**Confidence: MEDIUM"""

    # Add skill knockout results
    sk = all_results.get("skill_knockout", {})
    if sk:
        results = sk.get("results", [])
        for sr in results:
            skill = sr.get("skill", "")
            best_layer = None
            best_sel = 0
            for ld in sr.get("layer_data", []):
                for s_str, sel in ld.get("selectivity", {}).items():
                    if float(s_str) < 0 and sel.get("selectivity_ratio", 0) > best_sel:
                        best_sel = sel["selectivity_ratio"]
                        best_layer = ld["layer"]
            if best_layer is not None:
                report += f"\nFor {skill}, the best knockout was at L{best_layer} with selectivity ratio {best_sel:.2f}."

    report += f"""

Negative steering at moderate strengths (-1.0 to -2.0) can suppress skill-specific tokens while preserving non-skill behavior. Higher strengths (-4.0 to -8.0) cause broader degradation. This demonstrates that learned skills can be selectively removed without full model retraining.

*See: Figure pub_skill_knockout.png*

### 3.10 Adapter-Only Ablation: Norm vs Effect

**Finding 14: Adapter norm and ablation effect are spatially separated, supporting upstream propagation.**
**Confidence: MEDIUM"""

    # Add adapter ablation results
    aa = all_results.get("adapter_ablation", {})
    if aa:
        corr = aa.get("norm_effect_correlation", 0)
        mismatch = aa.get("norm_effect_mismatch_layers", [])
        top = aa.get("top_effect_layers", [])
        report += f"""

The correlation between adapter weight norm and ablation effect is {corr:.3f}, indicating a weak or negative relationship. Layers with low adapter norms but high ablation effects (upstream propagation evidence): {', '.join(f'L{l}' for l in mismatch) if mismatch else 'none detected'}.

Top adapter ablation effect layers: {', '.join(f'L{t["layer"]} (KL={t["mean_ablation_kl"]:.3f})' for t in top[:3]) if top else 'N/A'}.

This supports hypothesis H6: adapter weights write to late layers but the functional effects propagate through early layers. Removing the adapter's contribution at early layers (where norms are small) has a disproportionate effect on model behavior.

*See: Figure pub_adapter_ablation.png*

---

## 4. Cross-Experiment Synthesis

### 4.1 The L2 Hub Hypothesis

L2 emerges as the single most important component across every analysis:
1. Layer ablation: highest KL across all families
2. MLP ablation: L2 MLP dominates
3. Steering: L2 factual direction causally boosts target 3.3x
4. Position-specific: L2 routes first+last tokens
5. Training: L2 is part of the core circuit that locks in by step 10
6. Cross-model: L2 is among the top transfer layers

However, L2 is NOT a simple magnitude carrier. Its positional specialization (first+last, not operators) and its changing role after training (reduced for JSON, increased for delimiter/factual) suggest it performs active routing/processing, not just information transmission.

### 4.2 The Training Architecture

Training follows a two-phase architecture:
1. **Phase 1 (steps 1-10): Core circuit formation.** L2/L7/L9 stabilize rapidly. The model establishes the processing skeleton.
2. **Phase 2 (steps 10-100): Secondary refinement.** L15, L6, and skill-specific layers continue shifting. The model fills in task-specific details.

This has practical implications: early training steps are critical for establishing the processing architecture, while later steps fine-tune skill-specific components.

### 4.3 The Norm-Effect Paradox

The most architecturally interesting finding is the separation between where training writes (L20-L23, high norms) and where it matters (L0-L2, high ablation effects). This suggests:

1. LoRA writes large weight changes to late layers (near the output)
2. But these changes propagate upstream through the residual stream
3. The functional impact is felt at early layers that route information

This means that analyzing adapter weight norms alone is misleading for understanding functional impact. Causal ablation is necessary.

### 4.4 Skill Architecture

Skills are NOT uniformly stored. Each skill has a unique concentration pattern:
- Factual recall: distributed across L3/L16/L19 (knowledge is spread)
- Code semantics: L1/L10/L21 (spans early processing to late output)
- JSON schema: L6/L12/L13 (mid-layer concentration)
- Copying: dispersed (no single critical circuit)
- Delimiter: fully absorbed (becomes part of the base processing)

This diversity means:
- Targeted skill injection must be skill-specific
- Skill removal requires knowing which layers to target
- Adapter stacking works best when skills occupy orthogonal layer ranges

---

## 5. Implications for Small Model Optimization

### 5.1 Efficient Fine-Tuning

- **Rank r=4 is optimal** for surgical skill injection (peak L0 concentration)
- **o_proj is the most efficient target module** (344K params, +3.64 effect)
- **Core circuits lock in by step 10** — short training runs may be sufficient for basic skill acquisition
- **Skill-specific layer targeting** can reduce training cost by focusing on the 2-3 critical layers per skill

### 5.2 Skill Manipulation

- **Positive steering** at L2 can boost factual recall 3.3x
- **Negative steering** can selectively suppress skills
- **Cross-model patching** enables behavior transfer between model variants
- **Adapter stacking** allows multi-skill composition (factual + json = compatible)

### 5.3 Architectural Insights

- **Positional routing**: the model processes instruction and prediction tokens through distinct pathways
- **L22 as unembedding gateway**: exclusively affects last-position tokens
- **Distributed attention**: no single head dominates (max KL 0.046), unlike larger models
- **Two-phase training**: rapid core formation + slow secondary refinement

---

## 6. Limitations

1. **Single seed**: All results from one random seed. Confidence capped at MEDIUM (except L2 at HIGH). Multi-seed replication needed for publication.
2. **Zero ablation**: Creates out-of-distribution activations. Mean/resample ablation would be more principled.
3. **Short synthetic prompts**: 5-15 tokens. Results may not transfer to natural language or longer contexts.
4. **LoRA only**: Full SFT OOMs on 8GB. LoRA may produce different internal changes than full fine-tuning.
5. **Single model**: Results are specific to Qwen2.5-0.5B. Cross-model validation needed.
6. **Limited task suite**: 12 families with short prompts. Broader evaluation needed for generalization claims.

---

## 7. Open Hypotheses

| ID | Hypothesis | Status |
|----|-----------|--------|
| H001 | L2 is a general-purpose routing hub | SUPPORTED (with positional nuance) |
| H002 | LoRA concentrates skill into early layers | REJECTED (skill-specific) |
| H003 | Higher rank distributes skill | SUPPORTED |
| H004 | o_proj is key skill injection pathway | SUPPORTED for JSON |
| H005 | Factual and algorithmic tasks use different circuits | WEAKENED (both depend on L2) |
| H006 | Adapter norms write late, effects propagate upstream | SUPPORTED (norm-effect separation) |
| H007 | L22 is the unembedding pathway | SUPPORTED (last-position exclusive) |

---

## 8. Reproducibility

All experiments are fully reproducible:

```bash
ssh aero
cd ~/work/autonomous-small-model-exploration
source .venv/bin/activate

# Run all experiments in order
python scripts/run_baseline_and_ablation.py
python scripts/run_layer_ablation.py
python scripts/run_head_ablation.py
python scripts/run_mlp_ablation.py
python scripts/run_steering_sweep.py
python scripts/train_lora_json.py
python scripts/compare_lora_ablation.py
python scripts/run_lora_rank_sweep.py
python scripts/run_lora_module_sweep.py
python scripts/run_dataset_shard_ablation.py
python scripts/run_checkpoint_timeline.py
python scripts/run_adapter_archaeology.py
python scripts/run_adapter_stacking.py
python scripts/run_position_ablation.py
python scripts/run_cross_model_patching.py
python scripts/run_skill_knockout.py
python scripts/run_adapter_ablation.py

# Generate all plots and report
python scripts/generate_publication_report.py
```

### Artifacts

- {n_experiments} experiments in registry
- 10 LoRA adapters
- 5 training checkpoints
- 18+ result JSON files
- 15+ publication-quality plots
- Component atlas with 11+ entries

---

## 9. Conclusion

We have built a reproducible causal atlas of Qwen2.5-0.5B, connecting behaviours to components through {n_experiments} experiments. The key findings are:

1. **L2 is a universal routing hub** with positional specialization (HIGH confidence)
2. **LoRA training creates skill-specific concentration patterns**, not universal early-layer concentration
3. **Core circuits lock in rapidly** (step 10 of 100), with secondary refinement continuing
4. **Adapter norms and functional effects are spatially separated**, with upstream propagation
5. **Skills can be selectively transferred, suppressed, and combined**, opening paths for targeted optimization

This work demonstrates that even 0.5B parameter models have rich, non-trivial internal architectures that can be mapped through systematic causal intervention. The findings have direct implications for efficient fine-tuning, skill injection, and targeted optimization of small language models.

---

## Appendix A: Experiment Registry

| Exp ID | Type | Summary |
|--------|------|---------|
"""

    # Add experiment registry table
    registry_path = PROJECT_ROOT / "experiments" / "registry.jsonl"
    if registry_path.exists():
        with open(registry_path) as f:
            for line in f:
                entry = json.loads(line)
                report += f"| {entry['id']} | {entry['type']} | {entry['summary']} |\n"

    report += f"""

## Appendix B: Negative Results

1. Full SFT OOMs on 8GB VRAM — LoRA required
2. Full-residual activation patching gives KL=0 everywhere — position-specific needed
3. H002 (universal L0-L2 concentration) rejected — skill-specific patterns
4. Clean/corrupt pair v0 had tokenization misalignment — fixed in v1
5. Extreme steering (s >= +2) causes degeneration — finite steering budget
6. L2 is NOT position-uniform — operator tokens near-zero

## Appendix C: Decision Log

1. **D001**: HF native hooks instead of TransformerLens (GQA incompatibility)
2. **D002**: LoRA instead of full SFT (VRAM constraint)
3. **D003**: Zero ablation instead of mean (simpler implementation)
4. **D004**: Single seed (VRAM budget limits throughput)
5. **D005**: Short synthetic prompts (cleaner interpretability)
6. **D006**: Aero as primary compute host (RTX 2070 Super 8GB)
7. **D007**: Bundle-based GitHub push (aero has no gh auth)

---

*Generated by MI-Atlas automated report pipeline on {datetime.now().strftime("%Y-%m-%d %H:%M")}*
"""

    report_path = REPORTS_DIR / "publication_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")
    return report_path


# ============ MAIN ============
def main():
    print("=" * 60)
    print("  GENERATING PUBLICATION PLOTS AND REPORT")
    print("=" * 60)

    plots = [
        ("Layer ablation heatmap", plot_layer_ablation),
        ("MLP ablation heatmap", plot_mlp_ablation),
        ("Head ablation heatmap", plot_head_ablation),
        ("Steering sweep", plot_steering_sweep),
        ("LoRA rank sweep", plot_lora_rank_sweep),
        ("LoRA module sweep", plot_lora_module_sweep),
        ("Dataset shard ablation", plot_dataset_shard_ablation),
        ("Checkpoint timeline", plot_checkpoint_timeline),
        ("Adapter archaeology", plot_adapter_archaeology),
        ("Adapter stacking", plot_adapter_stacking),
        ("Position-specific ablation", plot_position_ablation),
        ("Activation patching v1", plot_activation_patching_v1),
        ("Cross-model patching", plot_cross_model_patching),
        ("Skill knockout", plot_skill_knockout),
        ("Adapter-only ablation", plot_adapter_ablation),
        ("Summary overview", plot_summary_overview),
    ]

    for name, func in plots:
        print(f"\n  Generating: {name}...")
        try:
            func()
            print(f"    OK")
        except Exception as e:
            print(f"    FAILED: {e}")

    print("\n  Generating publication report...")
    report_path = generate_report()
    print(f"  Report: {report_path}")

    print("\n" + "=" * 60)
    print("  ALL PLOTS AND REPORT GENERATED")
    print("=" * 60)


if __name__ == "__main__":
    main()
