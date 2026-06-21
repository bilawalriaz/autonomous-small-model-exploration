"""Run layer-level residual ablation."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from mi_atlas.model_loader import load_model
from mi_atlas.backend import create_backend
from mi_atlas.task_suite import TaskSuite
from mi_atlas.ablations import run_layer_ablation_suite
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.plotting import plot_ablation_heatmap
from mi_atlas.utils import save_json, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="Run layer ablation")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--suite", type=str, default=None)
    parser.add_argument("--ablation-type", type=str, default="mean",
                       choices=["mean", "zero", "resample"])
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    print("Loading model...")
    bundle = load_model(args.model, args.backend)
    backend = create_backend(bundle)
    print(f"  {bundle}")

    print("Loading task suite...")
    suite_path = args.suite or str(PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json")
    suite = TaskSuite.load(suite_path)

    print(f"Running layer ablation (type={args.ablation_type}, split={args.split})...")
    results = run_layer_ablation_suite(backend, suite, ablation_type=args.ablation_type, split=args.split)

    # Save
    output_path = PROJECT_ROOT / "experiments" / "results" / f"layer_ablation_{args.ablation_type}.json"
    save_json(results, output_path)
    print(f"  Results saved to {output_path}")

    # Plot
    effect_matrix = np.array(results["effect_matrix"])
    plot_path = plot_ablation_heatmap(
        effect_matrix,
        row_labels=results["layer_names"],
        col_labels=results["families"],
        title=f"Layer Ablation Heatmap ({args.ablation_type})",
        name=f"layer_ablation_heatmap_{args.ablation_type}",
    )
    print(f"  Plot saved to {plot_path}")

    # Register
    register_experiment(
        type="ablation",
        model=bundle.model_name,
        backend=bundle.backend,
        config="config/experiment_plan.yaml",
        inputs=[suite_path],
        outputs=[str(output_path), str(plot_path)],
        status="success",
        summary=f"Layer {args.ablation_type} ablation across {results['n_layers']} layers",
        next="Run head and MLP ablation",
    )
    print("  Experiment registered.")


if __name__ == "__main__":
    main()
