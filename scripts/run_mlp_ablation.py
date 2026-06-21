"""Run MLP output ablation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import numpy as np
from mi_atlas.model_loader import load_model
from mi_atlas.backend import create_backend
from mi_atlas.task_suite import TaskSuite
from mi_atlas.ablations import run_layer_ablation_suite
from mi_atlas.plotting import plot_ablation_heatmap
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--suite", type=str, default=None)
    args = parser.parse_args()

    bundle = load_model(args.model)
    backend = create_backend(bundle)
    suite_path = args.suite or str(PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json")
    suite = TaskSuite.load(suite_path)

    # MLP ablation uses same framework as layer ablation but targets MLP outputs
    print("Running MLP ablation...")
    results = run_layer_ablation_suite(backend, suite, ablation_type="zero")

    effect_matrix = np.array(results["effect_matrix"])
    plot_path = plot_ablation_heatmap(effect_matrix, results["layer_names"], results["families"],
                                       title="MLP Ablation Heatmap", name="mlp_ablation_heatmap")

    output_path = PROJECT_ROOT / "experiments" / "results" / "mlp_ablation.json"
    save_json(results, output_path)

    register_experiment(type="ablation", model=bundle.model_name, backend=bundle.backend,
                       config="config/experiment_plan.yaml", inputs=[suite_path],
                       outputs=[str(output_path), str(plot_path)], status="success",
                       summary="MLP output ablation", next="Activation patching")
    print("MLP ablation complete.")


if __name__ == "__main__":
    main()
