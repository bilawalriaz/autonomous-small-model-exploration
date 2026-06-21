"""Compare LoRA adapters: interference, specificity, merging."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import glob
from mi_atlas.training.adapter_analysis import adapter_norms_by_layer, effective_rank
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters", type=str, nargs="+", default=None)
    args = parser.parse_args()

    adapter_paths = args.adapters or glob.glob(str(PROJECT_ROOT / "experiments" / "adapters" / "*/adapter"))

    results = []
    for path in adapter_paths:
        try:
            weights = __import__("mi_atlas.training.adapter_analysis", fromlist=["load_adapter_weights"]).load_adapter_weights(path)
            norms = adapter_norms_by_layer(weights)
            ranks = effective_rank(weights)
            results.append({"path": path, "layer_norms": norms, "effective_ranks": ranks})
        except Exception as e:
            results.append({"path": path, "error": str(e)})

    output_path = PROJECT_ROOT / "experiments" / "results" / "adapter_comparison.json"
    save_json(results, output_path)
    register_experiment(type="comparison", model="adapters", backend="hf",
                       config="config/training_plan.yaml", inputs=adapter_paths,
                       outputs=[str(output_path)], status="success",
                       summary=f"Compared {len(adapter_paths)} adapters", next="SAE training")
    print("Adapter comparison complete.")


if __name__ == "__main__":
    main()
