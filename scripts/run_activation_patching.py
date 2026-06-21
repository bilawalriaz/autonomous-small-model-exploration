"""Run activation patching on clean/corrupt pairs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from mi_atlas.model_loader import load_model
from mi_atlas.backend import create_backend
from mi_atlas.task_suite import TaskSuite
from mi_atlas.patching import run_patching_suite
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, load_json, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--pairs", type=str, default=None)
    args = parser.parse_args()

    bundle = load_model(args.model)
    backend = create_backend(bundle)

    pairs_path = args.pairs or str(PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v0.json")
    raw_pairs = load_json(pairs_path)

    results = run_patching_suite(backend, raw_pairs)

    output_path = PROJECT_ROOT / "experiments" / "results" / "activation_patching.json"
    save_json(results, output_path)

    register_experiment(type="patching", model=bundle.model_name, backend=bundle.backend,
                       config="config/experiment_plan.yaml", inputs=[pairs_path],
                       outputs=[str(output_path)], status="success",
                       summary=f"Patching: {results['n_pairs']} pairs x {results['n_components']} components",
                       next="Path patching on top components")
    print("Activation patching complete.")


if __name__ == "__main__":
    main()
