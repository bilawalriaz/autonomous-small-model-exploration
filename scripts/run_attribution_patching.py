"""Run attribution patching (gradient-based approximation)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import torch
from mi_atlas.model_loader import load_model
from mi_atlas.backend import create_backend
from mi_atlas.attribution import compute_gradient_attribution
from mi_atlas.task_suite import TaskSuite
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

    results = []
    for ex in list(suite)[:20]:  # Limit for speed
        inputs = backend.tokenize(ex.clean_prompt)
        input_ids = inputs["input_ids"].to(backend.device)
        target_ids = backend.tokenizer.encode(ex.target, add_special_tokens=False)
        if target_ids:
            attr = compute_gradient_attribution(backend.model, input_ids, target_ids[0])
            results.append({"example_id": ex.id, "family": ex.family, "attribution_norm": attr.norm().item()})

    output_path = PROJECT_ROOT / "experiments" / "results" / "attribution_patching.json"
    save_json(results, output_path)

    register_experiment(type="patching", model=bundle.model_name, backend=bundle.backend,
                       config="config/experiment_plan.yaml", inputs=[suite_path],
                       outputs=[str(output_path)], status="success",
                       summary=f"Attribution patching on {len(results)} examples",
                       next="Steering sweeps")
    print("Attribution patching complete.")


if __name__ == "__main__":
    main()
