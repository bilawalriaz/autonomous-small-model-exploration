"""Run training hyperparameter sweep."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from mi_atlas.model_loader import load_model_hf
from mi_atlas.task_suite import build_default_suite
from mi_atlas.training.datasets import prepare_sft_dataset, split_dataset
from mi_atlas.training.lora import train_lora
from mi_atlas.training.hyperparam_sweeps import get_sweep_plan
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default="lora_rank")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    plan = get_sweep_plan(args.sweep)
    bundle = load_model_hf(args.model)
    suite = build_default_suite()
    dataset = prepare_sft_dataset(suite)
    splits = split_dataset(dataset)

    results = []
    for value in plan["values"]:
        print(f"Sweep: {plan['param']}={value}")
        overrides = dict(plan["fixed"])
        overrides[plan["param"]] = value
        try:
            output_dir = str(PROJECT_ROOT / "experiments" / "adapters" / f"sweep_{args.sweep}_{value}")
            result = train_lora(bundle.model, bundle.tokenizer, splits["train"], output_dir,
                               rank=overrides.get("rank", 8), config_override=overrides)
            results.append({"value": value, "status": "success", "loss": result["train_loss"]})
        except Exception as e:
            results.append({"value": value, "status": "failed", "error": str(e)})

    output_path = PROJECT_ROOT / "experiments" / "results" / f"sweep_{args.sweep}.json"
    save_json({"sweep": args.sweep, "results": results}, output_path)

    register_experiment(type="training", model=bundle.model_name, backend="hf",
                       config="config/training_plan.yaml", inputs=[],
                       outputs=[str(output_path)], status="success",
                       summary=f"Sweep {args.sweep}: {len(results)} runs",
                       next="Compare results")
    print("Sweep complete.")


if __name__ == "__main__":
    main()
