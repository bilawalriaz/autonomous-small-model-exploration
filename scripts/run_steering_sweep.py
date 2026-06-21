"""Run steering vector sweeps."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from mi_atlas.model_loader import load_model
from mi_atlas.backend import create_backend
from mi_atlas.steering import compute_steering_vector, steering_strength_sweep
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--layer", type=str, default="blocks.0.hook_resid_post")
    parser.add_argument("--family", type=str, default="json_schema")
    args = parser.parse_args()

    bundle = load_model(args.model)
    backend = create_backend(bundle)

    # Example: steer toward valid JSON
    pos_prompts = [
        'Return valid JSON: {"name": "Alice", "age": 31}',
        'Return valid JSON: {"x": 1, "y": 2}',
    ]
    neg_prompts = [
        "Tell me about Alice who is 31.",
        "What are x and y?",
    ]

    print(f"Computing steering vector for {args.family} at {args.layer}...")
    sv = compute_steering_vector(backend, pos_prompts, neg_prompts, args.layer)

    print("Running strength sweep...")
    prompt = 'Return exactly valid JSON with keys name and age. Bob is 25.\n'
    results = steering_strength_sweep(backend, prompt, args.layer, sv)

    output_path = PROJECT_ROOT / "experiments" / "results" / f"steering_{args.family}.json"
    save_json({"family": args.family, "layer": args.layer, "sweep": [
        {"strength": r.get("strength"), "status": r.get("status")} for r in results
    ]}, output_path)

    register_experiment(type="steering", model=bundle.model_name, backend=bundle.backend,
                       config="config/experiment_plan.yaml", inputs=[],
                       outputs=[str(output_path)], status="success",
                       summary=f"Steering sweep for {args.family} at {args.layer}",
                       next="Probes")
    print("Steering sweep complete.")


if __name__ == "__main__":
    main()
