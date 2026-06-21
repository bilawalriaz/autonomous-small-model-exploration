"""Run baseline evaluation on the target model."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mi_atlas.model_loader import load_model
from mi_atlas.backend import create_backend
from mi_atlas.task_suite import TaskSuite
from mi_atlas.eval_runner import evaluate_suite
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.plotting import plot_task_scores
from mi_atlas.utils import save_json, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--backend", type=str, default=None, help="Backend to use")
    parser.add_argument("--suite", type=str, default=None, help="Path to task suite JSON")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate")
    args = parser.parse_args()

    print("Loading model...")
    bundle = load_model(args.model, args.backend)
    backend = create_backend(bundle)
    print(f"  {bundle}")

    print("Loading task suite...")
    suite_path = args.suite or str(PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json")
    suite = TaskSuite.load(suite_path)
    print(f"  {suite.summary()}")

    print(f"Running baseline evaluation (split={args.split})...")
    results = evaluate_suite(backend, suite, split=args.split)

    # Save results
    output_path = PROJECT_ROOT / "experiments" / "results" / "baseline_eval.json"
    save_json(results, output_path)
    print(f"  Results saved to {output_path}")

    # Print summary
    print("\nBaseline Results Summary:")
    print(f"  Total examples: {results['summary']['total_examples']}")
    print(f"  Errors: {results['summary']['errors']}")
    print(f"  Overall mean: {results['summary']['overall_mean']:.3f}")
    print("\n  Per-family scores:")
    for fam, score in results["summary"]["primary_metric_by_family"].items():
        print(f"    {fam}: {score:.3f}")

    # Generate plot
    if results["summary"]["primary_metric_by_family"]:
        plot_path = plot_task_scores(
            results["summary"]["primary_metric_by_family"],
            title="Baseline Task Scores",
        )
        print(f"\n  Plot saved to {plot_path}")

    # Register experiment
    register_experiment(
        type="baseline",
        model=bundle.model_name,
        backend=bundle.backend,
        config="config/model.yaml",
        inputs=[suite_path],
        outputs=[str(output_path)],
        status="success",
        summary=f"Baseline eval: overall mean={results['summary']['overall_mean']:.3f}",
        key_metrics=results["summary"]["primary_metric_by_family"],
        next="Run layer-level residual ablation",
    )
    print("\n  Experiment registered.")


if __name__ == "__main__":
    main()
