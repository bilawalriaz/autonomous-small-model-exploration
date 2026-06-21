"""Compare checkpoints: eval multiple and produce comparison table."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import glob
from mi_atlas.task_suite import TaskSuite
from mi_atlas.training.checkpoint_eval import evaluate_checkpoints
from mi_atlas.comparisons.checkpoint_diff import compare_checkpoint_metrics
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None)
    parser.add_argument("--suite", type=str, default=None)
    args = parser.parse_args()

    ckpt_paths = args.checkpoints or glob.glob(str(PROJECT_ROOT / "experiments" / "checkpoints" / "*"))
    suite_path = args.suite or str(PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json")
    suite = TaskSuite.load(suite_path)

    results = evaluate_checkpoints(ckpt_paths, suite)
    comparison = compare_checkpoint_metrics(results)

    output_path = PROJECT_ROOT / "experiments" / "results" / "checkpoint_comparison.json"
    save_json(comparison, output_path)

    register_experiment(type="comparison", model="multiple", backend="hf",
                       config="config/experiment_plan.yaml", inputs=ckpt_paths,
                       outputs=[str(output_path)], status="success",
                       summary=f"Compared {len(ckpt_paths)} checkpoints",
                       next="Adapter comparison")
    print("Checkpoint comparison complete.")


if __name__ == "__main__":
    main()
