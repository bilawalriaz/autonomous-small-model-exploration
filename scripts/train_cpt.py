"""Train continued pretraining (CPT) on the target model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from mi_atlas.model_loader import load_model_hf
from mi_atlas.task_suite import TaskSuite, build_default_suite
from mi_atlas.training.datasets import prepare_cpt_dataset, split_dataset
from mi_atlas.training.cpt import train_cpt
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--corpus", type=str, default=None, help="Path to CPT corpus JSON")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    bundle = load_model_hf(args.model)
    output_dir = args.output or str(PROJECT_ROOT / "experiments" / "checkpoints" / "cpt_default")

    # Build CPT corpus from task suite prompts if no external corpus
    if args.corpus:
        import json
        with open(args.corpus) as f:
            chunks = json.load(f)
    else:
        suite = build_default_suite()
        chunks = [ex.clean_prompt + ex.target for ex in suite]

    dataset = prepare_cpt_dataset(chunks)
    splits = split_dataset(dataset)

    print(f"Training CPT: {len(splits['train'])} train examples")
    result = train_cpt(bundle.model, bundle.tokenizer, splits["train"], output_dir)

    register_experiment(type="training", model=bundle.model_name, backend="hf",
                       config="config/training_plan.yaml", inputs=[],
                       outputs=[output_dir], status="success",
                       summary=f"CPT: loss={result['train_loss']:.4f}", next="SFT")
    print("CPT complete.")


if __name__ == "__main__":
    main()
