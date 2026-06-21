"""Train SFT on the target model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from mi_atlas.model_loader import load_model_hf
from mi_atlas.task_suite import build_default_suite
from mi_atlas.training.datasets import prepare_sft_dataset, split_dataset
from mi_atlas.training.sft import train_sft
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    bundle = load_model_hf(args.model)
    output_dir = args.output or str(PROJECT_ROOT / "experiments" / "checkpoints" / "sft_default")

    suite = build_default_suite()
    dataset = prepare_sft_dataset(suite)
    splits = split_dataset(dataset)

    print(f"Training SFT: {len(splits['train'])} train examples")
    result = train_sft(bundle.model, bundle.tokenizer, splits["train"], output_dir)

    register_experiment(type="training", model=bundle.model_name, backend="hf",
                       config="config/training_plan.yaml", inputs=[],
                       outputs=[output_dir], status="success",
                       summary=f"SFT: loss={result['train_loss']:.4f}", next="LoRA")
    print("SFT complete.")


if __name__ == "__main__":
    main()
