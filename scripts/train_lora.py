"""Train LoRA adapter on the target model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from mi_atlas.model_loader import load_model_hf
from mi_atlas.task_suite import build_default_suite
from mi_atlas.training.datasets import prepare_sft_dataset, split_dataset
from mi_atlas.training.lora import train_lora
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    bundle = load_model_hf(args.model)
    output_dir = args.output or str(PROJECT_ROOT / "experiments" / "adapters" / f"lora_r{args.rank}")

    suite = build_default_suite()
    dataset = prepare_sft_dataset(suite)
    splits = split_dataset(dataset)

    print(f"Training LoRA rank={args.rank}: {len(splits['train'])} train examples")
    result = train_lora(bundle.model, bundle.tokenizer, splits["train"], output_dir, rank=args.rank)

    register_experiment(type="training", model=bundle.model_name, backend="hf",
                       config="config/training_plan.yaml", inputs=[],
                       outputs=[result["adapter_path"]], status="success",
                       summary=f"LoRA r={args.rank}: loss={result['train_loss']:.4f}",
                       next="Compare checkpoints")
    print("LoRA training complete.")


if __name__ == "__main__":
    main()
