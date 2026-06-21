"""Supervised fine-tuning (SFT)."""

from pathlib import Path
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

from ..utils import load_config, PROJECT_ROOT


def train_sft(
    model,
    tokenizer,
    dataset: Dataset,
    output_dir: str | Path,
    config_override: dict | None = None,
) -> dict:
    """Run SFT training.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        dataset: Training dataset with 'text' field
        output_dir: Where to save checkpoints
        config_override: Override default SFT config

    Returns:
        dict with training stats
    """
    config = load_config("training_plan")
    sft_config = config.get("sft", {})
    if config_override:
        sft_config.update(config_override)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=sft_config.get("learning_rate", 2e-5),
        per_device_train_batch_size=sft_config.get("batch_size", 4),
        gradient_accumulation_steps=sft_config.get("gradient_accumulation_steps", 4),
        max_steps=sft_config.get("max_steps", 500),
        warmup_steps=sft_config.get("warmup_steps", 25),
        weight_decay=sft_config.get("weight_decay", 0.01),
        save_steps=sft_config.get("save_steps", 100),
        eval_steps=sft_config.get("eval_steps", 50),
        logging_steps=sft_config.get("logging_steps", 10),
        fp16=sft_config.get("fp16", False),
        bf16=sft_config.get("bf16", True),
        seed=sft_config.get("seed", 42),
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    result = trainer.train()

    return {
        "output_dir": str(output_dir),
        "train_loss": result.training_loss,
        "train_steps": result.global_step,
        "metrics": result.metrics,
    }
