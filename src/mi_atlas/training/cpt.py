"""Continued pretraining (CPT)."""

from pathlib import Path
from transformers import TrainingArguments, Trainer
from datasets import Dataset

from ..utils import load_config, PROJECT_ROOT


def train_cpt(
    model,
    tokenizer,
    dataset: Dataset,
    output_dir: str | Path,
    config_override: dict | None = None,
) -> dict:
    """Run continued pretraining.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        dataset: Training dataset with 'text' field
        output_dir: Where to save checkpoints
        config_override: Override default CPT config

    Returns:
        dict with training stats
    """
    config = load_config("training_plan")
    cpt_config = config.get("cpt", {})
    if config_override:
        cpt_config.update(config_override)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=cpt_config.get("max_seq_length", 512),
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=cpt_config.get("learning_rate", 1e-4),
        per_device_train_batch_size=cpt_config.get("batch_size", 4),
        gradient_accumulation_steps=cpt_config.get("gradient_accumulation_steps", 4),
        max_steps=cpt_config.get("max_steps", 1000),
        warmup_steps=cpt_config.get("warmup_steps", 50),
        weight_decay=cpt_config.get("weight_decay", 0.01),
        save_steps=cpt_config.get("save_steps", 250),
        eval_steps=cpt_config.get("eval_steps", 100),
        logging_steps=cpt_config.get("logging_steps", 10),
        fp16=cpt_config.get("fp16", False),
        bf16=cpt_config.get("bf16", True),
        seed=cpt_config.get("seed", 42),
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        processing_class=tokenizer,
    )

    result = trainer.train()

    return {
        "output_dir": str(output_dir),
        "train_loss": result.training_loss,
        "train_steps": result.global_step,
        "metrics": result.metrics,
    }
