"""LoRA adapter training."""

from pathlib import Path
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
from datasets import Dataset

from ..utils import load_config, PROJECT_ROOT


def create_lora_config(
    rank: int = 8,
    target_modules: list[str] | None = None,
    alpha: int = 16,
    dropout: float = 0.05,
) -> LoraConfig:
    """Create a LoRA configuration."""
    config = load_config("training_plan")
    lora_config = config.get("lora", {})

    return LoraConfig(
        r=rank,
        lora_alpha=alpha or lora_config.get("alpha", 16),
        target_modules=target_modules or lora_config.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=dropout or lora_config.get("dropout", 0.05),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def train_lora(
    model,
    tokenizer,
    dataset: Dataset,
    output_dir: str | Path,
    rank: int = 8,
    target_modules: list[str] | None = None,
    config_override: dict | None = None,
) -> dict:
    """Train a LoRA adapter.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        dataset: Training dataset with 'text' field
        output_dir: Where to save adapter
        rank: LoRA rank
        target_modules: Which modules to target
        config_override: Override default config

    Returns:
        dict with training stats and adapter path
    """
    config = load_config("training_plan")
    lora_config = config.get("lora", {})
    if config_override:
        lora_config.update(config_override)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create LoRA model
    peft_config = create_lora_config(
        rank=rank,
        target_modules=target_modules,
        alpha=lora_config.get("alpha", 16),
        dropout=lora_config.get("dropout", 0.05),
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=lora_config.get("max_seq_length", 512),
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=lora_config.get("learning_rate", 2e-4),
        per_device_train_batch_size=lora_config.get("batch_size", 4),
        max_steps=lora_config.get("max_steps", 500),
        warmup_steps=lora_config.get("warmup_steps", 25),
        save_steps=100,
        logging_steps=10,
        fp16=False,
        bf16=True,
        seed=lora_config.get("seed", 42),
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized,
        processing_class=tokenizer,
    )

    result = trainer.train()

    # Save adapter
    adapter_path = output_dir / "adapter"
    peft_model.save_pretrained(str(adapter_path))

    return {
        "output_dir": str(output_dir),
        "adapter_path": str(adapter_path),
        "rank": rank,
        "target_modules": target_modules,
        "train_loss": result.training_loss,
        "train_steps": result.global_step,
        "metrics": result.metrics,
    }
