"""Training dataset preparation."""

from pathlib import Path
from datasets import Dataset, DatasetDict

from ..task_suite import TaskSuite, TaskExample
from ..utils import save_json, load_json, PROJECT_ROOT


def task_suite_to_hf_dataset(suite: TaskSuite) -> Dataset:
    """Convert a TaskSuite to a Hugging Face Dataset."""
    records = []
    for ex in suite.examples:
        records.append({
            "id": ex.id,
            "family": ex.family,
            "prompt": ex.clean_prompt,
            "target": ex.target,
            "metric_type": ex.metric_type,
            "split": ex.split,
        })
    return Dataset.from_list(records)


def prepare_sft_dataset(
    suite: TaskSuite,
    response_template: str = "### Answer:\n",
) -> Dataset:
    """Prepare dataset for SFT training.

    Format: prompt + response_template + target
    """
    records = []
    for ex in suite.examples:
        text = ex.clean_prompt + response_template + ex.target
        records.append({
            "text": text,
            "family": ex.family,
            "id": ex.id,
        })
    return Dataset.from_list(records)


def prepare_cpt_dataset(
    text_chunks: list[str],
    families: list[str] | None = None,
) -> Dataset:
    """Prepare dataset for continued pretraining."""
    records = []
    for i, chunk in enumerate(text_chunks):
        record = {"text": chunk, "id": f"cpt_{i:06d}"}
        if families and i < len(families):
            record["family"] = families[i]
        records.append(record)
    return Dataset.from_list(records)


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    heldout_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """Split dataset into train/val/test/heldout."""
    n = len(dataset)
    shuffled = dataset.shuffle(seed=seed)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    test_end = val_end + int(n * test_ratio)

    return DatasetDict({
        "train": shuffled.select(range(train_end)),
        "val": shuffled.select(range(train_end, val_end)),
        "test": shuffled.select(range(val_end, test_end)),
        "heldout": shuffled.select(range(test_end, n)),
    })
