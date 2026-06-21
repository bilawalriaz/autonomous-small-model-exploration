"""Tests for training data preparation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from mi_atlas.task_suite import build_default_suite
from mi_atlas.training.datasets import (
    task_suite_to_hf_dataset, prepare_sft_dataset,
)


class TestTrainingData:
    def test_suite_to_hf(self):
        suite = build_default_suite()
        ds = task_suite_to_hf_dataset(suite)
        assert len(ds) == len(suite)
        assert "prompt" in ds.column_names
        assert "target" in ds.column_names

    def test_sft_dataset(self):
        suite = build_default_suite()
        ds = prepare_sft_dataset(suite)
        assert len(ds) == len(suite)
        assert "text" in ds.column_names
        # Check that text contains prompt and target
        sample = ds[0]["text"]
        assert len(sample) > 0
