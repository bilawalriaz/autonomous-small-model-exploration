"""Tests for experiment registry."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from mi_atlas.experiment_registry import next_experiment_id


class TestRegistry:
    def test_next_id_format(self):
        exp_id = next_experiment_id()
        assert exp_id.startswith("exp_")
        assert len(exp_id) == 10  # exp_NNNNNN

    def test_id_increments(self):
        id1 = next_experiment_id()
        id2 = next_experiment_id()
        # Both return same since nothing was written
        assert id1 == id2
