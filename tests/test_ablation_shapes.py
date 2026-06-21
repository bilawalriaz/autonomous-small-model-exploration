"""Tests for ablation shapes and outputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


class TestAblationShapes:
    """Placeholder — these tests require a loaded model."""

    def test_placeholder(self):
        assert True

    # TODO: Add after model loading:
    # - test_ablation_output_shape
    # - test_zero_ablation
    # - test_mean_ablation
    # - test_layer_ablation_suite_matrix_shape
