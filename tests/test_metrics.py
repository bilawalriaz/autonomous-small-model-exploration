"""Tests for metrics module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from mi_atlas.metrics import (
    exact_match_score, edit_distance, edit_distance_score,
    valid_json_score, required_json_keys, ast_parse_success,
    token_entropy, top_token_concentration, hallucination_flag,
    ablation_effect, normalized_recovery, patch_score,
    skill_delta, adapter_specificity, localization_score,
)


class TestBasicMetrics:
    def test_exact_match(self):
        assert exact_match_score("hello", "hello") == 1.0
        assert exact_match_score("hello", "world") == 0.0
        assert exact_match_score("  hello  ", "hello") == 1.0

    def test_edit_distance(self):
        assert edit_distance("hello", "hello") == 0
        assert edit_distance("hello", "helo") == 1
        assert edit_distance("abc", "xyz") == 3

    def test_edit_distance_score(self):
        assert edit_distance_score("hello", "hello") == 1.0
        assert edit_distance_score("hello", "helo") < 1.0
        assert edit_distance_score("hello", "helo") > 0.0

    def test_valid_json(self):
        assert valid_json_score('{"a": 1}') == 1.0
        assert valid_json_score("not json") == 0.0
        assert valid_json_score("") == 0.0
        assert valid_json_score("[]") == 1.0

    def test_required_keys(self):
        assert required_json_keys('{"a": 1, "b": 2}', ["a", "b"]) == 1.0
        assert required_json_keys('{"a": 1}', ["a", "b"]) == 0.5
        assert required_json_keys("not json", ["a"]) == 0.0

    def test_ast_parse(self):
        assert ast_parse_success("x = 1") == 1.0
        assert ast_parse_success("def f():\n    pass") == 1.0
        assert ast_parse_success("def f(\n    pass") == 0.0


class TestTensorMetrics:
    def test_token_entropy(self):
        low_entropy = torch.tensor([100.0, 0.0, 0.0])  # Very peaked
        high_entropy = torch.tensor([1.0, 1.0, 1.0])    # Uniform
        assert token_entropy(low_entropy) < token_entropy(high_entropy)

    def test_top_concentration(self):
        concentrated = torch.tensor([100.0, 0.0, 0.0, 0.0, 0.0])
        assert top_token_concentration(concentrated, k=1) > 0.99

    def test_hallucination_flag(self):
        assert hallucination_flag("I don't know the answer") == 1.0
        assert hallucination_flag("The population was 50,000") == 0.0
        assert hallucination_flag("unknown") == 1.0


class TestPatchingMetrics:
    def test_ablation_effect(self):
        assert ablation_effect(1.0, 0.5) == 0.5
        assert ablation_effect(0.5, 0.5) == 0.0

    def test_normalized_recovery(self):
        # Clean=1.0, Corrupt=0.0, Patched=0.6 => recovery=0.6
        assert abs(normalized_recovery(0.6, 0.0, 1.0) - 0.6) < 1e-6
        # Full recovery
        assert abs(normalized_recovery(1.0, 0.0, 1.0) - 1.0) < 1e-6
        # No recovery
        assert abs(normalized_recovery(0.0, 0.0, 1.0) - 0.0) < 1e-6

    def test_patch_score(self):
        assert patch_score(0.6, 0.0) == 0.6

    def test_skill_delta(self):
        assert skill_delta(1.0, 0.5) == 0.5

    def test_adapter_specificity(self):
        assert adapter_specificity(0.8, [0.1, 0.2, 0.1]) > 0.5

    def test_localization_score(self):
        score = localization_score(1.0, [0.1, 0.1, 0.1])
        assert score > 3.0
