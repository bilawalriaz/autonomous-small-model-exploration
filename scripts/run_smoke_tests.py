"""Smoke tests: verify repo scaffold, imports, and basic functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that all core modules import."""
    from mi_atlas import utils, model_loader, backend, tokenization
    from mi_atlas import task_suite, metrics, eval_runner, activation_cache
    from mi_atlas import ablations, patching, attribution, steering
    from mi_atlas import probes, component_atlas, plotting, experiment_registry
    from mi_atlas import report_writer
    from mi_atlas.training import datasets, cpt, sft, lora, curricula
    from mi_atlas.training import checkpoint_eval, adapter_analysis, hyperparam_sweeps
    from mi_atlas.comparisons import checkpoint_diff, activation_diff, weight_delta
    from mi_atlas.comparisons import cka, svcca, skill_localization
    print("  All imports OK")


def test_config_loading():
    """Test that all config files load."""
    from mi_atlas.utils import load_config
    for name in ["model", "tasks", "experiment_plan", "training_plan", "thresholds", "plotting", "compute"]:
        cfg = load_config(name)
        assert isinstance(cfg, dict), f"Config {name} did not load as dict"
    print("  All configs load OK")


def test_task_suite_generation():
    """Test task suite generation produces expected count."""
    from mi_atlas.task_suite import build_default_suite
    suite = build_default_suite()
    summary = suite.summary()
    assert summary["total"] >= 50, f"Expected >= 50 examples, got {summary['total']}"
    assert len(summary["families"]) >= 5, f"Expected >= 5 families, got {len(summary['families'])}"
    print(f"  Task suite: {summary['total']} examples, {len(summary['families'])} families")
    for fam, count in summary["families"].items():
        print(f"    {fam}: {count}")


def test_task_suite_save_load():
    """Test save/load roundtrip."""
    from mi_atlas.task_suite import build_default_suite, TaskSuite
    import tempfile
    suite = build_default_suite()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        suite.save(f.name)
        loaded = TaskSuite.load(f.name)
        assert len(loaded) == len(suite)
        assert loaded.families == suite.families
    print("  Task suite save/load OK")


def test_metrics():
    """Test metric functions."""
    from mi_atlas.metrics import (
        exact_match_score, edit_distance_score, valid_json_score,
        required_json_keys, token_entropy, top_token_concentration,
    )
    import torch

    assert exact_match_score("hello", "hello") == 1.0
    assert exact_match_score("hello", "world") == 0.0
    assert edit_distance_score("hello", "hello") == 1.0
    assert edit_distance_score("hello", "helo") < 1.0
    assert valid_json_score('{"a": 1}') == 1.0
    assert valid_json_score("not json") == 0.0
    assert required_json_keys('{"a": 1, "b": 2}', ["a", "b"]) == 1.0
    assert required_json_keys('{"a": 1}', ["a", "b"]) == 0.5

    # Test entropy
    # [1.0, 0.0, 0.0] softmax = [0.576, 0.212, 0.212] — not one-hot
    logits = torch.tensor([100.0, 0.0, 0.0])  # Very peaked after softmax
    assert token_entropy(logits) < 0.01
    logits_uniform = torch.tensor([1.0, 1.0, 1.0])
    assert token_entropy(logits_uniform) > token_entropy(logits)
    print("  Metrics OK")


def test_experiment_registry():
    """Test experiment registry."""
    from mi_atlas.experiment_registry import next_experiment_id
    exp_id = next_experiment_id()
    assert exp_id.startswith("exp_")
    print(f"  Registry OK (next ID: {exp_id})")


def test_directory_structure():
    """Test that expected directories exist."""
    root = Path(__file__).parent.parent
    expected_dirs = [
        "config", "data/prompts", "data/clean_corrupt_pairs",
        "data/generated", "data/training_corpora", "data/eval_sets",
        "data/cached_activations", "data/cached_logits", "data/metadata",
        "src/mi_atlas", "src/mi_atlas/training", "src/mi_atlas/comparisons",
        "scripts", "experiments", "experiments/results", "experiments/plots",
        "experiments/tables", "experiments/adapters", "experiments/checkpoints",
        "reports", "tests",
    ]
    for d in expected_dirs:
        assert (root / d).exists(), f"Missing directory: {d}"
    print("  Directory structure OK")


def main():
    print("MI-Atlas Smoke Tests")
    print("=" * 40)

    tests = [
        ("Directory structure", test_directory_structure),
        ("Config loading", test_config_loading),
        ("Imports", test_imports),
        ("Task suite generation", test_task_suite_generation),
        ("Task suite save/load", test_task_suite_save_load),
        ("Metrics", test_metrics),
        ("Experiment registry", test_experiment_registry),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"\n[TEST] {name}")
            fn()
            print(f"  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("All smoke tests passed!")


if __name__ == "__main__":
    main()
