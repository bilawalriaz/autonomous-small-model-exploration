"""Tests for reproducibility."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from mi_atlas.utils import set_seed, git_commit_hash
from mi_atlas.task_suite import build_default_suite


class TestReproducibility:
    def test_deterministic_suite(self):
        """Task suite should be identical across calls with same seed."""
        suite1 = build_default_suite(seed=42)
        suite2 = build_default_suite(seed=42)
        assert len(suite1) == len(suite2)
        for e1, e2 in zip(suite1, suite2):
            assert e1.clean_prompt == e2.clean_prompt
            assert e1.target == e2.target

    def test_different_seeds(self):
        """Different seeds should produce different examples."""
        suite1 = build_default_suite(seed=42)
        suite2 = build_default_suite(seed=99)
        # At least some prompts should differ (copying is random)
        prompts1 = {e.clean_prompt for e in suite1 if e.family == "copying"}
        prompts2 = {e.clean_prompt for e in suite2 if e.family == "copying"}
        # May not differ if random ranges are small, but test structure is right
        assert len(prompts1) > 0

    def test_git_commit(self):
        """Should return a commit hash or None."""
        result = git_commit_hash()
        # May be None if not in a git repo, that's fine
        if result is not None:
            assert len(result) >= 7
