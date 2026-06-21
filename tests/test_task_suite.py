"""Tests for task suite."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from mi_atlas.task_suite import (
    build_default_suite, TaskExample, TaskSuite,
    generate_copying_examples, generate_delimiter_examples,
    generate_json_examples, generate_factual_examples,
)


class TestTaskSuite:
    def test_build_default(self):
        suite = build_default_suite()
        assert len(suite) >= 50
        assert len(suite.families) >= 5

    def test_families_present(self):
        suite = build_default_suite()
        expected = {"copying", "delimiter_tracking", "json_schema", "factual_recall", "arithmetic"}
        assert expected.issubset(set(suite.families))

    def test_filter_by_family(self):
        suite = build_default_suite()
        copying = suite.filter_by_family("copying")
        assert len(copying) > 0
        assert all(e.family == "copying" for e in copying)

    def test_filter_by_split(self):
        suite = build_default_suite()
        train = suite.filter_by_split("train")
        test = suite.filter_by_split("test")
        assert len(train) > 0
        assert len(test) > 0

    def test_save_load_roundtrip(self, tmp_path):
        suite = build_default_suite()
        path = tmp_path / "suite.json"
        suite.save(path)
        loaded = TaskSuite.load(path)
        assert len(loaded) == len(suite)
        assert loaded.families == suite.families

    def test_summary(self):
        suite = build_default_suite()
        s = suite.summary()
        assert "total" in s
        assert "families" in s
        assert "splits" in s


class TestGenerators:
    def test_copying(self):
        examples = generate_copying_examples(n=5)
        assert len(examples) == 5
        assert all(e.family == "copying" for e in examples)
        assert all(e.target for e in examples)

    def test_delimiter(self):
        examples = generate_delimiter_examples(n=5)
        assert len(examples) == 5
        assert all(e.family == "delimiter_tracking" for e in examples)

    def test_json(self):
        examples = generate_json_examples(n=5)
        assert len(examples) == 5
        assert all(e.family == "json_schema" for e in examples)

    def test_factual(self):
        examples = generate_factual_examples(n=5)
        assert len(examples) == 5
        assert all(e.family == "factual_recall" for e in examples)
