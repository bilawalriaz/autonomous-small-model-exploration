#!/usr/bin/env python3
"""Deterministically validate and score paired S01 MiniCPM outputs.

No model or heuristic judge is used.  Code is tested in a short-lived isolated
Python process; structured outputs are parsed and schema-validated.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import jsonschema
import yaml

VERSION = "s01-deterministic-verifier-v1"


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # S01 v1 records predate an explicit eval_id field.  The GGUF evaluator
    # deterministically uses this same zero-based fallback, so preserve it for
    # scoring rather than rerunning 1,600 completed generations.
    return {row.get("eval_id", row.get("id", f"eval_{index:04d}")): row for index, row in enumerate(rows)}


def normalize_number(text: str) -> str | None:
    marked = re.findall(r"####\s*([^\n]+)", text)
    if marked:
        value = marked[-1]
    else:
        tokens = re.findall(r"(?<![\w.])-?[\d][\d,]*(?:\.\d+)?(?:/[\d]+)?(?![\w.])", text)
        if not tokens:
            return None
        value = tokens[-1]
    return value.strip().replace(",", "").rstrip(".")


def code_from_response(text: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S | re.I)
    return fenced[-1].strip() if fenced else text.strip()


def score(row: dict, text: str) -> tuple[bool, str | None]:
    category = row["category"]
    if category == "math":
        return normalize_number(text) == str(row["expected"]), None
    if category == "concise_instruction":
        return text.strip() == row["expected"], None
    if category == "json_yaml_schema":
        try:
            parsed = yaml.safe_load(text)
            jsonschema.validate(parsed, row["json_schema"])
            return parsed == row["expected"], None
        except (yaml.YAMLError, jsonschema.ValidationError, TypeError, ValueError) as error:
            return False, f"schema:{type(error).__name__}"
    if category == "executable_code":
        prelude = "from collections import *\nfrom functools import *\nfrom itertools import *\nfrom math import *\nfrom typing import *"
        program = "\n".join([prelude, row.get("test_setup_code", ""), code_from_response(text), *row["tests"]])
        try:
            with tempfile.TemporaryDirectory(prefix="s01-eval-") as temp_dir:
                result = subprocess.run([sys.executable, "-I", "-c", program], cwd=temp_dir, capture_output=True,
                                        text=True, timeout=5, check=False)
            return result.returncode == 0, None if result.returncode == 0 else f"test:returncode={result.returncode}"
        except subprocess.TimeoutExpired:
            return False, "test:timeout"
    raise ValueError(f"Unknown category {category}")


def bootstrap(deltas: list[int], n: int = 20000) -> list[float]:
    rng, size = random.Random(42), len(deltas)
    means = sorted(sum(deltas[rng.randrange(size)] for _ in range(size)) / size for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n) - 1]]


def fixtures(path: Path) -> None:
    failures = []
    for fixture in json.loads(path.read_text()):
        actual, _ = score(fixture["row"], fixture["response"])
        if actual != fixture["expected_pass"]:
            failures.append(fixture["name"])
    if failures:
        raise SystemExit(f"S01 verifier fixtures failed: {failures}")
    print(f"S01 verifier fixtures passed: {len(json.loads(path.read_text()))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="data/eval/minicpm5_s01_heldout.jsonl")
    parser.add_argument("--base")
    parser.add_argument("--merged")
    parser.add_argument("--output")
    parser.add_argument("--manual-review")
    parser.add_argument("--experiment-id", default=None, help="Report identifier; inferred from the eval-set filename when omitted.")
    parser.add_argument("--fixtures", default="tests/fixtures/minicpm5_s01_verifier.json")
    parser.add_argument("--validate-fixtures", action="store_true")
    args = parser.parse_args()
    fixtures(Path(args.fixtures))
    if args.validate_fixtures:
        return
    if not all((args.base, args.merged, args.output, args.manual_review)):
        parser.error("--base, --merged, --output, and --manual-review are required for scoring")
    data, base, merged = load_jsonl(Path(args.eval_set)), load_jsonl(Path(args.base)), load_jsonl(Path(args.merged))
    if set(data) != set(base) or set(data) != set(merged):
        raise SystemExit("eval_id sets differ between suite and model outputs")
    integrity, families, reviews = {}, {}, []
    for label, run in (("base", base), ("merged", merged)):
        vals = list(run.values())
        integrity[label] = {"count": len(vals), "nonzero_returncodes": sum(v.get("returncode", 0) != 0 for v in vals),
                            "prompt_echoes": sum(bool(v.get("prompt_echo_removed")) for v in vals),
                            "template_residue": sum(bool(v.get("template_residue_detected")) for v in vals),
                            "empty_outputs": sum(not v.get("raw_stdout", "") for v in vals)}
    for family in ("math", "json_yaml_schema", "concise_instruction", "executable_code"):
        ids = sorted(key for key, row in data.items() if row["category"] == family)
        pairs = []
        for key in ids:
            b, b_reason = score(data[key], base[key]["generated_response"])
            m, m_reason = score(data[key], merged[key]["generated_response"])
            pairs.append((int(b), int(m)))
            reviews.append({"eval_id": key, "category": family, "prompt": data[key]["prompt"], "base_response": base[key]["generated_response"], "merged_response": merged[key]["generated_response"], "base_pass": b, "merged_pass": m, "base_failure": b_reason, "merged_failure": m_reason})
        deltas = [m - b for b, m in pairs]
        families[family] = {"n": len(ids), "base_rate": sum(b for b, _ in pairs) / len(ids), "merged_rate": sum(m for _, m in pairs) / len(ids), "delta": sum(deltas) / len(ids), "delta_ci95": bootstrap(deltas), "wins": sum(m > b for b, m in pairs), "losses": sum(m < b for b, m in pairs), "ties": sum(m == b for b, m in pairs)}
    inferred_id = re.search(r"minicpm5_(s\d+)_", Path(args.eval_set).name, re.I)
    experiment_id = args.experiment_id or (inferred_id.group(1).upper() if inferred_id else "S01")
    report = {"experiment_id": experiment_id, "verifier_version": VERSION, "fixture_validation": "passed", "integrity": integrity, "families": families}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    rng = random.Random(42)
    sample = []
    for family in families:
        candidates = [r for r in reviews if r["category"] == family]
        sample += rng.sample(candidates, min(12, len(candidates)))
    Path(args.manual_review).write_text(json.dumps({"experiment_id": experiment_id, "selection": "stratified deterministic random seed 42", "rows": sample}, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
