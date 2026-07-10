#!/usr/bin/env python3
"""Build and audit S01 MiniCPM single-shot data and deterministic eval assets.

The script fetches only public, version-pinned-by-fingerprint datasets and
creates all split manifests before writing any training rows.  It intentionally
does not train or run a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import jsonschema
import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data"
SEED = 20260709


def sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalized(*values: object) -> str:
    text = " ".join(str(v) for v in values)
    return re.sub(r"\s+", " ", text.lower()).strip()


def answer_value(answer: str) -> str:
    match = re.search(r"####\s*([^\n]+)", answer)
    if not match:
        raise ValueError(f"GSM8K answer lacks #### marker: {answer[-100:]!r}")
    return match.group(1).strip().replace(",", "")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def schema_row(i: int, split: str) -> dict:
    names = ["Ada", "Bo", "Cy", "Di", "Eve", "Fox", "Gia", "Hal", "Ivy", "Jay"]
    name, age, active = names[i % len(names)], 18 + (i * 7) % 63, bool(i % 2)
    tags = [f"t{i % 7}", f"g{(i * 3) % 11}"]
    obj = {"name": name, "age": age, "active": active, "tags": tags}
    schema = {"type": "object", "additionalProperties": False,
              "required": ["name", "age", "active", "tags"],
              "properties": {"name": {"type": "string", "const": name},
                             "age": {"type": "integer", "const": age},
                             "active": {"type": "boolean", "const": active},
                             "tags": {"type": "array", "items": {"type": "string"},
                                      "minItems": 2, "maxItems": 2, "const": tags}}}
    fmt = "yaml" if i % 3 == 0 else "json"
    rendered = yaml.safe_dump(obj, sort_keys=False).strip() if fmt == "yaml" else json.dumps(obj, separators=(",", ":"))
    prompt = (f"Return only valid {fmt.upper()} matching this exact schema: name (string), age (integer), "
              f"active (boolean), tags (an array of exactly two strings). Values: name={name}; age={age}; "
              f"active={str(active).lower()}; tags must be the array {json.dumps(tags)}. No markdown.")
    return {"source": "S01 deterministic schema generator", "source_license": "CC0-1.0",
            "source_revision": "s01-schema-v1", "split": split, "category": "json_yaml_schema",
            "prompt": prompt, "expected": obj, "serialization": fmt, "json_schema": schema,
            "canonical_response": rendered, "content_hash": sha([prompt, obj])}


def concise_row(i: int, split: str) -> dict:
    modes = [
        ("Convert '{x}' to uppercase. Output only the result.", lambda x: x.upper()),
        ("Reverse the characters in '{x}'. Output only the result.", lambda x: x[::-1]),
        ("Remove all spaces from '{x}'. Output only the result.", lambda x: x.replace(" ", "")),
        ("List these integers in ascending order, comma-separated with no spaces: {x}.",
         lambda x: ",".join(map(str, sorted(map(int, x.split(",")))))),
    ]
    words = ["blue river", "atlas", "quiet moon", "coda", "small model", "delta", "amber sky", "verifier"]
    template, fn = modes[i % len(modes)]
    value = f"{words[i % len(words)]}-{i}" if i % len(modes) != 3 else ",".join(str((i * k + i) % 997) for k in (3, 7, 11, 13))
    prompt, expected = template.format(x=value), fn(value)
    return {"source": "S01 deterministic instruction generator", "source_license": "CC0-1.0",
            "source_revision": "s01-instruction-v1", "split": split, "category": "concise_instruction",
            "prompt": prompt, "expected": expected, "content_hash": sha([prompt, expected])}


def heldout_schema_row(i: int) -> dict:
    """A schema shape intentionally absent from the flat S01 training rows."""
    label = f"item-{i:03d}"
    values = [i % 19, (i * 7 + 3) % 29]
    enabled = bool((i + 1) % 2)
    obj = {"record": {"label": label, "measurements": values}, "enabled": enabled}
    schema = {"type": "object", "additionalProperties": False, "required": ["record", "enabled"],
              "properties": {"record": {"type": "object", "additionalProperties": False,
                                        "required": ["label", "measurements"],
                                        "properties": {"label": {"const": label},
                                                       "measurements": {"type": "array", "const": values,
                                                                        "minItems": 2, "maxItems": 2}}},
                             "enabled": {"const": enabled}}}
    fmt = "yaml" if i % 2 else "json"
    rendered = yaml.safe_dump(obj, sort_keys=False).strip() if fmt == "yaml" else json.dumps(obj, separators=(",", ":"))
    prompt = (f"Return only {fmt.upper()}. Create an object with exactly two top-level keys: record and enabled. "
              f"record is an object whose label is {json.dumps(label)} and whose measurements is the integer array "
              f"{json.dumps(values)}. enabled is {str(enabled).lower()}. No markdown or commentary.")
    return {"source": "S02 deterministic structurally-heldout schema generator", "source_license": "CC0-1.0",
            "source_revision": "s02-schema-heldout-v1", "split": "heldout", "category": "json_yaml_schema",
            "prompt": prompt, "expected": obj, "serialization": fmt, "json_schema": schema,
            "canonical_response": rendered, "content_hash": sha([prompt, obj])}


def heldout_concise_row(i: int) -> dict:
    """Task operations intentionally disjoint from S01 train/validation modes."""
    words = ["red cedar", "quiet lake", "model atlas", "copper fox", "north star", "blue fern", "small proof", "delta map"]
    text = f"{words[i % len(words)]} {i}"
    mode = i % 4
    if mode == 0:
        prompt, expected = f"Count the words in: {json.dumps(text)}. Output only the integer.", str(len(text.split()))
    elif mode == 1:
        prompt, expected = f"Replace every vowel in {json.dumps(text)} with '*'. Output only the result.", re.sub(r"[aeiouAEIOU]", "*", text)
    elif mode == 2:
        prompt, expected = f"Output the first and last character of {json.dumps(text)}, separated by one colon and nothing else.", f"{text[0]}:{text[-1]}"
    else:
        prompt, expected = f"Replace each space in {json.dumps(text)} with an underscore. Output only the result.", text.replace(" ", "_")
    return {"source": "S02 deterministic structurally-heldout instruction generator", "source_license": "CC0-1.0",
            "source_revision": "s02-instruction-heldout-v1", "split": "heldout", "category": "concise_instruction",
            "prompt": prompt, "expected": expected, "content_hash": sha([prompt, expected])}


def gsm_rows(dataset, split: str, start: int, count: int) -> list[dict]:
    rows = []
    for item in dataset.select(range(start, start + count)):
        expected = answer_value(item["answer"])
        prompt = f"Solve this problem. Reply with only the final answer, with no explanation.\n\n{item['question']}"
        rows.append({"source": "openai/gsm8k", "source_license": "MIT", "source_revision": str(dataset._fingerprint),
                     "split": split, "category": "math", "prompt": prompt, "expected": expected,
                     "source_question": item["question"], "content_hash": sha([item["question"], expected])})
    return rows


def mbpp_rows(dataset, split: str, count: int) -> list[dict]:
    rows = []
    for item in dataset.select(range(count)):
        tests = list(item["test_list"])
        setup = item.get("test_setup_code") or ""
        # Admission: reference implementation must pass exactly the frozen tests.
        rows.append({"source": "google-research-datasets/mbpp", "source_license": "Apache-2.0",
                     "source_revision": str(dataset._fingerprint), "split": split,
                     "category": "executable_code", "prompt": f"Write Python code only. {item['text']}",
                     "reference_code": item["code"], "tests": tests, "test_setup_code": setup,
                     "task_id": item["task_id"], "content_hash": sha([item["text"], tests])})
    return rows


def validate_schema(row: dict) -> None:
    parsed = yaml.safe_load(row["canonical_response"])
    jsonschema.validate(parsed, row["json_schema"])
    if parsed != row["expected"]:
        raise ValueError("schema canonical response mismatch")


def reference_code_passes(row: dict) -> tuple[bool, str]:
    """Execute the public reference solution and its frozen tests in isolation."""
    prelude = "from collections import *\nfrom functools import *\nfrom itertools import *\nfrom math import *\nfrom typing import *"
    program = "\n".join([prelude, row.get("test_setup_code", ""), row["reference_code"], *row["tests"]])
    with tempfile.TemporaryDirectory(prefix="s01-mbpp-") as temp_dir:
        result = subprocess.run([sys.executable, "-I", "-c", program], cwd=temp_dir,
                                capture_output=True, text=True, timeout=5, check=False)
    return result.returncode == 0, result.stderr[-500:]


def admit_code(rows: list[dict], minimum: int | None = None) -> tuple[list[dict], list[dict]]:
    accepted, rejected = [], []
    for row in rows:
        passed, reason = reference_code_passes(row)
        if passed:
            accepted.append(row)
            if minimum is not None and len(accepted) >= minimum:
                break
        else:
            rejected.append({"task_id": row["task_id"], "reason": reason})
    if minimum is not None and len(accepted) < minimum:
        raise ValueError(f"only {len(accepted)} executable MBPP rows admitted; need {minimum}")
    return accepted if minimum is None else accepted[:minimum], rejected


def shingle_set(row: dict, width: int = 5) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9_]+", normalized(row["prompt"], row.get("expected", "")))
    return {tuple(tokens[i:i + width]) for i in range(max(0, len(tokens) - width + 1))}


def near_duplicate_audit(splits: dict[str, list[dict]], threshold: float = 0.80) -> dict:
    """Cross-split 5-token-shingle Jaccard audit with four-way MinHash candidates."""
    def minhash_signature(shingles: set[tuple[str, ...]]) -> tuple[int, ...]:
        if not shingles:
            return (0, 0, 0, 0)
        return tuple(min(int.from_bytes(hashlib.blake2b((str(seed) + " " + " ".join(shingle)).encode(), digest_size=8).digest(), "big")
                         for shingle in shingles) for seed in range(4))
    fingerprints = {name: [shingle_set(row) for row in rows] for name, rows in splits.items()}
    signatures = {name: [minhash_signature(shingles) for shingles in rows] for name, rows in fingerprints.items()}
    result = {"method": "5-token-shingle Jaccard with four-permutation MinHash full-signature candidates", "threshold": threshold, "comparisons": {}}
    for left, right in (("train", "validation"), ("train", "heldout"), ("validation", "heldout")):
        index: dict[tuple[int, ...], set[int]] = {}
        for right_index, signature in enumerate(signatures[right]):
            index.setdefault(signature, set()).add(right_index)
        candidates = {(left_index, right_index) for left_index, signature in enumerate(signatures[left])
                      for right_index in index.get(signature, ())}
        matches = []
        for left_index, right_index in candidates:
            intersection = len(fingerprints[left][left_index] & fingerprints[right][right_index])
            union = len(fingerprints[left][left_index] | fingerprints[right][right_index])
            score = intersection / union if union else 0.0
            if score >= threshold:
                matches.append({"left_index": left_index, "right_index": right_index, "jaccard": round(score, 6)})
        result["comparisons"][f"{left}__{right}"] = {"candidate_pairs": len(candidates), "near_duplicate_count": len(matches), "examples": matches[:20]}
    return result


def remove_code_near_duplicates(rows: list[dict], prior_rows: list[dict], threshold: float = 0.80) -> tuple[list[dict], list[dict]]:
    """Remove code prompts overlapping earlier source splits before split admission."""
    prior = [(row.get("task_id"), shingle_set(row)) for row in prior_rows]
    kept, rejected = [], []
    for row in rows:
        shingles = shingle_set(row)
        duplicate = next((task_id for task_id, other in prior
                          if len(shingles & other) / len(shingles | other) >= threshold), None)
        if duplicate is None:
            kept.append(row)
        else:
            rejected.append({"task_id": row["task_id"], "reason": f"near_duplicate_of_task_{duplicate}"})
    return kept, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-prefix", default="minicpm5_s01")
    parser.add_argument("--structurally-diverse-heldout", action="store_true")
    parser.add_argument("--allow-near-duplicates", action="store_true", help="Write an audit-failure manifest for diagnosis; never use it for evaluation.")
    args = parser.parse_args()
    manifest_path = OUT / "manifests" / f"{args.output_prefix}_manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"{manifest_path} exists; use --force to rebuild")

    gsm = load_dataset("openai/gsm8k", "main")
    mbpp = load_dataset("google-research-datasets/mbpp")
    print("datasets loaded", flush=True)
    # Official source splits are kept disjoint.  GSM validation is a deterministic
    # prefix of its training split and is excluded from its training rows.
    train = gsm_rows(gsm["train"], "train", 500, len(gsm["train"]) - 500)
    validation = gsm_rows(gsm["train"], "validation", 0, 500)
    heldout_math = gsm_rows(gsm["test"], "heldout", 0, 200)
    train += [schema_row(i, "train") for i in range(600)]
    validation += [schema_row(600 + i, "validation") for i in range(100)]
    heldout_schema = [heldout_schema_row(i) for i in range(200)] if args.structurally_diverse_heldout else [schema_row(700 + i, "heldout") for i in range(200)]
    train += [concise_row(i, "train") for i in range(600)]
    validation += [concise_row(600 + i, "validation") for i in range(100)]
    heldout_instruction = [heldout_concise_row(i) for i in range(200)] if args.structurally_diverse_heldout else [concise_row(700 + i, "heldout") for i in range(200)]
    admitted_train_code, rejected_train_code = admit_code(mbpp_rows(mbpp["train"], "train", len(mbpp["train"])))
    print("train code admitted", flush=True)
    admitted_validation_code, rejected_validation_code = admit_code(mbpp_rows(mbpp["validation"], "validation", len(mbpp["validation"])))
    admitted_validation_code, validation_near_rejected = remove_code_near_duplicates(admitted_validation_code, admitted_train_code)
    rejected_validation_code += validation_near_rejected
    print("validation code admitted", flush=True)
    heldout_candidates, rejected_heldout_code = admit_code(mbpp_rows(mbpp["test"], "heldout", 260))
    heldout_candidates, heldout_near_rejected = remove_code_near_duplicates(heldout_candidates, admitted_train_code + admitted_validation_code)
    rejected_heldout_code += heldout_near_rejected
    if len(heldout_candidates) < 200:
        raise ValueError(f"only {len(heldout_candidates)} heldout code rows after near-duplicate removal")
    heldout_code = heldout_candidates[:200]
    print("heldout code admitted", flush=True)
    train += admitted_train_code
    validation += admitted_validation_code

    for row in [r for r in train + validation + heldout_schema if r["category"] == "json_yaml_schema"]:
        validate_schema(row)

    all_splits = {"train": train, "validation": validation,
                  "heldout": heldout_math + heldout_schema + heldout_instruction + heldout_code}
    for index, row in enumerate(all_splits["heldout"]):
        row["eval_id"] = f"eval_{index:04d}"
    hashes = {name: {row["content_hash"] for row in rows} for name, rows in all_splits.items()}
    intersections = {f"{a}__{b}": len(hashes[a] & hashes[b])
                     for a, b in (("train", "validation"), ("train", "heldout"), ("validation", "heldout"))}
    if any(intersections.values()):
        raise ValueError(f"cross-split exact duplicate hashes: {intersections}")
    near_duplicates = near_duplicate_audit(all_splits)
    print("near-duplicate audit completed", flush=True)
    near_duplicate_counts = {key: value["near_duplicate_count"] for key, value in near_duplicates["comparisons"].items()}
    if any(near_duplicate_counts.values()) and not args.allow_near_duplicates:
        raise ValueError(f"cross-split near duplicates: {near_duplicate_counts}")
    counts = Counter(row["category"] for row in all_splits["heldout"])
    if any(counts[family] < 200 for family in ("math", "json_yaml_schema", "concise_instruction", "executable_code")):
        raise ValueError(f"underpowered heldout suite: {counts}")

    for name, rows in all_splits.items():
        write_jsonl(OUT / "sft" / f"{args.output_prefix}_{name}.jsonl", rows)
    write_jsonl(OUT / "eval" / f"{args.output_prefix}_heldout.jsonl", all_splits["heldout"])
    manifest = {"experiment_id": "S01", "created_at": datetime.now(timezone.utc).isoformat(), "seed": SEED,
                "builder": "scripts/data/build_minicpm5_s01_assets.py", "sources": [
                    {"name": "openai/gsm8k", "config": "main", "license": "MIT", "splits": {"train": len(gsm['train']), "test": len(gsm['test'])}, "fingerprint": str(gsm['train']._fingerprint)},
                    {"name": "google-research-datasets/mbpp", "license": "Apache-2.0", "splits": {k: len(v) for k, v in mbpp.items()}, "fingerprint": str(mbpp['train']._fingerprint)},
                    {"name": "S01/S02 deterministic schema/instruction generators", "license": "CC0-1.0", "revision": "s02-v1" if args.structurally_diverse_heldout else "s01-v1"}],
                "files": {name: {"path": str(OUT / ("eval" if name == "heldout" else "sft") / f"{args.output_prefix}_{'heldout' if name == 'heldout' else name}.jsonl"),
                                 "count": len(rows), "sha256": sha(rows)} for name, rows in all_splits.items()},
                "heldout_family_counts": dict(counts), "exact_hash_intersections": intersections, "near_duplicate_audit": near_duplicates,
                "data_audit_passed": not any(near_duplicate_counts.values()),
                "admission": {"schema_rows_validated": 900, "code_rows_executed": len(admitted_train_code) + len(admitted_validation_code) + len(heldout_code), "code_rows_rejected": {"train": rejected_train_code, "validation": rejected_validation_code, "heldout": rejected_heldout_code}, "benchmark_test_in_train": False}}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "heldout_counts": counts, "intersections": intersections}, indent=2))


if __name__ == "__main__":
    main()
