#!/usr/bin/env python3
"""Aggregate judge scores into summary metrics and comparison tables.

Enhanced with programmatic scorers that run on raw outputs without a judge:
  - JSON validity
  - Schema validity (expected_keys)
  - Entity/extraction F1 (expected_fields)
  - Exact-match factual (gold_answer)
  - Arithmetic/numeric exact match (gold_numeric)
  - Slop phrase rate
  - Output length statistics
  - Constraint following (max_words, required_format)

CLI:
    python scripts/eval/aggregate_eval_results.py \
        --run-id lfm2_230m_format_ablation_multi_turn_concise_20260629 \
        --compare-with lfm2_230m_base_20260629 lfm2_230m_format_ablation_alpaca_flat_20260629

    # Only aggregate api-judged scores
    python scripts/eval/aggregate_eval_results.py \
        --run-id some_run --judge-source api

    # Only aggregate mock-judged scores
    python scripts/eval/aggregate_eval_results.py \
        --run-id some_run --judge-source mock
"""

import argparse
import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SLOP_PHRASES = [
    "as an ai", "i apologize", "i'm sorry, but", "as a language model",
    "i don't have personal", "it's important to note that",
    "please note that", "i hope this helps",
]

DIMENSIONS = [
    "correctness", "instruction_following", "output_format",
    "concision", "usefulness", "hallucination_risk", "overall",
]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Programmatic scorers — deterministic, no LLM calls
# ---------------------------------------------------------------------------

def _parse_json_from_text(text: str):
    """Try to parse JSON from text. Returns (parsed_object, True) or (None, False)."""
    # Try the whole text first
    try:
        return json.loads(text), True
    except (json.JSONDecodeError, ValueError):
        pass
    # Try extracting JSON objects from the text
    # Use a more robust approach: find balanced braces
    start = text.find('{')
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1]), True
                    except (json.JSONDecodeError, ValueError):
                        break
        start = text.find('{', start + 1)

    # Try array-style JSON
    start = text.find('[')
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '[':
                depth += 1
            elif text[i] == ']':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1]), True
                    except (json.JSONDecodeError, ValueError):
                        break
        start = text.find('[', start + 1)

    return None, False


def scorer_json_validity(text: str, _eval_entry: dict) -> dict | None:
    """Check if response contains valid JSON."""
    _, is_valid = _parse_json_from_text(text)
    return {"valid": is_valid}


def scorer_schema_validity(text: str, eval_entry: dict) -> dict | None:
    """Check that expected JSON keys exist in parsed JSON."""
    expected_keys = eval_entry.get("expected_keys")
    if not expected_keys:
        return None
    parsed, is_valid = _parse_json_from_text(text)
    if not is_valid:
        return {"valid_json": False, "keys_present": [], "keys_missing": expected_keys, "all_keys_found": False}
    if not isinstance(parsed, dict):
        return {"valid_json": True, "keys_present": [], "keys_missing": expected_keys, "all_keys_found": False}
    present = [k for k in expected_keys if k in parsed]
    missing = [k for k in expected_keys if k not in parsed]
    return {
        "valid_json": True,
        "keys_present": present,
        "keys_missing": missing,
        "all_keys_found": len(missing) == 0,
    }


def scorer_entity_f1(text: str, eval_entry: dict) -> dict | None:
    """Compute precision/recall/F1 of extracted fields vs expected_fields."""
    expected_fields = eval_entry.get("expected_fields")
    if not expected_fields:
        return None
    if not isinstance(expected_fields, dict):
        return None

    text_lower = text.lower()
    tp = 0
    for key, expected_val in expected_fields.items():
        expected_str = str(expected_val).lower()
        if expected_str in text_lower:
            tp += 1

    fp = 0  # We can't reliably count false positives without knowing all possible fields
    fn = len(expected_fields) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_expected": len(expected_fields),
    }


def scorer_exact_match(text: str, eval_entry: dict) -> dict | None:
    """Check if gold_answer appears in response (case-insensitive)."""
    gold = eval_entry.get("gold_answer")
    if gold is None:
        return None
    gold_str = str(gold).lower()
    found = gold_str in text.lower()
    return {"gold_answer": gold, "found": found}


def scorer_numeric_match(text: str, eval_entry: dict) -> dict | None:
    """Extract numbers from response and compare to gold_numeric."""
    gold = eval_entry.get("gold_numeric")
    if gold is None:
        return None
    try:
        gold_num = float(gold)
    except (ValueError, TypeError):
        return None

    # Extract all numbers from text (integers and floats, including negatives)
    number_pattern = re.compile(r'-?\d+\.?\d*')
    found_numbers = []
    for match in number_pattern.finditer(text):
        try:
            found_numbers.append(float(match.group()))
        except ValueError:
            continue

    exact_match = gold_num in found_numbers
    # Also check for approximate match (within 0.01)
    approx_match = any(abs(n - gold_num) < 0.01 for n in found_numbers) if not exact_match else False

    return {
        "gold_numeric": gold_num,
        "numbers_found": found_numbers[:10],  # cap for readability
        "exact_match": exact_match,
        "approx_match": approx_match,
    }


def scorer_slop_rate(text: str, _eval_entry: dict) -> dict | None:
    """Count slop phrases in text."""
    text_lower = text.lower()
    count = sum(1 for p in SLOP_PHRASES if p in text_lower)
    matched = [p for p in SLOP_PHRASES if p in text_lower]
    return {"slop_count": count, "has_slop": count > 0, "matched_phrases": matched}


def scorer_output_length(text: str, _eval_entry: dict) -> dict | None:
    """Compute output length statistics."""
    # Simple whitespace-based token count (deterministic, no external deps)
    tokens = text.split()
    chars = len(text)
    lines = text.count('\n') + 1 if text else 0
    return {
        "token_count": len(tokens),
        "char_count": chars,
        "line_count": lines,
    }


def scorer_constraint_following(text: str, eval_entry: dict) -> dict | None:
    """Check compliance with constraints like max_words, required_format."""
    constraints = {}

    # max_words constraint
    max_words = eval_entry.get("max_words")
    if max_words is not None:
        word_count = len(text.split())
        constraints["max_words"] = {
            "limit": max_words,
            "actual": word_count,
            "compliant": word_count <= max_words,
        }

    # required_format constraint
    required_format = eval_entry.get("required_format")
    if required_format is not None:
        fmt = required_format.lower().strip()
        if fmt == "json":
            _, is_valid_json = _parse_json_from_text(text)
            constraints["required_format"] = {
                "required": "json",
                "compliant": is_valid_json,
            }
        elif fmt == "number" or fmt == "numeric":
            is_numeric = bool(re.match(r'^\s*-?\d+\.?\d*\s*$', text.strip()))
            constraints["required_format"] = {
                "required": fmt,
                "compliant": is_numeric,
            }
        else:
            constraints["required_format"] = {
                "required": fmt,
                "compliant": None,  # Unknown format type
                "note": f"Unknown required_format: {fmt}",
            }

    # Also check hard_constraints if present
    hard_constraints = eval_entry.get("hard_constraints", [])
    if hard_constraints:
        constraints["hard_constraints"] = hard_constraints

    if not constraints:
        return None

    # Overall compliance
    compliant_flags = [v.get("compliant") for k, v in constraints.items()
                       if isinstance(v, dict) and v.get("compliant") is not None]
    overall_compliant = all(compliant_flags) if compliant_flags else None

    constraints["overall_compliant"] = overall_compliant
    return constraints


ALL_SCORERS = [
    ("json_validity", scorer_json_validity),
    ("schema_validity", scorer_schema_validity),
    ("entity_f1", scorer_entity_f1),
    ("exact_match", scorer_exact_match),
    ("numeric_match", scorer_numeric_match),
    ("slop_phrases", scorer_slop_rate),
    ("output_length", scorer_output_length),
    ("constraint_following", scorer_constraint_following),
]


def run_programmatic_scorers(outputs: list[dict], eval_data_map: dict) -> dict:
    """Run all programmatic scorers on outputs. Returns aggregated results."""
    # Results structure: scorer_name -> list of per-item results
    raw_results = defaultdict(list)
    per_category = defaultdict(lambda: defaultdict(list))

    for output in outputs:
        eval_id = output.get("eval_id", "")
        category = output.get("category", "unknown")
        response = output.get("generated_response", "")
        eval_entry = eval_data_map.get(eval_id, {})

        for scorer_name, scorer_fn in ALL_SCORERS:
            result = scorer_fn(response, eval_entry)
            if result is not None:
                result["eval_id"] = eval_id
                result["category"] = category
                raw_results[scorer_name].append(result)
                per_category[scorer_name][category].append(result)

    # Aggregate each scorer
    aggregated = {}

    # json_validity
    if "json_validity" in raw_results:
        items = raw_results["json_validity"]
        valid_count = sum(1 for r in items if r.get("valid"))
        aggregated["json_validity"] = {
            "total_checked": len(items),
            "valid_count": valid_count,
            "valid_rate": round(valid_count / len(items), 4) if items else 0,
            "per_category": {},
        }
        for cat, cat_items in per_category["json_validity"].items():
            cat_valid = sum(1 for r in cat_items if r.get("valid"))
            aggregated["json_validity"]["per_category"][cat] = {
                "total": len(cat_items),
                "valid": cat_valid,
                "valid_rate": round(cat_valid / len(cat_items), 4) if cat_items else 0,
            }

    # schema_validity
    if "schema_validity" in raw_results:
        items = raw_results["schema_validity"]
        all_found = sum(1 for r in items if r.get("all_keys_found"))
        aggregated["schema_validity"] = {
            "total_checked": len(items),
            "all_keys_found_count": all_found,
            "all_keys_found_rate": round(all_found / len(items), 4) if items else 0,
            "per_category": {},
        }
        for cat, cat_items in per_category["schema_validity"].items():
            cat_found = sum(1 for r in cat_items if r.get("all_keys_found"))
            aggregated["schema_validity"]["per_category"][cat] = {
                "total": len(cat_items),
                "all_keys_found": cat_found,
                "rate": round(cat_found / len(cat_items), 4) if cat_items else 0,
            }

    # entity_f1
    if "entity_f1" in raw_results:
        items = raw_results["entity_f1"]
        f1_vals = [r["f1"] for r in items]
        precision_vals = [r["precision"] for r in items]
        recall_vals = [r["recall"] for r in items]
        aggregated["entity_f1"] = {
            "total_checked": len(items),
            "avg_precision": round(sum(precision_vals) / len(precision_vals), 4) if precision_vals else 0,
            "avg_recall": round(sum(recall_vals) / len(recall_vals), 4) if recall_vals else 0,
            "avg_f1": round(sum(f1_vals) / len(f1_vals), 4) if f1_vals else 0,
            "perfect_f1_count": sum(1 for f in f1_vals if f >= 1.0),
            "per_category": {},
        }
        for cat, cat_items in per_category["entity_f1"].items():
            cat_f1 = [r["f1"] for r in cat_items]
            cat_prec = [r["precision"] for r in cat_items]
            cat_rec = [r["recall"] for r in cat_items]
            aggregated["entity_f1"]["per_category"][cat] = {
                "total": len(cat_items),
                "avg_f1": round(sum(cat_f1) / len(cat_f1), 4) if cat_f1 else 0,
                "avg_precision": round(sum(cat_prec) / len(cat_prec), 4) if cat_prec else 0,
                "avg_recall": round(sum(cat_rec) / len(cat_rec), 4) if cat_rec else 0,
            }

    # exact_match
    if "exact_match" in raw_results:
        items = raw_results["exact_match"]
        found_count = sum(1 for r in items if r.get("found"))
        aggregated["exact_match"] = {
            "total_checked": len(items),
            "match_count": found_count,
            "match_rate": round(found_count / len(items), 4) if items else 0,
            "per_category": {},
        }
        for cat, cat_items in per_category["exact_match"].items():
            cat_match = sum(1 for r in cat_items if r.get("found"))
            aggregated["exact_match"]["per_category"][cat] = {
                "total": len(cat_items),
                "match_count": cat_match,
                "match_rate": round(cat_match / len(cat_items), 4) if cat_items else 0,
            }

    # numeric_match
    if "numeric_match" in raw_results:
        items = raw_results["numeric_match"]
        exact = sum(1 for r in items if r.get("exact_match"))
        approx = sum(1 for r in items if r.get("approx_match"))
        aggregated["numeric_match"] = {
            "total_checked": len(items),
            "exact_match_count": exact,
            "exact_match_rate": round(exact / len(items), 4) if items else 0,
            "approx_match_count": exact + approx,
            "approx_match_rate": round((exact + approx) / len(items), 4) if items else 0,
            "per_category": {},
        }
        for cat, cat_items in per_category["numeric_match"].items():
            cat_exact = sum(1 for r in cat_items if r.get("exact_match"))
            cat_approx = sum(1 for r in cat_items if r.get("approx_match"))
            aggregated["numeric_match"]["per_category"][cat] = {
                "total": len(cat_items),
                "exact_match": cat_exact,
                "approx_match": cat_exact + cat_approx,
            }

    # slop_phrases
    if "slop_phrases" in raw_results:
        items = raw_results["slop_phrases"]
        slop_count = sum(1 for r in items if r.get("has_slop"))
        total_slop_phrases = sum(r.get("slop_count", 0) for r in items)
        aggregated["slop_phrases"] = {
            "total_checked": len(items),
            "responses_with_slop": slop_count,
            "slop_response_rate": round(slop_count / len(items), 4) if items else 0,
            "total_slop_phrases": total_slop_phrases,
            "avg_slop_per_response": round(total_slop_phrases / len(items), 4) if items else 0,
            "per_category": {},
        }
        for cat, cat_items in per_category["slop_phrases"].items():
            cat_slop = sum(1 for r in cat_items if r.get("has_slop"))
            cat_total = sum(r.get("slop_count", 0) for r in cat_items)
            aggregated["slop_phrases"]["per_category"][cat] = {
                "total": len(cat_items),
                "responses_with_slop": cat_slop,
                "total_phrases": cat_total,
            }

    # output_length
    if "output_length" in raw_results:
        items = raw_results["output_length"]
        tokens = [r["token_count"] for r in items]
        chars = [r["char_count"] for r in items]
        lines_list = [r["line_count"] for r in items]

        def _percentile(sorted_vals, p):
            if not sorted_vals:
                return 0
            k = (len(sorted_vals) - 1) * p / 100
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_vals[int(k)]
            return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

        tokens_sorted = sorted(tokens)
        chars_sorted = sorted(chars)

        aggregated["output_length"] = {
            "total_checked": len(items),
            "token_count": {
                "mean": round(sum(tokens) / len(tokens), 2) if tokens else 0,
                "median": round(_percentile(tokens_sorted, 50), 2),
                "min": min(tokens) if tokens else 0,
                "max": max(tokens) if tokens else 0,
                "p25": round(_percentile(tokens_sorted, 25), 2),
                "p75": round(_percentile(tokens_sorted, 75), 2),
                "stdev": round(
                    math.sqrt(sum((t - sum(tokens)/len(tokens))**2 for t in tokens) / len(tokens)), 2
                ) if tokens else 0,
            },
            "char_count": {
                "mean": round(sum(chars) / len(chars), 2) if chars else 0,
                "median": round(_percentile(chars_sorted, 50), 2),
                "min": min(chars) if chars else 0,
                "max": max(chars) if chars else 0,
            },
            "line_count": {
                "mean": round(sum(lines_list) / len(lines_list), 2) if lines_list else 0,
            },
            "per_category": {},
        }
        for cat, cat_items in per_category["output_length"].items():
            cat_tokens = [r["token_count"] for r in cat_items]
            cat_chars = [r["char_count"] for r in cat_items]
            cat_sorted = sorted(cat_tokens)
            aggregated["output_length"]["per_category"][cat] = {
                "total": len(cat_items),
                "token_mean": round(sum(cat_tokens) / len(cat_tokens), 2) if cat_tokens else 0,
                "token_median": round(_percentile(cat_sorted, 50), 2),
                "token_min": min(cat_tokens) if cat_tokens else 0,
                "token_max": max(cat_tokens) if cat_tokens else 0,
            }

    # constraint_following
    if "constraint_following" in raw_results:
        items = raw_results["constraint_following"]
        compliant_count = sum(1 for r in items if r.get("overall_compliant") is True)
        checked = sum(1 for r in items if r.get("overall_compliant") is not None)
        aggregated["constraint_following"] = {
            "total_checked": len(items),
            "compliant_count": compliant_count,
            "compliance_rate": round(compliant_count / checked, 4) if checked > 0 else None,
            "per_category": {},
        }
        for cat, cat_items in per_category["constraint_following"].items():
            cat_compliant = sum(1 for r in cat_items if r.get("overall_compliant") is True)
            cat_checked = sum(1 for r in cat_items if r.get("overall_compliant") is not None)
            aggregated["constraint_following"]["per_category"][cat] = {
                "total": len(cat_items),
                "compliant": cat_compliant,
                "compliance_rate": round(cat_compliant / cat_checked, 4) if cat_checked > 0 else None,
            }

    return aggregated


# ---------------------------------------------------------------------------
# Judge score aggregation
# ---------------------------------------------------------------------------

def check_json_validity(text: str) -> bool:
    """Check if response contains valid JSON (legacy helper)."""
    _, is_valid = _parse_json_from_text(text)
    return is_valid


def check_slop(text: str) -> int:
    """Count slop phrases in text."""
    text_lower = text.lower()
    return sum(1 for p in SLOP_PHRASES if p in text_lower)


def aggregate_pointwise(scores: list[dict]) -> dict:
    """Aggregate pointwise scores into summary metrics."""
    if not scores:
        return {}

    by_category = defaultdict(list)
    for s in scores:
        cat = s.get("category", "unknown")
        by_category[cat].append(s)

    # Overall averages
    avg_scores = {}
    for dim in DIMENSIONS:
        vals = [s["scores"].get(dim, 0) for s in scores if s.get("scores")]
        avg_scores[dim] = round(sum(vals) / len(vals), 3) if vals else 0

    # Per-category averages
    category_scores = {}
    for cat, cat_scores in by_category.items():
        cat_avg = {}
        for dim in DIMENSIONS:
            vals = [s["scores"].get(dim, 0) for s in cat_scores if s.get("scores")]
            cat_avg[dim] = round(sum(vals) / len(vals), 3) if vals else 0
        category_scores[cat] = {
            "avg_scores": cat_avg,
            "count": len(cat_scores),
        }

    return {
        "count": len(scores),
        "avg_scores": avg_scores,
        "category_scores": category_scores,
    }


def compute_format_metrics(outputs: list[dict]) -> dict:
    """Compute format-specific metrics from outputs."""
    json_tasks = [o for o in outputs if o.get("category") in ("json_structured", "gamefaq_extraction", "structured_terse")]
    json_valid = sum(1 for o in json_tasks if check_json_validity(o.get("generated_response", "")))
    json_rate = round(json_valid / len(json_tasks), 3) if json_tasks else None

    lengths = [o.get("tokens_generated", 0) for o in outputs]
    avg_length = round(sum(lengths) / len(lengths), 1) if lengths else 0

    slop_counts = [check_slop(o.get("generated_response", "")) for o in outputs]
    total_slop = sum(slop_counts)
    slop_rate = round(sum(1 for c in slop_counts if c > 0) / len(outputs), 3) if outputs else 0

    return {
        "json_format_validity_rate": json_rate,
        "json_task_count": len(json_tasks),
        "avg_output_length": avg_length,
        "slop_phrase_rate": slop_rate,
        "slop_phrase_total": total_slop,
        "total_outputs": len(outputs),
    }


def compute_regression_summary(current: dict, baseline: dict) -> dict:
    """Compare current vs baseline, count regressions."""
    if not baseline:
        return {"regressions": 0, "improvements": 0, "details": []}

    current_cats = current.get("category_scores", {})
    baseline_cats = baseline.get("category_scores", {})

    regressions = []
    improvements = []

    for cat in set(list(current_cats.keys()) + list(baseline_cats.keys())):
        cur = current_cats.get(cat, {}).get("avg_scores", {})
        base = baseline_cats.get(cat, {}).get("avg_scores", {})
        cur_overall = cur.get("overall", 0)
        base_overall = base.get("overall", 0)

        if cur_overall < base_overall - 0.3:
            regressions.append({
                "category": cat,
                "current_overall": cur_overall,
                "baseline_overall": base_overall,
                "delta": round(cur_overall - base_overall, 3),
            })
        elif cur_overall > base_overall + 0.3:
            improvements.append({
                "category": cat,
                "current_overall": cur_overall,
                "baseline_overall": base_overall,
                "delta": round(cur_overall - base_overall, 3),
            })

    return {
        "regressions": len(regressions),
        "improvements": len(improvements),
        "regression_details": regressions,
        "improvement_details": improvements,
    }


def compute_winrate(current: dict, baseline: dict) -> float | None:
    """Compute overall win rate from pairwise scores or pointwise comparison."""
    if not baseline:
        return None
    cur_overall = current.get("avg_scores", {}).get("overall", 0)
    base_overall = baseline.get("avg_scores", {}).get("overall", 0)
    if cur_overall + base_overall == 0:
        return 0.5
    return round(cur_overall / (cur_overall + base_overall), 3)


def compute_judge_source_summary(scores: list[dict]) -> dict:
    """Summarize judge_source distribution across scores."""
    source_counts = defaultdict(int)
    for s in scores:
        source = s.get("judge_source", "unknown")
        source_counts[source] += 1
    return dict(source_counts)


def print_summary_table(run_id: str, metrics: dict, comparisons: dict, judge_source_summary: dict | None = None):
    """Print a readable summary table."""
    print(f"\n{'='*80}")
    print(f"  EVAL SUMMARY: {run_id}")
    print(f"{'='*80}")

    # Judge source info
    if judge_source_summary:
        total_scores = sum(judge_source_summary.values())
        print(f"\n  Judge Source Breakdown:")
        for source, count in sorted(judge_source_summary.items()):
            pct = round(100 * count / total_scores, 1) if total_scores > 0 else 0
            marker = " ⚠" if source == "mock" else ""
            print(f"    {source}: {count} ({pct}%){marker}")
        if "mock" in judge_source_summary:
            print(f"  ⚠ WARNING: Some scores are from mock judging — not real API evaluation.")

    # Scores
    avg = metrics.get("pointwise", {}).get("avg_scores", {})
    if avg:
        print(f"\n  Overall Scores (1-5):")
        print(f"  {'Dimension':<25} {'Score':>6}")
        print(f"  {'-'*31}")
        for dim in DIMENSIONS:
            val = avg.get(dim, "N/A")
            bar = "█" * int(float(val) * 4) if isinstance(val, (int, float)) else ""
            print(f"  {dim:<25} {val:>6}  {bar}")

    # Format metrics
    fmt = metrics.get("format", {})
    if fmt:
        print(f"\n  Format Metrics:")
        print(f"  {'Metric':<35} {'Value':>8}")
        print(f"  {'-'*43}")
        for k, v in fmt.items():
            print(f"  {k:<35} {v:>8}")

    # Programmatic metrics summary
    prog = metrics.get("programmatic_metrics", {})
    if prog:
        print(f"\n  Programmatic Metrics:")
        print(f"  {'Scorer':<30} {'Key Metric':<25} {'Value':>8}")
        print(f"  {'-'*63}")
        if "json_validity" in prog:
            r = prog["json_validity"]
            print(f"  {'json_validity':<30} {'valid_rate':<25} {r.get('valid_rate', 'N/A'):>8}")
        if "schema_validity" in prog:
            r = prog["schema_validity"]
            print(f"  {'schema_validity':<30} {'all_keys_found_rate':<25} {r.get('all_keys_found_rate', 'N/A'):>8}")
        if "entity_f1" in prog:
            r = prog["entity_f1"]
            print(f"  {'entity_f1':<30} {'avg_f1':<25} {r.get('avg_f1', 'N/A'):>8}")
        if "exact_match" in prog:
            r = prog["exact_match"]
            print(f"  {'exact_match':<30} {'match_rate':<25} {r.get('match_rate', 'N/A'):>8}")
        if "numeric_match" in prog:
            r = prog["numeric_match"]
            print(f"  {'numeric_match':<30} {'exact_match_rate':<25} {r.get('exact_match_rate', 'N/A'):>8}")
        if "slop_phrases" in prog:
            r = prog["slop_phrases"]
            print(f"  {'slop_phrases':<30} {'slop_response_rate':<25} {r.get('slop_response_rate', 'N/A'):>8}")
        if "output_length" in prog:
            r = prog["output_length"]
            tc = r.get("token_count", {})
            print(f"  {'output_length':<30} {'token_mean':<25} {tc.get('mean', 'N/A'):>8}")
        if "constraint_following" in prog:
            r = prog["constraint_following"]
            rate = r.get("compliance_rate")
            rate_str = str(rate) if rate is not None else "N/A"
            print(f"  {'constraint_following':<30} {'compliance_rate':<25} {rate_str:>8}")

    # Comparisons
    for comp_name, comp_data in comparisons.items():
        wr = comp_data.get("win_rate")
        reg = comp_data.get("regression", {})
        if wr is not None:
            print(f"\n  vs {comp_name}: win_rate={wr}")
        if reg.get("regressions", 0) > 0:
            print(f"  ⚠ Regressions: {reg['regressions']}")
            for r in reg.get("regression_details", []):
                print(f"    - {r['category']}: {r['baseline_overall']} → {r['current_overall']} ({r['delta']:+.2f})")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate eval results into summary metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Programmatic scorers run on raw outputs without a judge:\n"
            "  json_validity, schema_validity, entity_f1, exact_match,\n"
            "  numeric_match, slop_phrases, output_length, constraint_following\n\n"
            "Each scorer activates only when the relevant metadata exists\n"
            "in the eval prompt (e.g., expected_fields, gold_answer, etc.).\n"
        ),
    )
    parser.add_argument("--run-id", required=True, help="Run ID to aggregate")
    parser.add_argument("--compare-with", nargs="*", default=[], help="Baseline run IDs to compare against")
    parser.add_argument(
        "--judge-source", choices=["api", "mock", "all"], default="all",
        help="Filter judge scores by source: api, mock, or all (default: all)",
    )
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "results" / "evals" / args.run_id

    # Load data
    scores_path = run_dir / "judge_scores.jsonl"
    outputs_path = run_dir / "outputs.jsonl"
    eval_set_path = PROJECT_ROOT / "data" / "eval" / "small_model_eval_v1.jsonl"

    scores = load_jsonl(scores_path)
    outputs = load_jsonl(outputs_path)

    if not scores:
        log.warning(f"No judge scores found at {scores_path}")
    if not outputs:
        log.error(f"No outputs found at {outputs_path}")
        sys.exit(1)

    log.info(f"Loaded {len(scores)} scores, {len(outputs)} outputs")

    # Filter scores by judge_source
    if args.judge_source != "all":
        original_count = len(scores)
        scores = [s for s in scores if s.get("judge_source", "unknown") == args.judge_source]
        log.info(f"Filtered to {len(scores)} scores with judge_source='{args.judge_source}' (was {original_count})")
        if not scores:
            log.warning(f"No scores remain after filtering by judge_source='{args.judge_source}'")

    # Compute judge_source summary BEFORE filtering for the summary field
    all_scores = load_jsonl(scores_path)  # Reload all for the summary
    judge_source_summary = compute_judge_source_summary(all_scores)

    # Aggregate pointwise scores
    pointwise_scores = [s for s in scores if s.get("mode") == "pointwise"]
    pairwise_scores = [s for s in scores if s.get("mode") == "pairwise"]

    pointwise_agg = aggregate_pointwise(pointwise_scores)
    format_metrics = compute_format_metrics(outputs)

    # Pairwise win rates
    pairwise_agg = {}
    if pairwise_scores:
        wins = sum(1 for s in pairwise_scores if s.get("winner") == "model_a")
        losses = sum(1 for s in pairwise_scores if s.get("winner") == "model_b")
        ties = sum(1 for s in pairwise_scores if s.get("winner") == "tie")
        total = len(pairwise_scores)
        pairwise_agg = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": round(wins / total, 3) if total else 0,
            "total": total,
        }

    # Run programmatic scorers
    eval_data_map = {}
    if eval_set_path.exists():
        for entry in load_jsonl(eval_set_path):
            eid = entry.get("id") or entry.get("eval_id", "")
            if eid:
                eval_data_map[eid] = entry
        log.info(f"Loaded {len(eval_data_map)} eval entries for programmatic scoring")
    else:
        log.warning(f"Eval set not found at {eval_set_path}, programmatic scorers will have limited metadata")

    # Merge output eval_ids with eval data
    for output in outputs:
        eid = output.get("eval_id", "")
        if eid and eid not in eval_data_map:
            eval_data_map[eid] = {}  # No metadata, scorers will skip

    programmatic_metrics = run_programmatic_scorers(outputs, eval_data_map)
    log.info(f"Programmatic scorers produced metrics for {len(programmatic_metrics)} scorer types")

    # Comparisons with baselines
    comparisons = {}
    for baseline_id in args.compare_with:
        baseline_dir = PROJECT_ROOT / "results" / "evals" / baseline_id
        baseline_agg_path = baseline_dir / "aggregate.json"
        baseline_agg = load_json(baseline_agg_path)
        if not baseline_agg:
            log.warning(f"No aggregate found for baseline {baseline_id}, skipping")
            continue

        baseline_pointwise = baseline_agg.get("pointwise", {})
        win_rate = compute_winrate(pointwise_agg, baseline_pointwise)
        regression = compute_regression_summary(pointwise_agg, baseline_pointwise)
        comparisons[baseline_id] = {
            "win_rate": win_rate,
            "regression": regression,
            "baseline_avg_scores": baseline_pointwise.get("avg_scores", {}),
        }

    # Build aggregate
    aggregate = {
        "run_id": args.run_id,
        "pointwise": pointwise_agg,
        "pairwise": pairwise_agg,
        "format": format_metrics,
        "comparisons": comparisons,
        "programmatic_metrics": programmatic_metrics,
        "judge_source_summary": judge_source_summary,
    }

    # Save
    agg_path = run_dir / "aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    log.info(f"Saved aggregate to {agg_path}")

    # Print summary
    has_mock = "mock" in judge_source_summary
    print_summary_table(args.run_id, aggregate, comparisons,
                        judge_source_summary=judge_source_summary if has_mock else None)


if __name__ == "__main__":
    main()
