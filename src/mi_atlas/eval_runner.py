"""Baseline evaluation runner."""

import torch
from typing import Any

from .backend import BackendBase
from .task_suite import TaskSuite, TaskExample
from .metrics import (
    exact_match_score, valid_json_score, required_json_keys,
    ast_parse_success, hallucination_flag, token_entropy,
    top_token_concentration, edit_distance_score,
    format_violation_count,
)
from .utils import save_json, now_iso, PROJECT_ROOT


def evaluate_example(
    backend: BackendBase,
    example: TaskExample,
    max_new_tokens: int = 50,
) -> dict:
    """Evaluate a single task example.

    Returns dict with metrics and generation info.
    """
    prompt = example.clean_prompt
    target = example.target

    # Generate
    generated = backend.generate(prompt, max_new_tokens=max_new_tokens)

    # Get next-token logits for logprob-based metrics
    try:
        next_logits = backend.get_next_token_logits(prompt)
        next_logprobs = torch.log_softmax(next_logits, dim=-1)
        entropy = token_entropy(next_logits)
        top5_conc = top_token_concentration(next_logits, k=5)
    except Exception:
        next_logprobs = None
        entropy = None
        top5_conc = None

    # Compute target logprob if target tokenizes to something
    target_logprob_val = None
    if next_logprobs is not None:
        try:
            target_ids = backend.tokenizer.encode(target, add_special_tokens=False)
            if len(target_ids) == 1:
                target_logprob_val = next_logprobs[target_ids[0]].item()
        except Exception:
            pass

    # Metric computation
    metrics = {}

    # Always compute exact match
    metrics["exact_match"] = exact_match_score(generated, target)

    # Compute edit distance
    metrics["edit_distance"] = edit_distance_score(generated, target)

    # Family-specific metrics
    if example.family == "json_schema":
        metrics["valid_json"] = valid_json_score(generated)
        required = example.metadata.get("required_keys", [])
        if required:
            metrics["required_keys"] = required_json_keys(generated, required)
        metrics["format_violation_count"] = format_violation_count(generated, "json_schema")
    elif example.family == "code_syntax" or example.family == "code_semantics":
        metrics["ast_parse_success"] = ast_parse_success(generated)
    elif example.family == "uncertainty_signalling":
        metrics["hallucination_flag"] = hallucination_flag(generated)

    # Universal
    if target_logprob_val is not None:
        metrics["target_logprob"] = target_logprob_val
    if entropy is not None:
        metrics["token_entropy"] = entropy
    if top5_conc is not None:
        metrics["top5_concentration"] = top5_conc
    metrics["generation_length"] = len(generated)

    return {
        "example_id": example.id,
        "family": example.family,
        "prompt": prompt[:200],
        "target": target,
        "generated": generated[:200],
        "metrics": metrics,
    }


def evaluate_suite(
    backend: BackendBase,
    suite: TaskSuite,
    max_new_tokens: int = 50,
    split: str | None = None,
) -> dict:
    """Evaluate entire task suite.

    Returns dict with per-example results and summary statistics.
    """
    if split:
        suite = suite.filter_by_split(split)

    results = []
    for example in suite:
        try:
            result = evaluate_example(backend, example, max_new_tokens)
            results.append(result)
        except Exception as e:
            results.append({
                "example_id": example.id,
                "family": example.family,
                "error": str(e),
                "metrics": {},
            })

    # Aggregate by family
    family_scores: dict[str, dict[str, list[float]]] = {}
    for r in results:
        fam = r["family"]
        if fam not in family_scores:
            family_scores[fam] = {}
        for metric_name, value in r.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                if metric_name not in family_scores[fam]:
                    family_scores[fam][metric_name] = []
                family_scores[fam][metric_name].append(value)

    # Compute means
    family_means: dict[str, dict[str, float]] = {}
    for fam, metrics in family_scores.items():
        family_means[fam] = {}
        for metric_name, values in metrics.items():
            family_means[fam][metric_name] = sum(values) / len(values) if values else 0.0

    # Overall mean of primary metric per family
    primary_metric_by_family = {}
    for fam, means in family_means.items():
        # Use exact_match as primary if available, else target_logprob
        if "exact_match" in means:
            primary_metric_by_family[fam] = means["exact_match"]
        elif "target_logprob" in means:
            primary_metric_by_family[fam] = means["target_logprob"]

    summary = {
        "total_examples": len(results),
        "errors": sum(1 for r in results if "error" in r),
        "family_means": family_means,
        "primary_metric_by_family": primary_metric_by_family,
        "overall_mean": (
            sum(primary_metric_by_family.values()) / len(primary_metric_by_family)
            if primary_metric_by_family else 0.0
        ),
    }

    return {
        "timestamp": now_iso(),
        "results": results,
        "summary": summary,
    }
