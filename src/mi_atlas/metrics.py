"""Metrics for evaluating model behaviour on task families."""

import math
import json
import re
from typing import Any

import torch
import numpy as np


def target_logprob(logprobs: torch.Tensor, target_token_id: int) -> float:
    """Log probability of a single target token."""
    return logprobs[target_token_id].item()


def target_logprob_sequence(logprobs: torch.Tensor, target_ids: list[int]) -> float:
    """Sum of log probabilities for a sequence of target tokens."""
    total = 0.0
    for i, tid in enumerate(target_ids):
        if i < logprobs.shape[0]:
            total += logprobs[i, tid].item() if logprobs.dim() == 2 else logprobs[tid].item()
    return total


def target_vs_wrong_diff(
    logprobs: torch.Tensor, target_id: int, wrong_id: int
) -> float:
    """Difference in log probability between target and wrong token."""
    return logprobs[target_id].item() - logprobs[wrong_id].item()


def exact_match(predicted: str, target: str) -> bool:
    """Exact string match after stripping whitespace."""
    return predicted.strip() == target.strip()


def exact_match_score(predicted: str, target: str) -> float:
    """1.0 if exact match, 0.0 otherwise."""
    return 1.0 if exact_match(predicted, target) else 0.0


def valid_json(text: str) -> bool:
    """Check if text is valid JSON."""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def valid_json_score(text: str) -> float:
    """1.0 if valid JSON, 0.0 otherwise."""
    return 1.0 if valid_json(text) else 0.0


def required_json_keys(text: str, required: list[str]) -> float:
    """Fraction of required keys present in JSON output."""
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return 0.0
        present = sum(1 for k in required if k in obj)
        return present / len(required) if required else 1.0
    except (json.JSONDecodeError, TypeError):
        return 0.0


def ast_parse_success(code: str) -> float:
    """1.0 if code parses as valid Python AST, 0.0 otherwise."""
    import ast
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError:
        return 0.0


def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[len(s2)]


def edit_distance_score(predicted: str, target: str) -> float:
    """Normalized edit distance score (1.0 = perfect match)."""
    dist = edit_distance(predicted.strip(), target.strip())
    max_len = max(len(predicted.strip()), len(target.strip()), 1)
    return 1.0 - (dist / max_len)


def token_entropy(logprobs: torch.Tensor) -> float:
    """Entropy of the token distribution."""
    probs = torch.softmax(logprobs, dim=-1)
    log_probs = torch.log_softmax(logprobs, dim=-1)
    entropy = -(probs * log_probs).sum().item()
    return entropy


def top_token_concentration(logprobs: torch.Tensor, k: int = 5) -> float:
    """Fraction of probability mass in top-k tokens."""
    probs = torch.softmax(logprobs, dim=-1)
    topk = torch.topk(probs, k)
    return topk.values.sum().item()


def format_violation_count(text: str, family: str = "json_schema") -> int:
    """Count format violations. Higher = worse."""
    violations = 0
    if family == "json_schema":
        if not valid_json(text):
            violations += 1
        if len(text.strip()) == 0:
            violations += 1
    elif family == "delimiter_tracking":
        # Count unbalanced brackets
        for open_b, close_b in [("(", ")"), ("[", "]"), ("{", "}")]:
            if text.count(open_b) != text.count(close_b):
                violations += 1
    return violations


def hallucination_flag(text: str, expected_refusal: bool = True) -> float:
    """Check if model expresses uncertainty or refuses appropriately.

    Returns 1.0 if appropriate uncertainty expressed, 0.0 if hallucinated.
    """
    uncertainty_phrases = [
        "i don't know", "i'm not sure", "uncertain", "unknown",
        "cannot determine", "not enough information", "insufficient",
        "no way to know", "impossible to determine", "i can't",
        "cannot be determined", "not available", "no data",
        "i do not have", "beyond my", "outside my",
    ]
    text_lower = text.strip().lower()

    # Check if any uncertainty phrase is present
    for phrase in uncertainty_phrases:
        if phrase in text_lower:
            return 1.0

    # If the answer is "unknown" and model says so
    if text_lower in ["unknown", "undefined", "n/a"]:
        return 1.0

    # Model likely hallucinated a specific answer
    return 0.0


# ── Patching / ablation metrics ─────────────────────────────────────

def ablation_effect(metric_original: float, metric_ablated: float) -> float:
    """Ablation effect: original - ablated. Positive = ablation hurt."""
    return metric_original - metric_ablated


def normalized_recovery(
    metric_patched: float,
    metric_corrupt: float,
    metric_clean: float,
    epsilon: float = 1e-8,
) -> float:
    """Normalized recovery from patching.

    = (patched - corrupt) / max(epsilon, clean - corrupt)
    """
    denom = max(epsilon, metric_clean - metric_corrupt)
    return (metric_patched - metric_corrupt) / denom


def patch_score(metric_patched: float, metric_corrupt: float) -> float:
    """Raw patch score: patched - corrupt."""
    return metric_patched - metric_corrupt


def skill_delta(metric_trained: float, metric_base: float) -> float:
    """Skill change after training."""
    return metric_trained - metric_base


def adapter_effect(metric_with_adapter: float, metric_base: float) -> float:
    """Effect of adapter: with - base."""
    return metric_with_adapter - metric_base


def adapter_specificity(
    target_skill_delta: float, control_skill_deltas: list[float]
) -> float:
    """How specific an adapter is to target skill vs controls."""
    mean_control = sum(control_skill_deltas) / len(control_skill_deltas) if control_skill_deltas else 0.0
    return target_skill_delta - mean_control


def localization_score(
    component_delta_target: float, component_delta_controls: list[float], epsilon: float = 1e-8
) -> float:
    """Component delta for target / mean component delta for controls."""
    mean_control = sum(component_delta_controls) / len(component_delta_controls) if component_delta_controls else epsilon
    return component_delta_target / max(epsilon, abs(mean_control))


# ── Metric registry ─────────────────────────────────────────────────

METRIC_FUNCTIONS = {
    "target_logprob": target_logprob,
    "target_vs_wrong_diff": target_vs_wrong_diff,
    "exact_match": exact_match_score,
    "valid_json": valid_json_score,
    "required_keys": required_json_keys,
    "ast_parse_success": ast_parse_success,
    "edit_distance": edit_distance_score,
    "token_entropy": token_entropy,
    "top_concentration": top_token_concentration,
    "format_violation_count": format_violation_count,
    "hallucination_flag": hallucination_flag,
}


def compute_metric(metric_type: str, **kwargs) -> float:
    """Compute a metric by name."""
    fn = METRIC_FUNCTIONS.get(metric_type)
    if fn is None:
        raise ValueError(f"Unknown metric: {metric_type}")
    return fn(**kwargs)
