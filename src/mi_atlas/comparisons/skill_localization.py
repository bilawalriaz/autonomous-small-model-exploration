"""Skill localisation: identify which components matter for which skills."""

import torch
import numpy as np


def skill_localization_score(
    ablation_effects: dict[str, dict[str, float]],
    families: list[str],
) -> dict[str, dict[str, float]]:
    """Compute localisation scores: how specific is each component to each family?

    Args:
        ablation_effects: {component: {family: effect_size}}
        families: List of family names

    Returns:
        {component: {family: localisation_ratio}}
    """
    scores = {}
    for comp, family_effects in ablation_effects.items():
        values = [family_effects.get(f, 0.0) for f in families]
        total = sum(abs(v) for v in values)
        if total < 1e-10:
            scores[comp] = {f: 0.0 for f in families}
            continue

        scores[comp] = {}
        for f in families:
            eff = family_effects.get(f, 0.0)
            scores[comp][f] = abs(eff) / total

    return scores


def find_top_components(
    localization: dict[str, dict[str, float]],
    family: str,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Find top-k components most localised to a family."""
    scores = [(comp, loc.get(family, 0.0)) for comp, loc in localization.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
