"""Training curriculum definitions."""

# Curriculum definitions for experiment family G

CURRICULA = {
    "copying_first": ["copying", "delimiter_tracking", "json_schema", "code_semantics"],
    "code_first": ["code_semantics", "json_schema", "delimiter_tracking", "copying"],
    "easy_to_hard": ["copying", "factual_recall", "delimiter_tracking", "arithmetic", "json_schema", "code_semantics"],
    "hard_to_easy": ["code_semantics", "json_schema", "arithmetic", "delimiter_tracking", "factual_recall", "copying"],
    "mixed": ["mixed"],
    "style_first": ["verbosity_control", "refusal_compliance", "code_semantics", "copying"],
    "code_then_style": ["code_semantics", "json_schema", "verbosity_control", "refusal_compliance"],
}


def get_curriculum(name: str) -> list[str]:
    """Get curriculum order by name."""
    if name not in CURRICULA:
        raise ValueError(f"Unknown curriculum: {name}. Available: {list(CURRICULA.keys())}")
    return CURRICULA[name]
