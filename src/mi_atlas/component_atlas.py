"""Component atlas: cataloguing causal component-behaviour relationships."""

from .utils import append_jsonl, load_jsonl, save_json, PROJECT_ROOT
from pathlib import Path

ATLAS_PATH = PROJECT_ROOT / "reports" / "component_atlas.jsonl"
ATLAS_MD_PATH = PROJECT_ROOT / "reports" / "component_atlas.md"


def add_atlas_entry(entry: dict) -> None:
    """Append an entry to the component atlas JSONL."""
    # Validate required fields
    required = [
        "component_id", "component_type", "layer",
        "claimed_behaviour", "task_families",
        "positive_effects", "confidence",
    ]
    for field in required:
        if field not in entry:
            raise ValueError(f"Missing required field: {field}")

    append_jsonl(entry, ATLAS_PATH)


def load_atlas() -> list[dict]:
    """Load the full component atlas."""
    return load_jsonl(ATLAS_PATH)


def get_entries_for_layer(layer: int) -> list[dict]:
    """Get all atlas entries for a specific layer."""
    return [e for e in load_atlas() if e.get("layer") == layer]


def get_entries_for_family(family: str) -> list[dict]:
    """Get all atlas entries mentioning a task family."""
    return [e for e in load_atlas() if family in e.get("task_families", [])]


def confidence_distribution() -> dict[str, int]:
    """Count entries by confidence level."""
    counts = {"low": 0, "medium": 0, "high": 0, "very_high": 0}
    for entry in load_atlas():
        conf = entry.get("confidence", "low")
        if conf in counts:
            counts[conf] += 1
    return counts


def render_atlas_markdown() -> str:
    """Render the atlas as a human-readable markdown file."""
    entries = load_atlas()
    dist = confidence_distribution()

    lines = [
        "# Component Atlas",
        "",
        f"Total entries: {len(entries)}",
        "",
        "## Confidence distribution",
        "",
        "| Level | Count |",
        "|-------|-------|",
    ]
    for level, count in dist.items():
        lines.append(f"| {level} | {count} |")

    lines.append("")
    lines.append("## Entries")
    lines.append("")

    for entry in entries:
        lines.append(f"### {entry['component_id']}")
        lines.append(f"- **Type**: {entry['component_type']}")
        lines.append(f"- **Layer**: {entry['layer']}")
        lines.append(f"- **Behaviour**: {entry['claimed_behaviour']}")
        lines.append(f"- **Task families**: {', '.join(entry.get('task_families', []))}")
        lines.append(f"- **Confidence**: {entry['confidence']}")
        if entry.get("limitations"):
            lines.append(f"- **Limitations**: {entry['limitations']}")
        lines.append("")

    return "\n".join(lines)
