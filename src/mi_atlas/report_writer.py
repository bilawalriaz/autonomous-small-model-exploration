"""Report generation utilities."""

from pathlib import Path
from datetime import datetime

from .utils import PROJECT_ROOT


def update_progress(completed_items: dict[str, bool] | None = None) -> None:
    """Update progress.md with completion status."""
    path = PROJECT_ROOT / "progress.md"
    content = path.read_text()

    # Update date
    now = datetime.now().strftime("%Y-%m-%d")
    content = content.replace("- Date:", f"- Date: {now}")

    if completed_items:
        for item, done in completed_items.items():
            checkbox = "[x]" if done else "[ ]"
            content = content.replace(f"- [ ] {item}", f"- {checkbox} {item}")

    path.write_text(content)


def append_to_findings(section: str, text: str) -> None:
    """Append text to a section in current_findings.md."""
    path = PROJECT_ROOT / "reports" / "current_findings.md"
    content = path.read_text()

    marker = f"## {section}"
    if marker in content:
        idx = content.index(marker) + len(marker)
        content = content[:idx] + "\n" + text + "\n" + content[idx:]
    else:
        content += f"\n## {section}\n{text}\n"

    path.write_text(content)
