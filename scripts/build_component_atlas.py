"""Build component atlas from experiment results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mi_atlas.component_atlas import load_atlas, confidence_distribution, render_atlas_markdown
from mi_atlas.experiment_registry import load_registry
from mi_atlas.utils import PROJECT_ROOT


def main():
    atlas = load_atlas()
    dist = confidence_distribution()

    print(f"Component Atlas: {len(atlas)} entries")
    for level, count in dist.items():
        print(f"  {level}: {count}")

    # Render markdown
    md = render_atlas_markdown()
    md_path = PROJECT_ROOT / "reports" / "component_atlas.md"
    md_path.write_text(md)
    print(f"Markdown rendered to {md_path}")


if __name__ == "__main__":
    main()
