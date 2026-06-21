"""Build final report from experiment results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mi_atlas.experiment_registry import load_registry, count_by_status, count_by_type
from mi_atlas.component_atlas import load_atlas, confidence_distribution
from mi_atlas.utils import PROJECT_ROOT


def main():
    registry = load_registry()
    atlas = load_atlas()
    status_counts = count_by_status()
    type_counts = count_by_type()
    conf_dist = confidence_distribution()

    print("=== Report Summary ===")
    print(f"Total experiments: {len(registry)}")
    print(f"By status: {status_counts}")
    print(f"By type: {type_counts}")
    print(f"Atlas entries: {len(atlas)}")
    print(f"Confidence: {conf_dist}")
    print(f"\nSee reports/final_report.md for the full report.")


if __name__ == "__main__":
    main()
