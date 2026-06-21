"""Build the task suite and save to data/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mi_atlas.task_suite import build_default_suite
from mi_atlas.utils import PROJECT_ROOT


def main():
    print("Building task suite...")
    suite = build_default_suite()

    summary = suite.summary()
    print(f"  Total examples: {summary['total']}")
    print(f"  Families: {len(summary['families'])}")
    for fam, count in summary["families"].items():
        print(f"    {fam}: {count}")
    print(f"  Splits: {summary['splits']}")

    # Save
    output_path = PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json"
    suite.save(output_path)
    print(f"  Saved to: {output_path}")

    # Save clean/corrupt pairs separately
    pairs = []
    for ex in suite:
        if ex.corrupt_prompt:
            pairs.append({
                "id": ex.id,
                "family": ex.family,
                "clean": ex.clean_prompt,
                "corrupt": ex.corrupt_prompt,
                "target": ex.target,
                "wrong_target": ex.wrong_target,
            })

    pairs_path = PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v0.json"
    import json
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"  Clean/corrupt pairs: {len(pairs)} saved to {pairs_path}")


if __name__ == "__main__":
    main()
