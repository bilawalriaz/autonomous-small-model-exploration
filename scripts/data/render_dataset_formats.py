#!/usr/bin/env python3
"""Render canonical JSONL into all 6 format variants for Phase 9 ablation.

Reads a YAML experiment config and canonical source, produces all format
variants plus a manifest JSON.

Usage:
    python scripts/data/render_dataset_formats.py \
        --config configs/experiments/format_ablation_quality.yaml \
        --canonical data/canonical/phase9_pilot_300.jsonl \
        --output-dir data/sft/format_ablation/

Also accepts --seed (default 42) for deterministic rendering.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

# Import compile logic directly
from compile_sft_dataset import (
    ALL_FORMATS,
    compile_dataset,
    load_canonical,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_yaml_config(path: str) -> dict:
    """Minimal YAML parser for our config files (stdlib only).

    Handles the flat structure of format_ablation_quality.yaml.
    Returns dict with 'formats', 'canonical', 'experiment', etc.
    """
    result = {}
    current_section = None
    current_sub = None
    formats_list = []
    current_format = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip()
            if not stripped or stripped.lstrip().startswith("#"):
                continue

            # Count leading spaces for nesting
            indent = len(line) - len(line.lstrip())

            content = stripped.lstrip()
            if ":" not in content:
                continue

            key, _, val = content.partition(":")
            key = key.strip()
            val = val.strip()

            # Top-level sections
            if indent == 0:
                current_section = key
                current_sub = None
                if current_section not in result:
                    result[current_section] = {}
                continue

            # Second level
            if indent <= 4:
                if current_section == "formats" and key == "name":
                    # New format entry
                    if current_format:
                        formats_list.append(current_format)
                    current_format = {"name": val.strip('"').strip("'")}
                    continue
                elif current_format and val:
                    current_format[key] = val.strip('"').strip("'")
                    continue

                if val:
                    result[current_section] = result.get(current_section, {})
                    result[current_section][key] = val.strip('"').strip("'")
                current_sub = key

            # Third level (lists etc)
            if indent > 4 and current_format and val:
                current_format[key] = val.strip('"').strip("'")

    if current_format:
        formats_list.append(current_format)

    result["_formats_list"] = formats_list
    return result


def render_all_formats(
    canonical_path: str,
    output_dir: str,
    config_path: str | None = None,
    seed: int = 42,
) -> dict:
    """Render all 6 format variants and produce manifest.

    Returns manifest dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load config to get format-specific output paths if available
    config_formats = {}
    if config_path and os.path.exists(config_path):
        config = parse_yaml_config(config_path)
        for fmt_conf in config.get("_formats_list", []):
            name = fmt_conf.get("name", "")
            if name:
                config_formats[name] = fmt_conf

    # Load canonical to get count
    examples = load_canonical(canonical_path)
    canonical_count = len(examples)

    formats_meta = {}
    for fmt in ALL_FORMATS:
        # Determine output path
        fmt_conf = config_formats.get(fmt, {})
        if "output_path" in fmt_conf:
            output_path = fmt_conf["output_path"]
        else:
            output_path = os.path.join(output_dir, f"{fmt}.jsonl")

        logger.info("Rendering format: %s -> %s", fmt, output_path)
        meta = compile_dataset(
            canonical_path=canonical_path,
            fmt=fmt,
            output_path=output_path,
            seed=seed,
        )
        formats_meta[fmt] = meta

    manifest = {
        "canonical_source": canonical_path,
        "canonical_count": canonical_count,
        "formats": formats_meta,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Manifest written to %s", manifest_path)

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Render canonical JSONL into all 6 format variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to experiment YAML config (optional, for output paths).",
    )
    parser.add_argument(
        "--canonical",
        required=True,
        help="Path to canonical JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for all format variants.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for determinism (default: 42).",
    )
    args = parser.parse_args()

    manifest = render_all_formats(
        canonical_path=args.canonical,
        output_dir=args.output_dir,
        config_path=args.config,
        seed=args.seed,
    )

    # Summary
    print("\n=== Render Complete ===")
    print(f"Canonical source: {manifest['canonical_source']}")
    print(f"Canonical count:  {manifest['canonical_count']}")
    print(f"Formats rendered:  {len(manifest['formats'])}")
    for name, meta in manifest["formats"].items():
        print(f"  {name}: {meta['count']} examples -> {meta['path']}")
    print(f"Manifest: {args.output_dir}/manifest.json")


if __name__ == "__main__":
    main()
