#!/usr/bin/env python3
"""Generate a claim card from an experiment result file.

Usage:
    python scripts/generate_claim_card.py --experiment P2-STEER-001 --result experiments/results/steering_migration_0.5b.json
    python scripts/generate_claim_card.py --experiment P2-ABL-001 --result experiments/results/ablation_controls_0.5b.json --verdict confirmed
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_git_hash():
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_gpu_info():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "N/A"


def get_python_version():
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_torch_version():
    if TORCH_AVAILABLE:
        return torch.__version__
    return "N/A"


def hash_file(path):
    """SHA256 of a file."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        return "N/A"


def load_experiment_from_registry(exp_id, registry_path=None):
    """Find experiment entry in registry.jsonl."""
    if registry_path is None:
        registry_path = Path(__file__).parent.parent / "experiments" / "registry.jsonl"
    
    with open(registry_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("id") == exp_id:
                return entry
    return None


def load_result(result_path):
    """Load result JSON."""
    with open(result_path) as f:
        return json.load(f)


def extract_key_metrics(result_data):
    """Extract key metrics from result data."""
    metrics = {}
    
    # Try common metric keys
    for key in ["key_metrics", "summary_metrics", "metrics"]:
        if key in result_data:
            if isinstance(result_data[key], dict):
                metrics.update(result_data[key])
    
    # Extract from nested results
    if "results" in result_data and isinstance(result_data["results"], list):
        for r in result_data["results"]:
            if isinstance(r, dict) and "metrics" in r:
                for k, v in r["metrics"].items():
                    if k not in metrics:
                        metrics[k] = v
    
    return metrics


def extract_seeds(result_data):
    """Extract per-seed results if available."""
    seeds = {}
    if "seed_results" in result_data:
        seeds = result_data["seed_results"]
    elif "seeds" in result_data:
        seeds = result_data["seeds"]
    return seeds


def extract_controls(result_data):
    """Extract control results if available."""
    controls = []
    if "controls" in result_data:
        controls = result_data["controls"]
    elif "results" in result_data:
        for r in result_data.get("results", []):
            if isinstance(r, dict) and r.get("is_control"):
                controls.append(r)
    return controls


def generate_claim_card(exp_id, result_path, title=None, claim=None, verdict="inconclusive"):
    """Generate a claim card markdown."""
    
    registry_entry = load_experiment_from_registry(exp_id)
    result_data = load_result(result_path)
    
    # Build metadata
    model = result_data.get("model", registry_entry.get("model", "unknown") if registry_entry else "unknown")
    seed = result_data.get("seed", result_data.get("seeds", [1]))
    config = result_data.get("config", "configs/experiment_defaults.yaml")
    
    metrics = extract_key_metrics(result_data)
    seeds = extract_seeds(result_data)
    controls = extract_controls(result_data)
    
    # Build markdown
    title = title or (registry_entry.get("title", exp_id) if registry_entry else exp_id)
    claim = claim or "To be filled in."
    
    lines = []
    lines.append(f"# {exp_id}: {title}")
    lines.append("")
    lines.append("## Claim")
    lines.append(claim)
    lines.append("")
    lines.append("## Result")
    
    if metrics:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            else:
                lines.append(f"| {k} | {v} |")
    else:
        lines.append("No key metrics extracted. Check raw result file.")
    lines.append("")
    
    lines.append("## Controls")
    if controls:
        lines.append("| Control | Result | Delta vs Treatment |")
        lines.append("|---------|--------|--------------------|")
        for c in controls:
            name = c.get("name", "unknown")
            result = c.get("result", "N/A")
            delta = c.get("delta", "N/A")
            lines.append(f"| {name} | {result} | {delta} |")
    else:
        lines.append("No controls recorded. Run control experiments before claiming.")
    lines.append("")
    
    lines.append("## Seeds")
    if seeds:
        lines.append("| Seed | Metric | Value |")
        lines.append("|------|--------|-------|")
        for seed_id, seed_data in seeds.items():
            if isinstance(seed_data, dict):
                for k, v in seed_data.items():
                    lines.append(f"| {seed_id} | {k} | {v} |")
            else:
                lines.append(f"| {seed_id} | - | {seed_data} |")
    else:
        lines.append(f"Single seed: {seed}")
    lines.append("")
    
    lines.append("## Artifacts")
    lines.append(f"- Raw output: `{result_path}`")
    lines.append(f"- Config: `{config}`")
    lines.append(f"- Script: `scripts/{exp_id.lower().replace('-', '_')}.py`")
    lines.append(f"- Git commit: {get_git_hash()}")
    lines.append("")
    
    lines.append("## Environment")
    lines.append(f"- Model: {model}")
    lines.append(f"- Git commit: {get_git_hash()}")
    lines.append(f"- Python: {get_python_version()}")
    lines.append(f"- Torch: {get_torch_version()}")
    lines.append(f"- GPU: {get_gpu_info()}")
    lines.append(f"- Timestamp: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    
    lines.append("## Interpretation")
    lines.append("To be filled in after analysis.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("1. Single-seed results (need 3 seeds for HIGH confidence)")
    lines.append("2. Zero ablation only (need mean/resample controls)")
    lines.append("3. Synthetic tasks only (need real-task validation)")
    lines.append("")
    
    lines.append("## Verdict")
    lines.append(verdict)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate claim card")
    parser.add_argument("--experiment", required=True, help="Experiment ID (e.g., P2-STEER-001)")
    parser.add_argument("--result", required=True, help="Path to result JSON file")
    parser.add_argument("--title", default=None, help="Override title")
    parser.add_argument("--claim", default=None, help="Claim text")
    parser.add_argument("--verdict", default="inconclusive",
                       choices=["confirmed", "partially_confirmed", "rejected", "inconclusive"])
    parser.add_argument("--output", default=None, help="Output path (default: reports/claims/<exp_id>.md)")
    args = parser.parse_args()
    
    if not os.path.exists(args.result):
        print(f"ERROR: Result file not found: {args.result}")
        sys.exit(1)
    
    card = generate_claim_card(args.experiment, args.result, args.title, args.claim, args.verdict)
    
    output_path = args.output
    if output_path is None:
        output_path = Path(__file__).parent.parent / "reports" / "claims" / f"{args.experiment}.md"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(card)
    
    print(f"Claim card written: {output_path}")
    print(f"Experiment: {args.experiment}")
    print(f"Verdict: {args.verdict}")


if __name__ == "__main__":
    main()
