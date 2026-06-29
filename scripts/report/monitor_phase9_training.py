#!/usr/bin/env python3
"""Monitor Phase 9 format ablation training on aero.
When all 6 formats complete, collect results and update the report.

Usage: python scripts/report/monitor_phase9_training.py [--poll-once] [--update-now]
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AERO = "billz@aero.tail9cc5b.ts.net"
REMOTE_DIR = "~/work/autonomous-small-model-exploration"

FORMATS = [
    "alpaca_flat", "single_turn_chat", "multi_turn_concise",
    "multi_turn_verbose", "structured_terse", "bad_format_control"
]

def ssh_cmd(cmd: str) -> tuple[int, str]:
    """Run a command on aero via SSH."""
    result = subprocess.run(
        ["ssh", AERO, f"cd {REMOTE_DIR} && {cmd}"],
        capture_output=True, text=True, timeout=30
    )
    return result.returncode, result.stdout.strip()

def check_training_status() -> dict:
    """Check which formats have completed training."""
    status = {}
    for fmt in FORMATS:
        run_id = f"lfm2_230m_format_ablation_{fmt}"
        rc, out = ssh_cmd(f"cat adapters/{run_id}/metadata.json 2>/dev/null")
        if rc == 0 and out:
            try:
                meta = json.loads(out)
                status[fmt] = {
                    "complete": True,
                    "loss": meta.get("final_loss"),
                    "steps": meta.get("global_step"),
                    "adapter_dir": f"adapters/{run_id}",
                    "metadata": meta
                }
            except json.JSONDecodeError:
                status[fmt] = {"complete": False, "error": "invalid metadata"}
        else:
            # Check if still running
            rc2, ps_out = ssh_cmd(f"ps aux | grep train_lfm2 | grep '{fmt}' | grep -v grep")
            if ps_out:
                status[fmt] = {"complete": False, "running": True}
            else:
                status[fmt] = {"complete": False, "running": False}
    return status

def collect_results(status: dict) -> dict:
    """Collect all training results into a summary."""
    results = {}
    for fmt, info in status.items():
        if info["complete"]:
            meta = info["metadata"]
            results[fmt] = {
                "final_loss": meta.get("final_loss"),
                "global_step": meta.get("global_step"),
                "train_size": meta.get("train_size"),
                "eval_size": meta.get("eval_size"),
                "adapter_config": meta.get("adapter_config"),
                "training_metrics": meta.get("training_metrics"),
            }
    return results

def update_report(results: dict):
    """Update the Phase 9 HTML report with actual training results."""
    report_path = PROJECT_ROOT / "docs" / "09-data-format-ablation.html"
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        return

    content = report_path.read_text()

    # Build results table rows
    rows = []
    for fmt in FORMATS:
        r = results.get(fmt)
        if r:
            loss = f"{r['final_loss']:.4f}" if r['final_loss'] else "N/A"
            rows.append(
                f'<tr><td>{fmt.replace("_", " ").title()}</td>'
                f'<td>{loss}</td>'
                f'<td style="color: #999;">—</td>'
                f'<td style="color: #999;">—</td>'
                f'<td style="color: #999;">—</td>'
                f'<td style="color: #999;">—</td>'
                f'<td style="color: #999;">—</td></tr>'
            )
        else:
            rows.append(
                f'<tr><td>{fmt.replace("_", " ").title()}</td>'
                f'<td colspan="6" style="color: #999; font-style: italic;">Not completed</td></tr>'
            )

    table_html = "\n        ".join(rows)

    # Replace the placeholder table
    old_table_start = '<tr><td>Alpaca Flat</td><td colspan="6" style="color: #999; font-style: italic;">Training…</td></tr>'
    old_table_end = '<tr><td>Bad Format Control</td><td colspan="6" style="color: #999; font-style: italic;">Queued</td></tr>'

    if old_table_start in content:
        # Find the old rows section and replace it
        import re
        pattern = r'<tr><td>Alpaca Flat.*?</tr>\s*<tr><td>Single-Turn Chat.*?</tr>\s*<tr><td>Multi-Turn Concise.*?</tr>\s*<tr><td>Multi-Turn Verbose.*?</tr>\s*<tr><td>Structured Terse.*?</tr>\s*<tr><td>Bad Format Control.*?</tr>'
        replacement = table_html
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Update the status callout
    if "Training in progress" in content:
        n_complete = sum(1 for r in results.values() if r.get("final_loss"))
        content = content.replace(
            '<strong>Training in progress.</strong>',
            f'<strong>Training complete.</strong> {n_complete}/{len(FORMATS)} format ablations finished.'
        )

    report_path.write_text(content)
    print(f"Updated report: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Monitor Phase 9 training")
    parser.add_argument("--poll-once", action="store_true", help="Check once and exit")
    parser.add_argument("--update-now", action="store_true", help="Update report with current results")
    args = parser.parse_args()

    print(f"Checking training status on aero at {datetime.now(timezone.utc).isoformat()}...")
    status = check_training_status()

    complete = [f for f, s in status.items() if s["complete"]]
    running = [f for f, s in status.items() if s.get("running")]

    print(f"  Complete: {len(complete)}/{len(FORMATS)} — {', '.join(complete) if complete else 'none'}")
    if running:
        print(f"  Running: {', '.join(running)}")

    for fmt, info in status.items():
        if info["complete"]:
            loss = info.get("loss", "?")
            print(f"  ✓ {fmt}: loss={loss:.4f}" if isinstance(loss, float) else f"  ✓ {fmt}: {loss}")
        elif info.get("running"):
            print(f"  ⏳ {fmt}: training...")
        else:
            print(f"  ✗ {fmt}: not started / failed")

    if len(complete) == len(FORMATS) or args.update_now:
        print("\nAll formats complete! Collecting results...")
        results = collect_results(status)

        # Save results locally
        results_dir = PROJECT_ROOT / "results" / "evals" / "format_ablation_quality"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {results_path}")

        # Update the report
        update_report(results)

        # Save summary JSON
        summary = {
            "experiment": "format_ablation_quality",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "formats": results,
            "best_format": min(results.items(), key=lambda x: x[1].get("final_loss", float("inf")))[0] if results else None,
        }
        summary_path = results_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Best format: {summary['best_format']}")
        print(f"Summary: {summary_path}")
        return 0

    if args.poll_once:
        return 0 if len(complete) == len(FORMATS) else 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
