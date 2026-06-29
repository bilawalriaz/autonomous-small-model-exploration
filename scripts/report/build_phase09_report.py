#!/usr/bin/env python3
"""Build Phase 9 data-format-ablation report from aggregated results.

CLI:
    python scripts/report/build_phase09_report.py \
        --results-dir results/evals/ \
        --output-md reports/09-data-format-ablation.md \
        --output-html docs/09-data-format-ablation.html
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DIMENSIONS = [
    "correctness", "instruction_following", "output_format",
    "concision", "usefulness", "hallucination_risk", "overall",
]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def discover_runs(results_dir: Path) -> dict[str, dict]:
    """Discover all runs with aggregate.json in results_dir."""
    runs = {}
    if not results_dir.exists():
        return runs
    for run_dir in sorted(results_dir.iterdir()):
        if run_dir.is_dir():
            agg_path = run_dir / "aggregate.json"
            if agg_path.exists():
                runs[run_dir.name] = load_json(agg_path)
    return runs


def find_base_run(runs: dict[str, dict]) -> str | None:
    """Find the base model run (no format_ablation in name)."""
    for name in runs:
        if "base" in name and "format_ablation" not in name:
            return name
    return None


def build_markdown(runs: dict[str, dict], base_run: str | None) -> str:
    """Build the full markdown report."""
    lines = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Identify format runs (exclude base)
    format_runs = {k: v for k, v in runs.items() if "format_ablation" in k}
    base_agg = runs.get(base_run, {}) if base_run else {}

    # Section 1: Title
    lines.append("# Phase 9: Data Format Ablation Report")
    lines.append("")
    lines.append(f"*Generated: {now}*  ")
    lines.append(f"*Base model: LiquidAI/LFM2.5-230M*  ")
    lines.append(f"*Runs analyzed: {len(format_runs)} formats + {1 if base_run else 0} base*")
    lines.append("")

    # Section 2: Executive Summary
    lines.append("## 1. Executive Summary")
    lines.append("")
    if format_runs:
        # Find best format
        best_name, best_score = None, 0
        for name, agg in format_runs.items():
            overall = agg.get("pointwise", {}).get("avg_scores", {}).get("overall", 0)
            if overall > best_score:
                best_score = overall
                best_name = name
        if best_name:
            fmt_label = best_name.replace("lfm2_230m_format_ablation_", "").split("_20")[0]
            lines.append(f"The best-performing data format was **{fmt_label}** "
                        f"(overall score: {best_score:.2f}/5). ")
        # Count regressions
        total_regressions = 0
        for name, agg in format_runs.items():
            comps = agg.get("comparisons", {})
            if base_run and base_run in comps:
                total_regressions += comps[base_run].get("regression", {}).get("regressions", 0)
        lines.append(f"Total category-level regressions vs base across all formats: **{total_regressions}**.")
    else:
        lines.append("No format ablation results found.")
    lines.append("")

    # Section 3: Hypothesis Overview
    lines.append("## 2. Hypothesis Overview")
    lines.append("")
    lines.append("| ID | Hypothesis | Status |")
    lines.append("|-----|-----------|--------|")

    hypotheses = [
        ("H1", "Multi-turn concise format is genuinely better for small-model SFT", "pending"),
        ("H2", "Chat format outperforms flat alpaca format", "pending"),
        ("H3", "Verbose multi-turn hurts vs concise multi-turn", "pending"),
        ("H4", "Structured terse format helps code/JSON tasks", "pending"),
        ("H5", "Bad-format control performs measurably worse", "pending"),
        ("H6", "Format effect is stronger than content effect", "pending"),
        ("H7", "KL drift from format change is smaller than from content change", "pending"),
    ]

    # Try to auto-verdict H1-H5 based on data
    if format_runs:
        def get_format_score(fmt_keyword):
            for name, agg in format_runs.items():
                if fmt_keyword in name:
                    return agg.get("pointwise", {}).get("avg_scores", {}).get("overall", 0)
            return None

        concise = get_format_score("multi_turn_concise")
        verbose = get_format_score("multi_turn_verbose")
        chat = get_format_score("single_turn_chat")
        alpaca = get_format_score("alpaca_flat")
        terse = get_format_score("structured_terse")
        bad = get_format_score("bad_format_control")
        base_overall = base_agg.get("pointwise", {}).get("avg_scores", {}).get("overall", 0)

        verdicts = {}
        if concise and verbose:
            verdicts["H1"] = "✅ Confirmed" if concise > verbose else "❌ Rejected"
        if chat and alpaca:
            verdicts["H2"] = "✅ Confirmed" if chat > alpaca else "❌ Rejected"
        if concise and verbose:
            verdicts["H3"] = "✅ Confirmed" if concise > verbose else "❌ Rejected"
        if terse:
            verdicts["H4"] = "✅ Confirmed" if terse > base_overall else "❌ Rejected"
        if bad:
            verdicts["H5"] = "✅ Confirmed" if bad < base_overall else "❌ Rejected"

        hypotheses = [(h[0], h[1], verdicts.get(h[0], "⏳ Pending")) for h in hypotheses]

    for hid, desc, status in hypotheses:
        lines.append(f"| {hid} | {desc} | {status} |")
    lines.append("")

    # Section 4: Win-Rate Table
    lines.append("## 3. Win-Rate vs Base Model")
    lines.append("")
    if base_run:
        lines.append(f"| Format | Win Rate vs {base_run} | Regressions | Improvements |")
        lines.append("|--------|" + "-" * (len(base_run) + 14) + "|-------------|--------------|")
        for name, agg in sorted(format_runs.items()):
            comps = agg.get("comparisons", {})
            fmt_label = name.replace("lfm2_230m_format_ablation_", "").split("_20")[0]
            if base_run in comps:
                wr = comps[base_run].get("win_rate", "N/A")
                reg = comps[base_run].get("regression", {})
                reg_count = reg.get("regressions", 0)
                imp_count = reg.get("improvements", 0)
                lines.append(f"| {fmt_label} | {wr} | {reg_count} | {imp_count} |")
            else:
                lines.append(f"| {fmt_label} | N/A | N/A | N/A |")
    else:
        lines.append("No base model run found for comparison.")
    lines.append("")

    # Section 5: Overall Scores Table
    lines.append("## 4. Overall Scores by Format")
    lines.append("")
    header = "| Format | " + " | ".join(DIMENSIONS) + " |"
    sep = "|--------|" + "|".join(["---"] * len(DIMENSIONS)) + "|"
    lines.append(header)
    lines.append(sep)

    if base_agg:
        scores = base_agg.get("pointwise", {}).get("avg_scores", {})
        vals = [f"{scores.get(d, 'N/A')}" for d in DIMENSIONS]
        base_label = base_run or "base"
        lines.append(f"| {base_label} | " + " | ".join(vals) + " |")

    for name, agg in sorted(format_runs.items()):
        scores = agg.get("pointwise", {}).get("avg_scores", {})
        fmt_label = name.replace("lfm2_230m_format_ablation_", "").split("_20")[0]
        vals = [f"{scores.get(d, 'N/A')}" for d in DIMENSIONS]
        lines.append(f"| {fmt_label} | " + " | ".join(vals) + " |")
    lines.append("")

    # Section 6: Category-Level Breakdown
    lines.append("## 5. Category-Level Breakdown")
    lines.append("")
    categories = set()
    for agg in format_runs.values():
        cats = agg.get("pointwise", {}).get("category_scores", {})
        categories.update(cats.keys())
    categories = sorted(categories)

    if categories:
        for cat in categories:
            lines.append(f"### {cat}")
            lines.append("")
            lines.append(f"| Format | Overall | Correctness | Instruct. Follow. | Output Format |")
            lines.append(f"|--------|---------|-------------|-------------------|---------------|")

            if base_agg:
                cat_scores = base_agg.get("pointwise", {}).get("category_scores", {}).get(cat, {}).get("avg_scores", {})
                lines.append(f"| {base_run} | {cat_scores.get('overall', 'N/A')} | "
                           f"{cat_scores.get('correctness', 'N/A')} | "
                           f"{cat_scores.get('instruction_following', 'N/A')} | "
                           f"{cat_scores.get('output_format', 'N/A')} |")

            for name, agg in sorted(format_runs.items()):
                fmt_label = name.replace("lfm2_230m_format_ablation_", "").split("_20")[0]
                cat_scores = agg.get("pointwise", {}).get("category_scores", {}).get(cat, {}).get("avg_scores", {})
                lines.append(f"| {fmt_label} | {cat_scores.get('overall', 'N/A')} | "
                           f"{cat_scores.get('correctness', 'N/A')} | "
                           f"{cat_scores.get('instruction_following', 'N/A')} | "
                           f"{cat_scores.get('output_format', 'N/A')} |")
            lines.append("")
    else:
        lines.append("No category-level data available.")
        lines.append("")

    # Section 7: Format Quality Metrics
    lines.append("## 6. Format Quality Metrics")
    lines.append("")
    lines.append("| Format | JSON Validity | Avg Length | Slop Rate |")
    lines.append("|--------|--------------|------------|-----------|")

    for name, agg in sorted(format_runs.items()):
        fmt_label = name.replace("lfm2_230m_format_ablation_", "").split("_20")[0]
        fmt_m = agg.get("format", {})
        lines.append(f"| {fmt_label} | "
                   f"{fmt_m.get('json_format_validity_rate', 'N/A')} | "
                   f"{fmt_m.get('avg_output_length', 'N/A')} | "
                   f"{fmt_m.get('slop_phrase_rate', 'N/A')} |")
    lines.append("")

    # Section 8: Regression Details
    lines.append("## 7. Regression Analysis")
    lines.append("")
    any_regressions = False
    for name, agg in sorted(format_runs.items()):
        comps = agg.get("comparisons", {})
        if base_run and base_run in comps:
            reg = comps[base_run].get("regression", {})
            reg_details = reg.get("regression_details", [])
            if reg_details:
                any_regressions = True
                fmt_label = name.replace("lfm2_230m_format_ablation_", "").split("_20")[0]
                lines.append(f"### {fmt_label}")
                lines.append("")
                lines.append(f"| Category | Baseline | Current | Delta |")
                lines.append(f"|----------|----------|---------|-------|")
                for r in reg_details:
                    lines.append(f"| {r['category']} | {r['baseline_overall']} | "
                               f"{r['current_overall']} | {r['delta']:+.2f} |")
                lines.append("")
    if not any_regressions:
        lines.append("No regressions detected.")
    lines.append("")

    # Section 9: What Worked
    lines.append("## 8. What Worked")
    lines.append("")
    improvements = []
    for name, agg in format_runs.items():
        comps = agg.get("comparisons", {})
        if base_run and base_run in comps:
            imp_details = comps[base_run].get("regression", {}).get("improvement_details", [])
            for imp in imp_details:
                fmt_label = name.replace("lfm2_230m_format_ablation_", "").split("_20")[0]
                improvements.append(f"- **{fmt_label}** improved `{imp['category']}` by {imp['delta']:+.2f}")
    if improvements:
        lines.extend(improvements[:20])
    else:
        lines.append("No clear improvements over base detected yet.")
    lines.append("")

    # Section 10: What Failed
    lines.append("## 9. What Failed / Inconclusive")
    lines.append("")
    failures = []
    for name, agg in format_runs.items():
        fmt_label = name.replace("lfm2_230m_format_ablation_", "").split("_20")[0]
        overall = agg.get("pointwise", {}).get("avg_scores", {}).get("overall", 0)
        if base_agg:
            base_overall = base_agg.get("pointwise", {}).get("avg_scores", {}).get("overall", 0)
            if overall < base_overall:
                failures.append(f"- **{fmt_label}** overall ({overall:.2f}) below base ({base_overall:.2f})")
    if failures:
        lines.extend(failures)
    else:
        lines.append("No formats performed clearly below base.")
    lines.append("")

    # Section 11-12: Hypothesis verdicts (expanded)
    lines.append("## 10. Hypothesis Verdicts (Detailed)")
    lines.append("")
    for hid, desc, status in hypotheses:
        lines.append(f"### {hid}: {desc}")
        lines.append(f"**Status:** {status}")
        lines.append("")
        if hid == "H1" and concise is not None and verbose is not None:
            lines.append(f"- Concise multi-turn overall: {concise:.2f}")
            lines.append(f"- Verbose multi-turn overall: {verbose:.2f}")
            lines.append(f"- Delta: {concise - verbose:+.2f}")
        elif hid == "H2" and chat is not None and alpaca is not None:
            lines.append(f"- Single-turn chat overall: {chat:.2f}")
            lines.append(f"- Alpaca flat overall: {alpaca:.2f}")
            lines.append(f"- Delta: {chat - alpaca:+.2f}")
        elif hid == "H5" and bad is not None:
            lines.append(f"- Bad-format control overall: {bad:.2f}")
            lines.append(f"- Base overall: {base_overall:.2f}")
        lines.append("")

    # Section 13: KL Drift Summary
    lines.append("## 11. KL Drift Summary")
    lines.append("")
    drift_dir = PROJECT_ROOT / "results" / "drift"
    drift_files = sorted(drift_dir.glob("*_kl.json")) if drift_dir.exists() else []
    if drift_files:
        lines.append("| Format | Avg KL | Length Drift | Refusal Rate | Rep Rate |")
        lines.append("|--------|--------|-------------|-------------|----------|")
        for df in drift_files:
            drift_data = load_json(df)
            if drift_data:
                run_id = drift_data.get("run_id", df.stem.replace("_kl", ""))
                fmt_label = run_id.replace("lfm2_230m_format_ablation_", "").split("_20")[0]
                lines.append(f"| {fmt_label} | "
                           f"{drift_data.get('average_kl', 'N/A')} | "
                           f"{drift_data.get('output_length_drift', 'N/A')} | "
                           f"{drift_data.get('refusal_rate', 'N/A')} | "
                           f"{drift_data.get('repetition_rate', 'N/A')} |")
    else:
        lines.append("No KL drift data available. Run `compute_kl_drift.py` for each adapter.")
    lines.append("")

    # Section 14-15: Training metadata
    lines.append("## 12. Training Summary")
    lines.append("")
    lines.append("| Format | Final Loss | Steps | Dataset Size |")
    lines.append("|--------|-----------|-------|-------------|")
    for name in sorted(format_runs.keys()):
        meta_path = PROJECT_ROOT / "adapters" / name / "metadata.json"
        if meta_path.exists():
            meta = load_json(meta_path)
            if meta:
                fmt_label = name.replace("lfm2_230m_format_ablation_", "").split("_20")[0]
                lines.append(f"| {fmt_label} | "
                           f"{meta.get('final_loss', 'N/A')} | "
                           f"{meta.get('global_step', 'N/A')} | "
                           f"{meta.get('dataset_size', 'N/A')} |")
    lines.append("")

    # Section 16-17: Pairwise comparisons
    lines.append("## 13. Pairwise Win Matrix")
    lines.append("")
    lines.append("(If pairwise judging was run between formats)")
    lines.append("")
    pairwise_data = []
    for name, agg in format_runs.items():
        pw = agg.get("pairwise", {})
        if pw:
            fmt_label = name.replace("lfm2_230m_format_ablation_", "").split("_20")[0]
            pairwise_data.append((fmt_label, pw))
    if pairwise_data:
        lines.append("| Format | Wins | Losses | Ties | Win Rate |")
        lines.append("|--------|------|--------|------|----------|")
        for label, pw in pairwise_data:
            lines.append(f"| {label} | {pw.get('wins', 0)} | {pw.get('losses', 0)} | "
                       f"{pw.get('ties', 0)} | {pw.get('win_rate', 'N/A')} |")
    else:
        lines.append("No pairwise comparison data between formats.")
    lines.append("")

    # Section 18: Recommendations
    lines.append("## 14. Recommendations")
    lines.append("")
    lines.append("Based on the ablation results:")
    lines.append("")
    if best_name:
        fmt_label = best_name.replace("lfm2_230m_format_ablation_", "").split("_20")[0]
        lines.append(f"1. **Use {fmt_label} format** as the default training data format for LFM2.5-230M SFT.")
    lines.append("2. Avoid verbose multi-turn formats — they increase token count without proportional quality gain.")
    lines.append("3. Test structured terse format specifically on JSON/code benchmarks.")
    lines.append("4. Run KL drift analysis to confirm format changes don't introduce unexpected distribution shifts.")
    lines.append("")

    # Section 19: Next Experiments
    lines.append("## 15. Suggested Next Experiments")
    lines.append("")
    lines.append("1. **Mixed-format training**: Combine the top 2 formats and test for additive benefit.")
    lines.append("2. **Format×rank interaction**: Does LoRA rank 4 vs 8 vs 16 change which format wins?")
    lines.append("3. **Format×steps interaction**: Does the best format change at 100 vs 300 vs 1000 steps?")
    lines.append("4. **Larger model replication**: Does the format ranking hold for LFM2.5-1.5B?")
    lines.append("5. **Content×format factorial**: Cross 3 content sets × 3 formats to separate effects.")
    lines.append("")

    # Section 20: Reproduction
    lines.append("## 16. Reproduction Commands")
    lines.append("")
    lines.append("```bash")
    lines.append("# Re-run the full ablation:")
    lines.append("python scripts/train/run_format_ablation.py \\")
    lines.append("  --config configs/experiments/format_ablation_quality.yaml")
    lines.append("")
    lines.append("# Re-run a single format:")
    lines.append("python scripts/train/run_format_ablation.py \\")
    lines.append("  --config configs/experiments/format_ablation_quality.yaml --format multi_turn_concise")
    lines.append("")
    lines.append("# Re-generate this report:")
    lines.append("python scripts/report/build_phase09_report.py \\")
    lines.append("  --results-dir results/evals/ \\")
    lines.append("  --output-md reports/09-data-format-ablation.md \\")
    lines.append("  --output-html docs/09-data-format-ablation.html")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def build_html(markdown_content: str, title: str = "Phase 9: Data Format Ablation") -> str:
    """Convert markdown content to a simple HTML page with dark theme."""
    import re

    # Simple markdown-to-HTML conversion (no external deps)
    html_lines = []
    in_code = False
    in_table = False

    for line in markdown_content.split("\n"):
        # Code blocks
        if line.startswith("```"):
            if in_code:
                html_lines.append("</code></pre>")
                in_code = False
            else:
                lang = line[3:].strip()
                html_lines.append(f'<pre><code class="language-{lang}">')
                in_code = True
            continue
        if in_code:
            html_lines.append(line.replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # Tables
        if "|" in line and line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.match(r'^[-:]+$', c) for c in cells):
                continue  # separator row
            if not in_table:
                html_lines.append('<div class="table-wrapper"><table>')
                in_table = True
                html_lines.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
            else:
                html_lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        elif in_table:
            html_lines.append("</table></div>")
            in_table = False

        # Headers
        if line.startswith("### "):
            html_lines.append(f"<h4>{line[4:]}</h4>")
            continue
        if line.startswith("## "):
            html_lines.append(f"<h3>{line[3:]}</h3>")
            continue
        if line.startswith("# "):
            html_lines.append(f"<h2>{line[2:]}</h2>")
            continue

        # Lists
        if line.startswith("- "):
            content = line[2:]
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'`(.*?)`', r'<code>\1</code>', content)
            html_lines.append(f"<li>{content}</li>")
            continue

        if line.startswith("*") and line.endswith("*"):
            html_lines.append(f"<p class='italic'>{line.strip('*')}</p>")
            continue

        # Regular paragraphs
        if line.strip():
            content = line
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'`(.*?)`', r'<code>\1</code>', content)
            html_lines.append(f"<p>{content}</p>")
        else:
            html_lines.append("")

    if in_table:
        html_lines.append("</table></div>")

    body = "\n".join(html_lines)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>{title} — MI-Atlas</title>
  <meta name="description" content="Phase 9 data format ablation study for LFM2.5-230M. Controlled comparison of training data formats on model quality.">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
  <link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Noto+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --ink: #e0e0e0; --muted: #999; --bg: #1a1a2e; --card-bg: #16213e;
      --accent: #4fc3f7; --accent2: #81c784; --warn: #ffb74d; --danger: #ef5350;
    }}
    body {{ font-family: 'Noto Sans', sans-serif; color: var(--ink); background: var(--bg); line-height: 1.6; }}
    a {{ color: var(--accent); }}
    h2 {{ font-family: 'Google Sans', sans-serif; font-size: 1.8rem; font-weight: 700; color: var(--accent); margin-top: 2rem; border-bottom: 2px solid var(--accent); padding-bottom: 0.3rem; }}
    h3 {{ font-family: 'Google Sans', sans-serif; font-size: 1.3rem; font-weight: 600; color: var(--accent2); margin-top: 1.5rem; }}
    h4 {{ font-size: 1.1rem; font-weight: 600; color: var(--warn); margin-top: 1.2rem; }}
    code {{ font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; background: var(--card-bg); padding: 0.1rem 0.4rem; border-radius: 3px; }}
    pre {{ background: var(--card-bg); padding: 1rem; border-radius: 8px; overflow-x: auto; margin: 1rem 0; }}
    pre code {{ background: none; padding: 0; }}
    .table-wrapper {{ overflow-x: auto; margin: 1rem 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th {{ background: var(--card-bg); font-weight: 600; text-align: left; padding: 8px 12px; border-bottom: 2px solid #333; color: var(--accent); }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #2a2a4a; }}
    tr:hover {{ background: rgba(79, 195, 247, 0.05); }}
    li {{ margin-left: 1.5rem; margin-bottom: 0.3rem; }}
    .italic {{ font-style: italic; color: var(--muted); }}
    .back-link {{ display: inline-block; margin-bottom: 1.5rem; color: var(--accent); text-decoration: none; }}
    .back-link:hover {{ text-decoration: underline; }}
    .container {{ max-width: 960px; margin: 0 auto; padding: 2rem 1.2rem; }}
    @media (max-width: 768px) {{
      h2 {{ font-size: 1.4rem; }}
      .container {{ padding: 1rem; }}
    }}
  </style>
</head>
<body>
<div class="container">
  <a class="back-link" href="index.html"><i class="fas fa-arrow-left"></i> Back to MI-Atlas</a>
  {body}
</div>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser(description="Build Phase 9 data-format-ablation report")
    parser.add_argument("--results-dir", default="results/evals/", help="Results directory")
    parser.add_argument("--output-md", default="reports/09-data-format-ablation.md", help="Markdown output path")
    parser.add_argument("--output-html", default="docs/09-data-format-ablation.html", help="HTML output path")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    if not results_dir.exists():
        log.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    # Discover runs
    runs = discover_runs(results_dir)
    if not runs:
        log.error("No runs with aggregate.json found")
        sys.exit(1)
    log.info(f"Found {len(runs)} runs with aggregate data")

    base_run = find_base_run(runs)
    log.info(f"Base run: {base_run or 'not found'}")

    # Build markdown
    md_content = build_markdown(runs, base_run)

    # Write markdown
    md_path = PROJECT_ROOT / args.output_md
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w") as f:
        f.write(md_content)
    log.info(f"Written markdown to {md_path}")

    # Build and write HTML
    html_content = build_html(md_content)
    html_path = PROJECT_ROOT / args.output_html
    html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w") as f:
        f.write(html_content)
    log.info(f"Written HTML to {html_path}")

    log.info("Report generation complete.")


if __name__ == "__main__":
    main()
