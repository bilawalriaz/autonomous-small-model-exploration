#!/usr/bin/env python3
"""Compare outputs of base GGUF and SFT fine-tuned GGUF.

Example:
    python scripts/eval/compare_gguf_results.py \
        --base-outputs results/evals/lfm25_12b_base_gguf/outputs.jsonl \
        --sft-outputs results/evals/lfm25_12b_sft_gguf/outputs.jsonl \
        --eval-set data/eval/small_model_eval_v1.jsonl \
        --output results/evals/gguf_comparison_report.md
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

SLOP_PHRASES = [
    "as an ai", "i apologize", "i'm sorry, but", "as a language model",
    "i hope this helps", "please note that", "it's important to note"
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-outputs", required=True)
    parser.add_argument("--sft-outputs", required=True)
    parser.add_argument("--eval-set", default="data/eval/small_model_eval_v1.jsonl")
    parser.add_argument("--output", default="results/evals/gguf_comparison_report.md")
    return parser.parse_args()

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def extract_json_object(text: str) -> bool:
    """Check if output contains a valid JSON block."""
    text = text.strip()
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        pass

    # Try parsing text inside first { and last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            json.loads(text[start:end+1])
            return True
        except (json.JSONDecodeError, ValueError):
            pass

    # Try array
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            json.loads(text[start:end+1])
            return True
        except (json.JSONDecodeError, ValueError):
            pass

    return False

def check_factual_match(output_text: str, eval_entry: dict) -> bool:
    """Check if response contains expected answers."""
    # Simple check for fact presence
    expected = eval_entry.get("expected_behavior", "")
    if not expected:
        return True
    
    # Lowercase clean check
    clean_out = output_text.lower()
    
    # If there are hard constraints
    constraints = eval_entry.get("hard_constraints", [])
    for c in constraints:
        if c.lower() in clean_out:
            return True
            
    # Simple substring check
    if expected.lower() in clean_out:
        return True
        
    return False

def check_constraint_following(output_text: str, eval_entry: dict) -> bool:
    """Evaluate instruction following constraints."""
    constraints = eval_entry.get("hard_constraints", [])
    if not constraints:
        return True
        
    clean_out = output_text.lower()
    
    # Check word limit constraints
    for c in constraints:
        if "exactly" in c and "word" in c:
            # Extract number
            match = re.search(r'\d+', c)
            if match:
                expected_len = int(match.group())
                words = [w for w in output_text.split() if w.strip()]
                # Allow a tiny tolerance or check exact
                if "exactly one word" in c:
                    return len(words) == 1
                return len(words) == expected_len
                
        if "numbers only" in c:
            # check if there's no letters
            if any(char.isalpha() for char in output_text):
                return False
                
        if "no explanation" in c:
            # Check length or slop
            if len(output_text.split()) > 20:
                return False
                
    return True

def count_slop_phrases(text: str) -> int:
    clean = text.lower()
    return sum(1 for phrase in SLOP_PHRASES if phrase in clean)

def main() -> int:
    args = parse_args()

    base_records = {r["eval_id"]: r for r in load_jsonl(Path(args.base_outputs))}
    sft_records = {r["eval_id"]: r for r in load_jsonl(Path(args.sft_outputs))}
    eval_set = load_jsonl(Path(args.eval_set))

    print(f"Loaded {len(base_records)} base results, {len(sft_records)} SFT results")

    category_stats = defaultdict(lambda: {
        "count": 0,
        "base_len": 0, "sft_len": 0,
        "base_json_ok": 0, "sft_json_ok": 0,
        "base_fact_ok": 0, "sft_fact_ok": 0,
        "base_const_ok": 0, "sft_const_ok": 0,
        "base_slop_count": 0, "sft_slop_count": 0
    })

    for entry in eval_set:
        eid = entry.get("id") or entry.get("eval_id")
        category = entry.get("category", "unknown")
        
        if eid not in base_records or eid not in sft_records:
            continue
            
        base_out = base_records[eid]["generated_response"]
        sft_out = sft_records[eid]["generated_response"]
        
        stats = category_stats[category]
        stats["count"] += 1
        
        # Word counts
        stats["base_len"] += len(base_out.split())
        stats["sft_len"] += len(sft_out.split())
        
        # JSON formatting check
        if category in ["json_structured", "gamefaq_extraction"]:
            if extract_json_object(base_out):
                stats["base_json_ok"] += 1
            if extract_json_object(sft_out):
                stats["sft_json_ok"] += 1
                
        # Factual qa check
        if category == "factual_qa":
            if check_factual_match(base_out, entry):
                stats["base_fact_ok"] += 1
            if check_factual_match(sft_out, entry):
                stats["sft_fact_ok"] += 1
                
        # Constraints check
        if category in ["instruction_following", "concision_antislip"]:
            if check_constraint_following(base_out, entry):
                stats["base_const_ok"] += 1
            if check_constraint_following(sft_out, entry):
                stats["sft_const_ok"] += 1
                
        # Slop checks
        stats["base_slop_count"] += count_slop_phrases(base_out)
        stats["sft_slop_count"] += count_slop_phrases(sft_out)

    # Build report
    report_lines = [
        "# GGUF Model Comparison Report: Base vs SFT",
        "",
        "This report compares the output characteristics of the base `LiquidAI/LFM2.5-1.2B-Instruct` model and the fine-tuned SFT version (`lfm25_12b_instruct_sft_q8_strict`) using quantized `Q4_K_M` GGUF files.",
        "",
        "## Summary Metrics by Category",
        "",
        "| Category | Count | Metric Type | Base GGUF | SFT GGUF | SFT Delta |",
        "| :--- | :---: | :--- | :---: | :---: | :---: |"
    ]

    total_count = 0
    total_base_len = 0
    total_sft_len = 0
    total_base_slop = 0
    total_sft_slop = 0

    for cat, stats in sorted(category_stats.items()):
        cnt = stats["count"]
        if cnt == 0:
            continue
            
        total_count += cnt
        total_base_len += stats["base_len"]
        total_sft_len += stats["sft_len"]
        total_base_slop += stats["base_slop_count"]
        total_sft_slop += stats["sft_slop_count"]
        
        avg_base_len = stats["base_len"] / cnt
        avg_sft_len = stats["sft_len"] / cnt
        report_lines.append(f"| {cat} | {cnt} | Avg Length (words) | {avg_base_len:.1f} | {avg_sft_len:.1f} | {avg_sft_len - avg_base_len:+.1f} |")
        
        if cat in ["json_structured", "gamefaq_extraction"]:
            base_json_rate = (stats["base_json_ok"] / cnt) * 100
            sft_json_rate = (stats["sft_json_ok"] / cnt) * 100
            report_lines.append(f"| | | JSON Validity Rate | {base_json_rate:.1f}% | {sft_json_rate:.1f}% | {sft_json_rate - base_json_rate:+.1f}% |")
            
        if cat == "factual_qa":
            base_fact_rate = (stats["base_fact_ok"] / cnt) * 100
            sft_fact_rate = (stats["sft_fact_ok"] / cnt) * 100
            report_lines.append(f"| | | Factual Accuracy Rate | {base_fact_rate:.1f}% | {sft_fact_rate:.1f}% | {sft_fact_rate - base_fact_rate:+.1f}% |")
            
        if cat in ["instruction_following", "concision_antislip"]:
            base_const_rate = (stats["base_const_ok"] / cnt) * 100
            sft_const_rate = (stats["sft_const_ok"] / cnt) * 100
            report_lines.append(f"| | | Constraint Adherence | {base_const_rate:.1f}% | {sft_const_rate:.1f}% | {sft_const_rate - base_const_rate:+.1f}% |")
            
        if stats["base_slop_count"] > 0 or stats["sft_slop_count"] > 0:
            base_slop_avg = stats["base_slop_count"] / cnt
            sft_slop_avg = stats["sft_slop_count"] / cnt
            report_lines.append(f"| | | Slop Rate (phrases/resp) | {base_slop_avg:.2f} | {sft_slop_avg:.2f} | {sft_slop_avg - base_slop_avg:+.2f} |")

    report_lines.extend([
        "",
        "## Overall Comparison Summary",
        "",
        f"- **Total Prompts Evaluated**: {total_count}",
        f"- **Average Base Length**: {total_base_len / total_count:.1f} words",
        f"- **Average SFT Length**: {total_sft_len / total_count:.1f} words",
        f"- **Total Base Assistant Slop Phrases**: {total_base_slop}",
        f"- **Total SFT Assistant Slop Phrases**: {total_sft_slop}",
        "",
        "## Qualitative Differences & Examples",
        ""
    ])

    # Show a few examples of where they differ
    examples_shown = 0
    for entry in eval_set:
        eid = entry.get("id") or entry.get("eval_id")
        category = entry.get("category", "unknown")
        
        if eid not in base_records or eid not in sft_records:
            continue
            
        base_out = base_records[eid]["generated_response"]
        sft_out = sft_records[eid]["generated_response"]
        
        # Look for differences in JSON validity, length, or fact accuracy
        is_json_diff = extract_json_object(base_out) != extract_json_object(sft_out)
        is_length_diff = abs(len(base_out.split()) - len(sft_out.split())) > 30
        is_qa_diff = category == "factual_qa" and (check_factual_match(base_out, entry) != check_factual_match(sft_out, entry))
        
        if (is_json_diff or is_qa_diff or (is_length_diff and examples_shown < 2)) and examples_shown < 5:
            examples_shown += 1
            report_lines.extend([
                f"### Example {examples_shown}: {category} ({eid})",
                "",
                f"**Prompt**:",
                f"> {entry.get('prompt')}",
                "",
                f"**Base GGUF Output**:",
                "```",
                base_out,
                "```",
                "",
                f"**SFT GGUF Output**:",
                "```",
                sft_out,
                "```",
                "",
                "---",
                ""
            ])

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved comparison report to {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
