"""
Build Component Atlas from all experiment results.
Formalizes findings into the atlas schema (component_atlas.jsonl + component_atlas.md).
"""
import json, os
from pathlib import Path
from datetime import datetime, timezone

REPO = Path(__file__).parent.parent
RESULTS_DIR = REPO / "experiments" / "results"
REPORTS_DIR = REPO / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_result(name):
    path = RESULTS_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    # Load all results
    layer_ablation = load_result("layer_ablation_zero")
    head_ablation = load_result("head_ablation")
    mlp_ablation = load_result("mlp_ablation")
    steering = load_result("steering_sweep")
    lora_comparison = load_result("lora_ablation_comparison")
    rank_sweep = load_result("lora_rank_sweep")
    module_sweep = load_result("lora_module_sweep")
    shard_ablation = load_result("dataset_shard_ablation")
    archaeology = load_result("adapter_archaeology")
    stacking = load_result("adapter_stacking")
    timeline = load_result("checkpoint_timeline")
    position = load_result("position_specific_ablation")
    baseline = load_result("baseline_eval")

    entries = []
    md_lines = ["# Component Atlas\n", f"Generated: {datetime.now(timezone.utc).isoformat()}\n"]

    # ── L2 (Layer 2 residual stream) ──────────────────────────────────────────
    entries.append({
        "component_id": "layer_02_residual",
        "component_type": "residual_stream",
        "layer": 2,
        "head": None,
        "claimed_behaviour": "Universal processing hub — largest ablation effect across all 12 task families",
        "task_families": ["copying", "delimiter_tracking", "factual_recall", "code_semantics", "json_schema",
                          "arithmetic", "dead_code", "variable_renaming", "uncertainty_signalling",
                          "refusal_compliance", "verbosity_control", "code_syntax"],
        "positive_effects": [
            {"experiment_id": "exp_000005", "metric": "ablation_kl", "effect_size": 11.54,
             "detail": "L2 ablation causes KL 0.5-11.5 across all families; factual_recall most affected"},
            {"experiment_id": "exp_000007", "metric": "mlp_ablation_kl", "effect_size": 10.5,
             "detail": "L2 MLP ablation: major contributor to L2 effect"},
            {"experiment_id": "exp_000008", "metric": "steering_boost", "effect_size": 3.3,
             "detail": "Factual direction at L2: 'rome' logprob 0.064->0.213 (3.3x) at s=+4.0"},
        ],
        "negative_effects": [
            {"experiment_id": "exp_000018", "metric": "position_effect", "effect_size": 0,
             "detail": "L2 is NOT uniformly positional — effects concentrated at first+last tokens (mean 3.34/5.03), operators near-zero"}
        ],
        "training_delta_evidence": [
            {"experiment_id": "exp_000014", "variant": "json_schema_lora",
             "summary": "L2 importance for json_schema unchanged (21.15->23.38, +2.2). But L2 decreased for copying (-0.75) after JSON LoRA."},
            {"experiment_id": "exp_000017", "variant": "checkpoint_timeline",
             "summary": "L2 for json_schema stable from step 10 (23.37) through step 100 (23.38). Core circuit established early."},
        ],
        "controls": [
            {"experiment_id": "exp_000018", "result": "Operator tokens have near-zero effect at L2 (mean -0.09)"},
            {"experiment_id": "exp_000014", "result": "L2 effect persists across all 5 family-specific adapters"},
        ],
        "steering": {"tested": True, "summary": "Factual direction at L2 boosts target 3.3x. Negative steering suppresses. Oversteering at s>=+2 causes degeneration (Chinese chars)."},
        "adapter_evidence": {"tested": True, "summary": "All 5 family adapters affect L2 importance. delimiter adapter eliminates L9/L22/L4 dependence, making L2 more dominant."},
        "confidence": "high",
        "limitations": "Tested on short synthetic prompts only. Ablation is zero-ablation (not mean/resample). Steering tested on factual recall only.",
        "repro_command": "python scripts/run_layer_ablation.py && python scripts/run_steering_sweep.py && python scripts/run_position_ablation.py",
    })

    # ── L0 MLP ────────────────────────────────────────────────────────────────
    entries.append({
        "component_id": "layer_00_mlp",
        "component_type": "mlp",
        "layer": 0,
        "head": None,
        "claimed_behaviour": "Second-strongest ablation target across all families; absorbs JSON skill after LoRA training",
        "task_families": ["copying", "delimiter_tracking", "factual_recall", "code_semantics", "json_schema"],
        "positive_effects": [
            {"experiment_id": "exp_000007", "metric": "mlp_ablation_kl", "effect_size": 10.8,
             "detail": "L0 MLP second-strongest across all families"},
            {"experiment_id": "exp_000009", "metric": "lora_delta", "effect_size": 2.99,
             "detail": "L0 MLP importance for JSON: 10.85->13.84 (+2.99) after LoRA training"},
        ],
        "training_delta_evidence": [
            {"experiment_id": "exp_000009", "variant": "json_lora_r8",
             "summary": "L0 MLP absorbs JSON skill: +2.99 KL delta"},
            {"experiment_id": "exp_000010", "variant": "rank_sweep",
             "summary": "L0 MLP peaks at r=4 (15.77), declines at higher rank"},
            {"experiment_id": "exp_000011", "variant": "module_sweep",
             "summary": "o_proj-only LoRA achieves +3.64 L0 effect with only 344K params"},
        ],
        "confidence": "medium",
        "limitations": "JSON-specific training only. Other skill families concentrate elsewhere (dataset_shard_ablation).",
        "repro_command": "python scripts/compare_lora_ablation.py && python scripts/run_lora_rank_sweep.py",
    })

    # ── L22 (unembedding pathway) ─────────────────────────────────────────────
    entries.append({
        "component_id": "layer_22_residual",
        "component_type": "residual_stream",
        "layer": 22,
        "head": None,
        "claimed_behaviour": "Final token prediction layer — almost exclusively affects last-position tokens",
        "task_families": ["copying", "delimiter_tracking", "factual_recall", "code_semantics", "json_schema"],
        "positive_effects": [
            {"experiment_id": "exp_000018", "metric": "position_effect", "effect_size": 14.55,
             "detail": "Mean last-position effect 14.55 nats. All other positions ~0."},
            {"experiment_id": "exp_000005", "metric": "ablation_kl", "effect_size": 9.25,
             "detail": "L22 second-strongest layer for delimiter_tracking in base model"},
        ],
        "confidence": "medium",
        "limitations": "Position-specific ablation only tested on base model, not on adapted models.",
        "repro_command": "python scripts/run_position_ablation.py",
    })

    # ── L9 (instruction-sensitive) ────────────────────────────────────────────
    entries.append({
        "component_id": "layer_09_residual",
        "component_type": "residual_stream",
        "layer": 9,
        "head": None,
        "claimed_behaviour": "Instruction-sensitive layer — highest first-position effect among mid-layers, strong delimiter tracking",
        "task_families": ["delimiter_tracking", "json_schema"],
        "positive_effects": [
            {"experiment_id": "exp_000018", "metric": "position_effect", "effect_size": 5.66,
             "detail": "First-position mean 5.66, last-position mean 9.20"},
            {"experiment_id": "exp_000005", "metric": "ablation_kl", "effect_size": 10.0,
             "detail": "L9 strongest layer for delimiter_tracking in base model"},
        ],
        "confidence": "medium",
        "limitations": "Position analysis on short prompts only.",
        "repro_command": "python scripts/run_position_ablation.py",
    })

    # ── L12 H8 (strongest attention head) ─────────────────────────────────────
    entries.append({
        "component_id": "layer_12_head_08",
        "component_type": "attention_head",
        "layer": 12,
        "head": 8,
        "claimed_behaviour": "Strongest individual attention head across multiple families",
        "task_families": ["copying", "delimiter_tracking", "factual_recall", "code_semantics", "json_schema"],
        "positive_effects": [
            {"experiment_id": "exp_000006", "metric": "head_ablation_kl", "effect_size": 2.5,
             "detail": "L12 H8 strongest head across 5 families"},
        ],
        "confidence": "medium",
        "limitations": "Head ablation only. No patching or steering on individual heads.",
        "repro_command": "python scripts/run_head_ablation.py",
    })

    # ── L1 (skill injection point) ────────────────────────────────────────────
    entries.append({
        "component_id": "layer_01_residual",
        "component_type": "residual_stream",
        "layer": 1,
        "head": None,
        "claimed_behaviour": "Appears as universal skill injection point — positive delta across 3+ family adapters",
        "task_families": ["factual_recall", "code_semantics", "json_schema"],
        "positive_effects": [
            {"experiment_id": "exp_000014", "metric": "adapter_delta", "effect_size": 6.51,
             "detail": "L1 positive delta for factual_recall adapter (+6.51 on factual tasks)"},
            {"experiment_id": "exp_000014", "metric": "adapter_delta", "effect_size": 4.70,
             "detail": "L1 positive delta for code_semantics adapter (+4.70 on json tasks)"},
            {"experiment_id": "exp_000014", "metric": "adapter_delta", "effect_size": 3.00,
             "detail": "L1 positive delta for json_schema adapter (+3.00 on delimiter tasks)"},
        ],
        "confidence": "medium",
        "limitations": "Correlational — L1 delta appears but causal mechanism unknown. Could be adapter weight injection, not functional routing.",
        "repro_command": "python scripts/run_dataset_shard_ablation.py",
    })

    # ── LoRA concentration: skill-specific layer patterns ─────────────────────
    entries.append({
        "component_id": "skill_concentration_pattern",
        "component_type": "emergent_pattern",
        "layer": None,
        "head": None,
        "claimed_behaviour": "Each skill concentrates in different layers after LoRA training — no universal pattern",
        "task_families": ["copying", "delimiter_tracking", "factual_recall", "code_semantics", "json_schema"],
        "positive_effects": [
            {"experiment_id": "exp_000014", "metric": "delta_top3", "effect_size": 0,
             "detail": "factual_recall: L3/16/19. code: L1/10/21. json: L6/12/13. delimiter: fully absorbed. copying: dispersed."},
        ],
        "negative_effects": [
            {"experiment_id": "exp_000014", "metric": "h002_test", "effect_size": 0,
             "detail": "H002 (universal L0-L2 concentration) REJECTED. Each skill has unique concentration pattern."},
        ],
        "confidence": "medium",
        "limitations": "Only 5 families tested. Only r=8 adapters. Only 100 training steps.",
        "repro_command": "python scripts/run_dataset_shard_ablation.py",
    })

    # ── Adapter norm/effect spatial separation ────────────────────────────────
    entries.append({
        "component_id": "norm_effect_separation",
        "component_type": "emergent_pattern",
        "layer": None,
        "head": None,
        "claimed_behaviour": "Adapter weight norms peak at L20-L23 but ablation effects peak at L0-L2",
        "task_families": [],
        "positive_effects": [
            {"experiment_id": "exp_000015", "metric": "norm_vs_effect", "effect_size": 0,
             "detail": "Rank sweep: norms peak L22/L23. Ablation effects peak L0/L2. Spatial separation."},
        ],
        "confidence": "medium",
        "limitations": "Norm analysis doesn't prove causal direction. Could be that late layers carry more parameters but early layers do the processing.",
        "repro_command": "python scripts/run_adapter_archaeology.py",
    })

    # ── Adapter stacking: factual+json compatible ─────────────────────────────
    entries.append({
        "component_id": "adapter_stack_factual_json",
        "component_type": "adapter_interaction",
        "layer": None,
        "head": None,
        "claimed_behaviour": "factual_recall + json_schema adapters stack cleanly with positive synergy",
        "task_families": ["factual_recall", "json_schema"],
        "positive_effects": [
            {"experiment_id": "exp_000016", "metric": "stack_interference", "effect_size": 2.35,
             "detail": "factual+json: +2.35 synergy on factual, +1.17 on json, <0.3 interference elsewhere"},
        ],
        "confidence": "medium",
        "limitations": "Only weighted merge (0.5/0.5) tested. No sweep of merge weights.",
        "repro_command": "python scripts/run_adapter_stacking.py",
    })

    # ── Delimiter adapter is destructive when stacked ─────────────────────────
    entries.append({
        "component_id": "adapter_stack_delimiter_destructive",
        "component_type": "adapter_interaction",
        "layer": None,
        "head": None,
        "claimed_behaviour": "delimiter_tracking adapter is destructive when combined with other adapters",
        "task_families": ["delimiter_tracking"],
        "positive_effects": [],
        "negative_effects": [
            {"experiment_id": "exp_000016", "metric": "stack_interference", "effect_size": -15.51,
             "detail": "delimiter+code: -15.51 nats on delimiter task. Consistent across all pairs."},
        ],
        "confidence": "medium",
        "limitations": "May be an artifact of the delimiter task evaluation (target token is single char, model generates multi-char completions).",
        "repro_command": "python scripts/run_adapter_stacking.py",
    })

    # ── Checkpoint timeline: early circuit establishment ───────────────────────
    entries.append({
        "component_id": "early_circuit_establishment",
        "component_type": "training_phenomenon",
        "layer": None,
        "head": None,
        "claimed_behaviour": "Core component structure for JSON schema locks in by step 10 (first 10% of training)",
        "task_families": ["json_schema"],
        "positive_effects": [
            {"experiment_id": "exp_000017", "metric": "timeline_stability", "effect_size": 0,
             "detail": "L2/L7/L9 for json_schema: step10=23.37/21.62/19.00, step100=23.38/21.75/18.88. <1% drift after step 10."},
        ],
        "confidence": "medium",
        "limitations": "Only JSON schema family tested. Only 100 training steps. Different families might establish later.",
        "repro_command": "python scripts/run_checkpoint_timeline.py",
    })

    # Save JSONL
    atlas_path = REPORTS_DIR / "component_atlas.jsonl"
    with open(atlas_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Build markdown
    md_lines.append(f"Total entries: {len(entries)}\n")
    md_lines.append(f"Confidence distribution:\n")
    conf_counts = {}
    for e in entries:
        c = e.get("confidence", "unknown")
        conf_counts[c] = conf_counts.get(c, 0) + 1
    for c, n in sorted(conf_counts.items()):
        md_lines.append(f"- {c}: {n}\n")

    md_lines.append("\n---\n\n")
    for entry in entries:
        md_lines.append(f"## {entry['component_id']}\n\n")
        md_lines.append(f"**Type:** {entry['component_type']}  \n")
        if entry.get("layer") is not None:
            md_lines.append(f"**Layer:** {entry['layer']}  \n")
        if entry.get("head") is not None:
            md_lines.append(f"**Head:** {entry['head']}  \n")
        md_lines.append(f"**Confidence:** {entry.get('confidence', 'unknown')}  \n")
        md_lines.append(f"\n**Claim:** {entry['claimed_behaviour']}\n\n")

        if entry.get("task_families"):
            md_lines.append(f"**Task families:** {', '.join(entry['task_families'])}\n\n")

        if entry.get("positive_effects"):
            md_lines.append("**Positive effects:**\n")
            for eff in entry["positive_effects"]:
                md_lines.append(f"- [{eff['experiment_id']}] {eff.get('detail', eff.get('summary', ''))}\n")
            md_lines.append("\n")

        if entry.get("negative_effects"):
            md_lines.append("**Negative effects:**\n")
            for eff in entry["negative_effects"]:
                md_lines.append(f"- [{eff['experiment_id']}] {eff.get('detail', eff.get('summary', ''))}\n")
            md_lines.append("\n")

        if entry.get("steering"):
            md_lines.append(f"**Steering:** {entry['steering']['summary']}\n\n")

        if entry.get("adapter_evidence"):
            md_lines.append(f"**Adapter evidence:** {entry['adapter_evidence']['summary']}\n\n")

        if entry.get("limitations"):
            md_lines.append(f"**Limitations:** {entry['limitations']}\n\n")

        if entry.get("repro_command"):
            md_lines.append(f"**Repro:** `{entry['repro_command']}`\n\n")

        md_lines.append("---\n\n")

    with open(REPORTS_DIR / "component_atlas.md", "w") as f:
        f.writelines(md_lines)

    print(f"Atlas: {len(entries)} entries written to {atlas_path}")
    print(f"Markdown: {REPORTS_DIR / 'component_atlas.md'}")


if __name__ == "__main__":
    main()
