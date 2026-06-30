# Artifact Index

## Adapters

| Name | Path | Training data | Main skill | Eval score | Notes |
|------|------|---------------|------------|------------|-------|
| lora_json_r1 | experiments/adapters/lora_json_r1/ | JSON schema x300 | JSON format | loss=1.085 | Rank 1, all-linear |
| lora_json_r2 | experiments/adapters/lora_json_r2/ | JSON schema x300 | JSON format | loss=0.829 | Rank 2, all-linear |
| lora_json_r4 | experiments/adapters/lora_json_r4/ | JSON schema x300 | JSON format | loss=0.629 | Rank 4, all-linear |
| lora_json_r8 | experiments/adapters/lora_json_r8/ | JSON schema x300 | JSON format | loss=0.500 | Rank 8, all-linear. Primary adapter for comparison experiments. |
| lora_json_r16 | experiments/adapters/lora_json_r16/ | JSON schema x300 | JSON format | loss=0.409 | Rank 16, all-linear |
| lora_copying_r8 | experiments/adapters/lora_copying_r8/ | Copying x300 | Pattern repetition | loss=0.178, acc=89.5% | Rank 8, all-linear |
| lora_delimiter_tracking_r8 | experiments/adapters/lora_delimiter_tracking_r8/ | Delimiter x300 | Bracket completion | loss=0.164, acc=90.9% | Rank 8, all-linear. Fully absorbs skill (0 ablation sensitivity). |
| lora_factual_recall_r8 | experiments/adapters/lora_factual_recall_r8/ | Factual x300 | Capital cities etc | loss=0.552, acc=80.0% | Rank 8, all-linear |
| lora_code_semantics_r8 | experiments/adapters/lora_code_semantics_r8/ | Code x300 | Python output prediction | loss=0.050, acc=100% | Rank 8, all-linear |
| lora_json_schema_r8 | experiments/adapters/lora_json_schema_r8/ | JSON schema x300 | JSON format | loss=0.062, acc=96.2% | Rank 8, all-linear. Used in dataset shard ablation. |

## Checkpoints

| Name | Path | Training regime | Steps | Main result | Notes |
|------|------|-----------------|-------|-------------|-------|
| json_timeline_step10 | experiments/checkpoints/json_timeline_step10/ | LoRA r=8 JSON | 10 | Core circuit already established | Loss 0.587 |
| json_timeline_step25 | experiments/checkpoints/json_timeline_step25/ | LoRA r=8 JSON | 25 | L2/L7/L9 stable | Loss 0.154 |
| json_timeline_step50 | experiments/checkpoints/json_timeline_step50/ | LoRA r=8 JSON | 50 | Diminishing returns begin | Loss 0.104 |
| json_timeline_step75 | experiments/checkpoints/json_timeline_step75/ | LoRA r=8 JSON | 75 | Secondary layers still shifting | Loss 0.086 |
| json_timeline_step100 | experiments/checkpoints/json_timeline_step100/ | LoRA r=8 JSON | 100 | Final. L15/L6 drift max +2.85/+2.73 | Loss 0.062 |

## Plots

| Plot | Path | Experiment | Meaning |
|------|------|------------|---------|
| baseline_task_scores.png | experiments/plots/ | exp_000004 | Per-family baseline logprobs |
| layer_ablation_heatmap_zero.png | experiments/plots/ | exp_000005 | KL by layer x family |
| head_ablation_heatmap.png | experiments/plots/ | exp_000006 | Head ablation effects |
| mlp_ablation_heatmap.png | experiments/plots/ | exp_000007 | MLP ablation effects |
| lora_comparison_base.png | experiments/plots/ | exp_000009 | Base model ablation map |
| lora_comparison_lora.png | experiments/plots/ | exp_000009 | LoRA model ablation map |
| lora_comparison_diff.png | experiments/plots/ | exp_000009 | Delta map (LoRA - base) |
| activation_patching_heatmap_v1.png | experiments/plots/ | exp_000012 | Patching recovery heatmap |
| patching_kl_heatmap_v1.png | experiments/plots/ | exp_000013 | KL-based patching heatmap |

## Results

| File | Path | Experiment | Content |
|------|------|------------|---------|
| tokenizer_diagnostics.json | experiments/results/ | exp_000002 | Token-level analysis |
| baseline_eval.json | experiments/results/ | exp_000004 | Per-task baseline scores |
| layer_ablation_zero.json | experiments/results/ | exp_000005 | 24-layer x 12-family KL |
| head_ablation.json | experiments/results/ | exp_000006 | Head-level ablation |
| mlp_ablation.json | experiments/results/ | exp_000007 | MLP-level ablation |
| steering_sweep.json | experiments/results/ | exp_000008 | Steering strengths -8 to +8 |
| lora_ablation_comparison.json | experiments/results/ | exp_000009 | Base vs LoRA ablation maps |
| lora_json_comparison.json | experiments/results/ | exp_000009 | JSON-specific comparison |
| lora_rank_sweep.json | experiments/results/ | exp_000010 | Rank 1/2/4/8/16 results |
| lora_module_sweep.json | experiments/results/ | exp_000011 | q/v/o/mlp/attn/all results |
| activation_patching.json | experiments/results/ | exp_000012 | First patching attempt |
| activation_patching_v1.json | experiments/results/ | exp_000012 | Aligned pairs patching |
| patching_kl_v1.json | experiments/results/ | exp_000013 | KL-based patching |
| dataset_shard_ablation.json | experiments/results/ | exp_000014 | 5-family adapter comparison |
| adapter_archaeology.json | experiments/results/ | exp_000015 | Norm/rank analysis of 10 adapters |
| adapter_stacking.json | experiments/results/ | exp_000016 | 5-pair weighted merge interference |
| checkpoint_timeline.json | experiments/results/ | exp_000017 | 5-checkpoint ablation maps |
| position_specific_ablation.json | experiments/results/ | exp_000018 | Per-token position effects |

## Reports

| Report | Path | Status |
|--------|------|--------|
| Current Findings | reports/current_findings.md | Updated (18 experiments) |
| Open Hypotheses | reports/open_hypotheses.md | Updated (6 hypotheses) |
| Decision Log | reports/decision_log.md | Updated (7 decisions) |
| Negative Results | reports/negative_results.md | Updated (6 entries) |
| Component Atlas | reports/component_atlas.md | Complete (11 entries) |
| Component Atlas JSONL | reports/component_atlas.jsonl | Complete (11 entries) |
| Final Report | reports/final_report.md | Not started |
| Blog Post Outline | reports/blog_post_outline.md | Stub |
| Paper Outline | reports/paper_outline.md | Stub |
| Limitations | reports/limitations.md | Stub |

## GitHub Pages

| Artifact | Path | Status |
|----------|------|--------|
| Published site index | docs/index.html | Updated with Phase 3 link and Phase 1-3 summary |
| One-page MI-Atlas share page | docs/mi-atlas.html | Shareable public summary of the atlas, audited claims, caveats, negative results, and practical rules. No RAG benchmark content included |
| Shared HYFL-inspired stylesheet | docs/assets/hyfl-mi.css | New shared responsive design layer for all published pages |
| Phase 2 page | docs/05-phase2-repeatability.html | Updated to link Phase 3 and note the L26 -> L14 revision |
| Phase 3 page | docs/06-phase3-gap-closure.html | Restyled to match the site and list remaining work |
| Phase 7 LFM2 Atlas | docs/07-lfm2-230m-atlas.html | Complete atlas of LFM2.5-230M |
| Phase 8 SFT Sweep | docs/08-lfm2-230m-sft-sweep.html | 39 SFT experiments, best recipe identified |
| Phase 9 Format Ablation | docs/09-data-format-ablation.html | **CAVEAT: contains mock-judge data — see NR014** |
| Phase 9R Report | reports/09-data-format-ablation.md | Honest report with evidence tiers, training loss only |
| Bilawal.net mirror source | ../pretty-blog-python/pages/mi-atlas.html | Standalone bilawal.net copy with Swetrix analytics and absolute links back to the research repo/pages |
| Pre-sync safety branch | backup/pre-sync-20260630-010555 | Local branch preserving the pre-sync MI repo HEAD before merging origin/main |
| Pre-sync safety stash | stash@{0}: pre-sync-safety-20260630-010555 | Local stash preserving the dirty working tree before the sync/deploy repair |

## Phase 9 Scripts (2026-06-29)

| Script | Path | Purpose | Status |
|--------|------|---------|--------|
| judge_outputs.py | scripts/eval/judge_outputs.py | Judge eval outputs (pointwise/pairwise) | Rewritten in Phase 9R — explicit --mock, judge_source metadata |
| aggregate_eval_results.py | scripts/eval/aggregate_eval_results.py | Aggregate scores + programmatic metrics | Enhanced in Phase 9R — 8 programmatic scorers |
| generate_blind_review.py | scripts/eval/generate_blind_review.py | Stratified blind review (60+ examples, 9 categories) | New in Phase 9R |
| run_phase9r_eval.py | scripts/eval/run_phase9r_eval.py | One-command pipeline for aero eval | New in Phase 9R |
| run_eval_harness.py | scripts/eval/run_eval_harness.py | Generate model responses on eval set | Unchanged |
| export_manual_review.py | scripts/eval/export_manual_review.py | Export examples for human review | Unchanged |
| run_format_ablation.py | scripts/train/run_format_ablation.py | Orchestrate format ablation training | Unchanged |
| render_dataset_formats.py | scripts/data/render_dataset_formats.py | Render 6 format variants from canonical | Unchanged |
