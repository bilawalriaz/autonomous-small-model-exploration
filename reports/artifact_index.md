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
|---|---|---|---|
| M02 GGUF smoke | results/evals/M02_base_smoke/, results/evals/M02_merged_smoke/ | M02 | Raw deterministic boundary-validation outputs and template provenance |
| M03 paired metrics | results/evals/M03_corrected_q4_paired_metrics.json | M03 | Frozen held-out programmatic paired metrics and bootstrap CIs |

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
|---|---|---|
| MiniCPM single-shot master plan | reports/minicpm5_single_shot_master_plan.md | Active roadmap; Stage 0 next |
| MiniCPM control center | projects/minicpm5/ | Navigation, current status, and overnight handoff prompt |

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
| One-page MI-Atlas share page | docs/mi-atlas.html | Shareable public summary of the atlas, audited claims, LFM2 SFT sweep, caveats, negative results, and practical rules. Every audited claim links to full details. No RAG benchmark content included |
| Shared HYFL-inspired stylesheet | docs/assets/hyfl-mi.css | Shared responsive design layer for all published pages, including unified nav styling, mobile table/code wrapping, neutral cards, and removal of LFM2 colored row borders |
| Phase 2 page | docs/05-phase2-repeatability.html | Updated to link Phase 3 and note the L26 -> L14 revision |
| Phase 3 page | docs/06-phase3-gap-closure.html | Restyled to match the site and list remaining work |
| Phase 7 LFM2 Atlas | docs/07-lfm2-230m-atlas.html | Complete atlas of LFM2.5-230M, now using shared nav/style and conservative visible summary wording |
| Phase 8 SFT Sweep | docs/08-lfm2-230m-sft-sweep.html | 39 SFT experiments, best recipe identified |
| Phase 9 Format Ablation | docs/09-data-format-ablation.html | **CAVEAT: contains mock-judge data — see NR014** |
| Phase 9R Report | reports/09-data-format-ablation.md | Honest report with evidence tiers, training loss only |
| Bilawal.net mirror source | ../pretty-blog-python/pages/mi-atlas.html | Standalone bilawal.net copy with Swetrix analytics and absolute links back to the research repo/pages, including full-detail links for each audited claim |
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

## Distillation Scripts (2026-07-08)

| Script | Path | Purpose | Status |
|--------|------|---------|--------|
| teacher_scoring.py | teacher_scoring.py | Rank generated rollouts, select best generation, condense reasoning, append durable labels to `scored.jsonl` | Updated for OpenRouter `tencent/hy3:free`, generic OpenAI-compatible providers, provider metadata, and stop/resume on quota/auth/rate-limit exhaustion |
| TEACHER_SETUP.md | experiments/lfm25_8b/TEACHER_SETUP.md | Teacher setup and scoring commands | Updated with Tencent HY3/OpenRouter command and provider-swap resume instructions |
| DISTILL_PROGRESS.md | DISTILL_PROGRESS.md | Distillation pipeline status and next steps | Updated to make Tencent HY3/OpenRouter the recommended no-local-heat scoring path |

## Provider Smoke Tests (2026-07-09)

| Artifact | Path | Purpose | Status |
|----------|------|---------|--------|
| test_teacher_providers.py | scripts/eval/test_teacher_providers.py | Sanitized two-provider smoke test harness for HY3 rollout judging and opencode-go direct generation | New |
| hy3_rollout_judge_1783552065.json | results/provider_tests/hy3_rollout_judge_1783552065.json | Two OpenRouter `tencent/hy3:free` rollout-judge calls, each including all 6 generations for the prompt group | Complete |
| opencode_go_mimo_1783552085.json | results/provider_tests/opencode_go_mimo_1783552085.json | Two opencode-go `mimo-v2.5` direct generation calls with timing and local quality checks | Complete |
| teacher_scoring.py mixed mode | teacher_scoring.py | Split `MAX_WORKERS` across OpenRouter HY3 and opencode-go prompt-group workers while recording per-row provider metadata | Updated |

## Scored Label Exports (2026-07-09)

| Artifact | Path | Purpose | Status |
|----------|------|---------|--------|
| export_scored_sft.py | scripts/data/export_scored_sft.py | Export teacher-clean q>=8 labels into SFT JSONL and quarantine JSONL | New |
| validate_scored_sft.py | scripts/data/validate_scored_sft.py | Deterministically parse and schema-validate teacher-clean labels before SFT | New |
| scored.jsonl | /Users/bilawalriaz/scored/scored.jsonl | Mimo-v2.5 teacher labels for 1,728 rollout prompt groups | Complete except 2 unresolved prompts |
| sft_clean_q8_response.jsonl | /Users/bilawalriaz/scored/exports/sft_clean_q8_response.jsonl | Teacher-clean response-only SFT candidates (`format_valid`, q>=8, correct/na) | 1,240 rows |
| sft_clean_q8_reasoning.jsonl | /Users/bilawalriaz/scored/exports/sft_clean_q8_reasoning.jsonl | Teacher-clean reasoning+answer SFT candidates | 1,240 rows |
| sft_strict_q8_response.jsonl | /Users/bilawalriaz/scored/exports/sft_strict_q8_response.jsonl | Deterministically parse/schema-valid response-only SFT candidates | 827 rows |
| sft_strict_q8_reasoning.jsonl | /Users/bilawalriaz/scored/exports/sft_strict_q8_reasoning.jsonl | Deterministically parse/schema-valid reasoning+answer SFT candidates | 827 rows |
| sft_strict_manifest_q8_response.json | /Users/bilawalriaz/scored/exports/sft_strict_manifest_q8_response.json | Strict validation manifest and rejection counts | Complete |
| sft_strict_manifest_q8_reasoning.json | /Users/bilawalriaz/scored/exports/sft_strict_manifest_q8_reasoning.json | Strict validation manifest for reasoning+answer export | Complete |

## Agent Trajectory Lab (2026-07-09)

| Artifact | Path | Purpose | Status |
|----------|------|---------|--------|
| Hermes task runner | tools/agent_trajectory_lab/run_hermes_tasks.py | Runs Hermes one-shot tasks in isolated git workspaces and captures stdout/stderr, usage, diffs, verifier results, and session export attempts; subprocesses use process-group timeout termination and restarted workers can clean incomplete no-result run dirs | Updated |
| Seed agent tasks | tools/agent_trajectory_lab/agent_tasks.json | Twelve varied code/data/config/documentation tasks for initial trajectory collection | New |
| Lab README | tools/agent_trajectory_lab/README.md | Commands for lenovo/aero Hermes trajectory collection | New |
| lenovo lab copy | /home/billz/agent_trajectory_lab | Remote synced copy used for smoke tests | Created on lenovo |
| Atropos API on lenovo | http://lenovo:8000 | Running `run-api` service for future custom environment integration | Running via `nohup` |
| Sysadmin task generator | tools/agent_trajectory_lab/generate_sysadmin_tasks.py | Generates 125 sysadmin/security/deployment trajectory tasks | New |
| Sysadmin task queue | tools/agent_trajectory_lab/sysadmin_tasks.json | Generated queue with 125 tasks across eight operations families | New |
| Sysadmin fast shards | tools/agent_trajectory_lab/sysadmin_tasks_fast_shard_*.json | Four shard files for faster parallel collection across 115 non-host-audit tasks | New |
| Trajectory dataset exporter | tools/agent_trajectory_lab/export_trajectory_dataset.py | Converts passed task run directories into JSONL records with prompts, final answers, diffs, verifier results, changed files, and trace-file pointers/previews; filters generated cache/binary artifacts from `changed_files` | Updated |
| lenovo full collection run | /home/billz/agent_trajectory_lab/runs/sysadmin_stepfun_full | Original serial run; stopped after preserving first passed host-audit trace due slow host-audit throughput | Stopped |
| lenovo sharded collection runs | /home/billz/agent_trajectory_lab/runs/sysadmin_stepfun_fast_shard_{0..3} | Four Hermes collection workers using Nous `stepfun/step-3.7-flash:free` over the non-host-audit sysadmin queue | Complete |
| lenovo recovery run | /home/billz/agent_trajectory_lab/runs/sysadmin_stepfun_recovery_0 | Recovery pass for remaining shard-3 port-analysis and deployment-automation tasks | Complete |
| lenovo full sysadmin trajectory dataset | /home/billz/agent_trajectory_lab/datasets/sysadmin_stepfun_20260709.jsonl | Passed sysadmin/deployment Hermes trajectories, including a few records with ambiguous early trace exports | 106 records; 105 with trace files |
| lenovo clean sysadmin trajectory dataset | /home/billz/agent_trajectory_lab/datasets/sysadmin_stepfun_20260709_clean.jsonl | Strict SFT candidate export requiring verifier pass plus prompt-matched Hermes session trace | 102 records; 102 with matched trace files |
| Agentic scale task generator | tools/agent_trajectory_lab/generate_agentic_scale_tasks.py | Generates 700 templates and 7,000 parameterized verifier-backed task instances across seven agentic behavior families | New |
| Agentic scale task queue | tools/agent_trajectory_lab/agentic_scale_tasks.json | Full 7,000-instance HY3 collection queue | New |
| Agentic scale balanced queue | tools/agent_trajectory_lab/agentic_scale_tasks_balanced.json | Family-interleaved 7,000-instance queue for balanced collection throughput | New |
| Agentic scale templates | tools/agent_trajectory_lab/agentic_scale_tasks_templates.json | 700 template records, 100 per requested family | New |
| Agentic scale shards | tools/agent_trajectory_lab/agentic_scale_tasks_shard_*.json | Sixteen resumable shard files for parallel HY3 collection | New |
| Agentic scale balanced shards | tools/agent_trajectory_lab/agentic_scale_tasks_balanced_shard_*.json | Sixteen family-interleaved shard files; preferred for active HY3 collection | New |
| Agentic scale pilot | tools/agent_trajectory_lab/agentic_scale_tasks_pilot_7.json | One task per requested family for HY3 smoke/pilot | 7/7 passed on lenovo |
| Preference pair exporter | tools/agent_trajectory_lab/export_preference_pairs.py | Groups same-task rollouts into DPO pairs, preferring passed clean efficient traces and excluding transient API failures by default; sanitizes repository-internal/cache path noise from trainable transcript text | Updated |
| HY3 balanced worker supervisor | tools/agent_trajectory_lab/manage_hy3_balanced_workers.py | Keeps HY3 balanced shard collection moving while capping active shard workers and passing `--clean-incomplete` to recover no-result run dirs | Running on lenovo as PID 553902 with max-workers 7 |
| Training-ready split exporter | tools/agent_trajectory_lab/export_training_ready_splits.py | Creates deterministic stratified SFT/DPO train-validation splits and manifest from current trajectory exports; SFT rows include compact matched-trace `trajectory` and `trajectory_messages` fields; trainable transcripts sanitize repository-internal/cache path noise while preserving raw trace paths for audit | Updated |
| HY3 export refresher | tools/agent_trajectory_lab/refresh_hy3_exports.py | Periodically refreshes clean SFT export, DPO pairs, and stratified training-ready splits while collection workers run | Running on lenovo as PID 511189 |
| lenovo HY3 pilot run | /home/billz/agent_trajectory_lab/runs/hy3_agentic_scale_pilot | Seven-family Hermes/HY3 pilot through Nous Portal | 7 passed, 7 matched traces |
| lenovo HY3 family-sorted shards | /home/billz/agent_trajectory_lab/runs/hy3_agentic_scale_shard_{00..03}_r4 | Initial production workers; stopped after eight-worker rate-limit artifacts appeared | Partial |
| lenovo HY3 balanced shards | /home/billz/agent_trajectory_lab/runs/hy3_agentic_balanced_shard_{00..06}_r4 | Active balanced workers, four rollouts per task, transient retry/backoff and incomplete-run cleanup enabled | Running; seven-worker profile |
| lenovo HY3 partial balanced shards | /home/billz/agent_trajectory_lab/runs/hy3_agentic_balanced_shard_{06..07}_r4 | Briefly started and then stopped to avoid exceeding safe HY3 concurrency; completed outputs preserved for export/resume | Partial |
| lenovo HY3 state-compaction recovery | /home/billz/agent_trajectory_lab/runs/hy3_state_compaction_recovery_{0..1}_r4 | Targeted recovery workers used after verifier fix; stopped after the family crossed 100 clean records | Partial; preserved |
| lenovo HY3 partial clean dataset | /home/billz/agent_trajectory_lab/datasets/hy3_agentic_scale_partial_clean.jsonl | Strict partial export requiring verifier pass plus prompt-matched Hermes trace; cache/binary changed files filtered | 1,160 records; all seven families >100 |
| lenovo HY3 partial DPO pairs | /home/billz/agent_trajectory_lab/datasets/hy3_agentic_scale_partial_dpo.jsonl | Same-task rollout preference pairs, excluding transient API failures by default and tagging explicit rejected failure modes | 284 pairs |
| lenovo HY3 training-ready splits | /home/billz/agent_trajectory_lab/datasets/hy3_agentic_scale_training_ready | Stratified train/validation SFT and DPO JSONL files with manifest; SFT rows include compact sanitized tool-use trajectories, DPO rows include chosen/rejected chat messages and hard/efficiency tier files | SFT 1102/58, DPO 270/14, hard DPO 33/1, efficiency DPO 237/13 |

## LFM2.5-8B-A1B SFT Attempt (2026-07-09)

| Artifact | Path | Purpose | Status |
|----------|------|---------|--------|
| Unsloth QLoRA trainer | scripts/train/train_lfm25_8b_unsloth_qlora.py | Standalone SFT trainer for `LiquidAI/LFM2.5-8B-A1B` using 4-bit loading, fp16, rank-8 LoRA, tiny micro-batches, and auto target-module detection | New; dry-run OK; 8GB load failed before training |
| 8GB QLoRA config snapshot | configs/sft/lfm25_8b_a1b_unsloth_qlora_8gb.json | Reproducibility record for aero QLoRA attempts and failure modes | New |
| aero trainer copy | /home/billz/work/autonomous-small-model-exploration/scripts/train/train_lfm25_8b_unsloth_qlora.py | Remote copy used for aero load attempts | Synced |
| aero strict scored dataset copy | /home/billz/scored/exports/sft_strict_q8_response.jsonl | Strict 827-row response-only SFT candidate dataset copied to aero | Synced |
| aero training log | /home/billz/results/lfm25_8b_sft_q8_strict/train.log | Forced-CUDA load attempt log; failed with CUDA OOM during weight loading | Failed before training |
| aero embedding-offload log | /home/billz/results/lfm25_8b_sft_q8_strict/train_offload_embedding.log | Embedding-offload fallback log; failed with CUDA OOM during weight loading | Failed before training |

## LFM2.5-1.2B-Instruct SFT Run (2026-07-09)

| Artifact | Path | Purpose | Status |
|----------|------|---------|--------|
| 1.2B QLoRA config snapshot | configs/sft/lfm25_12b_instruct_unsloth_qlora_8gb.json | Reproducibility record for the successful aero 1.2B QLoRA run and smoke tests | New |
| aero 1.2B config copy | /home/billz/work/autonomous-small-model-exploration/configs/sft/lfm25_12b_instruct_unsloth_qlora_8gb.json | Remote copy used for run documentation | Synced |
| aero 1.2B adapter | /home/billz/results/lfm25_12b_instruct_sft_q8_strict/adapter | Final LoRA adapter trained from strict 827-row scored dataset | Complete; train loss 1.372 |
| aero 1.2B metadata | /home/billz/results/lfm25_12b_instruct_sft_q8_strict/metadata.json | Run metadata with model, dataset rows, training args, and metrics | Complete |
| aero 1.2B train log | /home/billz/results/lfm25_12b_instruct_sft_q8_strict/train.log | Full 300-step training log | Complete |
| aero 1.2B checkpoints | /home/billz/results/lfm25_12b_instruct_sft_q8_strict/checkpoints/checkpoint-{100,200,300} | Intermediate LoRA checkpoints | Complete |
| stalled dropout log | /home/billz/results/lfm25_12b_instruct_sft_q8_strict/train_stalled_dropout005.log | First 1.2B run with LoRA dropout 0.05; loaded but stalled at step 0 | Preserved diagnostic |

## GGUF Evaluation (2026-07-09)

| Artifact | Path | Purpose | Status |
|----------|------|---------|--------|
| prepare_mixed_blend.py | scripts/data/prepare_mixed_blend.py | Prepare SFT mixed dataset blend (formatting + coding + math) | New |
| mixed dataset blend | data/sft/mixed_blend_4k.jsonl | Shuffled 4,000-example SFT training dataset | New |
| mixed SFT config | configs/sft/lfm25_12b_instruct_unsloth_qlora_mixed.json | Training configuration snapshot for mixed run | New |
| export_gguf.py | scripts/train/export_gguf.py | Export Unsloth models/adapters to GGUF format | New |
| run_gguf_eval.py | scripts/eval/run_gguf_eval.py | Run evaluation on GGUF models on aero using llama-completion | New |
| run_gsm8k_eval.py | scripts/eval/run_gsm8k_eval.py | Run GSM8K math reasoning evaluation on GGUF models | New |
| compare_gguf_results.py | scripts/eval/compare_gguf_results.py | Compare base GGUF vs SFT GGUF outputs programmatically | New |
| | export_scaled_gguf.py | scripts/train/export_scaled_gguf.py | Export PEFT adapters with custom weight scaling | New |
| base model GGUF | /home/billz/models/LFM2.5-1.2B-Instruct-GGUF_gguf/LFM2.5-1.2B-Instruct.Q4_K_M.gguf | Quantized base model GGUF file | Created on aero |
| SFT model GGUF | /home/billz/results/lfm25_12b_instruct_sft_q8_strict/gguf_gguf/LFM2.5-1.2B-Instruct.Q4_K_M.gguf | Quantized fine-tuned model GGUF file | Created on aero |
| mixed model GGUF | /home/billz/results/lfm25_12b_instruct_sft_mixed/gguf_gguf/LFM2.5-1.2B-Instruct.Q4_K_M.gguf | Quantized mixed-blend fine-tuned model GGUF file | Created on aero |
| step-200 model GGUF | /home/billz/results/lfm25_12b_instruct_sft_q8_strict/gguf_step200_gguf/LFM2.5-1.2B-Instruct.Q4_K_M.gguf | Quantized step-200 fine-tuned model GGUF file | Created on aero |
| scale-0.3 model GGUF | /home/billz/results/lfm25_12b_instruct_sft_q8_strict/gguf_scaled_0.3_gguf/LFM2.5-1.2B-Instruct.Q4_K_M.gguf | Quantized scale-0.3 merged GGUF file | Created on aero |
| scale-0.7 model GGUF | /home/billz/results/lfm25_12b_instruct_sft_q8_strict/gguf_scaled_0.7_gguf/LFM2.5-1.2B-Instruct.Q4_K_M.gguf | Quantized scale-0.7 merged GGUF file | Created on aero |
| GGUF Comparison Report | results/evals/gguf_comparison_report.md | Comparative report detailing structured output, accuracy, length, and slop improvements | Complete |
| mixed GGUF Comparison Report | results/evals/gguf_mixed_comparison_report.md | Comparative report for mixed-blend model vs base model | Complete |
| step-200 GGUF Comparison Report | results/evals/gguf_step200_comparison_report.md | Comparative report for step-200 SFT model vs base model | Complete |
| scale-0.3 GGUF Comparison Report | results/evals/gguf_scaled_0.3_comparison_report.md | Comparative report for scale-0.3 SFT model vs base model | Complete |
| scale-0.7 GGUF Comparison Report | results/evals/gguf_scaled_0.7_comparison_report.md | Comparative report for scale-0.7 SFT model vs base model | Complete |
| SFT GGUF Outputs | results/evals/lfm25_12b_sft_gguf/outputs.jsonl | Inference outputs for SFT model | Complete |
| mixed GGUF Outputs | results/evals/lfm25_12b_sft_mixed_gguf/outputs.jsonl | Inference outputs for mixed-blend SFT model | Complete |
| step-200 GGUF Outputs | results/evals/lfm25_12b_sft_step200_gguf/outputs.jsonl | Inference outputs for step-200 SFT model | Complete |
| scale-0.3 GGUF Outputs | results/evals/lfm25_12b_sft_scaled_0.3_gguf/outputs.jsonl | Inference outputs for scale-0.3 merged model | Complete |
| scale-0.7 GGUF Outputs | results/evals/lfm25_12b_sft_scaled_0.7_gguf/outputs.jsonl | Inference outputs for scale-0.7 merged model | Complete |
| Base GGUF Outputs | results/evals/lfm25_12b_base_gguf/outputs.jsonl | Inference outputs for base model | Complete |
| SFT GSM8K Outputs | results/evals/gsm8k_sft_results.jsonl | GSM8K evaluation responses for SFT model | Complete |
| mixed GSM8K Outputs | results/evals/gsm8k_sft_mixed_results.jsonl | GSM8K evaluation responses for mixed-blend SFT model | Complete |
| step-200 GSM8K Outputs | results/evals/gsm8k_sft_step200_results.jsonl | GSM8K evaluation responses for step-200 SFT model | Complete |
| scale-0.3 GSM8K Outputs | results/evals/gsm8k_sft_scaled_0.3_results.jsonl | GSM8K evaluation responses for scale-0.3 SFT model | Complete |
| scale-0.7 GSM8K Outputs | results/evals/gsm8k_sft_scaled_0.7_results.jsonl | GSM8K evaluation responses for scale-0.7 SFT model | Complete |
| Base GSM8K Outputs | results/evals/gsm8k_base_results.jsonl | GSM8K evaluation responses for base model | Complete |
## Math and Formatting Multi-Adapter Merger (2026-07-09)

| Artifact | Path | Purpose | Status |
|----------|------|---------|--------|
| prepare_gsm8k_train.py | scripts/data/prepare_gsm8k_train.py | Downloads HF openai/gsm8k dataset and formats to ChatML, stripping calculator tags | New |
| merge_multi_adapters.py | scripts/train/merge_multi_adapters.py | Merges multiple adapters using PEFT add_weighted_adapter (LFM incompatible) | New |
| merge_lora_direct.py | scripts/train/merge_lora_direct.py | Merges multiple adapters directly into the base weights in PyTorch | New |
| math-only adapter | /home/billz/results/lfm25_12b_math_adapter/adapter | Final math reasoning LoRA adapter trained from cleaned GSM8K split | Complete; train loss 0.6861 |
| direct-merge GGUF | /home/billz/results/lfm25_12b_direct_merge_m1.0_f0.7_gguf/lfm25_12b_direct_merge_m1.0_f0.7.Q4_K_M.gguf | Quantized directly-merged multi-adapter GGUF file | Created on aero |
| direct-merge comparison report | results/evals/gguf_direct_merge_comparison_report.md | Comparative evaluation report for directly merged model vs base model | Complete |
| direct-merge JSON outputs | results/evals/lfm25_12b_direct_merge_m1.0_f0.7_gguf/outputs.jsonl | Inference outputs on 153 formatting prompts | Complete |
| direct-merge GSM8K outputs | results/evals/gsm8k_direct_merge_m1.0_f0.7_results.jsonl | GSM8K evaluation responses for directly merged model | Complete |

## MiniCPM5-1B Migration and Multi-Adapter Stacking (2026-07-09)

| Artifact | Path | Purpose | Status |
|----------|------|---------|--------|
| MiniCPM5-1B formatting adapter | /home/billz/results/minicpm5_1b_format_adapter/adapter | LoRA formatting adapter trained on strict 827-row scored dataset | Complete; loss 0.086 |
| MiniCPM5-1B math adapter | /home/billz/results/minicpm5_1b_math_adapter/adapter | LoRA math adapter trained on cleaned GSM8K split | Complete; loss 0.720 |
| Directly-merged GGUF model | /home/billz/results/minicpm5_1b_direct_merge_m1.0_f0.7_gguf/minicpm5_1b_direct_merge_m1.0_f0.7.Q4_K_M.gguf | Quantized directly merged Math (1.0) and Format (0.7) GGUF file | Created on aero |
| Formatting eval outputs (Base) | results/evals/minicpm5_1b_base_gguf/outputs.jsonl | Formatting outputs of base model | Complete |
| Formatting eval outputs (Format) | results/evals/minicpm5_1b_format_gguf/outputs.jsonl | Formatting outputs of format-only model | Complete |
| Formatting eval outputs (Math) | results/evals/minicpm5_1b_math_gguf/outputs.jsonl | Formatting outputs of math-only model | Complete |
| Formatting eval outputs (Merge) | results/evals/minicpm5_1b_direct_merge_m1.0_f0.7_gguf/outputs.jsonl | Formatting outputs of directly merged model | Complete |
| GSM8K evaluation responses (Base) | results/evals/gsm8k_minicpm5_1b_base_results.jsonl | GSM8K responses of base model | Complete |
| GSM8K evaluation responses (Math) | results/evals/gsm8k_minicpm5_1b_math_results.jsonl | GSM8K responses of math-only model | Complete |
| GSM8K evaluation responses (Merge) | results/evals/gsm8k_minicpm5_1b_direct_merge_m1.0_f0.7_results.jsonl | GSM8K responses of directly merged model | Complete |
| GSM8K accuracy summary (Base) | results/evals/gsm8k_minicpm5_1b_base_results_summary.json | GSM8K accuracy summary for base model | Complete |
| GSM8K accuracy summary (Math) | results/evals/gsm8k_minicpm5_1b_math_results_summary.json | GSM8K accuracy summary for math model | Complete |
| GSM8K accuracy summary (Merge) | results/evals/gsm8k_minicpm5_1b_direct_merge_m1.0_f0.7_results_summary.json | GSM8K accuracy summary for directly merged model | Complete |
| M01 experiment card | experiments/cards/M01.md | Pre-registered RS-LoRA merge-equivalence validation | Complete |
| M01 config | configs/M01_minicpm5_merge_validation.json | Reproducible merger validation configuration | Complete |
| Corrected direct merger | scripts/train/merge_lora_direct.py | RS-LoRA-aware, union-key direct delta sum | Complete |
| M01 equivalence result | results/evals/M01_minicpm5_merge_equivalence_rslora.json | FP16 tensor/logit comparison to sequential PEFT merge | Complete |
| Corrected MiniCPM Q4 smoke | results/evals/M01_merge_q4_smoke/outputs.jsonl | Export/load smoke; not valid behavioral scoring due template residue | Complete, negative harness finding |
| MiniCPM capability plan | reports/minicpm5_capability_plan.md | Staged expert, merge, consolidation, and release gates | Complete |
