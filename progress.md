# Progress

## Current phase: Phase 3 — Gap Closure and Gem Discovery

**Started:** 2026-06-23
**Goal:** Convert exploratory findings into reviewer-grade, falsifiable, reproducible claims. Hunt for surprising exceptions.
**Hardware:** aero (RTX 2070 Super 8GB)
**Repository:** bilawalriaz/autonomous-small-model-exploration

## Phase 3 status

### Completed
- [x] Full claims audit (20 claims classified in claims.md)
- [x] Threats to validity catalogued (14 threats in threats.md)
- [x] Gems inventory (10 candidate gems in gems.md)
- [x] Phase 3 file structure created
- [x] Phase 3 infrastructure: 16 experiment scripts (6560 lines), orchestrator (24 blocks)
- [x] Phase 3 gap closure report (reports/phase3_gap_closure.md)
- [x] Small model surgery skill (skills/small_model_surgery.md, 10 rules)
- [x] R1: 0.5B multi-seed ablation — hub at L2, std=0.0, ROBUST
- [x] R2: 1.5B multi-seed ablation — hub at L14 (REVISED from L26), std=0.0, ROBUST

### In progress
- [x] Phase 3 infrastructure COMPLETE — 16 scripts, orchestrator, all docs
- [x] R1-R5: Multi-seed replication — ALL COMPLETE (hub at L2/L14/L34, std=0.0)
- [x] L1-L3: Atlas-guided LoRA — COMPLETE (13.8x fewer params, equal accuracy on JSON)
- [x] L4-L5: Rank/module sweep — COMPLETE
- [x] C1-C3: Better causal tests — COMPLETE (position ablation, module ablation, method comparison)
- [x] P1-P3: Prompt robustness — COMPLETE (NL hubs validated, coder hub at L22!)
- [x] Q1: Quantization steering — COMPLETE (476x amplification at 4-bit)
- [x] GitHub Pages navigation/design pass — all pages share a HYFL-inspired responsive style, Phase 2 links Phase 3, and mobile overflow checks pass at 390px
- [ ] C4: Steering controls — needs HF-native steering rewrite
- [ ] G1: Steering direction transfer — needs memory optimization
- [ ] G3: Checkpoint lock-in — needs PEFT wrapper fix for ablation
- [ ] G4: Atlas-guided skip — needs recovery finetune DataLoader fix

### Blocked
- [ ] Remaining GPU-dependent experiments blocked on aero (offline as of 2026-06-23)

### Next actions (priority order)
1. **C4 steering controls** — Rewrite steering controls against the HF-native steering API; add random-vector and shuffled-label baselines.
2. **G1 steering direction transfer** — Reduce memory footprint for cross-scale direction transfer; avoid simultaneous full 2-model GPU residency where possible.
3. **G3 checkpoint lock-in at 1.5B** — Fix PEFT wrapper attribute access so checkpoint ablation can run on the 1.5B adapter timeline.
4. **G4 atlas-guided layer skip + recovery** — Fix the recovery finetune DataLoader and rerun skip+recovery evaluation.
5. **Publish docs** — Push the updated GitHub Pages HTML once reviewed locally.

## Phase 3 experiment plan

### Priority 1: Replication (closes T01, strengthens all claims)
- P3-REPL-001: 0.5B layer ablation x3 seeds (42, 137, 256)
- P3-REPL-002: 1.5B layer ablation x3 seeds
- P3-REPL-003: 3B layer ablation x3 seeds
- P3-REPL-004: 0.5B steering x3 seeds (at L2, L8, L12, L19)
- P3-REPL-005: 1.5B steering x3 seeds (at L6, L21, L26)

### Priority 2: Atlas-guided LoRA (closes C04, C05, C06 — the most valuable experiments)
- P3-LORA-001: Atlas-guided vs random-layer vs all-linear LoRA on 0.5B (JSON)
- P3-LORA-002: Atlas-guided vs random-layer vs all-linear LoRA on 0.5B (factual)
- P3-LORA-003: Atlas-guided vs random-layer vs all-linear LoRA on 0.5B (code)
- P3-LORA-004: Rank sweep with task accuracy (r=2,4,8,16) on 0.5B
- P3-LORA-005: Module sweep with task accuracy (o_proj, v_proj, q_proj, k_proj, all) on 0.5B

### Priority 3: Causal method improvements (closes T04, strengthens evidence quality)
- P3-CAUSAL-001: Full ablation method comparison at all layers (0.5B)
- P3-CAUSAL-002: Token-position-specific ablation at all layers (not just key layers)
- P3-CAUSAL-003: Module-specific ablation (q/k/v/o/up/down/gate) at hub layers
- P3-CAUSAL-004: Random-vector and shuffled-label controls for steering

### Priority 4: Prompt robustness (closes T03, T05)
- P3-PROMPT-001: Hub identification with 50+ natural language prompts (0.5B)
- P3-PROMPT-002: Steering effectiveness vs prompt length (short/medium/long)
- P3-PROMPT-003: Hub identification on Qwen2.5-Coder-0.5B

### Priority 5: Quantization (closes T06)
- P3-QUANT-001: Layer ablation on 4-bit NF4 0.5B
- P3-QUANT-002: Steering on 4-bit NF4 0.5B
- P3-QUANT-003: Layer ablation on 4-bit NF4 1.5B

### Priority 6: Gem hunting (anomaly detection)
- P3-GEM-001: Steering direction transfer across scales
- P3-GEM-002: Knockout controls (random-vector baseline)
- P3-GEM-003: Checkpoint lock-in at 1.5B
- P3-GEM-004: Atlas-guided layer skip + recovery finetune

## Key findings so far (carried from Phase 1-2)

See claims.md for full audit. Summary:
- 4 claims at HIGH confidence (C01, C02, C10, C13)
- 3 claims at MEDIUM-HIGH (C08, C14, C20)
- 8 claims at MEDIUM (C03, C04, C07, C09, C12, C15, C18, and one refuted C19)
- 3 claims at LOW (C06, C11, C16, C17)
- 10 candidate gems identified (G01-G10)
- 14 methodological threats identified (T01-T14)

## Phase 1-2 summary (archived)

21 Phase 1 experiments + Phase 2 blocks (A-I partial). 3 model scales (0.5B, 1.5B, 3B) + 1 cross-family (SmolLM2-1.7B). 40+ result files. GitHub Pages site published. Key infrastructure: run_full_atlas.py (Phase 1), run_full_phase2_atlas.py (Phase 2), experiment registry, claim cards, task suite (4300 examples).

---

## Phase 9: Data Format Ablation (2026-06-29)

**Goal:** Determine the optimal information shape for fine-tuning 230M-500M language models through controlled format ablation.
**Motivation:** Phase 8 showed dataset format dominates hyperparameters (5x impact). Phase 9 isolates format from content.
**Hardware:** aero (RTX 2070 Super 8GB)

### Phase 9 status

#### Completed
- [x] Directory structure created (configs/sft/, configs/eval/, configs/experiments/, data/canonical/, data/sft/, data/eval/, scripts/data/, scripts/train/, scripts/eval/, scripts/report/, adapters/, results/evals/, results/drift/, reports/phase9/)
- [x] Baseline configs frozen (quality + surgical)
- [x] Eval config created
- [x] 4 experiment configs created (format_ablation_quality, format_ablation_surgical, bsmagpie_v1_quality, bsmagpie_v1_surgical)
- [x] Experiment index initialized (10 planned runs)
- [x] AGENTS.md updated with Phase 9 rules

#### In progress
- [ ] Canonical dataset generation (300 examples, 9 domains)
- [ ] Eval dataset generation (150 prompts, 9 categories)
- [ ] Data pipeline scripts (compile, render, validate)
- [ ] Eval pipeline scripts (harness, judge, aggregate, manual review, KL drift)
- [ ] Training scripts (train_lfm2_sft, run_format_ablation)
- [ ] Report generator

#### Planned
- [ ] Render 6 format variants
- [ ] Validate format variants
- [ ] Train 6 format ablation adapters (quality)
- [ ] Train 2 surgical adapters
- [ ] Evaluate all adapters + base
- [ ] Judge all outputs
- [ ] Aggregate results
- [ ] Generate manual review samples
- [ ] Compute KL drift
- [ ] Train bilawal_smol_magpie_v1 adapters
- [ ] Write Phase 9 report

#### Hypotheses
- H1: Multi-turn concise is genuinely better for small-model SFT
- H2: smol-magpie advantage is partly format
- H3: Small models benefit from dense compact examples
- H4: Training loss doesn't correlate perfectly with behavioral quality
- H5: Surgical LoRA preserves base model while adding useful behavior
- H6: Structured terse outperforms verbose on JSON/code/extraction
- H7: There is a small-model-native data style

#### Pipeline
```
generate_canonical_dataset.py → phase9_pilot_300.jsonl
render_dataset_formats.py → 6 format variants
validate_dataset_formats.py → validation report
run_format_ablation.py → train all adapters
run_eval_harness.py → generate outputs
judge_outputs.py → score outputs
aggregate_eval_results.py → summary metrics
export_manual_review.py → human review
compute_kl_drift.py → drift analysis
build_phase09_report.py → final report
```

## 2026-06-29 17:40 — Phase 9 Infrastructure Complete

### Completed
- [x] All 7 configs frozen (2 SFT baseline, 1 eval, 4 experiment)
- [x] Canonical dataset: 300 examples, 9 domains (data/canonical/phase9_pilot_300.jsonl)
- [x] Eval dataset: 153 prompts, 9 categories (data/eval/small_model_eval_v1.jsonl)
- [x] Dataset compiler (scripts/data/compile_sft_dataset.py) — 6 format renderers
- [x] Format renderer (scripts/data/render_dataset_formats.py) — batch render + manifest
- [x] Dataset validator (scripts/data/validate_dataset_formats.py) — all checks pass
- [x] 6 format variants rendered and validated (data/sft/format_ablation/)
- [x] Eval harness (scripts/eval/run_eval_harness.py)
- [x] Judge scorer (scripts/eval/judge_outputs.py) — pointwise + pairwise, mock fallback
- [x] Aggregator (scripts/eval/aggregate_eval_results.py)
- [x] Manual review exporter (scripts/eval/export_manual_review.py)
- [x] KL drift (scripts/eval/compute_kl_drift.py) — proxy + full mode
- [x] Training script (scripts/train/train_lfm2_sft.py) — handles alpaca + chat formats
- [x] Format ablation runner (scripts/train/run_format_ablation.py) — orchestrator with dry-run
- [x] Report generator (scripts/report/build_phase09_report.py)
- [x] Experiment index (experiments/index.jsonl) — 10 planned runs
- [x] AGENTS.md updated with Phase 9 rules
- [x] Pipeline dry-run verified: 26 steps across 6 formats

### Next actions (requires GPU on aero)
1. Render datasets on aero: python scripts/data/render_dataset_formats.py
2. Run full ablation: python scripts/train/run_format_ablation.py --config configs/experiments/format_ablation_quality.yaml
3. Or run individual format: python scripts/train/run_format_ablation.py --config configs/experiments/format_ablation_quality.yaml --format multi_turn_concise

## 2026-06-29 18:28 — Phase 9 Training In Progress (3/6 complete)

### Monitor script bug fix
The monitor script `scripts/report/monitor_phase9_training.py` was constructing run_ids
without the date suffix (`_20260629`), so it couldn't find completed adapter dirs.
Fixed to use glob matching on aero.

### Training status (as of 18:28 UTC)
| Format | Status | Final Loss | Runtime |
|--------|--------|-----------|---------|
| alpaca_flat | ✅ complete | 1.7321 | 895s (~15min) |
| single_turn_chat | ✅ complete | 1.7475 | 400s (~7min) |
| multi_turn_concise | ✅ complete | 1.5156 | 1005s (~17min) |
| multi_turn_verbose | ⏳ training | — | ~4min elapsed |
| structured_terse | queued | — | — |
| bad_format_control | queued | — | — |

### Early observation
multi_turn_concise has the lowest training loss so far (1.516 vs 1.73-1.75 for other formats).
This is consistent with Phase 8 finding that dataset format dominates hyperparameters.

### Next
- Wait for all 6 formats to complete (~45 min remaining)
- Run --update-now to update the report HTML
- Commit and push
- Proceed to eval harness
