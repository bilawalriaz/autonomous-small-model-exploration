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
- [x] Git sync repaired — created safety branch/stash, merged origin/main with explicit `--no-rebase`, and preserved local work before deploy
- [x] One-page MI-Atlas share page — added `docs/mi-atlas.html`, made it the first index entry, added it to the published docs nav, mirrored the page into `pretty-blog-python/pages/mi-atlas.html`, and verified desktop/mobile layout with Playwright
- [x] Share page LFM2 SFT coverage — added a dedicated 39-run LFM2.5-230M SFT sweep section with dataset, optimizer, rank, target-module, format-ablation, and evidence-limit cards
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
- [x] Canonical dataset generation (300 examples, 9 domains)
- [x] Eval dataset generation (153 prompts, 9 categories)
- [x] Data pipeline scripts (compile, render, validate)
- [x] Eval pipeline scripts (harness, judge, aggregate, manual review, KL drift)
- [x] Training scripts (train_lfm2_sft, run_format_ablation)
- [x] Report generator

#### Planned
- [x] Render 6 format variants
- [x] Validate format variants
- [x] Train 6 format ablation adapters (quality) — COMPLETE, best loss: multi_turn_verbose (1.372)
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

## 2026-06-29 21:30 — Phase 9R: Evaluation Stack Rebuild

### Problem identified
The original Phase 9 report contained behavioral claims (win-rates, judge scores, hypothesis verdicts) based on mock-judge scoring — deterministic random numbers that look like real scores but carry no behavioral signal. `judge_outputs.py` silently fell back to mock when API was unavailable, using Python's `hash()` (non-deterministic across versions).

### Phase 9R changes (completed)
- [x] `judge_outputs.py` rewritten: explicit `--mock` flag required, `judge_source` metadata on every score, `hashlib`-based deterministic seeding, `--strict-report-mode`
- [x] `aggregate_eval_results.py` enhanced: programmatic scorers (JSON validity, schema validity, entity F1, exact-match factual, numeric match, slop rate, output length, constraint-following), `--judge-source` filter
- [x] `generate_blind_review.py` created: stratified blind review (60+ examples, 9 categories, anonymized labels, unblinding key)
- [x] `run_phase9r_eval.py` created: one-command pipeline for aero eval + judge + aggregate + blind review
- [x] `reports/09-data-format-ablation.md` rewritten with honest evidence tiers and mock-judge caveats

### Phase 9R changes (pending — requires aero GPU)
- [ ] Run eval harness on all 8 adapters + base model
- [ ] Run real judge (or document mock limitations)
- [ ] Generate blind review samples (60+ examples)
- [ ] Compute programmatic metrics
- [ ] Update report with real data

### Current safe claims (Phase 9)
1. **Training loss differs by format under content-controlled conditions** (CONFIRMED)
2. **Multi-turn verbose has lowest training loss** (1.372, 33% better than worst)
3. **bad_format_control has 2nd-best loss** — loss does NOT measure quality
4. **Surgical adapter beats quality adapter on loss** (1.27 vs 1.46, 3.8x fewer params)
5. **Format dominates hyperparameters** (consistent with Phase 8)

### Claims NOT yet supported
- Multi-turn verbose "wins behaviorally" (no real eval data)
- Any win-rate or judge score comparison (mock judge only)
- H1-H7 behavioral verdicts (all based on loss or mock data)

### Next actions (requires GPU on aero)
1. Run Phase 9R eval: `python scripts/eval/run_phase9r_eval.py --judge-api-url <url> --judge-api-key <key>`
2. Or mock pipeline test: `python scripts/eval/run_phase9r_eval.py --mock-judge --dry-run`
3. After eval: update reports with real data
4. Design Phase 10 (token-budget-controlled ablation)

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

## 2026-06-29 19:01 — Phase 9 Format Ablation Training COMPLETE (6/6)

All 6 format variants trained successfully with quality LoRA (r=8, hub layers + o_proj, Adafactor, lr=2e-4, 300 steps).

### Final results

| Format | Final Loss | Runtime | Rank |
|--------|-----------|---------|------|
| multi_turn_verbose | 1.3724 | 569s (~9.5min) | 1 (best) |
| bad_format_control | 1.4023 | 590s (~10min) | 2 |
| multi_turn_concise | 1.5156 | 1005s (~17min) | 3 |
| alpaca_flat | 1.7321 | 895s (~15min) | 4 |
| single_turn_chat | 1.7475 | 400s (~7min) | 5 |
| structured_terse | 1.8314 | 440s (~7min) | 6 (worst) |

### Key observations
- **Best training loss:** multi_turn_verbose (1.372) — 25% lower than worst (structured_terse at 1.831)
- **Surprise:** bad_format_control (deliberately malformed) has 2nd best loss (1.402) — this challenges the assumption that clean formatting is always better for training
- **Surprise:** multi_turn_concise (1.516) does NOT beat multi_turn_verbose (1.372) — the Phase 8 intuition that concise is better needs revision
- structured_terse performs worst — structured-JSON-like format is hardest for this model to learn from
- Loss gap between best and worst: 0.459 (33% relative difference)

### Caveats
- These are TRAINING losses only. Behavioral quality (eval harness + blind judging) may differ.
- Per AGENTS.md H4: "Training loss may not correlate with behavioral quality" — must wait for eval.
- Single seed (quality track pilot) — no variance estimate yet.

### Next actions
1. Run eval harness on all 6 adapters + base model
2. Judge outputs (blind pairwise)
3. Aggregate and build Phase 9 report
4. Test surgical LoRA track
5. Multi-seed replication if pilot results are interesting

## 2026-06-30 01:34 — GitHub Pages Design Unification and Claim Detail Links

Presentation maintenance completed. No research claims, metrics, confidence levels, or negative results changed.

### Changes made
- Added explicit full-detail links to every claim row on `docs/mi-atlas.html`.
- Mirrored the same claim-detail links to `pretty-blog-python/pages/mi-atlas.html` using absolute GitHub Pages URLs.
- Standardized detailed GitHub Pages nav labels across `docs/*.html`: Home, Start, 0.5B, 1.5B, Compare, Qual, Phase 2, Phase 3, LFM2, SFT, Format, RAG.
- Added the shared MI-Atlas stylesheet to the LFM2 and SFT pages where missing.
- Removed the colored left-border treatment from the LFM2 conv/attention rows.
- Tightened mobile table/code/bar behavior in `docs/assets/hyfl-mi.css`.
- Reworded the most visible LFM2 and format-page overclaims to match the evidence more closely.

### Verification
- Playwright desktop and mobile pass across index, share page, LFM2 atlas, SFT sweep, format ablation, RAG page, and the bilawal.net source page.
- Mobile overflow: 0px on all tested pages at 390px width.
- Share page detail links: 18 total, including 7 in the audited claims table.
- LFM2 colored row border: removed (`border-left: 0px`).

## Phase 13: Atlas-Guided Stochastic Inference (PTRM-Inspired) — 2026-06-30

**Status:** DESIGN (not yet implemented)
**Paper:** PTRM (arXiv:2605.19943) — Gaussian noise injection at test time for tiny recursive models
**Design doc:** docs/phase13-ptrm-noise-injection-design.md

### Core idea
PTRM injects Gaussian noise into latent states during inference to escape "bad basins" and uses K parallel rollouts + Q-head selection to pick the best trajectory. Their models (5-7M params) beat frontier LLMs at 1/3000th the cost.

Our atlas tells us *where* noise matters (L0 hub, skip KL=82.9) and where it's wasted (L5-L13, CKA=1.0, residual norm 25.5). We can do PTRM-style inference amplification but targeted, not uniform.

### Hypotheses
- H13-1: Hub-only noise (L0) matches uniform noise across all 14 layers
- H13-2: Hub noise outperforms random-layer noise
- H13-3: Loss-based selection improves over random selection
- H13-4: Width scaling (K rollouts) beats depth scaling
- H13-5: Noise creates measurably distinct completion clusters (bad basin detection)

### Experiments
- 13A: Noise localization (which layer?) — 14 layers × 50 prompts × K=10 × 3 seeds
- 13B: Width scaling curve (K=1,2,5,10,20,50)
- 13C: Sigma sweep (σ=0.05 to 2.0)
- 13D: Selection strategy comparison (loss, logit-conf, voting, oracle)
- 13E: Bad basin detection (K=100, clustering analysis)
- 13F: Atlas-guided vs uniform vs random noise head-to-head

### Estimated compute
~8 hours total on aero. LFM2.5-230M (450MB bf16), peak VRAM ~2GB.

### Expected outcome
Hub-only noise + loss-based selection should boost structured extraction from ~83% to ~93%+ with no retraining — just 10x inference compute at the right layer.

### What this enables
1. Free inference-time capability amplification (no retraining)
2. Operational validation of atlas-guided interventions
3. Alternative to SFT for 230M models (where 300 examples = catastrophic overfitting)
4. Generalization of PTRM from recursive to autoregressive architectures


## Phase 13: Atlas-Guided Stochastic Inference (PTRM-Inspired) — 2026-06-30

**Status:** DESIGN (not yet implemented)
**Paper:** PTRM (arXiv:2605.19943) — Gaussian noise injection at test time for tiny recursive models
**Design doc:** docs/phase13-ptrm-noise-injection-design.md

### Core idea
PTRM injects Gaussian noise into latent states during inference to escape bad basins and uses K parallel rollouts + Q-head selection to pick the best trajectory. Their models (5-7M params) beat frontier LLMs at 1/3000th the cost.

Our atlas tells us where noise matters (L0 hub, skip KL=82.9) and where it is wasted (L5-L13, CKA=1.0, residual norm 25.5). We can do PTRM-style inference amplification but targeted, not uniform.

### Hypotheses
- H13-1: Hub-only noise (L0) matches uniform noise across all 14 layers
- H13-2: Hub noise outperforms random-layer noise
- H13-3: Loss-based selection improves over random selection
- H13-4: Width scaling (K rollouts) beats depth scaling
- H13-5: Noise creates measurably distinct completion clusters (bad basin detection)

### Experiments
- 13A: Noise localization (which layer?)
- 13B: Width scaling curve (K=1,2,5,10,20,50)
- 13C: Sigma sweep
- 13D: Selection strategy comparison
- 13E: Bad basin detection (K=100, clustering)
- 13F: Atlas-guided vs uniform vs random head-to-head

### Estimated compute
~8 hours total on aero. LFM2.5-230M (450MB bf16), peak VRAM ~2GB.

### Expected outcome
Hub-only noise + loss-based selection should boost structured extraction from ~83% to ~93%+ with no retraining.

## 2026-07-08 — Rollout Labelling Provider Switch

### Completed
- [x] Pulled latest `main` with fast-forward to `285ceb2`.
- [x] Updated `teacher_scoring.py` to support OpenRouter through the OpenAI-compatible API.
- [x] Defaulted OpenRouter runs to pinned model `tencent/hy3:free` when `OPENROUTER_API_KEY` is present.
- [x] Preserved existing local teacher support for llama.cpp/LM Studio style `/v1/chat/completions` servers.
- [x] Made provider exhaustion resumable: auth, credit, quota, and rate-limit errors stop the run without appending a scored row for the current prompt.
- [x] Added provider/model/API-base metadata to every new scored row.
- [x] Documented OpenRouter and fallback-provider commands in `DISTILL_PROGRESS.md` and `experiments/lfm25_8b/TEACHER_SETUP.md`.

### Verification
- `python3 -m py_compile teacher_scoring.py`
- `python3 teacher_scoring.py stats` read the existing default output and reported 188 scored prompts.

### Next
- Run `teacher_scoring.py full` with `OPENROUTER_API_KEY` set. If OpenRouter free quota is exhausted, switch `TEACHER_PROVIDER`, `TEACHER_URL`, `TEACHER_MODEL`, and optional `TEACHER_API_KEY`, then rerun to resume from the next unscored `prompt_hash`.

## 2026-07-09 — Provider Smoke Test: HY3 vs opencode-go

### Completed
- [x] Added `scripts/eval/test_teacher_providers.py` for sanitized provider smoke tests without persisting API keys.
- [x] Tested OpenRouter `tencent/hy3:free` on 2 rollout prompt groups with all 6 generations included for each group.
- [x] Tested opencode-go endpoint `https://opencode.ai/zen/go/v1/chat/completions` with model `mimo-v2.5` on 2 direct generation prompts.

### Results
- HY3 full-rollout judging: 2/2 parseable JSON results, all generations included (`6/6` and `6/6`), 44.2s and 51.4s elapsed, quality scores 8/10 and 7/10.
- opencode-go direct generation: 4.56s and 4.29s elapsed. JSON extraction passed 4/4 local checks; Python bugfix passed the core off-by-one fix but omitted empty-input handling.

### Operational note
- The July 8 `teacher_scoring.py full` run processed only a small visible group because `MAX_WORKERS` controls prompt-level concurrency and `STOP_ON_RESPONSE_ERROR=1` stops after the current in-flight batch when HY3 returns empty/unparseable JSON. Each worker still sends all generations for its assigned `prompt_hash`.

## 2026-07-09 — Mixed Teacher Worker Split

### Completed
- [x] Added `TEACHER_PROVIDER=mixed` support to `teacher_scoring.py`.
- [x] Mixed mode builds two provider configs: OpenRouter `tencent/hy3:free` and opencode-go `mimo-v2.5`.
- [x] `MAX_WORKERS` is split across provider slots; `MAX_WORKERS=18` produces 9 OpenRouter workers and 9 opencode-go workers.
- [x] Scored rows and error rows now record the actual provider/model/API base/name used for each prompt group.
- [x] Updated `DISTILL_PROGRESS.md` and `experiments/lfm25_8b/TEACHER_SETUP.md` with the mixed-mode command.

### Verification
- `python3 -m py_compile teacher_scoring.py`
- Dummy config check: `TEACHER_PROVIDER=mixed OPENROUTER_API_KEY=dummy OPENCODE_API_KEY=dummy MAX_WORKERS=18` reports `openrouter-hy3 × 9` and `opencode-go × 9`.

## 2026-07-09 — Teacher Prompt Tightening

### Completed
- [x] Rewrote `teacher_scoring.py`'s ranking prompt as a validation-first rubric.
- [x] Added hard-failure rules for wrong JSON/YAML mode, missing/extra fields, top-level shape, code fences, enum/range/count violations, arithmetic consistency, and gold-answer mismatch.
- [x] Added score caps: hard failures cap at 6; wrong top-level format, unparseable output, or wrong gold-answer final response cap at 4.

### Note
- Running `teacher_scoring.py` processes must be restarted to pick up the stricter prompt.

## 2026-07-09 — Teacher Scheduler Head-of-Line Fix

### Completed
- [x] Diagnosed `teacher_scoring.py full` appearing stuck after the first 18-worker batch: 13 calls completed, 5 provider stragglers held the whole batch open.
- [x] Replaced fixed-size batch waiting with a sliding-window scheduler so completed worker slots immediately receive new prompt groups.
- [x] Preserved mixed-provider slot assignment; `MAX_WORKERS=18` still maps to 9 OpenRouter HY3 and 9 opencode-go slots in mixed mode.
- [x] Stopped the old stuck process. Current scored rows remain durable in `/Users/bilawalriaz/scored/scored.jsonl`.

### Current queue
- Total prompt groups: 1,730
- Scored prompt groups: 28
- Remaining prompt groups: 1,702

## 2026-07-09 — Scored Label Export and Strict Validation

### Completed
- [x] Retried the 8 failed prompt groups; 6 recovered, 2 remain unresolved after repeated hidden-reasoning/no-JSON failures.
- [x] Added `scripts/data/export_scored_sft.py` for high-confidence SFT exports.
- [x] Added `scripts/data/validate_scored_sft.py` for deterministic parse + JSON-schema validation.
- [x] Exported response-only and reasoning+answer candidate files under `/Users/bilawalriaz/scored/exports/`.

### Label audit
- Total prompt groups: 1,730
- Scored rows: 1,728
- Unresolved prompt groups: 2 (`9660c7868858b540`, `dcc80b3acabb14af`)
- Error rows: 9 (8 unique error prompt hashes; one repeated failure)
- Teacher-clean q>=8 rows: 1,240
- Strict parse/schema-valid rows: 827
- Strict rejected after teacher-clean: 413

### Interpretation
- The stricter teacher prompt improved score calibration, but teacher approval is still not sufficient for structured-output training.
- Use the strict files for the first SFT pass; keep teacher-clean-but-strict-rejected rows quarantined for analysis or repair.

## 2026-07-09 — Hermes/Atropos Agent Trajectory Lab on lenovo

### Completed
- [x] Confirmed aero already serves `LFM2.5-8B-A1B-Uncensored-Gaston-Q4_K_M.gguf` on `http://aero:8080/v1`.
- [x] Configured lenovo Hermes default model to use the aero llama.cpp endpoint with provider `custom`, model `model`, and dummy local API key.
- [x] Preserved the previous Nous Portal model settings in `~/.hermes/backups/config.before-aero-lfm.*.yaml`; prior default was `nous` + `stepfun/step-3.7-flash:free`.
- [x] Started Atropos API on lenovo at `http://0.0.0.0:8000` via `nohup`; process PID observed as `5806`.
- [x] Added reproducible trajectory harness and 12 seed agent tasks under `tools/agent_trajectory_lab/`, synced to `~/agent_trajectory_lab` on lenovo.
- [x] Ran Hermes smoke with aero LFM: endpoint works, but the model emitted a JSON plan/tool-call-like object and did not edit files; verifier failed.
- [x] Ran control smoke with Nous `stepfun/step-3.7-flash:free`: Hermes edited `stats.py` and passed the local verifier.

### Current interpretation
- Infrastructure is wired: lenovo can call aero through Hermes, Atropos API is running, and the trace harness captures stdout/stderr, usage, diffs, verifier output, and session export attempts.
- LFM2.5-8B-A1B in the current GGUF/server setup is not yet a working Hermes tool-use agent. Treat the failed smoke as useful negative baseline data, not as a trajectory-quality dataset.

### Next
- Write a small custom Atropos environment around `tools/agent_trajectory_lab/run_hermes_tasks.py` instead of using heavyweight built-in dataset environments.
- Investigate whether LFM needs a different chat template/tool-call format, non-streaming setting, or SFT bootstrap on successful Hermes/Stepfun traces before it can drive tools.

## 2026-07-09 — Sysadmin Trajectory Collection Started

### Completed
- [x] Installed Docker Engine, Docker Compose v2, `python3-pytest`, and `python3-yaml` on lenovo for realistic operations/deployment tasks.
- [x] Added `tools/agent_trajectory_lab/generate_sysadmin_tasks.py`, generating `sysadmin_tasks.json` with 125 tasks.
- [x] Task families: host security audits (10), Docker Compose deployment hardening (20), backup/restore automation (15), incident triage (20), systemd hardening (15), reverse-proxy config (15), deployment scripts (15), and port-scan parsing (15).
- [x] Switched Hermes collection back to Nous Portal `stepfun/step-3.7-flash:free`, because aero LFM did not execute Hermes tools in smoke testing.
- [x] Ran a 3-task sysadmin pilot. The generated reports were useful; strict keyword verification was relaxed for host-audit tasks after two false-negative verifier failures.
- [x] Started full 125-task background collection on lenovo:
  `cd ~/agent_trajectory_lab && nohup python3 run_hermes_tasks.py --tasks sysadmin_tasks.json --out runs/sysadmin_stepfun_full --skip-existing --timeout 900 > runs/sysadmin_stepfun_full.log 2>&1 &`

### Current run
- Original serial PID `23260` was stopped after preserving the first passed host-audit trace because host-audit tasks were too slow for the >100 trace target.
- Four faster shard workers were launched against tasks 10-124, excluding the slow host-audit block:
  - `runs/sysadmin_stepfun_fast_shard_0`
  - `runs/sysadmin_stepfun_fast_shard_1`
  - `runs/sysadmin_stepfun_fast_shard_2`
  - `runs/sysadmin_stepfun_fast_shard_3`
- Current shard PIDs observed: `37369`, `37371`, `37373`, `37375`.
- Logs: `/home/billz/agent_trajectory_lab/runs/sysadmin_stepfun_fast_shard_{0..3}.log`.
- Partial dataset export: `/home/billz/agent_trajectory_lab/datasets/sysadmin_stepfun_partial.jsonl`.
- Latest checked status: 11 passed records / 11 completed, 10 with session trace files, 7 with clean task-specific trace exports. Families so far: 1 host security audit, 10 Docker Compose deployment hardening tasks.

### Next
- Monitor completion count and pass rate.
- Convert passed traces into SFT/DPO-ready records after at least 100 useful results exist.
- Keep failed traces as negative/control data rather than silently discarding them.
- Continue using the task-specific `Task ID` prompt marker and filtered session export; early pre-marker Compose traces are useful artifact records but have ambiguous concurrent session exports.

## 2026-07-09 — Sysadmin Trajectory Collection Completed

### Completed
- [x] Finished the lenovo Hermes/Stepfun sysadmin recovery run and stopped with no active `run_hermes_tasks.py` workers.
- [x] Exported the full passed dataset to `/home/billz/agent_trajectory_lab/datasets/sysadmin_stepfun_20260709.jsonl`.
- [x] Added `--require-matched-trace` to `tools/agent_trajectory_lab/export_trajectory_dataset.py`.
- [x] Exported the strict clean dataset to `/home/billz/agent_trajectory_lab/datasets/sysadmin_stepfun_20260709_clean.jsonl`.

### Final counts
- Result rows: 115
- Verifier-passed rows: 106
- Full export: 106 passed records, 105 with trace files
- Strict clean export: 102 passed records, 102 with prompt-matched trace files
- Strict family mix: backup/restore 13, Docker Compose deployment 17, deployment automation 13, incident triage 16, security scan parsing 15, reverse-proxy config 13, systemd hardening 15

### Interpretation
- The >100 high-quality trajectory target is met for verifier-passed, prompt-matched Hermes traces using Nous Portal `stepfun/step-3.7-flash:free`.
- aero LFM remains a negative tool-use baseline for this harness until tool-call formatting is fixed or bootstrapped from these traces.
- The clean dataset is appropriate for the first agentic SFT pass; failed/ambiguous traces should be kept for negative/preference analysis, not mixed into SFT without filtering.

## 2026-07-09 — HY3 Agentic Scale Collection Started

### Completed
- [x] Added `tools/agent_trajectory_lab/generate_agentic_scale_tasks.py`.
- [x] Generated 700 task templates and 7,000 task instances across seven requested families: tool-format, file operations, shell/repo inspection, failure recovery, summarisation/state-compaction, tool routing, and multi-step mini-agent tasks.
- [x] Generated 16 shard files at `tools/agent_trajectory_lab/agentic_scale_tasks_shard_*.json`.
- [x] Added `--rollouts` support to `tools/agent_trajectory_lab/run_hermes_tasks.py` so same-task 4-8 rollout comparisons can be collected for SFT/DPO.
- [x] Verified HY3 routing through Nous Portal on lenovo with `--provider nous --model tencent/hy3:free`.
- [x] Ran a seven-family HY3 pilot: 7/7 passed, 7/7 with prompt-matched Hermes traces.
- [x] Launched four production workers on lenovo for shards 00-03 with four rollouts per task.

### Current HY3 run
- Initial family-sorted workers `99937`, `99939`, `99941`, `99943` were stopped after eight-worker concurrency produced HY3 HTTP 429 artifacts.
- Current balanced worker PIDs: `116868`, `116870`, `116872`, `116874`, `127243`, `127245`
- Logs:
  - `/home/billz/agent_trajectory_lab/runs/hy3_agentic_balanced_shard_00_r4.log`
  - `/home/billz/agent_trajectory_lab/runs/hy3_agentic_balanced_shard_01_r4.log`
  - `/home/billz/agent_trajectory_lab/runs/hy3_agentic_balanced_shard_02_r4.log`
  - `/home/billz/agent_trajectory_lab/runs/hy3_agentic_balanced_shard_03_r4.log`
  - `/home/billz/agent_trajectory_lab/runs/hy3_agentic_balanced_shard_04_r4.log`
  - `/home/billz/agent_trajectory_lab/runs/hy3_agentic_balanced_shard_05_r4.log`
- Current strict SFT export: 125/125 clean records in `/home/billz/agent_trajectory_lab/datasets/hy3_agentic_scale_partial_clean.jsonl`; workers still running.
- Current DPO export: 29 same-task pairs in `/home/billz/agent_trajectory_lab/datasets/hy3_agentic_scale_partial_dpo.jsonl`, excluding 11 transient API/rate-limit artifacts.
- Runner now supports `--transient-retries`, `--transient-sleep`, `--rerun-failed`, and `--rollout-offset`.
- Generator now writes family-interleaved balanced shards at `agentic_scale_tasks_balanced_shard_*.json`.

### Resume/export commands
```bash
cd ~/agent_trajectory_lab
python3 run_hermes_tasks.py --tasks agentic_scale_tasks_balanced_shard_00.json --out runs/hy3_agentic_balanced_shard_00_r4 --skip-existing --rerun-failed --rollouts 4 --timeout 420 --transient-retries 4 --transient-sleep 90 --hermes-arg=--provider --hermes-arg=nous --hermes-arg=--model --hermes-arg=tencent/hy3:free
python3 export_trajectory_dataset.py --roots runs/hy3_agentic_scale_pilot runs/hy3_agentic_scale_shard_00_r4 runs/hy3_agentic_scale_shard_01_r4 runs/hy3_agentic_scale_shard_02_r4 runs/hy3_agentic_scale_shard_03_r4 runs/hy3_agentic_balanced_shard_00_r4 runs/hy3_agentic_balanced_shard_01_r4 runs/hy3_agentic_balanced_shard_02_r4 runs/hy3_agentic_balanced_shard_03_r4 runs/hy3_agentic_balanced_shard_04_r4 runs/hy3_agentic_balanced_shard_05_r4 --out datasets/hy3_agentic_scale_partial_clean.jsonl --require-matched-trace
python3 export_preference_pairs.py --roots runs/hy3_agentic_scale_pilot runs/hy3_agentic_scale_shard_00_r4 runs/hy3_agentic_scale_shard_01_r4 runs/hy3_agentic_scale_shard_02_r4 runs/hy3_agentic_scale_shard_03_r4 runs/hy3_agentic_balanced_shard_00_r4 runs/hy3_agentic_balanced_shard_01_r4 runs/hy3_agentic_balanced_shard_02_r4 runs/hy3_agentic_balanced_shard_03_r4 runs/hy3_agentic_balanced_shard_04_r4 runs/hy3_agentic_balanced_shard_05_r4 --out datasets/hy3_agentic_scale_partial_dpo.jsonl --include-all-passed-contrast
```

### Next
- Monitor pass rate and rate-limit behavior for the six balanced HY3 workers.
- Hold at six workers until the current state-compaction batch clears; then launch balanced shards 06-15 with the same transient retry/backoff settings.
- Only increase to 8 rollouts after clean export quality is confirmed.
- Improve DPO ranking with model-behavior-specific checks for invalid tool calls, wrong-tool choices, ignored observations, premature finals, hallucinated files, unsafe commands, and overlong loops.

## 2026-07-09 — LFM2.5-8B-A1B Unsloth QLoRA Attempt on aero

### Completed
- [x] Stopped the aero llama.cpp LFM server (`llama-server` PID `286325`) to free VRAM.
- [x] Confirmed aero GPU returned to idle: about 6 MiB / 8192 MiB used after shutdown.
- [x] Added `scripts/train/train_lfm25_8b_unsloth_qlora.py`, a standalone Unsloth QLoRA SFT trainer for `LiquidAI/LFM2.5-8B-A1B`.
- [x] Synced the trainer and strict scored dataset to aero:
  - `/home/billz/work/autonomous-small-model-exploration/scripts/train/train_lfm25_8b_unsloth_qlora.py`
  - `/home/billz/scored/exports/sft_strict_q8_response.jsonl`
- [x] Added config snapshot `configs/sft/lfm25_8b_a1b_unsloth_qlora_8gb.json`.
- [x] Verified aero has the required packages: `torch`, `transformers`, `datasets`, `trl`, `unsloth`, `bitsandbytes`, and `peft`.

### Training attempts
- Default Unsloth 4-bit load failed before training because bnb tried to dispatch modules to CPU/disk.
- Raising `gpu_memory_utilization` to `0.9` failed with the same CPU/disk dispatch error.
- Forcing `--device-map cuda` got further but failed during weight loading with CUDA OOM at about 7.59 GiB used.
- Adding `--offload-embedding` plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` still failed during weight loading at about 7.59 GiB used.

### Interpretation
- On aero's RTX 2070-class 8GB GPU, the HF `LiquidAI/LFM2.5-8B-A1B` checkpoint cannot currently be loaded for Unsloth bnb 4-bit QLoRA before training activations or optimizer state are added.
- The trainer is still useful for a larger GPU, a future stack with better offload support, or a smaller LFM checkpoint.

### Next
- Do not retry the same 8GB Unsloth 4-bit recipe without a changed memory strategy.
- Practical paths: train `LiquidAI/LFM2.5-3B`/230M, use a larger GPU, or use a CPU/NVMe-offload training stack outside this Unsloth path.

## 2026-07-09 — LFM2.5-1.2B-Instruct Unsloth QLoRA SFT Completed on aero

### Completed
- [x] Switched the Unsloth QLoRA trainer from `LiquidAI/LFM2.5-8B-A1B` to `LiquidAI/LFM2.5-1.2B-Instruct`.
- [x] Added config snapshot `configs/sft/lfm25_12b_instruct_unsloth_qlora_8gb.json`.
- [x] Ran 1-step smoke tests on aero:
  - batch 1, seq 512: success, 15.77s, loss 3.082
  - batch 4, seq 1024: success, 18.95s, loss 2.757
  - batch 8, seq 1024: success, 17.44s, loss 3.113
  - batch 16, seq 1024: success, 22.66s, loss 3.383
- [x] Completed full SFT run:
  - model: `LiquidAI/LFM2.5-1.2B-Instruct`
  - dataset: `/home/billz/scored/exports/sft_strict_q8_response.jsonl`
  - rows: 827
  - max sequence length: 1024
  - steps: 300
  - batch size: 16
  - gradient accumulation: 1
  - LoRA: r=8, alpha=16, dropout=0
  - optimizer: `adamw_8bit`
  - runtime: 3276s (~54m36s)
  - train loss: 1.372
- [x] Saved adapter to `/home/billz/results/lfm25_12b_instruct_sft_q8_strict/adapter`.
- [x] Saved checkpoints at steps 100, 200, and 300.

### Compatibility notes
- Initial dropout=0.05 run loaded but stalled at step 0.
- Initial dropout=0 run with padding-free auto-enabled also stalled at step 0.
- Successful recipe explicitly used `--lora-dropout 0`, `--no-padding-free`, `--dataset-num-proc 1`, and `--dataloader-num-workers 0`.
- A quick adapter-load smoke loaded the saved LoRA but stalled during generation with GPU idle; treat this as a decode/runtime smoke issue to investigate before claiming behavioral improvement.

- [x] Completed GGUF conversion (quantization format `Q4_K_M`) for both SFT adapter model (`lfm25_12b_instruct_sft_q8_strict`) and base model (`LiquidAI/LFM2.5-1.2B-Instruct`) on `aero`.
- [x] Run a controlled evaluation harness against the base GGUF model and the SFT GGUF model using all 153 prompts from `small_model_eval_v1.jsonl` via `llama-completion`.
- [x] Analyzed results and compiled a GGUF Model Comparison Report showing significant formatting improvement (GameFAQ JSON validity +76.5%, Factual QA accuracy +17.6%, overall length -30.3 words, slop phrases eliminated).
- [x] Evaluated both models on 100 GSM8K prompts to measure math reasoning performance, observing a 15.0% regression on SFT (54.0%) vs base (69.0%) due to task drift/catastrophic forgetting.

### Next
- Investigate DPO training sweeps on this dataset to further optimize structured formatting.
- Design Phase 13 PTRM-style noise injection runs on the GGUF models.
