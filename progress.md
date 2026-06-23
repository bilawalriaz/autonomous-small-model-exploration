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
