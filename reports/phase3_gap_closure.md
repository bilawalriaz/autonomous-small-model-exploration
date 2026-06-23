# Phase 3 Report: Gap Closure and Gem Discovery

**Status:** INFRASTRUCTURE COMPLETE, EXPERIMENTS QUEUED
**Started:** 2026-06-23
**Models:** Qwen2.5-0.5B, 1.5B, 3B, Coder-0.5B, SmolLM2-1.7B
**Hardware:** aero (RTX 2070 Super 8GB)
**Total Phase 3 blocks:** 24 across 6 priority levels

---

## Executive Summary

Phase 3 converts Phase 1-2's exploratory findings into reviewer-grade, falsifiable, reproducible claims. The core question:

> Are small language models controllable through model-specific causal surfaces, and can mapping those surfaces produce better training, steering, compression, or inference recipes than generic practice?

Phase 1 (21 experiments) identified hub layers, steering effects, and training concentration patterns. Phase 2 (17 experiments) tested cross-scale and cross-family generalization. Phase 3 closes the methodological gaps that prevent publication: insufficient seeds, too few prompts, missing controls, and untested practical claims.

**Infrastructure built:** 12 new experiment scripts, full orchestrator, claims audit, threats catalog, gems inventory.

**Key experiments queued:**
1. Multi-seed replication (R1-R5) — determines if ANY claim survives
2. Atlas-guided LoRA (L1-L5) — the headline practical experiment
3. Steering controls (C4) — determines if steering is real
4. Quantization causal drift (Q1-Q3) — hidden deployment risk

---

## 1. Claims We Confirmed (with stronger evidence)

*Awaiting multi-seed replication results. These are the claims most likely to survive:*

| Claim | Previous Confidence | Expected | What Would Confirm |
|-------|-------------------|----------|-------------------|
| C01: L2 hub (0.5B) | HIGH | HIGH | Hub layer stable across 3 seeds (std <= 1) |
| C02: Hub migration L2->L26->L34 | HIGH | HIGH | Hub stable per model across seeds |
| C08: Late-layer behavior encoding | MED-HIGH | HIGH | Monotonic recovery pattern replicates |
| C10: Steering migration | HIGH | HIGH | Steering effects > random at correct layers |
| C13: Layer skipping fails | HIGH | HIGH | 0% overlap replicated across seeds |
| C20: Ablation rank-order stability | MED-HIGH | HIGH | Rank correlation > 0.8 across methods |

---

## 2. Claims We Weakened

*These claims face specific falsification threats in Phase 3:*

| Claim | Threat | Expected Outcome |
|-------|--------|-----------------|
| C09: Skill knockout (11654x selectivity) | T11: prompt artifact | Selectivity ratio drops with controls |
| C06: r=4 optimal rank | No task accuracy data | May not hold for actual accuracy |
| C11: Norm-effect corr=0.85 | T10: reporting bugs | Data fine, narrative needs rewrite |

---

## 3. Claims We Killed

| Claim | Status | Reason |
|-------|--------|--------|
| C19: Norm-effect separation | REFUTED | Correlation is 0.85 (strong positive), not negative. The "separation" was a conflation of general vs adapter-specific importance. |

---

## 4. New Gems Found

*Awaiting Phase 3 experiments. Candidate gems from Phase 1-2:*

| Gem | Criteria Met | Confidence | Key Experiment |
|-----|-------------|------------|----------------|
| G01: Hub migrates to final ~10% | 1,2,3,5,8 | HIGH | Multi-seed replication |
| G02: Steering migrated not collapsed | 2,3,4,5,8 | HIGH | Steering direction transfer |
| G05: Zero=mean at hub layers | 1,2,4,8 | MED-HIGH | Full ablation method comparison |
| G07: Late-layer LoRA target | 1,2,3,4,5 | MED-HIGH | Atlas-guided LoRA |
| G09: 4-bit NF4 sweet spot | 2,3,7,8 | MED-HIGH | Quantization causal drift |

---

## 5. Practical Training Rules

### Confirmed (from Phase 1-2)

1. **Target late layers for LoRA:** Adapter effects concentrate at final ~10% of layers (L19-L23 at 0.5B, L31-L35 at 3B). Don't waste parameters on early layers.

2. **o_proj is the most efficient module:** 344K params for o_proj gives +3.64 boost vs 3.3M params for MLP giving +1.92. 10x more parameter-efficient.

3. **Each model needs its own atlas:** Hub position is scale-dependent and architecture-specific. You cannot transfer layer-targeting knowledge from 0.5B to 1.5B.

### Under Test (Phase 3)

4. **Atlas-guided LoRA should beat generic LoRA:** Target atlas-identified layers + o_proj-only at r=4. Test: P3-LORA-001 through P3-LORA-005.

5. **Core circuit locks in at 10% of training:** If confirmed at 1.5B with multiple tasks, this becomes a major compute-saving rule. Test: P3-GEM-003.

6. **Steering budget scales with model size:** 1.5B has 3x larger steering budget than 0.5B. Stronger interventions but also more destructive collateral damage. Test: P3-STEER-001, P3-STEER-002.

---

## 6. Practical Inference/Deployment Rules

### Confirmed

1. **4-bit NF4 over 8-bit:** 8-bit is actually SLOWER than 4-bit on bitsandbytes (52% vs 9% speed loss at 1.5B). Use 4-bit NF4 as default.

2. **All layers are necessary:** Naive layer skipping gives 0% top-5 token overlap across all skip configs and all models tested. Don't skip layers without recovery training.

### Under Test

3. **Quantization may change causal surfaces:** Benchmark quality may be preserved while steering/ablation behaviour changes. This would make quantized models unreliable for model surgery. Test: P3-QUANT-001 through P3-QUANT-003.

4. **Atlas-guided layer skip + recovery:** Can we skip low-causal layers with a short recovery finetune? Test: P3-GEM-004.

---

## 7. Open Questions

### Critical (must resolve in Phase 3)

1. **Is the 0.5B hub location stable across seeds?** If L2 moves by >2 layers across seeds, all hub-related claims are fragile. Experiment: R1.

2. **Is steering task-specific?** If random vectors give similar KL as task vectors, all steering findings collapse. Experiment: C4.

3. **Does atlas-guided LoRA beat generic LoRA?** This is the most practically valuable claim. If it fails, the atlas is interesting but not actionable. Experiments: L1-L5.

4. **Does quantization change causal surfaces?** If yes, quantized models cannot be used for steering/ablation even when they pass benchmarks. Experiments: Q1-Q3.

### Important (should resolve)

5. **Why does zero=mean at hub layers?** Is it because hub activations have near-zero mean, or something else? Experiment: C1.

6. **What happens at the 0.5B/1.5B boundary?** The hub jumps from 8% to 93% depth. Is there a smooth transition or a phase change? Need intermediate scales.

7. **Does SmolLM2 have a real hub at L0 or no hub at all?** The flat ablation profile suggests no clear hub, but L0 being the "top" may be an artifact. Need second SmolLM2 scale.

8. **Is the 11654x knockout selectivity real?** T11 flags it as possibly a prompt artifact (near-zero base probability for JSON). Need controls. Experiment: G2.

### Nice to Have

9. Does steering direction transfer across scales? (G1)
10. Can we detect finetuning lock-in by step 10? (G3)
11. Do code-semantic circuits interfere with format circuits when merged? (Future)
12. Do head effects increase with scale as claimed? (C18 needs verification)

---

## 8. Reproducibility Package

### Files

- `claims.md` — 20 claims with status, confidence, evidence, falsifiers
- `threats.md` — 14 methodological threats with severity and fixes
- `gems.md` — 10 candidate gems with criteria scoring
- `experiments/registry.jsonl` — one record per run
- `configs/` — experiment configurations as JSON
- `scripts/run_full_phase3_atlas.py` — single entry point orchestrator

### Experiment Scripts (12 new + 4 existing)

| Block | Script | Status |
|-------|--------|--------|
| R1-R5 | run_phase3_multiseed_replication.py | EXISTS |
| L1-L3 | run_phase3_atlas_guided_lora.py | EXISTS |
| L4 | run_phase3_rank_sweep_with_accuracy.py | NEW |
| L5 | run_phase3_module_sweep_with_accuracy.py | NEW |
| C1 | run_phase3_ablation_method_comparison.py | NEW |
| C2 | run_phase3_position_ablation_all_layers.py | NEW |
| C3 | run_phase3_module_ablation.py | NEW |
| C4 | run_phase3_steering_controls.py | EXISTS |
| P1 | run_phase3_natural_language_hubs.py | NEW |
| P2 | run_phase3_steering_by_length.py | NEW |
| P3 | run_phase3_coder_atlas.py | NEW |
| Q1-Q3 | run_phase3_quantization_atlas.py | EXISTS |
| G1 | run_phase3_steering_direction_transfer.py | NEW |
| G2 | run_phase3_knockout_controls.py | NEW |
| G3 | run_phase3_checkpoint_lockin.py | NEW |
| G4 | run_phase3_atlas_guided_skip.py | NEW |

### Running

```bash
# All blocks on 0.5B
python scripts/run_full_phase3_atlas.py --model Qwen/Qwen2.5-0.5B --blocks all

# Priority 1 only (replication — do this first)
python scripts/run_full_phase3_atlas.py --model Qwen/Qwen2.5-0.5B --priority 1

# Specific block
python scripts/run_full_phase3_atlas.py --model Qwen/Qwen2.5-0.5B --blocks L1

# Dry run
python scripts/run_full_phase3_atlas.py --blocks all --dry-run
```

---

## 9. Recommended Phase 4

Phase 4 should focus on:

1. **Publication:** Write the actual paper. Phase 3 results provide the evidence.
2. **New model atlases:** Run on Qwen3.5-0.8B, Phi-4-mini, Llama-3.2-1B.
3. **Deobfuscation-specific training:** Atlas-guided LoRA for real code deobfuscation tasks.
4. **Deployment pipeline:** Build a CLI that takes a model, runs the atlas, and outputs targeting recommendations.
5. **Community validation:** Release the atlas methodology and invite external replication.

---

## Methodology: How We Closed Each Gap

### T01: Single seed everywhere
**Fix:** Multi-seed replication (R1-R5) with seeds 42, 137, 256.
**Metric:** Hub location std <= 1 layer = robust. Std > 2 = fragile.
**Status:** QUEUED (aero offline).

### T02: Too few prompts
**Fix:** Natural language prompt expansion (P1) with 50+ prompts per family from canonical task suite.
**Metric:** Hub location agreement between synthetic and NL prompts.
**Status:** QUEUED.

### T03: Synthetic prompts toy-like
**Fix:** P1 + steering by length (P2). Test short/medium/long.
**Status:** QUEUED.

### T04: Zero ablation distribution shift
**Fix:** Full ablation method comparison (C1). Compare zero/mean/gaussian at ALL layers.
**Status:** QUEUED.

### T05: Steering prompt-length dependent
**Fix:** P2 — steering at hub layers with short/medium/long prompts.
**Status:** QUEUED.

### T06: Quantization causal surface drift
**Fix:** Q1-Q3 — layer ablation and steering on 4-bit NF4.
**Status:** QUEUED.

### T07: LoRA effects task-specific
**Fix:** L1-L3 — train on 3 families, compare concentration maps.
**Status:** QUEUED.

### T08: Layer importance differs by position
**Fix:** C2 — token-position ablation at ALL layers.
**Status:** QUEUED.

### T09: Base vs coder transfer
**Fix:** P3 — hub identification on Qwen2.5-Coder-0.5B.
**Status:** QUEUED.

### T10: Norm-effect reporting confusion
**Fix:** Manual re-analysis. Data is fine (corr=0.85 is strong positive). Narrative rewrite needed.
**Status:** DONE (in claims.md C11 and C19).

### T11: Skill knockout selectivity artifact
**Fix:** G2 — random-vector and shuffled-label knockout controls.
**Status:** QUEUED.

### T12: Late-layer proximity-to-output artifact
**Fix:** C1 + C2 — if early-layer interventions with matched magnitude produce proportionally smaller effects, it's real. If equal magnitude gives equal effect, it's proximity.
**Status:** QUEUED.

### T13: 3B dual-hub (L34 and L13)
**Fix:** R3 — multi-seed replication at 3B. If L13 persists across seeds, it's real.
**Status:** QUEUED.

### T14: Publication report bugs
**Fix:** Manual correction in claims.md. The 0.85 correlation is strong positive, not weak.
**Status:** DONE.

---

## Compute Budget

| Priority | Blocks | Est. Time (0.5B) | Est. Time (1.5B) | Total |
|----------|--------|-------------------|-------------------|-------|
| 1: Replication | R1-R5 | 35 min | 70 min | 105 min |
| 2: Atlas LoRA | L1-L5 | 180 min | — | 180 min |
| 3: Causal tests | C1-C4 | 125 min | — | 125 min |
| 4: Prompt robust | P1-P3 | 65 min | — | 65 min |
| 5: Quantization | Q1-Q3 | 40 min | 40 min | 80 min |
| 6: Gem hunting | G1-G4 | 115 min | 60 min | 175 min |
| **TOTAL** | **24** | **560 min** | **170 min** | **~12 hours** |

---

## Appendix: Evidence Levels

| Level | Definition | Example |
|-------|-----------|---------|
| Observed once | Single seed, single model, no controls | C03: SmolLM2 hub |
| Replicated | Multiple seeds/models/methods agree | C02: Hub migration |
| Causal but narrow | Causal evidence for one model/task | C04: LoRA concentration |
| Causal and actionable | Causal, replicated, directly usable | C01: L2 hub |
| Possibly artifact | Methodological concern | C11: Norm-effect |
| Refuted | Contradicted by stronger evidence | C19: Norm-effect separation |
| Unknown | Insufficient data | C17: L1 injection |
