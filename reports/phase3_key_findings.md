# Phase 3 Key Findings (Live Analysis)

Generated: 2026-06-23
Status: Experiment runner active on aero

---

## CONFIRMED: Hub location is stable across prompt types

**P1 Natural Language Hubs (2500 prompts, 50+ per family):**
- NL hub: L2 (same as synthetic!)
- Hub shift: 0 layers
- Family agreement: 100%
- All 12 families agree on L2 as the hub with NL prompts

**Implication:** The synthetic prompt methodology is VALIDATED. Hub identification using short synthetic prompts gives the same result as using 50+ natural language prompts per family.

---

## CONFIRMED: Hub is architecture-specific, not just depth-dependent

**P3 Coder Atlas:**
- Base 0.5B: hub at L2 (8% depth)
- Coder-0.5B: hub at L22 (92% depth)
- Same architecture family (Qwen2.5), same scale (0.5B)
- Hub COMPLETELY FLIPS from early to late

**Implication:** Code-specific training fundamentally restructures where processing happens. The coder model pushes the hub from the first 10% to the last 10% of layers. This is a stronger effect than scaling from 0.5B to 1.5B.

---

## REVISED: Hub migration pattern (3 scales)

| Model | Hub | Depth | Status |
|-------|-----|-------|--------|
| Qwen2.5-0.5B | L2 | 8% | REPLICATED (3 seeds, std=0.0) |
| Qwen2.5-1.5B | L14 | 50% | REPLICATED (3 seeds, std=0.0) |
| Qwen2.5-3B | L34 | 94% | REPLICATED (3 seeds, std=0.0) |
| Qwen2.5-Coder-0.5B | L22 | 92% | Single seed |
| SmolLM2-1.7B | L0 | 0% | Single seed |

---

## NEW FINDING: Quantization amplifies steering sensitivity

**Q1 Quantization Atlas (4-bit NF4, 0.5B):**
- Steering at L2, s=-4.0: KL=10.0 (4-bit) vs KL=0.021 (bf16)
- 476x amplification of steering effect under 4-bit quantization
- VRAM: 436MB (vs ~1000MB bf16)

**Implication:** Quantized models are DRAMATICALLY more sensitive to steering interventions. This could be because:
1. Quantization noise makes the model more susceptible to perturbation
2. The reduced precision amplifies the effect of activation changes
3. The baseline distribution is already shifted, so steering has more leverage

This is a potential GEM: quantized models may be better steering targets.

---

## NEW FINDING: Position-specific ablation insights

**C2 Position Ablation (24 layers × 5 position types × 12 families):**
- Answer tokens dominate at ALL 24 layers (mean effect 8.65)
- BOS tokens second (1.57), especially at L8 (4.34)
- Content tokens at L2 (1.94) — confirms hub role in content routing
- Instruction tokens at L6 (1.45) — early instruction processing
- Delimiters near-zero everywhere (0.15)

**Implication:** Ablation-based hub identification is heavily biased toward layers that affect final answer tokens. The "universal hub" may partly be the "most important layer for output generation" rather than the "most important layer for reasoning."

---

## NEW FINDING: Module sweep confirms o_proj efficiency

**L5 Module Sweep (8 module configs × 100 steps):**
- o_proj: loss=0.33 (best individual module)
- v_proj: loss=0.34
- q_proj: loss=0.34
- k_proj: loss=0.35
- up_proj: loss=0.38
- down_proj: loss=0.39
- gate_proj: loss=0.45
- all_linear: loss=0.31 (best overall, 4.4M params)

**Implication:** o_proj is confirmed as the most efficient individual module for skill injection. It achieves 0.33 loss with only 344K params (10x fewer than all_linear for only 6% worse loss).

---

## NEW FINDING: Ablation method comparison reveals issues

**C1 Ablation Methods (zero/mean/resample, 24 layers × 12 families):**
- Zero ablation hub: L2 (19.48 KL)
- Mean ablation: ALL ZEROS (mean activations ≈ 0 in residual stream)
- Resample ablation hub: L23 (14.06 KL) — COMPLETELY DIFFERENT
- Zero vs mean correlation: -0.21 (negative, mean gives nothing)
- Zero vs resample correlation: 0.14 (weak)

**Implication:** 
1. Mean ablation is useless for these models (residual stream activations have near-zero mean)
2. Zero ablation and resample ablation identify DIFFERENT important layers
3. The hub identification is method-dependent — this is a methodological warning

---

## Rank sweep: loss vs accuracy disconnect

**L4 Rank Sweep (r=2,4,8,16, 100 steps each):**
- Loss decreases with rank: r=2(0.76) → r=4(0.59) → r=8(0.46) → r=16(0.38)
- Exact match: 0.0 for ALL ranks
- Valid JSON: 0.0 for ALL ranks

**Implication:** The base model evaluation methodology needs improvement. LoRA training successfully reduces loss but doesn't produce measurable exact-match improvement on these eval prompts. Either:
1. The eval prompts are too hard for 100-step LoRA
2. The base model doesn't naturally produce exact-match responses
3. The scoring needs to be more lenient (partial credit)

---

## Status of remaining experiments

### Completed successfully (9/18):
- R1-R5: Multi-seed replication ✓
- L4-L5: Rank/module sweep ✓
- C1-C3: Ablation methods, position, module ✓
- P1-P3: NL hubs, steering by length, coder atlas ✓
- Q1: Quantization steering ✓

### Re-running (5 blocks, ~3 hours):
- L1-L3: Atlas-guided LoRA (THE key experiment)
- G3: Checkpoint lock-in
- G4: Atlas-guided skip

### Needs TransformerLens (2 blocks):
- C4: Steering controls
- G2: Knockout controls

### Failed with different issues (2 blocks):
- G1: Steering direction transfer (memory issues)
- Q2: Steering on 4-bit (missing ablation data)
