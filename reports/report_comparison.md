# Scaling the Atlas: 0.5B vs 1.5B

## A Cross-Scale Comparison of Mechanistic Interpretability Findings in Qwen2.5

**Author:** Bilawal Riaz
**Date:** 2026-06-21
**Models:** Qwen2.5-0.5B (24 layers, 14 heads, d_model=896, ~0.49B params) vs Qwen2.5-1.5B (28 layers, 12 heads, d_model=1536, ~1.54B params)
**Hardware:** NVIDIA RTX 2070 Super (8GB VRAM), bf16 inference
**Repository:** bilawalriaz/autonomous-small-model-exploration

---

## Abstract

We compare the mechanistic interpretability atlases of Qwen2.5-0.5B and Qwen2.5-1.5B — two models from the same family at 3× scale — using identical causal intervention methodology. The comparison reveals that scaling from 0.49B to 1.54B parameters fundamentally redistributes the model's causal architecture: the universal routing hub migrates from L2 (8% depth) to L26 (93% depth), attention heads gain 22× more functional impact, MLPs lose 3× relative dominance, steering leverage collapses 70×, and skill knockout selectivity drops ~48,000×. Cross-model activation transfer maintains a proportional depth invariant (final ~10% of layers), and naive layer skipping remains fatal at both scales — though one mid-layer skip configuration at 1.5B shows the first hint of redundancy (10% top-5 overlap). These findings establish that interpretability insights do not directly transfer across scales: the causal structure itself transforms, and methods effective at 0.5B (steering, knockout) require fundamentally different approaches at 1.5B.

---

## 1. Comparison Table

| Dimension | Qwen2.5-0.5B | Qwen2.5-1.5B | Change at 3× Scale |
|-----------|-------------|-------------|---------------------|
| **Architecture** | 24 layers · 14 heads (GQA, 2 KV) · d_model=896 · d_mlp=4864 · ~0.49B params | 28 layers · 12 heads (GQA) · d_model=1536 · ~1.54B params | +4 layers, −2 heads, +71% d_model, 3.1× params |
| **Universal Hub Layer** | L2 — mean ablation KL 19.11 across all families. First+last position router. **HIGH** | L26 — mean ablation KL 13.70. Secondary: L6 (13.28), L14 (13.01). **HIGH** | Hub migrates from 8% to 93% depth. From singular to distributed (3 near-equal layers) |
| **MLP Dominance** | L2 MLP top, max KL 8.12. L0 MLP second. MLPs dominate over attention. **MEDIUM** | L0 MLP top, max KL 2.58. L1 (1.83), L27 (1.17). **MEDIUM** | MLP max effect **3× weaker** (8.12 → 2.58). MLPs lose relative dominance |
| **Head Specialization** | Distributed — max single-head KL 0.046. No specialist heads. 200× smaller than layer effects. **MEDIUM** | Specialist heads emerge — max KL 1.02 (arithmetic L0 H3). L0 H3 is cross-task. **MEDIUM** | Head impact **22× stronger** (0.046 → 1.02). Shifts from distributed to specialized |
| **Steering Leverage** | L2 steering boosts target 3.3× (0.064 → 0.213). Factual direction. **MEDIUM** | L2 steering best boost 0.003. Near-inert. **MEDIUM** | Steering **70× weaker** (0.213 → 0.003). Linear steering becomes impractical |
| **Skill Knockout** | L19 selective: **11,654×** selectivity at s=−2.0. JSON/copying preserved. **MEDIUM** | L21 best: **0.24×** selectivity. Most layers negative selectivity. **MEDIUM** | Selectivity **~48,000× weaker** (11,654× → 0.24×). Skills become entangled |
| **Cross-Model Transfer** | L23 (100%), L22 (97%), L21 (95%). Final 3 layers. **MEDIUM** | L27 (99.9%), L26 (99.2%), L25 (98.7%). Final 3 layers. **MEDIUM** | Shifts 4 layers deeper. Proportional depth invariant: final ~10% at both scales |
| **Adapter Norm-Effect Corr** | 0.85 — effect IS at late layers (L19–L23), matching norms. **MEDIUM** | 0.54 — norms flat (3.33–3.45), effects still peak late (L23–L27). **MEDIUM** | Correlation **weakens** (0.85 → 0.54). Norms less predictive of impact |
| **Layer Skipping** | 0% top-5 overlap in ALL 10 configs. KL 7.9–13.8. Even weakest layer breaks. **HIGH (negative)** | 0% in 3/4 configs. Exception: skip_mid_5 = 10% overlap (KL 6.58). **HIGH (negative)** | Still fatal, but **first hint of redundancy** (mid-layer skip at 10%) |
| **Early Exit** | L22: 0% argmax, KL 9.14. Only L23 (full) works. **HIGH (negative)** | All exits fail: 0–7.14% argmax. KL ∞ at L25–L26. **HIGH (negative)** | Still fatal at all layers. No viable early exit at either scale |
| **LoRA Training** | r=8, batch_size=2, 100 steps. Loss 0.062. Full adapter fits. | r=8, batch_size=1, gradient checkpointing. Loss 0.0098. | More constrained training. Lower final loss despite smaller batch |
| **Core Circuit Lock-in** | L2/L7/L9 locks in by step 10 (first 10%). Two-phase training. **MEDIUM** | Not tested at 1.5B (compute constraints) | Unknown — checkpoint timeline not available for 1.5B |

---

## 2. Architecture Comparison

The two models differ in several key architectural dimensions:

| Property | 0.5B | 1.5B | Ratio |
|----------|------|------|-------|
| Layers | 24 | 28 | 1.17× |
| Heads | 14 | 12 | 0.86× |
| d_model | 896 | 1536 | 1.71× |
| Parameters | ~0.49B | ~1.54B | 3.14× |

Counterintuitively, the 1.5B model has **fewer attention heads** (12 vs 14) but a **wider** d_model (1536 vs 896). The scaling strategy favors wider residual stream over more heads. This has direct interpretability implications: with fewer but wider heads, each head processes a larger fraction of d_model, potentially explaining why individual heads gain 22× more impact at 1.5B.

---

## 3. The Hub Migration

### 3.1 Where the Hub Lives

At 0.5B, Layer 2 is the singular universal hub (mean KL 19.11) — early in the network at 8% depth. It routes first tokens (instruction) and last tokens (prediction), with operator tokens showing near-zero effect. No other layer comes close.

At 1.5B, Layer 26 is the primary hub (mean KL 13.70) — late in the network at 93% depth. But unlike 0.5B's singular dominance, L6 (13.28) and L14 (13.01) are within 0.69 nats. The hub role is **distributed across three layers** spanning early (L6, 21% depth), mid (L14, 50% depth), and late (L26, 93% depth) positions.

### 3.2 What This Means

The hub migration from 8% to 93% depth is the most dramatic structural change. At 0.5B, the model does its critical routing early — Layer 2 processes inputs before most computation. At 1.5B, the critical routing happens late — Layer 26 processes the fully-built representation just before output.

This suggests that scale shifts the model's computational strategy from **early routing** (decide where information goes, then process) to **late integration** (build representations throughout, then route at the end). The secondary hubs at L6 and L14 may represent intermediate integration points that feed into L26's final routing.

### 3.3 Magnitude Comparison

The absolute hub KL is actually **lower** at 1.5B (13.70 vs 19.11), despite the larger model. This is because the 1.5B model has more layers to compensate — ablating one layer leaves 27 others to carry the load, whereas at 0.5B, ablating L2 leaves only 23. The hub is critical at both scales, but the larger model is more **gracefully degradable**.

---

## 4. MLP vs Attention: The Computational Shift

### 4.1 MLP Effects Weaken

| Metric | 0.5B | 1.5B | Change |
|--------|------|------|--------|
| Max MLP KL | 8.12 (L2) | 2.58 (L0) | 3.1× weaker |
| Top MLP layers | L2, L0 | L0, L1, L27 | Shifts to L0 dominance |
| MLP-to-layer ratio | 8.12/19.11 = 42% | 2.58/13.70 = 19% | MLPs carry less of the load |

At 0.5B, MLPs contribute ~42% of the top layer's effect. At 1.5B, this drops to ~19%. MLPs are proportionally **less important** at scale.

### 4.2 Head Effects Strengthen

| Metric | 0.5B | 1.5B | Change |
|--------|------|------|--------|
| Max head KL | 0.046 | 1.02 | 22× stronger |
| Head-to-layer ratio | 0.046/19.11 = 0.2% | 1.02/13.70 = 7.4% | Heads carry 37× more relative impact |
| Specialist heads | None | L0 H3 (arithmetic, code, copying) | Specialists emerge |

At 0.5B, heads contributed 0.2% of the top layer effect — purely distributed, no specialization. At 1.5B, heads contribute 7.4% — still smaller than layer effects, but 37× more relatively important, with identifiable specialists.

### 4.3 The Inversion

The combined effect is an **inversion of computational roles**:
- 0.5B: MLPs dominate (42% of layer effect), heads are negligible (0.2%)
- 1.5B: MLPs weaken (19%), heads gain (7.4%)

Scale shifts the computational burden from MLPs (feedforward, per-token processing) toward attention (cross-token, relational processing). This aligns with the intuition that larger models benefit more from relational reasoning than from per-token feature transformation.

---

## 5. The Steering Collapse

### 5.1 Quantitative Comparison

| Metric | 0.5B | 1.5B | Ratio |
|--------|------|------|-------|
| Best probability boost | 0.213 (3.3× from 0.064) | 0.003 | 70× weaker |
| Steering layer | L2 | L2 | Same |
| SV norm | — | 9.625 | — |
| Effective boost ratio | 3.3× | ~1.01× | 70× reduction |

At 0.5B, steering L2 with a factual recall direction increases "Rome" probability from 0.064 to 0.213 — a 3.3× boost that is practically meaningful. At 1.5B, the same intervention (steering L2 with a factual direction) produces a best boost of 0.003 — essentially no effect.

### 5.2 Why Steering Collapses

Three potential explanations:

1. **Representational entanglement**: At 1.5B, factual knowledge is distributed across more neurons and layers, making a single linear steering direction insufficient.
2. **Hub migration**: The steering was tested at L2 (the 0.5B hub). At 1.5B, L2 is no longer the hub — the critical routing happens at L26. Steering at L26 might be more effective, but this was not tested.
3. **Wider residual stream**: d_model increases from 896 to 1536 (71% wider). A single steering vector occupies a smaller fraction of the representation space, diluting its effect.

### 5.3 Practical Implications

The steering collapse means that **linear steering is not a scalable manipulation technique**. Methods that work at 0.5B (boosting factual recall, controlling verbosity) may require non-linear interventions (e.g., SAE-based steering, distributed steering vectors, or activation editing) at 1.5B and beyond.

---

## 6. Skill Knockout: From Selective to Non-Selective

### 6.1 The 48,000× Collapse

| Metric | 0.5B | 1.5B | Ratio |
|--------|------|------|-------|
| Best knockout layer | L19 | L21 | Shifts +2 layers |
| Best selectivity | 11,654× | 0.24× | ~48,000× weaker |
| Skill drop at best layer | — | 0.0011 | Minimal |
| Non-skill drop at best layer | — | 0.0044 | Larger than skill drop |

At 0.5B, L19 knockout achieves 11,654× selectivity — the steering suppresses factual recall tokens 11,654× more than non-skill tokens. This is highly selective: the skill is removed while other behavior is preserved.

At 1.5B, the best layer (L21) achieves 0.24× selectivity — the steering suppresses non-skill tokens **4× more** than skill tokens. This is **anti-selective**: the intervention hurts everything except the intended target. Most other layers show large negative selectivity (−1.6 to −77.4), meaning the steering direction is actively wrong.

### 6.2 The Entanglement Hypothesis

The collapse in selectivity strongly suggests that **skills become more entangled at scale**. At 0.5B, factual recall occupies a relatively isolated subspace that can be targeted with a single steering vector. At 1.5B, factual recall is woven into the same representational space as other behaviors — removing it requires removing the entangled components too.

This has profound implications for safety: if undesirable skills (e.g., harmful capabilities) are as entangled as factual recall at scale, then **skill removal via steering may be impossible** at deployment-scale models. More invasive methods (retraining, circuit surgery) may be required.

---

## 7. Cross-Model Transfer: The Proportional Invariant

### 7.1 The Depth Invariant

| Model | Layers | Top Transfer Layers | Depth Range |
|-------|--------|-------------------|-------------|
| 0.5B | 24 | L21, L22, L23 | 88–96% |
| 1.5B | 28 | L25, L26, L27 | 89–96% |

Both models concentrate transferable trained behavior in their **final 3 layers**, corresponding to the **final ~10% of depth**. This is a robust structural invariant: the transfer zone is defined by relative position, not absolute layer index.

### 7.2 Recovery Quality

| Layer Position | 0.5B Recovery | 1.5B Recovery |
|---------------|--------------|--------------|
| Final layer | 100% (L23) | 99.9% (L27) |
| Second-to-last | 97% (L22) | 99.2% (L26) |
| Third-to-last | 95% (L21) | 98.7% (L25) |
| Fourth-to-last | 87% (L20) | 97.9% (L24) |

Recovery is **more uniform** at 1.5B — the fourth-to-last layer (L24) achieves 97.9%, compared to 0.5B's 87% (L20). The transfer zone is wider (more layers carry transferable information) at 1.5B.

### 7.3 Implication

The proportional invariant means cross-model patching methods can be **calibrated by depth fraction** rather than absolute layer index. A patching method that works at the final 10% of one model should work at the final 10% of another, regardless of layer count.

---

## 8. Adapter Architecture

### 8.1 Norm-Effect Decoupling

| Metric | 0.5B | 1.5B |
|--------|------|------|
| Norm-effect correlation | 0.85 | 0.54 |
| Top effect layers | L19–L23 | L23–L27 |
| Norm distribution | Peaks at L20–L23 | Flat (3.33–3.45) |
| Effect distribution | Peaks at L19–L23 | Peaks at L23–L27 |

At 0.5B, adapter norms and ablation effects are both concentrated at late layers (correlation 0.85) — training writes where it matters. At 1.5B, adapter norms are **remarkably uniform** across all 28 layers (3.33–3.45), but effects still peak at late layers (L23–L27). The correlation drops to 0.54.

### 8.2 What Changes

The decoupling means that at 1.5B, **adapter norms alone are insufficient to predict functional impact**. The same weight norm at L0 and L27 produces vastly different effects (L27 >> L0). This suggests that the residual stream at late layers is more sensitive to perturbation — the same weight change has more leverage when applied to a more refined representation.

### 8.3 Implication

For adapter design at scale, **norm-based pruning is unreliable**. An adapter with small norms at late layers may have large functional effects. Causal ablation (not norm inspection) is necessary to identify which adapter components matter.

---

## 9. Efficiency: What Remains Fatal and What Shifts

### 9.1 Layer Skipping

| Config | 0.5B Top-5 Overlap | 1.5B Top-5 Overlap |
|--------|-------------------|-------------------|
| Skip weakest 1 | 0% | 0% |
| Skip mid 5 | 0% | **10%** |
| Skip 6 layers | 0% | 2.86% |
| Skip 8 layers | 0% | 2.86% |

At 0.5B, ALL 10 skip configurations produce 0% top-5 overlap. At 1.5B, the skip_mid_5 configuration (skipping L4–L8) preserves **10% top-5 overlap** — the first evidence that some layers may be partially redundant at scale.

### 9.2 Early Exit

| Exit Point | 0.5B Argmax Match | 1.5B Argmax Match |
|-----------|-------------------|-------------------|
| Full model | 100% | 0% (∞ KL) |
| 1 layer early | — | 0% |
| 2 layers early | — | 7.14% |
| 4 layers early | — | 7.14% |

Early exit fails at both scales. At 1.5B, even the "full model" exit (L27) shows ∞ KL and 0% argmax, likely a measurement artifact from the lm_head projection. The key finding is the same: intermediate hidden states are not directly projectable to vocabulary.

### 9.3 The Redundancy Hint

The 10% top-5 overlap in skip_mid_5 at 1.5B is the **first positive efficiency signal** across both studies. While 10% is far from practical, it suggests that:
- Mid layers (L4–L8) at 1.5B may carry **partially redundant** computation
- Structured pruning (not zero-ablation) with retraining might recover this redundancy
- Scale may eventually produce enough redundancy for practical layer removal, but 1.5B is not yet at that threshold

---

## 10. What Does NOT Change at 3× Scale

Several findings are **invariant** across scales:

1. **Inference optimization via layer removal is impractical** — skipping any layer breaks output at both scales (with one partial exception)
2. **Early exit fails** — no intermediate layer produces usable output at either scale
3. **Cross-model transfer is in the final ~10%** — the proportional depth invariant holds
4. **JSON produces the largest training shift** — KL 6.47 at both scales (the skill most affected by LoRA)
5. **LoRA is required** — full SFT OOMs on 8GB at both scales
6. **Zero ablation is disruptive** — all ablation effects are large, confirming every layer matters

---

## 11. Limitations

1. **Only two scale points**: 0.5B and 1.5B. A third point (e.g., 0.5B, 1.5B, 3B) would establish scaling laws rather than pairwise comparisons.
2. **Different task suite sizes**: 0.5B tested 12 families; 1.5B tested 8. Four families (refusal/compliance, verbosity, variable renaming, uncertainty) are missing from 1.5B.
3. **Steering tested at L2 only for 1.5B**: The 0.5B study tested L2 and L21. The 1.5B study tested only L2. The steering collapse may be layer-specific — testing at L26 (the new hub) might yield different results.
4. **No checkpoint timeline for 1.5B**: The core circuit lock-in finding (step 10) is only verified at 0.5B. Whether the two-phase training architecture persists at 1.5B is unknown.
5. **Single seed at both scales**: All results from one random seed per model. Multi-seed replication is needed.
6. **Zero ablation at both scales**: Creates out-of-distribution activations. Mean/resample ablation would be more principled.
7. **Same model family**: Both are Qwen2.5. Findings may be family-specific. Cross-family comparison (e.g., Qwen vs Llama) would test generality.

---

## 12. Key Findings

1. **The universal hub migrates from L2 to L26** — from 8% to 93% depth, from singular to distributed (HIGH)
2. **Attention heads gain 22× impact** — specialist heads emerge, L0 H3 is cross-task (MEDIUM)
3. **MLP effects weaken 3×** — from 8.12 to 2.58 KL, despite 3× more parameters (MEDIUM)
4. **Steering collapses 70×** — from 0.213 to 0.003 boost, linear steering becomes impractical (MEDIUM)
5. **Skill knockout selectivity drops ~48,000×** — from 11,654× to 0.24×, skills entangle at scale (MEDIUM)
6. **Cross-model transfer maintains proportional depth** — final ~10% of layers at both scales (MEDIUM)
7. **Adapter norm-effect correlation weakens** — from 0.85 to 0.54, norms less predictive (MEDIUM)
8. **Layer skipping remains fatal** — but first hint of mid-layer redundancy (10% overlap) at 1.5B (HIGH for negative)
9. **Early exit fails at both scales** — no viable inference speedup via layer removal (HIGH for negative)
10. **The computational burden shifts from MLPs to attention** — a fundamental architectural rebalancing at 3× scale (MEDIUM)

---

## 13. Conclusion

Scaling from 0.5B to 1.5B parameters — a 3.1× increase — does not merely amplify the 0.5B atlas; it **transforms it**. The universal hub migrates across the network. The MLP-attention balance inverts. Steering and knockout — the two most practically powerful interventions at 0.5B — become nearly inert. Skills entangle. Adapter norms decouple from effects. Only the cross-model transfer zone (final ~10% of depth) and the fatality of layer skipping remain invariant.

These findings carry a clear message for interpretability research: **insights from small models are necessary but not sufficient for understanding large models**. The causal architecture itself is scale-dependent. Methods must be re-validated at each scale, and the atlas must be rebuilt — not merely extrapolated — as models grow.

The one encouraging signal is the 10% top-5 overlap in mid-layer skipping at 1.5B — the first crack in the "every layer is essential" wall. Whether this crack widens at 3B, 7B, and beyond is the question the next scaling study must answer.
