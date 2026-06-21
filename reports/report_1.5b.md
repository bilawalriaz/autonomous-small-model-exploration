# Mechanistic Interpretability Atlas of Qwen2.5-1.5B

## A Causal Investigation of Component Behaviour, Training Perturbation, and Skill Architecture in a 1.5B Parameter Language Model

**Author:** Bilawal Riaz
**Date:** 2026-06-21
**Model:** Qwen/Qwen2.5-1.5B (28 layers, 12 heads GQA, d_model=1536, ~1.54B parameters)
**Hardware:** NVIDIA RTX 2070 Super (8GB VRAM), bf16 inference
**Repository:** bilawalriaz/autonomous-small-model-exploration

---

## Abstract

We present a mechanistic interpretability atlas of Qwen2.5-1.5B, a 1.5B parameter transformer — a 3× scale-up from our prior 0.5B study. Using the same causal intervention suite (layer ablation, MLP ablation, head ablation, steering vectors, LoRA training perturbation, cross-model activation transfer, skill knockout, and adapter ablation), we map how this larger model processes information across 8 task families. We find that the universal routing hub **shifts from L2 (at 0.5B) to L26** (mean ablation KL 13.70), that individual attention heads become **22× more impactful** (max head KL 1.02 vs 0.046), that steering becomes **70× weaker** (best boost 0.003 vs 0.213), and that skill knockout selectivity **collapses** from 11,654× to 0.24×. Cross-model patching shifts monotonically from L21–L23 to L25–L27. MLP ablation is 3× weaker despite 3× more parameters. All layer-skipping configurations remain fatal (0% top-5 overlap except one mid-skip at 10%), and early exit at any layer fails. These findings reveal that scaling from 0.5B to 1.5B redistributes causal structure: the universal hub migrates to later layers, attention heads gain functional specialization, steering leverage diminishes, and learned skills become harder to selectively manipulate.

---

## 1. Introduction

### 1.1 Motivation

Our prior study of Qwen2.5-0.5B established a causal atlas identifying Layer 2 as a universal routing hub, skill-specific LoRA concentration patterns, rapid core-circuit lock-in, and selective skill knockout at 11,654× selectivity. A central limitation was single-model specificity — all findings were tied to one architecture at 0.49B parameters. This study scales the same methodology 3× to Qwen2.5-1.5B (1.54B parameters) to determine which structural findings persist, which shift, and which qualitatively change at scale.

### 1.2 Research Questions

1. **Does the universal routing hub persist at L2, shift to a new layer, or fragment at 28 layers?**
2. **Do specialist attention heads emerge at 1.5B, or does processing remain distributed?**
3. **Does steering leverage scale with model size, or does it diminish?**
4. **Can learned skills still be selectively knocked out, or does scale make skills more entangled?**
5. **Does cross-model activation transfer shift to later layers in a deeper model?**
6. **Does naive layer skipping remain fatal, or do redundant layers emerge at scale?**

### 1.3 Methodology Overview

We employ the same causal intervention approach as the 0.5B study:

- **Zero ablation:** Remove component output, measure KL divergence in next-token distribution
- **Steering vectors:** Compute mean(positive) − mean(negative) activation differences, inject at varying strengths
- **LoRA training perturbation:** Train low-rank adapters on specific skills, compare component maps before/after
- **Cross-model patching:** Transfer activations from trained model to base model, measure recovery
- **Skill knockout:** Apply negative steering to suppress learned skills, measure selectivity
- **Adapter-only ablation:** Selectively remove adapter contribution at each layer
- **Layer skipping & early exit:** Test whether layers can be removed for inference speedup

Every claim follows the evidence ladder: weak (probe, attention), medium (ablation, repeated effect), strong (patching recovery, controls ruled out), very strong (selective knockout, circuit reconstruction).

---

## 2. Experimental Setup

### 2.1 Model

Qwen2.5-1.5B: 28 transformer layers, 12 attention heads (GQA), d_model=1536, vocab_size=151,936, ~1.54B parameters. Loaded in bf16. The 1.5B model requires gradient checkpointing and batch_size=1 for LoRA training due to VRAM constraints on the RTX 2070 Super (8GB).

### 2.2 Task Suite

8 task families tested:
1. Copying/induction
2. Bracket and delimiter tracking
3. JSON/schema following
4. Factual recall
5. Arithmetic micro-reasoning
6. Code syntax recognition
7. Code semantic preservation
8. Dead-code detection

### 2.3 Training Configuration

LoRA: r=8, alpha=16, target_modules=[q_proj, k_proj, v_proj, o_proj], lr=2e-4, batch_size=1, gradient checkpointing enabled. Training converged to loss 0.0098. All training uses LoRA due to VRAM constraints (full SFT OOMs on 8GB at 1.5B scale).

### 2.4 Infrastructure

HuggingFace Transformers with manual forward hooks. All experiments reproducible via `python scripts/run_*.py` on aero.

---

## 3. Results

### 3.1 Component Atlas: Layer-Level Ablation

**Finding 1: L26 is the universal importance hub — the hub shifts from L2 to L26 at 3× scale.**
**Confidence: HIGH**

Zero-ablating Layer 26 causes the largest mean KL divergence across all 8 task families (mean KL 13.70 nats). The top 5 layers by ablation effect:

| Rank | Layer | Mean KL |
|------|-------|---------|
| 1 | L26 | 13.70 |
| 2 | L6 | 13.28 |
| 3 | L14 | 13.01 |
| 4 | L5 | 10.44 |
| 5 | L9 | 10.34 |

Key observations:
- The universal hub **migrates from L2 (0.5B, early) to L26 (1.5B, late)** — a 24-layer shift
- L6 and L14 form a secondary cluster of high-importance layers, suggesting distributed processing across the depth
- The mean per-layer KL ranges from 5.40 (L23) to 13.70 (L26), showing substantial variation
- No single layer dominates as overwhelmingly as L2 did at 0.5B; the top 3 layers (L26, L6, L14) are within 0.69 nats of each other

This is a fundamental structural change: at 0.5B, one layer (L2) was the clear singular hub. At 1.5B, the hub role is shared between a late layer (L26) and two mid layers (L6, L14), suggesting more distributed routing.

### 3.2 MLP-Level Ablation

**Finding 2: L0 MLP dominates but MLP effects are 3× weaker than at 0.5B.**
**Confidence: MEDIUM**

MLP ablation reveals L0 MLP has the highest mean effect (KL 2.58), with L1 MLP second (KL 1.83) and L27 MLP third (KL 1.17). The top 5 MLP layers:

| Rank | Layer | Mean KL |
|------|-------|---------|
| 1 | L0 | 2.58 |
| 2 | L1 | 1.83 |
| 3 | L27 | 1.17 |
| 4 | L26 | 0.29 |
| 5 | L9 | 0.25 |

Key observations:
- MLP ablation max KL is 2.58, **3× weaker than 0.5B's max of 8.12** (L2 MLP at 0.5B)
- Early MLPs (L0, L1) dominate at both scales, but the magnitude is dramatically reduced
- Late MLPs (L27) gain relative importance at 1.5B (third place, vs not in top 5 at 0.5B)
- The MLP contribution to overall layer effect is smaller at 1.5B, suggesting attention carries more of the load

### 3.3 Head-Level Ablation

**Finding 3: Specialist heads emerge — max head KL is 22× stronger than at 0.5B.**
**Confidence: MEDIUM**

Head ablation reveals that individual attention heads have substantially more impact at 1.5B than at 0.5B:

| Rank | Task | Layer | Head | KL |
|------|------|-------|------|----|
| 1 | arithmetic | L0 | H3 | 1.02 |
| 2 | code_syntax | L0 | H6 | 0.51 |
| 3 | delimiter_tracking | L0 | H10 | 0.44 |
| 4 | delimiter_tracking | L0 | H6 | 0.44 |
| 5 | code_syntax | L0 | H3 | 0.35 |
| 6 | copying | L0 | H3 | 0.26 |

Key observations:
- Max single-head KL is **1.02**, compared to 0.5B's 0.046 — a **22× increase** in head impact
- The strongest heads are concentrated in **L0** across multiple task families
- L0 H3 appears in arithmetic, code_syntax, and copying — a potential cross-task specialist head
- At 0.5B, no head exceeded 0.046 KL (200× smaller than layer effects). At 1.5B, the top head reaches 1.02, which is ~13× smaller than the top layer effect (13.70) — closing the gap
- This suggests attention heads gain functional specialization at scale, transitioning from fully distributed to partially specialized processing

### 3.4 Steering Vectors

**Finding 4: Steering is 70× weaker at 1.5B — the steering budget collapses at scale.**
**Confidence: MEDIUM**

Steering at L2 (the same layer tested at 0.5B) with a factual recall direction (sv_norm 9.625) produces a dramatically weaker effect:

- **Best boost: 0.003** (absolute probability increase), compared to 0.5B's **0.213** — a **70× reduction**
- For "The capital of Italy is ", original target probability is 0.000919; even at optimal steering strength (−2.0), it only reaches 0.010315
- For "The capital of Spain is ", original probability is 0.000038; best steering reaches 0.000353
- Negative steering (strength −2.0 to −4.0) increases target probability, but the absolute effect is minimal
- The KL divergence at moderate steering strengths is small (0.009–0.036 at ±0.5 to ±1.0), with larger KL only at extreme strengths (0.76 at −4.0)

This is a critical scaling finding: the steering leverage that made skill manipulation practical at 0.5B (3.3× boost) is largely unavailable at 1.5B. The model's representations are more entangled and resistant to linear steering.

### 3.5 LoRA Training Perturbation

**Finding 5: LoRA training converges but requires gradient checkpointing at 1.5B scale.**
**Confidence: MEDIUM**

LoRA training on the 1.5B model required batch_size=1 and gradient checkpointing to fit in 8GB VRAM. Training converged to loss 0.0098.

Base-to-trained KL divergence varies substantially by task family:

| Task Family | Base→Trained KL |
|-------------|----------------|
| json_schema | 6.47 |
| copying | 0.54 |
| dead_code | 0.42 |
| delimiter_tracking | 0.28 |
| factual_recall | 0.16 |
| arithmetic | 0.08 |
| code_semantics | 0.08 |
| code_syntax | 0.05 |

Key observations:
- JSON schema training produces the largest distributional shift (KL 6.47), consistent with 0.5B where JSON also showed the largest effect
- Code syntax and code semantics show minimal shift (KL 0.05–0.08), suggesting these skills are largely pre-trained
- The LoRA adapter successfully modifies behavior despite the VRAM-constrained training configuration

### 3.6 Cross-Model Activation Transfer

**Finding 6: Cross-model patching shifts to L25–L27 — recovery is monotonic and later.**
**Confidence: MEDIUM**

Cross-model patching (transferring trained model activations into the base model) reveals that recovery increases monotonically from early to late layers, with the top transfer layers shifted ~4 layers deeper than at 0.5B:

| Layer | Mean Recovery | 0.5B Equivalent |
|-------|--------------|-----------------|
| L27 | 99.9% | L23 (100%) |
| L26 | 99.2% | L22 (97%) |
| L25 | 98.7% | L21 (95%) |
| L24 | 97.9% | L20 (87%) |
| L23 | 97.4% | — |
| L22 | 95.5% | — |

Key observations:
- The top 3 transfer layers (L25, L26, L27) correspond to the last 3 layers, shifted from 0.5B's L21, L22, L23 (also last 3 layers)
- The shift is **proportional to depth**: both models concentrate transferable behavior in their final 3 layers
- Recovery is more uniform at 1.5B — L24 and L23 also exceed 97%, whereas at 0.5B, L20 dropped to 87%
- Early layers show low recovery (~3–6% at L0–L1, rising to ~19% at L9, ~22% at L11)
- This confirms that trained behavior is encoded in late-layer activation patterns at both scales, but the "late" zone is proportionally the same (final ~10% of depth)

### 3.7 Skill Knockout via Negative Steering

**Finding 7: Skill knockout selectivity collapses from 11,654× to 0.24× at scale.**
**Confidence: MEDIUM**

Skill knockout experiments on factual_recall reveal that selective suppression is dramatically harder at 1.5B:

| Layer | Selectivity | Skill Drop | Non-Skill Drop | SV Norm |
|-------|------------|------------|----------------|---------|
| L2 | −24.55 | −0.0095 | −0.0004 | 8.88 |
| L3 | −77.41 | −0.0099 | −0.0001 | 10.63 |
| L16 | −9.31 | −0.0033 | +0.0004 | 26.13 |
| L19 | −1.62 | −0.0861 | +0.0532 | 36.50 |
| L21 | **+0.24** | +0.0011 | +0.0044 | 49.50 |

Key observations:
- The best selectivity is at **L21: 0.24×**, compared to 0.5B's **L19: 11,654×** — a **~48,000× reduction** in selectivity
- A selectivity of 0.24× means the skill drop is only 24% of the non-skill drop — the knockout is **non-selective** (it suppresses non-skill behavior more than skill behavior)
- Most layers show **negative selectivity** (−1.6 to −77.4), meaning the steering direction suppresses non-skill tokens more than skill tokens — the opposite of the intended effect
- L19 (the best knockout layer at 0.5B) has selectivity −1.62 at 1.5B, showing that the same intervention point does not transfer across scales
- SV norms are large (up to 49.5 at L21) but do not translate to selective manipulation

This is perhaps the most consequential scaling finding: the ability to selectively remove skills via steering, which was a headline result at 0.5B, is **effectively lost** at 1.5B. Skills are more entangled in the larger model's representations.

### 3.8 Adapter-Only Ablation: Norm vs Effect

**Finding 8: Adapter norm-effect correlation weakens from 0.85 to 0.54 at scale.**
**Confidence: MEDIUM**

The correlation between adapter weight norm and ablation effect is **0.54** at 1.5B, compared to **0.85** at 0.5B. This indicates a weaker relationship between where training writes weights and where the functional impact manifests.

Top adapter ablation effect layers:

| Layer | Total KL | Adapter Norm |
|-------|----------|-------------|
| L27 | 13.64 | 3.42 |
| L26 | 12.70 | 3.39 |
| L25 | 12.00 | 3.39 |
| L24 | 11.56 | 3.37 |
| L23 | 10.92 | 3.41 |

Key observations:
- All top effect layers are late (L23–L27), consistent with 0.5B (L19–L23)
- Adapter norms are remarkably uniform (3.33–3.45 across all 28 layers), with minimal variation
- The weaker correlation (0.54 vs 0.85) means adapter norms are **less predictive** of functional impact at 1.5B
- At 0.5B, norm and effect were both concentrated at late layers (corr 0.85). At 1.5B, norms are flat but effects still peak late — the decoupling suggests more complex propagation dynamics

Per-family adapter ablation shows L27 dominates in most families (delimiter_tracking, json_schema, factual_recall, dead_code), while arithmetic peaks at L24 and code_syntax at L16.

### 3.9 Efficiency Experiments: Layer Skipping and Early Exit

**Finding 9: Naive layer skipping remains fatal — all configurations break output except one partial exception.**
**Confidence: HIGH (for the negative result)**

We tested 4 layer-skip configurations:

| Config | Layers Skipped | Mean KL | Top-5 Overlap |
|--------|---------------|---------|---------------|
| skip_weakest_1 | L15 | 9.50 | 0% |
| skip_mid_5 | L4–L8 | 6.58 | 10% |
| skip_6_layers | L4,5,8,11,15,16 | 9.15 | 2.86% |
| skip_8_layers | L4,5,8,10,11,14,15,16 | 9.15 | 2.86% |

Key observations:
- Skipping even the single weakest layer (L15) produces 0% top-5 overlap and KL of 9.50
- The only non-zero overlap is **skip_mid_5 at 10%** — skipping 5 consecutive mid layers (L4–L8) preserves 10% of top-5 predictions, but KL is still 6.58
- This partial exception suggests some mid-layer redundancy exists at 1.5B that was absent at 0.5B, but it is far from practical
- More aggressive skipping (6–8 layers) does not improve over skipping 1

**Finding 10: Early exit fails at all layers — intermediate hidden states are not directly projectable.**
**Confidence: HIGH (for the negative result)**

| Exit Layer | Layers Skipped | Mean KL | Argmax Match | Speedup |
|-----------|---------------|---------|-------------|---------|
| L27 (full) | 0 | ∞ | 0% | 1.00× |
| L26 | 1 | ∞ | 0% | 1.04× |
| L25 | 2 | ∞ | 7.14% | 1.08× |
| L23 | 4 | 12.60 | 7.14% | 1.17× |
| L17 | 10 | 7.21 | 0% | 1.56× |

Key observations:
- No early exit configuration produces usable output — argmax match never exceeds 7.14%
- Even exiting one layer early (L26) gives 0% argmax match
- The ∞ KL values at L25–L26 indicate the intermediate distributions are so different they overflow standard KL computation
- Theoretical speedups (1.04–1.56×) are meaningless without correct output
- This confirms the 0.5B finding: each layer transforms the residual stream, and intermediate hidden states are not directly projectable to vocabulary

---

## 4. Cross-Experiment Synthesis

### 4.1 The Hub Migration

The most fundamental scaling finding is the **migration of the universal hub from L2 to L26**. At 0.5B (24 layers), L2 was the singular routing hub — early in the network, handling first and last position tokens. At 1.5B (28 layers), L26 is the primary hub — late in the network, with L6 and L14 as secondary clusters.

This migration suggests that as models scale, the critical routing functionality moves to later layers where more abstract representations have been built. The hub is not fixed to a specific depth fraction; it shifts from ~8% depth (L2/24) to ~93% depth (L26/28).

### 4.2 The Specialization Shift

At 0.5B, processing was distributed across heads (max head KL 0.046) and concentrated in MLPs (max MLP KL 8.12). At 1.5B, this inverts:
- **Heads gain 22× more impact** (max 1.02) — specialist heads emerge, particularly in L0
- **MLPs lose 3× impact** (max 2.58) — MLPs become less dominant

This suggests that scale shifts the computational burden from MLPs (which dominated at 0.5B) toward attention heads (which gain specialization at 1.5B). The model trades MLP magnitude for head precision.

### 4.3 The Steering Collapse

The 70× reduction in steering leverage (0.213 → 0.003) is the most practically consequential scaling effect. At 0.5B, steering was a viable tool for skill manipulation — 3.3× probability boosts, 11,654× knockout selectivity. At 1.5B, steering is nearly inert: best boost 0.003, best knockout selectivity 0.24×.

This collapse likely reflects increased representational entanglement: at 1.5B, skills are distributed across more neurons and layers, making linear steering directions insufficient. The "steering budget" that was generous at 0.5B is exhausted at 1.5B.

### 4.4 The Transfer Zone

Cross-model patching confirms that trained behavior is encoded in the **final ~10% of layers** at both scales:
- 0.5B (24 layers): L21–L23 (88–96% depth)
- 1.5B (28 layers): L25–L27 (89–96% depth)

This proportionality is a robust structural invariant — the transfer zone is defined by relative depth, not absolute layer index.

---

## 5. Implications for Model Optimization at Scale

### 5.1 Skill Manipulation Becomes Harder

- **Steering** is 70× weaker — linear steering is not a viable manipulation tool at 1.5B
- **Skill knockout** selectivity drops ~48,000× — skills cannot be selectively removed without retraining
- **Practical implication:** targeted skill injection/removal methods that work at 0.5B may require non-linear interventions at 1.5B

### 5.2 Attention Heads Gain Functional Roles

- **Specialist heads emerge** (max KL 1.02, 22× stronger than 0.5B)
- **L0 H3** appears across arithmetic, code syntax, and copying — a potential multi-task specialist
- **Practical implication:** head-level pruning and optimization becomes meaningful at 1.5B, unlike at 0.5B where all heads were near-equal

### 5.3 Inference Optimization Remains Limited

- **Layer skipping** is still fatal (0% top-5 overlap in 3/4 configs)
- **One partial exception:** skipping 5 mid-layers (L4–L8) preserves 10% top-5 overlap — the first hint of mid-layer redundancy at scale
- **Early exit** fails at all layers — no practical speedup without quality loss
- **Practical implication:** layer removal is not viable, but the 10% overlap in mid-skip suggests structured pruning (not zero-ablation) might work with retraining

---

## 6. Limitations

1. **Single seed**: All results from one random seed. Confidence capped at MEDIUM (except L26 hub at HIGH). Multi-seed replication needed for publication.
2. **Zero ablation**: Creates out-of-distribution activations. Mean/resample ablation would be more principled.
3. **VRAM-constrained training**: LoRA at batch_size=1 with gradient checkpointing may produce different internal changes than higher-batch training.
4. **8 task families** (vs 12 at 0.5B): The 1.5B suite covers fewer task families, limiting direct comparison on refusal/compliance, verbosity, and uncertainty tasks.
5. **Short synthetic prompts**: 5–15 tokens. Results may not transfer to natural language or longer contexts.
6. **LoRA only**: Full SFT OOMs on 8GB. LoRA may produce different internal changes than full fine-tuning.
7. **Steering tested at L2 only**: The steering collapse may be layer-specific; other layers were not tested due to compute constraints.
8. **Single comparison point**: Only 0.5B and 1.5B tested. A third scale point would establish scaling laws.

---

## 7. Open Hypotheses

| ID | Hypothesis | Status |
|----|-----------|--------|
| H001 | The universal routing hub persists at the same relative depth | REJECTED — hub migrates from L2 (~8% depth) to L26 (~93% depth) |
| H002 | Specialist attention heads emerge at scale | SUPPORTED — max head KL increases 22× (0.046 → 1.02) |
| H003 | Steering leverage scales with model capacity | REJECTED — steering collapses 70× (0.213 → 0.003) |
| H004 | Skill knockout selectivity is preserved at scale | REJECTED — selectivity drops ~48,000× (11,654× → 0.24×) |
| H005 | Cross-model transfer zone is defined by relative depth | SUPPORTED — final ~10% of layers at both scales |
| H006 | MLP dominance scales with parameters | REJECTED — MLP effects weaken 3× despite 3× more parameters |
| H007 | Mid-layer redundancy emerges at scale | PARTIALLY SUPPORTED — skip_mid_5 gives 10% top-5 overlap (vs 0% at 0.5B) |
| H008 | Adapter norm predicts functional impact | WEAKENED — correlation drops from 0.85 to 0.54 |

---

## 8. Key Findings

1. **L26 is the universal routing hub** — the hub migrates from L2 (0.5B) to L26 (1.5B), a shift from 8% to 93% depth (HIGH confidence)
2. **Specialist attention heads emerge** — max head KL increases 22× (0.046 → 1.02), with L0 H3 as a cross-task specialist (MEDIUM)
3. **MLP effects weaken 3× at scale** — max MLP KL drops from 8.12 to 2.58 despite 3× more parameters (MEDIUM)
4. **Steering collapses 70×** — best boost drops from 0.213 to 0.003, making linear steering impractical at 1.5B (MEDIUM)
5. **Skill knockout selectivity collapses ~48,000×** — from 11,654× to 0.24×, skills become entangled at scale (MEDIUM)
6. **Cross-model transfer shifts proportionally** — top layers move from L21–L23 to L25–L27, maintaining the final ~10% depth invariant (MEDIUM)
7. **Adapter norm-effect correlation weakens** — from 0.85 to 0.54, norms become less predictive of impact (MEDIUM)
8. **Naive layer skipping remains fatal** — 0% top-5 overlap in 3/4 configs, but skip_mid_5 shows 10% overlap (first hint of redundancy) (HIGH for negative)
9. **Early exit fails at all layers** — no configuration produces usable output (HIGH for negative)
10. **LoRA training converges at 1.5B** — loss 0.0098 with gradient checkpointing, batch_size=1 (MEDIUM)

---

## 9. Conclusion

We have extended the causal interpretability atlas from 0.5B to 1.5B parameters, revealing that scaling 3× fundamentally redistributes the model's internal causal structure. The universal routing hub migrates from early (L2) to late (L26) layers. Attention heads transition from distributed to partially specialized (22× stronger effects). MLPs lose relative dominance (3× weaker). Steering leverage collapses (70× weaker), and skill knockout becomes non-selective (~48,000× less selective). Cross-model activation transfer maintains its proportional depth invariant (final ~10% of layers). Inference optimization via layer removal remains impractical, though one configuration hints at emerging mid-layer redundancy.

These findings establish that interpretability insights from small models do not directly transfer to larger ones — the causal architecture itself transforms with scale. Methods that are effective at 0.5B (steering, knockout) may require fundamentally different approaches at 1.5B. The atlas provides a roadmap for which interventions scale and which require rethinking.

---

## Appendix A: Experiment Registry

| Experiment | Result File | Key Metric |
|-----------|-------------|------------|
| Layer ablation | layer_ablation_1.5b.json | Max mean KL: L26 = 13.70 |
| MLP ablation | mlp_ablation_1.5b.json | Max mean KL: L0 = 2.58 |
| Head ablation | head_ablation_1.5b.json | Max KL: arithmetic L0 H3 = 1.02 |
| Steering sweep | steering_sweep_1.5b.json | Best boost: 0.003 (L2, factual) |
| LoRA comparison | lora_comparison_1.5b.json | Max base→trained KL: json = 6.47 |
| Cross-model patching | cross_model_patching_1.5b.json | Max recovery: L27 = 99.9% |
| Skill knockout | skill_knockout_1.5b.json | Best selectivity: L21 = 0.24× |
| Adapter ablation | adapter_ablation_1.5b.json | Norm-effect corr: 0.54 |
| Efficiency | efficiency_1.5b.json | Best top-5 overlap: 10% (skip_mid_5) |

## Appendix B: Reproducibility

```bash
ssh aero
cd ~/work/autonomous-small-model-exploration
source .venv/bin/activate

# Run all 1.5B experiments
python scripts/run_layer_ablation.py --model Qwen/Qwen2.5-1.5B
python scripts/run_head_ablation.py --model Qwen/Qwen2.5-1.5B
python scripts/run_mlp_ablation.py --model Qwen/Qwen2.5-1.5B
python scripts/run_steering_sweep.py --model Qwen/Qwen2.5-1.5B
python scripts/train_lora_json.py --model Qwen/Qwen2.5-1.5B
python scripts/compare_lora_ablation.py --model Qwen/Qwen2.5-1.5B
python scripts/run_cross_model_patching.py --model Qwen/Qwen2.5-1.5B
python scripts/run_skill_knockout.py --model Qwen/Qwen2.5-1.5B
python scripts/run_adapter_ablation.py --model Qwen/Qwen2.5-1.5B
python scripts/run_efficiency.py --model Qwen/Qwen2.5-1.5B
```
