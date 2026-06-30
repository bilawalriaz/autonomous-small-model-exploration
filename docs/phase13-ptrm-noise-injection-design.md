# Phase 13 Design: Atlas-Guided Stochastic Inference (PTRM-Inspired)

## Status: DESIGN (not yet implemented)

## Paper Reference

Probabilistic Tiny Recursive Model (PTRM) — arXiv:2605.19943

Key idea: inject Gaussian noise into latent states during inference to escape "bad basins" that trap models into incorrect completions. Uses K parallel noisy rollouts + Q-head selection to pick the best trajectory. Achieves 91.2% on PPBench with 7M params vs 34.7% for Claude Opus.

## Why This Matters for MI-Atlas

PTRM validates a hypothesis implicit in our atlas data: **small models have the capability to solve harder problems but get trapped by deterministic inference.** Their fix is brute-force uniform noise + learned selection. Our atlas tells us *exactly where* noise matters and *where it's wasted*. We can do better.

### What Our Atlas Already Tells Us (LFM2.5-230M)

| Finding | Implication for Noise Injection |
|---------|--------------------------------|
| L0 (conv) is universal hub (skip KL=82.9) | Noise here has maximum downstream impact |
| L5-L13 CKA=1.0 (identical reps) | Noise at L5-L13 is wasted — reps don't change |
| L0 residual norm = 2.0 | σ=0.2 noise = 10% perturbation (huge relative effect) |
| L6-L13 residual norm = 25.5 | σ=0.2 noise = 0.8% perturbation (negligible) |
| Steering most effective at L0-L6 | Same principle: early layers are intervention targets |
| Conv MLPs 2.12x > Attn MLPs | Noise in conv layers has more impact per unit |
| L4 (attn) secondary hub | Second-best noise target |

**Core prediction:** Atlas-guided noise at L0 only should match or beat PTRM's uniform noise across all layers, at 1/14th the compute (1 rollout target vs 14).

## Hypotheses

### H13-1: Hub-Only Noise Matches Uniform Noise
Atlas-guided noise at L0 only (1 layer) produces equivalent task performance to uniform noise across all 14 layers, at K=10 rollouts.

**Falsifier:** If uniform noise significantly outperforms hub-only noise (>5% accuracy gap), the atlas is missing important noise targets.

**Rationale:** L0 skip KL=82.9 accounts for ~50% of total layer importance. Noise at L0 propagates through all downstream layers. Noise at L5-L13 is absorbed by the high residual norm (25.5) and identical representations (CKA=1.0).

### H13-2: Hub Noise Outperforms Random-Layer Noise
Noise at L0 (atlas-identified hub) outperforms noise at a randomly selected layer (e.g., L7, L9) at matched σ and K.

**Falsifier:** If random-layer noise matches hub noise, the atlas doesn't predict noise sensitivity.

**Rationale:** L7 skip KL=7.26, L9 skip KL=7.07 — both are 11.6x weaker than L0. Noise injected into low-importance layers should have minimal effect on output.

### H13-3: Q-Value Selection Improves Over Random Selection
Selecting the best rollout using KL-divergence-to-prompt (or loss proxy) outperforms random rollout selection at K=10.

**Falsifier:** If random selection matches Q-value selection, the selection signal is no better than chance.

**Rationale:** PTRM's Q-head strongly correlates with correctness. We don't have a trained Q-head, but we can use the model's own loss (negative log-likelihood on the prompt tokens) as a proxy — lower loss = more confident = likely correct.

### H13-4: Width Scaling Beats Depth Scaling
K=10 noisy rollouts outperforms D=10 recursive refinement steps (if applicable to the architecture).

**Falsifier:** If recursive refinement outperforms noisy rollouts, depth > width for this model.

**Rationale:** PTRM shows width scaling (parallel rollouts) dominates depth scaling (sequential refinement). Our model doesn't have native recursion, but we can approximate depth with iterative self-refinement prompts.

### H13-5: Noise-Induced Diversity Creates Measurably Different Completions
At K=10 with hub noise, the model produces at least 3 distinct completion clusters (measured by embedding cosine similarity), not just 10 near-identical outputs.

**Falsifier:** If all K=10 outputs are near-identical (cosine > 0.99), noise doesn't actually explore the solution space.

**Rationale:** PTRM shows ~8% of rollouts escape bad basins. If noise at the hub creates genuine diversity, we should see distinct clusters.

## Experimental Design

### Experiment 13A: Noise Localization (Which Layer?)

**Question:** Which layer's noise injection gives the biggest accuracy boost?

**Method:**
1. Take LFM2.5-230M (base model, no adapter)
2. Select 50 structured extraction prompts (JSON schema, entity extraction) — tasks where model scores ~83%
3. For each layer L0-L13:
   a. Run K=10 rollouts with Gaussian noise (σ=0.2) injected at that layer only
   b. Score each rollout on exact match / JSON validity
   c. Record the best-of-K accuracy
4. Compare best-of-K accuracy per layer

**Predicted results:**
- L0: best_of_10 ≈ 93-95% (hub noise, high impact)
- L4: best_of_10 ≈ 90-92% (secondary hub)
- L5-L13: best_of_10 ≈ 85-87% (marginal improvement over no noise)
- No noise: best_of_10 ≈ 83% (single-pass baseline)

**Config:**
```json
{
  "experiment_id": "P13A",
  "model": "LiquidAI/LFM2.5-230M",
  "task": "structured_extraction",
  "n_prompts": 50,
  "K": 10,
  "sigma": 0.2,
  "layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
  "seeds": [42, 137, 2026],
  "metrics": ["exact_match", "json_validity", "best_of_K_accuracy", "completion_diversity"],
  "hardware": "aero (RTX 2070 Super 8GB)",
  "estimated_time": "14 layers × 50 prompts × 10 rollouts × 3 seeds = ~2 hours"
}
```

### Experiment 13B: Width Scaling Curve (How Many Rollouts?)

**Question:** How does accuracy scale with K (number of noisy rollouts)?

**Method:**
1. Use L0-only noise (from 13A winner)
2. Same 50 prompts
3. Run K = 1, 2, 5, 10, 20, 50
4. Plot accuracy vs K

**Predicted results (following PTRM's scaling curve):**
- K=1: 83% (baseline, no noise diversity)
- K=5: 90-92%
- K=10: 93-95%
- K=20: 94-96% (diminishing returns)
- K=50: 95-97% (plateau)

**Config:**
```json
{
  "experiment_id": "P13B",
  "model": "LiquidAI/LFM2.5-230M",
  "layer": 0,
  "K_values": [1, 2, 5, 10, 20, 50],
  "sigma": 0.2,
  "seeds": [42],
  "estimated_time": "~45 minutes"
}
```

### Experiment 13C: Sigma Sweep (How Much Noise?)

**Question:** What's the optimal noise magnitude at the hub layer?

**Method:**
1. L0-only, K=10, 50 prompts
2. σ = 0.05, 0.1, 0.2, 0.5, 1.0, 2.0
3. Plot accuracy vs σ

**Predicted results:**
- σ=0.05: minimal diversity, ~84% (too small to escape basins)
- σ=0.2: sweet spot, ~93% (PTRM's default for PPBench)
- σ=1.0: too noisy, ~88% (destroys useful representations)
- σ=2.0: collapse, ~60% (garbage in, garbage out)

**Note:** L0 residual norm = 2.0, so σ=0.2 is a 10% perturbation. σ=1.0 is a 50% perturbation. The optimal σ should be proportional to the residual norm at the target layer.

### Experiment 13D: Selection Strategy (How to Pick the Best?)

**Question:** What's the best way to select among K noisy rollouts?

**Methods to compare:**
1. **Random selection** (baseline): pick one rollout at random
2. **Loss-based selection**: compute prompt NLL for each rollout, pick lowest
3. **Logit-conf selection**: pick rollout with highest softmax max-prob on first generated token
4. **Ensemble voting**: pick the most common answer across K rollouts (majority vote)
5. **Oracle selection**: pick the rollout with highest ground-truth match (upper bound)

**Predicted ranking:** Oracle > Loss-based ≈ Logit-conf > Ensemble > Random

**Rationale for loss-based:** PTRM uses a trained Q-head. We don't have one, but the model's own loss on the prompt is a free proxy that should correlate with completion quality. If this works, it means we can skip training a separate verifier.

### Experiment 13E: Bad Basin Detection (Does the Hypothesis Hold?)

**Question:** Do "bad basins" actually exist in LFM2.5-230M's latent space?

**Method:**
1. Take 20 prompts where the model gets the wrong answer (single-pass)
2. Run K=100 noisy rollouts at L0 (σ=0.2)
3. Cluster completions by embedding similarity (cosine threshold 0.95)
4. Measure: how many distinct clusters? What fraction are correct?

**If PTRM's mechanism applies:**
- Expect 2-5 distinct clusters per prompt
- One cluster = correct answer (the "good basin")
- Other clusters = incorrect variants (the "bad basins")
- The correct cluster should contain 5-15% of rollouts (matching PTRM's ~8% escape rate)

**If bad basins DON'T exist:**
- All rollouts produce similar outputs (1 cluster)
- Or all rollouts are equally wrong (no "good basin" reachable by noise)

This experiment is the most diagnostic — it tells us whether the PTRM framework applies to autoregressive models at all, or only to recursive architectures.

### Experiment 13F: Atlas-Guided vs Uniform Noise (Head-to-Head)

**Question:** Does knowing the hub layer actually help, or is any noise good enough?

**Conditions:**
1. **No noise** (baseline): K=1 single-pass
2. **Uniform noise**: σ=0.2 at all 14 layers, K=10
3. **Hub-only noise**: σ=0.2 at L0 only, K=10
4. **Random-layer noise**: σ=0.2 at one random layer (L7), K=10
5. **Scaled noise**: σ proportional to layer importance (σ_L0=0.5, σ_L5=0.3, σ_L7=0.1, σ_L13=0.05), K=10
6. **Early-phase noise**: σ=0.2 at L0-L5 only, K=10

**Predicted ranking:** Scaled ≈ Hub-only > Early-phase > Uniform > Random-layer > No noise

**Key comparison:** Hub-only vs Uniform. If hub-only matches uniform at 1/14th the compute, atlas-guidance is validated. If uniform wins, the atlas is missing something.

## Implementation Plan

### Phase 1: Infrastructure (Day 1)

1. Create `scripts/ptrm/noisy_inference.py`:
   - Hook-based noise injection at specified layers
   - Support for Gaussian noise with configurable σ per layer
   - K-parallel rollout manager
   - Loss-based selection logic
   - Completion clustering (embedding similarity)

2. Create `scripts/ptrm/selection_strategies.py`:
   - Random selection
   - Loss-based selection (prompt NLL)
   - Logit-confidence selection
   - Majority voting
   - Oracle selection (requires ground truth)

3. Create `scripts/ptrm/eval_pipeline.py`:
   - Reuse existing task suite (structured extraction)
   - Score exact match, JSON validity, completion diversity
   - Aggregate across seeds

4. Create `configs/P13A.json` through `configs/P13F.json`

### Phase 2: Run 13A (Day 1-2)

1. Run noise localization experiment
2. Identify winning layer(s)
3. Update atlas with noise sensitivity map

### Phase 3: Run 13B-13D (Day 2)

1. Width scaling curve at winning layer
2. Sigma sweep at winning layer
3. Selection strategy comparison

### Phase 4: Run 13E (Day 2-3)

1. Bad basin detection (K=100 on 20 wrong-answer prompts)
2. Cluster analysis
3. Determine if PTRM framework applies

### Phase 5: Run 13F (Day 3)

1. Head-to-head comparison
2. Final report with recommendations

### Phase 6: Report and Publish (Day 3-4)

1. Write Phase 13 report
2. Update GitHub Pages
3. Update MI-Atlas skill with findings

## Estimated Compute

| Experiment | Time (aero) | VRAM |
|------------|-------------|------|
| 13A: Noise localization | ~2 hours | ~1 GB |
| 13B: Width scaling | ~45 min | ~1 GB |
| 13C: Sigma sweep | ~30 min | ~1 GB |
| 13D: Selection strategy | ~1 hour | ~1 GB |
| 13E: Bad basin detection | ~2 hours | ~2 GB (K=100) |
| 13F: Head-to-head | ~1.5 hours | ~1 GB |
| **Total** | **~8 hours** | **~2 GB peak** |

All well within aero's 8GB VRAM budget. LFM2.5-230M is only 450MB in bf16.

## What This Enables (If It Works)

1. **Practical inference-time amplification for LFM2.5-230M**: If hub noise + loss selection boosts structured extraction from 83% to 93%+, that's a free 10pp gain with no retraining — just 10x inference compute.

2. **Validation of atlas-guided interventions**: If hub-only noise matches uniform noise, it proves the atlas is operationally useful, not just academically interesting.

3. **Bridge to SFT findings**: Phase 9 showed 300 examples = catastrophic overfitting at 230M. Noise-based inference amplification could be the alternative path: instead of training more data, explore more at inference time.

4. **PTRM generalization**: If this works on autoregressive models (not just recursive TRM), it's a significant finding — the technique is more general than the paper claims.

5. **Cheap verifier discovery**: If loss-based selection works as well as a trained Q-head, we've found a zero-cost verification mechanism for small models.

## Open Questions

1. **Does noise at L0 interact with the conv gate structure?** The gated convolution (B, C, x) is multiplicative — noise on the residual stream enters *after* the conv operation. We might need to inject noise into the conv's internal state (in_proj output) rather than the residual stream.

2. **Is σ=0.2 optimal for L0's residual norm of 2.0?** PTRM tuned σ per task. Our atlas says L0 norm = 2.0, but the optimal perturbation ratio might not be 10%.

3. **Does this work with adapters?** Phase 9's best adapter (multi-turn verbose, loss 1.37) might have different noise sensitivity than the base model. The hub could shift after LoRA training.

4. **Can we train a lightweight Q-head?** If loss-based selection is mediocre, we could train a small MLP on top of L0's hidden states to predict completion quality — essentially PTRM's Q-head but using atlas-guided features.

---

*Date: 2026-06-30*
*Inspired by: PTRM (arXiv:2605.19943)*
*Target model: LiquidAI/LFM2.5-230M*
*Hardware: aero (RTX 2070 Super 8GB)*
*Prerequisites: LFM2.5-230M Phase 1 atlas (complete), Phase 9 SFT sweep (complete)*
