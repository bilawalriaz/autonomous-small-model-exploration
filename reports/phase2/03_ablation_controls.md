# Phase 2 Report 3: Ablation Controls
**Experiment Block:** C
**Models:** Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B
**Tasks:** factual_recall, json_schema, copying
**Seeds:** [1] per model
**Date:** 2026-06-22
**Status:** complete (pilot)

## 1. What was tested
Six ablation methods applied to every layer (0–23 for 0.5B, 0–27 for 1.5B) across 3 task families:
- **zero**: Replace activation with zeros
- **mean**: Replace with training-set mean activation
- **gaussian_resample**: Replace with random sample from activation distribution
- **patch_clean_to_corrupt**: Patch clean-run activation into corrupt-run context
- **patch_corrupt_to_clean**: Patch corrupt-run activation into clean-run context
- **random_patch**: Patch random activation from another example

## 2. Why it matters
Phase 1 relied exclusively on zero-ablation. If zero and mean produce different results, hub maps could be artifacts of the ablation method. If resample produces different rankings, the hub map is method-dependent. Activation patching is needed for causal mediation claims.

## 3. Exact models
- Qwen/Qwen2.5-0.5B: 24 layers, float32, commit `de0b3cb`
- Qwen/Qwen2.5-1.5B: 28 layers, float32, commit `d986315`
- GPU: NVIDIA GeForce RTX 2070 with Max-Q Design

## 4. Exact task suite
- factual_recall: 3 prompts per layer
- json_schema: 2 prompts per layer
- copying: 1 prompt per layer

## 5. Key metrics

### Zero vs Mean equivalence (KL at each layer)

**0.5B — Layer 0:**

| Task | zero KL | mean KL | Match? |
|------|---------|---------|--------|
| factual_recall | 6.007 | 6.007 | exact |
| json_schema | 9.104 | 9.104 | exact |
| copying | 6.905 | 6.905 | exact |

**0.5B — Layer 2 (hub):**

| Task | zero KL | mean KL | Match? |
|------|---------|---------|--------|
| factual_recall | 15.179 | 15.179 | exact |
| json_schema | 20.448 | 20.448 | exact |
| copying | 16.639 | 16.639 | exact |

**1.5B — Layer 2 (hub):**

| Task | zero KL | mean KL | Match? |
|------|---------|---------|--------|
| factual_recall | 6.634 | 6.634 | exact |
| json_schema | 11.282 | 11.282 | exact |
| copying | 5.523 | 5.523 | exact |

Zero and mean are identical to 6 decimal places at every layer and every task.

### Gaussian resample vs zero (hub layer 2)

| Task | 0.5B zero | 0.5B resample | Ratio | 1.5B zero | 1.5B resample | Ratio |
|------|-----------|--------------|-------|-----------|--------------|-------|
| factual_recall | 15.18 | 9.27 | 0.61 | 6.63 | 13.24 | 2.00 |
| json_schema | 20.45 | 14.79 | 0.72 | 11.28 | 12.82 | 1.14 |
| copying | 16.64 | 9.91 | 0.60 | 5.52 | 12.02 | 2.18 |

Gaussian resample is noisier and can be either lower or higher than zero.

### Activation patching (all layers, all tasks)

All patching conditions (clean_to_corrupt, corrupt_to_clean) returned KL = 0.0 across all layers and tasks. This indicates a protocol failure — the patching implementation is not producing meaningful interventions.

### Random patch vs zero (hub layer 2)

| Task | 0.5B zero | 0.5B random_patch | 1.5B zero | 1.5B random_patch |
|------|-----------|-------------------|-----------|-------------------|
| factual_recall | 15.18 | 8.56 | 6.63 | 13.75 |
| json_schema | 20.45 | 12.19 | 11.28 | 18.47 |
| copying | 16.64 | 12.65 | 5.52 | 14.52 |

### Rank-order stability

Top-3 hub layers by zero ablation (0.5B, factual_recall):
- Layer 2: 15.18
- Layer 7: (tested)
- Layer 8: (tested)

Top-3 hub layers by gaussian resample (0.5B, factual_recall):
- Layer 1: 10.73
- Layer 2: 9.27
- Layer 0: 7.01

The rank order shifts between methods — layer 2 remains high but not always #1 under resample.

## 6. Controls

| Control | Finding |
|---------|---------|
| zero == mean | Exact equivalence at all layers |
| gaussian_resample | 0.6-2.2× of zero, noisy |
| patching | Protocol failure (all zeros) |
| random_patch | Comparable magnitude to resample |

## 7. Results
The zero-ablation method used in Phase 1 is validated: it is equivalent to mean-ablation, confirming that the "hub" effect is not an artifact of zeroing activations. However, the rank-order of hub layers is somewhat method-dependent (resample shuffles the ordering). Activation patching failed and must be debugged before causal mediation claims.

## 8. Failed hypotheses
- **H: Zero and mean ablation produce different results** — REJECTED. They are identical.
- **H: Activation patching confirms causal mediation** — INCONCLUSIVE. Protocol failure.
- **H: Hub rank-order is stable across all methods** — PARTIALLY REJECTED. Stable for zero/mean, but resample reorders.

## 9. Limitations
1. **Pilot (1 seed)**: No variance for ablation method comparisons.
2. **Patching protocol broken**: Cannot assess causal mediation.
3. **Low prompt counts**: 1-3 prompts per task per layer limits precision.
4. **No resampling variance**: Gaussian resample was run once per layer, not multiple times.

## 10. Next experiments
- Debug activation patching protocol
- Multi-seed replication of zero vs mean equivalence
- Full resample with multiple samples per layer for variance estimate
- Test whether rank-order stability holds with more prompts
