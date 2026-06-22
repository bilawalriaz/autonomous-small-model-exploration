# Phase 2 Report 4: Third Scale Point
**Experiment Block:** D
**Models:** Qwen/Qwen2.5-3B
**Tasks:** factual_recall, json_schema, copying, code, arithmetic, code_syntax, code_semantics, dead_code, verbosity_control, uncertainty_signalling, refusal_compliance, delimiter_tracking (12 task families)
**Seeds:** [1]
**Date:** 2026-06-22
**Status:** partial (layer/head/mlp/steering/lora complete; patching/skip/knockout failed)

## 1. What was tested
Full atlas at 3B scale: layer ablation (all 36 layers), head ablation (6 representative layers × 16 heads), MLP ablation (all 36 layers), steering sweep (5 candidate layers), and LoRA JSON injection. This is the third scale point (after 0.5B and 1.5B) to test whether hub concentration is scale-invariant.

## 2. Why it matters
Three scale points (0.5B, 1.5B, 3B) are the minimum for establishing a trend. If hubs persist and maintain proportional position, the atlas is scale-invariant. If hubs disappear or shift dramatically, the atlas is scale-dependent.

## 3. Exact models
- Qwen/Qwen2.5-3B: 36 layers, 16 KV heads (GQA), 128 query heads
- Architecture: grouped query attention (16 KV heads, 128 query heads)
- This GQA architecture caused tensor size mismatch errors in patching/skip/knockout

## 4. Exact task suite
12 task families evaluated with 15 examples each:
copying, delimiter_tracking, json_schema, factual_recall, arithmetic, code_syntax, code_semantics, dead_code, verbosity_control, uncertainty_signalling, refusal_compliance, instruction_following

## 5. Key metrics

### Layer ablation (top-5 layers by mean effect across tasks)

From the effect_matrix (36 layers × 12 tasks):

| Layer | Mean Effect | Max Effect | Top Task |
|-------|-------------|------------|----------|
| 4 | ~10.5 | 14.31 | json_schema |
| 5 | ~10.2 | 16.38 | (varies) |
| 2 | ~8.0 | 10.81 | (varies) |
| 3 | ~7.9 | 9.50 | (varies) |
| 6 | ~7.0 | 11.06 | (varies) |

### MLP ablation (top-5 layers by mean effect)

| Layer | Factual | JSON | Copying | Code |
|-------|---------|------|---------|------|
| 0 | 4.53 | 0.58 | 2.75 | 1.09 |
| 1 | 0.32 | 0.16 | 0.17 | 0.03 |
| 2 | 0.10 | 0.53 | 0.26 | 0.27 |
| 11 | 0.20 | 0.91 | 0.04 | 0.03 |
| 12 | 0.16 | 0.63 | 0.05 | 0.16 |

Layer 0 MLP dominates with 4.53 effect on factual_recall — 8× higher than any other layer.

### Head ablation (layer 2, top heads)

| Head | Mean Effect | Max Effect |
|------|-------------|------------|
| head_00 | 1.092 | 2.547 |
| head_02 | 0.532 | 1.219 |
| head_14 | 0.173 | 0.439 |
| head_06 | 0.085 | 0.268 |
| head_12 | 0.068 | 0.125 |
| all others | <0.055 | <0.16 |

Only 2 of 16 heads at layer 2 are significantly active.

### Steering sweep

| Layer | SV Norm | KL at -4.0 (JSON prompt 1) | KL at +4.0 |
|-------|---------|---------------------------|------------|
| 2 | 17.0 | 7.53 | 5.47 |
| 18 | (tested) | (data available) | — |
| 26 | (tested) | (data available) | — |
| 34 | (tested) | (data available) | — |
| 35 | (tested) | (data available) | — |

### LoRA JSON injection

| Metric | Before | After (train loss 0.517) |
|--------|--------|--------------------------|
| json_schema exact_match | 0.0% | pending full eval |
| json_schema valid_json | 0.0% | pending |
| train_loss | N/A | 0.517 |
| peak_memory_gb | N/A | 2.06 |

## 6. Controls

| Experiment | Status | Notes |
|-----------|--------|-------|
| Layer ablation (all 36) | complete | Full coverage |
| Head ablation (6 layers) | complete | Representative sample |
| MLP ablation (all 36) | complete | Full coverage |
| Steering (5 layers) | complete | Candidate hub layers |
| Patching | FAILED | Tensor size mismatch (GQA: 16 KV vs 128 Q) |
| Layer skip | FAILED | Same GQA error |
| Knockout | FAILED | Same GQA error |

## 7. Results
The 3B model shows clear hubs in early layers (0-5). The MLP concentration at layer 0 is a new finding not present at smaller scales. Head specialization is extreme (2/16 heads carry the signal at layer 2). The GQA architecture prevents standard patching/skip/knockout experiments — these require architecture-aware implementations.

### Scale trend summary

| Property | 0.5B (24L) | 1.5B (28L) | 3B (36L) |
|----------|-----------|-----------|---------|
| Early hub range | L2-4 | L2-4 | L0-5 |
| Late hub range | L21-23 | L25-27 | pending |
| MLP hub | distributed | distributed | L0 extreme |
| Head specialization | moderate | moderate | extreme (2/16) |
| Steering vector norm (L2) | 1.49 | 9.08 | 17.0 |

## 8. Failed hypotheses
- **H: Patching confirms causal mediation at 3B** — BLOCKED by GQA architecture mismatch.
- **H: MLP effect is distributed** — REJECTED. Layer 0 MLP dominates (4.53 vs ~0.1 for others).

## 9. Limitations
1. **Single seed**: No variance.
2. **GQA blocks patching/skip/knockout**: Architecture-specific implementation needed.
3. **Head ablation only 6 of 36 layers**: May miss late-layer heads.
4. **LoRA eval incomplete**: Only train loss available, no post-training eval metrics.

## 10. Next experiments
- Implement GQA-aware patching for 3B
- Full head ablation across all 36 layers
- Investigate why MLP layer 0 is so dominant at 3B
- Multi-seed replication
