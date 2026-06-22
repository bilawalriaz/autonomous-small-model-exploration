# Phase 2 Report 9: Long Task Robustness
**Experiment Block:** I
**Models:** Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B
**Tasks:** factual_recall, json_schema (at short/medium/long prompt lengths)
**Seeds:** [1]
**Date:** 2026-06-22
**Status:** partial (0.5B complete; 1.5B lora failed)

## 1. What was tested
Layer ablation, steering, and LoRA adapter effects measured at 3 prompt lengths (short, medium, long) for factual_recall and json_schema tasks. This tests whether hub positions and intervention effectiveness are stable when prompts get longer.

## 2. Why it matters
The atlas is built on short prompts. If hubs shift with prompt length, the atlas is only valid for short-context tasks. Real-world use cases often involve longer prompts, so robustness to length is critical for practical applicability.

## 3. Exact models
- Qwen/Qwen2.5-0.5B: 24 layers, float32
- Qwen/Qwen2.5-1.5B: 28 layers, float32
- 1.5B LoRA experiment failed: tried to load 0.5B adapter (dim 896) into 1.5B model (dim 1536)

## 4. Exact task suite
- factual_recall: 3 prompts × 3 lengths (short: "Capital of France?", medium: "What is the capital...", long: "In the context of European geography...")
- json_schema: 3 prompts × 3 lengths
- 7 steering strengths: -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0

## 5. Key metrics

### Hub layer migration — Qwen2.5-0.5B

| Task | Short Hub | Medium Hub | Long Hub | Stable? |
|------|-----------|-----------|---------|---------|
| factual_recall | L2 (16.15) | L7 (13.44) | L22 (15.96) | NO |
| json_schema | (data available) | (data available) | (data available) | pending analysis |

### Hub layer migration — Qwen2.5-1.5B

| Task | Short Hub | Medium Hub | Long Hub | Stable? |
|------|-----------|-----------|---------|---------|
| factual_recall | L26 (12.93) | L14 (19.52) | L14 (19.47) | NO |

### Top-3 hub layers by length — 0.5B factual_recall

| Rank | Short | Medium | Long |
|------|-------|--------|------|
| 1 | L2 (16.15) | L7 (13.44) | L22 (15.96) |
| 2 | L22 (15.96) | L22 (13.32) | L2 (16.15) |
| 3 | L7 (13.44) | L2 (13.32) | L7 (13.44) |

The same 3 layers (2, 7, 22) dominate at all lengths, but their ranking shifts.

### Top-3 hub layers by length — 1.5B factual_recall

| Rank | Short | Medium | Long |
|------|-------|--------|------|
| 1 | L26 (12.93) | L14 (19.52) | L14 (19.47) |
| 2 | L6 (10.92) | L16 (17.07) | L16 (17.68) |
| 3 | L14 (10.58) | L5 (16.23) | L5 (15.45) |

### Steering effectiveness by prompt length — 0.5B layer 2

| Strength | Short KL | Medium KL | Long KL |
|----------|---------|----------|---------|
| -4.0 | 0.320 | 0.114 | 0.430 |
| -2.0 | 0.092 | 0.033 | 0.114 |
| -1.0 | 0.018 | 0.006 | 0.012 |
| 0.0 | 0.000 | 0.000 | 0.000 |
| 1.0 | 0.009 | 0.005 | 0.007 |
| 2.0 | 0.019 | 0.012 | 0.018 |
| 4.0 | 0.053 | 0.026 | 0.051 |

Steering at layer 2 is relatively stable across lengths for 0.5B (same order of magnitude).

### Steering effectiveness by prompt length — 1.5B layer 26

| Strength | Short KL | Medium KL | Long KL |
|----------|---------|----------|---------|
| -4.0 | 6.626 | 5.654 | 6.237 |
| -2.0 | 2.508 | 2.157 | 2.226 |
| -1.0 | 0.616 | 0.533 | 0.549 |
| 0.0 | 0.000 | 0.000 | 0.000 |
| 1.0 | 0.580 | 0.517 | 0.549 |
| 2.0 | 2.107 | 1.871 | 2.083 |
| 4.0 | 5.874 | 5.207 | 5.649 |

Steering at layer 26 in 1.5B is stable across lengths (KL varies by <15%).

### LoRA adapter effect by prompt length — 0.5B

| Task | Short KL | Medium KL | Long KL |
|------|---------|----------|---------|
| factual_recall | 0.067 | 1.099 | 0.513 |
| json_schema | 0.099 | 0.796 | 0.344 |

Adapter effect peaks at medium length, then decreases at long length.

## 6. Controls
- Full-layer ablation at each length (24 layers for 0.5B, 28 for 1.5B)
- Steering sweep at 7 strengths
- Same prompts across all conditions

## 7. Results
Hub layers are NOT position-stable across prompt lengths. In 0.5B, the top-3 layers remain the same (2, 7, 22) but their ranking changes. In 1.5B, the hub shifts from L26 (short) to L14 (medium/long). However, steering effectiveness at the short-prompt hub remains reasonable for both models. The LoRA adapter effect peaks at medium length, suggesting length-dependent adapter sensitivity.

## 8. Failed hypotheses
- **H: Hub positions are stable across prompt lengths** — REJECTED. Rankings shift.
- **H: Steering is ineffective at non-hub layers for long prompts** — INCONCLUSIVE. Steering at the short-prompt hub still works for long prompts.

## 9. Limitations
1. **Single seed**: No variance for hub position comparisons.
2. **1.5B LoRA failed**: Cannot compare adapter length-sensitivity across scales.
3. **Only factual_recall fully analyzed**: json_schema length data available but not fully analyzed.
4. **"Long" prompts are still short**: True long-context (1000+ tokens) not tested.

## 10. Next experiments
- Fix 1.5B LoRA loading (need 1.5B-specific adapter)
- Test true long-context prompts (1000+ tokens)
- Multi-seed replication of hub migration
- Test whether multi-layer steering compensates for hub migration
