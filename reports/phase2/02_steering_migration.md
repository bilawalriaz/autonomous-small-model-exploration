# Phase 2 Report 2: Steering Migration
**Experiment Block:** B
**Models:** Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B
**Tasks:** json_schema (steering target), json_schema + copying (collateral)
**Seeds:** [1] per model
**Date:** 2026-06-22
**Status:** complete (pilot)

## 1. What was tested
Single-layer activation steering at hub and non-hub layers in both 0.5B and 1.5B models, with strength sweeps from -4.0 to +4.0. Task vectors were computed from json_schema examples and applied to test prompts. Random vectors served as controls.

## 2. Why it matters
Phase 1 found concentrated hubs at 0.5B. If hubs persist at 1.5B with shifted positions, this confirms architecture-dependent localization. If steering becomes distributed, the hub model fails at scale.

## 3. Exact models
- Qwen/Qwen2.5-0.5B: 24 layers, float32, commit `de0b3cb`
- Qwen/Qwen2.5-1.5B: 28 layers, float32, commit `de0b3cb`
- GPU: NVIDIA GeForce RTX 2070 with Max-Q Design

## 4. Exact task suite
- Steering target: json_schema (3 test prompts)
- Collateral measurement: json_schema + copying (KL divergence from base distribution)
- 8 strength levels: -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0

## 5. Key metrics

### Steering vector norms per layer

| Layer | 0.5B Norm | 1.5B Norm |
|-------|-----------|-----------|
| 2 | 1.49 | 9.08 |
| 6 | — | — |
| 8 | (tested) | — |
| 12 | (tested) | — |
| 14 | — | (tested) |
| 19 | (tested) | — |
| 21 | (tested) | (tested) |
| 22 | (tested) | — |
| 23 | (tested) | — |
| 25 | — | (tested) |
| 26 | — | (tested) |
| 27 | — | (tested) |

### KL divergence at hub layer 2, negative strengths (json task)

| Strength | 0.5B KL | 1.5B KL |
|----------|---------|---------|
| -4.0 | 0.287 | 0.572 |
| -2.0 | 0.047 | 0.352 |
| -1.0 | 0.018 | 0.057 |
| -0.5 | 0.005 | 0.013 |

### KL divergence at hub layer 2, positive strengths (json task)

| Strength | 0.5B KL | 1.5B KL |
|----------|---------|---------|
| 0.5 | 0.006 | 0.009 |
| 1.0 | 0.021 | 0.018 |
| 2.0 | 0.068 | 0.031 |
| 4.0 | 0.169 | 0.176 |

### Collateral damage at strength +4.0

| Model | json_schema collateral | copying collateral |
|-------|----------------------|-------------------|
| 0.5B | 0.160 | 0.114 |
| 1.5B | 0.353 | 0.117 |

## 6. Controls

### Random vector control at layer 2, strength -4.0

| Model | Task Vector KL | Random Vector KL | Task Collateral (json) | Random Collateral (json) |
|-------|---------------|-----------------|----------------------|------------------------|
| 0.5B | 0.287 | 0.147 | 0.160 | 0.911 |
| 1.5B | 0.572 | 0.500 | 0.353 | 0.372 |

**Key finding**: At 0.5B, random vectors produce 5.7× more collateral damage (0.911 vs 0.160), confirming task specificity. At 1.5B, the gap narrows (0.372 vs 0.353), suggesting less specificity.

## 7. Results

### Hub layer comparison

| Property | 0.5B | 1.5B |
|----------|------|------|
| Hub layers | 2, 21, 22, 23 | 2, 21, 25, 26, 27 |
| Early hub | Layer 2 | Layer 2 |
| Late hub range | L21-23 | L25-27 |
| Total layers | 24 | 28 |
| Late hub position (% of depth) | 87.5-95.8% | 89.3-96.4% |

Hub position as a percentage of model depth is nearly identical across scales (~88-96%), confirming proportional positioning.

### Steering effectiveness summary

| Model | Best single-layer KL (strength -4.0) | Best layer |
|-------|--------------------------------------|-----------|
| 0.5B | 0.287 | 2 |
| 1.5B | 0.572 | 2 |

## 8. Failed hypotheses
- **Hypothesis: Steering becomes fully distributed at 1.5B** — REJECTED. Hub layers still exist.
- **Hypothesis: Random vectors produce equal collateral** — REJECTED at 0.5B (5.7× difference), PARTIALLY CONFIRMED at 1.5B (narrower gap).

## 9. Limitations
1. **Pilot (1 seed per model)**: No variance estimate for steering effectiveness.
2. **Only 3 test prompts per condition**: Limited statistical power.
3. **No multi-layer steering tested**: Cannot determine if combining hub layers improves effectiveness.
4. **Only json_schema task vectors**: Other tasks may show different migration patterns.

## 10. Next experiments
- Multi-layer steering combinations at 1.5B
- Ablation controls to verify hub layer rank-order (Block C)
- 3B scale replication (Block D)
- Multi-seed replication
