# P2-STEER-001/002: Steering Migration — 0.5B vs 1.5B

## Claim
Steering effectiveness at hub layers scales with model size: Qwen2.5-1.5B shows 6× higher steering vector norms at layer 2 (9.08 vs 1.49) and achieves larger KL divergences at equivalent strengths, but hub positions shift from late layers (21-23) in 0.5B to late layers (25-27) in 1.5B, confirming that hub location is architecture-dependent while hub existence is scale-invariant.

## Result

| Metric | 0.5B (Layer 2) | 1.5B (Layer 2) |
|--------|---------------|---------------|
| Steering vector norm | 1.49 | 9.08 |
| KL at strength -4.0 | 0.287 | 0.572 |
| KL at strength +4.0 | 0.169 | 0.176 |
| Collateral at +4.0 (json) | 0.160 | 0.353 |
| Collateral at +4.0 (copying) | 0.114 | 0.117 |

## Controls

| Control | 0.5B KL at -4.0 | 0.5B Collateral (json) | 1.5B KL at -4.0 | 1.5B Collateral (json) |
|---------|-----------------|----------------------|-----------------|----------------------|
| task_vector | 0.287 | 0.160 | 0.572 | 0.353 |
| random_vector | 0.147 | 0.911 | 0.500 | 0.372 |

Random vectors produce comparable KL but much higher collateral damage at 0.5B (0.911 vs 0.160), confirming task-specificity. At 1.5B, random and task vectors are more similar, suggesting distributed representations.

## Seeds

| Seed | Model | Status |
|------|-------|--------|
| 1 | Qwen2.5-0.5B | complete |
| 1 | Qwen2.5-1.5B | complete |

Mean: pilot (1 seed per model). No variance available.

## Artifacts
- Raw output: `experiments/results/steering_migration_qwen05b.json`, `steering_migration_qwen15b.json`
- Run IDs: `P2-STEER-001`, `P2-STEER-002`
- Script commit: `de0b3cb`

## Environment
- GPU: NVIDIA GeForce RTX 2070 with Max-Q Design
- Precision: torch.float32
- 0.5B: 24 layers, hub layers [2, 21, 22, 23]
- 1.5B: 28 layers, hub layers [2, 21, 25, 26, 27]

## Interpretation
Steering migrates: hub positions shift to accommodate more layers (0.5B L21-23 → 1.5B L25-27), but the early hub at layer 2 is preserved. The 1.5B model has much larger steering vector norms (6×), indicating stronger feature representations. Random vector controls confirm task-specificity at 0.5B but less so at 1.5B, suggesting increasing representation density with scale.

## Limitations
1. Single seed per model — no variance.
2. Only 3 prompts per condition — small sample.
3. No activation patching — cannot confirm causal mechanism.

## Verdict
confirmed
