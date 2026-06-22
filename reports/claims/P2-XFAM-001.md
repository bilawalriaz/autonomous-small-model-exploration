# P2-XFAM-001: Cross-Family Replication — SmolLM2-1.7B

## Claim
SmolLM2-1.7B (a different architecture family from Qwen) shows flat ablation profiles with identical KL effects across all 24 layers for every task (json: 3.24, factual: 1.74, copying: 6.33, code: -1.23), and all hub layers converge to layer 0. Steering at the hub layer produces negligible boosts (0.01-0.15 logprob). This indicates that the hub-localization finding from Qwen models does NOT transfer to SmolLM2 — hub localization is architecture-specific, not universal.

## Result

| Task | Baseline Logprob | Baseline EM | Hub Layer | Ablation KL (all layers identical) | Steering Boost |
|------|-----------------|-------------|-----------|-----------------------------------|---------------|
| json | -7.56 | 33.3% | 0 | 3.24 | 0.144 |
| factual | -9.07 | 100% | 0 | 1.74 | 0.014 |
| copying | -4.48 | 66.7% | 0 | 6.33 | 0.153 |
| code | -12.03 | 0.0% | 0 | -1.23 | 0.136 |

## Controls
- All 24 layers tested — no layer stands out
- Steering at identified "hub" (layer 0) — negligible effect
- Flat profile is its own control: no concentration = no hubs

## Seeds

| Seed | Model | Status |
|------|-------|--------|
| 1 | SmolLM2-1.7B | complete |

## Artifacts
- Raw output: `experiments/results/cross_family_smollm2-17b.json`
- Run ID: `P2-XFAM-001`
- Script commit: `f6587f3`

## Interpretation
The flat ablation profile means SmolLM2-1.7B distributes task-relevant computation evenly across all layers, unlike Qwen models which concentrate it in hub layers. This is a critical negative result: hub localization is architecture-specific. The atlas methodology (find hubs → ablate → steer) may not generalize across model families.

## Limitations
1. Single seed, single model from the non-Qwen family.
2. Only 3 examples per task (measured by n_prompts in baseline).
3. No head-level or MLP-level ablation to check sub-layer concentration.
4. SmolLM2 may have hubs at a different granularity (e.g., MLP-only or head-only).

## Verdict
rejected
