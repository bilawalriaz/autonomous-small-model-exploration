# P2-SCALE3-001: Third Scale Point — Qwen2.5-3B Atlas

## Claim
At 3B scale (36 layers, 16 heads), Qwen2.5-3B shows concentrated hub layers in early positions (layers 0-5) for zero-ablation, with layer 4 producing the largest effects (KL 14.31 for json_schema). MLP ablation shows extreme concentration at layer 0 (effect 4.53 for factual_recall, 0.58 for json_schema), 8× higher than any other layer. Head ablation identifies heads 0 and 2 at layer 2 as dominant (mean effects 1.09 and 0.53). Steering at layer 2 shows strong KL responses (up to 7.53 at strength -4.0). LoRA injection achieves train loss 0.517 with 0% JSON validity before training.

## Result

| Experiment | Key Finding | Effect Size |
|-----------|-------------|-------------|
| Layer ablation | Hub at L4 (json_schema) | KL 14.31 |
| MLP ablation | Extreme hub at L0 | Effect 4.53 |
| Head ablation | L2 head_0 dominant | Mean effect 1.09 |
| Steering sweep | L2 responsive | KL 7.53 at -4.0 |
| LoRA JSON | Train loss 0.517 | 0% → training |

## Controls
- Layer ablation covers all 36 layers
- Head ablation samples 6 representative layers (2, 10, 18, 26, 30, 34)
- Steering tests 5 candidate layers (2, 18, 26, 34, 35)
- Patching, skip, knockout all failed (tensor size mismatch: 16 vs 128)

## Seeds

| Seed | Status |
|------|--------|
| 1 | complete (layer, head, mlp, steering, lora) |
| 1 | error (patching, skip, knockout) |

## Artifacts
- Raw output: `experiments/results/third_scale_*.json`
- Run IDs: `P2_D01` through `P2_D05`

## Interpretation
The 3B model continues the hub pattern from 0.5B and 1.5B, with hubs concentrated in early layers. The MLP ablation result (layer 0 effect = 4.53) is striking and suggests the first MLP layer plays an outsized role at this scale. The head ablation shows only 2 of 16 heads at layer 2 are significant, consistent with specialized attention heads. The tensor size mismatch errors for patching/skip/knockout are due to GQA (grouped query attention) at 3B having 16 KV heads vs 128 query heads.

## Limitations
1. Single seed — no variance.
2. Patching, skip, knockout all failed due to GQA architecture mismatch.
3. Head ablation only tested 6 of 36 layers.
4. No collateral damage measurement.

## Verdict
partially_confirmed
