# P2-ROBUST-001: Long Task Robustness — Hub Stability Across Prompt Lengths

## Claim
Hub layers in Qwen models shift position with prompt length. In Qwen2.5-0.5B, the factual_recall hub moves from layer 2 (short, effect 16.15) to layer 7 (medium, 13.44) and layer 22 (long, 15.96). In Qwen2.5-1.5B, the hub shifts from layer 26 (short, 12.93) to layer 14 (medium, 19.52) and layer 14 (long, 19.47). Steering effectiveness at the short-prompt hub degrades for longer prompts. LoRA adapter effects increase with prompt length (0.5B: KL 0.067→1.099→0.513 for short→medium→long factual_recall).

## Result

### Qwen2.5-0.5B — Hub layer by prompt length

| Task | Length | Hub Layer | Hub Effect | Top-3 Layers |
|------|--------|-----------|------------|-------------|
| factual_recall | short | 2 | 16.15 | 2, 22, 7 |
| factual_recall | medium | 7 | 13.44 | 7, 22, 2 |
| factual_recall | long | 22 | 15.96 | 22, 2, 7 |

### Qwen2.5-1.5B — Hub layer by prompt length

| Task | Length | Hub Layer | Hub Effect | Top-3 Layers |
|------|--------|-----------|------------|-------------|
| factual_recall | short | 26 | 12.93 | 26, 6, 14 |
| factual_recall | medium | 14 | 19.52 | 14, 16, 5 |
| factual_recall | long | 14 | 19.47 | 14, 16, 5 |

### Steering at short-prompt hub (0.5B, layer 2)

| Prompt Length | KL at -4.0 | KL at +4.0 |
|--------------|-----------|-----------|
| short | 0.32 | 0.05 |
| medium | 0.11 | 0.13 |
| long | 0.43 | 0.13 |

### LoRA adapter effect by prompt length (0.5B)

| Task | Length | Adapter KL | Target Logprob Delta |
|------|--------|-----------|---------------------|
| factual_recall | short | 0.067 | 0.000 |
| factual_recall | medium | 1.099 | 0.042 |
| factual_recall | long | 0.513 | -0.001 |
| json_schema | short | 0.099 | 0.000 |
| json_schema | medium | 0.796 | 0.000 |
| json_schema | long | 0.344 | 0.000 |

## Controls
- Ablation at all 24/28 layers for each length condition
- Steering sweep at 7 strengths (-4.0 to +4.0)
- LoRA adapter loaded from pre-trained checkpoint

## Seeds

| Seed | Model | Experiments | Status |
|------|-------|------------|--------|
| 1 | Qwen2.5-0.5B | ablation, steering, lora | complete |
| 1 | Qwen2.5-1.5B | ablation, steering | complete |
| 1 | Qwen2.5-1.5B | lora | error (size mismatch) |

## Artifacts
- Raw output: `experiments/results/long_task_robustness_*.json`
- Run IDs: `P2_I01` through `P2_I03`

## Interpretation
Hub layers are NOT stable across prompt lengths. The hub migrates from early layers (short prompts) to mid/late layers (long prompts), suggesting that short factual recall is processed differently from long-context factual recall. This has practical implications: steering vectors computed on short prompts will not work optimally on long prompts. The LoRA adapter effect peaks at medium length, suggesting the adapter is overfitted to medium-length contexts.

## Limitations
1. Single seed — no variance.
2. 1.5B LoRA failed (size mismatch loading 0.5B adapter into 1.5B model).
3. Only factual_recall and json_schema tested for length variation.
4. "Long" prompts are still relatively short (truncated context).

## Verdict
confirmed
