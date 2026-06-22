# P2-SURGERY-001: Adapter Surgery — Qwen2.5-0.5B

## Claim
LoRA adapter weights in Qwen2.5-0.5B show monotonically increasing norms from early to late layers (0.16→0.28 for json_formatting), with ablation effects concentrated in late layers (layer 23: 9.84, layer 22: 9.22 for json_schema). Rank truncation reveals near-uniform singular value spectra (all ~0.54-0.64), requiring all 8 ranks for full norm recovery. This confirms that adapters primarily modify late-layer representations and are not easily compressible.

## Result

### Adapter norm profile (json_formatting, selected layers)

| Layer | Total Norm | Q Norm | K Norm | V Norm | O Norm |
|-------|-----------|--------|--------|--------|--------|
| 0 | 0.166 | 0.103 | 0.033 | 0.040 | 0.119 |
| 6 | 0.174 | 0.109 | 0.041 | 0.041 | 0.123 |
| 12 | 0.189 | 0.115 | 0.044 | 0.049 | 0.134 |
| 18 | 0.240 | 0.146 | 0.060 | 0.065 | 0.169 |
| 23 | 0.278 | 0.171 | 0.061 | 0.068 | 0.199 |

### Ablation effect by layer (json_formatting adapter)

| Layer | Ablation Effect (KL) |
|-------|---------------------|
| 0 | 0.000004 |
| 5 | 0.000401 |
| 10 | 0.028 |
| 15 | 2.335 |
| 20 | 6.274 |
| 23 | 9.843 |

### Rank truncation (layer 0, k_proj lora_A)

| Ranks | Explained Norm |
|-------|---------------|
| 1 | 13.3% |
| 2 | 26.4% |
| 4 | 51.7% |
| 6 | 76.4% |
| 8 | 100% |

## Controls
- 4 adapters tested: json_formatting, factual_recall, delimiter_tracking, code_semantics
- All show similar late-concentration pattern
- Rank truncation serves as compressibility control

## Seeds

| Seed | Status |
|------|--------|
| 1 | complete |

## Artifacts
- Raw output: `experiments/results/adapter_surgery.json`
- Run ID: `P2_SURGERY_qwen05b_20260622_133047_seed1`
- LoRA config: r=8, alpha=16, target=[q,k,v,o_proj], lr=0.0002, steps=100

## Interpretation
Adapters primarily modify late-layer representations (layers 19-23 carry 95%+ of ablation effect). The monotonically increasing norms suggest adapters "write" more information into later layers where representations are more task-specific. The near-uniform singular value spectrum means LoRA rank cannot be reduced without proportional norm loss.

## Limitations
1. Single seed, single model.
2. Only 4 adapters tested — may not generalize to all skills.
3. No compatibility matrix eval (cross-adapter interactions not measured).

## Verdict
confirmed
