# LFM2.5-8B-A1B: Phase 1 Results Summary

## Architecture
- 24 layers: 18 conv + 6 attention + 22 MoE FFN + 2 dense FFN
- 8.3B total / 1.5B active per token
- 32 experts, 4 active per layer
- Q4_K_M GGUF: 4.9GB on aero RTX 2070 Super

## Capability Benchmark (95 tasks)
- **Overall: 63.2% (60/95)**
- Best: arithmetic 100%, code debug 100%, science 86%, geography 82%
- Worst: code gen 29%, biology 25%, physics 25%, astronomy 0%
- Interpretation: strong factual recall, weak on longer-form generation

## Sensitivity Analysis (53 tests)
- **All tests passed (53/53)**
- Diversity scores ALL low (0.10-0.20) — model is robust to perturbation
- Temperature sweep: diversity=0.12 — stable across temps
- Character swaps: diversity=0.10 — very robust
- Semantic perturbation: diversity=0.17 — paraphrase-resistant
- Context length: diversity=0.20 — most sensitive but still low
- Average speed: 28-52 tok/s depending on prompt complexity

## Routing Probe (20 prompts)
- Speed: 122-131 tok/s across all categories
- All categories produced coherent output
- No routing failures or anomalies observed

## Key Findings
1. Model is highly stable — perturbation barely affects routing
2. Code generation is the primary weakness (29%)
3. Strong factual/arithmetic performance
4. Conv layers dominate (75% of layers) — same as 230M finding
5. Dense-first design (layers 0-1) prevents routing overhead on raw embeddings
