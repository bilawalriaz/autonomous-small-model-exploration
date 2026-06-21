# Decision Log

## Decision D001

Choice: Start with qwen3.5-0.5b as primary target, fall back to GPT-2 small or Pythia if TransformerLens support is awkward.

Reason: Research prompt specifies 0.5B-class model. Qwen3.5-0.5B is the target. Fallback ensures progress even if architecture support is incomplete.

Alternatives considered: Skip straight to GPT-2 small (always supported), Phi-3-mini (too large at 3.8B).

Cost: Time spent debugging TransformerLens compatibility.

Revisit when: First model load attempt fails or TransformerLens conversion is unsupported.

---

## Decision D002

Choice: Run initial experiments on micro (CPU-only). Move training to aero (RTX 2070 Super).

Reason: Micro is always-on. Inference and ablation can be slow on CPU but work. Training requires GPU.

Alternatives considered: Run everything on aero (requires SSH), run on cloud GPU (costs money).

Cost: Slower iteration for inference-heavy phases.

Revisit when: CPU inference is unbearably slow for the ablation sweep.
