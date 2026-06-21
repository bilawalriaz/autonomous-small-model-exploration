# Decision Log

## Decision D001: Use HF native hooks instead of TransformerLens

Choice:
Use HuggingFace Transformers with manual forward hooks for ablation/patching instead of TransformerLens.

Reason:
TransformerLens had compatibility issues with Qwen2.5-0.5B architecture (GQA attention). HF native hooks work reliably and give full control.

Alternatives considered:
- TransformerLens (preferred by prompt but didn't support Qwen2.5 cleanly)
- NNsight (more complex setup, same hooking mechanism)

Cost:
More boilerplate code. No built-in activation patching utilities.

Revisit when:
TransformerLens adds Qwen2.5 support.

---

## Decision D002: LoRA instead of full SFT for training perturbations

Choice:
Use LoRA adapters (r=8, alpha=16, all-linear targets) for all training experiments instead of full fine-tuning.

Reason:
Full SFT OOMs on 8GB VRAM even with bf16 + gradient checkpointing. LoRA uses ~1GB for optimizer states vs ~4GB for full SFT.

Alternatives considered:
- Full SFT with gradient accumulation (still OOMs)
- CPU offloading (too slow for iteration)
- Quantized training (not supported by TRL SFTTrainer at the time)

Cost:
LoRA may produce different internal changes than full SFT. Results may not transfer.

Revisit when:
Running on larger GPU (>=16GB VRAM).

---

## Decision D003: Zero ablation instead of mean ablation

Choice:
Zero out component outputs during ablation (set to zeros) rather than replacing with mean activation.

Reason:
Simpler to implement. Mean ablation requires computing mean activations across a dataset first, adding a preprocessing step.

Alternatives considered:
- Mean ablation (more principled, less likely to create distribution shift)
- Resample ablation (replace with activation from different prompt)

Cost:
Zero ablation creates out-of-distribution activations that may cause cascade effects. Ablation effects may be overestimated.

Revisit when:
Implementing resample ablation for stronger causal claims.

---

## Decision D004: Single seed for all experiments

Choice:
All experiments use a single random seed (default PyTorch seed). No multi-seed replication yet.

Reason:
8GB VRAM budget limits throughput. Running 3 seeds would triple experiment time.

Alternatives considered:
- 3-seed replication (gold standard but 3x cost)
- Seed sweep on subset (partial solution)

Cost:
Results may not be robust to seed variation. Confidence levels capped at "medium" until replicated.

Revisit when:
Key findings need to be strengthened for publication. Run 3-seed replication on top 5 findings.

---

## Decision D005: Short synthetic prompts for all experiments

Choice:
Use short synthetic prompts (5-15 tokens) rather than natural language or longer contexts.

Reason:
Cleaner interpretability. Easier to align clean/corrupt pairs. Faster inference. Less confounded by context length effects.

Alternatives considered:
- Natural language prompts (more realistic but harder to control)
- Long-context tasks (OOM risk, harder to analyze positions)

Cost:
Results may not transfer to longer contexts or natural language. Position-specific effects may differ with longer inputs.

Revisit when:
Core findings validated on short prompts, extend to longer contexts.

---

## Decision D006: Aero as primary compute host

Choice:
All GPU experiments run on aero (RTX 2070 Super 8GB) via SSH. No micro usage.

Reason:
Micro has no GPU. Aero has CUDA-capable RTX 2070 Super with 8GB VRAM, sufficient for 0.5B model inference and LoRA training.

Alternatives considered:
- Cloud GPU (costs money)
- CPU-only on micro (too slow)
- Larger local GPU (not available)

Cost:
8GB VRAM limits batch size and model size. Cannot run full SFT. Cannot run larger models.

Revisit when:
Access to >=16GB VRAM GPU.

---

## Decision D007: Bundle-based GitHub push from aero

Choice:
Use git bundle + scp to push from aero (no SSH key) to micro (has gh auth) to GitHub.

Reason:
Aero has no GitHub SSH key or gh CLI auth configured. Micro has gh auth.

Alternatives considered:
- Configure gh auth on aero (requires token management)
- Direct push from aero (SSH key not registered)

Cost:
Manual push workflow. Not automated.

Revisit when:
Setting up gh auth on aero with a deploy key.
