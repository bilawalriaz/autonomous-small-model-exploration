# Negative Results

## NR001: Full SFT OOMs on 8GB VRAM
Experiment: Attempted full supervised fine-tuning of Qwen2.5-0.5B on JSON schema data.
Expected: Model + optimizer + gradients fit in 8GB VRAM with bf16 and gradient checkpointing.
Observed: OOM at first training step.
Interpretation: 0.5B model with AdamW optimizer needs ~6-7GB for training. 8GB is marginal.
Next: Use LoRA (r=8) which reduces optimizer states to ~50MB.

---

## NR002: Full-residual activation patching gives KL=0 everywhere
Experiment: Patch full residual stream at each layer from clean run into corrupt run.
Expected: Some layers show high recovery, others low.
Observed: KL=0 at every layer — full-residual patching trivially restores clean computation.
Next: Position-specific and component-specific patching needed.

---

## NR003: Clean/corrupt pair v0 tokenization misalignment
Experiment: Initial clean/corrupt pairs had multi-token targets.
Expected: Target token is single-token.
Observed: Some targets encode as 2+ tokens. Logprob scoring only uses first token.
Next: Built pairs_v1.json with verified single-token targets.

---

## NR004: H002 (universal L0-L2 concentration) rejected
Experiment: Dataset shard ablation — 5 skill families.
Expected: All families concentrate into L0-L2 after LoRA training.
Observed: Each family concentrates in different layers (factual_recall: L3/16/19, code: L1/10/21, json: L6/12/13).
Next: Skill-specific analysis required.

---

## NR005: Extreme steering causes degeneration
Experiment: Steering sweep at L2 with factual direction, strengths -8 to +8.
Expected: Monotonic improvement with positive steering.
Observed: At s>=+2, model generates Chinese characters and repetitive garbage.
Next: Find and respect the steering budget.

---

## NR006: L2 is position-dependent, not uniform
Experiment: Position-specific ablation at L2.
Expected: L2 effect is uniform across all positions.
Observed: L2 specifically handles first tokens (instruction) and last tokens (prediction). Operators near-zero.
Next: L2 has positional specialization, not simple residual magnitude.

---

## NR007: JSON skill knockout had limited effect
Experiment: Skill knockout via negative steering on JSON skill.
Expected: Negative steering suppresses JSON-specific token probabilities.
Observed: Base probability of JSON targets already near-zero — no room to suppress.
Next: Use prompts where adapter demonstrably changes target probability.

---

## NR008: H6 (upstream propagation) rejected
Experiment: Adapter-only ablation — remove adapter contribution at each layer.
Expected: Adapter ablation at early layers would have large effects (upstream propagation).
Observed: Norm-effect correlation = 0.85. Adapter effects ARE at same layers where norms peak (L19-L23).
Next: The separation between general importance (L0-L2) and adapter-specific importance (L19-L23) is the finding.

---

## NR009: PeftModel.from_pretrained modifies base model in-place
Experiment: Loading base and trained models as separate objects.
Expected: Two independent models.
Observed: PeftModel wraps and modifies base model in-place. Both objects produce adapter-active results.
Next: Use `disable_adapter()` context manager.

---

## NR010: SmolLM2 shows flat ablation profile — no identifiable hub
Experiment: Layer ablation on SmolLM2-1.7B.
Expected: Clear hub layer similar to Qwen2.5.
Observed: All 24 layers show IDENTICAL ablation effects within each family.
Next: Architecture-specific hub analysis required.

---

## NR011: 3B patching/skip/knockout fail with tensor dimension mismatch
Experiment: Cross-model patching on Qwen2.5-3B.
Expected: Same experiments as 0.5B/1.5B.
Observed: RuntimeError — GQA dimension mismatch (4 KV heads vs 14).
Next: Read head dimensions from model.config. Parameterize all code.

---

## NR012: 1.5B LoRA adapter cannot load into different-scale model
Experiment: Cross-scale adapter loading.
Expected: Adapter would produce measurable effects.
Observed: State dict size mismatch — A matrix dimensions match d_model per scale.
Next: Train separate adapters per scale.

---

## NR013: Gaussian resample preserves layer ranking
Experiment: Six ablation methods compared at 0.5B and 1.5B.
Expected: Different methods might change rank ordering.
Observed: Zero ≈ mean (mean activation ≈ 0). Gaussian resample preserves ranking with higher variance.
Next: Use zero ablation for efficiency.

---

## NR014: Mock judge produces plausible-looking but meaningless scores (Phase 9R)

Experiment:
Phase 9 eval pipeline used mock judge (deterministic random numbers seeded by eval_id) when API was unavailable. The pipeline silently fell back to mock without flagging it.

Expected:
Mock scores would be clearly distinguishable from real judge scores.

Observed:
Mock scores look plausible (1-5 range, some variation by category). The HTML report presented mock-judge win-rates and judge scores as if they were real behavioral evidence. Python's `hash()` was used for seeding — non-deterministic across Python versions (PYTHONHASHSEED randomization).

Interpretation:
Mock-judge scores are structurally indistinguishable from real scores in aggregate. The silent fallback was a pipeline integrity bug.

What this rules out:
Using mock judge without explicit flagging. Using Python hash() for deterministic seeding. Silent fallback from API to mock.

Next:
Phase 9R fixed this: explicit `--mock` flag required, `judge_source` metadata on every score, `hashlib`-based seeding, `--strict-report-mode`. All existing mock-judge results caveated as "pipeline validation only, not behavioural evidence."

---

## NR015: bad_format_control has unexpectedly low training loss (Phase 9)

Experiment:
Content-controlled format ablation on LFM2.5-230M. Bad-format-control (deliberately verbose, filler-heavy data) achieved 2nd-best training loss (1.402).

Expected:
Bad-format-control should have the worst or near-worst loss as a negative control.

Observed:
Loss rank: multi_turn_verbose (1.372) > bad_format_control (1.402) > multi_turn_concise (1.516) > alpaca_flat (1.732) > single_turn_chat (1.748) > structured_terse (1.831).

Interpretation:
Predictable, repetitive text is inherently easy for language models to predict. Low loss reflects statistical regularity, not quality. Strongest training-loss-only evidence that loss ≠ quality.

What this rules out:
Using training loss as a proxy for output quality without behavioral verification.

Next:
Run real eval to confirm: if bad_format_control has worst behavioral scores despite 2nd-best loss, H-P9-4 is confirmed.
