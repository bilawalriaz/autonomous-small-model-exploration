# Negative Results

## NR001: Full SFT OOMs on 8GB VRAM

Experiment:
Attempted full supervised fine-tuning of Qwen2.5-0.5B on JSON schema data.

Expected:
Model + optimizer + gradients fit in 8GB VRAM with bf16 and gradient checkpointing.

Observed:
OOM at first training step. Model in bf16 ~1GB. Optimizer states (AdamW, 2x model size) ~2GB. Gradients ~1GB. Activations ~2GB. Total ~6GB + CUDA overhead > 8GB.

Interpretation:
0.5B model with AdamW optimizer needs ~6-7GB for training. 8GB is marginal and OOMs due to CUDA fragmentation and activation memory.

What this rules out:
Full fine-tuning as a viable approach on 8GB VRAM for 0.5B models.

Next:
Use LoRA (r=8) which reduces optimizer states to ~50MB. All training experiments use LoRA.

---

## NR002: Full-residual activation patching gives KL=0 everywhere

Experiment:
Patch full residual stream at each layer from clean run into corrupt run.

Expected:
Some layers show high recovery (KL drop), others show low recovery. Differential patching reveals information flow.

Observed:
KL=0 at every layer. Patching the full residual at any layer completely overwrites the corrupt signal, making the patched model identical to the clean model.

Interpretation:
Full-residual patching is trivial for clean/corrupt pairs with identical prefixes. The residual stream at any layer carries the full computation state, so replacing it restores the clean computation regardless of which layer.

What this rules out:
Full-residual patching as a useful technique for these prompt pairs. Need position-specific or component-specific patching.

Next:
Position-specific patching (implemented in exp_000018). Component-specific patching (head/MLP outputs).

---

## NR003: Clean/corrupt pair v0 tokenization misalignment

Experiment:
Initial clean/corrupt pairs (pairs_v0.json) had multi-token targets.

Expected:
Target token is a single token that can be scored directly.

Observed:
Some targets encoded as 2+ tokens (e.g., "Paris" -> ["Par", "is"]). Logprob scoring only uses first token, missing the actual answer.

Interpretation:
Tokenizer splits some targets into multiple tokens. Need to verify all targets are single-token before using them for logprob comparison.

What this rules out:
Using arbitrary string targets without tokenizer verification.

Next:
Built pairs_v1.json with verified single-token targets (exp_000012).

---

## NR004: H002 (universal L0-L2 concentration) rejected

Experiment:
Dataset shard ablation — train LoRA adapters on 5 skill families, compare component maps.

Expected:
All skill families concentrate into L0-L2 after LoRA training (extrapolating from JSON results).

Observed:
Each family concentrates in different layers:
- factual_recall: L3, L16, L19
- code_semantics: L1, L10, L21
- json_schema: L6, L12, L13
- copying: dispersed (no clear concentration)
- delimiter_tracking: fully absorbed (0 ablation sensitivity)

Interpretation:
JSON's L0-L2 concentration was specific to format-heavy tasks, not a universal training effect. Different skills use different circuits and training writes to different locations.

What this rules out:
Universal "training concentrates in early layers" hypothesis. Each skill must be analyzed individually.

Next:
Train more adapters on sub-families. Test if concentration pattern is stable across seeds.

---

## NR005: Extreme steering causes degeneration

Experiment:
Steering sweep at L2 with factual direction, strengths -8 to +8.

Expected:
Monotonic improvement with positive steering strength.

Observed:
At s>=+2, model generates Chinese characters and repetitive garbage. At s=+4, factual recall is 3.3x better but generation quality degrades. At s=+8, output is incoherent.

Interpretation:
Activation addition at extreme strengths pushes the model out of distribution. The steering vector direction is useful but the magnitude must be carefully controlled.

What this rules out:
Unbounded steering as a reliable intervention. Need to find and respect the steering budget.

Next:
Test steering at finer granularity (s=0.25, 0.5, 0.75, 1.0). Test steering with norm-constrained vectors.

---

## NR006: Position-specific patching shows L2 is position-dependent, not uniform

Experiment:
Position-specific ablation at L2 across 11 tasks.

Expected:
L2 effect is uniform across all positions (supporting "universal hub" hypothesis).

Observed:
L2 effect concentrated at first (3.34 mean) and last (5.03 mean) positions. Operator tokens have near-zero effect (-0.09 mean). Content tokens moderate (0.79 mean).

Interpretation:
L2 is not a uniform processing layer. It specifically handles instruction/first tokens and prediction/last tokens. This is more nuanced than "universal hub."

What this rules out:
L2 as a simple residual magnitude carrier. It has positional specialization.

Next:
Test if L2's positional pattern changes after LoRA training. Test if L2 first-position effect is instruction-specific.
