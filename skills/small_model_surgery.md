# Small Model Surgery

**Version:** 3.0.0
**Last updated:** 2026-06-23
**Status:** Evidence-backed rules with confidence levels

---

## What This Is

A set of practical rules for training, steering, compressing, and deploying small language models (<3B params) based on mechanistic interpretability evidence. Each rule has a confidence level, the models it was tested on, and known failure cases.

---

## Rule 1: Each Model Needs Its Own Atlas

**Confidence:** HIGH
**Evidence:** 3 Qwen2.5 scales + SmolLM2 cross-family
**Models:** Qwen2.5-0.5B/1.5B/3B, SmolLM2-1.7B

The "universal processing hub" — the layer most responsible for routing information — moves dramatically with model scale and architecture:

- Qwen2.5-0.5B: L2 (8% depth)
- Qwen2.5-1.5B: L14 (50% depth) — REVISED from L26 (93%) by Phase 3 multi-seed with full 12-family suite
- Qwen2.5-3B: L34 (94% depth)
- SmolLM2-1.7B: L0 (0% depth)

**Implication:** You cannot transfer layer-targeting knowledge from one model to another. A 0.5B atlas is useless for a 1.5B model. Run the atlas per model.

**Failure case:** Only Qwen2.5 family and SmolLM2 tested. Other architectures may behave differently.

---

## Rule 2: Target Late Layers for LoRA

**Confidence:** MEDIUM-HIGH
**Evidence:** Adapter-only ablation across 3 scales, cross-model patching
**Models:** Qwen2.5-0.5B/1.5B/3B

LoRA adapter effects concentrate at the final ~10% of layers:
- 0.5B: L19-L23 (out of 24)
- 1.5B: L25-L27 (out of 28)
- 3B: L31-L35 (out of 36)

This is where the adapter norms peak AND where functional effects are strongest. The correlation between norm and effect is 0.85 (strong positive).

**Implication:** When doing LoRA, focus on the last 10% of layers. Don't waste parameters on early layers.

**Failure case:** Only tested with LoRA. Full SFT may write to different layers. The "final 10%" may partly be a proximity-to-output artifact.

---

## Rule 3: o_proj Is the Most Efficient Injection Module

**Confidence:** MEDIUM
**Evidence:** Module sweep on 0.5B, JSON family
**Models:** Qwen2.5-0.5B

Module efficiency for JSON skill injection:
- o_proj: +3.64 boost with 344K params (10.6 per param)
- v_proj: +2.75 with 197K params (13.9 per param)
- MLP: +1.92 with 3.3M params (0.6 per param)

o_proj writes directly to the residual stream, making it the most direct injection pathway.

**Implication:** Use o_proj-only LoRA for surgical skill injection. 10x more parameter-efficient than MLP.

**Failure case:** Only tested on JSON at 0.5B. May not hold for factual recall or code tasks. Needs multi-family and multi-scale replication.

---

## Rule 4: Steering Migrates, Not Collapses

**Confidence:** HIGH
**Evidence:** Steering sweeps at 0.5B/1.5B/3B
**Models:** Qwen2.5-0.5B/1.5B/3B

Steering effectiveness does not decrease with model size. It MIGRATES to different layers and gets STRONGER:

- 0.5B best: L8 delta=1.32, L12 delta=2.16
- 1.5B best: L21 delta=4.64 (3.5x stronger), L26 delta=3.54

The steering budget at 1.5B is ~3x larger (vector norm 81.39 vs 1.49 at 0.5B L2).

**Implication:** Always test ALL candidate hub layers, not just the smaller model's hub. Steering at the wrong layer looks like "steering failure" when it's really "wrong layer."

**Failure case:** No random-vector controls yet (Phase 3 experiment C4). The effect may not be task-specific.

---

## Rule 5: 4-bit NF4 Beats 8-bit for Deployment

**Confidence:** MEDIUM-HIGH
**Evidence:** Speed and quality benchmarks at 0.5B and 1.5B
**Models:** Qwen2.5-0.5B/1.5B

BitsAndBytes 8-bit is actually SLOWER than 4-bit NF4:
- 1.5B: bf16=18.8 tok/s, 8bit=9.0 tok/s (52% slower), 4bit=17.1 tok/s (9% slower)
- Constraint adherence: 4-bit approximately equals bf16

The reason: 8-bit uses row-wise quantization with online dequantization per matmul. 4-bit NF4 uses block-wise quantization with pre-computed dequant tables.

**Implication:** Use 4-bit NF4 as default for 1.5B+ models. 8-bit is worse in every metric.

**Failure case:** Causal surface drift not yet measured. 4-bit may preserve benchmarks while changing steering/ablation behavior (Phase 3 experiment Q1-Q3).

---

## Rule 6: All Layers Are Necessary

**Confidence:** HIGH (negative result)
**Evidence:** Layer skipping across 3 scales
**Models:** Qwen2.5-0.5B/1.5B/3B

Naive layer skipping (zeroing a layer's output) gives 0% top-5 token overlap for ALL skip configs tested. Every layer is necessary for maintaining output quality.

**Implication:** Don't skip layers for inference speed. The efficiency gains are in training (early lock-in), not inference (layer skipping).

**Failure case:** Atlas-guided skip + recovery finetune has not been tested (Phase 3 experiment G4).

---

## Rule 7: Core Circuit Locks in at 10% of Training

**Confidence:** MEDIUM
**Evidence:** Checkpoint timeline on 0.5B, JSON family
**Models:** Qwen2.5-0.5B

The core processing circuit (L2/L7/L9 for JSON) stabilizes within the first 10% of training. Loss continues dropping from 0.587 to 0.062 while the core circuit is frozen.

**Implication:** Save checkpoints at 10% intervals. If the core circuit is stable by step 10/100, you can stop early for basic skill acquisition.

**Failure case:** Only tested with JSON on 0.5B. Different tasks may have different lock-in points. Phase 3 experiment G3 tests at 1.5B.

---

## Rule 8: Zero Ablation = Mean Ablation at Hub Layers

**Confidence:** MEDIUM-HIGH
**Evidence:** Ablation controls at 0.5B and 1.5B
**Models:** Qwen2.5-0.5B/1.5B

At hub layers, zeroing and replacing with the mean activation give identical KL divergence. This is because hub activations have near-zero mean — the mean IS effectively zero in the relevant subspace.

Gaussian resample ablation IS less destructive (~40% reduction), suggesting hub activations have structure beyond just magnitude.

**Implication:** Zero ablation is a valid methodology for hub identification. It's not "more destructive" than mean at hub layers.

**Failure case:** May be specific to hub layers. Non-hub layers may show different zero/mean equivalence. Phase 3 experiment C1 tests this.

---

## Rule 9: Skills Are Separable to Different Degrees

**Confidence:** LOW-MEDIUM
**Evidence:** Skill separability benchmark on 0.5B
**Models:** Qwen2.5-0.5B

Skill separability scores (SSS):
- code_semantics: 0.36 (most separable — localized, steerable)
- json_schema: 0.22
- factual_recall: 0.22
- copying: 0.22
- delimiter_tracking: 0.15 (least separable)

**Implication:** Code skills are the best candidates for surgical editing. JSON and factual skills are more entangled with general processing.

**Failure case:** Zero collateral damage across all skills is suspicious (metric may not be sensitive enough). Needs validation with actual downstream task accuracy.

---

## Rule 10: Steering Vectors Are Task-Specific (PENDING)

**Confidence:** UNKNOWN (under test)
**Experiment:** Phase 3 C4

The steering control experiment tests whether:
- Target task vectors produce stronger effects than random same-norm vectors
- Shuffled-label vectors are weaker than correct-label vectors
- Unrelated-task vectors produce different effects than target vectors

If random vectors give similar KL as task vectors, all steering findings collapse.

---

## How to Run the Atlas on a New Model

```bash
# 1. Verify model loads
python -c "from mi_atlas.model_loader import load_model; b=load_model('Qwen/Qwen2.5-0.5B')"

# 2. Run Phase 1 atlas (component mapping)
python scripts/run_full_atlas.py --model Qwen/Qwen2.5-0.5B --suffix 0.5b

# 3. Run Phase 2 blocks (reproducibility)
python scripts/run_full_phase2_atlas.py --model Qwen/Qwen2.5-0.5B --blocks all

# 4. Run Phase 3 gap closure
python scripts/run_full_phase3_atlas.py --model Qwen/Qwen2.5-0.5B --blocks all

# 5. Generate reports
python scripts/generate_publication_report.py
```

See `research/mi-atlas-experimentation` skill for the full 7-phase workflow.

---

## Known Gotchas

1. **PeftModel modifies base model in-place.** Use `model.disable_adapter()` for base behavior.
2. **Single-token targets required.** Always verify `len(tokenizer(target)["input_ids"]) == 1`.
3. **Full-residual patching is trivial** for identical-prefix pairs. Use component-specific patching.
4. **Zero ablation breaks residual stream.** Valid for hub identification, but interpret with care.
5. **GQA head dimensions differ across models.** 0.5B has d_head=64, 3B has d_head=128. Parameterize from config.
6. **8-bit is slower than 4-bit.** bitsandbytes overhead. Use 4-bit NF4.
7. **Steering at extreme strengths causes degeneration.** Test fine granularity (0.25, 0.5, 0.75, 1.0).
8. **Head effects are tiny in small models.** 200x smaller than layer effects at 0.5B.
9. **Early exit doesn't work naively.** Intermediate hidden states can't be projected through lm_head.
10. **Each model size has different internal structure.** Don't transfer atlases across scales.
