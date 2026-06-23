# Gems — Surprising, Reusable, or Practically Valuable Discoveries

Generated: 2026-06-23
A gem must satisfy >= 3 of: (1) replicates across seeds/prompts, (2) contradicts common assumption, (3) actionable training/inference rule, (4) causal not correlational, (5) differs across scale/family, (6) predicts downstream behavior, (7) saves compute/memory, (8) would surprise a competent finetuner.

---

## G01: Hub migrates to final ~10% of layers at scale

**Claim:** The universal processing hub moves from early layers (L2, 8% depth) at 0.5B to the final ~10% of layers at 1.5B (L26, 93%) and 3B (L34, 94%).

**Evidence:**
- Replicated across 3 Qwen2.5 scales (0.5B, 1.5B, 3B)
- Ablation controls confirm rank-order stability at 0.5B and 1.5B
- Hub is top layer for 9/12 task families at 3B

**Criteria met:**
- (1) Replicated across scales (not seeds — single seed per scale)
- (2) Contradicts assumption that early layers are always most important
- (3) Actionable: when targeting layers for steering/LoRA, use atlas data not heuristics
- (5) Differs across scale in a meaningful way
- (8) Would surprise a competent finetuner who assumes "early layers = routing"

**Failure cases:** Single seed per model. Only Qwen family tested. SmolLM2 hub at L0 contradicts the "migration" pattern.

**Practical implication:** Every model scale needs its own atlas. You cannot transfer layer-targeting knowledge from 0.5B to 1.5B. This means atlas-guided LoRA must be re-run per model.

**Confidence:** HIGH (for the migration pattern), MEDIUM (for the specific percentages)

**Next decisive experiment:** Multi-seed replication at 1.5B and 3B. If the hub moves by >2 layers across seeds, the pattern is fragile.

---

## G02: Steering migrated, not collapsed, at 1.5B

**Claim:** Phase 1 concluded "steering sensitivity drops 70x at 1.5B" — this was wrong. The Phase 1 team only tested at L2 (the 0.5B hub). When testing at the correct 1.5B hub layers, steering is actually STRONGER at 1.5B than 0.5B.

**Evidence:**
- 0.5B best single-layer: L8 delta=1.32, L12 delta=2.16
- 1.5B best single-layer: L21 delta=4.65 (3.5x stronger than 0.5B best)
- Steering vector norm scales with model size (1.49 at 0.5B L2 vs 9.08 at 1.5B L2)

**Criteria met:**
- (2) Directly contradicts the published Phase 1 conclusion
- (3) Actionable: always test ALL candidate hub layers, not just the smaller model's hub
- (4) Causal: steering is an intervention, not observation
- (5) Differs across scale
- (8) Would surprise anyone who read the Phase 1 report

**Failure cases:** No random-vector control. Single seed. The "migration" could be wrong-layer testing at 0.5B too (maybe L8 was always the real hub).

**Practical implication:** The Phase 1 narrative about "steering breaks at scale" was premature. The real lesson: hub location must be discovered per model, and steering leverage INCREASES with scale when targeting the right layer.

**Confidence:** HIGH

**Next decisive experiment:** Test whether the steering DIRECTION (not just layer) transfers from 0.5B to 1.5B. If the direction is preserved but the layer changes, that's a different finding than if both change.

---

## G03: SmolLM2 hub at L0 — architecture beats scale

**Claim:** SmolLM2-1.7B has its universal hub at L0 (the very first layer), completely unlike Qwen's pattern. Hub position is architecture-specific, not depth-dependent.

**Evidence:**
- SmolLM2-1.7B: L0 is hub for all 4 tested families (json, factual, copying, code)
- hub_consistent=True
- Contradicts Qwen's L2->L26->L34 migration

**Criteria met:**
- (2) Contradicts the assumption that scaling laws determine hub position
- (5) Differs across family (Qwen vs SmolLM2)
- (8) Would surprise anyone assuming all transformers have similar internal structure

**Failure cases:** Single model, single seed, only 4 families. No ablation controls.

**Practical implication:** Architecture choice matters more than scale for internal structure. A 1.7B SmolLM2 is more different from a 1.5B Qwen internally than a 0.5B Qwen is from a 1.5B Qwen. This means model-specific atlases are essential, not optional.

**Confidence:** MEDIUM

**Next decisive experiment:** Test a second SmolLM2 scale (360M or 350M) and a different architecture family (Phi, Gemma) to see if L0-hub is SmolLM2-specific or a broader pattern.

---

## G04: Atlas-guided LoRA should beat generic LoRA

**Claim (untested):** LoRA adapters targeted at atlas-identified critical layers should outperform generic all-linear LoRA at equal or lower trainable parameters.

**Evidence:**
- Dataset shard ablation shows each skill concentrates in different layers (C04)
- o_proj-only is 10x more parameter-efficient than MLP-only for JSON (C05)
- Rank r=4 gives peak concentration (C06)
- These findings together suggest: target the right layers/modules/rank for each skill

**Criteria met (if proven):**
- (3) Directly actionable training rule
- (7) Saves compute/memory
- (8) Would surprise a competent finetuner who uses default all-linear LoRA

**Failure cases:** NOT YET TESTED. This is a hypothesis, not a finding.

**Practical implication:** If proven, this becomes the headline result: "Use the atlas to choose LoRA targets. Save 90% of parameters with equal or better skill acquisition."

**Confidence:** LOW (hypothesis only)

**Next decisive experiment:** THE KEY PHASE 3 EXPERIMENT. Train atlas-guided LoRA (target layers from dataset shard ablation) vs generic all-linear LoRA on 3+ task families. Measure task accuracy, not just ablation proxies. Compare trainable parameters.

---

## G05: Zero ablation = mean ablation at hub layers

**Claim:** Zeroing L2's output and replacing it with the mean activation give identical KL divergence (15.18 for factual_recall at 0.5B).

**Evidence:**
- 0.5B L2: zero=15.18, mean=15.18
- 1.5B L26: zero=14.13, mean=14.13
- But gaussian_resample gives lower effect (9.27/9.40)

**Criteria met:**
- (1) Replicated across 2 models
- (2) Contradicts the assumption that zero ablation is more destructive than mean ablation
- (4) Causal: both are interventions
- (8) Would surprise MI practitioners who prefer mean ablation as "less destructive"

**Failure cases:** May be specific to hub layers (the most important layer has activations far from the mean, so mean is effectively zero in the relevant subspace).

**Practical implication:** For hub layers, zero ablation is fine — it's not "more destructive" than mean. This validates the Phase 1 approach. But gaussian resample IS less destructive (40% reduction), suggesting the hub's activations have structure beyond just magnitude.

**Confidence:** MEDIUM-HIGH

**Next decisive experiment:** Test zero=mean equivalence at non-hub layers. If it holds everywhere, the two methods are equivalent. If it only holds at hub layers, there's something special about hub activation distributions.

---

## G06: Core circuit locks in at 10% of training

**Claim:** The core processing circuit (L2/L7/L9 for JSON) stabilizes within the first 10% of training and drifts <1% through the remaining 90%.

**Evidence:**
- 0.5B checkpoint timeline: L2/L7/L9 stable from step 10 to step 100
- Loss continues dropping (0.587 to 0.062) while core circuit is frozen
- Secondary layers (L15, L6) continue shifting

**Criteria met:**
- (3) Actionable: early stopping or checkpoint selection rule
- (7) Saves compute: if the core circuit locks at 10%, the remaining 90% is refinement
- (6) Predicts downstream behavior: step-10 checkpoint should perform similarly to step-100 on core skills
- (8) Would surprise practitioners who train for hundreds of steps

**Failure cases:** Only tested with JSON on 0.5B. Different tasks may have different lock-in points. The 10% figure may be absolute (10 steps) rather than relative.

**Practical implication:** Save checkpoints at 10% intervals. If the core circuit is stable by step 10/100, you can stop training early for basic skill acquisition. Use remaining compute for secondary refinement or different skills.

**Confidence:** MEDIUM

**Next decisive experiment:** Test lock-in at 1.5B with 3 task families. If the 10% rule generalizes, it becomes a major compute-saving training rule.

---

## G07: Late-layer LoRA is the universal training target

**Claim:** LoRA adapters' functional effects are concentrated at the final ~10% of layers (L19-L23 at 0.5B, L31-L35 at 3B), matching where the adapter norms peak.

**Evidence:**
- 0.5B: adapter-only ablation effect corr=0.85 with norm, peaks at L19-L23
- 3B: adapter knockout top5 = L31-L35
- Cross-model patching: monotonic recovery from early to late layers
- Resolves the H006 confusion: there is NO upstream propagation

**Criteria met:**
- (1) Replicated across 3 scales
- (2) Contradicts the "early layers are where training matters" intuition
- (3) Actionable: target late layers for surgical LoRA
- (4) Causal: ablation is an intervention
- (5) Differs across scale (but the ~10% pattern is consistent)

**Failure cases:** Only tested with LoRA. Full SFT may write to different layers. The "final 10%" may be a proximity-to-output artifact.

**Practical implication:** When doing LoRA, focus on the last 10% of layers for maximum effect per parameter. This is already partially captured by o_proj efficiency (G05's companion finding).

**Confidence:** MEDIUM-HIGH

**Next decisive experiment:** Test whether atlas-guided late-layer LoRA outperforms random-layer LoRA at equal parameters. This is the same experiment as G04 but focused on the late-layer hypothesis.

---

## G08: Skill knockout selectivity collapses at scale

**Claim:** Skill knockout (selective suppression via negative steering) works brilliantly at 0.5B (11654x selectivity at L19) but fails at 1.5B (0.24x) and 3B (knockout at L31-35 is not skill-specific).

**Evidence:**
- 0.5B: L19 factual knockout, 11654x selectivity
- 1.5B: 0.24x selectivity (essentially non-functional)
- 3B: adapter knockout top5 = L31-L35 (universal, not skill-specific)

**Criteria met:**
- (2) Contradicts the assumption that techniques that work at small scale work at larger scale
- (5) Differs dramatically across scale
- (8) Would surprise anyone who demonstrated knockout at 0.5B and assumed it would scale

**Failure cases:** The 0.5B selectivity may be a prompt artifact (NR007). No random-vector control.

**Practical implication:** Skill knockout via steering is a 0.5B parlor trick, not a scalable technique. At 1.5B+, skills become more entangled and cannot be surgically removed with a single vector.

**Confidence:** MEDIUM

**Next decisive experiment:** Test knockout with random-vector and shuffled-label controls at all 3 scales. If the controls show similar "selectivity," the original finding was noise.

---

## G09: Quantization affects small models more than large ones

**Claim:** 0.5B suffers 42-55% speed loss under quantization while 1.5B loses only 9% (4-bit NF4). Smaller models are more sensitive to quantization.

**Evidence:**
- 0.5B: 42-55% speed loss across quant levels
- 1.5B: 9% loss at 4-bit, 52% at 8-bit (counterintuitive: 8-bit is slower than 4-bit)
- Constraint adherence: 4-bit approximately equals bf16 for 1.5B

**Criteria met:**
- (2) Contradicts "8-bit is always the safe default" assumption
- (3) Actionable: use 4-bit NF4, not 8-bit, for deployment
- (7) Saves memory (4-bit is 4x compression)
- (8) Would surprise practitioners who default to 8-bit for "quality"

**Failure cases:** Only tested with bitsandbytes. GGUF quants may behave differently. No causal surface comparison.

**Practical implication:** 4-bit NF4 should be the default quantization for 1.5B+ models. 8-bit is actually worse in every metric (slower, larger, no quality advantage). For 0.5B, quantization hurts more — consider bf16 for edge deployment of tiny models.

**Confidence:** MEDIUM-HIGH

**Next decisive experiment:** Test whether steering vectors computed on bf16 models transfer to 4-bit models. If they don't, quantization changes causal surfaces and the "quality" measurement is insufficient.

---

## G10: Ablation controls preserve rank order

**Claim:** The top ablation layer remains the top layer regardless of ablation method (zero, mean, gaussian resample, random patch). This validates the hub identification methodology.

**Evidence:**
- 0.5B: L2 is top under all 4 methods
- 1.5B: L26 is top under all 4 methods
- Gaussian resample reduces effect size by ~40% but doesn't change rank order

**Criteria met:**
- (1) Replicated across 2 models
- (4) Causal: all methods are interventions
- (2) Contradicts the concern that zero ablation gives artifactual results

**Failure cases:** Only tested at hub layers. Non-hub layers may show different rank orders under different methods.

**Practical implication:** Hub identification is robust to methodology choice. The Phase 1 approach (zero ablation) gives the same hub as more principled methods. This is a validation of the entire atlas methodology.

**Confidence:** MEDIUM-HIGH

**Next decisive experiment:** Test rank-order stability at ALL layers (not just the hub) across methods. Create a full rank-order correlation matrix.
