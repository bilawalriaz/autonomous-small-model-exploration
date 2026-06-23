# Claims Audit — Phase 3 Gap Closure

Generated: 2026-06-23
Purpose: Every claim from Phase 1 & 2, classified, with falsifiers and replication status.

## Classification Key

- **Observed once** — single seed, single model, no controls
- **Replicated** — multiple seeds OR multiple models OR multiple methods agree
- **Causal but narrow** — causal evidence exists but only for one model/task/prompt regime
- **Causal and actionable** — causal, replicated, and directly usable for training/inference
- **Possibly artifact** — methodological concern makes the finding suspect
- **Refuted** — contradicted by stronger evidence
- **Unknown** — insufficient data to classify

---

## C01: L2 is a universal importance hub (0.5B)

**Status:** CAUSAL AND ACTIONABLE — REPLICATED
**Confidence:** HIGH (upgraded from single-seed)
**Evidence:**
- Zero ablation: L2 top layer for all 12 families (KL 0.5-11.5)
- MLP ablation: L2 MLP is the dominant MLP component
- Steering: L2 factual direction boosts target 3.3x (Phase 1)
- Position-specific: L2 routes first+last tokens (mean 3.34/5.03)
- Ablation controls (Phase 2): zero=mean=15.18, gaussian_resample=9.27, random_patch=8.56
  — rank order preserved across methods (L2 remains top)
- Cross-model patching: L2 among top transfer layers

**Seeds:** 3 (Phase 3 R1: seeds 42, 137, 256 — hub at L2 for ALL seeds, std=0.0, verdict=robust)
**Models:** Qwen2.5-0.5B only
**Controls:** zero, mean, gaussian_resample, clean/corrupt patch, random_patch (Phase 2)
**Prompts:** 3 per family (short synthetic)

**Falsifier:** ~~If L2 is not the top ablation layer in 2/3 seeds~~ TESTED: L2 is top in 3/3 seeds. Remaining falsifier: if natural language prompts shift the hub.

**Gap:** ~~Needs 3-seed replication.~~ DONE (hub_std=0.0). Needs natural language prompt validation (P1).

---

## C02: Hub migrates later with scale (L2 -> L14/L26 -> L34)

**Status:** REPLICATED — REVISED
**Confidence:** HIGH
**Evidence:**
- 0.5B: L2 hub (8% depth), mean_effect=19.11 — REPLICATED across 3 seeds (std=0.0)
- 1.5B: L14 hub (50% depth), mean_effect=15.39 — REPLICATED across 3 seeds (std=0.0)
  — **Phase 3 revision:** Phase 2 reported L26 (93% depth) as the hub, but that was tested with fewer families. Full 12-family suite reveals L14 as the true hub. L26 is #2 (12.84).
- 3B: L34 hub (94% depth), mean_effect=23.63
  — L34 is top for 9/12 families; L13 is secondary hub for 3/12 (json, dead_code, refusal)
- Pattern: hub at 8% -> 50% -> 94% depth (revised from 8% -> 93% -> 94%)

**Seeds:** 3 per model (Phase 3 R1, R2). Single seed for 3B (R3 running).
**Models:** 3 Qwen2.5 scales
**Controls:** Ablation controls at 0.5B and 1.5B confirm rank order stability

**Falsifier:** If a 4th scale (e.g., 7B) breaks the pattern, or if multi-seed shows high variance in hub location.

**Gap:** Single seed per model. The L13 secondary hub at 3B is unexplained. Need to test whether L13 is a genuine second hub or an artifact of the task suite.

---

## C03: SmolLM2 hub is at L0 (architecture-specific, not depth-dependent)

**Status:** Observed once
**Confidence:** MEDIUM
**Evidence:**
- SmolLM2-1.7B: L0 is hub for all 4 tested families (json, factual, copying, code)
- hub_consistent=True across all families
- Contradicts Qwen pattern entirely (Qwen hub migrates with scale; SmolLM2 hub fixed at L0)

**Seeds:** 1
**Models:** SmolLM2-1.7B only
**Controls:** None (no ablation controls, no steering replication)

**Falsifier:** If another SmolLM2 scale (e.g., 360M, 350M) has hub at a different layer, or if the finding is a task-suite artifact.

**Gap:** Only 4 families tested. No steering replication. No ablation controls. Needs a second SmolLM2 scale and a different architecture (e.g., Phi, Gemma) to confirm the "architecture-specific" claim.

---

## C04: LoRA training rewires where skills live (skill-specific concentration)

**Status:** Causal but narrow
**Confidence:** MEDIUM
**Evidence:**
- Dataset shard ablation (0.5B only):
  - factual_recall: L3/16/19
  - code_semantics: L1/10/21
  - json_schema: L6/12/13
  - copying: dispersed
  - delimiter: fully absorbed
- Rejects H002 (universal L0-L2 concentration)

**Seeds:** 1
**Models:** Qwen2.5-0.5B only
**Controls:** None (no replication, no different rank/module comparison)

**Falsifier:** If multi-seed shows different concentration patterns, or if full SFT produces different patterns than LoRA.

**Gap:** Critical gap — only tested with one LoRA config (r=8, all-linear). Atlas-guided vs random layer targeting not yet compared. This is one of the most important Phase 3 experiments.

---

## C05: o_proj is the most efficient skill injection pathway

**Status:** Causal but narrow
**Confidence:** MEDIUM
**Evidence:**
- Module sweep (0.5B): o_proj +3.64 with 344K params, v_proj +2.75 with 197K, MLP +1.92 with 3.3M
- Only tested on JSON schema family

**Seeds:** 1
**Models:** Qwen2.5-0.5B only
**Controls:** Compared across modules but no random-matched-layer control

**Falsifier:** If o_proj advantage disappears for factual recall or code tasks, or if at 1.5B+ the efficiency ranking changes.

**Gap:** Needs replication across 3+ task families and at least 2 model scales. The "o_proj writes to residual stream" explanation is plausible but not proven causal.

---

## C06: LoRA rank r=4 is optimal for surgical injection

**Status:** Observed once
**Confidence:** LOW-MEDIUM
**Evidence:**
- Rank sweep (0.5B): L0 MLP peaks at r=4 (15.77), declines at r=16 (13.94)
- Total norm scales linearly (6.14 to 22.92)
- Only measured on JSON schema

**Seeds:** 1
**Models:** Qwen2.5-0.5B only
**Controls:** No comparison to full-model baseline, no task accuracy measurement

**Falsifier:** If r=4 peak is specific to L0 MLP effect (not actual task performance), or if r=8 gives better accuracy with acceptable distribution.

**Gap:** The metric (L0 MLP ablation effect) is a proxy, not direct task performance. Need to measure actual JSON accuracy at each rank. Also needs replication at 1.5B+.

---

## C07: Core circuit locks in by step 10

**Status:** Observed once
**Confidence:** MEDIUM
**Evidence:**
- Checkpoint timeline (0.5B, JSON): L2/L7/L9 stabilize at step 10, drift <1% through step 100
- Loss: 0.587 to 0.062
- Secondary layers (L15/L6) continue shifting

**Seeds:** 1
**Models:** Qwen2.5-0.5B only
**Controls:** None

**Falsifier:** If different tasks have different lock-in points, or if the "drift <1%" is actually meaningful for task accuracy.

**Gap:** Only tested with JSON. Factual recall may lock in at a different step. Also: is step 10 specific to 100-step training, or does it scale? (e.g., does a 1000-step run also lock in at 10%?)

---

## C08: Trained behavior encoded in late-layer activations

**Status:** Replicated across models
**Confidence:** MEDIUM-HIGH
**Evidence:**
- 0.5B: Cross-model patching monotonic recovery L23=100%, L22=97%, L21=95%, L19=80%
- 1.5B: cross_model_patching result exists (file present)
- Adapter-only ablation: effect at L19-L23, corr=0.85 with norm
- 3B: adapter knockout top5 critical layers = L31-L35 (final ~10%)

**Seeds:** 1 per model
**Models:** 0.5B, 1.5B, 3B
**Controls:** Adapter-only ablation separates general from adapter-specific importance

**Falsifier:** If early layers show high recovery for some task families, or if the monotonic pattern breaks with mean ablation.

**Gap:** The "final ~10% of layers" pattern is consistent across scales (0.5B: L19-23/24=96-100%, 1.5B: L25-27/28=89-96%, 3B: L31-35/36=86-97%). But only tested with LoRA adapters, not full SFT. And the mechanism (why late layers?) is not explained.

---

## C09: Skill knockout via negative steering

**Status:** Causal but narrow
**Confidence:** MEDIUM
**Evidence:**
- 0.5B: L19 factual knockout, selectivity=11654x at s=-2.0
- L2 non-selective (universal hub)
- L21: 53x selectivity
- 1.5B: skill_knockout result exists but analysis incomplete
- 3B: adapter knockout top5 = L31-L35 (not skill-selective)

**Seeds:** 1
**Models:** 0.5B primarily
**Controls:** No random-vector control, no same-norm control

**Falsifier:** If the 11654x selectivity doesn't replicate across seeds, or if it's an artifact of the specific prompts used (near-zero base probability issue noted in NR007).

**Gap:** The 11654x number is suspiciously high. Needs random-vector and shuffled-label controls. Also, the selectivity collapse at 1.5B/3B (0.24x and knockout at L31-35 not skill-specific) is a major unexplained finding.

---

## C10: Steering migrates, not collapses, at 1.5B

**Status:** Replicated
**Confidence:** HIGH
**Evidence:**
- 0.5B: best single-layer steering at L8 (delta=1.32 at s=-2.0) and L12 (delta=2.16 at s=-4.0)
- 1.5B: best at L21 (delta=4.65 at s=-4.0), L26 (delta=3.54 at s=-2.0), L6 (delta=3.14 at s=+4.0)
- Steering vector norm 6x larger at 1.5B (9.08 vs 1.49 at L2)
- 3B: steering tested at L2/L18/L26/L34/L35; L35 has sv_norm=227

**Seeds:** 1 per model
**Models:** 0.5B, 1.5B, 3B
**Controls:** strength sweeps but no random-vector control

**Falsifier:** If steering at 0.5B hub layers (L2) gives equal effect at 1.5B when properly scaled, the "migration" is just wrong-layer testing.

**Gap:** The original Phase 1 "steering failure at 1.5B" was indeed a wrong-layer artifact. But the "migration" claim needs to show that the SAME direction vector (not just same layer type) doesn't transfer across scales. Also needs random-vector controls to confirm the effect is task-specific.

---

## C11: Adapter norm-effect correlation = 0.85

**Status:** Possibly artifact (of reporting, not data)
**Confidence:** LOW (revised down due to contradictory reporting)
**Evidence:**
- 0.5B adapter-only ablation: corr=0.85, effects peak at L19-L23
- Publication report says "weak or negative relationship" (corr=0.855) — self-contradictory
- H6 was simultaneously "supported" and "rejected" in different reports

**Problem:** The publication report (Section 3.10) states "correlation of 0.855, indicating a weak or negative relationship" — but 0.85 is a STRONG positive correlation. This is either a bug in the report generation or a fundamental confusion in the analysis. The same document claims H6 is both supported (Section 4.3) and rejected (NR008).

**Falsifier:** Already partially falsified by internal inconsistency.

**Gap:** The actual finding seems to be: adapter effects ARE at the same layers where norms peak (L19-L23), contradicting the "upstream propagation" hypothesis. The correlation is strong and positive. The report narrative is confused. Needs re-analysis with correct interpretation.

---

## C12: Adapters can be combined with varying interference

**Status:** Observed once
**Confidence:** MEDIUM
**Evidence:**
- factual+json: synergistic (+2.35/+1.17)
- code+json: compatible
- delimiter: destructive (-7 to -16 nats)

**Seeds:** 1
**Models:** 0.5B only
**Controls:** None

**Falsifier:** If multi-seed shows different interference patterns, or if the synergy is just noise.

**Gap:** Needs replication. Also needs to test whether interference is predictable from localization overlap (hypothesis H8).

---

## C13: Naive layer skipping destroys output

**Status:** Replicated across models
**Confidence:** HIGH (for the negative result)
**Evidence:**
- 0.5B: 0% top-5 overlap for all 10 skip configs
- 1.5B: efficiency_1.5b.json exists with similar result
- 3B: third_scale_layer_skip_seed1.json exists

**Seeds:** 1 per model
**Models:** 0.5B, 1.5B, 3B
**Controls:** Multiple skip configs tested

**Falsifier:** If atlas-guided skip (skipping low-causal layers) works better, or if skip + recovery finetune recovers quality.

**Gap:** Only tests naive zero-skip. The Phase 3 spec asks for atlas-guided skip + recovery finetune. That's the actionable experiment.

---

## C14: Quantization: 4-bit NF4 is the sweet spot

**Status:** Replicated
**Confidence:** MEDIUM-HIGH
**Evidence:**
- 1.5B: bf16=18.8 tok/s, 8bit=9.0 tok/s (52% slower), 4bit=17.1 tok/s (9% slower)
- 0.5B: 42-55% speed loss under quantization (more affected than 1.5B)
- Constraint adherence: 4bit approximately equals bf16 for 1.5B

**Seeds:** 1
**Models:** 0.5B, 1.5B
**Controls:** Multiple quant levels tested

**Falsifier:** If 4bit changes causal surfaces (steering vectors don't transfer), or if quality drops on harder tasks.

**Gap:** CRITICAL — only measured inference speed and constraint adherence. Did NOT measure whether quantization changes causal surfaces. This is one of the most important Phase 3 experiments: "Does quantization preserve benchmark quality while changing intervention behaviour?"

---

## C15: Positional specialization across layers

**Status:** Observed once
**Confidence:** MEDIUM
**Evidence:**
- L22: last-position only (14.55 nats mean effect)
- L0/L2: first+last routers
- L9: instruction-sensitive (first=5.66, last=9.20)
- Operators/delimiters: near-zero everywhere

**Seeds:** 1
**Models:** 0.5B only
**Controls:** None

**Falsifier:** If positional patterns change after LoRA training, or if they don't replicate at 1.5B+.

**Gap:** Only 0.5B. Needs cross-scale comparison. Also: does token-position importance change with prompt length?

---

## C16: code_semantics is most surgically separable

**Status:** Observed once
**Confidence:** LOW
**Evidence:**
- SSS scores (0.5B): code=0.36, json=0.22, factual=0.22, copying=0.22, delimiter=0.21
- code_semantics has highest insertion_gain (+3.70) while others are negative

**Seeds:** 1
**Models:** 0.5B only
**Controls:** Collateral damage was 0 for all skills (suspicious — may indicate the metric is not sensitive enough)

**Falsifier:** If collateral damage is actually non-zero with better measurement, or if the SSS ranking changes at 1.5B.

**Gap:** The zero collateral damage across all skills is suspicious. Also: the SSS formula weights are somewhat arbitrary. Needs validation with actual downstream task accuracy, not just ablation-based proxies.

---

## C17: L1 as universal skill injection point

**Status:** Unknown
**Confidence:** LOW
**Evidence:**
- "Positive delta across 3+ adapters" (not quantified in available results)
- Listed as "weak/tentative signal"

**Falsifier:** If the positive delta is within noise, or if it's specific to the adapter configuration used.

**Gap:** Not enough data to evaluate. Mark for targeted testing in Phase 3.

---

## C18: Head effects are small (distributed attention)

**Status:** Observed once
**Confidence:** MEDIUM
**Evidence:**
- 0.5B: max head ablation KL = 0.046 (200x smaller than layer effects)
- 1.5B: head_ablation_1.5b.json exists
- 3B: third_scale_head_ablation_seed1.json exists

**Seeds:** 1 per model
**Models:** 0.5B, 1.5B, 3B

**Falsifier:** If specific heads have large effects on specific tasks (the 0.5B result may be because GQA has only 2 KV heads).

**Gap:** Need to check if head specialization increases with scale (the skill says "22x stronger head specialization at 1.5B" but this needs verification from the actual data).

---

## C19: Norm-effect separation (adapter norms at late layers, effects at early layers)

**Status:** Refuted (the "separation" claim) / Resolved (the actual finding)
**Confidence:** HIGH (for the resolution)
**Evidence:**
- The publication report confused 0.85 correlation as "weak or negative"
- NR008 correctly identifies: adapter effects ARE at L19-L23 (where norms peak)
- The "separation" was between GENERAL layer importance (L0-L2) and ADAPTER-SPECIFIC importance (L19-L23)
- There is no upstream propagation

**Resolution:** The original claim of "norm-effect paradox" was based on conflating two different measurements. The actual finding is: general ablation at L0-L2 is devastating because those layers are universally important, NOT because the adapter propagates there. The adapter itself acts at L19-L23.

---

## C20: Ablation method rank-order stability

**Status:** Replicated (2 models)
**Confidence:** MEDIUM-HIGH
**Evidence:**
- 0.5B L2: zero=15.18, mean=15.18, gaussian=9.27, random_patch=8.56
  — L2 remains top under all methods
- 1.5B L26: zero=14.13, mean=14.13, gaussian=9.40, random_patch=10.14
  — L26 remains top under all methods
- Key: zero and mean give IDENTICAL results (mean ablation = zero ablation for these layers)

**Falsifier:** If gaussian resample changes the rank order (it doesn't for the top layer, but reduces effect size by ~40%).

**Gap:** The zero=mean equivalence is surprising and needs explanation. Also, clean/corrupt patching gives KL=0 everywhere (NR002), so that method is useless for rank ordering.

---

## Summary Table

| ID | Claim | Status | Confidence | Seeds | Models | Key Gap |
|----|-------|--------|------------|-------|--------|---------|
| C01 | L2 hub (0.5B) | Causal+actionable | HIGH | 1 | 1 | Multi-seed, NL prompts |
| C02 | Hub migration L2->L26->L34 | Replicated | HIGH | 1/model | 3 | Multi-seed, 4th scale |
| C03 | SmolLM2 hub at L0 | Observed once | MEDIUM | 1 | 1 | 2nd SmolLM2 scale |
| C04 | Skill-specific LoRA concentration | Causal+narrow | MEDIUM | 1 | 1 | Multi-seed, atlas-guided vs random |
| C05 | o_proj most efficient injection | Causal+narrow | MEDIUM | 1 | 1 | Multi-family, multi-scale |
| C06 | r=4 optimal rank | Observed once | LOW-MED | 1 | 1 | Task accuracy, multi-scale |
| C07 | Core circuit locks in at step 10 | Observed once | MEDIUM | 1 | 1 | Multi-task, step scaling |
| C08 | Late-layer behavior encoding | Replicated | MED-HIGH | 1/model | 3 | Multi-seed, mechanism |
| C09 | Skill knockout (L19 factual) | Causal+narrow | MEDIUM | 1 | 1 | Controls, selectivity collapse |
| C10 | Steering migrates not collapses | Replicated | HIGH | 1/model | 3 | Direction transfer, controls |
| C11 | Norm-effect corr=0.85 | Possibly artifact | LOW | 1 | 1 | Re-analysis needed |
| C12 | Adapter stacking interference | Observed once | MEDIUM | 1 | 1 | Multi-seed, predictability |
| C13 | Layer skipping fails | Replicated | HIGH | 1/model | 3 | Atlas-guided + recovery |
| C14 | 4-bit NF4 sweet spot | Replicated | MED-HIGH | 1 | 2 | Causal surface drift |
| C15 | Positional specialization | Observed once | MEDIUM | 1 | 1 | Cross-scale, prompt length |
| C16 | code_semantics most separable | Observed once | LOW | 1 | 1 | Better metrics, multi-scale |
| C17 | L1 universal injection point | Unknown | LOW | 1 | 1 | Targeted testing |
| C18 | Distributed attention | Observed once | MEDIUM | 1/model | 3 | Head specialization at scale |
| C19 | Norm-effect separation | Refuted | HIGH | 1 | 1 | Already resolved |
| C20 | Ablation rank-order stability | Replicated | MED-HIGH | 1/model | 2 | Why zero=mean? |
