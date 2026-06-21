# Component Atlas
Generated: 2026-06-21T20:51:14.363331+00:00
Total entries: 11
Confidence distribution:
- high: 1
- medium: 10

---

## layer_02_residual

**Type:** residual_stream  
**Layer:** 2  
**Confidence:** high  

**Claim:** Universal processing hub — largest ablation effect across all 12 task families

**Task families:** copying, delimiter_tracking, factual_recall, code_semantics, json_schema, arithmetic, dead_code, variable_renaming, uncertainty_signalling, refusal_compliance, verbosity_control, code_syntax

**Positive effects:**
- [exp_000005] L2 ablation causes KL 0.5-11.5 across all families; factual_recall most affected
- [exp_000007] L2 MLP ablation: major contributor to L2 effect
- [exp_000008] Factual direction at L2: 'rome' logprob 0.064->0.213 (3.3x) at s=+4.0

**Negative effects:**
- [exp_000018] L2 is NOT uniformly positional — effects concentrated at first+last tokens (mean 3.34/5.03), operators near-zero

**Steering:** Factual direction at L2 boosts target 3.3x. Negative steering suppresses. Oversteering at s>=+2 causes degeneration (Chinese chars).

**Adapter evidence:** All 5 family adapters affect L2 importance. delimiter adapter eliminates L9/L22/L4 dependence, making L2 more dominant.

**Limitations:** Tested on short synthetic prompts only. Ablation is zero-ablation (not mean/resample). Steering tested on factual recall only.

**Repro:** `python scripts/run_layer_ablation.py && python scripts/run_steering_sweep.py && python scripts/run_position_ablation.py`

---

## layer_00_mlp

**Type:** mlp  
**Layer:** 0  
**Confidence:** medium  

**Claim:** Second-strongest ablation target across all families; absorbs JSON skill after LoRA training

**Task families:** copying, delimiter_tracking, factual_recall, code_semantics, json_schema

**Positive effects:**
- [exp_000007] L0 MLP second-strongest across all families
- [exp_000009] L0 MLP importance for JSON: 10.85->13.84 (+2.99) after LoRA training

**Limitations:** JSON-specific training only. Other skill families concentrate elsewhere (dataset_shard_ablation).

**Repro:** `python scripts/compare_lora_ablation.py && python scripts/run_lora_rank_sweep.py`

---

## layer_22_residual

**Type:** residual_stream  
**Layer:** 22  
**Confidence:** medium  

**Claim:** Final token prediction layer — almost exclusively affects last-position tokens

**Task families:** copying, delimiter_tracking, factual_recall, code_semantics, json_schema

**Positive effects:**
- [exp_000018] Mean last-position effect 14.55 nats. All other positions ~0.
- [exp_000005] L22 second-strongest layer for delimiter_tracking in base model

**Limitations:** Position-specific ablation only tested on base model, not on adapted models.

**Repro:** `python scripts/run_position_ablation.py`

---

## layer_09_residual

**Type:** residual_stream  
**Layer:** 9  
**Confidence:** medium  

**Claim:** Instruction-sensitive layer — highest first-position effect among mid-layers, strong delimiter tracking

**Task families:** delimiter_tracking, json_schema

**Positive effects:**
- [exp_000018] First-position mean 5.66, last-position mean 9.20
- [exp_000005] L9 strongest layer for delimiter_tracking in base model

**Limitations:** Position analysis on short prompts only.

**Repro:** `python scripts/run_position_ablation.py`

---

## layer_12_head_08

**Type:** attention_head  
**Layer:** 12  
**Head:** 8  
**Confidence:** medium  

**Claim:** Strongest individual attention head across multiple families

**Task families:** copying, delimiter_tracking, factual_recall, code_semantics, json_schema

**Positive effects:**
- [exp_000006] L12 H8 strongest head across 5 families

**Limitations:** Head ablation only. No patching or steering on individual heads.

**Repro:** `python scripts/run_head_ablation.py`

---

## layer_01_residual

**Type:** residual_stream  
**Layer:** 1  
**Confidence:** medium  

**Claim:** Appears as universal skill injection point — positive delta across 3+ family adapters

**Task families:** factual_recall, code_semantics, json_schema

**Positive effects:**
- [exp_000014] L1 positive delta for factual_recall adapter (+6.51 on factual tasks)
- [exp_000014] L1 positive delta for code_semantics adapter (+4.70 on json tasks)
- [exp_000014] L1 positive delta for json_schema adapter (+3.00 on delimiter tasks)

**Limitations:** Correlational — L1 delta appears but causal mechanism unknown. Could be adapter weight injection, not functional routing.

**Repro:** `python scripts/run_dataset_shard_ablation.py`

---

## skill_concentration_pattern

**Type:** emergent_pattern  
**Confidence:** medium  

**Claim:** Each skill concentrates in different layers after LoRA training — no universal pattern

**Task families:** copying, delimiter_tracking, factual_recall, code_semantics, json_schema

**Positive effects:**
- [exp_000014] factual_recall: L3/16/19. code: L1/10/21. json: L6/12/13. delimiter: fully absorbed. copying: dispersed.

**Negative effects:**
- [exp_000014] H002 (universal L0-L2 concentration) REJECTED. Each skill has unique concentration pattern.

**Limitations:** Only 5 families tested. Only r=8 adapters. Only 100 training steps.

**Repro:** `python scripts/run_dataset_shard_ablation.py`

---

## norm_effect_separation

**Type:** emergent_pattern  
**Confidence:** medium  

**Claim:** Adapter weight norms peak at L20-L23 but ablation effects peak at L0-L2

**Positive effects:**
- [exp_000015] Rank sweep: norms peak L22/L23. Ablation effects peak L0/L2. Spatial separation.

**Limitations:** Norm analysis doesn't prove causal direction. Could be that late layers carry more parameters but early layers do the processing.

**Repro:** `python scripts/run_adapter_archaeology.py`

---

## adapter_stack_factual_json

**Type:** adapter_interaction  
**Confidence:** medium  

**Claim:** factual_recall + json_schema adapters stack cleanly with positive synergy

**Task families:** factual_recall, json_schema

**Positive effects:**
- [exp_000016] factual+json: +2.35 synergy on factual, +1.17 on json, <0.3 interference elsewhere

**Limitations:** Only weighted merge (0.5/0.5) tested. No sweep of merge weights.

**Repro:** `python scripts/run_adapter_stacking.py`

---

## adapter_stack_delimiter_destructive

**Type:** adapter_interaction  
**Confidence:** medium  

**Claim:** delimiter_tracking adapter is destructive when combined with other adapters

**Task families:** delimiter_tracking

**Negative effects:**
- [exp_000016] delimiter+code: -15.51 nats on delimiter task. Consistent across all pairs.

**Limitations:** May be an artifact of the delimiter task evaluation (target token is single char, model generates multi-char completions).

**Repro:** `python scripts/run_adapter_stacking.py`

---

## early_circuit_establishment

**Type:** training_phenomenon  
**Confidence:** medium  

**Claim:** Core component structure for JSON schema locks in by step 10 (first 10% of training)

**Task families:** json_schema

**Positive effects:**
- [exp_000017] L2/L7/L9 for json_schema: step10=23.37/21.62/19.00, step100=23.38/21.75/18.88. <1% drift after step 10.

**Limitations:** Only JSON schema family tested. Only 100 training steps. Different families might establish later.

**Repro:** `python scripts/run_checkpoint_timeline.py`

---

