# Phase 2 Report 8: Deobfuscation Surgery
**Experiment Block:** H
**Models:** Qwen/Qwen2.5-0.5B
**Tasks:** (planned: subskill decomposition of json_schema, factual_recall, code_semantics)
**Seeds:** [1]
**Date:** 2026-06-22
**Status:** not_run

## 1. What was tested
Deobfuscation surgery was planned to: (a) decompose composite skills into subskills, (b) train separate adapters for each subskill, (c) compare joint training vs composed adapters, and (d) measure cross-subskill transfer via activation patching. None of these were executed.

## 2. Why it matters
If composite skills (like json_schema) can be decomposed into independent subskills (like bracket matching, key ordering, value typing), then fine-grained model editing becomes possible. This would enable targeted modification of specific sub-behaviors without affecting the parent skill.

## 3. Exact models
- Qwen/Qwen2.5-0.5B: 24 layers
- LoRA config: r=8, alpha=16, target=[q,k,v,o_proj], lr=0.0002, steps=100
- Git commit: not recorded

## 4. Exact task suite
Planned but not executed:
- Subskill decomposition of json_schema (bracket matching, key ordering, value typing)
- Subskill decomposition of factual_recall (entity extraction, relation lookup)
- Subskill decomposition of code_semantics (variable scoping, control flow)

## 5. Key metrics
No metrics collected. All result fields are empty:
- `subskills_trained`: []
- `ablation_maps`: {}
- `eval_results`: {}
- `overlap_analysis`: {}
- `stacking_interference`: {}
- `semantic_preservation_signature`: {}

## 6. Controls
No controls executed.

## 7. Results
No results. The experiment was a placeholder with notes:
- "Full joint training comparison requires training a combined adapter (future work)"
- "Full cross-model patching requires matching architecture (deferred to P2-SEPARABILITY-001)"

## 8. Failed hypotheses
No hypotheses tested — experiment not executed.

## 9. Limitations
1. **No data**: Cannot draw any conclusions.
2. **Subskill decomposition not implemented**: The decomposition methodology is not defined.
3. **No evaluation framework**: Even if subskills were trained, there's no evaluation protocol.
4. **Deferred to other experiments**: Some components were deferred to P2-SEPARABILITY-001.

## 10. Next experiments
- Define subskill decomposition protocol
- Implement subskill training pipeline
- Design joint vs composed evaluation
- Execute the experiment
