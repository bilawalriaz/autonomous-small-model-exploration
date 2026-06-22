# Phase 2 Report Templates

Each report in reports/phase2/ must follow this structure.

## Required Sections

### Header
```
# Phase 2 Report N: [Title]
**Experiment Block:** [A-I]
**Models:** [list]
**Tasks:** [list]
**Seeds:** [list]
**Date:** [YYYY-MM-DD]
**Status:** [complete/partial/not_run]
```

### 1. What was tested
Precise description of the experiment. No vague language.

### 2. Why it matters
Connection to the Phase 2 thesis and specific hypotheses.

### 3. Exact models
Model IDs, revisions, tokenizer versions, quantization, dtype.

### 4. Exact task suite
Task families, number of examples, splits, scoring functions.

### 5. Key metrics
Primary metrics with values. Include effect sizes and confidence.

### 6. Controls
Every control that was run. What happened with each.

### 7. Results
Per-seed results. Tables with mean, std, min, max.
For causal maps: top layers, top heads, top MLPs, normalized importance.
Always include machine-readable CSV/JSON summaries.

### 8. Failed hypotheses
Which hypotheses were rejected or weakened. Why.

### 9. Limitations
What was not tested. What could confound results.
At least 3 limitations per report.

### 10. Next experiments
What should be tested next based on these results.

---

## Report-Specific Requirements

### 01_parity_verification.md
- Side-by-side 0.5B vs 1.5B comparison for ALL task families
- Symmetrical experiment coverage table
- Missing experiments filled with actual results

### 02_steering_migration.md
- Per-layer steering effectiveness heatmap
- Single vs multi-layer comparison
- Verdict: steering migrates / becomes distributed / fails despite hubs / edit cost quantified

### 03_ablation_controls.md
- Zero vs mean vs resample vs patch comparison
- Rank-order stability across methods
- Claims that survive non-zero ablation flagged

### 04_third_scale_point.md
- 3B atlas structure
- Trend continuation or break
- Hub position, MLP/attention ratio, steering leverage, knockout selectivity

### 05_cross_family_replication.md
- Cross-architecture comparison
- Universal vs architecture-specific findings

### 06_adapter_surgery.md
- Per-adapter norm and effect profiles
- Compatibility matrix
- Predictability from localization overlap

### 07_skill_separability_benchmark.md
- Skill separability scores per skill
- Component breakdown
- Most/least separable skills

### 08_deobfuscation_surgery.md
- Subskill atlas
- Joint vs composed adapter comparison
- Transfer via activation patching
- Practical deobfuscation recommendations

### 09_long_task_robustness.md
- Short vs medium vs long prompt hub maps
- Steering stability by prompt length
- Real-task transfer

### 10_final_phase2_findings.md
- Executive summary
- What Phase 1 claimed vs what Phase 2 confirmed/rejected
- Scale-dependent findings
- Cross-family findings
- Adapter surgery findings
- Deobfuscation findings
- Practical rules for small-model brain surgery
- Open questions
- Recommended Phase 3
- Reproducibility checklist
