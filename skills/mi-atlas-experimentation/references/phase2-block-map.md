# Phase 2 Block Map & Execution Dependencies

## Block Execution Order (by priority)

| Priority | Block | Script | Models | Est. Runtime | Dependencies |
|----------|-------|--------|--------|-------------|--------------|
| 1 | A: Parity | run_phase2_parity.py | 1.5B | ~2h | None |
| 2 | B: Steering | run_phase2_steering_migration.py | 0.5B, 1.5B | ~4h | None |
| 3 | C: Ablation Controls | run_phase2_ablation_controls.py | 0.5B, 1.5B | ~3h | None |
| 4 | F: Adapter Surgery | run_phase2_adapter_surgery.py | 0.5B, 1.5B | ~5h | Adapters from Phase 1 |
| 5 | H: Deobfuscation | run_phase2_deobfuscation.py | 0.5B, 1.5B | ~4h | None |
| 6 | G: Separability | run_phase2_skill_separability.py | 0.5B | ~2h | Adapters from Phase 1 |
| 7 | D: Third Scale | run_phase2_third_scale.py | 3B (4bit) | ~4h | None |
| 8 | E: Cross-Family | run_phase2_cross_family.py | Gemma-2B, SmolLM2 | ~3h | None |
| 9 | I: Robustness | run_phase2_long_task_robustness.py | 0.5B, 1.5B | ~3h | None |

Total estimated: ~25h on RTX 2070 Super (8GB VRAM)

## VRAM Budget

| Model | Precision | VRAM | Training | Notes |
|-------|-----------|------|----------|-------|
| Qwen2.5-0.5B | bf16 | ~1GB | LoRA bs=2 | Phase 1 primary |
| Qwen2.5-1.5B | bf16 | ~3GB | LoRA bs=1, grad_ckpt | Phase 1 secondary |
| Qwen2.5-3B | 4bit NF4 | ~2GB | LoRA bs=1, grad_ckpt | Phase 2 third scale |
| Gemma-2-2B | bf16 | ~4.5GB | LoRA bs=1, grad_ckpt | Cross-family |
| SmolLM2-1.7B | bf16 | ~3.5GB | LoRA bs=1, grad_ckpt | Cross-family |

## Block Dependencies Graph

```
A (parity) ─────────────────────────┐
B (steering) ───────────────────────┤
C (ablation controls) ──────────────┤
                                    ├──→ I (robustness) ──→ 10 (final report)
F (adapter surgery) ────────────────┤
G (separability) ←── needs adapters ┤
H (deobfuscation) ──────────────────┤
D (third scale) ────────────────────┤
E (cross-family) ───────────────────┘
```

Blocks A-H are independent of each other. Block I benefits from all prior results. Report 10 depends on all other reports.

## Hypothesis → Block Mapping

| Hypothesis | Tested By | Verdict Criteria |
|------------|-----------|------------------|
| H1: Hub migration | A, B, D, E | Hub layer shifts with scale across models |
| H2: MLP/attention shift | A, C, D | MLP dominance ratio changes with scale |
| H3: Steering moved not collapsed | B | Steering works at new hub layers in 1.5B |
| H4: Knockout selectivity decreases | A, G | Selectivity ratio drops from 0.5B to 1.5B |
| H5: Final 10% transfer invariant | F, H | Transfer recovery peaks at last 10% of layers |
| H6: LoRA norms insufficient | F | Norm-effect correlation varies by adapter |
| H7: Layer skipping invalid | A, D | Skipping destroys output at all scales |
| H8: Adapter compatibility predictable | F | Compatibility correlates with localization overlap |

## Key Metrics Per Block

- **A**: seed_variance (σ/μ), confidence_recalibration
- **B**: target_logit_delta, KL, collateral_damage, steering_strength_curve
- **C**: rank_order_stability, method_agreement_rate
- **D**: hub_layer, head_max_effect, steering_boost, knockout_selectivity (vs 0.5B/1.5B trend)
- **E**: hub_layer, architecture_specific_findings
- **F**: compatibility_class, norm_effect_correlation, collateral_damage_per_adapter
- **G**: skill_separability_score (SSS), component_breakdown
- **H**: subskill_localization_overlap, interference_class, hallucination_rate
- **I**: hub_stability_across_lengths, steering_degradation_rate

## Quality Gates

Phase 2 is not complete until:
- [ ] All experiments registered in registry.jsonl
- [ ] All raw results saved to experiments/results/
- [ ] All headline claims have controls
- [ ] All major results have ≥3 seeds (unless marked pilot)
- [ ] 0.5B and 1.5B comparisons are symmetrical
- [ ] At least one third scale point tested
- [ ] At least one cross-family replication attempted
- [ ] Steering migration directly tested
- [ ] Zero ablation claims checked against mean/resample/patch
- [ ] Adapter compatibility matrix exists
- [ ] Skill separability benchmark exists
- [ ] Deobfuscation surgery report exists
- [ ] Negative results documented
- [ ] Skill file updated
- [ ] progress.md accurate
