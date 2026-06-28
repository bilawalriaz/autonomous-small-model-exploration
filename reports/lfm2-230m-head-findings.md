# LFM2.5-230M: Head Ablation Findings

## Summary

Individual head ablation across 6 attention layers (96 heads total) reveals that **only 2-3 heads matter** and they are concentrated in L2 and L4. Later attention layers (L6-L12) have no specialist heads.

## Top Heads (mean KL across 9 task families)

| Rank | Head | Mean KL | Best Family | Role |
|------|------|---------|-------------|------|
| 1 | **L4_H11** | 0.948 | instruction_following (1.76) | **Universal head** |
| 2 | **L2_H9** | 0.563 | json_schema (0.83) | Structural specialist |
| 3 | **L2_H11** | 0.250 | factual_recall (0.32) | General processing |
| 4 | L2_H4 | 0.083 | factual_recall (0.08) | Minor |
| 5 | L4_H4 | 0.078 | instruction_following (0.28) | Minor |
| 6 | L2_H7 | 0.064 | json_schema (0.08) | Minor |
| 7 | L2_H13 | 0.052 | code_syntax (0.14) | Minor |
| 8 | L4_H13 | 0.043 | code_syntax (0.10) | Minor |
| 9 | L4_H9 | 0.042 | instruction_following (0.16) | Minor |
| 10 | L2_H6 | 0.035 | factual_recall (0.03) | Minor |

## Per-Layer Head Statistics

| Layer | Type | Mean KL | Max KL | Best Head | Worst Head |
|-------|------|---------|--------|-----------|------------|
| L2 | attn | 0.078 | 0.563 (H9) | H9 | H0 (0.013) |
| L4 | attn | 0.087 | 0.948 (H11) | H11 | H1 (0.016) |
| L6 | attn | 0.017 | 0.027 (H11) | H11 | H0 (0.013) |
| L8 | attn | 0.017 | 0.024 (H11) | H11 | H5 (0.012) |
| L10 | attn | 0.015 | 0.023 (H11) | H11 | H5 (0.011) |
| L12 | attn | 0.011 | 0.018 (H11) | H11 | H7 (0.008) |

## Key Findings

### 1. L4_H11 is the Universal Head
L4_H11 is the #1 or #2 head for ALL 9 task families. It has the highest single-head KL (1.76 for instruction_following). This is the LFM2 equivalent of an "induction head" — a single head that contributes significantly to all tasks.

### 2. L2_H9 is the Structural Specialist
L2_H9 ranks #1 for json_schema (0.83), variable_renaming (0.68), and dead_code (0.50). These are all structural/compositional tasks. This head likely handles structural pattern recognition.

### 3. H11 is Architecturally Special
Head index 11 is the best head in ALL 6 attention layers:
- L2_H11: 3rd overall
- L4_H11: 1st overall
- L6_H11, L8_H11, L10_H11, L12_H11: all best in their layer

This suggests H11 has learned a position or function that is universally useful, even in layers where individual head effects are small.

### 4. Early Attention Layers Have Specialist Heads; Late Layers Don't
- L2, L4: Mean KL ~0.08, with clear outliers (H9, H11) at 5-10x the mean
- L6-L12: Mean KL ~0.01-0.02, very uniform across heads (max/min ratio <2x)

This matches the operator ablation finding: L2 and L4 operators are critical (KL=22.95, 34.87), while L6-L12 operators are less important (KL=1.83-11.10).

### 5. Head Effects Are Small Relative to Layer Effects
The strongest single-head effect (L4_H11, KL=0.948) is 87x smaller than the strongest layer effect (L0 skip, KL=82.9). This means individual attention heads contribute very little compared to the conv layers and MLPs. In LFM2, the attention mechanism is secondary to the convolutional processing.

### 6. Specialist Head Distribution
- Arithmetic: L4_H11 (1.53), L2_H9 (0.51)
- Factual recall: L4_H11 (1.31), L2_H9 (0.43)
- Instruction following: L4_H11 (1.76), L2_H9 (0.78)
- JSON schema: L2_H9 (0.83), L4_H11 (0.66)
- Variable renaming: L2_H9 (0.68), L4_H11 (0.51)
- Copying: L4_H11 (0.77), L2_H9 (0.30)
- Dead code: L2_H9 (0.50), L4_H11 (0.48)
- Code semantics: L4_H11 (0.58), L2_H9 (0.46)
- Code syntax: L4_H11 (0.94), L2_H9 (0.57)

L4_H11 dominates all families except json_schema and variable_renaming where L2_H9 leads.

## Cross-Layer Patching Note

Full-layer residual patching shows 100% recovery at all layers. This is the known limitation of full-residual patching — replacing the entire hidden state at any layer replaces all information. Component-specific patching (operator-only or MLP-only) is needed for meaningful results.

---

*Date: 2026-06-29*
*Experiment: run_lfm2_head_atlas_clean.py*
*Results: lfm2_230m_head_atlas_seed42_20260629_005525.json*
