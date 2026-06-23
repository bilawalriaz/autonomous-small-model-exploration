# Phase 3 Report: Gap Closure and Gem Discovery

Status: IN PROGRESS
Started: 2026-06-23

## 1. Claims We Confirmed (with stronger evidence)

*To be filled after multi-seed replication.*

| Claim | Previous Confidence | New Confidence | What Changed |
|-------|-------------------|----------------|--------------|
| C01: L2 hub (0.5B) | HIGH | | |
| C02: Hub migration | HIGH | | |
| C08: Late-layer encoding | MED-HIGH | | |
| C10: Steering migration | HIGH | | |
| C13: Layer skipping fails | HIGH | | |
| C20: Ablation rank-order stability | MED-HIGH | | |

## 2. Claims We Weakened

*Claims that lost confidence after replication or better controls.*

| Claim | Previous Confidence | New Confidence | Why |
|-------|-------------------|----------------|-----|
| | | | |

## 3. Claims We Killed

*Claims refuted by Phase 3 experiments.*

| Claim | Evidence | What Killed It |
|-------|----------|----------------|
| | | |

## 4. New Gems Found

*From gems.md — only those that passed Phase 3 testing.*

| ID | Claim | Criteria Met | Confidence |
|----|-------|-------------|------------|
| G01 | Hub at final ~10% | 5/8 | |
| G02 | Steering migrated not collapsed | 5/8 | |
| G03 | SmolLM2 hub at L0 | 3/8 | |
| G04 | Atlas-guided LoRA beats generic | untested | |
| G05 | Zero=mean at hub layers | 4/8 | |
| G06 | Core circuit locks at 10% | 4/8 | |
| G07 | Late-layer LoRA target | 5/8 | |
| G08 | Knockout selectivity collapses at scale | 3/8 | |
| G09 | 4-bit NF4 sweet spot | 4/8 | |
| G10 | Ablation rank-order stable | 4/8 | |

## 5. Practical Training Rules

*Rules backed by replicated causal evidence.*

1. **Use atlas-guided LoRA targeting** — [status: untested]
2. **Target late layers (final ~10%) for LoRA** — [status: replicated across scales]
3. **r=4 for surgical injection** — [status: needs task accuracy validation]
4. **o_proj is the most efficient module** — [status: needs multi-family replication]
5. **Core circuit locks at 10% of training** — [status: needs multi-task validation]
6. **Every model scale needs its own atlas** — [status: replicated]

## 6. Practical Inference/Deployment Rules

1. **Use 4-bit NF4, not 8-bit** — 8-bit is slower, larger, and no better quality
2. **Quantization affects small models more** — 0.5B loses 42-55% speed; 1.5B loses 9%
3. **Steering requires per-model hub discovery** — Cannot transfer layer knowledge across scales
4. **Skill knockout does not scale** — Works at 0.5B, fails at 1.5B+

## 7. Open Questions

1. Does quantization change causal surfaces? (P3-QUANT experiments)
2. Does the 10% lock-in rule generalize to 1.5B and other tasks?
3. Is the L13 secondary hub at 3B real or noise?
4. Can steering directions transfer across scales if the layer is correct?
5. Does atlas-guided LoRA actually beat generic LoRA on task accuracy?
6. Is the 11654x knockout selectivity real or a prompt artifact?
7. Why does zero=mean at hub layers?
8. What mechanism drives the hub migration pattern?

## 8. Reproducibility Package

- [ ] All Phase 3 scripts committed with --model, --seed, --force args
- [ ] Registry entries for all Phase 3 experiments
- [ ] Multi-seed results with mean/std
- [ ] Natural language prompt variants included
- [ ] All claims.md entries updated with new evidence
- [ ] threats.md updated with resolved/remaining threats
- [ ] gems.md updated with tested/untested status

## 9. Recommended Phase 4

*To be filled at end of Phase 3.*

Potential directions:
- SAE training on identified hub layers
- Cross-architecture atlas (Phi, Gemma, Llama 3.2)
- Full SFT comparison (if VRAM allows)
- Deobfuscation-specific training improvement
- Deployment pipeline with atlas-guided quantization
- Mechanistic explanation for hub migration
