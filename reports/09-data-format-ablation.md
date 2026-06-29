# Phase 9: Data Format Ablation

## Executive Summary

Phase 8 showed that dataset format dominates hyperparameters for LFM2.5-230M SFT (5x more impact). Phase 9 isolates this finding through a controlled experiment: same canonical content rendered into 6 training formats, evaluated with a permanent judge-based harness.

**Status:** Infrastructure complete, pilot experiments planned.

---

## 1. What Was Tested

Six training formats applied to identical canonical content (300 examples spanning 9 domains):

1. Alpaca flat (instruction/input/output)
2. Single-turn ChatML
3. Multi-turn concise
4. Multi-turn verbose
5. Structured terse
6. Bad-format negative control

Each format trained with frozen quality config (Adafactor, lr=2e-4, r=8, hub all modules, 300 steps) and evaluated with the same permanent eval harness (150 prompts, 9 categories, judge-based scoring).

---

## 2. Why Loss Alone Is Insufficient

Training loss measures how well the model fits the training distribution. Different formats have different inherent "easiness" — flat verbose text is easier to model than structured terse output. Lower loss on Format A vs Format B could mean:
- Format A is genuinely better learned, OR
- Format A is inherently easier to predict

Therefore we measure:
- Behavioral win-rates (blind pairwise judging)
- Category-level instruction following
- JSON/schema compliance
- Output length and concision
- Hallucination tendency
- KL divergence from base model

---

## 3. Frozen Baseline Recipe

### Quality Adapter
- Model: LiquidAI/LFM2.5-230M
- LoRA: r=8, alpha=16, hub layers all modules (q/k/v/o_proj)
- Optimizer: Adafactor
- Learning rate: 2e-4
- Steps: 300
- Max sequence length: 1024
- Batch size: 4, gradient accumulation: 4

### Surgical Adapter
- Same as quality except:
- Target modules: o_proj only (hub-targeted)
- ~70% fewer parameters

---

## 4. Eval Harness Design

Permanent eval set: `data/eval/small_model_eval_v1.jsonl`
- 150 prompts across 9 categories
- Designed to reveal regressions, not flatter models
- Categories: instruction following, JSON/structured, GameFAQ extraction, coding, deobfuscation, reasoning, concision/anti-slop, factual Q&A, multi-turn

Generation settings: temp=0.2, top_p=0.9, max_new_tokens=512, seed=42

---

## 5. Dataset Compiler Design

Canonical input: `data/canonical/phase9_pilot_300.jsonl`
- 300 examples, 9 domains, content-independent of format
- Each example has: id, domain, difficulty, user_intent, context, ideal_answer, constraints

Renderer: `scripts/data/render_dataset_formats.py`
- Takes canonical input, outputs 6 format variants
- Tracks canonical IDs across all variants
- Validates content constancy

---

## 6. Controlled Format Ablation Results

[To be filled after experiments run]

---

## 7. Dataset Mixture Results

[To be filled after bilawal_smol_magpie_v1 experiments]

---

## 8. Quality vs Surgical Adapter Comparison

[To be filled]

---

## 9. Win-Rate Tables

| Format | Final Loss | KL | Win-rate vs Base | JSON Validity | Avg Tokens | Judge Score |
|--------|-----------|-----|-----------------|---------------|------------|-------------|
| alpaca_flat | | | | | | |
| single_turn_chat | | | | | | |
| multi_turn_concise | | | | | | |
| multi_turn_verbose | | | | | | |
| structured_terse | | | | | | |
| bad_format_control | | | | | | |

---

## 10. Category-Level Breakdown

[To be filled]

---

## 11. JSON/Schema Compliance Rates

[To be filled]

---

## 12. Output Length / Concision Analysis

[To be filled]

---

## 13. KL / Drift Analysis

[To be filled]

---

## 14. Qualitative Examples

[To be filled — from manual review samples]

---

## 15. Failure Cases

[To be filled]

---

## 16. What We Learned

[To be filled]

---

## 17. What Surprised Us

[To be filled]

---

## 18. What Remains Uncertain

[To be filled]

---

## 19. Next Experiments

[To be filled]

---

## 20. Hypothesis Verdicts

| Hypothesis | Status | Evidence |
|-----------|--------|----------|
| H1: Multi-turn concise is better | UNRESOLVED | Awaiting controlled ablation |
| H2: smol-magpie advantage is partly format | UNRESOLVED | Awaiting content-controlled comparison |
| H3: Small models prefer dense compact examples | UNRESOLVED | Awaiting verbose vs concise comparison |
| H4: Loss doesn't correlate with quality | UNRESOLVED | Awaiting loss vs judge score correlation |
| H5: Surgical LoRA preserves base model | UNRESOLVED | Awaiting KL drift comparison |
| H6: Structured terse wins on JSON/code | UNRESOLVED | Awaiting category-level analysis |
| H7: Small-model-native data style exists | UNRESOLVED | Awaiting format cluster analysis |

---

## Appendix: Experiment Registry

| Run ID | Format | Adapter | Status |
|--------|--------|---------|--------|
| lfm2_230m_format_ablation_alpaca_flat | alpaca_flat | quality | planned |
| lfm2_230m_format_ablation_single_turn_chat | single_turn_chat | quality | planned |
| lfm2_230m_format_ablation_multi_turn_concise | multi_turn_concise | quality | planned |
| lfm2_230m_format_ablation_multi_turn_verbose | multi_turn_verbose | quality | planned |
| lfm2_230m_format_ablation_structured_terse | structured_terse | quality | planned |
| lfm2_230m_format_ablation_bad_format_control | bad_format_control | quality | planned |
| lfm2_230m_surgical_multi_turn_concise | multi_turn_concise | surgical | planned |
| lfm2_230m_surgical_structured_terse | structured_terse | surgical | planned |
| lfm2_230m_quality_bsmagpie_v1 | multi_turn_concise | quality | planned |
| lfm2_230m_surgical_bsmagpie_v1 | multi_turn_concise | surgical | planned |
