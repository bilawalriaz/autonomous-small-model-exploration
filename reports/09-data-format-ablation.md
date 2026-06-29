# Phase 9: Data Format Ablation

## Executive Summary

Phase 8 showed that dataset format dominates hyperparameters for LFM2.5-230M SFT (5x more impact). Phase 9 isolates this finding through a controlled experiment: same canonical content rendered into 6 training formats, evaluated with a permanent judge-based harness.

**Status:** Training complete. Evaluation infrastructure rebuilt (Phase 9R). Awaiting real eval harness + judge runs on aero.

**Evidence tier of current claims:**
- Training loss rankings: VALID (reproducible, deterministic)
- Behavioral claims (win-rate, quality): UNVERIFIED — current eval data uses mock judge only
- Programmatic metrics (JSON validity, slop, length): PENDING — no eval outputs generated yet

---

## 0. Phase 9R: What Changed and Why

The original Phase 9 report contained behavioral claims (win-rates, judge scores, hypothesis verdicts) based on mock-judge scoring — deterministic random numbers that look like real scores but carry no behavioral signal. Phase 9R fixes this:

### Evaluation stack fixes (completed)
1. `judge_outputs.py`: mock judging now requires explicit `--mock` flag; `judge_source` metadata on every score; `hashlib`-based deterministic seeding (not Python `hash()`); `--strict-report-mode` for publishable runs
2. `aggregate_eval_results.py`: programmatic scorers for JSON validity, schema validity, entity F1, exact-match factual, numeric match, slop rate, output length, constraint-following; `--judge-source` filter; judge_source in every aggregate
3. `generate_blind_review.py`: new script for stratified blind review — 60+ examples across 9 categories, anonymized model labels, unblinding key stored separately
4. `run_phase9r_eval.py`: one-command pipeline for eval + judge + aggregate + blind review on aero

### What remains (requires aero GPU)
1. Run eval harness on all 8 adapters + base model
2. Run real judge (or document mock judge limitations)
3. Generate blind review samples
4. Compute programmatic metrics
5. Update this report with real data

---

## 1. What Was Tested

Six training formats applied to identical canonical content (300 examples spanning 9 domains):

1. Alpaca flat (instruction/input/output)
2. Single-turn ChatML
3. Multi-turn concise
4. Multi-turn verbose
5. Structured terse
6. Bad-format negative control

Plus bilawal_smol_magpie_v1 (345 examples, curated mixture) in both quality and surgical adapter configs.

Each format trained with frozen quality config (Adafactor, lr=2e-4, r=8, hub all modules, 300 steps) and evaluated with the same permanent eval harness (153 prompts, 9 categories, judge-based scoring).

---

## 2. Why Loss Alone Is Insufficient

Training loss measures how well the model fits the training distribution. Different formats have different inherent "easiness" — flat verbose text is easier to model than structured terse output. Lower loss on Format A vs Format B could mean:
- Format A is genuinely better learned, OR
- Format A is inherently easier to predict

Therefore we measure:
- Behavioral win-rates (blind pairwise judging)
- Programmatic metrics (JSON validity, entity F1, exact match, numeric match)
- Category-level instruction following
- Output length and concision
- Hallucination tendency
- Slop phrase rate
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
- Seed: 42

### Surgical Adapter
- Same as quality except:
- Target modules: o_proj only (hub-targeted)
- ~70% fewer parameters (65K vs 245K)

---

## 4. Eval Harness Design

Permanent eval set: `data/eval/small_model_eval_v1.jsonl`
- 153 prompts across 9 categories
- Categories: instruction following, JSON/structured, GameFAQ extraction, coding, deobfuscation, reasoning, concision/anti-slop, factual Q&A, multi-turn
- Each prompt has hard constraints and a scoring rubric

Generation settings: temp=0.2, top_p=0.9, max_new_tokens=512, seed=42

---

## 5. Controlled Format Ablation: Training Loss Results

These are real, reproducible training losses. No judge or behavioral inference involved.

| Format | Final Loss | Runtime | Rank | Relative to Best |
|--------|-----------|---------|------|-----------------|
| multi_turn_verbose | 1.3724 | 569s (~9.5min) | 1 | baseline |
| bad_format_control | 1.4023 | 590s (~10min) | 2 | +2.2% |
| multi_turn_concise | 1.5156 | 1005s (~17min) | 3 | +10.4% |
| alpaca_flat | 1.7321 | 895s (~15min) | 4 | +26.2% |
| single_turn_chat | 1.7475 | 400s (~7min) | 5 | +27.3% |
| structured_terse | 1.8314 | 440s (~7min) | 6 | +33.4% |

Loss gap: 0.459 between best and worst (33.4% relative difference).

**What this proves:** Training loss differs significantly by format under content-controlled conditions. Multi-turn verbose achieves the lowest loss. The gap is large enough to be meaningful (not noise).

**What this does NOT prove:** That multi-turn verbose produces better outputs. Loss and quality can decouple (see H4 below).

---

## 6. bilawal_smol_magpie_v1 Training Loss Results

| Adapter | Params | Loss |
|---------|--------|------|
| Surgical (out_proj only) | ~65K | 1.2714 |
| Quality (hub all modules) | 245K | 1.4642 |

Surgical beats quality on training loss with 3.8x fewer parameters.

---

## 7. Hypotheses: Evidence Status

| Hypothesis | Status | Evidence Level | Notes |
|-----------|--------|---------------|-------|
| H1: Multi-turn concise is genuinely better | REJECTED (loss only) | Training loss | Verbose beats concise on loss when content is held constant. Behavioral evidence pending. |
| H2: smol-magpie advantage is partly format | PLAUSIBLE | Training loss | Multi-turn formats have lower loss than flat. But content/format interaction not fully isolated. |
| H3: Small models prefer dense compact examples | REJECTED (loss only) | Training loss | Verbose > concise > structured terse on loss. More context helps at 230M scale. |
| H4: Loss doesn't correlate with quality | PLAUSIBLE | Loss ranks | bad_format_control has 2nd-best loss but should be worst on quality. Need real eval to confirm. |
| H5: Surgical LoRA preserves base model | PLAUSIBLE | Training loss | Surgical bsmagpie beats quality bsmagpie on loss with 3.8x fewer params. Need KL drift + eval. |
| H6: Structured terse wins on JSON/code | UNRESOLVED | None | No behavioral data. Loss suggests otherwise (structured_terse is worst). |
| H7: Small-model-native data style exists | PLAUSIBLE | Training loss | Multi-turn verbose is the loss winner. But "native" requires behavioral confirmation. |

**IMPORTANT CAVEAT:** All current verdicts are based on training loss ONLY. Training loss ranks may not transfer to behavioral quality. The original Phase 9 report's behavioral claims (win-rates, judge scores) were generated by a mock judge (deterministic random numbers) and should be treated as "pipeline validation only, not behavioural evidence."

---

## 8. What Remains to Be Done (Phase 9R)

### On aero (GPU required):
```bash
# 1. Full eval run with real judge
python scripts/eval/run_phase9r_eval.py \
    --judge-api-url http://localhost:8080 \
    --judge-api-key $JUDGE_API_KEY

# Or with mock judge for pipeline validation only:
python scripts/eval/run_phase9r_eval.py --mock-judge

# 2. After eval completes, update this report with real data
python scripts/report/build_phase09_report.py
```

### What the real eval will provide:
1. **Programmatic metrics:** JSON validity, schema validity, entity F1, exact match, numeric match, slop rate, output length — these are deterministic and don't need a judge
2. **Judge scores (if API available):** Pointwise (1-5 on 7 dimensions) and pairwise (win/tie/loss) — these require a real judge model
3. **Blind review samples:** 60+ examples stratified across 9 categories — for human verification

### What we can claim after real eval:
- If programmatic metrics confirm loss rankings: strong evidence
- If judge scores confirm loss rankings: strong evidence  
- If judge scores contradict loss rankings: H4 confirmed (loss ≠ quality)
- If blind review contradicts judge: meta-judge calibration data

---

## 9. Category-Level Predictions

Based on training loss, we predict:
- **JSON/structured:** Alpaca flat and single-turn chat may win (simpler output format, less multi-turn context to confuse formatting)
- **Coding:** Multi-turn verbose may win (more context for code explanation)
- **Reasoning:** Multi-turn verbose may win (more chain-of-thought examples in training)
- **Concision:** Structured terse may win (compact training data → compact outputs)
- **Factual Q&A:** Hard to predict — depends on whether verbose context helps or distracts

These are predictions. Real eval data will confirm or reject them.

---

## 10. KL Drift Analysis

Pending. Requires eval outputs to compute output-length drift, refusal rate, and repetition rate relative to base model.

---

## 11. Manual Review Samples

Pending. After eval harness runs, generate blind review:
```bash
python scripts/eval/generate_blind_review.py \
    --run-ids lfm2_230m_base_20260629 \
              lfm2_230m_quality_multi_turn_verbose_20260629 \
              lfm2_230m_quality_alpaca_flat_20260629 \
              lfm2_230m_quality_structured_terse_20260629 \
    --min-per-category 7 \
    --output results/evals/blind_review_phase9r.md
```

---

## 12. Next Experiments

### Phase 10: Token-Budget-Controlled Data-Shape Ablation

Phase 9 showed multi-turn verbose has the lowest training loss. But is that because:
(a) the verbose format provides more useful context for learning, OR
(b) verbose examples simply have more tokens, so the model sees more data per step?

Phase 10 controls for total training tokens:
- Same total token budget across all formats
- Measure: does verbose still win when token count is equalized?
- If yes → format genuinely helps
- If no → the advantage is just more tokens

### Steering Vectors for DSL Adherence

Extract steering vectors that control formatting/discipline at the 0.5B model's hub layers (L2). If a 0.5B model can be steered to reliably produce structured output matching a DSL, it achieves the parsing efficiency of a dedicated 25M model without a separate network.

### Circuit Shift Analysis via Training Perturbation

Track how causal graphs shift when applying LoRA/SFT for specialized workflows. Does fine-tuning create new circuits or upweight existing baseline circuits? Map internal states pre- and post-LoRA.

---

## Appendix: Experiment Registry

| Run ID | Format | Adapter | Training Loss | Eval Status |
|--------|--------|---------|--------------|-------------|
| lfm2_230m_base_20260629 | base | none | N/A | Pending |
| lfm2_230m_quality_alpaca_flat_20260629 | alpaca_flat | quality | 1.7321 | Pending |
| lfm2_230m_quality_single_turn_chat_20260629 | single_turn_chat | quality | 1.7475 | Pending |
| lfm2_230m_quality_multi_turn_concise_20260629 | multi_turn_concise | quality | 1.5156 | Pending |
| lfm2_230m_quality_multi_turn_verbose_20260629 | multi_turn_verbose | quality | 1.3724 | Pending |
| lfm2_230m_quality_structured_terse_20260629 | structured_terse | quality | 1.8314 | Pending |
| lfm2_230m_quality_bad_format_control_20260629 | bad_format_control | quality | 1.4023 | Pending |
| lfm2_230m_quality_bsmagpie_surgical_20260629 | multi_turn_verbose (bsmagpie) | quality | 1.4642 | Pending |
| lfm2_230m_surgical_bsmagpie_surgical_20260629 | multi_turn_verbose (bsmagpie) | surgical | 1.2714 | Pending |

---

## Appendix: Evidence Ladder for Phase 9 Claims

| Claim | Evidence Type | Current Level | Required for Publication |
|-------|-------------|---------------|------------------------|
| Multi-turn verbose has lowest loss | Training loss | CONFIRMED | Already sufficient |
| Loss ranks don't predict quality | Judge + programmatic | UNVERIFIED | Need real eval |
| Verbose wins behaviorally | Judge + manual review | UNVERIFIED | Need real eval |
| Surgical preserves base model | KL drift + eval | UNVERIFIED | Need real eval |
| Format matters more than hyperparams | Cross-format loss spread | PLAUSIBLE (from Phase 8+9) | Need real eval confirmation |

---

## Appendix: Mock Judge Caveat

The original Phase 9 HTML report (docs/09-data-format-ablation.html) contains behavioral data (win-rates, judge scores, category scores) generated by a mock judge. These values are deterministic random numbers seeded by eval_id — they carry NO information about actual model quality. Any section of that report referencing "win-rate", "judge overall", or "judge-based" metrics should be read as:

**"Pipeline validation only, not behavioural evidence. These numbers demonstrate that the eval pipeline produces structured output, not that one format outperforms another on behavioral metrics."**

To upgrade these claims, rerun the pipeline with `--judge-api-url` pointing to a real judge model.
