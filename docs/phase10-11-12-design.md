# Phase 10 Design: Token-Budget-Controlled Data-Shape Ablation

## Status: DESIGN (not yet implemented)

## Motivation

Phase 9 showed multi-turn verbose has the lowest training loss (1.372 vs 1.831 for worst). But this could be explained by two competing hypotheses:

**H-A: Format genuinely helps.** Multi-turn conversational structure teaches the model better patterns for generating useful outputs.

**H-B: More tokens per example helps.** Verbose examples simply have more tokens, so the model sees more data per step (300 steps × batch 16 = 4800 gradient updates, but verbose examples provide more context per update). The advantage is about total training tokens seen, not format structure.

Phase 10 distinguishes H-A from H-B by controlling total training tokens across formats.

## Design

### Core experiment

For each of 3 formats (multi-turn verbose, multi-turn concise, alpaca flat):
1. Measure tokens per example in each format
2. Adjust the number of examples or training steps to equalize total tokens
3. Train and evaluate

Three conditions:
- **Token-matched:** Same total tokens across formats. Verbose gets fewer examples (or fewer steps) to match concise/flat.
- **Example-matched:** Same number of examples across formats (current Phase 9 design). Already done.
- **Step-matched:** Same training steps but different total tokens (current Phase 9 design). Already done.

By comparing token-matched vs example-matched results, we can isolate the "more tokens" effect.

### Specific configs

Assume canonical dataset has 300 examples. Measure average tokens per example per format:
- alpaca_flat: ~80 tokens/example → 24,000 total tokens
- multi_turn_concise: ~120 tokens/example → 36,000 total tokens
- multi_turn_verbose: ~200 tokens/example → 60,000 total tokens

Token-matched condition (match to alpaca_flat at 24,000 tokens):
- alpaca_flat: 300 examples, 300 steps (baseline)
- multi_turn_concise: 200 examples, 300 steps (or 300 examples, 200 steps)
- multi_turn_verbose: 120 examples, 300 steps (or 300 examples, 120 steps)

### Controls
1. Same canonical content (subset of Phase 9 canonical)
2. Same model (LFM2.5-230M)
3. Same LoRA config (quality: r=8, hub all)
4. Same optimizer (Adafactor, lr=2e-4)
5. Same eval set (153 prompts)
6. Same decoding settings
7. Seed 42

### Metrics
1. Training loss (already measured in Phase 9)
2. Programmatic eval metrics (JSON validity, entity F1, exact match)
3. Judge scores (pointwise + pairwise)
4. Blind review

### Hypotheses

**H10-1: Token budget explains the verbose advantage.**
If verbose loses its advantage when total tokens are equalized, H-B is confirmed.
Falsifier: If verbose still wins on token-matched condition, format genuinely helps.

**H10-2: Example count matters more than token count.**
If example-matched (same N examples, different tokens) beats token-matched (same tokens, different N), learning from distinct examples matters more than seeing more tokens.
Falsifier: If token-matched beats example-matched, total data volume matters more.

**H10-3: Format interaction with token budget is non-linear.**
If the optimal format depends on token budget (verbose better at high budget, concise better at low budget), there's an interaction effect.

## Implementation plan

1. Tokenize Phase 9 canonical data in each format → measure tokens/example
2. Create token-matched subsets (fewer examples for verbose, same for flat)
3. Train 3 token-matched adapters
4. Run Phase 9R eval pipeline on all
5. Compare token-matched vs example-matched results

## Estimated compute

3 adapters × 300 steps × ~10min each = ~30 minutes on aero.
Plus eval: 3 × 153 prompts × ~2s each = ~15 minutes.
Total: ~1 hour on aero.

---

# Phase 11 Design: Steering Vectors for DSL Adherence

## Motivation

When a small model (e.g., 0.5B) is asked to produce output matching a specific DSL (JSON schema, structured format, specific output template), it relies on internal representations at hub layers to maintain formatting discipline. If we can extract the steering vectors that control this DSL adherence, we can dynamically boost the model's formatting reliability at inference — achieving the parsing efficiency of a dedicated smaller model without a separate network.

## Approach

1. **Identify DSL-relevant layers:** Use Phase 1 MI-Atlas data to identify which layers are most responsible for JSON/format compliance (L2, L6, L7, L9, L12, L13 based on dataset shard ablation).

2. **Collect contrastive pairs:**
   - Compliant outputs (valid JSON matching schema)
   - Non-compliant outputs (invalid JSON, wrong schema)
   - Same prompts, different generation seeds

3. **Extract steering vectors:**
   - Compute mean activation difference at hub layers between compliant and non-compliant runs
   - Validate by steering: add vector → does compliance rate improve?
   - Validate by anti-steering: subtract vector → does compliance rate degrade?

4. **Test on 0.5B model:**
   - Steering vectors at L2 for JSON schema compliance
   - Measure: JSON validity rate, schema compliance rate, output quality (non-DSL content should not degrade)

5. **Compare to dedicated small model:**
   - If steering achieves >90% compliance rate, it matches a dedicated 25M parser
   - But the 0.5B model retains general capability — the parser doesn't

## Expected results

- Steering at L2 should boost JSON compliance by 20-50%
- Safe steering range: s ∈ [+0.5, +2.0] (from Phase 1 steering sweep)
- Anti-steering should degrade compliance (confirming causality)
- Steering should NOT affect content quality (only formatting)

## Implementation plan

1. Build contrastive dataset: 100 JSON extraction prompts × 5 compliant + 5 non-compliant completions each
2. Record activations at L0, L2, L6, L7, L9, L12, L13 for all runs
3. Compute steering vectors per layer
4. Validate with steering sweep (s = -2 to +2, step 0.5)
5. Test on held-out prompts

## Estimated compute

~2 hours on aero (0.5B model, inference only — no training).

---

# Phase 12 Design: Circuit Shifts via Training Perturbations

## Motivation

When we fine-tune a small model with LoRA for specialized workflows (JSON extraction, code deobfuscation, multi-step reasoning), does the fine-tuning:
(a) Create entirely new causal circuits, or
(b) Simply upweight existing baseline circuits?

Mapping the internal states pre- and post-LoRA will validate the structural impact of perturbations and inform future model surgery.

## Approach

1. **Baseline causal atlas:** Use Phase 1-3 MI-Atlas data for Qwen2.5-0.5B as the pre-training baseline. Full layer ablation, head ablation, MLP ablation, steering maps.

2. **Post-training atlas:** After LoRA training on a specific task (e.g., JSON schema), repeat the same atlas experiments:
   - Layer ablation (which layers matter now?)
   - Head ablation (which heads are repurposed?)
   - MLP ablation (which MLPs carry the new skill?)
   - Steering (which layers respond to steering?)

3. **Compare atlases:**
   - Delta maps: pre-training vs post-training ablation effects
   - Hub migration: does the hub layer shift after training?
   - New circuit detection: are there layers/heads that become important ONLY after training?
   - Preservation: do baseline capabilities (factual recall, copying) survive?

4. **Specific questions:**
   - Does LoRA create new attention heads, or repurpose existing ones?
   - Do the same MLPs handle both base and trained behaviors?
   - Is there a "training budget" — a limit to how many circuits can be modified?

## Expected results

- Based on Phase 3 checkpoint timeline: core circuit (L2/L7/L9) locks in by step 10
- Based on dataset shard ablation: each skill concentrates in different layers
- Prediction: LoRA upweights existing circuits rather than creating new ones
- Prediction: the hub layer position does not change after training (only importance weights change)

## Implementation plan

1. Run baseline atlas on base LFM2.5-230M (if not already done)
2. Run post-training atlas on best adapter (multi-turn verbose quality)
3. Compute delta maps
4. Test on 2 more adapters (alpaca_flat, structured_terse) to see if format affects circuit creation

## Estimated compute

~8 hours on aero (full atlas per adapter: 24 layers × 12 tasks × multiple methods).
