# MiniCPM5-1B single-shot master plan

## North star

Make `openbmb/MiniCPM5-1B` unusually dependable for compact single-shot work:
math, strict JSON/YAML/schema transformations, concise instruction following,
and small executable code changes. The goal is *world-class for its size on
these measured tasks*, not an unsupported claim of universal capability.

Agent training is explicitly deferred. It begins only after the frozen
single-shot checkpoint passes the readiness gate below.

## Non-negotiables

- The MiniCPM HF chat template is canonical. Q4 evaluation uses M02's exact
  template, greedy decoding, explicit stops, raw stdout, and fixture checks.
- Use direct RS-LoRA merging or sequential native PEFT only; never PEFT `cat`
  or `linear` for this adapter pair without a new equivalence validation.
- No mock judge. Math, JSON/YAML, code, and tool-format metrics must be
  deterministic or verifier-backed.
- Split before generation/training. Validation selects weights; only the
  locked choice reaches held-out evaluation.
- Every run has a Phase 2 card, JSON config, seed-specific registry rows,
  raw outputs, and a claim card when it affects a decision.

## Correct interpretation of M03

M03 is an encouraging format result: corrected merge JSON validity improved
from 6/12 to 12/12 (+50pp, paired 95% CI +25 to +75). Math, code, and tool
point estimates were unchanged. Its 12-example controls are too small to
certify a −10pp preservation budget, but they do **not** demonstrate a
regression. Treat M03 as format improvement plus neutral observed controls;
repeat it with properly powered suites before changing training direction.

## Stage 0 — measurement and data admission

### Deliverables

1. Freeze versioned `train`, `validation`, and `heldout` manifests before
   training. Deduplicate across every split by normalized prompt, answer, and
   near-duplicate embedding/hash checks.
2. Expand deterministic single-shot evaluation to 200 examples per family:
   math, JSON/YAML/schema, concise instruction constraints, and executable
   Python/code repair. Keep tool-call *syntax* as a neutral fifth control.
3. Store each evaluator, expected result, verifier version, prompt hash, and
   dataset source/revision. Do not include evaluation templates in SFT.

### Data admission policy

| Family | Primary source | Admission condition |
|---|---|---|
| Math | `openai/gsm8k` train, MIT | Exact answer extracted and independently recomputed where possible. GSM8K test remains held out. |
| Schema | Existing strict validated rows + generated gaps | Parse and JSON-schema/YAML validator must pass; examples must be compact. |
| General replay | H4 No Robots, CC-BY-NC-4.0, only if license suits the intended use | Preserve concise natural-language instruction following; no need for a large slice. |
| Code | MBPP plus selected permissively licensed CommitPackFT rows | Execute tests in a sandbox. Store tests, return code, and source license. |
| Tool syntax only | xLAM-style public calls | Convert to MiniCPM XML exactly; validator must parse. No claim of agent capability. |

Mimo-v2.5 augmentation is allowed only for missing coverage. Each output is a
candidate, never a label: deterministic parser/test/schema validation decides
admission. Save model/version, prompt hash, candidate hash, validator result,
and failure reason. Spend calls on hard schema edge cases and executable code
tasks, not bulk unverified prose.

### Readiness gate

Run the fixed base and corrected merge on the larger suite. This establishes
the honest baseline and the validation data needed for weight selection; it is
not a training-effect experiment.

## Stage 1 — targeted single-shot adapters (three seeds)

Train from the same frozen base with seeds **42, 137, 2026**. Keep rank,
targets, precision, sequence length, optimizer, decoding, and step budget
fixed across candidates.

1. **Schema adapter:** JSON/YAML/schema validity, exact field preservation,
   compact output.
2. **Math-retention adapter:** verified GSM8K-style work; optimize final
   answers, not unbounded hidden reasoning verbosity.
3. **Code adapter:** only executable one-shot tasks (function completion,
   small patch, output prediction); tests are the judge.
4. **Balanced single-shot adapter:** a domain-balanced mixture of the first
   three plus concise replay. This is the likely release candidate and is
   evaluated against the direct-merge route, not assumed better.

For each adapter, use a validation split to choose checkpoint/early stopping;
report the locked held-out result once. A positive target lower confidence
bound is required. Non-target preservation is assessed by a sufficiently large
suite and both an observed-delta budget and confidence interval—not a tiny
suite whose CI makes any −10pp budget impossible to pass.

## Stage 2 — selection and consolidation

For each seed, compare base, each specialist, corrected direct merge, and the
balanced adapter. Select weights on validation only using a constrained utility:

`schema + math + code gains`, subject to no material degradation in the other
single-shot families.

Then test **one** locked candidate on held-out data. If direct merging wins,
retain direct merging. If the balanced adapter wins, prefer it: one coherent
adapter is simpler and less error-prone than a pile of deltas. Consider a
low-LR consolidation LoRA only if both paths show complementary, replicated
gains.

## Stage 3 — release-quality single-shot gate

The candidate advances only when all are true:

- Three-seed target gains replicate with effect-size standard deviation below
  20% of the mean effect where a positive claim is made.
- Exact JSON/YAML/schema and code-test results improve over base with 95% CIs.
- Math does not materially regress on both GSM8K-style and independently
  generated verifier-backed arithmetic.
- Q4 behavior agrees with the F16 selected candidate on a fixed smoke suite.
- Raw prompts, outputs, manifests, adapter hashes, merge weights, and license
  inventory are published in the artifact index.

If a candidate fails, preserve it as a negative result and change one factor
only: data composition, target modules, rank, or training budget.

## Stage 4 — agent capability (only after Stage 3)

Start with tool-call syntax and short observation/action loops, then move to
the existing verifier-backed trajectory corpus. Preserve 30–40% single-shot
replay in every agent blend. Score agent completion, recovery, and efficiency
separately; an agent score may never replace a single-shot score.

## First next session

Create the Stage 0 experiment card/config, build the 200-per-family frozen
suite and data manifests, validate source licenses/revisions, and run only the
baseline/merge measurement. Do not train until the audit passes.

## Continuation protocol

At session start read `AGENTS.md`, `progress.md`, the registry, current
findings, hypotheses, decision log, negative results, this file, and the most
recent MiniCPM card/claim. At shutdown update the required six files plus this
plan's “First next session” line/status if the priority changed.
