# Overnight autonomy prompt — MiniCPM5-1B

Copy the prompt below into a new Codex task rooted at this repository.

```text
Continue the MiniCPM5-1B project autonomously overnight in:
/Users/bilawalriaz/autonomous-small-model-exploration

Mission
Make openbmb/MiniCPM5-1B an unusually strong, dependable *single-shot*
specialist for compact math, strict JSON/YAML/schema work, concise instruction
following, and executable one-shot code tasks. “World class” means excellent
for a 1B model on the frozen measured suite; do not claim general intelligence
or synergy without the required held-out evidence. Agent training is out of
scope until the single-shot release gate passes.

Start by reading, in full:
1. AGENTS.md
2. progress.md
3. experiments/registry.jsonl
4. reports/current_findings.md
5. reports/open_hypotheses.md
6. reports/decision_log.md
7. reports/negative_results.md
8. projects/minicpm5/README.md
9. projects/minicpm5/STATUS.md
10. reports/minicpm5_single_shot_master_plan.md
11. experiments/cards/M01.md, M02.md, M03.md and their claim cards/configs.

Known facts that must not be rediscovered incorrectly
- Existing MiniCPM math and formatting adapters are rank-8 RS-LoRA, alpha 16.
  Direct merge scaling is alpha/sqrt(rank), not alpha/r.
- M01 validated 168/168 adapted projections and FP16 equivalence of corrected
  direct merge with sequential native PEFT merge. Never use PEFT `linear` or
  `cat` for this pair.
- M02 repaired the llama.cpp GGUF harness. Use scripts/eval/run_gguf_eval.py:
  exact HF MiniCPM chat template, temperature 0, top_p 1, top_k 0, seed 42,
  explicit `<|im_end|>`, `</s>`, and `[end of text]` stops, raw stdout, and
  fixture validation. Do not report behavioral scores if fixture checks fail.
- M03 is a narrow, credible JSON-format result (+50pp; paired CI +25 to +75)
  with unchanged observed math/code/tool controls. Its 12-example control
  suites are underpowered: this is neither a regression finding nor synergy.
- Corrected merge: /home/billz/results/minicpm5_1b_direct_merge_rslora_m1.0_f0.7
  Q4: /home/billz/results/minicpm5_1b_direct_merge_rslora_m1.0_f0.7_gguf/MiniCPM5-1B.Q4_K_M.gguf

Authority and autonomy
- You are authorized to inspect, create, edit, run, resume, and verify all
  in-repository work and aero-host work needed for this mission overnight.
- Make reasonable reversible scientific choices without asking questions.
- Preserve existing user changes; do not reset, checkout, delete, overwrite,
  or mass-reformat unrelated work. Do not push, publish, open a PR, spend
  money, change credentials, or contact people/services except normal public
  dataset downloads and the already-authorized Mimo-v2.5 inference route.
- If Mimo is used, use it only for candidate generation. Deterministic parsing,
  JSON-schema/YAML validation, or executable tests must decide admission. Log
  model, prompt hash, output hash, validator/verifier version, and failure
  reason. Never use a mock judge.

First objective: complete Stage 0, then continue only through passed gates
1. Before any run, create `experiments/cards/S01.md` and a JSON config in
   `configs/` meeting every Phase 2 requirement in AGENTS.md.
2. Build versioned source manifests and deduplicated train/validation/held-out
   splits. Use source revisions and licenses. Prefer: GSM8K train (math),
   strictly validator-passing JSON/YAML/schema rows, a small licensed replay
   slice, and executable MBPP/permissively licensed CommitPackFT code rows.
   Keep benchmark/test rows disjoint from training.
3. Build frozen, deterministic single-shot evaluation suites with at least 200
   examples each for math, JSON/YAML/schema, concise instruction following,
   and executable code. Tool-call XML syntax is a neutral control only.
   Store raw prompts, expected outputs/tests, task provenance, and validators.
4. Validate every evaluator against known fixtures. Run corrected Q4 base and
   current merge on the new suites; record raw outputs, paired metrics, 20k
   paired-bootstrap CIs, integrity counts, and a human-review sample. This
   establishes the baseline; do not call it a training effect.
5. If data/evaluator audit passes, create cards/configs for the next smallest
   decision-relevant experiment. Train three seeds (42, 137, 2026) from the
   same base: schema, math-retention, code, and balanced single-shot candidates
   only as justified by the data budget. Select checkpoints/weights on
   validation; lock choices; evaluate held-out once.
6. Compare base, specialists, corrected direct merge, and balanced adapter.
   Prefer the smallest robust winner. Direct merge must use the validated
   RS-LoRA-aware path. A consolidation adapter is permitted only after
   complementary gains replicate.

Scientific gates
- Each new experiment needs a card, config, resumability check, run IDs, raw
  outputs, and registry entry before executing.
- Report paired wins/losses/ties and paired bootstrap CIs. Do not call an
  unchanged point estimate a regression merely because an underpowered CI is
  wide; increase suite size before applying a tight preservation budget.
- A positive result needs target gains across three seeds, held-out replication,
  non-target regression analysis, and a claim card. Record nulls honestly.
- Quantized Q4 behavior must agree with selected F16 behavior on a fixed smoke
  set before release claims.
- No agent trajectories, DPO, or multi-step tool-use training until the
  single-shot release-quality gate in the master plan is met.

Operational behavior
- Keep working until there is no safe, useful next action or a real external
  blocker occurs. Recover from interrupted work using the registry; do not
  rerun completed seed/task combinations without `--force`.
- Use aero for GPU work. Keep logs and checkpoints resumable. Avoid concurrent
  jobs that exceed its 8GB VRAM. Emit concise commentary updates at least once
  per hour or on major gate transitions.
- At every decision, favor a smaller controlled experiment over an exciting
  but confounded one.

Required shutdown
Update progress.md, experiments/registry.jsonl, reports/current_findings.md,
reports/open_hypotheses.md, reports/negative_results.md when relevant,
reports/artifact_index.md, projects/minicpm5/STATUS.md, and this project’s
master plan if the next priority changes. Verify JSONL validity, config/card
presence, relevant tests/fixtures, and git diff --check. End with a concise
evidence-based report naming exact files, completed run IDs, blockers, and the
single highest-value next action.
```
