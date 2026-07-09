# Negative Results

## NR001: Full SFT OOMs on 8GB VRAM
Experiment: Attempted full supervised fine-tuning of Qwen2.5-0.5B on JSON schema data.
Expected: Model + optimizer + gradients fit in 8GB VRAM with bf16 and gradient checkpointing.
Observed: OOM at first training step.
Interpretation: 0.5B model with AdamW optimizer needs ~6-7GB for training. 8GB is marginal.
Next: Use LoRA (r=8) which reduces optimizer states to ~50MB.

---

## NR002: Full-residual activation patching gives KL=0 everywhere
Experiment: Patch full residual stream at each layer from clean run into corrupt run.
Expected: Some layers show high recovery, others low.
Observed: KL=0 at every layer — full-residual patching trivially restores clean computation.
Next: Position-specific and component-specific patching needed.

---

## NR003: Clean/corrupt pair v0 tokenization misalignment
Experiment: Initial clean/corrupt pairs had multi-token targets.
Expected: Target token is single-token.
Observed: Some targets encode as 2+ tokens. Logprob scoring only uses first token.
Next: Built pairs_v1.json with verified single-token targets.

---

## NR004: H002 (universal L0-L2 concentration) rejected
Experiment: Dataset shard ablation — 5 skill families.
Expected: All families concentrate into L0-L2 after LoRA training.
Observed: Each family concentrates in different layers (factual_recall: L3/16/19, code: L1/10/21, json: L6/12/13).
Next: Skill-specific analysis required.

---

## NR005: Extreme steering causes degeneration
Experiment: Steering sweep at L2 with factual direction, strengths -8 to +8.
Expected: Monotonic improvement with positive steering.
Observed: At s>=+2, model generates Chinese characters and repetitive garbage.
Next: Find and respect the steering budget.

---

## NR006: L2 is position-dependent, not uniform
Experiment: Position-specific ablation at L2.
Expected: L2 effect is uniform across all positions.
Observed: L2 specifically handles first tokens (instruction) and last tokens (prediction). Operators near-zero.
Next: L2 has positional specialization, not simple residual magnitude.

---

## NR007: JSON skill knockout had limited effect
Experiment: Skill knockout via negative steering on JSON skill.
Expected: Negative steering suppresses JSON-specific token probabilities.
Observed: Base probability of JSON targets already near-zero — no room to suppress.
Next: Use prompts where adapter demonstrably changes target probability.

---

## NR008: H6 (upstream propagation) rejected
Experiment: Adapter-only ablation — remove adapter contribution at each layer.
Expected: Adapter ablation at early layers would have large effects (upstream propagation).
Observed: Norm-effect correlation = 0.85. Adapter effects ARE at same layers where norms peak (L19-L23).
Next: The separation between general importance (L0-L2) and adapter-specific importance (L19-L23) is the finding.

---

## NR009: PeftModel.from_pretrained modifies base model in-place
Experiment: Loading base and trained models as separate objects.
Expected: Two independent models.
Observed: PeftModel wraps and modifies base model in-place. Both objects produce adapter-active results.
Next: Use `disable_adapter()` context manager.

---

## NR010: SmolLM2 shows flat ablation profile — no identifiable hub
Experiment: Layer ablation on SmolLM2-1.7B.
Expected: Clear hub layer similar to Qwen2.5.
Observed: All 24 layers show IDENTICAL ablation effects within each family.
Next: Architecture-specific hub analysis required.

---

## NR011: 3B patching/skip/knockout fail with tensor dimension mismatch
Experiment: Cross-model patching on Qwen2.5-3B.
Expected: Same experiments as 0.5B/1.5B.
Observed: RuntimeError — GQA dimension mismatch (4 KV heads vs 14).
Next: Read head dimensions from model.config. Parameterize all code.

---

## NR012: 1.5B LoRA adapter cannot load into different-scale model
Experiment: Cross-scale adapter loading.
Expected: Adapter would produce measurable effects.
Observed: State dict size mismatch — A matrix dimensions match d_model per scale.
Next: Train separate adapters per scale.

---

## NR013: Gaussian resample preserves layer ranking
Experiment: Six ablation methods compared at 0.5B and 1.5B.
Expected: Different methods might change rank ordering.
Observed: Zero ≈ mean (mean activation ≈ 0). Gaussian resample preserves ranking with higher variance.
Next: Use zero ablation for efficiency.

---

## NR014: Mock judge produces plausible-looking but meaningless scores (Phase 9R)

Experiment:
Phase 9 eval pipeline used mock judge (deterministic random numbers seeded by eval_id) when API was unavailable. The pipeline silently fell back to mock without flagging it.

Expected:
Mock scores would be clearly distinguishable from real judge scores.

Observed:
Mock scores look plausible (1-5 range, some variation by category). The HTML report presented mock-judge win-rates and judge scores as if they were real behavioral evidence. Python's `hash()` was used for seeding — non-deterministic across Python versions (PYTHONHASHSEED randomization).

Interpretation:
Mock-judge scores are structurally indistinguishable from real scores in aggregate. The silent fallback was a pipeline integrity bug.

What this rules out:
Using mock judge without explicit flagging. Using Python hash() for deterministic seeding. Silent fallback from API to mock.

Next:
Phase 9R fixed this: explicit `--mock` flag required, `judge_source` metadata on every score, `hashlib`-based seeding, `--strict-report-mode`. All existing mock-judge results caveated as "pipeline validation only, not behavioural evidence."

---

## NR015: bad_format_control has unexpectedly low training loss (Phase 9)

Experiment:
Content-controlled format ablation on LFM2.5-230M. Bad-format-control (deliberately verbose, filler-heavy data) achieved 2nd-best training loss (1.402).

Expected:
Bad-format-control should have the worst or near-worst loss as a negative control.

Observed:
Loss rank: multi_turn_verbose (1.372) > bad_format_control (1.402) > multi_turn_concise (1.516) > alpaca_flat (1.732) > single_turn_chat (1.748) > structured_terse (1.831).

Interpretation:
Predictable, repetitive text is inherently easy for language models to predict. Low loss reflects statistical regularity, not quality. Strongest training-loss-only evidence that loss ≠ quality.

What this rules out:
Using training loss as a proxy for output quality without behavioral verification.

Next:
Run real eval to confirm: if bad_format_control has worst behavioral scores despite 2nd-best loss, H-P9-4 is confirmed.

---

## NR016: Teacher-clean labels still fail deterministic schema validation

Experiment:
Mimo-v2.5 judged 1,728 rollout prompt groups after prompt tightening. Rows were filtered to teacher-clean labels using `format_valid=true`, `quality_score>=8`, and `correctness in {correct, na}`.

Expected:
Most teacher-clean labels would parse and satisfy the prompt metadata JSON schema.

Observed:
Teacher-clean q>=8 rows: 1,240. Deterministic parse + JSON-schema validation accepted only 827 and rejected 413. Major rejection modes included YAML parser failures, JSON parse failures, wrong wrapper/top-level shape, enum violations, range violations, count violations, and missing required fields.

Interpretation:
The teacher is useful for ranking and reasoning condensation, but it still over-accepts structured outputs. For structured-output SFT, teacher labels must be gated by deterministic validators before training.

What this rules out:
Using teacher `format_valid=true` and high `quality_score` as the sole criterion for training-data quality.

Next:
Use `/Users/bilawalriaz/scored/exports/sft_strict_q8_response.jsonl` or `/Users/bilawalriaz/scored/exports/sft_strict_q8_reasoning.jsonl` for the first SFT pass. Keep teacher-clean-but-strict-rejected rows quarantined for repair or negative/preference data.

---

## NR017: Current LFM2.5-8B-A1B GGUF does not drive Hermes tools in first smoke

Experiment:
Configured Hermes on lenovo to use aero's OpenAI-compatible llama.cpp endpoint for `LFM2.5-8B-A1B-Uncensored-Gaston-Q4_K_M.gguf`, then ran `agent_code_bugfix_001` through `tools/agent_trajectory_lab/run_hermes_tasks.py`.

Expected:
Hermes would use shell/file tools to inspect `stats.py`, edit the median and trimmed-mean bugs, and pass the verifier.

Observed:
Hermes reached the aero endpoint and returned successfully, but the model emitted a JSON plan/tool-call-like object instead of completing executable tool calls. No workspace diff was produced. The verifier failed on the original bugs. A control run with Nous `stepfun/step-3.7-flash:free` edited `stats.py` and passed the same verifier.

Interpretation:
The trajectory harness and Hermes tools work, but the current LFM server/model/chat-template combination is not yet tool-call compatible enough for unattended Hermes agent traces.

What this rules out:
Assuming that a raw OpenAI-compatible LFM2.5-8B-A1B endpoint can immediately collect high-quality Hermes tool-use trajectories without a tool-format bootstrap or adapter.

Next:
Test alternate llama.cpp chat/tool templates and non-streaming settings. If still failing, collect successful teacher traces with a tool-capable provider, then SFT/ DPO LFM on those traces before using it as the trajectory generator.

---

## NR018: LFM2.5-8B-A1B Unsloth QLoRA does not fit on aero 8GB GPU

Experiment:
Stopped the aero llama.cpp LFM server, then attempted to load `LiquidAI/LFM2.5-8B-A1B` through Unsloth for bnb 4-bit QLoRA SFT using the strict 827-row scored dataset. Config snapshot: `configs/sft/lfm25_8b_a1b_unsloth_qlora_8gb.json`.

Expected:
The 4-bit base model plus rank-8 LoRA setup would load on aero's RTX 2070-class 8GB GPU, allowing batch size 1 with gradient accumulation.

Observed:
Default Unsloth/bnb placement failed before training because modules were dispatched to CPU/disk. Raising `gpu_memory_utilization` to `0.9` failed the same way. Forcing all quantized modules onto CUDA got further, but failed during weight loading with CUDA OOM at about 7.59 GiB used on a 7.604 GiB usable GPU. Adding `--offload-embedding`, reducing context to 768, and setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` still failed during weight loading.

Interpretation:
With the current aero software stack (`unsloth` 2026.5.2, `transformers` 5.3.0, bnb 4-bit) and RTX 2070-class 8GB VRAM, the HF `LiquidAI/LFM2.5-8B-A1B` checkpoint cannot be loaded for QLoRA training before activations or optimizer state are added.

What this rules out:
The same Unsloth bnb 4-bit 8GB recipe as a viable path for first LFM2.5-8B-A1B SFT on aero.

Next:
Use a smaller checkpoint, a larger GPU, or a training stack with practical CPU/NVMe offload. Keep `scripts/train/train_lfm25_8b_unsloth_qlora.py` for larger hardware or future offload-capable stack tests.

---

## NR019: Eight parallel HY3 Hermes workers trigger rate-limit artifacts

Experiment:
Started four family-sorted HY3 workers and then four additional balanced HY3 workers on lenovo, all using `tencent/hy3:free` through Nous Portal with four rollouts per task.

Expected:
Eight workers would increase throughput while preserving clean verifier-backed trajectories.

Observed:
The run began cleanly, but at eight workers several Hermes invocations returned final stdout `API call failed after 3 retries: HTTP 429...` with return code 0. Because no task files were produced, downstream verifiers failed with missing-output errors. These are provider/rate-limit artifacts, not model-quality failures.

Interpretation:
HY3 collection needs lower concurrency and explicit transient retry/backoff. Rate-limit artifacts must be excluded from DPO rejected examples because they are infrastructure failures, not invalid model trajectories.

What this rules out:
Treating every failed verifier row as a useful rejected preference example. Running eight HY3 workers without backoff on the current Nous Portal free route.

Next:
Use balanced shards with four workers, `--transient-retries 4 --transient-sleep 90`, and `--rerun-failed`. Keep `export_preference_pairs.py` defaulting to exclude transient API failures.

---

## NR020: Format-heavy SFT causes math reasoning regression on LFM2.5-1.2B-Instruct

Experiment:
Ran GSM8K evaluation (100 prompts) on `lfm25_12b_instruct_sft_q8_strict` GGUF and compared against the base `LFM2.5-1.2B-Instruct` GGUF.

Expected:
The SFT model would either maintain or slightly improve general reasoning performance compared to the base model.

Observed:
The SFT model accuracy on GSM8K dropped to **54.00%** (54/100 correct), representing a **15.0% regression** compared to the base model's **69.00%** (69/100 correct). Qualitatively, the SFT model frequently hallucinated invalid math reasoning steps or skipped constraints (e.g., ignoring multiplier terms).

Interpretation:
Supervised fine-tuning of a small model (1.2B parameters) on a narrow, format-intensive dataset (827 examples focused on strict JSON formatting and conciseness) causes catastrophic forgetting or behavioral drift on general multi-step reasoning tasks. The format alignment constraint acts as a behavioral tax on general intelligence.

What this rules out:
Fine-tuning small models on specialized formatting targets without including general instruction-following and reasoning task families (like GSM8K or general SFT tasks) in the training blend.

Next:
Include mixed-task data-blend training (e.g., combining formatting, general instruction following, and mathematical reasoning in SFT/DPO) to preserve general model intelligence while aligning formatting style.

---

## NR021: Hermes trajectory workers can wedge without result files unless subprocess groups are killed on timeout

Experiment:
Ran seven-family HY3/Hermes trajectory collection on lenovo using `run_hermes_tasks.py` with per-task timeout 420s and transient retry/backoff.

Expected:
When a task exceeded timeout or a Hermes/session subprocess hung, the runner would write stdout/stderr/result artifacts and continue to the next rollout.

Observed:
Six balanced shard workers held current run directories beyond the timeout without writing `stdout.txt`, `stderr.txt`, or `result.json`. No live Hermes child process was visible under `pstree`, but the Python runners remained alive and stopped advancing. The supervisor considered the workers active, so shards 06-15 could not start under the worker cap.

Interpretation:
`subprocess.run(..., timeout=...)` was not sufficient for this long-running Hermes/session-export workflow. The collector needs explicit process-group isolation, process-group termination on timeout, and cleanup of incomplete no-result run directories during restart.

What this rules out:
Relying on a parent Python worker being alive as evidence that collection is progressing. Restarting workers without cleaning incomplete run dirs.

Next:
Use the patched `run_hermes_tasks.py`, which launches subprocesses in their own process group and kills the group on timeout. Run shard workers with `--clean-incomplete`; the supervisor now passes this flag by default. Monitor latest result ages as well as process liveness.

---

## NR022: Standard PEFT weight merging corrupts LFM2.5 SSM/linear-RNN models

Experiment:
Loaded two LoRA adapters (Math and Formatting) on `LFM2.5-1.2B-Instruct` using `PeftModel`, combined them using `model.add_weighted_adapter` (linear type), called `model.merge_and_unload()`, and exported to GGUF.

Expected:
The merged model would load and combine both adapters cleanly.

Observed:
The PEFT-merged model collapsed completely on all benchmarks (0.0% GSM8K, 0.0% JSON validity), getting stuck in infinite token repetition loops (repeating `1.1.1.1` or `###`). The math-only control model merged using PEFT collapsed similarly, scoring only 5.0% on GSM8K.

Interpretation:
Standard PEFT utilities like `merge_and_unload` are designed for standard Transformer linear projection modules and do not respect the specific parameter mappings and state variables of Liquid Foundation Models (LFMs). Merging LoRA layers using standard PEFT methods corrupts the model parameters.

What this rules out:
Using PEFT's native `add_weighted_adapter` and `merge_and_unload` on LFM2.5 models.

Next:
Manually compute the low-rank delta products and sum them directly to the base weights in PyTorch ($W_{base} + \sum w_i s_i B_i A_i$), completely bypassing PEFT's API.

---

## NR023: SFT on raw GSM8K answers with calculator tags causes digit/equation repetition collapse

Experiment:
SFT-trained a completions-only math adapter on the raw `openai/gsm8k` train split containing calculator-guided tags (e.g. `<<16-3-4=9>>9`).

Expected:
The model would learn to output step-by-step mathematical reasoning.

Observed:
The model was heavily distorted, falling into severe repetition loops of formulas and calculator tags (e.g., repeating `1.5 hours * 30mph = <<1.5*30=45>>45` over and over).

Interpretation:
The base instruct model has never seen these calculator-guided tags (`<<...>>`) during pre-training. Forcing the model to output them in SFT represents an out-of-distribution vocabulary mismatch, causing logit distortion and generation collapse.

What this rules out:
Finetuning instruct models on raw calculator-guided datasets without formatting cleanup.

Next:
Strip all `<<.*?>>` calculator annotations from the SFT dataset using regular expressions before training.

---

## NR024: Mixed-blend SFT fails on small model under low-epoch constraints

Experiment:
Trained `LiquidAI/LFM2.5-1.2B-Instruct` on a mixed-task dataset (413 formatting examples, 2,000 Magicoder, 1,600 GSM8K examples) for 300 steps. Converted it to GGUF format and ran formatting and GSM8K evaluations.

Expected:
The mixed SFT model would preserve math reasoning performance (staying close to the base model's 69.0% accuracy) while aligning structured formatting.

Observed:
1. Math reasoning performance declined further to **48.00%** (worse than the formatting-only SFT model's 54.00%).
2. Structured output formatting alignment failed completely: GameFAQ JSON validity rate dropped to **11.8%** (compared to formatting-only SFT's 82.4%), and JSON Structured validity rate dropped to **52.9%** (worse than the base model's 76.5%).
3. The model showed extreme bias towards outputting code blocks (e.g., generating Python scripts to answer simple mathematical formatting prompts).

Interpretation:
Under low-epoch constraints (1.2 epochs over a 4,000-example mixed dataset in 300 steps), the model does not receive enough gradient updates per task family. Because 50% of the dataset consists of code generation, the model's output distribution becomes heavily skewed towards code syntax. Without significantly more training steps (e.g., 1,500 - 2,000 steps), a simple data mixture on small models is insufficient to align formatting and preserve general reasoning capabilities.

What this rules out:
Using a short, single-pass mixed training run with a dominant task (e.g., 50% code) to align formatting and preserve general reasoning.

Next:
Run sequential SFT passes (general reasoning and code first, followed by a light formatting alignment pass) or train on the mixed blend for a much higher step count (e.g., 5 epochs).

---

## NR025: Standard PEFT weight merging collapses openbmb/MiniCPM5-1B GGUF outputs

Experiment:
Loaded two trained LoRA adapters (Math and Formatting) on `openbmb/MiniCPM5-1B` using `PeftModel`, combined them using `model.add_weighted_adapter(combination_type="linear")`, called `model.merge_and_unload()`, and exported to GGUF using Unsloth.

Expected:
Since MiniCPM5-1B uses the standard Llama attention architecture, standard PEFT wrapper APIs should merge the weights cleanly without parameter corruption.

Observed:
The PEFT-merged GGUF model collapsed completely on GSM8K, scoring exactly **0.00%** accuracy and getting stuck in infinite repeating loops of token/prose blocks. In contrast, the directly merged model (Math 1.0 + Format 0.7) merged using direct PyTorch weight surgery (`merge_lora_direct.py`) achieved **49.0%** accuracy.

Interpretation:
Standard PEFT wrapper combination routines (`add_weighted_adapter` and `merge_and_unload`) can distort or corrupt the projection matrices of quantized or patched architectures under Unsloth's native layer wrapping. Direct tensor surgery ($W_{base} + \sum w_i s_i B_i A_i$) is required even for standard transformer models to avoid compilation/quantization artifacts.

What this rules out:
Relying on standard PEFT merging wrappers for Unsloth-trained adapters when quantizing to GGUF.

---

## NR026: Initial MiniCPM direct merger used the wrong RS-LoRA scale

Experiment:
M01 analytically audited the math and formatting adapter tensors and compared
the saved direct model with the expected weighted update.

Observed:
Both adapters have `use_rslora=true`, rank 8, and alpha 16. The initial
`merge_lora_direct.py` used `alpha / r` instead of `alpha / sqrt(r)`, applying
only 0.3536 of each stated delta. Its maximum discrepancy from the intended
analytic merge was 0.00661. Correcting the scale and re-merging reduced the
maximum FP16 save/load discrepancy to 0.000848 across all 168 adapted tensors.

Interpretation:
The previous MiniCPM "Math 1.0 + Format 0.7" artifact was not a merge at those
weights. Its behavioral metrics cannot be used as evidence of multi-adapter
synergy. The corrected direct merge agrees with sequential PEFT merging; PEFT
`linear` and `cat` still diverge in the current 0.18.1 runtime and are not
validated replacements.

---

## NR027: Current llama.cpp GGUF harness does not cleanly delimit MiniCPM outputs

Experiment:
Ran the corrected RS-LoRA merge through `run_gguf_eval.py` with deterministic
decoding on the first three evaluation prompts.

Observed:
The Q4 model loaded and generated, but one response was empty and two retained
`user` / `assistant` template residue and `[end of text]` in the captured text.

Interpretation:
This is an output-boundary/harness issue, not a model-quality result. The
existing GGUF behavioral scores must not be used to validate the corrected
merge until the evaluator captures only generated tokens and reproduces the
same prompts/stop conditions for base and candidate models.
## NR028: M03 control suite is underpowered for a strict MiniCPM preservation gate

Experiment: M03 deterministic Q4 paired held-out evaluation of base versus the
corrected RS-LoRA 1.0 math / 0.7 formatting direct merge.

Observed: JSON rose 50.0pp (95% paired bootstrap CI +25.0 to +75.0), but math
was unchanged at 75.0% (CI −33.3 to +33.3). Code was unchanged at 41.7% with
the same wide CI, whose lower bound violates the −10pp control regression
budget; tool format was 0/12 for both models. All 96 captures passed boundary
integrity checks.

Interpretation: the initial gate was miscalibrated for 12-example control
suites: unchanged point estimates cannot certify a −10pp lower-bound budget.
The JSON gain remains valid, while non-target preservation is inconclusive—not
negative. Expand the frozen suites before training or making a synergy claim.

---
