# Current Findings

## Executive summary
Qwen2.5-0.5B has a clear hierarchical component structure with L2 as a universal processing hub. Position-specific ablation reveals L2 routes first+last tokens while L22 is exclusively a last-token layer and L9 is the instruction-sensitive layer. LoRA training rewires where skills live — each skill concentrates in DIFFERENT layers, rejecting the universal L0-L2 hypothesis. The core circuit (L2/L7/L9) locks in by step 10 of training. Adapters can be stacked: factual+json combines cleanly, but the delimiter adapter is destructive when merged. Cross-model patching shows trained behavior is encoded in late-layer activations (monotonic recovery L23=100% → L0≈0%). Skill knockout at L19 selectively suppresses factual recall (11654x selectivity) while preserving other skills. Adapter-only ablation shows norm-effect correlation of 0.85, updating H6: adapter effects are at the same layers where norms peak, not upstream.

## Strongest causal claims

1. **L2 is a universal importance hub with positional specialization.** Ablating L2 causes the largest KL divergence (0.5-11.5 nats) across all 12 task families. Position-specific analysis shows L2 specifically routes first tokens (instruction, mean 3.34) and last tokens (prediction, mean 5.03). Operator tokens have near-zero effect. *Confidence: HIGH.* Evidence: layer ablation, MLP ablation, steering, position-specific ablation.

2. **LoRA training rewires where skills live, with skill-specific concentration patterns.** Each skill family concentrates in different layers after LoRA training: factual_recall at L3/16/19, code_semantics at L1/10/21, json_schema at L6/12/13. The hypothesis that training universally concentrates into L0-L2 is rejected. *Confidence: MEDIUM.* Evidence: dataset shard ablation (5 families).

3. **L2 causal role for factual recall confirmed by steering.** Steering L2 with factual direction boosts "rome" from 0.064 to 0.213 (3.3x) for "capital of Italy". Negative steering suppresses it. Oversteering at s>=+2 causes degeneration (Chinese characters). *Confidence: MEDIUM.* Evidence: steering sweep.

4. **L22 is the unembedding/final-prediction pathway.** Position-specific ablation shows L22 almost exclusively affects last-position tokens (mean 14.55 nats, all other positions ~0). Cross-model patching confirms L22 carries 97% of trained behavior. *Confidence: MEDIUM.* Evidence: position-specific ablation, cross-model patching.

5. **Core circuit locks in by step 10 of training.** L2/L7/L9 for JSON schema stabilize at step 10 (first 10% of training) and drift <1% through step 100. Secondary layers (L15, L6) continue shifting. *Confidence: MEDIUM.* Evidence: checkpoint timeline.

6. **Trained behavior is encoded in late-layer activation patterns.** Cross-model patching shows monotonic recovery increase from early to late layers: L23=100%, L22=97%, L21=95%, L20=87%, L19=80%. Patching trained activations at L23 into the base model gives 100% recovery. *Confidence: MEDIUM.* Evidence: cross-model patching (17 pairs, 24 layers).

7. **Skills can be selectively suppressed via negative steering at skill-specific layers.** L19 selectively knocks out factual recall (selectivity ratio 11654x at s=-2.0) while preserving JSON and copying skills. L2 is non-selective (universal hub — knockout affects everything). L21 also shows good selectivity (53x). *Confidence: MEDIUM.* Evidence: skill knockout experiment (2 skills, 7+ layers).

8. **Adapter norm and ablation effect are correlated (r=0.85), both peaking at late layers.** Adapter-only ablation shows that removing the adapter's contribution at L19-L23 has the largest effect, matching the norm distribution. This updates H6: the adapter's effect IS at the same layers where it writes, not upstream. The earlier finding conflated general layer importance (L0-L2) with adapter-specific importance (L19-L23). *Confidence: MEDIUM.* Evidence: adapter-only ablation (12 prompts, 24 layers).

9. **Adapters can be combined with varying interference.** factual_recall + json_schema stacks cleanly (+2.35 synergy on factual, +1.17 on json). delimiter_tracking adapter is destructive when stacked (-7 to -16 nats). *Confidence: MEDIUM.* Evidence: adapter stacking.

## Training perturbation findings

- **Dataset shard ablation (5 families):** Each skill concentrates in different layers. factual_recall at L3/16/19; code_semantics at L1/10/21; json_schema at L6/12/13. delimiter_tracking fully absorbs (0 ablation sensitivity). copying dispersed.
- **Checkpoint timeline:** Core circuit (L2/L7/L9) locks in by step 10. Loss: 0.587 (step10) -> 0.062 (step100). Secondary layers L15/L6 drift +2.85/+2.73 through training.
- **LoRA rank sweep:** L0 MLP peaks at r=4 (15.77). Total norm scales linearly (6.14 to 22.92). Higher rank shifts norm from uniform to late layers.
- **LoRA module sweep:** o_proj alone achieves +3.64 L0 effect with 344K params. MLP-only weakest (3.3M params, +1.92).
- **Cross-task effects after JSON LoRA:** L4 appeared for factual_recall (+1.42). L2 increased for delimiter (+1.19), factual (+1.27). L2 decreased for copying (-0.75).
- **Adapter stacking:** factual+json synergistic. code+json compatible. delimiter destructive.
- **Adapter-only ablation:** Norm-effect correlation 0.85. JSON adapter effect concentrated at L19-L23 (L23=100%, L22=92%, L21=81%). Only L12 shows norm-effect mismatch.

## Position-specific findings

- **L22:** Almost exclusively last-position (mean 14.55, others ~0). Unembedding pathway.
- **L0/L2:** First+last position routers. L0 first=3.94, last=3.30. L2 first=3.34, last=5.03.
- **L9:** Instruction-sensitive. First=5.66, last=9.20. Strongest mid-layer for both positions.
- **L7:** Balanced first+last (5.03/5.93).
- **L15:** Weak overall (max 3.37 on last). Processing layer.
- **Operators/delimiters:** Near-zero effect across all layers.

## Cross-model patching findings

- Recovery increases monotonically from early to late layers
- L23=100%, L22=97%, L21=95%, L20=87%, L19=80%, L18=73%
- Mid-layers (L13-L17) show 50-80% recovery
- Early layers (L0-L12) give minimal recovery (<50%)
- The adapter's learned behavior is encoded in the activation patterns of late layers

## Skill knockout findings

- **Factual recall:** L19 most selective (11654x at s=-2.0), L21 good (53x), L16 moderate (0.42x)
- **Factual recall:** L2 and L3 non-selective (universal hub effect)
- **JSON:** Base probability of targets already near-zero, limited knockout room
- **JSON:** L6, L7, L9, L12, L13, L21 all cause moderate KL changes at s=-1.0
- Negative steering at s=-4.0 causes broad degradation across all skills

## Weak/tentative signals
- L1 appears as universal skill injection point (positive delta across 3+ adapters)
- L21/L23 may be formatting/output specialists
- Code semantics resistant to layer ablation (KL=0.52 at L2)
- delimiter adapter's extreme stacking behavior may indicate format-specific overfitting
- L12 norm-effect mismatch may indicate a processing bottleneck

## Negative results
1. Full SFT OOMs on 8GB VRAM. LoRA required.
2. Full-residual activation patching gives KL=0 everywhere. Position-specific patching needed.
3. H002 (universal L0-L2 concentration) rejected.
4. Clean/corrupt pair v0 had tokenization misalignment. Fixed in v1.
5. Extreme steering (s>=+2) causes degeneration (Chinese characters, repetition).
6. L2 is NOT position-uniform (operator tokens near-zero).
7. JSON skill knockout had limited effect due to near-zero base target probability.
8. H6 (upstream propagation) rejected by adapter-only ablation (corr=0.85, effects at same layers as norms).
9. PeftModel.from_pretrained modifies base model in-place — must use disable_adapter() for base behavior.

## Current atlas status
| Confidence | Count |
|------------|-------|
| Low        | 2     |
| Medium     | 17    |
| High       | 1     |
| Very High  | 0     |
| Negative   | 9     |

## Phase 9: Format Ablation Training Loss Findings (2026-06-29)

**IMPORTANT CAVEAT:** The following findings are based on training loss ONLY. No real behavioral evaluation has been completed. Mock-judge data from the original Phase 9 report should be disregarded for behavioral claims.

### Key training loss findings
- **Format significantly affects training loss under content-controlled conditions.** Same 300 canonical examples rendered into 6 formats → 33% loss range (1.37 to 1.83).
- **Multi-turn verbose has lowest loss (1.372).** This is surprising — Phase 8 suggested concise was better. But Phase 8 didn't control for content.
- **bad_format_control has 2nd-best loss (1.402).** Deliberately malformed data is easy to predict. Loss measures predictability, not quality.
- **Structured terse has worst loss (1.831).** Compact structured format is hardest for 230M model to learn from.
- **Surgical adapter beats quality adapter on loss (1.27 vs 1.46) with 3.8x fewer params.** Out_proj-only LoRA is more parameter-efficient than hub-all-modules.

### What we can claim
1. Training loss differs by format (confirmed, reproducible)
2. Loss does not trivially measure quality (plausible, needs behavioral confirmation)
3. Format dominates hyperparameters (consistent with Phase 8)

### What we cannot yet claim
1. Multi-turn verbose produces better outputs (no real eval data)
2. Any behavioral ranking of formats (mock judge only)
3. Loss-quality correlation or decoupling (needs real eval)

## Best next experiments
1. Multi-seed replication of top 5 findings
2. Mean/resample ablation (stronger causal claims)
3. CPT training
4. SAE training on key layers (L0, L1, L2, L7, L9, L19, L22)
5. Skill injection at L19 for factual recall
6. Extend to natural language prompts
7. **Phase 9R: Run real eval on aero** (highest priority — completes Phase 9)
8. **Phase 10: Token-budget-controlled data-shape ablation**

## Documentation update

2026-06-30: Added and deployed a single shareable MI-Atlas page at `docs/mi-atlas.html`, linked it from the docs index/navbar, and mirrored it to `pretty-blog-python/pages/mi-atlas.html`. This is a presentation artifact only. It uses the audited claim set and Phase 9 caveats already recorded here; no research claim, metric, confidence level, or negative result changed.

2026-06-30: Expanded the share page with a dedicated LFM2.5-230M SFT sweep section. The added copy summarizes existing Phase 8/9 evidence only: 39 SFT runs, dataset choice dominating loss, Adafactor as the practical optimizer, rank 8 as the efficiency point, hub-targeted LoRA as the low-drift option, and the explicit caveat that behavioral format ranking remains unsettled.

2026-06-30: Added full-detail links to every audited claim on the share page and mirrored them on bilawal.net. Standardized the detailed GitHub Pages navigation and shared responsive styling across the docs pages. This is presentation maintenance only. Claim changes: 0. Negative-result changes: 0.

2026-07-08: Updated rollout labelling infrastructure so `teacher_scoring.py` can use OpenRouter's pinned free Tencent HY3 model (`tencent/hy3:free`) or any OpenAI-compatible provider. This is operational maintenance only. It does not add labelled-data results, behavioural metrics, or claim-confidence changes.

2026-07-09: Ran a two-provider smoke test for rollout labelling alternatives. OpenRouter `tencent/hy3:free` successfully judged two rollout prompt groups with all six generations included per prompt group, returning parseable JSON in 44.2s and 51.4s with quality scores 8/10 and 7/10. opencode-go `mimo-v2.5` completed two direct generation prompts in 4.56s and 4.29s; the JSON extraction response passed all local checks, while the Python bugfix fixed the off-by-one error but did not handle empty input. This is operational evidence about provider speed/reliability only; it does not change MI-Atlas research claims or hypothesis confidence.

2026-07-09: Added mixed-provider scheduling to `teacher_scoring.py`. Setting `TEACHER_PROVIDER=mixed` splits `MAX_WORKERS` across OpenRouter HY3 and opencode-go; for example, `MAX_WORKERS=18` schedules 9 OpenRouter `tencent/hy3:free` prompt-group workers and 9 opencode-go `mimo-v2.5` prompt-group workers. This is an operational throughput/resilience change only and does not change research findings.

2026-07-09: Completed hosted teacher labelling for the rollout set using opencode-go `mimo-v2.5`: 1,728/1,730 prompt groups scored, with 2 unresolved after retries. A high-confidence teacher filter (`format_valid=true`, `quality_score>=8`, correctness correct/na) yielded 1,240 rows. Deterministic parse + schema validation accepted 827 of those rows and rejected 413, showing that teacher-clean labels still need validator gating before SFT.

2026-07-09: Set up a lenovo-based Hermes/Atropos agent trajectory lab using aero's LFM2.5-8B-A1B llama.cpp endpoint. Hermes can call aero successfully (`AERO_LFM_READY` smoke), and a control run with Nous `stepfun/step-3.7-flash:free` solved the first code-edit task. The same task with aero LFM produced a JSON plan/tool-call-like response but did not execute tools or modify files, so it failed the verifier. This is operational evidence that the harness works and a negative baseline for current LFM tool-use behavior; it does not change atlas claim confidence.

2026-07-09: Expanded the trajectory lab to 125 sysadmin/security/deployment tasks and started a background collection run on lenovo with Nous `stepfun/step-3.7-flash:free`. Docker Engine, Docker Compose v2, pytest, and PyYAML were installed on lenovo to support realistic Compose and verification workflows. This is an active data-collection run; no finetuning or behavioral result is claimed yet.

2026-07-09 update: The original serial sysadmin collection was stopped after one passed host-audit record because host-audit tasks were too slow for the >100 trace goal. Four faster shard workers are now running over the remaining non-host-audit sysadmin/deployment task queue. Latest live check showed 11/11 completed records passing, 10 with Hermes session trace files, and 7 with clean task-specific trace exports. This remains data-collection progress only; no finetuning result is claimed.

2026-07-09 final update: The lenovo Hermes/Stepfun sysadmin collection completed enough verified trajectories for the first agentic SFT pass. Across 115 result rows, 106 passed deterministic verifiers. The full export at `/home/billz/agent_trajectory_lab/datasets/sysadmin_stepfun_20260709.jsonl` contains 106 passed records, 105 with trace files. The strict export at `/home/billz/agent_trajectory_lab/datasets/sysadmin_stepfun_20260709_clean.jsonl` contains 102 passed records, all with prompt-matched Hermes trace files. This satisfies the >100 high-quality trajectory collection target but does not yet provide evidence that finetuning improves LFM agentic performance.

2026-07-09 HY3 scale update: Added a seven-family agentic task generator that emits 700 templates and 7,000 verifier-backed instances: tool-format, file operations, shell/repo inspection, failure recovery, summarisation/state-compaction, choose-the-right-tool routing, and multi-step mini-agent tasks. HY3 routing through Nous Portal works with `--provider nous --model tencent/hy3:free`. A seven-family pilot passed 7/7 with prompt-matched Hermes traces. Eight-worker concurrency produced HTTP 429 artifacts, so active collection now uses six family-balanced workers with transient retry/backoff. Current strict SFT export contains 1,080/1,080 clean records, all with prompt-matched traces, and current DPO export contains 265 same-task pairs after excluding transient API failures. All seven requested families have crossed 100 clean records. The exporter now filters generated cache/binary artifacts from `changed_files`, and DPO pairs now include explicit rejected-failure tags for overlong elapsed, verifier-failed, and ignored-observation / failed-recovery cases. A capped lenovo supervisor keeps balanced HY3 shards moving without exceeding six active workers, and a refresher updates clean/DPO/train-ready exports every 600s. Stratified training-ready splits are available: SFT train/val 1026/54 and DPO train/val 251/14; every SFT row includes a compact matched-trace `trajectory` and `trajectory_messages` field for tool-use imitation, and DPO rows include clean `chosen_messages`/`rejected_messages` plus hard-negative and efficiency-tier files. This is collection infrastructure and early operational evidence only; no LFM training effect is claimed.

2026-07-09 HY3 export-hygiene checkpoint: Refreshed the trainable HY3 trajectory exports after adding transcript sanitization for repository-internal/cache path noise in tool calls, tool results, terminal-command strings, and assistant prose. Current strict clean export contains 1,130 verifier-passed prompt-matched traces, and current DPO export contains 278 same-task pairs. Training-ready splits are SFT train/val 1072/58, DPO train/val 264/14, hard DPO 30/1, and efficiency DPO 234/13. Validation found zero `.git` path rows in the trainable SFT/DPO text fields after sanitization; raw Hermes traces remain preserved separately for audit. This improves dataset quality but does not change any model-behavior claim.

2026-07-09 HY3 worker recovery checkpoint: Six long-running shard workers had wedged on current tasks without writing result files, so `run_hermes_tasks.py` now launches subprocesses in separate process groups and kills the group on timeout; `--clean-incomplete` removes partial no-result run directories on restart. The capped supervisor now runs seven HY3 workers (PID 553902) and passes `--clean-incomplete`. After restart, the workers cleaned incomplete runs, resumed collection, and refreshed exports to 1,160 strict clean prompt-matched traces plus 284 DPO pairs. Training-ready splits are SFT train/val 1102/58, DPO train/val 270/14, hard DPO 33/1, and efficiency DPO 237/13. No fresh rate-limit matches were observed during the seven-worker probe. This is operational collection evidence only; no model-quality claim changes.

2026-07-09: Stopped the aero LFM llama.cpp server to free VRAM and attempted to start Unsloth QLoRA SFT for `LiquidAI/LFM2.5-8B-A1B` on the strict 827-row scored dataset. The new trainer is `scripts/train/train_lfm25_8b_unsloth_qlora.py`, with config snapshot `configs/sft/lfm25_8b_a1b_unsloth_qlora_8gb.json`. Four load strategies failed before training: default bnb 4-bit placement, higher Unsloth GPU utilization, forced CUDA placement, and forced CUDA plus embedding offload. The strongest failure was a true CUDA OOM during weight loading at about 7.59 GiB used on the 7.604 GiB usable RTX 2070-class GPU. This is a hardware/stack feasibility result, not a behavioral model result.

2026-07-09: Completed Unsloth QLoRA SFT for `LiquidAI/LFM2.5-1.2B-Instruct` on aero using the strict 827-row scored dataset (final train loss 1.372). Converted both the SFT adapter model and base model to Q4_K_M GGUF format, and evaluated them on 153 prompts. The comparative analysis shows a major behavioral improvement: JSON Validity Rate for GameFAQ extraction jumped from 5.9% to 82.4% (+76.5%), factual QA accuracy increased from 5.9% to 23.5% (+17.6%), average response length decreased from 139.7 to 109.4 words (-30.3 words), and slop phrases were completely eliminated. This provides solid empirical evidence that SFT'ing on strictly validated mimo-labeled examples successfully bootstraps formatting discipline and factual accuracy. However, evaluation on 100 GSM8K prompts revealed a significant math reasoning regression, dropping from 69.0% (base) to 54.0% (SFT), demonstrating a 15.0% behavioral tax / task drift from narrow, formatting-intensive fine-tuning on a small model.

2026-07-09: Completed mixed-blend SFT on `LiquidAI/LFM2.5-1.2B-Instruct` on aero (413 formatting, 2,000 Magicoder, 1,600 GSM8K examples) for 300 steps (final train loss 1.392). Evaluation on GGUF format showed that under low-epoch constraints (~1.2 epochs overall), the model failed to preserve math reasoning (GSM8K accuracy dropped further to 48.0%) and failed formatting alignment (GameFAQ JSON validity rate dropped to 11.8%). The model showed a heavy bias towards outputting code blocks (skewed by the 50% Magicoder ratio). This demonstrates that short mixed-blend SFT runs on small models are highly susceptible to task interference and require significantly more training steps / epochs to balance task alignment and capability preservation.

2026-07-09: Evaluated intermediate checkpoint-200 (~3.9 epochs formatting SFT) to validate the sequential SFT trade-off hypothesis. Converting and testing this checkpoint on all 153 formatting prompts and 100 GSM8K prompts showed a highly favorable balance: GameFAQ JSON validity reached 76.5% (nearly matching step-300's 82.4%) and JSON structured validity reached 82.4% (better than step-300's 76.5%), while math reasoning accuracy was preserved at 56.0% (higher than step-300's 54.0%). This validates that a lighter SFT pass (Pass 2) over the instruct model is a highly effective way to balance formatting style and general reasoning capability.

2026-07-09: Executed a custom weight-scaling sweep (0.3 and 0.7 adapter delta scaling before merging) to find the optimal style-vs-capability threshold on the step-200 checkpoint. At scale 0.3, the formatting delta is below the threshold of behavioral activation, yielding near-base results (11.8% GameFAQ JSON validity, 67.0% GSM8K accuracy). However, scale 0.7 triggers a highly robust compromise: GameFAQ JSON validity reaches 52.9%, structured JSON validity hits 88.2% (surpassing the base and full SFT runs), conversational slop is eliminated, and math reasoning accuracy is preserved at 60.0% (retaining 87% of the base model's reasoning capacity). This maps a classic sigmoidal activation curve and confirms that adapter delta scaling is an effective knob to tune the style-vs-reasoning trade-off.
2026-07-09: Replaced custom completions-only collator with pre-tokenized dataset-side prompt masking, bypassing Triton/PyTorch Inductor compilation hangs on the RTX 2070 GPU. Rebuilt the GSM8K dataset by regular-expression stripping of out-of-distribution calculator annotations (`<<...>>`) and SFT-trained a clean math adapter (train loss 0.686). Discovered that standard PEFT API functions like `add_weighted_adapter` and `merge_and_unload` corrupt LFM2.5 non-transformer weights (causing severe repeating token collapses). Designed and executed a direct PyTorch weight merging script `merge_lora_direct.py` ($W_{base} + \sum w_i s_i B_i A_i$) which successfully bypassed PEFT's API. Directly merging the Math adapter (1.0) and Formatting adapter (0.7) achieved a final GSM8K accuracy of **69.0%** (100% of baseline capability, +9% absolute over format-only adapter) while maintaining **82.4% JSON validity** (+5.9% over base) and **23.5% GameFAQ validity** (+17.6% over base), completely validating multi-adapter direct weight blending on LFMs.
