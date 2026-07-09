# On-Policy Distillation Pipeline — Progress & Next Steps

## Status: Rollout phase COMPLETE. Teacher scoring script built, test run done. Full scoring next.

## What We Built

### On-Policy Distillation Pipeline
Student: LFM2.5-8B-A1B (aero: RTX 2070, Mac: M3 Max, deck: Steam Deck)
Teacher: OpenAI-compatible scorer. Originally Gemma4-26B-A4B locally; now supports OpenRouter `tencent/hy3:free` or any compatible provider.

### Rollouts Accumulated (23,220 total)
| Location | Count | Dataset Mix |
|---|---|---|
| aero `/home/billz/rollouts/` | 10,386 | ifstruct/gsm8k/arc/mmlu |
| aero `/home/billz/rollouts-deck/` | 2,272 | ifstruct/gsm8k/arc/mmlu |
| mac `/Users/bilawalriaz/rollouts/` | 10,380 | ifstruct/gsm8k/arc/mmlu |
| mac `/Users/bilawalriaz/rollouts-deck/` | 182 | ifstruct/gsm8k/arc/mmlu |

Each rollout contains:
- `prompt` — full input
- `response` — student output
- `reasoning` — full thinking chain (4k avg chars)
- `logprobs` — every token's log probability
- `temperature` — one of [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
- `prompt_meta` — gold_answer, dataset, entity_type, json_schema
- `tokens_per_second`, `elapsed_seconds`, `finish_reason`

### Datasets (5,191 prompts × 6 temps = 31,146 expected)
- **ifstruct-v1.0**: 2,000 structured output prompts (JSON/YAML)
- **GSM8K**: 1,319 math reasoning (gold answers)
- **ARC-Challenge**: 1,172 science reasoning (gold answers)
- **MMLU**: 700 college knowledge, 5 subjects (gold answers)

### Key Scripts
- `rollout_multigen.py` — multi-gen rollout generator (resumable)
- `compress_aggressive.py` — strips duplicated reasoning (63% savings) [DEPRECATED — use teacher condensation]
- `teacher_scoring.py` — provider-configurable ranking + reasoning condensation (replaces brute-force compression)
- `classifier/` — quality classifier (84% F1, 76k pred/sec) → separate repo

### Hardware Map
| Machine | IP | GPU | Role |
|---|---|---|---|
| aero | 100.100.211.67 | RTX 2070 8GB | Student + classifier |
| mac (m3) | 100.100.61.28 | M3 Max 36GB | Student + Teacher (Gemma4) |
| deck | 100.113.55.70 | Steam Deck (Vulkan) | Student (currently idle) |

### Server Ports
- Student: aero:8080, mac:8081
- Teacher: mac:8082 (Gemma4-26B-A4B Q4_K_M, 128k ctx)

## Next Steps

### 1. Teacher Scoring (READY — script built, test passed)
- `teacher_scoring.py` ranks 6 generations per prompt, picks winner, condenses reasoning, and appends one durable row per prompt to `scored.jsonl`
- Test results from local teacher: 3 prompts scored, avg quality 9.3/10, 83% reasoning compression, 100% correctness
- Current recommended teacher: OpenRouter Tencent HY3 (`tencent/hy3:free`) to avoid running Gemma4-26B-A4B locally on the laptop
- Need to: consolidate rollouts to Mac, run full scoring (~3,900 prompts, provider quota permitting)
- Report: `docs/scoring_report.html` — full examples with reasoning comparison

#### OpenRouter scoring command

```bash
export OPENROUTER_API_KEY="..."
export TEACHER_PROVIDER=openrouter
export TEACHER_MODEL=tencent/hy3:free
export TEACHER_CONTEXT_TOKENS=262000
export MAX_WORKERS=6
export INPUT_DIR=/Users/bilawalriaz/rollouts
export OUTPUT_DIR=/Users/bilawalriaz/scored
python3 teacher_scoring.py score-one
python3 teacher_scoring.py full
```

`teacher_scoring.py` sends one API call per `prompt_hash`, with all 6 rollout generations in the same prompt so the teacher can compare them jointly. Run `score-one` first to validate HY3 returns parseable JSON before spending the full queue. HY3 uses `RESPONSE_FORMAT=json_schema` by default because its OpenRouter provider rejects `json_object`. `MAX_WORKERS=6` runs six prompt-hash groups concurrently during `full`; each worker still sends all 6 generations in one request. The HY3/OpenRouter defaults use the 262k context setting and no prompt/reasoning/response truncation; if a prompt is estimated to exceed the configured context, the scorer stops before sending it. Empty/unparseable responses stop the run and save diagnostics under `$OUTPUT_DIR/bad_teacher_responses/`. If the free quota/rate limit is hit, `teacher_scoring.py` stops, leaving the current prompt unscored. After setting a different provider/model/key, rerun the same command; existing `prompt_hash` rows in `scored.jsonl` are skipped and the next unscored rollout is processed.

To finish using another OpenAI-compatible provider:

```bash
export TEACHER_PROVIDER=local
export TEACHER_URL=http://100.100.61.28:1234
export TEACHER_MODEL=gemma4-26b-a4b-qat-uncensored-hauhaucs-balanced-mtp
python3 teacher_scoring.py full
```

To split a full run across OpenRouter HY3 and opencode-go, use mixed mode:

```bash
export TEACHER_PROVIDER=mixed
export OPENROUTER_API_KEY="..."
export OPENCODE_API_KEY="..."
export MAX_WORKERS=18
export INPUT_DIR=/Users/bilawalriaz/rollouts
export OUTPUT_DIR=/Users/bilawalriaz/scored
python3 teacher_scoring.py full
```

In mixed mode, `MAX_WORKERS` is split across the two providers. With `MAX_WORKERS=18`, the scorer schedules 9 OpenRouter `tencent/hy3:free` prompt groups and 9 opencode-go `mimo-v2.5` prompt groups per batch. Each prompt group still sends all rollout generations in one API call. opencode-go defaults to prompt-only JSON (`OPENCODE_RESPONSE_FORMAT=none`) because response-format support is provider-specific; set `OPENCODE_RESPONSE_FORMAT=json_object` only after validating the endpoint accepts it.

### 2. Training (Unsloth on aero)
- Scored pairs with condensed reasoning as training targets
- LoRA fine-tune LFM2.5-8B-A1B
- Targets: q/k/v/o_proj + gate_up_proj/down_proj (MoE-specific)
- r=16, alpha=32, 3 epochs, lr=2e-4, bf16
- Unsloth already installed on aero

### 3. Evaluation
- Run trained model through same benchmarks
- Compare against base LFM2.5-8B-A1B

## Repos
- Pipeline: github.com/bilawalriaz/autonomous-small-model-exploration (private)
- Classifier: github.com/bilawalriaz/quality-classifier (private)
