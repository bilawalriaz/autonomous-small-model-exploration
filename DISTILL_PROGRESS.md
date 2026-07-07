# On-Policy Distillation Pipeline — Progress & Next Steps

## Status: Rollout phase paused. Teacher scoring next.

## What We Built

### On-Policy Distillation Pipeline
Student: LFM2.5-8B-A1B (aero: RTX 2070, Mac: M3 Max, deck: Steam Deck)
Teacher: Gemma4-26B-A4B (Mac, port 8082)

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
- `compress_aggressive.py` — strips duplicated reasoning (63% savings)
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

### 1. Teacher Scoring (BLOCKED — waiting for Gemma4 GGUF on Mac)
- Download Gemma4-26B-A4B Q4_K_M GGUF to Mac
- Start teacher: `~/llama.cpp/build/bin/llama-server -m <path> --port 8082 -ngl 999 --ctx-size 131072 --flash-attn on --cont-batching --parallel 1 --mlock`
- Score all 23,220 rollouts against teacher
- For GSM8K/ARC/MMLU: check answer correctness against gold
- Output: scored pairs with teacher judgments

### 2. Reasoning Compression
- Run `compress_aggressive.py` on scored rollouts
- Strips duplicated output from reasoning chains
- Student learns concise planning, not verbose reasoning

### 3. Training (Unsloth on aero)
- LoRA fine-tune LFM2.5-8B-A1B on scored+compressed pairs
- Targets: q/k/v/o_proj + gate_up_proj/down_proj (MoE-specific)
- r=16, alpha=32, 3 epochs, lr=2e-4, bf16
- Unsloth already installed on aero

### 4. Evaluation
- Run trained model through same benchmarks
- Compare against base LFM2.5-8B-A1B

## Repos
- Pipeline: github.com/bilawalriaz/autonomous-small-model-exploration (private)
- Classifier: github.com/bilawalriaz/quality-classifier (private)
