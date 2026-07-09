# Teacher Setup

## OpenRouter Tencent HY3 Free (No Local Heat)

Use OpenRouter when local Gemma/Qwen teacher inference makes the laptop too hot. The scorer uses an OpenAI-compatible API, so only the base URL, model, and key change.

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

Resume behavior:
- Successful labels are appended to `$OUTPUT_DIR/scored.jsonl`.
- On quota, auth, credit, or rate-limit errors, scoring stops without writing a scored row for the current prompt.
- Rerun with another provider/model/key and completed `prompt_hash` values are skipped automatically.
- Each prompt hash is scored in one API call containing all 6 generations, so OpenRouter's free-model per-call billing is used efficiently and the teacher compares all candidates jointly.
- `score-one` should return a valid scored JSON object before `full` is started. Empty or unparseable provider responses stop the run by default and save diagnostics under `$OUTPUT_DIR/bad_teacher_responses/`.
- `MAX_WORKERS=6` sends six prompt-hash groups concurrently during `full`; each worker still sends all six generations for its prompt in a single request.
- HY3 uses `RESPONSE_FORMAT=json_schema` by default because its OpenRouter provider rejects `json_object`.

HY3 defaults in `teacher_scoring.py` use `TEACHER_CONTEXT_TOKENS=262000` and no prompt/reasoning/response truncation (`MAX_*_CHARS=0`). If a specific rollout exceeds the provider's true context limit, the scorer stops before sending it; then set explicit character limits only for that retry.

## Mixed OpenRouter HY3 + opencode-go

Use mixed mode when you want one `teacher_scoring.py full` run to split workers across both hosted providers.

```bash
export TEACHER_PROVIDER=mixed
export OPENROUTER_API_KEY="..."
export OPENCODE_API_KEY="..."
export MAX_WORKERS=18
export INPUT_DIR=/Users/bilawalriaz/rollouts
export OUTPUT_DIR=/Users/bilawalriaz/scored
python3 teacher_scoring.py full
```

With `MAX_WORKERS=18`, the scheduler assigns 9 prompt-group workers to OpenRouter `tencent/hy3:free` and 9 prompt-group workers to opencode-go `mimo-v2.5`. Each prompt group still contains all six rollout generations in one teacher call. Scored rows record the actual `teacher_provider`, `teacher_model`, `teacher_api_base`, and `teacher_name`, so mixed-provider labels remain auditable.

Defaults:
- OpenRouter: `OPENROUTER_TEACHER_MODEL=tencent/hy3:free`, `OPENROUTER_RESPONSE_FORMAT=json_schema`, `OPENROUTER_CONTEXT_TOKENS=262000`.
- opencode-go: `OPENCODE_MODEL=mimo-v2.5`, `OPENCODE_URL=https://opencode.ai/zen/go/v1/chat/completions`, `OPENCODE_RESPONSE_FORMAT=none`, `OPENCODE_CONTEXT_TOKENS=100000`.

If the opencode-go endpoint is confirmed to accept OpenAI `response_format`, set `OPENCODE_RESPONSE_FORMAT=json_object` for stricter JSON. Otherwise the prompt still asks for strict JSON and parse failures remain unscored by default.

## Local Qwen3.6-35B-A3B Teacher on Mac

## Quick Start (MLX)

```bash
# Install MLX
pip install mlx-lm

# Run the server (Q4 fits in 36GB at ~21GB)
python -m mlx_lm.server \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --port 8081 \
  --trust-remote-code

# Verify
curl http://localhost:8081/v1/models
```

## Alternative: llama.cpp on Mac

```bash
# Build with Metal
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build -DGGML_METAL=ON && cmake --build build --config Release

# Download GGUF (Q5_K_M ~26GB, fits 36GB)
huggingface-cli download Qwen/Qwen3.6-35B-A3B-GGUF qwen3.6-35b-a3b-q5_k_m.gguf --local-dir models/

# Run server
./build/bin/llama-server \
  -m models/qwen3.6-35b-a3b-q5_k_m.gguf \
  --host 0.0.0.0 --port 8081 \
  -ngl 999 --ctx-size 32768 \
  --flash-attn
```

## Alternative: Ollama on Mac

```bash
ollama pull qwen3:30b-a3b
ollama serve  # Runs on port 11434 by default
```

## Network Setup (Tailscale)

Mac and aero need to communicate. Options:

1. **Tailscale** (recommended): Both machines on tailnet → use Tailscale IP
2. **SSH tunnel**: `ssh -L 8081:localhost:8081 aero` from Mac
3. **Direct IP**: If on same LAN, use local IP

### Finding Mac's IP:
```bash
# Tailscale
tailscale ip -4

# LAN
ipconfig getifaddr en0
```

## Logprobs Support

MLX server supports logprobs via the OpenAI-compatible API:
```python
resp = requests.post("http://MAC_IP:8081/v1/completions", json={
    "model": "qwen3.6-35b-a3b",
    "prompt": "The capital of France is",
    "max_tokens": 10,
    "logprobs": True,
    "echo": True,  # Return logprobs for the prompt tokens too
})
# Response includes logprobs.token_logprobs[]
```

## Performance (M3 Max 36GB)

| Quant | Memory | Speed | Quality |
|-------|--------|-------|---------|
| Q4 | ~21GB | 40-52 tok/s | Good |
| Q5_K_M | ~26GB | 35-45 tok/s | Better |
| Q8 | ~35GB | 25-35 tok/s | Best |

## Verify Teacher Works

```python
import requests

resp = requests.post("http://MAC_IP:8081/v1/chat/completions", json={
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 100,
    "temperature": 0.2,
})

data = resp.json()
print(data["choices"][0]["message"]["content"])
print(f"Tokens: {data['usage']['completion_tokens']}")
```
