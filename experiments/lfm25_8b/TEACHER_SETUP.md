# Qwen3.6-35B-A3B Teacher Setup on Mac

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
