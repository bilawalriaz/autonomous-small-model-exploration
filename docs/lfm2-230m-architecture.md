# LFM2.5-230M: Complete Architecture Document

## Overview

**LiquidAI/LFM2.5-230M** is a 230M-parameter instruction-tuned hybrid language model from Liquid AI.
It combines gated short convolutions with grouped query attention in an alternating pattern.
Designed for on-device inference with 128K context and 2x faster prefill/decode on CPUs vs pure transformers.

**Training:** 28T tokens pretraining + reinforcement learning post-training (instruction tuning).
**Recommended use:** Data extraction, lightweight agentic pipelines.
**NOT recommended for:** Reasoning-heavy workloads (advanced math, code generation, creative writing).

---

## Architecture Summary

| Property | Value |
|---|---|
| Architecture class | `Lfm2ForCausalLM` |
| Model type | `lfm2` |
| Total parameters | 229,693,184 (229.7M) |
| Trainable parameters | 229,693,184 (229.7M) |
| Hidden size | 1024 |
| Intermediate size (FFN) | 2560 |
| Num layers | 14 |
| Layer types | 8 conv + 6 attention (alternating) |
| Attention heads | 16 (Q) / 8 (KV) = GQA ratio 2:1 |
| Head dimension | 64 |
| Max position embeddings | 128,000 |
| Vocab size | 65,536 |
| Tie embeddings | Yes (embedding = LM head) |
| RoPE theta | 1,000,000 |
| Activation | SwiGLU (MLP) |
| Norm | RMSNorm (eps=1e-5) |
| Dtype | bfloat16 |
| VRAM (bf16) | 450 MB |

---

## Layer Map

```
L0:  CONV     (Lfm2ShortConv)  12.06M params
L1:  CONV     (Lfm2ShortConv)  12.06M params
L2:  ATTN     (Lfm2Attention)  11.01M params
L3:  CONV     (Lfm2ShortConv)  12.06M params
L4:  ATTN     (Lfm2Attention)  11.01M params
L5:  CONV     (Lfm2ShortConv)  12.06M params
L6:  ATTN     (Lfm2Attention)  11.01M params
L7:  CONV     (Lfm2ShortConv)  12.06M params
L8:  ATTN     (Lfm2Attention)  11.01M params
L9:  CONV     (Lfm2ShortConv)  12.06M params
L10: ATTN     (Lfm2Attention)  11.01M params
L11: CONV     (Lfm2ShortConv)  12.06M params
L12: ATTN     (Lfm2Attention)  11.01M params
L13: CONV     (Lfm2ShortConv)  12.06M params

Total layers: 162.58M + embedding 67.11M = 229.69M
```

Pattern: `C C A C A C A C A C A C A C` (conv bookends, alternating interior)
Conv layers: L0, L1, L3, L5, L7, L9, L11, L13 (8 total)
Attn layers: L2, L4, L6, L8, L10, L12 (6 total)

---

## Per-Layer Structure

Every layer (Lfm2DecoderLayer) has:
1. `operator_norm` — RMSNorm (1,024 params) — pre-norm for operator
2. **Operator** — either `self_attn` (Lfm2Attention) or `conv` (Lfm2ShortConv)
3. `ffn_norm` — RMSNorm (1,024 params) — pre-norm for MLP
4. `feed_forward` — Lfm2MLP (7,864,320 params) — SwiGLU FFN

Forward flow per layer:
```
residual = hidden_states
hidden_states = residual + operator(operator_norm(residual))
hidden_states = hidden_states + feed_forward(ffn_norm(hidden_states))
```

---

## Component Details

### Lfm2ShortConv (Conv Layers)

**Parameters per layer:** 4,197,376

| Component | Shape | Description |
|---|---|---|
| `in_proj` | [3072, 1024] | Projects input to 3x hidden (B, C, x) |
| `conv1d.weight` | [3072, 4] | Causal 1D convolution, kernel_size=4 |
| `out_proj` | [1024, 1024] | Projects back to hidden_size |

**Forward flow:**
```
projected = in_proj(x)           # [B, T, 3072]
B, C, x = chunk(projected, 3)   # each [B, T, 1024]
Bx = B * x                      # gate: element-wise multiplication
conv_out = conv1d(Bx)            # causal convolution (kernel_size=4)
y = C * conv_out                 # output gate
output = out_proj(y)             # project back to 1024
```

**Key insight:** This is a GATED convolution, not a simple conv. The input is projected into
three components — a gate (B), a carrier (C), and the signal (x). The gate modulates the signal
before convolution, and the carrier modulates the convolution output. This gives the conv layer
a form of input-dependent gating similar to attention.

**Conv state caching:** Maintains `conv_state` for autoregressive generation (past 3 tokens).

### Lfm2Attention (Attention Layers)

**Parameters per layer:** 3,145,856

| Component | Shape | Description |
|---|---|---|
| `q_proj` | [1024, 1024] | Query projection (16 heads × 64 dim) |
| `k_proj` | [512, 1024] | Key projection (8 KV heads × 64 dim) |
| `v_proj` | [512, 1024] | Value projection (8 KV heads × 64 dim) |
| `out_proj` | [1024, 1024] | Output projection |
| `q_layernorm` | [64] | Per-head Q normalization (RMSNorm) |
| `k_layernorm` | [64] | Per-head K normalization (RMSNorm) |

**Configuration:**
- Query heads: 16
- KV heads: 8 (GQA ratio 2:1)
- Head dimension: 64
- Position encoding: RoPE (theta=1M)
- Attention backend: SDPA by default (fallback to eager)

**Forward flow:**
```
Q = q_layernorm(q_proj(x).view(16, 64))  # per-head norm
K = k_layernorm(k_proj(x).view(8, 64))   # per-head norm
V = v_proj(x).view(8, 64)
Q, K = apply_rotary_pos_emb(Q, K, cos, sin)  # RoPE
attn_output = scaled_dot_product_attention(Q, K, V, causal_mask)
output = out_proj(attn_output)
```

**Key insight:** Q and K have per-head layer normalization before RoPE. This stabilizes training
and is common in modern architectures (e.g., Gemma 2). The GQA ratio of 2:1 means each KV head
is shared across 2 query heads.

### Lfm2MLP (All Layers)

**Parameters per layer:** 7,864,320

| Component | Shape | Description |
|---|---|---|
| `gate_proj` | [2560, 1024] | Gate projection (for SwiGLU) |
| `up_proj` | [2560, 1024] | Up projection |
| `down_proj` | [1024, 2560] | Down projection |

**Forward:** `down_proj(silu(gate_proj(x)) * up_proj(x))`

**Same in every layer** — 7.86M params per layer regardless of conv/attn type.
MLPs constitute 52% of layer parameters (7.86M / ~11-12M per layer).

### Embeddings and LM Head

- `embed_tokens`: [65536, 1024] = 67.1M params
- `lm_head`: [65536, 1024] = 67.1M params (tied with embed_tokens)
- `embedding_norm`: RMSNorm [1024] — applied after all layers, before LM head

**Tied embeddings** mean the embedding table IS the LM head. This saves 67.1M params.
The embedding constitutes 29.2% of total parameters.

---

## Residual Stream Analysis

Residual stream norms (measured on "The capital of France is"):

| Layer | Type | Norm | Mean | Std | Phase |
|-------|------|------|------|-----|-------|
| L0 | conv | 1.94 | -0.0001 | 0.025 | Quiet |
| L1 | conv | 2.21 | -0.0000 | 0.028 | Quiet |
| L2 | attn | 2.42 | 0.0001 | 0.031 | Quiet |
| L3 | conv | 1.76 | -0.0001 | 0.022 | Quiet |
| L4 | attn | 2.56 | 0.0002 | 0.033 | Quiet |
| L5 | conv | 1.68 | 0.0000 | 0.022 | Quiet |
| **L6** | **attn** | **25.55** | -0.004 | **0.326** | **EXPLOSION** |
| L7 | conv | 25.55 | -0.004 | 0.326 | High |
| L8 | attn | 25.59 | -0.004 | 0.326 | High |
| L9 | conv | 25.62 | -0.003 | 0.327 | High |
| L10 | attn | 25.60 | -0.004 | 0.327 | High |
| L11 | conv | 25.72 | -0.004 | 0.328 | High |
| L12 | attn | 26.16 | -0.003 | 0.334 | High |
| L13 | conv | 26.58 | -0.005 | 0.339 | High |
| **L14** | **norm** | **189.54** | -0.038 | **2.418** | **OUTPUT** |

**Critical finding: The residual stream has THREE distinct phases:**

1. **Quiet phase (L0-L5):** Norms ~1.7-2.6. The model is building up initial representations
   through convolutions and early attention. Very stable, low-magnitude activations.

2. **High phase (L6-L13):** Norms ~25.5-26.6 (10x jump from L5→L6). There is a massive
   discontinuity at L6 where the residual stream magnitude jumps 10x. This happens at the
   THIRD attention layer (L6). After this jump, norms are very stable with slow growth.

3. **Output phase (L14 = embedding_norm):** Norm = 189.5 (7.4x jump). The embedding_norm
   projects into a much higher magnitude space for the LM head.

**The L5→L6 transition is the most important architectural boundary in this model.**
L6 is the third attention layer. Something about the accumulated representation through
L0-L5 (2 convs, 1 attn, 1 conv, 1 attn, 1 conv) creates a representation that, when
processed by L6's attention, triggers a 10x magnitude increase. This could be:
- A learned scaling that gates "real" processing vs "warmup"
- A consequence of the gated conv architecture accumulating energy
- A design choice to separate "feature extraction" (L0-L5) from "reasoning" (L6-L13)

---

## Tokenizer

| Property | Value |
|---|---|
| Vocab size | 65,536 (64K) |
| BOS token | `<\|startoftext\|>` (id=1) |
| EOS token | `<\|im_end\|>` (id=7) |
| PAD token | `<\|pad\|>` (id=0) |
| Padding side | right |

**Tokenization examples:**
- "Hello world" → 3 tokens: [`<\|startoftext\|>`, `Hello`, ` world`]
- '{"key": "value"}' → 8 tokens (decent JSON handling)
- 'def fibonacci(n):' → 7 tokens (splits `fibonacci` into `fib` + `on` + `acci`)

**Vocab is 4x smaller than Qwen (65K vs 152K).** This means:
- More tokens per text (less efficient per token)
- But smaller embedding table (67M vs 152M for 0.5B)
- Task suite targets need to be verified as single-token in THIS tokenizer

---

## Generation Quality (Baseline)

Tested with temperature=0.1, top_k=50:

- "The capital of France is" → "Paris." ✓
- '{"name": "test", "value":' → '{"name": "test", "value": "test"}\n}' ✓ (valid JSON!)
- 'def hello_world():' → Full function with print, return, and `if __name__` guard ✓

**This is a surprisingly capable 230M model.** Despite the model card saying "not recommended
for reasoning/code", it handles JSON and code generation well at basic levels.

---

## Comparison to Qwen2.5-0.5B

| Property | LFM2.5-230M | Qwen2.5-0.5B |
|---|---|---|
| Params | 229.7M | 494M |
| Layers | 14 (8 conv + 6 attn) | 24 (all attn) |
| Hidden size | 1024 | 896 |
| Attn heads | 16Q / 8KV | 14 (GQA) |
| Vocab | 65K | 152K |
| Context | 128K | 32K |
| Layer types | Hybrid (conv+attn) | Uniform (attn-only) |
| MLP type | SwiGLU | SwiGLU |
| Norm | RMSNorm | RMSNorm |
| Positional | RoPE (theta=1M) | RoPE (theta=1M) |
| Architecture | Lfm2ForCausalLM | Qwen2ForCausalLM |

**Key differences for MI work:**
1. Hybrid layers — ablation must handle conv AND attn layers differently
2. Conv layers have no attention heads — head ablation only applies to 6/14 layers
3. The residual stream has a massive discontinuity at L6 (not present in Qwen)
4. Conv layers use gated convolution (B, C, x decomposition) — ablation of the gate vs carrier
   is a novel experiment not possible in pure transformers
5. Smaller vocab means different tokenization characteristics

---

## MI-Atlas Adaptation Notes

### What works unchanged:
- Layer-level ablation (hook `model.layers[i]` output → zero/mean/resample)
- MLP ablation (hook `model.layers[i].feed_forward` output)
- Steering vectors (add to residual stream at any layer)
- Position-specific ablation
- LoRA training (target_modules work for both conv and attn submodules)
- Activation patching (same residual stream hooks)

### What needs adaptation:
- Head ablation only applies to layers L2, L4, L6, L8, L10, L12 (6 layers)
- Conv-specific experiments: gate ablation (B), carrier ablation (C), signal ablation (x)
- Conv kernel analysis: what does each of the 4 kernel positions learn?
- Conv state analysis: how does the 3-token conv state affect generation?
- Module sweep must include conv-specific modules (in_proj, conv1d, out_proj)
- Task suite must be re-verified for single-token targets with 65K vocab

### Novel experiments unique to LFM2.5:
1. **Conv vs Attn contribution ratio:** What fraction of total processing is done by conv vs attn?
2. **Gate analysis:** Ablate B (gate) vs C (carrier) vs x (signal) independently in conv layers
3. **L6 boundary analysis:** What causes the 10x norm jump? Is L6 structurally different?
4. **Conv kernel depth analysis:** Does kernel_size=4 mean conv layers only "see" 4 tokens back?
5. **Alternation pattern importance:** What if we run all-attn or all-conv? (requires training)
6. **Conv→Attn handoff:** How does information flow from conv to attn layers?
7. **Early conv warmup:** L0-L1 are both conv before the first attn at L2. Why?
8. **Late conv closure:** L13 is conv after the last attn at L12. Why?
9. **Conv state analysis:** What information is stored in the 3-token conv state?
10. **Head dim analysis:** With head_dim=64 and GQA 2:1, how do the 8 KV heads specialize?

---

## File: LFM2.5-230M Architecture Document
## Date: 2026-06-28
## Status: Complete — ready for MI-Atlas pipeline adaptation
