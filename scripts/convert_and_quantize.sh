#!/bin/bash
# convert_and_quantize.sh — Convert HF models to GGUF and quantize at multiple levels.
# Usage: ./scripts/convert_and_quantize.sh Qwen/Qwen2.5-0.5B 0.5b
#        ./scripts/convert_and_quantize.sh Qwen/Qwen2.5-1.5B 1.5b

set -e

MODEL_NAME="$1"
SUFFIX="$2"
LLAMA_DIR="$HOME/llama.cpp"
LLAMA_BIN="$HOME/llama.cpp/build/bin"
GGUF_DIR="$HOME/gguf_models"

mkdir -p "$GGUF_DIR"

echo "============================================================"
echo "  GGUF CONVERSION + QUANTIZATION"
echo "  Model: $MODEL_NAME"
echo "  Suffix: $SUFFIX"
echo "============================================================"

# Step 1: Convert HF model to GGUF (fp16)
echo ""
echo "  [Step 1] Converting $MODEL_NAME to GGUF (fp16)..."
FP16_PATH="$GGUF_DIR/${SUFFIX}_fp16.gguf"

if [ -f "$FP16_PATH" ]; then
    echo "    Already exists: $FP16_PATH"
else
    python3 "$LLAMA_DIR/convert_hf_to_gguf.py" \
        "$MODEL_NAME" \
        --outfile "$FP16_PATH" \
        --outtype f16 2>&1
    echo "    Saved: $FP16_PATH ($(du -h "$FP16_PATH" | cut -f1))"
fi

# Step 2: Quantize to q8_0 (8-bit)
echo ""
echo "  [Step 2] Quantizing to Q8_0 (8-bit)..."
Q8_PATH="$GGUF_DIR/${SUFFIX}_q8_0.gguf"

if [ -f "$Q8_PATH" ]; then
    echo "    Already exists: $Q8_PATH"
else
    "$LLAMA_BIN/llama-quantize" \
        "$FP16_PATH" \
        "$Q8_PATH" \
        Q8_0 2>&1
    echo "    Saved: $Q8_PATH ($(du -h "$Q8_PATH" | cut -f1))"
fi

# Step 3: Quantize to q4_K_M (4-bit, K-quants medium)
echo ""
echo "  [Step 3] Quantizing to Q4_K_M (4-bit K-quants medium)..."
Q4_PATH="$GGUF_DIR/${SUFFIX}_q4_k_m.gguf"

if [ -f "$Q4_PATH" ]; then
    echo "    Already exists: $Q4_PATH"
else
    "$LLAMA_BIN/llama-quantize" \
        "$FP16_PATH" \
        "$Q4_PATH" \
        Q4_K_M 2>&1
    echo "    Saved: $Q4_PATH ($(du -h "$Q4_PATH" | cut -f1))"
fi

# Summary
echo ""
echo "  CONVERSION COMPLETE"
echo "  Files:"
ls -lh "$GGUF_DIR"/${SUFFIX}_*.gguf 2>/dev/null
echo ""
echo "  Usage for qualitative analysis:"
echo "    python scripts/run_qualitative_analysis.py --model $FP16_PATH --suffix ${SUFFIX}_fp16 --backend llama"
echo "    python scripts/run_qualitative_analysis.py --model $Q8_PATH --suffix ${SUFFIX}_q8 --backend llama"
echo "    python scripts/run_qualitative_analysis.py --model $Q4_PATH --suffix ${SUFFIX}_q4 --backend llama"
