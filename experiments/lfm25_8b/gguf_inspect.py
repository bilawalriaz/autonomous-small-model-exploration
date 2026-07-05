#!/usr/bin/env python3
"""Inspect LFM2.5-8B-A1B GGUF: extract tensor map, shapes, sizes, config."""
import sys, json, re, os
from collections import defaultdict
sys.path.insert(0, os.path.expanduser("~/gguf-env/lib/python3.11/site-packages"))
from gguf import GGUFReader
import numpy as np

GGUF_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/LFM2.5-8B-A1B-Uncensored-Gaston-Q4_K_M.gguf")

print(f"Loading {GGUF_PATH}...")
reader = GGUFReader(GGUF_PATH)

# Extract metadata
print("\n" + "="*80)
print("METADATA")
print("="*80)
for key, val in reader.fields.items():
    if hasattr(val, 'parts'):
        # Try to get the value
        try:
            parts = [p.tolist() if hasattr(p, 'tolist') else p for p in val.parts]
            if len(parts) == 1:
                v = parts[0]
                if isinstance(v, list) and len(v) == 1:
                    v = v[0]
                print(f"  {key}: {v}")
            else:
                print(f"  {key}: {parts}")
        except:
            print(f"  {key}: <complex>")
    else:
        print(f"  {key}: {val}")

# Tensor analysis
print("\n" + "="*80)
print("TENSOR MAP")
print("="*80)

tensors_by_block = defaultdict(list)
all_tensors = []

for tensor in reader.tensors:
    name = tensor.name
    shape = tensor.data.shape
    n_params = 1
    for s in shape:
        n_params *= s
    n_bytes = tensor.data.nbytes
    
    # Parse block index
    m = re.match(r'blk\.(\d+)\.(.*)', name)
    if m:
        block_idx = int(m.group(1))
        suffix = m.group(2)
        tensors_by_block[block_idx].append((suffix, shape, n_params, n_bytes))
    else:
        all_tensors.append((name, shape, n_params, n_bytes))

# Print top-level tensors
print("\n--- Top-level tensors ---")
total_params = 0
total_bytes = 0
for name, shape, n_params, n_bytes in all_tensors:
    print(f"  {name:40s} shape={str(shape):20s} params={n_params:>12,}  size={n_bytes:>12,} bytes")
    total_params += n_params
    total_bytes += n_bytes

# Print per-block tensors
print("\n--- Per-block tensors ---")
for block_idx in sorted(tensors_by_block.keys()):
    tensors = tensors_by_block[block_idx]
    block_params = sum(p for _, _, p, _ in tensors)
    block_bytes = sum(b for _, _, _, b in tensors)
    total_params += block_params
    total_bytes += block_bytes
    
    # Classify block
    has_exps = any('exps' in t[0] for t in tensors)
    has_shortconv = any('shortconv' in t[0] for t in tensors)
    has_attn = any('attn_q' in t[0] for t in tensors)
    
    layer_type = "DENSE" if not has_exps else "MoE"
    sub_type = "conv" if has_shortconv else "attn" if has_attn else "?"
    
    print(f"\n  Layer {block_idx:2d} [{layer_type:4s}] [{sub_type:4s}] — {block_params:>12,} params, {block_bytes:>12,} bytes")
    for suffix, shape, n_params, n_bytes in tensors:
        print(f"    {suffix:45s} shape={str(shape):20s} params={n_params:>12,}")

print(f"\n{'='*80}")
print(f"TOTAL: {total_params:,} params, {total_bytes:,} bytes ({total_bytes/1e9:.2f} GB)")
print(f"{'='*80}")

# Weight statistics (sample from a few tensors)
print("\n" + "="*80)
print("WEIGHT STATISTICS (sample)")
print("="*80)

# Sample some key tensors
sample_names = ['token_embd.weight']
for block_idx in [0, 2, 10, 23]:
    for tensor in reader.tensors:
        if tensor.name == f'blk.{block_idx}.attn_q.weight':
            sample_names.append(tensor.name)
            break
    for tensor in reader.tensors:
        if tensor.name == f'blk.{block_idx}.ffn_gate.weight' or tensor.name == f'blk.{block_idx}.ffn_gate_exps.weight':
            sample_names.append(tensor.name)
            break
    for tensor in reader.tensors:
        if tensor.name == f'blk.{block_idx}.shortconv.conv.weight':
            sample_names.append(tensor.name)
            break

for tensor in reader.tensors:
    if tensor.name in sample_names:
        data = tensor.data.astype(np.float32)
        print(f"\n  {tensor.name}:")
        print(f"    shape: {data.shape}")
        print(f"    dtype: {tensor.data.dtype}")
        print(f"    min: {data.min():.6f}")
        print(f"    max: {data.max():.6f}")
        print(f"    mean: {data.mean():.6f}")
        print(f"    std: {data.std():.6f}")
        print(f"    near-zero (<1e-4): {(np.abs(data) < 1e-4).sum() / data.size * 100:.1f}%")
        print(f"    |val|>1: {(np.abs(data) > 1).sum() / data.size * 100:.1f}%")
        sample_names.remove(tensor.name)
