#!/usr/bin/env python3
"""
Comprehensive architecture inspection of LFM2.5-230M.
Maps every layer, module, parameter, and architectural detail.
"""
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LiquidAI/LFM2.5-230M"

def count_params(module, name=""):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def inspect_module_tree(model, prefix="", depth=0, max_depth=3, file=None):
    """Print module tree up to max_depth."""
    for name, child in model.named_children():
        total, trainable = count_params(child)
        line = f"{'  ' * depth}{name}: {child.__class__.__name__} ({total:,} params)"
        print(line, file=file)
        if depth < max_depth:
            inspect_module_tree(child, prefix=f"{prefix}.{name}", depth=depth+1, max_depth=max_depth, file=file)

def main():
    print("=" * 80)
    print("LFM2.5-230M Architecture Deep Inspection")
    print("=" * 80)
    
    # Load tokenizer
    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Model max length: {tokenizer.model_max_length}")
    print(f"  Padding side: {tokenizer.padding_side}")
    print(f"  Special tokens: {tokenizer.special_tokens_map}")
    print(f"  BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"  PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    
    # Test tokenization
    test_prompts = [
        "Hello world",
        "The capital of France is",
        '{"key": "value"}',
        "def fibonacci(n):",
    ]
    print("\n  Tokenization tests:")
    for prompt in test_prompts:
        ids = tokenizer.encode(prompt)
        tokens = [tokenizer.decode([tid]) for tid in ids]
        print(f"    '{prompt}' -> {len(ids)} tokens: {tokens}")
    
    # Load model
    print("\n[2] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Inspect on CPU first
        trust_remote_code=True,
    )
    
    # Config
    config = model.config
    print("\n[3] Model Config:")
    config_dict = {k: v for k, v in config.to_dict().items() if not k.startswith('_')}
    for key, value in sorted(config_dict.items()):
        if isinstance(value, (int, float, str, bool, type(None))):
            print(f"  {key}: {value}")
        elif isinstance(value, list) and len(value) < 30:
            print(f"  {key}: {value}")
        elif isinstance(value, dict):
            print(f"  {key}: {json.dumps(value, indent=2)}")
    
    # Total parameters
    total, trainable = count_params(model)
    print(f"\n[4] Parameter Counts:")
    print(f"  Total: {total:,} ({total/1e6:.1f}M)")
    print(f"  Trainable: {trainable:,} ({trainable/1e6:.1f}M)")
    
    # Embedding parameters
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_total, _ = count_params(model.model.embed_tokens)
        print(f"  Embedding: {embed_total:,} ({embed_total/1e6:.2f}M)")
    if hasattr(model, 'lm_head'):
        lm_total, _ = count_params(model.lm_head)
        print(f"  LM head: {lm_total:,} ({lm_total/1e6:.2f}M)")
    
    # Layer types
    print(f"\n[5] Layer Type Map:")
    if hasattr(config, 'layer_types'):
        for i, ltype in enumerate(config.layer_types):
            print(f"  L{i}: {ltype}")
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for i, layer in enumerate(model.model.layers):
            print(f"  L{i}: {layer.__class__.__name__}")
    
    # Module tree
    print(f"\n[6] Module Tree (depth 3):")
    inspect_module_tree(model, max_depth=3)
    
    # Per-layer parameter breakdown
    print(f"\n[7] Per-Layer Parameter Breakdown:")
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for i, layer in enumerate(model.model.layers):
            total, _ = count_params(layer)
            layer_type = config.layer_types[i] if hasattr(config, 'layer_types') else layer.__class__.__name__
            print(f"  L{i} ({layer_type}): {total:,} params ({total/1e6:.2f}M)")
            
            # Sub-module breakdown
            for name, child in layer.named_children():
                sub_total, _ = count_params(child)
                print(f"    {name}: {child.__class__.__name__} ({sub_total:,})")
    
    # Attention-specific inspection
    print(f"\n[8] Attention Layer Details:")
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for i, layer in enumerate(model.model.layers):
            layer_type = config.layer_types[i] if hasattr(config, 'layer_types') else layer.__class__.__name__
            if 'attention' in layer_type.lower() or 'Attention' in layer.__class__.__name__:
                # Find attention module
                attn = None
                for name, child in layer.named_children():
                    if 'attn' in name.lower() or 'attention' in name.lower():
                        attn = child
                        break
                if attn:
                    print(f"\n  L{i} ({layer_type}):")
                    for attr in ['num_heads', 'num_key_value_heads', 'head_dim', 'hidden_size',
                                 'num_attention_heads', 'd_model']:
                        if hasattr(attn, attr):
                            print(f"    {attr}: {getattr(attn, attr)}")
                    for name, child in attn.named_children():
                        sub_total, _ = count_params(child)
                        print(f"    {name}: {child.__class__.__name__} ({sub_total:,})")
    
    # Conv-specific inspection
    print(f"\n[9] Conv Layer Details:")
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for i, layer in enumerate(model.model.layers):
            layer_type = config.layer_types[i] if hasattr(config, 'layer_types') else layer.__class__.__name__
            if 'conv' in layer_type.lower() or 'Conv' in layer.__class__.__name__:
                print(f"\n  L{i} ({layer_type}):")
                for name, child in layer.named_children():
                    sub_total, _ = count_params(child)
                    sub_type = child.__class__.__name__
                    print(f"    {name}: {sub_type} ({sub_total:,})")
                    # Print shape info for conv layers
                    if hasattr(child, 'weight'):
                        print(f"      weight shape: {child.weight.shape}")
                    if hasattr(child, 'kernel_size'):
                        print(f"      kernel_size: {child.kernel_size}")
                    if hasattr(child, 'L_cache'):
                        print(f"      L_cache: {child.L_cache}")
                    # Check for custom attributes
                    for attr_name in dir(child):
                        if not attr_name.startswith('_') and attr_name not in ['weight', 'bias', 'training']:
                            try:
                                val = getattr(child, attr_name)
                                if isinstance(val, (int, float, bool, str)):
                                    print(f"      {attr_name}: {val}")
                            except:
                                pass
    
    # MLP/FFN inspection
    print(f"\n[10] MLP/FFN Details:")
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for i, layer in enumerate(model.model.layers):
            mlp = None
            for name, child in layer.named_children():
                if 'mlp' in name.lower() or 'ffn' in name.lower():
                    mlp = child
                    break
            if mlp:
                total, _ = count_params(mlp)
                print(f"  L{i}: {mlp.__class__.__name__} ({total:,} params)")
                for name, child in mlp.named_children():
                    sub_total, _ = count_params(child)
                    print(f"    {name}: {child.__class__.__name__} {list(child.weight.shape) if hasattr(child, 'weight') else ''} ({sub_total:,})")
    
    # Normalization layers
    print(f"\n[11] Normalization Layers:")
    norm_count = 0
    for name, module in model.named_modules():
        if 'norm' in name.lower() and 'Norm' in module.__class__.__name__:
            total, _ = count_params(module)
            attrs = {}
            if hasattr(module, 'eps'):
                attrs['eps'] = module.eps
            if hasattr(module, 'variance_epsilon'):
                attrs['eps'] = module.variance_epsilon
            print(f"  {name}: {module.__class__.__name__} ({total:,}) {attrs}")
            norm_count += 1
    print(f"  Total norm layers: {norm_count}")
    
    # Forward pass test
    print(f"\n[12] Forward Pass Test:")
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    test_input = "The capital of France is"
    inputs = tokenizer(test_input, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    
    print(f"  Input shape: {inputs['input_ids'].shape}")
    print(f"  Logits shape: {outputs.logits.shape}")
    print(f"  Number of hidden states: {len(outputs.hidden_states)}")
    print(f"  Hidden state shapes:")
    for i, hs in enumerate(outputs.hidden_states):
        print(f"    L{i}: {hs.shape} (dtype={hs.dtype})")
    
    if outputs.attentions:
        print(f"  Number of attention outputs: {len(outputs.attentions)}")
        for i, attn in enumerate(outputs.attentions):
            if attn is not None:
                print(f"    L{i}: {attn.shape}")
            else:
                print(f"    L{i}: None (conv layer)")
    
    # Generated text
    print(f"\n[13] Generation Test:")
    for prompt in ["The capital of France is", '{"name": "test", "value":', "def hello_world():"]:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(input_ids, max_new_tokens=30, temperature=0.1, do_sample=True, top_k=50)
        output = tokenizer.decode(gen[0], skip_special_tokens=True)
        print(f"  '{prompt}' -> '{output}'")
    
    # Residual stream analysis
    print(f"\n[14] Residual Stream Analysis:")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    for i, hs in enumerate(outputs.hidden_states):
        norm = hs.float().norm().item()
        mean = hs.float().mean().item()
        std = hs.float().std().item()
        print(f"  L{i}: norm={norm:.4f}, mean={mean:.6f}, std={std:.4f}")
    
    # Layer type classification summary
    print(f"\n[15] Architecture Summary:")
    if hasattr(config, 'layer_types'):
        from collections import Counter
        type_counts = Counter(config.layer_types)
        print(f"  Layer types: {dict(type_counts)}")
        print(f"  Total layers: {len(config.layer_types)}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Intermediate size: {config.intermediate_size}")
        print(f"  Num attention heads: {config.num_attention_heads}")
        print(f"  Num KV heads: {config.num_key_value_heads}")
        print(f"  Max position embeddings: {config.max_position_embeddings}")
        print(f"  Vocab size: {config.vocab_size}")
        print(f"  Tie embeddings: {config.tie_word_embeddings}")
        print(f"  RoPE theta: {config.rope_parameters.get('rope_theta', 'N/A')}")
    
    # VRAM footprint
    if torch.cuda.is_available():
        print(f"\n[16] VRAM Footprint:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        print(f"  Reserved: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
    
    print("\n" + "=" * 80)
    print("Inspection complete.")
    print("=" * 80)

if __name__ == "__main__":
    main()
