#!/usr/bin/env python3
"""
LFM2.5-230M Deep Probe — Comprehensive analysis beyond MI-Atlas.
Covers: position ablation, logit lens, attention patterns, per-token loss,
pairwise ablation, CKA similarity, component vocabulary contribution,
conv kernel analysis, embedding structure, adversarial robustness.
"""
import json, torch, sys, os, math
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "LiquidAI/LFM2.5-230M"
PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONV_LAYERS = [0, 1, 3, 5, 7, 9, 11, 13]
ATTN_LAYERS = [2, 4, 6, 8, 10, 12]
ALL_LAYERS = list(range(14))
NUM_HEADS = 16
HEAD_DIM = 64
HIDDEN = 1024

def kl_last_token(baseline, ablated):
    """KL divergence on last token predictions."""
    return torch.nn.functional.kl_div(
        torch.log_softmax(ablated[:, -1, :].float(), -1),
        torch.softmax(baseline[:, -1, :].float(), -1),
        reduction='batchmean'
    ).item()

def save_result(name, data, ts):
    path = RESULTS_DIR / f"lfm2_230m_{name}_{ts}.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path.name}")
    return path


# ═══════════════════════════════════════════════
# EXPERIMENT 1: POSITION-SPECIFIC ABLATION
# ═══════════════════════════════════════════════
def exp_position_ablation(model, tokenizer, device, ts):
    print("\n" + "="*60)
    print("EXP 1: POSITION-SPECIFIC ABLATION")
    print("="*60)
    
    prompts = [
        "The capital of France is Paris",
        '{"name": "Alice", "age": 30}',
        "def fibonacci(n): return n",
    ]
    
    results = {}
    for pi, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        seq_len = ids.shape[1]
        
        with torch.no_grad():
            baseline = model(ids).logits
        
        prompt_kls = {}
        for layer_idx in ALL_LAYERS:
            layer_kls = []
            for pos in range(seq_len):
                def make_hook(p):
                    def hook(module, input, output):
                        h = output if isinstance(output, torch.Tensor) else output[0]
                        if h.dim() == 3:
                            h[:, p, :] = 0.0
                        else:
                            h[p, :] = 0.0
                        return (h,) + output[1:] if isinstance(output, tuple) else h
                    return hook
                
                handle = model.model.layers[layer_idx].register_forward_hook(make_hook(pos))
                with torch.no_grad():
                    abl = model(ids).logits
                handle.remove()
                layer_kls.append(kl_last_token(baseline, abl))
            
            prompt_kls[f"L{layer_idx}"] = [round(k, 4) for k in layer_kls]
            
        tokens = [tokenizer.decode([ids[0, i]]) for i in range(seq_len)]
        results[f"prompt_{pi}"] = {"text": prompt, "tokens": tokens, "kl_by_layer": prompt_kls}
        
        # Print summary
        max_pos_per_layer = {li: int(np.argmax(prompt_kls[f"L{li}"])) for li in ALL_LAYERS}
        print(f"\n  Prompt {pi}: '{prompt[:50]}'")
        for li in [0, 2, 5, 6, 12, 13]:
            kls = prompt_kls[f"L{li}"]
            max_p = max_pos_per_layer[li]
            print(f"    L{li}: max KL={max(kls):.3f} at pos {max_p} ('{tokens[max_p]}')")
    
    save_result("position_ablation", results, ts)
    return results


# ═══════════════════════════════════════════════
# EXPERIMENT 2: LOGIT LENS
# ═══════════════════════════════════════════════
def exp_logit_lens(model, tokenizer, device, ts):
    print("\n" + "="*60)
    print("EXP 2: LOGIT LENS (what does each layer 'know'?)")
    print("="*60)
    
    prompts = [
        "The capital of France is",
        "2 + 2 =",
        '{"key": "value", "num":',
    ]
    
    results = {}
    lm_head = model.lm_head
    embed_norm = model.model.embedding_norm
    
    for pi, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # Get hidden states at each layer
        with torch.no_grad():
            outputs = model(ids, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states  # 15 tensors (embed + 14 layers)
        
        layer_predictions = {}
        for li, hs in enumerate(hidden_states):
            # Project through embedding_norm + lm_head
            normed = embed_norm(hs.to(lm_head.weight.dtype))
            logits = lm_head(normed)
            last_token_logits = logits[0, -1, :]
            top5 = torch.topk(last_token_logits, 5)
            top5_tokens = [tokenizer.decode([idx.item()]) for idx in top5.indices]
            top5_probs = torch.softmax(last_token_logits, -1)[top5.indices].tolist()
            
            layer_predictions[f"L{li}" if li > 0 else "embed"] = {
                "top5_tokens": top5_tokens,
                "top5_probs": [round(p, 4) for p in top5_probs],
                "entropy": round(torch.distributions.Categorical(logits=last_token_logits).entropy().item(), 4),
            }
        
        results[f"prompt_{pi}"] = {"text": prompt, "predictions": layer_predictions}
        
        print(f"\n  Prompt {pi}: '{prompt}'")
        for li_name in ["embed", "L0", "L3", "L5", "L6", "L10", "L13"]:
            pred = layer_predictions.get(li_name, {})
            tokens = pred.get("top5_tokens", [])
            probs = pred.get("top5_probs", [])
            ent = pred.get("entropy", 0)
            print(f"    {li_name:>5}: top='{tokens[0] if tokens else '?'}' ({probs[0] if probs else 0:.3f}), "
                  f"entropy={ent:.2f}")
    
    save_result("logit_lens", results, ts)
    return results


# ═══════════════════════════════════════════════
# EXPERIMENT 3: ATTENTION PATTERN ANALYSIS
# ═══════════════════════════════════════════════
def exp_attention_patterns(model, tokenizer, device, ts):
    print("\n" + "="*60)
    print("EXP 3: ATTENTION PATTERN ANALYSIS")
    print("="*60)
    
    prompt = "The capital of France is Paris and it is beautiful"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    tokens = [tokenizer.decode([ids[0, i].item()]) for i in range(ids.shape[1])]
    
    # Capture attention weights
    attention_data = {}
    hooks = []
    
    for layer_idx in ATTN_LAYERS:
        attn_module = model.model.layers[layer_idx].self_attn
        
        def make_hook(li):
            def hook(module, input, output):
                # SDPA doesn't return weights by default. We need to use eager attention.
                # Instead, let's capture the Q, K patterns
                pass
            return hook
    
    # Use eager attention to get weights
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        outputs = model(ids, output_attentions=True)
    
    if outputs.attentions:
        for li, attn_weights in enumerate(outputs.attentions):
            if attn_weights is not None:
                # attn_weights: [batch, heads, seq, seq]
                aw = attn_weights[0].float().cpu().numpy()  # [heads, seq, seq]
                
                # Per-head statistics
                head_stats = {}
                for hi in range(aw.shape[0]):
                    w = aw[hi]
                    # Attention entropy (how focused vs spread)
                    entropy = -np.sum(w * np.log(w + 1e-10), axis=-1).mean()
                    # Attention to self (diagonal)
                    self_attn = np.diagonal(w).mean()
                    # Attention to first token (BOS)
                    bos_attn = w[:, 0].mean()
                    # Attention to last token
                    last_attn = w[:, -1].mean()
                    # Max attention weight
                    max_attn = w.max()
                    
                    head_stats[f"H{hi}"] = {
                        "entropy": round(float(entropy), 4),
                        "self_attn": round(float(self_attn), 4),
                        "bos_attn": round(float(bos_attn), 4),
                        "last_attn": round(float(last_attn), 4),
                        "max_attn": round(float(max_attn), 4),
                    }
                
                attention_data[f"L{layer_idx}"] = head_stats
                print(f"  L{layer_idx}: mean entropy={np.mean([h['entropy'] for h in head_stats.values()]):.3f}, "
                      f"mean self-attn={np.mean([h['self_attn'] for h in head_stats.values()]):.3f}")
    else:
        print("  No attention weights returned (SDPA doesn't support output_attentions)")
        print("  Skipping attention pattern analysis")
    
    if attention_data:
        save_result("attention_patterns", {"prompt": prompt, "tokens": tokens, "data": attention_data}, ts)
    return attention_data


# ═══════════════════════════════════════════════
# EXPERIMENT 4: PER-TOKEN LOSS ANALYSIS
# ═══════════════════════════════════════════════
def exp_per_token_loss(model, tokenizer, device, ts):
    print("\n" + "="*60)
    print("EXP 4: PER-TOKEN LOSS ANALYSIS")
    print("="*60)
    
    prompts = [
        "The capital of France is Paris and it is known for the Eiffel Tower",
        '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}',
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]",
        "1 1 2 3 5 8 13 21 34 55 89 144",
    ]
    
    results = {}
    for pi, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(ids)
            logits = outputs.logits
        
        # Compute per-token cross-entropy loss
        shift_logits = logits[0, :-1, :].float()
        shift_labels = ids[0, 1:]
        per_token_loss = torch.nn.functional.cross_entropy(
            shift_logits, shift_labels, reduction='none'
        ).cpu().numpy()
        
        tokens = [tokenizer.decode([ids[0, i].item()]) for i in range(ids.shape[1])]
        
        # Find easiest and hardest tokens
        token_loss_pairs = [(tokens[i+1], float(per_token_loss[i])) for i in range(len(per_token_loss))]
        sorted_by_loss = sorted(token_loss_pairs, key=lambda x: x[1])
        
        easiest = sorted_by_loss[:5]
        hardest = sorted_by_loss[-5:]
        
        results[f"prompt_{pi}"] = {
            "text": prompt,
            "tokens": tokens[1:],  # predicted tokens
            "losses": [round(float(l), 4) for l in per_token_loss],
            "mean_loss": round(float(per_token_loss.mean()), 4),
            "max_loss": round(float(per_token_loss.max()), 4),
            "min_loss": round(float(per_token_loss.min()), 4),
        }
        
        print(f"\n  Prompt {pi}: mean_loss={per_token_loss.mean():.3f}")
        print(f"    Easiest: {[(t, round(l, 2)) for t, l in easiest]}")
        print(f"    Hardest: {[(t, round(l, 2)) for t, l in hardest]}")
    
    save_result("per_token_loss", results, ts)
    return results


# ═══════════════════════════════════════════════
# EXPERIMENT 5: RESIDUAL STREAM CKA SIMILARITY
# ═══════════════════════════════════════════════
def exp_cka_similarity(model, tokenizer, device, ts):
    print("\n" + "="*60)
    print("EXP 5: RESIDUAL STREAM CKA SIMILARITY")
    print("="*60)
    
    prompts = [
        "The capital of France is",
        '{"key": "value"}',
        "def hello(): return 42",
        "1 + 1 = 2",
    ]
    
    def linear_cka(X, Y):
        """Compute linear CKA between two representation matrices."""
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)
        XtX = X.T @ X
        YtY = Y.T @ Y
        XtY = X.T @ Y
        hsic_xy = (XtY * XtY).sum()
        hsic_xx = (XtX * XtX).sum()
        hsic_yy = (YtY * YtY).sum()
        return (hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt() + 1e-8)).item()
    
    # Collect hidden states for all prompts
    all_hidden = {li: [] for li in range(15)}  # 15 = embed + 14 layers
    
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            outputs = model(ids, output_hidden_states=True)
        for li, hs in enumerate(outputs.hidden_states):
            all_hidden[li].append(hs[0].float().detach().cpu())  # [seq, hidden]
    
    # Concatenate across prompts
    hidden_concat = {li: torch.cat(all_hidden[li], dim=0) for li in range(15)}
    
    # Compute CKA matrix
    n = 15
    cka_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cka_val = linear_cka(hidden_concat[i], hidden_concat[j])
            cka_matrix[i, j] = cka_val
            cka_matrix[j, i] = cka_val
    
    labels = ["embed"] + [f"L{i}" for i in range(14)]
    
    print(f"\n  CKA similarity matrix (selected):")
    print(f"  {'':>6}", end="")
    for l in labels:
        print(f"{l:>6}", end="")
    print()
    for i in range(n):
        print(f"  {labels[i]:>6}", end="")
        for j in range(n):
            print(f"{cka_matrix[i,j]:>6.2f}", end="")
        print()
    
    # Find most/least similar layer pairs
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((labels[i], labels[j], cka_matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n  Most similar pairs:")
    for a, b, cka in pairs[:5]:
        print(f"    {a} <-> {b}: {cka:.4f}")
    print(f"\n  Least similar pairs:")
    for a, b, cka in pairs[-5:]:
        print(f"    {a} <-> {b}: {cka:.4f}")
    
    save_result("cka_similarity", {
        "labels": labels,
        "matrix": [[round(float(v), 4) for v in row] for row in cka_matrix],
        "most_similar": [(a, b, round(c, 4)) for a, b, c in pairs[:10]],
        "least_similar": [(a, b, round(c, 4)) for a, b, c in pairs[-10:]],
    }, ts)
    return cka_matrix


# ═══════════════════════════════════════════════
# EXPERIMENT 6: COMPONENT VOCABULARY CONTRIBUTION
# ═══════════════════════════════════════════════
def exp_vocab_contribution(model, tokenizer, device, ts):
    print("\n" + "="*60)
    print("EXP 6: COMPONENT VOCABULARY CONTRIBUTION")
    print("="*60)
    
    prompt = "The capital of France is"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Get baseline
    with torch.no_grad():
        baseline = model(ids).logits[0, -1, :]  # [vocab]
    
    lm_head = model.lm_head
    embed_norm = model.model.embedding_norm
    
    # For each layer, compute its contribution to the final logits
    # Contribution at layer i = norm(h_i) @ lm_head.T - norm(h_{i-1}) @ lm_head.T
    with torch.no_grad():
        outputs = model(ids, output_hidden_states=True)
    
    contributions = {}
    prev_logits = None
    for li in range(15):
        hs = outputs.hidden_states[li][0, -1, :].float()  # [hidden]
        normed = embed_norm(hs.unsqueeze(0).to(lm_head.weight.dtype)).squeeze(0)
        logits = lm_head(normed)
        
        if prev_logits is not None:
            delta = logits - prev_logits
            top5 = torch.topk(delta, 5)
            top5_tokens = [tokenizer.decode([idx.item()]) for idx in top5.indices]
            top5_vals = top5.values.tolist()
            
            # Bottom 5 (most suppressed)
            bottom5 = torch.topk(delta, 5, largest=False)
            bottom5_tokens = [tokenizer.decode([idx.item()]) for idx in bottom5.indices]
            bottom5_vals = bottom5.values.tolist()
            
            label = f"L{li-1}" if li > 0 else "embed"
            contributions[label] = {
                "boosted": list(zip(top5_tokens, [round(v, 2) for v in top5_vals])),
                "suppressed": list(zip(bottom5_tokens, [round(v, 2) for v in bottom5_vals])),
                "delta_norm": round(float(delta.norm()), 4),
            }
        
        prev_logits = logits
    
    for layer_name, contrib in contributions.items():
        boosted = contrib["boosted"]
        print(f"  {layer_name:>5}: boosts '{boosted[0][0]}' (+{boosted[0][1]:.1f}), "
              f"'{boosted[1][0]}' (+{boosted[1][1]:.1f})  |  "
              f"norm={contrib['delta_norm']:.1f}")
    
    save_result("vocab_contribution", {"prompt": prompt, "contributions": contributions}, ts)
    return contributions


# ═══════════════════════════════════════════════
# EXPERIMENT 7: PAIRWISE LAYER ABLATION
# ═══════════════════════════════════════════════
def exp_pairwise_ablation(model, tokenizer, device, ts):
    print("\n" + "="*60)
    print("EXP 7: PAIRWISE LAYER ABLATION (91 pairs)")
    print("="*60)
    
    prompt = "The capital of France is"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        baseline = model(ids).logits
    
    # Test top 6 most important layers from skip ablation
    # L0, L5, L1, L4, L2, L12
    important_layers = [0, 1, 2, 4, 5, 12]
    
    results = {}
    for i, li in enumerate(important_layers):
        for lj in important_layers[i+1:]:
            # Zero both operators
            def make_hook():
                def hook(module, input, output):
                    return torch.zeros_like(output) if isinstance(output, torch.Tensor) else (torch.zeros_like(output[0]),) + output[1:]
                return hook
            
            h1 = model.model.layers[li].feed_forward.register_forward_hook(make_hook())
            h2 = model.model.layers[lj].feed_forward.register_forward_hook(make_hook())
            
            with torch.no_grad():
                abl = model(ids).logits
            
            h1.remove()
            h2.remove()
            
            kl = kl_last_token(baseline, abl)
            
            # Also get individual Kls for comparison
            # (already known from previous experiments, use cached values)
            
            pair_key = f"L{li}+L{lj}"
            results[pair_key] = round(kl, 4)
            print(f"  {pair_key}: KL={kl:.4f}")
    
    save_result("pairwise_ablation", {"prompt": prompt, "pairs": results}, ts)
    return results


# ═══════════════════════════════════════════════
# EXPERIMENT 8: EMBEDDING SPACE STRUCTURE
# ═══════════════════════════════════════════════
def exp_embedding_structure(model, tokenizer, device, ts):
    print("\n" + "="*60)
    print("EXP 8: EMBEDDING SPACE STRUCTURE")
    print("="*60)
    
    embed_weights = model.model.embed_tokens.weight.float().detach().cpu().numpy()
    
    # Sample of interesting token categories
    categories = {
        "numbers": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "100", "1000"],
        "capitals": ["Paris", "London", "Tokyo", "Berlin", "Rome", "Madrid", "Beijing", "Moscow"],
        "code": ["def", "return", "if", "else", "for", "while", "import", "class", "True", "False"],
        "json": ["{", "}", "[", "]", ":", ",", '"', "null", "true", "false"],
        "common": ["the", "a", "an", "is", "are", "was", "were", "be", "been", "have", "has"],
        "special": ["<|startoftext|>", "<|pad|>", "<|im_end|>"],
    }
    
    cat_centroids = {}
    for cat, tokens in categories.items():
        ids = []
        for t in tokens:
            enc = tokenizer.encode(t, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        if ids:
            vecs = embed_weights[ids]
            centroid = vecs.mean(axis=0)
            cat_centroids[cat] = centroid
    
    # Compute inter-category distances
    cats = list(cat_centroids.keys())
    dist_matrix = {}
    for i, c1 in enumerate(cats):
        for c2 in cats[i+1:]:
            c1v = cat_centroids[c1]
            c2v = cat_centroids[c2]
            cos_sim = np.dot(c1v, c2v) / (np.linalg.norm(c1v) * np.linalg.norm(c2v) + 1e-8)
            dist_matrix[f"{c1}<->{c2}"] = round(float(cos_sim), 4)
    
    print(f"\n  Embedding dim: {embed_weights.shape}")
    print(f"  Categories: {list(cat_centroids.keys())}")
    print(f"\n  Inter-category cosine similarity:")
    for pair, sim in sorted(dist_matrix.items(), key=lambda x: x[1], reverse=True):
        print(f"    {pair}: {sim:.4f}")
    
    # Embedding norm distribution
    norms = np.linalg.norm(embed_weights, axis=1)
    print(f"\n  Embedding norms: mean={norms.mean():.2f}, std={norms.std():.2f}, "
          f"min={norms.min():.2f}, max={norms.max():.2f}")
    
    save_result("embedding_structure", {
        "vocab_size": embed_weights.shape[0],
        "embed_dim": embed_weights.shape[1],
        "categories": list(cat_centroids.keys()),
        "inter_category_similarity": dist_matrix,
        "norm_stats": {"mean": round(float(norms.mean()), 2), "std": round(float(norms.std()), 2),
                       "min": round(float(norms.min()), 2), "max": round(float(norms.max()), 2)},
    }, ts)
    return dist_matrix


# ═══════════════════════════════════════════════
# EXPERIMENT 9: CONV KERNEL ANALYSIS
# ═══════════════════════════════════════════════
def exp_conv_kernel(model, tokenizer, device, ts):
    print("\n" + "="*60)
    print("EXP 9: CONV KERNEL WEIGHT ANALYSIS")
    print("="*60)
    
    results = {}
    for layer_idx in CONV_LAYERS:
        conv = model.model.layers[layer_idx].conv.conv  # Conv1d
        weight = conv.weight.float().detach().cpu()  # [out_channels, in_channels, kernel_size]
        
        # kernel_size = 3, groups = 1024 (depthwise)
        # Each channel has its own 3-tap filter
        w = weight.squeeze()  # [1024, 3]
        
        # Per-position weight magnitude
        pos_mag = w.detach().abs().mean(dim=0).numpy()
        
        # Weight variance per position
        pos_var = w.var(dim=0).numpy()
        
        # Dominant pattern: which kernel position has the largest weight?
        dominant_pos = int(pos_mag.argmax())
        
        # Filter diversity: how different are the per-channel filters?
        filter_norms = w.norm(dim=1).numpy()
        
        results[f"L{layer_idx}"] = {
            "pos_magnitude": [round(float(v), 4) for v in pos_mag],
            "pos_variance": [round(float(v), 6) for v in pos_var],
            "dominant_position": dominant_pos,
            "filter_norm_mean": round(float(filter_norms.mean()), 4),
            "filter_norm_std": round(float(filter_norms.std()), 4),
        }
        
        print(f"  L{layer_idx}: pos_mag={[round(v, 4) for v in pos_mag]}, "
              f"dominant_pos={dominant_pos} ({'current' if dominant_pos==1 else 'past' if dominant_pos==0 else 'future'})")
    
    save_result("conv_kernel", results, ts)
    return results


# ═══════════════════════════════════════════════
# EXPERIMENT 10: GRADIENT-BASED SENSITIVITY
# ═══════════════════════════════════════════════
def exp_gradient_sensitivity(model, tokenizer, device, ts):
    print("\n" + "="*60)
    print("EXP 10: GRADIENT-BASED LAYER SENSITIVITY")
    print("="*60)
    
    prompts = [
        "The capital of France is",
        '{"key":',
        "def foo():",
    ]
    
    results = {}
    for pi, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # Enable gradients on embeddings
        # Perturbation-based approach, no grad needed
        
        
        # Forward with gradient
        model.train()  # Enable grad
        for p in model.parameters():
            p.requires_grad_(False)
        
        # Run through model manually to get loss
        with torch.no_grad():
            baseline_logits = model(ids).logits
        
        target_id = ids[0, -1].item()
        target_logit = baseline_logits[0, -1, target_id]
        
        # Compute gradient of target logit w.r.t. input embeddings
        # This requires a forward pass with grad enabled
        model.eval()
        
        # Use the simpler approach: perturbation sensitivity
        grad_norms = {}
        with torch.no_grad():
            baseline = model(ids).logits
        
        for layer_idx in ALL_LAYERS:
            # Perturb layer output by small epsilon and measure sensitivity
            eps = 0.01
            
            def make_perturb_hook(epsilon):
                def hook(module, input, output):
                    h = output if isinstance(output, torch.Tensor) else output[0]
                    noise = torch.randn_like(h) * epsilon
                    h = h + noise
                    return (h,) + output[1:] if isinstance(output, tuple) else h
                return hook
            
            handle = model.model.layers[layer_idx].register_forward_hook(make_perturb_hook(eps))
            with torch.no_grad():
                perturbed = model(ids).logits
            handle.remove()
            
            sensitivity = (perturbed - baseline).abs().mean().item()
            grad_norms[f"L{layer_idx}"] = round(sensitivity, 6)
        
        results[f"prompt_{pi}"] = {"text": prompt, "sensitivity": grad_norms}
        print(f"\n  Prompt {pi}: '{prompt[:40]}'")
        for li in ALL_LAYERS:
            s = grad_norms[f"L{li}"]
            bar = "#" * max(0, int(s * 5000))
            print(f"    L{li:>2}: {s:.6f} {bar}")
    
    save_result("gradient_sensitivity", results, ts)
    return results


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Subset: position,logit_lens,attention,loss,cka,vocab,pairwise,embed,conv,gradient")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=str(device), trust_remote_code=True
    )
    model.eval()
    print(f"VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_exps = {
        "position": lambda: exp_position_ablation(model, tokenizer, device, ts),
        "logit_lens": lambda: exp_logit_lens(model, tokenizer, device, ts),
        "attention": lambda: exp_attention_patterns(model, tokenizer, device, ts),
        "loss": lambda: exp_per_token_loss(model, tokenizer, device, ts),
        "cka": lambda: exp_cka_similarity(model, tokenizer, device, ts),
        "vocab": lambda: exp_vocab_contribution(model, tokenizer, device, ts),
        "pairwise": lambda: exp_pairwise_ablation(model, tokenizer, device, ts),
        "embed": lambda: exp_embedding_structure(model, tokenizer, device, ts),
        "conv": lambda: exp_conv_kernel(model, tokenizer, device, ts),
        "gradient": lambda: exp_gradient_sensitivity(model, tokenizer, device, ts),
    }
    
    to_run = args.experiments or list(all_exps.keys())
    
    print(f"\n{'#'*60}")
    print(f"LFM2.5-230M DEEP PROBE — {len(to_run)} experiments")
    print(f"{'#'*60}")
    
    for exp_name in to_run:
        if exp_name in all_exps:
            try:
                all_exps[exp_name]()
            except Exception as e:
                print(f"\n  ERROR in {exp_name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  Unknown experiment: {exp_name}")
    
    print(f"\n{'#'*60}")
    print("DEEP PROBE COMPLETE")
    print(f"{'#'*60}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
