#!/usr/bin/env python3
"""
Compress reasoning traces from rollouts.
Removes repeated schema definitions, self-validation loops, redundant confirmation.
Keeps: planning, key decisions, final validation.

Applied BEFORE teacher scoring so the teacher sees concise reasoning.
"""
import json, re, sys
from pathlib import Path

def compress_reasoning(text):
    """Compress a reasoning trace, keeping the useful parts."""
    if not text or len(text) < 200:
        return text
    
    lines = text.split('\n')
    compressed = []
    seen_schemas = set()
    prev_line = ""
    skip_next = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines that follow empty lines
        if not stripped and not prev_line:
            continue
        
        # Skip repeated schema definitions
        schema_key = stripped[:100].lower()
        if any(w in stripped.lower() for w in ['fields:', 'each item has', 'each sample', 'properties:', 'type:']):
            if schema_key in seen_schemas:
                continue
            seen_schemas.add(schema_key)
        
        # Skip self-validation loops
        if re.match(r'^check (if|whether|that|all|any|no)', stripped.lower()):
            # Keep only the first validation, skip repeated ones
            if any('check' in c.lower() for c in compressed[-5:]):
                continue
        
        # Skip redundant confirmation
        if stripped.lower() in [
            'that should satisfy.', 'that seems correct.', 'that should work.',
            'good.', 'all good.', 'perfect.', 'excellent.', 'done.',
            'thus output exactly that.', 'thus final answer.',
            'provide as bare list.', 'no extra commentary.',
            'make sure no extra fields.', 'that should satisfy',
        ]:
            continue
        
        # Skip "Make sure..."重复 (keep first only)
        if stripped.lower().startswith('make sure') and any(c.lower().startswith('make sure') for c in compressed[-10:]):
            continue
        
        # Skip repeated "We need to..." at start of reasoning
        if stripped.lower().startswith('we need to') and i > 20:
            # Count how many "we need to" we've seen
            need_count = sum(1 for c in compressed if c.lower().startswith('we need to'))
            if need_count >= 2:
                continue
        
        # Skip lines that just restate the output format
        if re.match(r'^(so the|the output|output format|format is|the format)', stripped.lower()):
            if any(re.match(r'^(so the|the output|output format|format is|the format)', c.lower()) for c in compressed[-10:]):
                continue
        
        # Keep everything else
        compressed.append(stripped)
        prev_line = stripped
    
    result = '\n'.join(compressed)
    
    # Remove duplicated paragraphs (exact or near-exact)
    paragraphs = result.split('\n\n')
    seen = set()
    deduped = []
    for p in paragraphs:
        key = p.strip()[:200].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    
    return '\n\n'.join(deduped)

def process_rollouts(input_path, output_path):
    """Process a rollouts.jsonl file, adding compressed reasoning."""
    stats = {
        "total": 0,
        "compressed": 0,
        "total_reasoning_chars": 0,
        "compressed_reasoning_chars": 0,
    }
    
    with open(input_path) as fin, open(output_path, 'w') as fout:
        for line in fin:
            rec = json.loads(line)
            stats["total"] += 1
            
            reasoning = rec.get("reasoning", "")
            stats["total_reasoning_chars"] += len(reasoning)
            
            if reasoning and len(reasoning) > 200:
                compressed = compress_reasoning(reasoning)
                rec["reasoning_compressed"] = compressed
                rec["reasoning_compressed_chars"] = len(compressed)
                stats["compressed_reasoning_chars"] += len(compressed)
                stats["compressed"] += 1
            else:
                rec["reasoning_compressed"] = reasoning
                rec["reasoning_compressed_chars"] = len(reasoning)
                stats["compressed_reasoning_chars"] += len(reasoning)
            
            fout.write(json.dumps(rec, default=str) + "\n")
    
    ratio = stats["compressed_reasoning_chars"] / max(stats["total_reasoning_chars"], 1)
    saved = (1 - ratio) * 100
    print(f"Compressed {stats['compressed']}/{stats['total']} reasoning traces")
    print(f"  Before: {stats['total_reasoning_chars']:,} chars")
    print(f"  After:  {stats['compressed_reasoning_chars']:,} chars")
    print(f"  Saved:  {saved:.1f}% ({stats['total_reasoning_chars'] - stats['compressed_reasoning_chars']:,} chars)")
    
    return stats

if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/home/billz/rollouts/rollouts.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".jsonl", "_compressed.jsonl")
    
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    process_rollouts(input_path, output_path)
