#!/usr/bin/env python3
"""
Aggressive reasoning compression:
1. Strip JSON/code blocks from reasoning (they're duplicated in response)
2. Keep only the planning phase (first ~40% of lines)
3. Truncate to budget
"""
import json, re, sys

def aggressive_compress(text, max_chars=2000):
    if not text or len(text) < max_chars:
        return text
    
    lines = text.split('\n')
    
    # Phase 1: Strip lines that look like they contain the actual output
    # (JSON objects, code blocks, long strings that match the response)
    planning_lines = []
    in_code_block = False
    
    for line in lines:
        stripped = line.strip()
        
        # Track code blocks
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        
        # Skip JSON-like lines (contain quotes, braces, colons in JSON pattern)
        if re.match(r'^[\s]*[\{\["\'].*[\}\]"\']', stripped) and len(stripped) > 50:
            continue
        if re.match(r'^[\s]*"[\w_]+"\s*:', stripped):
            continue
        
        # Skip lines that are clearly the output being drafted
        if any(p in stripped for p in ['"parser_review_label"', '"sample_text"', '"chapter_title"', 
                                        '"trial_title"', '"config_snippet"', '"audit_focus"']):
            continue
        
        planning_lines.append(stripped)
    
    # Phase 2: Keep only the first ~40% of planning lines (the actual reasoning)
    cutoff = max(1, len(planning_lines) // 2 + len(planning_lines) // 10)  # ~60% = first 60%
    result = '\n'.join(planning_lines[:cutoff])
    
    # Phase 3: Truncate to budget
    if len(result) > max_chars:
        # Find a good cut point (end of sentence or line)
        cut = result[:max_chars].rfind('\n')
        if cut < max_chars // 2:
            cut = max_chars
        result = result[:cut]
    
    return result

input_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".jsonl", "_compact.jsonl")

total_orig = 0
total_comp = 0
count = 0

with open(input_path) as fin, open(output_path, 'w') as fout:
    for line in fin:
        rec = json.loads(line)
        count += 1
        
        orig = rec.get("reasoning", "")
        total_orig += len(orig)
        
        comp = aggressive_compress(orig)
        total_comp += len(comp)
        
        rec["reasoning_compact"] = comp
        rec["reasoning_compact_chars"] = len(comp)
        fout.write(json.dumps(rec, default=str) + "\n")

saved_pct = (1 - total_comp / max(total_orig, 1)) * 100
print(f"Processed {count} rollouts")
print(f"  Original: {total_orig:,} chars (avg {total_orig//count:,})")
print(f"  Compact:  {total_comp:,} chars (avg {total_comp//count:,})")
print(f"  Saved:    {saved_pct:.1f}%")
