#!/usr/bin/env python3
"""
Routing Heatmap Analysis for LFM2.5-8B-A1B
============================================
Behavioral routing analysis: sends 40 prompts (8 categories × 5 variations)
to the model and measures response divergence to infer expert activation patterns.

Hypothesis: Categories with HIGH output diversity = experts fire differently
per variation. LOW diversity = same experts always fire for that task type.
"""

import json
import time
import statistics
import requests
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────
BASE_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "lfm2.5-8b-a1b"
RESULTS_DIR = Path("/home/billz/work/autonomous-small-model-exploration/results")
DOCS_DIR = Path("/home/billz/work/autonomous-small-model-exploration/docs")
MAX_TOKENS = 256
TEMPERATURE = 0.7
TIMEOUT = 60

# ── Prompt Categories ──────────────────────────────────────────────────
CATEGORIES = {
    "English Prose": [
        "Write a short paragraph about the ocean at sunset.",
        "Describe the feeling of walking through a quiet forest.",
        "Write a few sentences about an old lighthouse on a cliff.",
        "Describe a busy market in a small European town.",
        "Write a paragraph about the first snow of winter.",
    ],
    "Python Code": [
        "Write a Python function that checks if a number is prime.",
        "Write a Python function to reverse a linked list.",
        "Write a Python function that finds the longest palindrome in a string.",
        "Write a Python function to merge two sorted arrays.",
        "Write a Python function that converts decimal to binary.",
    ],
    "Math Reasoning": [
        "What is the derivative of x^3 + 2x^2 - 5x + 7?",
        "Solve for x: 3x + 7 = 22.",
        "What is the integral of sin(x)cos(x) dx?",
        "If f(x) = 2x^2 - 3x + 1, find f'(2).",
        "What is the sum of the first 20 positive integers?",
    ],
    "French Text": [
        "Écrivez une courte phrase sur la beauté de Paris.",
        "Décrivez une journée typique à la campagne française.",
        "Écrivez quelques mots sur la cuisine française.",
        "Décrivez le marché de Noël à Strasbourg.",
        "Écrivez un paragraphe sur les saisons en France.",
    ],
    "JSON Extraction": [
        'Extract the name and age from: "John is 30 years old and lives in NYC." Return JSON.',
        'Parse: "Apple Inc was founded by Steve Jobs in 1976." Return JSON with company, founder, year.',
        'From "The book has 350 pages and costs $24.99", extract book details as JSON.',
        'Parse: "Tokyo has 14 million people and is the capital of Japan." Return JSON.',
        'From "Dr. Smith treated 5 patients today at General Hospital", extract structured JSON.',
    ],
    "Medical Domain": [
        "What are the main symptoms of Type 2 diabetes?",
        "Explain how mRNA vaccines work in simple terms.",
        "What is the difference between bacterial and viral infections?",
        "Describe the role of white blood cells in the immune system.",
        "What are the common treatments for hypertension?",
    ],
    "Creative Writing": [
        "Write the opening line of a mystery novel.",
        "Write a haiku about artificial intelligence.",
        "Write a short dialogue between two old friends meeting after years.",
        "Write a poem about the passage of time.",
        "Write a dark fairy tale opening about a cursed forest.",
    ],
    "Step-by-Step Instructions": [
        "Explain how to make a cup of coffee step by step.",
        "Give step-by-step instructions to tie a bowline knot.",
        "Explain how to change a flat tire step by step.",
        "Describe the steps to bake chocolate chip cookies.",
        "Explain how to set up a basic home network step by step.",
    ],
}

# ── Utilities ──────────────────────────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def send_prompt(prompt: str) -> dict:
    """Send a prompt to the model and return response + timing."""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    t0 = time.time()
    try:
        resp = requests.post(BASE_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        wall_time = time.time() - t0
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {
            "content": content,
            "wall_time": wall_time,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    except Exception as e:
        wall_time = time.time() - t0
        return {
            "content": "",
            "wall_time": wall_time,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": str(e),
        }


def compute_pairwise_diversity(responses: list[str]) -> dict:
    """Compute diversity metrics across a set of responses."""
    if len(responses) < 2:
        return {
            "mean_levenshtein": 0,
            "max_levenshtein": 0,
            "normalized_diversity": 0,
            "pairwise_distances": [],
        }

    distances = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            d = levenshtein_distance(responses[i], responses[j])
            distances.append(d)

    mean_len = statistics.mean([len(r) for r in responses]) or 1
    return {
        "mean_levenshtein": statistics.mean(distances),
        "max_levenshtein": max(distances),
        "normalized_diversity": statistics.mean(distances) / mean_len if mean_len else 0,
        "pairwise_distances": distances,
    }


# ── Main Execution ─────────────────────────────────────────────────────

def run_experiment():
    """Run all prompts and collect results."""
    all_results = {}
    total = sum(len(v) for v in CATEGORIES.values())
    done = 0

    print(f"Routing Analysis: {total} prompts across {len(CATEGORIES)} categories")
    print("=" * 70)

    for category, prompts in CATEGORIES.items():
        cat_results = []
        for i, prompt in enumerate(prompts):
            done += 1
            print(f"  [{done}/{total}] {category} — variation {i+1}/5...", end=" ", flush=True)
            result = send_prompt(prompt)
            cat_results.append({
                "prompt": prompt,
                "response": result["content"],
                "wall_time": result["wall_time"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
            })
            status = "OK" if "error" not in result else f"ERR: {result['error']}"
            print(f"{status} ({result['wall_time']:.2f}s, {result.get('completion_tokens', '?')} tokens)")
            time.sleep(0.3)  # be polite to the server

        # ── Compute category-level metrics ──
        responses = [r["response"] for r in cat_results if r["response"]]
        wall_times = [r["wall_time"] for r in cat_results]
        completion_tokens = [r["completion_tokens"] for r in cat_results]

        diversity = compute_pairwise_diversity(responses)

        timing_mean = statistics.mean(wall_times) if wall_times else 0
        timing_std = statistics.stdev(wall_times) if len(wall_times) > 1 else 0
        timing_cv = timing_std / timing_mean if timing_mean else 0

        token_mean = statistics.mean(completion_tokens) if completion_tokens else 0
        token_std = statistics.stdev(completion_tokens) if len(completion_tokens) > 1 else 0
        token_cv = token_std / token_mean if token_mean else 0

        all_results[category] = {
            "trials": cat_results,
            "metrics": {
                "mean_levenshtein": round(diversity["mean_levenshtein"], 2),
                "max_levenshtein": round(diversity["max_levenshtein"], 2),
                "normalized_diversity": round(diversity["normalized_diversity"], 4),
                "timing_mean_s": round(timing_mean, 3),
                "timing_std_s": round(timing_std, 3),
                "timing_cv": round(timing_cv, 4),
                "token_mean": round(token_mean, 1),
                "token_std": round(token_std, 1),
                "token_cv": round(token_cv, 4),
                "pairwise_distances": diversity["pairwise_distances"],
            },
        }
        print(f"    → diversity={diversity['normalized_diversity']:.4f}  timing_cv={timing_cv:.4f}  token_cv={token_cv:.4f}")

    # ── Save results JSON ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "routing_heatmap_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    return all_results


# ── HTML Visualization ─────────────────────────────────────────────────

def generate_html(results: dict):
    """Generate a comprehensive HTML heatmap visualization."""
    categories = list(results.keys())

    # Collect metrics for heatmap
    heatmap_data = []
    for cat in categories:
        m = results[cat]["metrics"]
        heatmap_data.append({
            "category": cat,
            "normalized_diversity": m["normalized_diversity"],
            "timing_cv": m["timing_cv"],
            "token_cv": m["token_cv"],
            "mean_levenshtein": m["mean_levenshtein"],
            "timing_mean_s": m["timing_mean_s"],
            "token_mean": m["token_mean"],
        })

    # Build per-variation timing data for bar charts
    timing_by_category = {}
    for cat in categories:
        timing_by_category[cat] = [t["wall_time"] for t in results[cat]["trials"]]

    # Generate sample responses for the detail table
    response_preview = []
    for cat in categories:
        for trial in results[cat]["trials"]:
            preview = trial["response"][:200] + "..." if len(trial["response"]) > 200 else trial["response"]
            response_preview.append({
                "category": cat,
                "prompt": trial["prompt"][:80],
                "preview": preview,
                "wall_time": trial["wall_time"],
                "tokens": trial["completion_tokens"],
            })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LFM2.5-8B-A1B Routing Heatmap Analysis</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0a0a0f;
    color: #e0e0e8;
    padding: 2rem;
    line-height: 1.6;
  }}
  h1 {{
    font-size: 1.8rem;
    margin-bottom: 0.3rem;
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  .subtitle {{ color: #888; font-size: 0.9rem; margin-bottom: 2rem; }}
  h2 {{
    font-size: 1.3rem;
    margin: 2rem 0 1rem;
    color: #a78bfa;
    border-bottom: 1px solid #222;
    padding-bottom: 0.4rem;
  }}

  /* Heatmap Grid */
  .heatmap-container {{
    overflow-x: auto;
    margin-bottom: 2rem;
  }}
  .heatmap {{ border-collapse: collapse; width: 100%; min-width: 700px; }}
  .heatmap th {{
    padding: 0.7rem 1rem;
    text-align: left;
    font-weight: 600;
    color: #ccc;
    font-size: 0.85rem;
    border-bottom: 2px solid #333;
  }}
  .heatmap td {{
    padding: 0.7rem 1rem;
    text-align: center;
    font-size: 0.85rem;
    border-bottom: 1px solid #1a1a2e;
  }}
  .heatmap td:first-child {{
    text-align: left;
    font-weight: 500;
    min-width: 160px;
    color: #d4d4e8;
  }}
  .heatmap td:not(:first-child) {{
    border-radius: 4px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }}
  .heatmap tr:hover td:not(:first-child) {{
    filter: brightness(1.15);
  }}

  /* Metric interpretation legend */
  .legend {{
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    color: #aaa;
  }}
  .legend-swatch {{
    width: 16px;
    height: 16px;
    border-radius: 3px;
  }}

  /* Bar charts */
  .chart-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
  }}
  .chart-card {{
    background: #12121e;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 1rem;
  }}
  .chart-card h3 {{
    font-size: 0.9rem;
    color: #a78bfa;
    margin-bottom: 0.7rem;
  }}
  .bar-row {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
    font-size: 0.78rem;
  }}
  .bar-label {{
    width: 24px;
    text-align: right;
    color: #888;
    font-size: 0.72rem;
  }}
  .bar-track {{
    flex: 1;
    height: 18px;
    background: #1a1a2e;
    border-radius: 3px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s;
  }}
  .bar-value {{
    width: 55px;
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: #aaa;
    font-size: 0.72rem;
  }}

  /* Response preview table */
  .detail-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
  }}
  .detail-table th {{
    padding: 0.5rem;
    text-align: left;
    color: #888;
    border-bottom: 1px solid #333;
    font-weight: 500;
  }}
  .detail-table td {{
    padding: 0.5rem;
    border-bottom: 1px solid #1a1a2e;
    vertical-align: top;
  }}
  .detail-table td:first-child {{ color: #a78bfa; white-space: nowrap; }}
  .detail-table td:nth-child(2) {{ color: #ccc; max-width: 300px; }}
  .detail-table td:nth-child(3) {{ color: #aaa; font-size: 0.75rem; max-width: 450px; }}

  .hypothesis-box {{
    background: #12121e;
    border-left: 3px solid #7c3aed;
    padding: 1rem 1.2rem;
    border-radius: 0 6px 6px 0;
    margin: 1.5rem 0;
    font-size: 0.88rem;
    color: #ccc;
  }}
  .hypothesis-box strong {{ color: #a78bfa; }}
</style>
</head>
<body>

<h1>🔍 LFM2.5-8B-A1B Routing Heatmap</h1>
<p class="subtitle">Behavioral expert activation analysis · {datetime.now().strftime('%Y-%m-%d %H:%M')} · {sum(len(v) for v in CATEGORIES.values())} prompts · {len(CATEGORIES)} categories</p>

<div class="hypothesis-box">
  <strong>Hypothesis:</strong> Categories with <em>high</em> output diversity (high Levenshtein, high timing CV)
  indicate that different expert subsets fire across variations. Categories with <em>low</em> diversity
  suggest the same experts always activate for that task type — stable routing.
</div>

<h2>📊 Heatmap — Routing Signature per Category</h2>

<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#22c55e"></div> Stable (low value) = consistent expert routing</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#eab308"></div> Moderate variation</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#ef4444"></div> Diverse (high value) = varied expert activation</div>
</div>

<div class="heatmap-container">
<table class="heatmap" id="heatmapTable">
<thead>
<tr>
  <th>Category</th>
  <th>Normalized Diversity<br><span style="font-weight:400;font-size:0.72rem;color:#666">Levenshtein / mean length</span></th>
  <th>Timing CV<br><span style="font-weight:400;font-size:0.72rem;color:#666">std / mean response time</span></th>
  <th>Token Count CV<br><span style="font-weight:400;font-size:0.72rem;color:#666">std / mean output length</span></th>
  <th>Mean Levenshtein<br><span style="font-weight:400;font-size:0.72rem;color:#666">raw edit distance</span></th>
  <th>Mean Time (s)<br><span style="font-weight:400;font-size:0.72rem;color:#666">wall clock</span></th>
</tr>
</thead>
<tbody id="heatmapBody">
</tbody>
</table>
</div>

<h2>⏱ Response Time per Variation</h2>
<div class="chart-grid" id="timingCharts"></div>

<h2>📝 Sample Responses</h2>
<table class="detail-table">
<thead><tr><th>Category</th><th>Prompt</th><th>Response (truncated)</th><th>Time</th><th>Tokens</th></tr></thead>
<tbody id="responseBody"></tbody>
</table>

<h2>📐 Methodology Notes</h2>
<div class="hypothesis-box">
  <strong>Approach:</strong> We cannot directly observe which experts fire (that requires modifying llama.cpp internals).
  Instead, we use <em>behavioral routing analysis</em>: send the same base prompt type with 5 variations and measure
  output divergence. Different prompt variations may activate different expert subsets in the MoE layers
  (8 total experts, 1 active per layer in this architecture). High Levenshtein distance between responses
  to semantically similar prompts suggests that the router sends different variations to different experts.
  <br><br>
  <strong>Limitations:</strong> Output diversity is also affected by temperature, token sampling, and prompt ambiguity.
  Timing variance includes network overhead. This is a proxy signal, not a direct expert activation measurement.
  <br><br>
  <strong>Temperature:</strong> {TEMPERATURE} · <strong>Max tokens:</strong> {MAX_TOKENS} · <strong>Model:</strong> LFM2.5-8B-A1B (Q4_K_M GGUF)
</div>

<script>
// ── Data ──
const heatmapData = {json.dumps(heatmap_data)};
const timingData = {json.dumps({cat: timing_by_category[cat] for cat in categories})};
const responsePreviews = {json.dumps(response_preview)};

// ── Color interpolation: green (0) → yellow (0.5) → red (1) ──
function diversityColor(value, maxVal) {{
  const t = Math.min(value / (maxVal || 1), 1);
  let r, g, b;
  if (t < 0.5) {{
    // green → yellow
    const p = t / 0.5;
    r = Math.round(34 + p * 200);
    g = Math.round(197 + p * (-14));
    b = Math.round(94 + p * (-68));
  }} else {{
    // yellow → red
    const p = (t - 0.5) / 0.5;
    r = Math.round(234 + p * 1);
    g = Math.round(183 + p * (-147));
    b = Math.round(26 + p * (-10));
  }}
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function timingColor(value, maxVal) {{
  // Lower is more stable → green. Higher → red.
  return diversityColor(value, maxVal);
}}

// ── Heatmap ──
const maxDiv = Math.max(...heatmapData.map(d => d.normalized_diversity)) || 1;
const maxTcv = Math.max(...heatmapData.map(d => d.timing_cv)) || 1;
const maxTocv = Math.max(...heatmapData.map(d => d.token_cv)) || 1;

const tbody = document.getElementById('heatmapBody');
heatmapData.forEach(d => {{
  const row = document.createElement('tr');
  row.innerHTML = `
    <td>${{d.category}}</td>
    <td style="background:${{diversityColor(d.normalized_diversity, maxDiv)}};color:#000">
      ${{d.normalized_diversity.toFixed(4)}}
    </td>
    <td style="background:${{timingColor(d.timing_cv, maxTcv)}};color:#000">
      ${{d.timing_cv.toFixed(4)}}
    </td>
    <td style="background:${{timingColor(d.token_cv, maxTocv)}};color:#000">
      ${{d.token_cv.toFixed(4)}}
    </td>
    <td style="color:#999">${{d.mean_levenshtein.toFixed(1)}}</td>
    <td style="color:#999">${{d.timing_mean_s.toFixed(3)}}s</td>
  `;
  tbody.appendChild(row);
}});

// ── Timing bar charts ──
const chartsDiv = document.getElementById('timingCharts');
const chartColors = ['#7c3aed','#06b6d4','#f59e0b','#10b981','#ef4444','#ec4899','#3b82f6','#f97316'];
Object.entries(timingData).forEach(([cat, times], ci) => {{
  const maxT = Math.max(...times);
  const card = document.createElement('div');
  card.className = 'chart-card';
  card.innerHTML = `<h3>${{cat}}</h3>`;
  times.forEach((t, i) => {{
    const pct = (t / maxT * 100).toFixed(1);
    card.innerHTML += `
      <div class="bar-row">
        <span class="bar-label">v${{i+1}}</span>
        <div class="bar-track">
          <div class="bar-fill" style="width:${{pct}}%;background:${{chartColors[ci % chartColors.length]}}"></div>
        </div>
        <span class="bar-value">${{t.toFixed(3)}}s</span>
      </div>`;
  }});
  chartsDiv.appendChild(card);
}});

// ── Response previews ──
const rbody = document.getElementById('responseBody');
responsePreviews.forEach(r => {{
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${{r.category}}</td>
    <td>${{r.prompt}}</td>
    <td>${{r.preview}}</td>
    <td>${{r.wall_time.toFixed(2)}}s</td>
    <td>${{r.tokens}}</td>
  `;
  rbody.appendChild(tr);
}});
</script>

</body>
</html>"""

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = DOCS_DIR / "12-lfm25-8b-routing-analysis.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"HTML visualization saved to {html_path}")
    return html_path


# ── Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_experiment()
    generate_html(results)
    print("\n✅ Routing analysis complete!")
