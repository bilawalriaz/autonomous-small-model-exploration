#!/usr/bin/env python3
"""
LFM2.5-230M Qualitative SFT Evaluator
Tests trained adapters across diverse prompt categories:
factual, structured output, code, reasoning, instruction-following,
multilingual, creativity, NER, math, text transformation.
"""
import json, torch, sys, os, time, gc
from pathlib import Path
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL = "LiquidAI/LFM2.5-230M"
PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "experiments" / "results"
ADAPTERS_DIR = PROJECT / "experiments" / "adapters" / "lfm2_sft_sweep"

# ══════════════════════════════════════════════════════════════
# COMPREHENSIVE EVALUATION PROMPTS (10 categories, 50 prompts)
# ══════════════════════════════════════════════════════════════
EVAL_CATEGORIES = {
    "factual_knowledge": [
        {"prompt": "What is the capital of Australia?", "expect_contains": "Canberra"},
        {"prompt": "Who wrote 'To Kill a Mockingbird'?", "expect_contains": "Harper Lee"},
        {"prompt": "What is the chemical formula for table salt?", "expect_contains": "NaCl"},
        {"prompt": "In what year did World War II end?", "expect_contains": "1945"},
        {"prompt": "What planet is closest to the Sun?", "expect_contains": "Mercury"},
    ],
    "structured_output": [
        {"prompt": 'Return a JSON object with name, age, and city for a person named "Bob" aged 30 from London.',
         "expect_json": True, "expect_keys": ["name", "age", "city"]},
        {"prompt": 'Create a JSON array of 3 programming languages with their year of creation.',
         "expect_json": True},
        {"prompt": 'Write a YAML config for a web server on port 8080 with 4 workers.',
         "expect_contains": "8080"},
        {"prompt": 'Return a CSV with columns: fruit, color, price for 3 fruits.',
         "expect_contains": ","},
        {"prompt": 'Create a JSON object representing a book with title, author, year, and genres (array).',
         "expect_json": True, "expect_keys": ["title", "author"]},
    ],
    "code_generation": [
        {"prompt": "Write a Python function to check if a number is prime.", "expect_contains": "def "},
        {"prompt": "Write a Python function to reverse a linked list.", "expect_contains": "def "},
        {"prompt": "Write a bash script to find all .log files larger than 100MB.", "expect_contains": "find"},
        {"prompt": "Write a Python function to merge two sorted arrays.", "expect_contains": "def "},
        {"prompt": "Write a SQL query to find the top 5 customers by total orders.", "expect_contains": "SELECT"},
    ],
    "reasoning": [
        {"prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "expect_contains": "5 minutes"},
        {"prompt": "A bat and a ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?", "expect_contains": "0.05"},
        {"prompt": "If you have 3 apples and take away 2, how many do YOU have?", "expect_contains": "2"},
        {"prompt": "What comes next in the sequence: 2, 6, 12, 20, 30, ?", "expect_contains": "42"},
        {"prompt": "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly?", "expect_contains": "No"},
    ],
    "instruction_following": [
        {"prompt": "List exactly 5 European capital cities, one per line, numbered.", "expect_lines": 5},
        {"prompt": "Write a sentence that contains the words: quantum, banana, and bicycle.", "expect_contains": "quantum"},
        {"prompt": "Explain gravity in exactly 3 sentences.", "expect_sentences": 3},
        {"prompt": "Translate 'Hello, how are you?' to Japanese.", "expect_contains": None},
        {"prompt": "Write an acrostic poem using the word 'STAR'.", "expect_contains": "S"},
    ],
    "math": [
        {"prompt": "What is 847 × 23?", "expect_contains": "19481"},
        {"prompt": "Solve: 3x + 7 = 22. What is x?", "expect_contains": "5"},
        {"prompt": "What is the square root of 144?", "expect_contains": "12"},
        {"prompt": "If a rectangle has length 12 and width 5, what is its area?", "expect_contains": "60"},
        {"prompt": "What is 15% of 240?", "expect_contains": "36"},
    ],
    "multilingual": [
        {"prompt": "Translate 'The weather is beautiful today' to French.", "expect_contains": None},
        {"prompt": "Translate 'I love programming' to Spanish.", "expect_contains": None},
        {"prompt": "Translate 'Good morning' to German.", "expect_contains": None},
        {"prompt": "Translate 'Thank you very much' to Japanese.", "expect_contains": None},
        {"prompt": "Translate 'How much does this cost?' to Mandarin Chinese.", "expect_contains": None},
    ],
    "text_transformation": [
        {"prompt": "Summarize this in one sentence: The mitochondria is the powerhouse of the cell. It generates most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy.", "expect_short": True},
        {"prompt": "Rewrite this in formal English: Hey, wanna grab lunch? I'm starving.", "expect_contains": None},
        {"prompt": "Convert this sentence to passive voice: The dog chased the cat.", "expect_contains": "cat"},
        {"prompt": "Expand this into a paragraph: AI is transforming healthcare.", "expect_longer": True},
        {"prompt": "Rewrite this as a haiku: The snow falls gently on the mountain.", "expect_short": True},
    ],
    "ner_and_extraction": [
        {"prompt": "Extract all person names from: 'Barack Obama met Angela Merkel in Berlin to discuss NATO.'",
         "expect_contains": "Obama"},
        {"prompt": "Extract dates from: 'The company was founded on March 15, 2004 and went public on June 8, 2012.'",
         "expect_contains": "2004"},
        {"prompt": "Extract all email addresses from: 'Contact john@example.com or support@company.org for help.'",
         "expect_contains": "@example.com"},
        {"prompt": "Identify the programming language: 'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)'",
         "expect_contains": "Python"},
        {"prompt": "Extract the main topic from: 'Recent advances in quantum computing have led to breakthroughs in cryptography and drug discovery.'",
         "expect_contains": "quantum"},
    ],
    "creativity": [
        {"prompt": "Write a one-sentence story about a robot learning to dream.", "expect_contains": None},
        {"prompt": "Come up with 3 creative names for a coffee shop that also sells books.", "expect_lines_min": 3},
        {"prompt": "Write a limerick about a programmer who lost their code.", "expect_contains": None},
        {"prompt": "Describe a color that doesn't exist to someone who can see.", "expect_longer": True},
        {"prompt": "Write a product tagline for an AI-powered umbrella.", "expect_contains": None},
    ],
}


def load_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    return model, tok


def generate_text(model, tok, prompt, max_new_tokens=256):
    model.eval()
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id
        )
    dt = time.time() - t0
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    n_tok = out.shape[1] - ids.shape[1]
    return gen, n_tok, round(dt, 2)


def score_response(response, spec):
    """Score a response against expected criteria. Returns (score, notes)."""
    notes = []
    score = 0  # 0-3 scale

    if not response or len(response.strip()) < 2:
        return 0, "Empty or degenerate response"

    # Basic quality: non-empty, reasonable length
    score = 1
    words = response.split()

    if spec.get("expect_contains"):
        target = spec["expect_contains"]
        if target.lower() in response.lower():
            score = 3
            notes.append(f"Contains '{target}'")
        else:
            notes.append(f"Missing '{target}'")

    if spec.get("expect_json"):
        # Try to parse JSON from response
        try:
            # Find JSON in response
            start = response.find("{")
            start_arr = response.find("[")
            if start_arr >= 0 and (start < 0 or start_arr < start):
                start = start_arr
            if start >= 0:
                parsed = json.loads(response[start:])
                score = 3
                notes.append("Valid JSON")
                if spec.get("expect_keys"):
                    missing = [k for k in spec["expect_keys"] if k not in parsed]
                    if missing:
                        score = 2
                        notes.append(f"Missing keys: {missing}")
        except json.JSONDecodeError:
            notes.append("Invalid JSON")
            score = 1

    if spec.get("expect_lines"):
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        if len(lines) >= spec["expect_lines"]:
            score = 3
            notes.append(f"{len(lines)} lines (expected {spec['expect_lines']})")
        else:
            score = 2
            notes.append(f"Only {len(lines)} lines (expected {spec['expect_lines']})")

    if spec.get("expect_lines_min"):
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        if len(lines) >= spec["expect_lines_min"]:
            score = 3
        else:
            score = 1

    if spec.get("expect_short"):
        if len(words) < 50:
            score = 3
            notes.append("Appropriate length")
        else:
            score = 2
            notes.append("Too long")

    if spec.get("expect_longer"):
        if len(words) > 30:
            score = 3
        else:
            score = 1
            notes.append("Too short")

    if spec.get("expect_sentences"):
        sents = [s.strip() for s in response.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        if abs(len(sents) - spec["expect_sentences"]) <= 1:
            score = 3
        else:
            score = 2
            notes.append(f"~{len(sents)} sentences (expected {spec['expect_sentences']})")

    if score == 0:
        score = 1  # at least it responded

    return score, "; ".join(notes) if notes else "OK"


def run_qualitative_eval(adapter_path=None, adapter_name="base"):
    """Run comprehensive qualitative eval. If adapter_path is None, eval base model."""
    from peft import PeftModel

    print(f"\n{'='*60}")
    print(f"QUALITATIVE EVALUATION: {adapter_name}")
    print(f"{'='*60}")

    model, tok = load_model_and_tokenizer()

    if adapter_path:
        print(f"  Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    results = {}
    total_score = 0
    total_tests = 0

    for category, tests in EVAL_CATEGORIES.items():
        print(f"\n  [{category}]")
        cat_results = []
        cat_score = 0

        for test in tests:
            gen, n_tok, dt = generate_text(model, tok, test["prompt"])
            score, notes = score_response(gen, test)

            status = "✓" if score >= 3 else "~" if score >= 2 else "✗"
            print(f"    {status} [{score}/3] {test['prompt'][:50]}... → {gen[:80]}...")
            if notes:
                print(f"         {notes}")

            cat_results.append({
                "prompt": test["prompt"],
                "response": gen[:500],  # truncate for storage
                "score": score,
                "notes": notes,
                "n_tokens": n_tok,
                "time_s": dt,
            })
            cat_score += score
            total_score += score
            total_tests += 1

        avg = cat_score / len(tests) if tests else 0
        print(f"    Category avg: {avg:.1f}/3.0")
        results[category] = {"tests": cat_results, "avg_score": round(avg, 2)}

    overall = total_score / total_tests if total_tests else 0
    print(f"\n  OVERALL SCORE: {overall:.2f}/3.0 ({total_tests} tests)")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "adapter": adapter_name,
        "overall_score": round(overall, 2),
        "total_tests": total_tests,
        "categories": results,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None,
                       help="Path to adapter dir. Omit for base model eval.")
    parser.add_argument("--name", type=str, default="base",
                       help="Name for this eval run")
    parser.add_argument("--all-adapters", action="store_true",
                       help="Eval all adapters in the sweep directory")
    parser.add_argument("--categories", type=str, default=None,
                       help="Comma-separated categories to eval (default: all)")
    args = parser.parse_args()

    if args.categories:
        cats = [c.strip() for c in args.categories.split(",")]
        global EVAL_CATEGORIES
        EVAL_CATEGORIES = {k: v for k, v in EVAL_CATEGORIES.items() if k in cats}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.all_adapters:
        # Eval base + all adapters
        all_results = []

        # Base model
        print("\n### BASE MODEL ###")
        base_result = run_qualitative_eval(None, "base")
        all_results.append(base_result)

        # All adapters
        if ADAPTERS_DIR.exists():
            for adapter_dir in sorted(ADAPTERS_DIR.iterdir()):
                if adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").exists():
                    print(f"\n### ADAPTER: {adapter_dir.name} ###")
                    result = run_qualitative_eval(str(adapter_dir), adapter_dir.name)
                    all_results.append(result)

        # Save combined
        out_file = RESULTS_DIR / f"lfm2_qualitative_comparison_{timestamp}.json"
        with open(out_file, "w") as f:
            json.dump({"model": MODEL, "timestamp": timestamp, "results": all_results}, f, indent=2)

        # Print comparison table
        print(f"\n{'='*70}")
        print(f"{'Adapter':<30} {'Overall':>8} {'Factual':>8} {'JSON':>8} {'Code':>8} {'Reason':>8}")
        print("-" * 70)
        for r in all_results:
            cats = r.get("categories", {})
            print(f"{r['adapter']:<30} {r['overall_score']:>8.2f} "
                  f"{cats.get('factual_knowledge', {}).get('avg_score', 0):>8.2f} "
                  f"{cats.get('structured_output', {}).get('avg_score', 0):>8.2f} "
                  f"{cats.get('code_generation', {}).get('avg_score', 0):>8.2f} "
                  f"{cats.get('reasoning', {}).get('avg_score', 0):>8.2f}")

        print(f"\nSaved: {out_file}")

    else:
        result = run_qualitative_eval(args.adapter, args.name)
        out_file = RESULTS_DIR / f"lfm2_qualitative_{args.name}_{timestamp}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
