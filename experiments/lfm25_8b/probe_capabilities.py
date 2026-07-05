#!/usr/bin/env python3
"""
Capabilities Benchmark: Comprehensive capability evaluation of LFM2.5-8B
across factual QA, code generation, math, multilingual, and instruction following.

Each task is scored (correct/partial/wrong) and results are saved as JSON
for comparison across different model configurations and quantizations.
"""

import json
import time
import re
import requests
import sys
from datetime import datetime
from pathlib import Path

BASE_URL = "http://localhost:8080"
ENDPOINT = f"{BASE_URL}/v1/chat/completions"

# ──────────────────────────────────────────────────────────────
# TASKS
# ──────────────────────────────────────────────────────────────

FACTUAL_QA = [
    {"id": "f01", "q": "What is the capital of France?", "a": "Paris", "category": "geography"},
    {"id": "f02", "q": "What planet is closest to the Sun?", "a": "Mercury", "category": "science"},
    {"id": "f03", "q": "Who wrote '1984'?", "a": "George Orwell", "category": "literature"},
    {"id": "f04", "q": "What is the chemical symbol for gold?", "a": "Au", "category": "chemistry"},
    {"id": "f05", "q": "How many bones are in the adult human body?", "a": "206", "category": "biology"},
    {"id": "f06", "q": "What year did World War II end?", "a": "1945", "category": "history"},
    {"id": "f07", "q": "What is the speed of light in m/s?", "a": "299792458", "category": "physics"},
    {"id": "f08", "q": "What is the largest ocean on Earth?", "a": "Pacific", "category": "geography"},
    {"id": "f09", "q": "Who painted the Mona Lisa?", "a": "Leonardo da Vinci", "category": "art"},
    {"id": "f10", "q": "What is the square root of 144?", "a": "12", "category": "math"},
    {"id": "f11", "q": "What is the currency of Japan?", "a": "Yen", "category": "geography"},
    {"id": "f12", "q": "What gas do plants absorb from the atmosphere?", "a": "carbon dioxide", "category": "science"},
    {"id": "f13", "q": "Who invented the telephone?", "a": "Alexander Graham Bell", "category": "invention"},
    {"id": "f14", "q": "What is the boiling point of water at sea level in Celsius?", "a": "100", "category": "science"},
    {"id": "f15", "q": "What is the smallest prime number?", "a": "2", "category": "math"},
    {"id": "f16", "q": "How many continents are there?", "a": "7", "category": "geography"},
    {"id": "f17", "q": "What element has atomic number 1?", "a": "Hydrogen", "category": "chemistry"},
    {"id": "f18", "q": "What is the longest river in the world?", "a": "Nile", "category": "geography"},
    {"id": "f19", "q": "Who developed the theory of relativity?", "a": "Albert Einstein", "category": "physics"},
    {"id": "f20", "q": "What is the hardest natural substance on Earth?", "a": "Diamond", "category": "science"},
    {"id": "f21", "q": "What is the capital of Australia?", "a": "Canberra", "category": "geography"},
    {"id": "f22", "q": "How many sides does a hexagon have?", "a": "6", "category": "math"},
    {"id": "f23", "q": "What year was the Declaration of Independence signed?", "a": "1776", "category": "history"},
    {"id": "f24", "q": "What is the powerhouse of the cell?", "a": "Mitochondria", "category": "biology"},
    {"id": "f25", "q": "What is the freezing point of water in Fahrenheit?", "a": "32", "category": "science"},
    {"id": "f26", "q": "Who was the first person to walk on the Moon?", "a": "Neil Armstrong", "category": "history"},
    {"id": "f27", "q": "What is the chemical formula for table salt?", "a": "NaCl", "category": "chemistry"},
    {"id": "f28", "q": "How many degrees are in a circle?", "a": "360", "category": "math"},
    {"id": "f29", "q": "What is the largest planet in our solar system?", "a": "Jupiter", "category": "science"},
    {"id": "f30", "q": "What is the main language spoken in Brazil?", "a": "Portuguese", "category": "geography"},
    {"id": "f31", "q": "What force keeps us on the ground?", "a": "Gravity", "category": "physics"},
    {"id": "f32", "q": "How many players are on a soccer team?", "a": "11", "category": "sports"},
    {"id": "f33", "q": "What is the capital of Canada?", "a": "Ottawa", "category": "geography"},
    {"id": "f34", "q": "What is the pH of pure water?", "a": "7", "category": "chemistry"},
    {"id": "f35", "q": "Who wrote 'Romeo and Juliet'?", "a": "Shakespeare", "category": "literature"},
    {"id": "f36", "q": "What is the derivative of x²?", "a": "2x", "category": "math"},
    {"id": "f37", "q": "How many chromosomes do humans have?", "a": "46", "category": "biology"},
    {"id": "f38", "q": "What is the capital of Germany?", "a": "Berlin", "category": "geography"},
    {"id": "f39", "q": "What year was the internet invented?", "a": "1969", "category": "history"},
    {"id": "f40", "q": "What is the nearest star to Earth?", "a": "Sun", "category": "astronomy"},
    {"id": "f41", "q": "What type of rock is formed by volcanic activity?", "a": "Igneous", "category": "geology"},
    {"id": "f42", "q": "What is the SI unit of force?", "a": "Newton", "category": "physics"},
    {"id": "f43", "q": "How many zeros are in one million?", "a": "6", "category": "math"},
    {"id": "f44", "q": "What is the main gas in Earth's atmosphere?", "a": "Nitrogen", "category": "science"},
    {"id": "f45", "q": "What continent is Egypt in?", "a": "Africa", "category": "geography"},
    {"id": "f46", "q": "What is the square root of 256?", "a": "16", "category": "math"},
    {"id": "f47", "q": "What is the most abundant element in the universe?", "a": "Hydrogen", "category": "chemistry"},
    {"id": "f48", "q": "How many planets are in our solar system?", "a": "8", "category": "astronomy"},
    {"id": "f49", "q": "What is the capital of Italy?", "a": "Rome", "category": "geography"},
    {"id": "f50", "q": "What does DNA stand for?", "a": "Deoxyribonucleic acid", "category": "biology"},
]

CODE_TASKS = [
    {
        "id": "c01",
        "q": "Write a Python function that checks if a number is prime.",
        "check": lambda r: "def " in r and "return" in r,
        "category": "code_gen",
    },
    {
        "id": "c02",
        "q": "Write a Python function to reverse a string.",
        "check": lambda r: "def " in r and ("[::-1]" in r or "reverse" in r.lower()),
        "category": "code_gen",
    },
    {
        "id": "c03",
        "q": "Write a Python function to find the factorial of a number.",
        "check": lambda r: "def " in r and ("factorial" in r.lower() or "return" in r),
        "category": "code_gen",
    },
    {
        "id": "c04",
        "q": "Write a Python function to check if a string is a palindrome.",
        "check": lambda r: "def " in r and ("palindrome" in r.lower() or "[::-1]" in r),
        "category": "code_gen",
    },
    {
        "id": "c05",
        "q": "What is the output of this code?\n```python\nprint(2 ** 10)\n```",
        "check": lambda r: "1024" in r,
        "category": "code_debug",
    },
    {
        "id": "c06",
        "q": "Find the bug: def add(a, b): return a - b",
        "check": lambda r: "subtraction" in r.lower() or "minus" in r.lower() or "-" in r or "add" in r.lower(),
        "category": "code_debug",
    },
    {
        "id": "c07",
        "q": "Write a Python function to find the largest element in a list.",
        "check": lambda r: "def " in r and ("max" in r.lower() or "largest" in r.lower()),
        "category": "code_gen",
    },
    {
        "id": "c08",
        "q": "Write a Python function that counts vowels in a string.",
        "check": lambda r: "def " in r and ("vowel" in r.lower()),
        "category": "code_gen",
    },
    {
        "id": "c09",
        "q": "What does this code output?\n```python\nx = [1,2,3]\nprint(x[1])\n```",
        "check": lambda r: "2" in r,
        "category": "code_debug",
    },
    {
        "id": "c10",
        "q": "Write a Python function to check if a number is even.",
        "check": lambda r: "def " in r and ("%" in r or "even" in r.lower()),
        "category": "code_gen",
    },
    {
        "id": "c11",
        "q": "Write a Python function that returns the Fibonacci sequence up to n.",
        "check": lambda r: "def " in r and ("fibonacci" in r.lower()),
        "category": "code_gen",
    },
    {
        "id": "c12",
        "q": "What is the time complexity of binary search? Answer in O() notation.",
        "check": lambda r: "O(log n)" in r or "O(log" in r,
        "category": "cs_knowledge",
    },
    {
        "id": "c13",
        "q": "Write a Python function to merge two sorted lists.",
        "check": lambda r: "def " in r and ("merge" in r.lower() or "sorted" in r.lower()),
        "category": "code_gen",
    },
    {
        "id": "c14",
        "q": "What does this SQL do? SELECT COUNT(*) FROM users WHERE age > 18;",
        "check": lambda r: "count" in r.lower() and ("18" in r or "age" in r.lower()),
        "category": "code_debug",
    },
    {
        "id": "c15",
        "q": "Write a Python class for a stack with push and pop methods.",
        "check": lambda r: "class " in r and ("push" in r.lower() or "pop" in r.lower()),
        "category": "code_gen",
    },
    {
        "id": "c16",
        "q": "Write a Python decorator that logs function execution time.",
        "check": lambda r: "def " in r and ("decorator" in r.lower() or "@" in r or "time" in r.lower()),
        "category": "code_gen",
    },
    {
        "id": "c17",
        "q": "Write a Python function to flatten a nested list.",
        "check": lambda r: "def " in r and ("flatten" in r.lower() or "nested" in r.lower()),
        "category": "code_gen",
    },
    {
        "id": "c18",
        "q": "What is a hash table? Give a one-sentence answer.",
        "check": lambda r: len(r) < 200 and ("hash" in r.lower() or "key" in r.lower()),
        "category": "cs_knowledge",
    },
    {
        "id": "c19",
        "q": "Write a Python function to find duplicates in a list.",
        "check": lambda r: "def " in r and ("duplicate" in r.lower()),
        "category": "code_gen",
    },
    {
        "id": "c20",
        "q": "Write a Python one-liner to get the even numbers from a list.",
        "check": lambda r: "def " in r or "[x" in r or "filter" in r or "%" in r,
        "category": "code_gen",
    },
]

MATH_TASKS = [
    {"id": "m01", "q": "What is 127 + 384?", "a": "511", "category": "arithmetic"},
    {"id": "m02", "q": "What is 15 × 23?", "a": "345", "category": "arithmetic"},
    {"id": "m03", "q": "What is 1000 / 8?", "a": "125", "category": "arithmetic"},
    {"id": "m04", "q": "What is 2^10?", "a": "1024", "category": "arithmetic"},
    {"id": "m05", "q": "What is the derivative of x³ + 2x?", "a": "3x² + 2", "category": "calculus"},
    {"id": "m06", "q": "Solve for x: 2x + 5 = 13", "a": "4", "category": "algebra"},
    {"id": "m07", "q": "What is the area of a circle with radius 5? Use π = 3.14", "a": "78.5", "category": "geometry"},
    {"id": "m08", "q": "What is gcd(12, 18)?", "a": "6", "category": "number_theory"},
    {"id": "m09", "q": "Convert 1011 binary to decimal.", "a": "11", "category": "conversion"},
    {"id": "m10", "q": "What is the sum of interior angles of a pentagon?", "a": "540", "category": "geometry"},
]

MULTILINGUAL_TASKS = [
    {"id": "ml01", "q": "Translate to French: 'The cat is on the table.'", "check": lambda r: "chat" in r.lower() or "table" in r.lower(), "lang": "french"},
    {"id": "ml02", "q": "Translate to Spanish: 'I love programming.'", "check": lambda r: "programar" in r.lower() or "amor" in r.lower() or "coding" in r.lower(), "lang": "spanish"},
    {"id": "ml03", "q": "What does 'konnichiwa' mean in English?", "check": lambda r: "hello" in r.lower() or "good afternoon" in r.lower(), "lang": "japanese"},
    {"id": "ml04", "q": "Translate to German: 'Good morning, how are you?'", "check": lambda r: "guten" in r.lower() or "morgen" in r.lower(), "lang": "german"},
    {"id": "ml05", "q": "What is 'hello' in Mandarin Chinese?", "check": lambda r: "nǐ hǎo" in r.lower() or "ni hao" in r.lower() or "你好" in r, "lang": "chinese"},
    {"id": "ml06", "q": "Translate to Portuguese: 'The weather is beautiful today.'", "check": lambda r: "tempo" in r.lower() or "bonito" in r.lower() or "lindo" in r.lower(), "lang": "portuguese"},
    {"id": "ml07", "q": "What does 'bon voyage' mean?", "check": lambda r: "good journey" in r.lower() or "safe trip" in r.lower() or "travel" in r.lower(), "lang": "french"},
    {"id": "ml08", "q": "Translate to Italian: 'I want coffee.'", "check": lambda r: "caffè" in r.lower() or "cafe" in r.lower() or "voglio" in r.lower(), "lang": "italian"},
    {"id": "ml09", "q": "What is 'thank you' in Korean?", "check": lambda r: "kamsahamnida" in r.lower() or "감사합니다" in r, "lang": "korean"},
    {"id": "ml10", "q": "Translate to Arabic: 'Peace be upon you.'", "check": lambda r: "salam" in r.lower() or "peace" in r.lower(), "lang": "arabic"},
]

INSTRUCTION_TASKS = [
    {
        "id": "i01",
        "q": "List exactly 5 fruits, numbered 1-5.",
        "check": lambda r: all(str(i) in r for i in range(1, 6)) and len(r) < 500,
        "category": "format",
    },
    {
        "id": "i02",
        "q": "Answer with ONLY 'yes' or 'no': Is the Earth round?",
        "check": lambda r: r.strip().lower() in ("yes", "no"),
        "category": "constraint",
    },
    {
        "id": "i03",
        "q": "Write a haiku about artificial intelligence (5-7-5 syllables).",
        "check": lambda r: len(r) < 300 and "\n" in r,
        "category": "format",
    },
    {
        "id": "i04",
        "q": "Respond to everything in exactly one sentence.",
        "check": lambda r: r.count(".") <= 2 and len(r) < 300,
        "category": "constraint",
    },
    {
        "id": "i05",
        "q": "Count from 1 to 10, separated by commas, nothing else.",
        "check": lambda r: all(str(i) in r for i in range(1, 11)) and len(r) < 100,
        "category": "format",
    },
]


def check_server():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def query(prompt, temperature=0.1, max_tokens=256):
    payload = {
        "model": "lfm25-8b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    start = time.perf_counter()
    try:
        r = requests.post(ENDPOINT, json=payload, timeout=120)
        elapsed = time.perf_counter() - start
        data = r.json()
        usage = data.get("usage", {})
        content = data["choices"][0]["message"]["content"]
        return {
            "ok": True,
            "response": content,
            "completion_tokens": usage.get("completion_tokens", 0),
            "wall_time_s": round(elapsed, 4),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def score_factual(answer, expected):
    """Score factual answer: correct if expected text appears in response."""
    a = answer.lower().strip()
    e = expected.lower().strip()
    if e in a:
        return "correct"
    # Check for numeric equivalence
    try:
        if float(e) in [float(x) for x in re.findall(r"-?\d+\.?\d*", a)]:
            return "correct"
    except (ValueError, TypeError):
        pass
    # Partial: first word matches
    words = e.split()
    if words and words[0] in a:
        return "partial"
    return "wrong"


def score_math(answer, expected):
    """Score math answer."""
    a = answer.lower().strip()
    e = expected.lower().strip()
    # Extract numbers from response
    nums = re.findall(r"-?\d+\.?\d*", a)
    if e in nums:
        return "correct"
    # Check if expected number appears
    try:
        if float(e) in [float(x) for x in nums]:
            return "correct"
    except (ValueError, TypeError):
        pass
    if e in a:
        return "correct"
    return "wrong"


def run_benchmark():
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("LFM2.5-8B Capabilities Benchmark")
    print("=" * 70)

    if not check_server():
        print("\n❌ Server not running at", BASE_URL)
        print("Start it with:")
        print("  /home/billz/llama.cpp/build/bin/llama-server -m /home/billz/LFM2.5-8B-A1B-Uncensored-Gaston-Q4_K_M.gguf --host 0.0.0.0 --port 8080 --ctx-size 131072 --cache-type-k q4_0 --cache-type-v q4_0 --no-kv-offload --flash-attn auto --mlock --cont-batching --parallel 1 -ngl 99")
        sys.exit(1)

    print(f"\n✅ Server running at {BASE_URL}")
    print(f"Tasks: {len(FACTUAL_QA)} factual + {len(CODE_TASKS)} code + {len(MATH_TASKS)} math + {len(MULTILINGUAL_TASKS)} multilingual + {len(INSTRUCTION_TASKS)} instruction")
    print(f"Total: {len(FACTUAL_QA) + len(CODE_TASKS) + len(MATH_TASKS) + len(MULTILINGUAL_TASKS) + len(INSTRUCTION_TASKS)} tasks\n")

    all_results = []
    scores = {"correct": 0, "partial": 0, "wrong": 0, "error": 0}

    # Factual QA
    print("📚 Factual QA")
    print("-" * 60)
    for t in FACTUAL_QA:
        print(f"  [{t['id']}] {t['q'][:50]}...", end=" ", flush=True)
        r = query(t["q"])
        if r["ok"]:
            score = score_factual(r["response"], t["a"])
            scores[score] += 1
            symbol = {"correct": "✅", "partial": "🟡", "wrong": "❌"}[score]
            print(f"{symbol} → {r['response'][:60].replace(chr(10), ' ')}")
            all_results.append({"task": t, "score": score, "result": r})
        else:
            scores["error"] += 1
            print(f"💥 {r['error']}")
            all_results.append({"task": t, "score": "error", "result": r})

    # Code
    print("\n💻 Code Tasks")
    print("-" * 60)
    for t in CODE_TASKS:
        print(f"  [{t['id']}] {t['q'][:50]}...", end=" ", flush=True)
        r = query(t["q"], max_tokens=512)
        if r["ok"]:
            score = "correct" if t["check"](r["response"]) else "wrong"
            scores[score] += 1
            symbol = {"correct": "✅", "wrong": "❌"}[score]
            print(f"{symbol} ({r['completion_tokens']}t)")
            all_results.append({"task": t, "score": score, "result": r})
        else:
            scores["error"] += 1
            print(f"💥 {r['error']}")
            all_results.append({"task": t, "score": "error", "result": r})

    # Math
    print("\n🔢 Math Tasks")
    print("-" * 60)
    for t in MATH_TASKS:
        print(f"  [{t['id']}] {t['q'][:50]}...", end=" ", flush=True)
        r = query(t["q"])
        if r["ok"]:
            score = score_math(r["response"], t["a"])
            scores[score] += 1
            symbol = {"correct": "✅", "partial": "🟡", "wrong": "❌"}[score]
            print(f"{symbol} → {r['response'][:60].replace(chr(10), ' ')}")
            all_results.append({"task": t, "score": score, "result": r})
        else:
            scores["error"] += 1
            print(f"💥 {r['error']}")
            all_results.append({"task": t, "score": "error", "result": r})

    # Multilingual
    print("\n🌍 Multilingual Tasks")
    print("-" * 60)
    for t in MULTILINGUAL_TASKS:
        print(f"  [{t['id']}] {t['q'][:50]}...", end=" ", flush=True)
        r = query(t["q"])
        if r["ok"]:
            score = "correct" if t["check"](r["response"]) else "wrong"
            scores[score] += 1
            symbol = {"correct": "✅", "wrong": "❌"}[score]
            print(f"{symbol} → {r['response'][:60].replace(chr(10), ' ')}")
            all_results.append({"task": t, "score": score, "result": r})
        else:
            scores["error"] += 1
            print(f"💥 {r['error']}")
            all_results.append({"task": t, "score": "error", "result": r})

    # Instruction following
    print("\n📋 Instruction Following")
    print("-" * 60)
    for t in INSTRUCTION_TASKS:
        print(f"  [{t['id']}] {t['q'][:50]}...", end=" ", flush=True)
        r = query(t["q"])
        if r["ok"]:
            score = "correct" if t["check"](r["response"]) else "wrong"
            scores[score] += 1
            symbol = {"correct": "✅", "wrong": "❌"}[score]
            print(f"{symbol} → {r['response'][:60].replace(chr(10), ' ')}")
            all_results.append({"task": t, "score": score, "result": r})
        else:
            scores["error"] += 1
            print(f"💥 {r['error']}")
            all_results.append({"task": t, "score": "error", "result": r})

    # Save
    total = scores["correct"] + scores["partial"] + scores["wrong"] + scores["error"]
    out_file = output_dir / "capabilities_benchmark_results.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": "LFM2.5-8B-A1B-Uncensored-Gaston-Q4_K_M",
        "total_tasks": total,
        "scores": scores,
        "accuracy": round((scores["correct"] + scores["partial"] * 0.5) / max(total, 1) * 100, 1),
    }
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "results": all_results}, f, indent=2, default=str)

    # Final report
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  ✅ Correct:  {scores['correct']:>3} ({scores['correct']/max(total,1)*100:.1f}%)")
    print(f"  🟡 Partial:  {scores['partial']:>3} ({scores['partial']/max(total,1)*100:.1f}%)")
    print(f"  ❌ Wrong:    {scores['wrong']:>3} ({scores['wrong']/max(total,1)*100:.1f}%)")
    print(f"  💥 Error:    {scores['error']:>3}")
    print(f"  📊 Accuracy: {summary['accuracy']}%")
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    run_benchmark()
