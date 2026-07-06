#!/usr/bin/env python3
"""Rerun capability benchmark with proper max_tokens for reasoning models."""
import requests, json, time

SERVER = "http://localhost:8080/v1/chat/completions"
MODEL = "lfm2.5-8b-a1b"
MAX_TOKENS = 4096  # Reasoning models need 2-8x more tokens for thinking + output

TASKS = [
    # Factual (15 tasks)
    {"id": "f01", "q": "What is the capital of France?", "a": "Paris", "cat": "factual"},
    {"id": "f02", "q": "What planet is closest to the Sun?", "a": "Mercury", "cat": "factual"},
    {"id": "f03", "q": "Who wrote '1984'?", "a": "George Orwell", "cat": "factual"},
    {"id": "f04", "q": "What is the chemical symbol for gold?", "a": "Au", "cat": "factual"},
    {"id": "f05", "q": "How many bones are in the adult human body?", "a": "206", "cat": "factual"},
    {"id": "f06", "q": "What year did World War II end?", "a": "1945", "cat": "factual"},
    {"id": "f07", "q": "What is the speed of light in m/s?", "a": "299792458", "cat": "factual"},
    {"id": "f08", "q": "What is the largest ocean on Earth?", "a": "Pacific", "cat": "factual"},
    {"id": "f09", "q": "Who painted the Mona Lisa?", "a": "Leonardo da Vinci", "cat": "factual"},
    {"id": "f10", "q": "What is the square root of 144?", "a": "12", "cat": "factual"},
    {"id": "f11", "q": "What is the currency of Japan?", "a": "yen", "cat": "factual"},
    {"id": "f12", "q": "What gas do plants absorb from the atmosphere?", "a": "carbon dioxide", "cat": "factual"},
    {"id": "f13", "q": "Who invented the telephone?", "a": "Alexander Graham Bell", "cat": "factual"},
    {"id": "f14", "q": "What is the boiling point of water in Celsius?", "a": "100", "cat": "factual"},
    {"id": "f15", "q": "How many continents are there?", "a": "7", "cat": "factual"},
    # Code gen (14 tasks)
    {"id": "c01", "q": "Write a Python function that checks if a number is prime.", "a": "def is_prime", "cat": "code_gen", "check": "has_def"},
    {"id": "c02", "q": "Write a Python function to reverse a string.", "a": "def reverse", "cat": "code_gen", "check": "has_def"},
    {"id": "c03", "q": "Write a Python function to find the factorial of a number.", "a": "def factorial", "cat": "code_gen", "check": "has_def"},
    {"id": "c04", "q": "Write a Python function to check if a string is a palindrome.", "a": "def is_palindrome", "cat": "code_gen", "check": "has_def"},
    {"id": "c05", "q": "What is the output of this code?\n```python\nprint(2 ** 10)\n```", "a": "1024", "cat": "code_debug"},
    {"id": "c06", "q": "Find the bug: def add(a, b): return a - b", "a": "should be a + b", "cat": "code_debug"},
    {"id": "c07", "q": "Write a Python function to find the largest element in a list.", "a": "def max_element", "cat": "code_gen", "check": "has_def"},
    {"id": "c08", "q": "Write a Python function that counts vowels in a string.", "a": "def count_vowels", "cat": "code_gen", "check": "has_def"},
    {"id": "c09", "q": "What does this code output?\n```python\nx = [1,2,3]\nprint(x[1])\n```", "a": "2", "cat": "code_debug"},
    {"id": "c10", "q": "Write a Python function to check if a number is even.", "a": "def is_even", "cat": "code_gen", "check": "has_def"},
    {"id": "c11", "q": "Write a Python function that returns the Fibonacci sequence up to n.", "a": "def fibonacci", "cat": "code_gen", "check": "has_def"},
    {"id": "c12", "q": "What is the time complexity of binary search? Answer with O() notation only.", "a": "O(log n)", "cat": "code_debug"},
    {"id": "c13", "q": "Write a Python function to merge two sorted lists.", "a": "def merge", "cat": "code_gen", "check": "has_def"},
    {"id": "c14", "q": "What does this SQL do? SELECT COUNT(*) FROM users WHERE age > 18;", "a": "count", "cat": "code_debug"},
    # Math (10 tasks)
    {"id": "m01", "q": "What is 15 * 17?", "a": "255", "cat": "math"},
    {"id": "m02", "q": "Solve for x: 2x + 5 = 13", "a": "4", "cat": "math"},
    {"id": "m03", "q": "What is the derivative of x^3?", "a": "3x^2", "cat": "math"},
    {"id": "m04", "q": "What is the area of a circle with radius 5?", "a": "78.5", "cat": "math"},
    {"id": "m05", "q": "What is the GCD of 12 and 18?", "a": "6", "cat": "math"},
    {"id": "m06", "q": "Convert 1011 binary to decimal.", "a": "11", "cat": "math"},
    {"id": "m07", "q": "What is 2^10?", "a": "1024", "cat": "math"},
    {"id": "m08", "q": "What is the sum of interior angles of a pentagon?", "a": "540", "cat": "math"},
    {"id": "m09", "q": "What is sqrt(144)?", "a": "12", "cat": "math"},
    {"id": "m10", "q": "What is 144 / 12?", "a": "12", "cat": "math"},
    # Multilingual (10 tasks)
    {"id": "ml01", "q": "Translate to French: 'The cat is on the table.'", "a": "chat", "cat": "multilingual"},
    {"id": "ml02", "q": "Translate to Spanish: 'I love programming.'", "a": "programación", "cat": "multilingual"},
    {"id": "ml03", "q": "What does 'konnichiwa' mean in English?", "a": "hello", "cat": "multilingual"},
    {"id": "ml04", "q": "Translate to German: 'Good morning, how are you?'", "a": "Guten Morgen", "cat": "multilingual"},
    {"id": "ml05", "q": "What is 'hello' in Mandarin Chinese?", "a": "nǐ hǎo", "cat": "multilingual"},
    {"id": "ml06", "q": "Translate to Portuguese: 'The weather is beautiful today.'", "a": "tempo", "cat": "multilingual"},
    {"id": "ml07", "q": "What does 'bon voyage' mean?", "a": "good journey", "cat": "multilingual"},
    {"id": "ml08", "q": "Translate to Italian: 'I want coffee.'", "a": "caffè", "cat": "multilingual"},
    {"id": "ml09", "q": "What is 'thank you' in Korean?", "a": "감사합니다", "cat": "multilingual"},
    {"id": "ml10", "q": "Translate to Arabic: 'Peace be upon you.'", "a": "السلام", "cat": "multilingual"},
    # Instruction following (5 tasks)
    {"id": "i01", "q": "List exactly 5 fruits, numbered 1-5.", "a": "5", "cat": "instruction"},
    {"id": "i02", "q": "Answer with ONLY 'yes' or 'no': Is the Earth round?", "a": "yes", "cat": "instruction"},
    {"id": "i03", "q": "Write a haiku about artificial intelligence (5-7-5 syllables).", "a": "haiku", "cat": "instruction"},
    {"id": "i04", "q": "Respond to everything in exactly one sentence.", "a": "sentence", "cat": "instruction"},
    {"id": "i05", "q": "Count from 1 to 10, separated by commas, nothing else.", "a": "1,2,3,4,5,6,7,8,9,10", "cat": "instruction"},
]

def score(task, response):
    resp = response.lower().strip()
    a = task["a"].lower()
    if task.get("check") == "has_def":
        return "correct" if "def " in resp or "function" in resp else "wrong"
    if task["cat"] == "math":
        # Check if the answer number appears anywhere
        return "correct" if a in resp else "wrong"
    if task["cat"] == "multilingual":
        return "correct" if a in resp else "wrong"
    if task["cat"] == "instruction":
        if task["id"] == "i01":
            return "correct" if "5" in resp and ("apple" in resp or "banana" in resp or "cherry" in resp or "1." in resp) else "wrong"
        if task["id"] == "i02":
            return "correct" if resp.startswith("yes") or resp.startswith("no") else "wrong"
        if task["id"] == "i05":
            return "correct" if "1,2,3,4,5,6,7,8,9,10" in resp.replace(" ", "") else "wrong"
        return "correct" if len(resp) > 10 else "wrong"
    # Factual/code_debug
    return "correct" if a in resp else "wrong"

results = []
cats = {}

for t in TASKS:
    t0 = time.time()
    try:
        resp = requests.post(SERVER, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": t["q"]}],
            "temperature": 0.2,
            "max_tokens": MAX_TOKENS
        }, timeout=120)
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        wall = time.time() - t0
    except Exception as e:
        content = f"ERROR: {e}"
        reasoning = ""
        tokens = 0
        wall = time.time() - t0

    s = score(t, content)
    cat = t["cat"]
    if cat not in cats:
        cats[cat] = {"total": 0, "correct": 0}
    cats[cat]["total"] += 1
    if s == "correct":
        cats[cat]["correct"] += 1

    results.append({
        "id": t["id"], "category": cat, "question": t["q"][:80],
        "score": s, "response_len": len(content), "reasoning_len": len(reasoning),
        "tokens": tokens, "time": round(wall, 2),
        "response_preview": content[:200]
    })
    marker = "✅" if s == "correct" else "❌"
    print(f"{marker} [{t['id']}] {t['q'][:55]}... ({tokens}t, {wall:.1f}s)")

print("\n" + "="*60)
print("CORRECTED BENCHMARK RESULTS (max_tokens=4096)")
print("="*60)
for cat, s in sorted(cats.items()):
    pct = s["correct"]/s["total"]*100 if s["total"] else 0
    bar = "█" * int(pct/5) + "░" * (20 - int(pct/5))
    print(f"  {cat:25s} {s['correct']:2d}/{s['total']:2d} {bar} {pct:.0f}%")
total_c = sum(s["correct"] for s in cats.values())
total = sum(s["total"] for s in cats.values())
print(f"\n  {'TOTAL':25s} {total_c:2d}/{total:2d}              {total_c/total*100:.1f}%")

with open("/home/billz/results/corrected_benchmark.json", "w") as f:
    json.dump({"results": results, "summary": cats, "max_tokens": MAX_TOKENS}, f, indent=2)
print(f"\nResults saved to /home/billz/results/corrected_benchmark.json")
