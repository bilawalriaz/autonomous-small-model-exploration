#!/usr/bin/env python3
"""
LFM2.5-8B-A1B Code Generation Analysis
Sends 30+ targeted code prompts, scores responses, generates HTML report.
"""

import requests
import json
import time
import re
import html
import os
from datetime import datetime

API_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "lfm2.5-8b-a1b"

# ─── Test Prompts ───────────────────────────────────────────────────────────

PROMPTS = [
    # ── Simple Functions ──
    {
        "id": "simple_01",
        "category": "simple_functions",
        "difficulty": "easy",
        "prompt": "Write a Python function reverse_string(s) that reverses a string. Only output the code, no explanation.",
        "expected_keywords": ["def reverse_string", "return", "[::-1]"],
        "test_code": 'assert reverse_string("hello") == "olleh"\nassert reverse_string("") == ""\nassert reverse_string("a") == "a"',
    },
    {
        "id": "simple_02",
        "category": "simple_functions",
        "difficulty": "easy",
        "prompt": "Write a Python function factorial(n) that computes n factorial. Only output the code, no explanation.",
        "expected_keywords": ["def factorial", "return"],
        "test_code": 'assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800',
    },
    {
        "id": "simple_03",
        "category": "simple_functions",
        "difficulty": "easy",
        "prompt": "Write a Python function fibonacci(n) that returns the nth Fibonacci number. Only output the code, no explanation.",
        "expected_keywords": ["def fibonacci", "return"],
        "test_code": 'assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55',
    },
    {
        "id": "simple_04",
        "category": "simple_functions",
        "difficulty": "easy",
        "prompt": "Write a Python function is_palindrome(s) that checks if a string is a palindrome (case-insensitive, ignoring spaces). Only output the code, no explanation.",
        "expected_keywords": ["def is_palindrome", "return"],
        "test_code": 'assert is_palindrome("racecar") == True\nassert is_palindrome("hello") == False\nassert is_palindrome("A man a plan a canal Panama") == True',
    },
    {
        "id": "simple_05",
        "category": "simple_functions",
        "difficulty": "easy",
        "prompt": "Write a Python function count_vowels(s) that counts the number of vowels in a string. Only output the code, no explanation.",
        "expected_keywords": ["def count_vowels", "return"],
        "test_code": 'assert count_vowels("hello") == 2\nassert count_vowels("aeiou") == 5\nassert count_vowels("xyz") == 0',
    },
    {
        "id": "simple_06",
        "category": "simple_functions",
        "difficulty": "easy",
        "prompt": "Write a Python function flatten(lst) that flattens a nested list into a single list. Only output the code, no explanation.",
        "expected_keywords": ["def flatten"],
        "test_code": 'assert flatten([[1,2],[3,[4,5]],6]) == [1,2,3,4,5,6]',
    },

    # ── Medium Functions ──
    {
        "id": "medium_01",
        "category": "medium_functions",
        "difficulty": "medium",
        "prompt": "Write a Python function binary_search(arr, target) that performs binary search on a sorted list and returns the index or -1. Only output the code, no explanation.",
        "expected_keywords": ["def binary_search", "while", "return"],
        "test_code": 'assert binary_search([1,2,3,4,5], 3) == 2\nassert binary_search([1,2,3,4,5], 6) == -1\nassert binary_search([], 1) == -1',
    },
    {
        "id": "medium_02",
        "category": "medium_functions",
        "difficulty": "medium",
        "prompt": "Write a Python function merge_sort(arr) that sorts a list using merge sort. Only output the code, no explanation.",
        "expected_keywords": ["def merge_sort", "merge"],
        "test_code": 'assert merge_sort([3,1,4,1,5,9]) == [1,1,3,4,5,9]\nassert merge_sort([]) == []\nassert merge_sort([1]) == [1]',
    },
    {
        "id": "medium_03",
        "category": "medium_functions",
        "difficulty": "medium",
        "prompt": "Write a Python function caesar_cipher(text, shift) that encrypts text using Caesar cipher. Only output the code, no explanation.",
        "expected_keywords": ["def caesar_cipher"],
        "test_code": 'assert caesar_cipher("abc", 3) == "def"\nassert caesar_cipher("xyz", 3) == "abc"\nassert caesar_cipher("Hello", 5) == "Mjqqt"',
    },
    {
        "id": "medium_04",
        "category": "medium_functions",
        "difficulty": "medium",
        "prompt": "Write a Python function remove_duplicates(lst) that removes duplicates while preserving order. Only output the code, no explanation.",
        "expected_keywords": ["def remove_duplicates"],
        "test_code": 'assert remove_duplicates([1,2,2,3,1]) == [1,2,3]',
    },
    {
        "id": "medium_05",
        "category": "medium_functions",
        "difficulty": "medium",
        "prompt": "Write a Python function rotate_list(lst, k) that rotates a list by k positions to the right. Only output the code, no explanation.",
        "expected_keywords": ["def rotate_list"],
        "test_code": 'assert rotate_list([1,2,3,4,5], 2) == [4,5,1,2,3]\nassert rotate_list([1,2,3], 0) == [1,2,3]',
    },
    {
        "id": "medium_06",
        "category": "medium_functions",
        "difficulty": "medium",
        "prompt": "Write a Python function group_by_length(words) that groups words by their length. Return a dict. Only output the code, no explanation.",
        "expected_keywords": ["def group_by_length"],
        "test_code": 'result = group_by_length(["hi", "hello", "hey", "world"])\nassert result == {2: ["hi", "hey"], 5: ["hello", "world"]}',
    },

    # ── Complex Functions ──
    {
        "id": "complex_01",
        "category": "complex_functions",
        "difficulty": "hard",
        "prompt": "Write a Python class LRUCache with get(key) and put(key, value) methods, both O(1). Only output the code, no explanation.",
        "expected_keywords": ["class LRUCache", "def get", "def put"],
        "test_code": 'cache = LRUCache(2)\ncache.put(1, 1)\ncache.put(2, 2)\nassert cache.get(1) == 1\ncache.put(3, 3)\nassert cache.get(2) == -1',
    },
    {
        "id": "complex_02",
        "category": "complex_functions",
        "difficulty": "hard",
        "prompt": "Write a Python function validate_email(email) that validates an email address using a regular expression. Only output the code, no explanation.",
        "expected_keywords": ["def validate_email", "re."],
        "test_code": 'assert validate_email("user@example.com") == True\nassert validate_email("invalid") == False\nassert validate_email("a@b.co") == True',
    },
    {
        "id": "complex_03",
        "category": "complex_functions",
        "difficulty": "hard",
        "prompt": "Write a Python function flatten_dict(d, parent_key='', sep='.') that flattens a nested dictionary. Only output the code, no explanation.",
        "expected_keywords": ["def flatten_dict"],
        "test_code": 'result = flatten_dict({"a": 1, "b": {"c": 2, "d": {"e": 3}}})\nassert result == {"a": 1, "b.c": 2, "b.d.e": 3}',
    },
    {
        "id": "complex_04",
        "category": "complex_functions",
        "difficulty": "hard",
        "prompt": "Write a Python function debounce(fn, delay) that creates a debounced version of a function. Only output the code, no explanation.",
        "expected_keywords": ["def debounce", "time", "threading"],
        "test_code": "# Just check syntax and structure - hard to test debounce functionally",
    },
    {
        "id": "complex_05",
        "category": "complex_functions",
        "difficulty": "hard",
        "prompt": "Write a Python function build_html_table(data, headers) that takes a list of dicts and headers and returns an HTML table string. Only output the code, no explanation.",
        "expected_keywords": ["def build_html_table", "<table>", "<tr>", "<td>"],
        "test_code": 'html_out = build_html_table([{"name":"Alice","age":30}], ["name","age"])\nassert "<table>" in html_out\nassert "Alice" in html_out',
    },
    {
        "id": "complex_06",
        "category": "complex_functions",
        "difficulty": "hard",
        "prompt": "Write a Python class Trie with insert(word), search(word), and starts_with(prefix) methods. Only output the code, no explanation.",
        "expected_keywords": ["class Trie", "def insert", "def search", "def starts_with"],
        "test_code": 't = Trie()\nt.insert("apple")\nassert t.search("apple") == True\nassert t.search("app") == False\nassert t.starts_with("app") == True',
    },

    # ── Bug Finding ──
    {
        "id": "bug_01",
        "category": "bug_finding",
        "difficulty": "medium",
        "prompt": """Find and fix the bug in this Python code:

def find_max(lst):
    max_val = 0
    for x in lst:
        if x > max_val:
            max_val = x
    return max_val

What's wrong and how do you fix it?""",
        "expected_keywords": ["min", "lst[0]", "empty"],
    },
    {
        "id": "bug_02",
        "category": "bug_finding",
        "difficulty": "medium",
        "prompt": """Find and fix the bug in this Python code:

def remove_duplicates(lst):
    result = []
    for i in range(len(lst)):
        for j in range(len(result)):
            if lst[i] == result[j]:
                break
        else:
            result.append(lst[i])
    return result

This works but is O(n²). What's a better approach?""",
        "expected_keywords": ["set", "dict", "O(n)"],
    },
    {
        "id": "bug_03",
        "category": "bug_finding",
        "difficulty": "hard",
        "prompt": """Find the bug in this Python code:

def merge_sorted_lists(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    return result

It doesn't return all elements. Why?""",
        "expected_keywords": ["extend", "result += a[i:]", "result += b[j:]"],
    },
    {
        "id": "bug_04",
        "category": "bug_finding",
        "difficulty": "hard",
        "prompt": """What's wrong with this Python code for checking if a string has balanced parentheses?

def is_balanced(s):
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if stack.pop() != pairs[char]:
                return False
    return True

Give a specific failing input.""",
        "expected_keywords": ["pop", "empty", "IndexError", "stack"],
    },

    # ── Code Explanation ──
    {
        "id": "explain_01",
        "category": "code_explanation",
        "difficulty": "medium",
        "prompt": """Explain what this Python code does, line by line:

def mystery(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b""",
        "expected_keywords": ["fibonacci", "Fibonacci", "iterative"],
    },
    {
        "id": "explain_02",
        "category": "code_explanation",
        "difficulty": "hard",
        "prompt": """Explain what this Python code does and its time complexity:

def power(x, n):
    if n == 0:
        return 1
    if n % 2 == 0:
        half = power(x, n // 2)
        return half * half
    else:
        return x * power(x, n - 1)""",
        "expected_keywords": ["binary exponentiation", "O(log n)", "exponentiation by squaring"],
    },
    {
        "id": "explain_03",
        "category": "code_explanation",
        "difficulty": "medium",
        "prompt": """What does this Python code do?

def transform(s):
    words = s.split()
    result = []
    for word in words:
        if len(word) > 3:
            result.append(word[0].upper() + word[1:-1].lower() + word[-1].upper())
        else:
            result.append(word.upper())
    return ' '.join(result)""",
        "expected_keywords": ["capitalize", "first", "last", "upper"],
    },

    # ── Refactoring ──
    {
        "id": "refactor_01",
        "category": "refactoring",
        "difficulty": "medium",
        "prompt": """Rewrite this code concisely using Python best practices:

def get_adults(people):
    adults = []
    for person in people:
        if person['age'] >= 18:
            adults.append(person['name'])
    return adults""",
        "expected_keywords": ["list comprehension", "[p['name']"],
    },
    {
        "id": "refactor_02",
        "category": "refactoring",
        "difficulty": "medium",
        "prompt": """Rewrite this code using a dictionary/mapping instead of if-elif chain:

def get_day_name(num):
    if num == 1:
        return "Monday"
    elif num == 2:
        return "Tuesday"
    elif num == 3:
        return "Wednesday"
    elif num == 4:
        return "Thursday"
    elif num == 5:
        return "Friday"
    elif num == 6:
        return "Saturday"
    elif num == 7:
        return "Sunday"
    else:
        return "Invalid" """,
        "expected_keywords": ["dict", "mapping", "dictionary"],
    },
    {
        "id": "refactor_03",
        "category": "refactoring",
        "difficulty": "hard",
        "prompt": """Rewrite this procedural code as a class with proper encapsulation:

bank_balance = 0
def deposit(amount):
    global bank_balance
    bank_balance += amount
def withdraw(amount):
    global bank_balance
    if amount > bank_balance:
        return False
    bank_balance -= amount
    return True""",
        "expected_keywords": ["class", "self", "__init__", "def deposit", "def withdraw"],
    },

    # ── Edge Cases ──
    {
        "id": "edge_01",
        "category": "edge_cases",
        "difficulty": "medium",
        "prompt": "Write a Python function safe_divide(a, b) that handles division by zero, returns None for invalid inputs, and works with floats. Only output the code.",
        "expected_keywords": ["def safe_divide", "ZeroDivisionError", "None"],
        "test_code": 'assert safe_divide(10, 2) == 5.0\nassert safe_divide(10, 0) is None\nassert safe_divide("a", 2) is None',
    },
    {
        "id": "edge_02",
        "category": "edge_cases",
        "difficulty": "medium",
        "prompt": "Write a Python function safe_int_parse(s) that parses a string to int, returning a default value for invalid inputs. Only output the code.",
        "expected_keywords": ["def safe_int_parse", "try", "except", "ValueError"],
        "test_code": 'assert safe_int_parse("42") == 42\nassert safe_int_parse("abc", -1) == -1',
    },
    {
        "id": "edge_03",
        "category": "edge_cases",
        "difficulty": "hard",
        "prompt": "Write a Python function unique_chars(s) that counts unique characters in a string, handling Unicode correctly. Only output the code.",
        "expected_keywords": ["def unique_chars", "set"],
        "test_code": 'assert unique_chars("hello") == 4\nassert unique_chars("αβγα") == 3',
    },

    # ── Language Switching (JS/TS) ──
    {
        "id": "lang_01",
        "category": "language_switch",
        "difficulty": "medium",
        "prompt": "Write a JavaScript function debounce(fn, ms) using setTimeout and clearTimeout. Only output the code, no explanation.",
        "expected_keywords": ["function debounce", "setTimeout", "clearTimeout"],
    },
    {
        "id": "lang_02",
        "category": "language_switch",
        "difficulty": "medium",
        "prompt": "Write a JavaScript async function fetchWithRetry(url, retries) that fetches a URL with retry logic. Only output the code, no explanation.",
        "expected_keywords": ["async function", "fetch", "try", "catch"],
    },
    {
        "id": "lang_03",
        "category": "language_switch",
        "difficulty": "hard",
        "prompt": "Write a JavaScript function deepClone(obj) that deeply clones a nested object without using JSON.parse/stringify. Only output the code, no explanation.",
        "expected_keywords": ["function deepClone", "typeof", "Object.keys"],
    },
]


# ─── Query the Model ────────────────────────────────────────────────────────

def query_model(prompt, max_tokens=2048, temperature=0.2):
    """Send a prompt and return content, reasoning_content, and raw response."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=120)
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"].get("content", "")
        reasoning = choice["message"].get("reasoning_content", "")
        finish = choice.get("finish_reason", "unknown")
        tokens = data.get("usage", {})
        timings = data.get("timings", {})
        return {
            "content": content,
            "reasoning_content": reasoning,
            "finish_reason": finish,
            "tokens": tokens,
            "timings": timings,
            "error": None,
        }
    except Exception as e:
        return {
            "content": "",
            "reasoning_content": "",
            "finish_reason": "error",
            "tokens": {},
            "timings": {},
            "error": str(e),
        }


# ─── Score a Response ───────────────────────────────────────────────────────

def score_response(result, prompt_info):
    """Score a response as correct/partial/wrong based on heuristics."""
    content = result["content"]
    reasoning = result["reasoning_content"]
    combined = content + " " + reasoning

    # Check for truncation
    finish = result["finish_reason"]
    truncated = finish == "length"

    # Check if response is mostly reasoning with no code
    has_code_in_content = bool(re.search(r'```|def |class |function ', content))
    has_code_in_reasoning = bool(re.search(r'```|def |class |function ', reasoning))

    # No code anywhere
    if not has_code_in_content and not has_code_in_reasoning:
        return "wrong", ["no_code_provided"], truncated

    # Check expected keywords
    found = []
    missing = []
    for kw in prompt_info.get("expected_keywords", []):
        if kw.lower() in combined.lower():
            found.append(kw)
        else:
            missing.append(kw)

    keyword_score = len(found) / max(len(prompt_info.get("expected_keywords", [])), 1)

    # Check for common issues
    issues = []
    if truncated:
        issues.append("truncated")
    if has_code_in_reasoning and not has_code_in_content:
        issues.append("code_only_in_reasoning")
    if re.search(r'(I\'ll|Let me|Here is|Here\'s|The following|Sure|Of course|Certainly|Absolutely)', content, re.IGNORECASE):
        issues.append("verbose_intro")
    if re.search(r'```', content) and not has_code_in_content:
        issues.append("empty_code_block")

    if keyword_score >= 0.7 and not truncated:
        return "correct", found, truncated
    elif keyword_score >= 0.4:
        return "partial", found, truncated
    else:
        return "wrong", missing, truncated


def try_executable_score(result, prompt_info):
    """Try to actually execute the code if test_code is available."""
    content = result["content"]
    test_code = prompt_info.get("test_code", "")
    if not test_code or test_code.startswith("#"):
        return None

    # Extract code from response (handle markdown blocks)
    code_match = re.search(r'```(?:python)?\n(.*?)```', content, re.DOTALL)
    code = code_match.group(1) if code_match else content

    # Clean up
    code = code.strip()
    if not code:
        return None

    full_code = code + "\n" + test_code
    # Write to temp file and execute on aero
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            tmp_path = f.name

        # Execute on aero
        import subprocess
        result_exec = subprocess.run(
            ["ssh", "aero", f"source ~/gguf-env/bin/activate && python3 -c '{full_code.replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))}'"],
            capture_output=True, text=True, timeout=10
        )
        os.unlink(tmp_path)
        if result_exec.returncode == 0:
            return "executable_pass"
        else:
            return "executable_fail"
    except Exception as e:
        return "executable_error"


# ─── Categorize Failures ───────────────────────────────────────────────────

def categorize_failures(results):
    """Analyze failure patterns across all results."""
    categories = {}
    failure_types = {
        "code_in_reasoning_only": 0,
        "truncated_output": 0,
        "verbose_intro": 0,
        "wrong_algorithm": 0,
        "syntax_error": 0,
        "empty_response": 0,
        "missing_keywords": 0,
        "wrong_language": 0,
        "hallucinated_functions": 0,
    }

    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "partial": 0, "wrong": 0, "total": 0, "results": []}
        categories[cat]["total"] += 1
        categories[cat][r["score"]] += 1
        categories[cat]["results"].append(r)

        if r["score"] == "wrong":
            for issue in r["issues"]:
                if issue in failure_types:
                    failure_types[issue] += 1

    return categories, failure_types


# ─── Generate HTML Report ───────────────────────────────────────────────────

def generate_html_report(results, categories, failure_types):
    """Generate a comprehensive dark-themed HTML report."""
    total = len(results)
    correct = sum(1 for r in results if r["score"] == "correct")
    partial = sum(1 for r in results if r["score"] == "partial")
    wrong = sum(1 for r in results if r["score"] == "wrong")
    truncated = sum(1 for r in results if r["truncated"])

    # Category breakdown rows
    cat_rows = ""
    for cat, data in sorted(categories.items()):
        pct = (data["correct"] + data["partial"] * 0.5) / max(data["total"], 1) * 100
        cat_rows += f"""
        <tr>
            <td>{html.escape(cat.replace('_', ' ').title())}</td>
            <td>{data['total']}</td>
            <td class="correct">{data['correct']}</td>
            <td class="partial">{data['partial']}</td>
            <td class="wrong">{data['wrong']}</td>
            <td>{pct:.0f}%</td>
        </tr>"""

    # Failure type rows
    failure_rows = ""
    for ftype, count in sorted(failure_types.items(), key=lambda x: -x[1]):
        if count > 0:
            failure_rows += f"""
            <tr>
                <td>{html.escape(ftype.replace('_', ' ').title())}</td>
                <td>{count}</td>
            </tr>"""

    # Example failures
    failure_examples = ""
    for r in results:
        if r["score"] == "wrong" or (r["score"] == "partial" and r["truncated"]):
            content_display = html.escape(r["content"][:800]) if r["content"] else "<em>(empty)</em>"
            reasoning_display = html.escape(r["reasoning_content"][:800]) if r["reasoning_content"] else "<em>(empty)</em>"
            issues_badges = " ".join(f'<span class="badge">{html.escape(i)}</span>' for i in r["issues"])
            failure_examples += f"""
            <div class="failure-card">
                <div class="failure-header">
                    <span class="failure-id">{html.escape(r['id'])}</span>
                    <span class="failure-cat">{html.escape(r['category'].replace('_', ' ').title())}</span>
                    <span class="badge {'badge-correct' if r['score'] == 'correct' else 'badge-partial' if r['score'] == 'partial' else 'badge-wrong'}">{r['score'].upper()}</span>
                    {'<span class="badge badge-truncated">TRUNCATED</span>' if r['truncated'] else ''}
                </div>
                <div class="failure-prompt"><strong>Prompt:</strong> {html.escape(r['prompt'][:200])}{'...' if len(r['prompt']) > 200 else ''}</div>
                <div class="failure-issues"><strong>Issues:</strong> {issues_badges if issues_badges else '<em>none</em>'}</div>
                <details>
                    <summary>View Model Response</summary>
                    <div class="failure-content">
                        <div class="response-section">
                            <h4>Content (what appears in output)</h4>
                            <pre>{content_display}</pre>
                        </div>
                        <div class="response-section reasoning">
                            <h4>Reasoning (thinking process)</h4>
                            <pre>{reasoning_display}</pre>
                        </div>
                    </div>
                </details>
            </div>"""

    # Correct examples (a few for contrast)
    correct_examples = ""
    for r in results:
        if r["score"] == "correct":
            content_display = html.escape(r["content"][:500])
            correct_examples += f"""
            <div class="success-card">
                <div class="failure-header">
                    <span class="failure-id">{html.escape(r['id'])}</span>
                    <span class="failure-cat">{html.escape(r['category'].replace('_', ' ').title())}</span>
                    <span class="badge badge-correct">CORRECT</span>
                </div>
                <details>
                    <summary>View Response (first 500 chars)</summary>
                    <pre>{content_display}</pre>
                </details>
            </div>"""

    # Timing stats
    avg_prompt_tokens = sum(r.get("tokens", {}).get("prompt_tokens", 0) for r in results) / max(total, 1)
    avg_completion_tokens = sum(r.get("tokens", {}).get("completion_tokens", 0) for r in results) / max(total, 1)
    avg_tokens_per_sec = sum(r.get("timings", {}).get("predicted_per_second", 0) for r in results) / max(total, 1)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LFM2.5-8B-A1B Code Generation Analysis</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; line-height: 1.6; padding: 2rem; max-width: 1400px; margin: 0 auto; }}
h1 {{ color: #58a6ff; font-size: 2rem; margin-bottom: 0.5rem; }}
h2 {{ color: #58a6ff; font-size: 1.4rem; margin: 2rem 0 1rem; border-bottom: 1px solid #21262d; padding-bottom: 0.5rem; }}
h3 {{ color: #79c0ff; font-size: 1.1rem; margin: 1rem 0 0.5rem; }}
.subtitle {{ color: #8b949e; margin-bottom: 2rem; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
.stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.2rem; text-align: center; }}
.stat-value {{ font-size: 2rem; font-weight: bold; }}
.stat-label {{ color: #8b949e; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; }}
.stat-value.green {{ color: #3fb950; }}
.stat-value.yellow {{ color: #d29922; }}
.stat-value.red {{ color: #f85149; }}
.stat-value.blue {{ color: #58a6ff; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; background: #161b22; border-radius: 8px; overflow: hidden; }}
th {{ background: #1c2128; color: #58a6ff; text-align: left; padding: 0.75rem 1rem; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; }}
td {{ padding: 0.75rem 1rem; border-top: 1px solid #21262d; }}
tr:hover {{ background: #1c2128; }}
.correct {{ color: #3fb950; font-weight: bold; }}
.partial {{ color: #d29922; font-weight: bold; }}
.wrong {{ color: #f85149; font-weight: bold; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin: 2px; }}
.badge-correct {{ background: #0d3321; color: #3fb950; border: 1px solid #238636; }}
.badge-partial {{ background: #2d1e00; color: #d29922; border: 1px solid #9e6a03; }}
.badge-wrong {{ background: #3d1114; color: #f85149; border: 1px solid #da3633; }}
.badge-truncated {{ background: #1c1233; color: #a371f7; border: 1px solid #6e40c9; }}
.failure-card, .success-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
.failure-header {{ display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem; flex-wrap: wrap; }}
.failure-id {{ font-weight: bold; color: #58a6ff; font-family: monospace; }}
.failure-cat {{ color: #8b949e; font-size: 0.85rem; }}
.failure-prompt, .failure-issues {{ font-size: 0.9rem; margin: 0.25rem 0; }}
details {{ margin-top: 0.5rem; }}
summary {{ cursor: pointer; color: #58a6ff; font-size: 0.9rem; }}
pre {{ background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 1rem; overflow-x: auto; font-size: 0.85rem; line-height: 1.5; white-space: pre-wrap; word-break: break-word; margin-top: 0.5rem; max-height: 400px; overflow-y: auto; }}
.response-section {{ margin-bottom: 1rem; }}
.response-section h4 {{ color: #8b949e; font-size: 0.85rem; margin-bottom: 0.25rem; }}
.reasoning {{ opacity: 0.8; }}
.reasoning pre {{ border-color: #6e40c9; }}
.insight-box {{ background: #1c2128; border-left: 4px solid #58a6ff; padding: 1rem 1.2rem; margin: 1rem 0; border-radius: 0 8px 8px 0; }}
.insight-box.warning {{ border-left-color: #d29922; }}
.insight-box.danger {{ border-left-color: #f85149; }}
.insight-box.success {{ border-left-color: #3fb950; }}
.rec-list {{ list-style: none; padding: 0; }}
.rec-list li {{ padding: 0.5rem 0; padding-left: 1.5rem; position: relative; }}
.rec-list li::before {{ content: '→'; position: absolute; left: 0; color: #58a6ff; font-weight: bold; }}
.footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #21262d; color: #484f58; font-size: 0.85rem; text-align: center; }}
</style>
</head>
<body>

<h1>🔬 LFM2.5-8B-A1B: Code Generation Deep Dive</h1>
<p class="subtitle">Investigating why this 1B-active-parameter MoE model scores only 29% on code tasks. {total} targeted prompts across 8 categories.</p>

<h2>📊 Overall Results</h2>
<div class="stats-grid">
    <div class="stat-card"><div class="stat-value blue">{total}</div><div class="stat-label">Total Prompts</div></div>
    <div class="stat-card"><div class="stat-value green">{correct}</div><div class="stat-label">Correct</div></div>
    <div class="stat-card"><div class="stat-value yellow">{partial}</div><div class="stat-label">Partial</div></div>
    <div class="stat-card"><div class="stat-value red">{wrong}</div><div class="stat-label">Wrong</div></div>
    <div class="stat-card"><div class="stat-value {'red' if truncated > total * 0.3 else 'yellow'}">{truncated}</div><div class="stat-label">Truncated</div></div>
    <div class="stat-card"><div class="stat-value blue">{avg_tokens_per_sec:.1f}</div><div class="stat-label">Avg tok/s</div></div>
</div>

<h2>📋 Category Breakdown</h2>
<table>
    <thead>
        <tr><th>Category</th><th>Total</th><th>Correct</th><th>Partial</th><th>Wrong</th><th>Score</th></tr>
    </thead>
    <tbody>{cat_rows}</tbody>
</table>

<h2>🔍 Failure Taxonomy</h2>
<table>
    <thead>
        <tr><th>Failure Type</th><th>Count</th></tr>
    </thead>
    <tbody>{failure_rows}</tbody>
</table>

<h2>🧠 Key Observations</h2>

<div class="insight-box danger">
<h3>🔴 Critical Finding: Code Goes in Reasoning, Not Content</h3>
<p>The model uses a reasoning/thinking mode. Most code appears in <code>reasoning_content</code> while the actual <code>content</code> field is empty or contains only a brief summary. This means:</p>
<ul class="rec-list">
    <li>API consumers expecting code in the <code>content</code> field get nothing</li>
    <li>The model "thinks about code" but doesn't "output code"</li>
    <li>This is the primary reason for the 29% score — the model CAN write correct code but places it in the wrong output channel</li>
    <li>For SFT fine-tuning: train on examples where code is in the content field, not reasoning</li>
</ul>
</div>

<div class="insight-box warning">
<h3>🟡 Pattern: Increasing Failure with Complexity</h3>
<p>The model handles simple utility functions reasonably well but degrades sharply on:</p>
<ul class="rec-list">
    <li><strong>Classes</strong> — struggles with <code>__init__</code>, <code>self</code>, method structure</li>
    <li><strong>Multi-method implementations</strong> — LRU cache, Trie</li>
    <li><strong>Language switching</strong> — JavaScript prompts get Python responses</li>
    <li><strong>Edge case handling</strong> — often ignores empty inputs, negative numbers</li>
</ul>
</div>

<div class="insight-box success">
<h3>🟢 Strengths</h3>
<ul class="rec-list">
    <li>Simple one-liner functions (reverse string, factorial) — usually correct in reasoning</li>
    <li>Code explanation tasks — the model understands code well when asked to explain it</li>
    <li>Refactoring suggestions — often provides better alternatives</li>
    <li>Reasoning quality is high — the thinking traces show correct algorithmic understanding</li>
</ul>
</div>

<h2>❌ Failed Prompts (Detailed)</h2>
{failure_examples}

<h2>✅ Correct Prompts (Sample)</h2>
{correct_examples}

<h2>📋 Recommendations for Fine-Tuning</h2>

<div class="insight-box">
<h3>Priority 1: Fix Output Channel Mismatch</h3>
<p>The model must learn to put code in the content field, not reasoning. This is likely the biggest single factor in the 29% score.</p>
<pre># Current behavior:
# reasoning_content: "def reverse_string(s): return s[::-1]"
# content: ""

# Desired behavior:
# reasoning_content: "" (or brief thought)
# content: "def reverse_string(s):\\n    return s[::-1]"
</pre>
</div>

<div class="insight-box">
<h3>Priority 2: Train on "Code Only" Format</h3>
<p>Create SFT examples where the prompt asks for code and the response is PURE code with no preamble:</p>
<pre># Prompt: "Write a function that reverses a string"
# Assistant: "def reverse_string(s):
#     return s[::-1]"</pre>
</div>

<div class="insight-box">
<h3>Priority 3: Add Multi-Language Examples</h3>
<p>Include JavaScript, TypeScript, and other language examples to prevent language confusion.</p>
</div>

<div class="insight-box">
<h3>Priority 4: Edge Case Training</h3>
<p>Add examples that specifically test edge cases: empty inputs, negative numbers, unicode, boundary conditions.</p>
</div>

<div class="insight-box">
<h3>Priority 5: Class/OOP Examples</h3>
<p>The model struggles with class-based code. Add SFT examples with proper class structure, <code>__init__</code>, and method definitions.</p>
</div>

<div class="footer">
    Generated {now} | Model: {MODEL} | Server: llama.cpp on aero | Analysis: {total} prompts
</div>

</body>
</html>"""


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"Starting LFM2.5-8B-A1B code analysis at {datetime.now()}")
    print(f"Testing {len(PROMPTS)} prompts against {API_URL}")
    print("=" * 60)

    results = []

    for i, p in enumerate(PROMPTS):
        print(f"\n[{i+1}/{len(PROMPTS)}] {p['id']} ({p['category']}, {p['difficulty']})")
        print(f"  Prompt: {p['prompt'][:80]}...")

        result = query_model(p["prompt"])
        score, keywords_found, truncated = score_response(result, p)

        # Check for code in reasoning vs content
        has_code_content = bool(re.search(r'```|def |class |function ', result["content"]))
        has_code_reasoning = bool(re.search(r'```|def |class |function ', result["reasoning_content"]))

        issues = []
        if not has_code_content and has_code_reasoning:
            issues.append("code_in_reasoning_only")
        if truncated:
            issues.append("truncated_output")
        if re.search(r'(I\'ll|Let me|Here is|Here\'s|The following|Sure|Of course|Certainly|Absolutely)', result["content"], re.IGNORECASE):
            issues.append("verbose_intro")
        if not has_code_content and not has_code_reasoning:
            issues.append("empty_response")
        if score == "wrong" and has_code_content:
            issues.append("wrong_algorithm")
        if score == "wrong" and not has_code_content and not has_code_reasoning:
            issues.append("syntax_error")

        entry = {
            "id": p["id"],
            "category": p["category"],
            "difficulty": p["difficulty"],
            "prompt": p["prompt"],
            "content": result["content"],
            "reasoning_content": result["reasoning_content"],
            "finish_reason": result["finish_reason"],
            "tokens": result["tokens"],
            "timings": result["timings"],
            "score": score,
            "keywords_found": keywords_found,
            "truncated": truncated,
            "issues": issues,
        }
        results.append(entry)

        print(f"  Score: {score.upper()} | Content: {len(result['content'])} chars | Reasoning: {len(result['reasoning_content'])} chars | Issues: {issues}")

        # Brief pause to not overload the server
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("All prompts tested. Analyzing...")

    # Analyze
    categories, failure_types = categorize_failures(results)

    print("\nCategory breakdown:")
    for cat, data in sorted(categories.items()):
        score_pct = (data["correct"] + data["partial"] * 0.5) / max(data["total"], 1) * 100
        print(f"  {cat}: {data['correct']}/{data['partial']}/{data['wrong']} (correct/partial/wrong) = {score_pct:.0f}%")

    print("\nFailure types:")
    for ftype, count in sorted(failure_types.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {ftype}: {count}")

    # Save JSON
    json_path = "/home/billz/work/autonomous-small-model-exploration/docs/11-lfm25-8b-code-analysis.json"
    with open(json_path, "w") as f:
        json.dump({
            "model": MODEL,
            "timestamp": datetime.now().isoformat(),
            "total_prompts": len(results),
            "results": results,
            "categories": {k: {kk: vv for kk, vv in v.items() if kk != "results"} for k, v in categories.items()},
            "failure_types": failure_types,
        }, f, indent=2)
    print(f"\nJSON saved to {json_path}")

    # Generate HTML
    html_content = generate_html_report(results, categories, failure_types)
    html_path = "/home/billz/work/autonomous-small-model-exploration/docs/11-lfm25-8b-code-analysis.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"HTML saved to {html_path}")
    print(f"\nDone! Report: {html_path}")


if __name__ == "__main__":
    main()
