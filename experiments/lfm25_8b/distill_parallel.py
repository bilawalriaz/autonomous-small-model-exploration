#!/usr/bin/env python3
"""
Parallel distillation pipeline: students generate while teacher scores.

Students: aero (localhost:8080) + deck (deck.tail9cc5b.ts.net:8080)
Teacher: Mac (100.100.61.28:8081)

Flow:
1. Generate prompts
2. Send prompts to both students in parallel
3. As results come in, queue them for teacher scoring
4. Teacher scores one at a time (bottleneck)
5. Accumulate scored pairs overnight
6. In morning: train with Unsloth on accumulated data
"""
import json, time, requests, os, sys, concurrent.futures
from pathlib import Path
from collections import deque

# Endpoints
STUDENTS = [
    {"name": "aero", "url": "http://localhost:8080"},
    {"name": "deck", "url": "http://deck.tail9cc5b.ts.net:8080"},
]
TEACHER_URL = "http://100.100.61.28:8081"

DATA_DIR = Path("/home/billz/results/distill_parallel")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Large prompt pool — will be distributed across students
PROMPTS = [
    "Solve: 3x + 7 = 22. Show your work.",
    "What is 15% of 240?",
    "Simplify: (2^3)(2^4)",
    "What is the GCD of 48 and 18?",
    "Solve for x: x^2 - 5x + 6 = 0",
    "What is the sum of the first 20 natural numbers?",
    "Convert 10110 binary to decimal.",
    "What is 7! ?",
    "If f(x) = 2x + 3, what is f(f(1))?",
    "What is the LCM of 12 and 18?",
    "Write a Python function to check if a number is prime.",
    "Write a Python function to reverse a string.",
    "Write a Python function to find the factorial of n.",
    "Write a Python function to check if a string is a palindrome.",
    "Write a Python function to count vowels in a string.",
    "If all cats are animals, and all animals are living things, are all cats living things?",
    "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball.",
    "If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?",
    "What comes next: 2, 6, 12, 20, 30, ?",
    "Explain why the sky is blue in one paragraph.",
    "List exactly 5 fruits, numbered 1-5.",
    "Explain quantum computing to a 10-year-old.",
    "Write a haiku about the ocean (5-7-5).",
    "Compare Python and JavaScript in 3 bullet points.",
    "What is the derivative of x^3?",
    "Solve: 2x - 5 = 15",
    "What is 2^10?",
    "How many bits in a byte?",
    "What is the time complexity of binary search?",
    "Write a Python function to merge two sorted lists.",
    "What is sqrt(144)?",
    "Convert 0xFF to decimal.",
    "What is the boiling point of water in Fahrenheit?",
    "Explain what a hash table is.",
    "What is the difference between a list and a tuple in Python?",
    "How does HTTPS work?",
    "What is REST?",
    "Explain recursion in one sentence.",
    "What is a closure?",
    "What is the CAP theorem?",
    "What is eventual consistency?",
    "Explain the difference between TCP and UDP.",
    "What is a deadlock?",
    "What is virtual memory?",
    "How does garbage collection work?",
    "What is a race condition?",
    "Explain the observer pattern.",
    "What is dependency injection?",
    "What is the difference between process and thread?",
    "How does DNS work?",
]

COMPLETED_FILE = DATA_DIR / "completed.jsonl"
SCORED_FILE = DATA_DIR / "scored.jsonl"
PROGRESS_FILE = DATA_DIR / "progress.json"


def load_completed():
    completed = set()
    if COMPLETED_FILE.exists():
        with open(COMPLETED_FILE) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    completed.add(d["prompt"])
    return completed


def load_scored():
    scored = []
    if SCORED_FILE.exists():
        with open(SCORED_FILE) as f:
            for line in f:
                if line.strip():
                    scored.append(json.loads(line))
    return scored


def student_generate(student_url, prompt, timeout=120):
    """Generate on a student and return response + logprobs."""
    try:
        r = requests.post(f"{student_url}/v1/chat/completions", json={
            "model": "model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048, "temperature": 0.7,
            "logprobs": True, "top_logprobs": 5,
        }, timeout=timeout)
        r.raise_for_status()
        choice = r.json()["choices"][0]
        content = choice["message"].get("content", "")
        lps = []
        if choice.get("logprobs") and "content" in choice["logprobs"]:
            for lp in choice["logprobs"]["content"]:
                lps.append({"token": lp["token"], "logprob": lp["logprob"]})
        return {"response": content, "logprobs": lps, "tokens": r.json().get("usage", {}).get("completion_tokens", 0)}
    except Exception as e:
        return {"response": "", "logprobs": [], "error": str(e)}


def teacher_generate(prompt, timeout=300):
    """Generate teacher reference + logprobs."""
    try:
        r = requests.post(f"{TEACHER_URL}/v1/chat/completions", json={
            "model": "model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024, "temperature": 0.3,
            "logprobs": True, "top_logprobs": 3,
        }, timeout=timeout)
        r.raise_for_status()
        choice = r.json()["choices"][0]
        content = choice["message"].get("content", "")
        lps = []
        if choice.get("logprobs") and "content" in choice["logprobs"]:
            for lp in choice["logprobs"]["content"]:
                lps.append({"token": lp["token"], "logprob": lp["logprob"]})
        return {"response": content, "logprobs": lps}
    except Exception as e:
        return {"response": "", "logprobs": [], "error": str(e)}


def check_student_health(student_url):
    """Check if a student is alive."""
    try:
        r = requests.get(f"{student_url}/health", timeout=5)
        return r.status_code == 200
    except:
        return False


def check_teacher_health():
    """Check if teacher is alive."""
    try:
        r = requests.get(f"{TEACHER_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        try:
            # llama.cpp might not have /health, try /v1/models
            r = requests.get(f"{TEACHER_URL}/v1/models", timeout=5)
            return r.status_code == 200
        except:
            return False


def update_progress(completed_count, scored_count, total):
    """Write progress to file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "completed": completed_count,
            "scored": scored_count,
            "total": total,
            "timestamp": time.time(),
        }, f, indent=2)


def step_parallel_generate_and_score():
    """Main loop: students generate, teacher scores as fast as it can."""
    print(f"\n{'='*60}")
    print("PARALLEL GENERATION + SCORING")
    print(f"{'='*60}")

    # Check connectivity
    alive_students = []
    for s in STUDENTS:
        if check_student_health(s["url"]):
            alive_students.append(s)
            print(f"  ✅ {s['name']}: {s['url']}")
        else:
            print(f"  ❌ {s['name']}: {s['url']} (unreachable)")

    if not alive_students:
        print("ERROR: No students reachable")
        return

    teacher_alive = check_teacher_health()
    print(f"  {'✅' if teacher_alive else '❌'} teacher: {TEACHER_URL}")
    if not teacher_alive:
        print("WARNING: Teacher unreachable. Will queue student responses for later scoring.")

    # Load existing progress
    completed = load_completed()
    scored_list = load_scored()
    remaining = [p for p in PROMPTS if p not in completed]

    print(f"\n  Completed: {len(completed)}/{len(PROMPTS)}")
    print(f"  Scored: {len(scored_list)}")
    print(f"  Remaining: {len(remaining)}")

    if not remaining:
        print("  All prompts completed!")
        return

    # Distribute prompts across students
    prompt_queue = deque(remaining)

    # Student generation loop
    def generate_batch(student, prompts_batch):
        """Generate on a student for a batch of prompts."""
        results = []
        for prompt in prompts_batch:
            print(f"  [{student['name']}] Generating: {prompt[:45]}...", end=" ", flush=True)
            result = student_generate(student["url"], prompt)
            if "error" not in result:
                print(f"✅ {result['tokens']} tokens, {len(result['logprobs'])} logprobs")
            else:
                print(f"❌ {result['error']}")
            results.append({"prompt": prompt, "student": student["name"], **result})
            time.sleep(0.5)
        return results

    # Score with teacher
    def score_pair(pair):
        """Score a student response with the teacher."""
        prompt = pair["prompt"]
        student_resp = pair.get("response", "")
        if not student_resp:
            return None

        print(f"  [teacher] Scoring: {prompt[:45]}...", end=" ", flush=True)
        teacher_result = teacher_generate(prompt)
        if "error" in teacher_result:
            print(f"❌ {teacher_result['error']}")
            return None

        # Simple agreement check
        s_words = set(student_resp.lower().split())
        t_words = set(teacher_result["response"].lower().split())
        agreement = len(s_words & t_words) > len(t_words) * 0.2

        pair["teacher_response"] = teacher_result["response"]
        pair["teacher_logprobs"] = teacher_result["logprobs"]
        pair["agreement"] = agreement
        print(f"{'✅' if agreement else '❌'} agree")
        return pair

    # Run: students generate in parallel, teacher scores sequentially
    batch_size = 3  # prompts per student per round
    round_num = 0

    while prompt_queue:
        round_num += 1
        print(f"\n--- Round {round_num} ---")

        # Get batch for each student
        student_batches = {}
        for s in alive_students:
            batch = []
            for _ in range(batch_size):
                if prompt_queue:
                    batch.append(prompt_queue.popleft())
            if batch:
                student_batches[s["name"]] = (s, batch)

        if not student_batches:
            break

        # Generate on students in parallel
        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(alive_students)) as executor:
            futures = {}
            for name, (student, batch) in student_batches.items():
                futures[executor.submit(generate_batch, student, batch)] = name

            for future in concurrent.futures.as_completed(futures):
                results = future.result()
                all_results.extend(results)

                # Save completed results
                for r in results:
                    with open(COMPLETED_FILE, "a") as f:
                        f.write(json.dumps({"prompt": r["prompt"], "student": r["student"],
                                           "response": r.get("response", ""),
                                           "logprobs": r.get("logprobs", []),
                                           "tokens": r.get("tokens", 0)}) + "\n")
                    completed.add(r["prompt"])

        # Teacher scores as results come in
        if teacher_alive:
            for r in all_results:
                if r.get("response"):
                    scored = score_pair(r)
                    if scored:
                        scored_list.append(scored)
                        with open(SCORED_FILE, "a") as f:
                            f.write(json.dumps(scored) + "\n")
                    update_progress(len(completed), len(scored_list), len(PROMPTS))

        print(f"  Progress: {len(completed)}/{len(PROMPTS)} completed, {len(scored_list)} scored")

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Completed: {len(completed)}")
    print(f"Scored: {len(scored_list)}")
    print(f"Data: {DATA_DIR}")


def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"

    if cmd == "run":
        step_parallel_generate_and_score()
    elif cmd == "status":
        completed = load_completed()
        scored = load_scored()
        print(f"Completed: {len(completed)}/{len(PROMPTS)}")
        print(f"Scored: {len(scored)}")
    else:
        print("Usage: python distill_parallel.py [run|status]")


if __name__ == "__main__":
    main()
