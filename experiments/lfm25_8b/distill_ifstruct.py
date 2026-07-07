#!/usr/bin/env python3
"""
Distillation pipeline using LiquidAI/ifstruct-v1.0 dataset.

Students: aero (localhost:8080) + deck (deck.tail9cc5b.ts.net:8080)
Teacher: Mac llama.cpp (100.100.61.28:8081)

2000 prompts from Liquid AI's structured output dataset.
Students generate rollouts in parallel while teacher scores.
"""
import json, time, requests, os, sys, concurrent.futures, random
from pathlib import Path
from collections import deque

STUDENTS = [
    {"name": "aero", "url": "http://localhost:8080"},
    {"name": "deck", "url": "http://deck.tail9cc5b.ts.net:8080"},
]
TEACHER_URL = "http://100.100.61.28:8081"
DATA_DIR = Path("/home/billz/results/distill_ifstruct")
DATA_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS_FILE = Path("/home/billz/ifstruct_prompts.jsonl")
COMPLETED_FILE = DATA_DIR / "completed.jsonl"
SCORED_FILE = DATA_DIR / "scored.jsonl"


def load_prompts():
    prompts = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                prompts.append(d["prompt"])
    return prompts


def load_completed():
    completed = set()
    if COMPLETED_FILE.exists():
        with open(COMPLETED_FILE) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    completed.add(d["prompt"])
    return completed


def student_generate(student_url, prompt, timeout=120):
    try:
        r = requests.post(f"{student_url}/v1/chat/completions", json={
            "model": "model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096, "temperature": 0.7,
            "logprobs": True, "top_logprobs": 3,
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
    try:
        r = requests.post(f"{TEACHER_URL}/v1/chat/completions", json={
            "model": "model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048, "temperature": 0.2,
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


def check_health(url):
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.status_code == 200
    except:
        try:
            r = requests.get(f"{url}/v1/models", timeout=5)
            return r.status_code == 200
        except:
            return False


def main():
    prompts = load_prompts()
    completed = load_completed()
    remaining = [p for p in prompts if p not in completed]

    print(f"{'='*60}")
    print(f"IFSTRUCT DISTILLATION PIPELINE")
    print(f"{'='*60}")

    # Check students
    alive = []
    for s in STUDENTS:
        ok = check_health(s["url"])
        print(f"  {'✅' if ok else '❌'} {s['name']}: {s['url']}")
        if ok:
            alive.append(s)

    teacher_ok = check_health(TEACHER_URL)
    print(f"  {'✅' if teacher_ok else '❌'} teacher: {TEACHER_URL}")

    print(f"\n  Total prompts: {len(prompts)}")
    print(f"  Completed: {len(completed)}")
    print(f"  Remaining: {len(remaining)}")
    print(f"  Students: {len(alive)}")

    if not alive:
        print("\nERROR: No students reachable")
        return
    if not teacher_ok:
        print("\nWARNING: Teacher unreachable, queuing for later")

    # Shuffle remaining
    random.shuffle(remaining)
    prompt_queue = deque(remaining)

    batch_size = max(1, len(remaining) // (len(alive) * 10))  # ~10 rounds per student
    batch_size = min(batch_size, 5)

    round_num = 0
    while prompt_queue:
        round_num += 1
        print(f"\n--- Round {round_num} ({len(prompt_queue)} remaining) ---")

        # Distribute to students
        student_batches = {}
        for s in alive:
            batch = []
            for _ in range(batch_size):
                if prompt_queue:
                    batch.append(prompt_queue.popleft())
            if batch:
                student_batches[s["name"]] = (s, batch)

        if not student_batches:
            break

        # Generate in parallel
        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(alive)) as ex:
            futures = {}
            for name, (student, batch) in student_batches.items():
                futures[ex.submit(lambda s, b: [(p, student_generate(s["url"], p)) for p in b], student, batch)] = name
            for f in concurrent.futures.as_completed(futures):
                results = f.result()
                for prompt, result in results:
                    all_results.append({"prompt": prompt, **result})
                    with open(COMPLETED_FILE, "a") as fp:
                        fp.write(json.dumps({"prompt": prompt, "response": result.get("response", ""),
                                            "tokens": result.get("tokens", 0)}) + "\n")
                    completed.add(prompt)

        # Teacher scores
        if teacher_ok:
            for r in all_results:
                if not r.get("response"):
                    continue
                print(f"  [teacher] {r['prompt'][:50]}...", end=" ", flush=True)
                t = teacher_generate(r["prompt"])
                if "error" not in t:
                    scored = {
                        "prompt": r["prompt"],
                        "student_response": r["response"],
                        "student_logprobs": r.get("logprobs", []),
                        "teacher_response": t["response"],
                        "teacher_logprobs": t.get("logprobs", []),
                    }
                    with open(SCORED_FILE, "a") as fp:
                        fp.write(json.dumps(scored) + "\n")
                    print("✅")
                else:
                    print(f"❌ {t['error']}")

                time.sleep(0.5)

        print(f"  Progress: {len(completed)}/{len(prompts)} completed")

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"Completed: {len(completed)}")
    print(f"Scored: {sum(1 for _ in open(SCORED_FILE)) if SCORED_FILE.exists() else 0}")
    print(f"Data: {DATA_DIR}")


if __name__ == "__main__":
    main()
