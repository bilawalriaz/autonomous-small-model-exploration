#!/usr/bin/env python3
"""
Three-machine overnight distillation pipeline.

Machines:
  aero   → LFM2.5-8B-A1B GGUF (port 8080, student rollouts)
  deck   → LFM2.5-8B-A1B GGUF (port 8080, student rollouts)
  mac    → Gemma 4 26B GGUF   (port 8081, teacher scoring tomorrow)

Each machine gets a unique slice of 2000 ifstruct prompts.
No prompt is processed by more than one machine.
Everything is saved: prompt, input tokens, output, logprobs, timing.

Usage:
  python overnight_rollouts.py run --machine aero   # run on aero
  python overnight_rollouts.py run --machine deck   # run on deck
  python overnight_rollouts.py run --machine mac    # Mac does student rollouts too
  python overnight_rollouts.py status               # check all machines
  python overnight_rollouts.py merge                # merge all results for training
"""
import json, time, hashlib, requests, sys, os
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
MACHINES = {
    "aero": {"url": "http://localhost:8080", "model": "lfm2.5-8b-a1b"},
    "deck": {"url": "http://deck.tail9cc5b.ts.net:8080", "model": "lfm2.5-8b-a1b"},
    "mac":  {"url": "http://100.100.61.28:8081", "model": "gemma4"},
}

PROMPTS_FILE = Path("/home/billz/ifstruct_prompts.jsonl")
RESULTS_DIR = Path("/home/billz/results/overnight_rollouts")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# PROMPT SLICING
# ============================================================
def load_prompts():
    prompts = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                prompts.append(d["prompt"])
    return prompts

def assign_machine(prompt, machines):
    """Deterministic assignment: hash prompt to machine index."""
    h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
    return machines[h % len(machines)]

def get_my_prompts(machine_name):
    """Get the prompts assigned to this machine."""
    all_prompts = load_prompts()
    machine_names = sorted(MACHINES.keys())
    my_prompts = [p for p in all_prompts if assign_machine(p, machine_names) == machine_name]
    return my_prompts

# ============================================================
# GENERATION
# ============================================================
def generate_with_logprobs(url, model, prompt, max_tokens=8192, temperature=0.7):
    """Generate and capture everything: response, logprobs, timing."""
    t0 = time.time()
    try:
        r = requests.post(f"{url}/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": 5,
        }, timeout=600)  # 10 min timeout for long outputs
        wall = time.time() - t0
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        content = choice["message"].get("content", "")
        reasoning = choice["message"].get("reasoning_content", "")
        usage = data.get("usage", {})

        # Extract logprobs
        logprobs = []
        if choice.get("logprobs") and "content" in choice["logprobs"]:
            for lp in choice["logprobs"]["content"]:
                logprobs.append({
                    "token": lp["token"],
                    "token_id": lp.get("id", -1),
                    "logprob": lp["logprob"],
                    "top_logprobs": [{"token": t["token"], "logprob": t["logprob"]}
                                     for t in lp.get("top_logprobs", [])],
                })

        return {
            "success": True,
            "response": content,
            "reasoning": reasoning,
            "logprobs": logprobs,
            "tokens_generated": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "wall_time": round(wall, 2),
            "tokens_per_sec": round(usage.get("completion_tokens", 0) / wall, 1) if wall > 0 else 0,
        }
    except Exception as e:
        wall = time.time() - t0
        return {
            "success": False,
            "response": "",
            "reasoning": "",
            "logprobs": [],
            "tokens_generated": 0,
            "prompt_tokens": 0,
            "wall_time": round(wall, 2),
            "error": str(e),
        }

# ============================================================
# MAIN LOOP
# ============================================================
def run_machine(machine_name):
    """Run rollouts for one machine."""
    cfg = MACHINES[machine_name]
    my_prompts = get_my_prompts(machine_name)

    # Load completed set
    out_file = RESULTS_DIR / f"rollouts_{machine_name}.jsonl"
    completed = set()
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    completed.add(d["prompt"])

    remaining = [p for p in my_prompts if p not in completed]

    print(f"{'='*60}")
    print(f"OVERNIGHT ROLLOUTS: {machine_name.upper()}")
    print(f"{'='*60}")
    print(f"  Machine: {machine_name} → {cfg['url']}")
    print(f"  Model: {cfg['model']}")
    print(f"  Assigned prompts: {len(my_prompts)}")
    print(f"  Already done: {len(completed)}")
    print(f"  Remaining: {len(remaining)}")
    print(f"  Max tokens: 8192")
    print(f"  Output: {out_file}")
    print()

    # Check connectivity
    try:
        r = requests.get(f"{cfg['url']}/health", timeout=5)
        if r.status_code != 200:
            raise Exception("not healthy")
        print(f"  ✅ {machine_name} reachable")
    except:
        try:
            r = requests.get(f"{cfg['url']}/v1/models", timeout=5)
            print(f"  ✅ {machine_name} reachable (via /v1/models)")
        except:
            print(f"  ❌ {machine_name} not reachable at {cfg['url']}")
            return

    # Generate
    total_tokens = 0
    total_time = 0
    for i, prompt in enumerate(remaining):
        print(f"  [{i+1}/{len(remaining)}] {prompt[:60]}...", end=" ", flush=True)

        result = generate_with_logprobs(cfg["url"], cfg["model"], prompt,
                                        max_tokens=8192, temperature=0.7)

        entry = {
            "prompt": prompt,
            "machine": machine_name,
            "model": cfg["model"],
            **result,
        }

        # Append to file immediately (crash-safe)
        with open(out_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        if result["success"]:
            total_tokens += result["tokens_generated"]
            total_time += result["wall_time"]
            print(f"✅ {result['tokens_generated']}t {result['wall_time']:.1f}s "
                  f"({result['tokens_per_sec']:.0f} t/s)")
        else:
            print(f"❌ {result.get('error', 'unknown')[:50]}")

        time.sleep(0.5)  # Gentle on the server

    print(f"\n  DONE: {len(remaining)} prompts, {total_tokens} tokens, "
          f"{total_time:.0f}s total")
    print(f"  Output: {out_file}")

def show_status():
    """Show status of all machines and rollouts."""
    print(f"{'='*60}")
    print("OVERNIGHT ROLLOUT STATUS")
    print(f"{'='*60}")

    all_prompts = load_prompts()
    machine_names = sorted(MACHINES.keys())

    for name in machine_names:
        cfg = MACHINES[name]
        out_file = RESULTS_DIR / f"rollouts_{name}.jsonl"

        # Count completed
        completed = 0
        total_tokens = 0
        if out_file.exists():
            with open(out_file) as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        completed += 1
                        total_tokens += d.get("tokens_generated", 0)

        assigned = sum(1 for p in all_prompts if assign_machine(p, machine_names) == name)

        # Check health
        healthy = False
        try:
            r = requests.get(f"{cfg['url']}/health", timeout=3)
            healthy = r.status_code == 200
        except:
            try:
                r = requests.get(f"{cfg['url']}/v1/models", timeout=3)
                healthy = r.status_code == 200
            except:
                pass

        print(f"\n  {name.upper()}:")
        print(f"    URL: {cfg['url']}")
        print(f"    Health: {'✅' if healthy else '❌'}")
        print(f"    Assigned: {assigned}")
        print(f"    Completed: {completed}/{assigned}")
        print(f"    Tokens: {total_tokens:,}")

    # Total
    total_completed = 0
    for name in machine_names:
        out_file = RESULTS_DIR / f"rollouts_{name}.jsonl"
        if out_file.exists():
            with open(out_file) as f:
                total_completed += sum(1 for l in f if l.strip())
    print(f"\n  TOTAL: {total_completed}/{len(all_prompts)} prompts completed")

def merge_results():
    """Merge all machine results into one file for training."""
    merged_file = RESULTS_DIR / "merged_rollouts.jsonl"
    all_entries = []
    for name in MACHINES:
        out_file = RESULTS_DIR / f"rollouts_{name}.jsonl"
        if out_file.exists():
            with open(out_file) as f:
                for line in f:
                    if line.strip():
                        all_entries.append(json.loads(line))

    with open(merged_file, "w") as f:
        for e in all_entries:
            f.write(json.dumps(e) + "\n")

    print(f"Merged {len(all_entries)} rollouts → {merged_file}")
    return merged_file

# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python overnight_rollouts.py [run --machine X | status | merge]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "run":
        machine = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--machine" else None
        if not machine:
            print("Usage: python overnight_rollouts.py run --machine aero|deck|mac")
            sys.exit(1)
        run_machine(machine)
    elif cmd == "status":
        show_status()
    elif cmd == "merge":
        merge_results()
    else:
        print("Usage: python overnight_rollouts.py [run --machine X | status | merge]")
