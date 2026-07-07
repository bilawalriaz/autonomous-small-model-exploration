#!/usr/bin/env python3
"""
Monitor aero's rollout progress. When it finishes its shard,
copy deck's remaining work to aero and restart there.
"""
import json, subprocess, time, os

AERO_SHARD = "/home/billz/mixed-shard.jsonl"
AERO_ROLLOUTS = "/home/billz/rollouts/rollouts.jsonl"
DECK_SHARD = "/home/billz/mixed-shard-deck.jsonl"
DECK_ROLLOUTS = "/home/billz/rollouts/rollouts.jsonl"

def get_line_count(host, path):
    """Get line count of a file on a remote host."""
    try:
        result = subprocess.run(
            ["ssh", host, f"wc -l < {path}"],
            capture_output=True, text=True, timeout=10
        )
        return int(result.stdout.strip())
    except:
        return 0

def get_completed_hashes(host):
    """Get set of completed prompt hashes from a machine."""
    try:
        result = subprocess.run(
            ["ssh", host, f"cat {host_path('completed.jsonl', host)}"],
            capture_output=True, text=True, timeout=30
        )
        hashes = set()
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                rec = json.loads(line)
                hashes.add(rec.get("prompt_hash", ""))
        return hashes
    except:
        return set()

def host_path(path, host):
    if host == "m3":
        return path.replace("/home/billz/", "/Users/bilawalriaz/")
    return path

def run_cmd(host, cmd, timeout=60):
    try:
        result = subprocess.run(
            ["ssh", host, cmd],
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except:
        return ""

def scp_file(src_host, src_path, dst_host, dst_path):
    """Copy file between hosts."""
    subprocess.run(
        ["scp", f"{src_host}:{src_path}", f"{dst_host}:{dst_path}"],
        timeout=300
    )

def main():
    AERO_TOTAL = 10386  # Expected rollouts for aero's shard
    
    print(f"Monitoring aero for completion (target: {AERO_TOTAL})...")
    
    while True:
        done = get_line_count("aero", AERO_ROLLOUTS)
        pct = done / AERO_TOTAL * 100
        print(f"  Aero: {done}/{AERO_TOTAL} ({pct:.1f}%)")
        
        if done >= AERO_TOTAL - 5:  # Close enough to done
            print(f"\n✅ Aero finished ({done} rollouts)")
            break
        
        time.sleep(60)  # Check every minute
    
    # Step 1: Get deck's completed hashes
    print("\nGetting deck's completed work...")
    deck_completed = get_line_count("deck", DECK_ROLLOUTS)
    print(f"  Deck has {deck_completed} completed rollouts")
    
    # Step 2: Copy deck's shard to aero
    print("Copying deck's shard to aero...")
    scp_file("deck", DECK_SHARD, "aero", "/home/billz/mixed-shard-deck.jsonl")
    
    # Step 3: Copy deck's completed hashes to aero (so it skips what deck already did)
    print("Copying deck's completed.jsonl to aero...")
    scp_file("deck", "/home/billz/rollouts/completed.jsonl", "aero", "/home/billz/completed-deck.jsonl")
    
    # Step 4: Merge completed hashes into aero's completed.jsonl
    print("Merging completed hashes...")
    run_cmd("aero", f"""
        cat /home/billz/completed-deck.jsonl >> /home/billz/rollouts/completed.jsonl
        rm /home/billz/completed-deck.jsonl
    """)
    
    # Step 5: Restart aero with deck's shard
    print("Restarting aero with deck's shard...")
    run_cmd("aero", """
        pkill -f rollout_multigen 2>/dev/null
        sleep 2
        cd /home/billz
        SHARD_FILE=/home/billz/mixed-shard-deck.jsonl \
        OUTPUT_DIR=/home/billz/rollouts-deck \
        nohup python3 /home/billz/rollout_multigen.py > /tmp/rollout-deck.log 2>&1 &
        echo "Deck shard started on aero, PID: $!"
    """)
    
    # Step 6: Kill deck
    print("Killing deck...")
    run_cmd("deck", "pkill -f rollout_multigen 2>/dev/null")
    run_cmd("deck", "kill $(lsof -t -i:8080 2>/dev/null) 2>/dev/null")
    
    print("\n✅ Handoff complete!")
    print("  - Aero now processing deck's shard")
    print("  - Deck freed up")

if __name__ == "__main__":
    main()
