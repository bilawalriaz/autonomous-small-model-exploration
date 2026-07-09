# Agent Trajectory Lab

Minimal harness for collecting Hermes Agent traces against an OpenAI-compatible
local model endpoint.

On lenovo, Hermes is configured to use aero's LFM2.5-8B-A1B llama.cpp server:

```bash
~/.hermes/hermes-agent/venv/bin/hermes config set model.provider custom
~/.hermes/hermes-agent/venv/bin/hermes config set model.default model
~/.hermes/hermes-agent/venv/bin/hermes config set model.base_url http://aero:8080/v1
~/.hermes/hermes-agent/venv/bin/hermes config set model.api_key dummy
```

Run a smoke task:

```bash
cd ~/agent_trajectory_lab
python3 run_hermes_tasks.py --tasks agent_tasks.json --out runs/aero_lfm_smoke --limit 1
```

Run all seed tasks:

```bash
cd ~/agent_trajectory_lab
python3 run_hermes_tasks.py --tasks agent_tasks.json --out runs/aero_lfm_seed12
```

Generate and run the sysadmin/deployment queue with the tool-capable Nous model:

```bash
cd ~/agent_trajectory_lab
python3 generate_sysadmin_tasks.py
~/.hermes/hermes-agent/venv/bin/hermes config set model.provider nous
~/.hermes/hermes-agent/venv/bin/hermes config set model.default stepfun/step-3.7-flash:free
~/.hermes/hermes-agent/venv/bin/hermes config set model.base_url https://inference-api.nousresearch.com/v1
nohup python3 run_hermes_tasks.py \
  --tasks sysadmin_tasks.json \
  --out runs/sysadmin_stepfun_full \
  --skip-existing \
  --timeout 900 \
  > runs/sysadmin_stepfun_full.log 2>&1 &
```

Each task writes `result.json`, `stdout.txt`, `stderr.txt`, `diff.patch`,
`usage.json`, and `hermes_sessions.jsonl`.
