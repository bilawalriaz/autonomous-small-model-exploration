#!/usr/bin/env python3
"""Generate a broad sysadmin trajectory task set for Hermes collection."""

from __future__ import annotations

import json
from pathlib import Path


def verifier_report(path: str, terms: list[str]) -> str:
    checks = "\n".join([f"assert {term!r} in text.lower()" for term in terms])
    return (
        "python3 - <<'PY'\n"
        f"from pathlib import Path\np=Path({path!r})\nassert p.exists(), f'missing {path}'\n"
        "text=p.read_text().lower()\n"
        f"assert len(text) > 400, len(text)\n{checks}\nPY"
    )


def task(task_id, family, prompt, files=None, verifiers=None):
    return {
        "id": task_id,
        "family": family,
        "prompt": prompt,
        "files": files or {},
        "verifiers": verifiers or [],
    }


tasks = []

host_audits = [
    ("listening_ports", "listening ports, owning processes, and exposure risk", ["commands", "finding", "evidence"]),
    ("users_groups", "local users, sudo-capable groups, and stale-login risk", ["commands", "finding", "evidence"]),
    ("auth_logs", "recent authentication failures using available logs/journal output", ["commands", "finding", "evidence"]),
    ("firewall", "firewall status, default policy, and safe allowlist recommendations", ["commands", "finding", "evidence"]),
    ("updates", "package update/reboot-needed posture without changing packages", ["commands", "finding", "evidence"]),
    ("services", "enabled/running services and suspicious exposure", ["commands", "finding", "evidence"]),
    ("ssh", "SSH daemon configuration and hardening opportunities", ["commands", "finding", "evidence"]),
    ("disk", "disk usage, large directories, and cleanup candidates", ["commands", "finding", "evidence"]),
    ("docker_host", "Docker daemon status, group membership, and container exposure risk", ["commands", "finding", "evidence"]),
    ("kernel", "kernel/security feature posture including AppArmor/seccomp where visible", ["commands", "finding", "evidence"]),
]
for i, (slug, focus, terms) in enumerate(host_audits, 1):
    tasks.append(
        task(
            f"sys_host_audit_{i:03d}_{slug}",
            "host_security_audit",
            "Perform a read-only sysadmin/security audit of this lenovo box focused on "
            f"{focus}. Do not change host state. Write `audit_report.md` with: commands run, findings, severity, evidence snippets, and concrete next actions.",
            verifiers=[{"name": "audit report", "cmd": verifier_report("audit_report.md", terms)}],
        )
    )

compose_base = {
    "app.py": "from http.server import BaseHTTPRequestHandler, HTTPServer\nimport os\nclass H(BaseHTTPRequestHandler):\n    def do_GET(self):\n        body=f\"ok {os.getenv('APP_NAME','app')}\".encode()\n        self.send_response(200); self.end_headers(); self.wfile.write(body)\nHTTPServer(('0.0.0.0', int(os.getenv('PORT','8080'))), H).serve_forever()\n",
    "Dockerfile": "FROM python:3.14-slim\nWORKDIR /app\nCOPY app.py .\nCMD [\"python\", \"app.py\"]\n",
}
for i in range(1, 21):
    port = 8100 + i
    files = dict(compose_base)
    files.update(
        {
            "compose.yaml": f"services:\n  web:\n    build: .\n    ports:\n      - \"{port}:8080\"\n    environment:\n      APP_NAME: demo{i}\n",
            "check_compose.py": "import subprocess, yaml, pathlib\nsubprocess.check_call(['docker','compose','config'], stdout=subprocess.DEVNULL)\nd=yaml.safe_load(pathlib.Path('compose.yaml').read_text())\nsvc=d['services']['web']\nassert 'healthcheck' in svc\nassert svc.get('restart') in ('unless-stopped','always')\nassert 'read_only' in svc and svc['read_only'] is True\nassert 'security_opt' in svc\n",
        }
    )
    tasks.append(
        task(
            f"sys_compose_harden_{i:03d}",
            "docker_compose_deployment",
            "Harden this Docker Compose deployment for a small internal service. Add a healthcheck, sensible restart policy, read-only container filesystem where feasible, and basic security options. Validate statically with `docker compose config`; do not run `docker compose up`, `build`, `pull`, or start containers for this task.",
            files,
            [{"name": "compose hardening", "cmd": "python3 check_compose.py"}],
        )
    )

for i in range(1, 16):
    files = {
        "important/app.conf": f"port={9000+i}\nmode=prod\n",
        "important/data.txt": "alpha\nbeta\ngamma\n",
        "check_backup.py": "import os, subprocess, tarfile, pathlib, tempfile\nsubprocess.check_call(['bash','backup.sh'])\narchives=list(pathlib.Path('backups').glob('backup-*.tar.gz'))\nassert archives, 'no archive'\nwith tarfile.open(archives[-1]) as t:\n    names=set(t.getnames())\nassert any(n.endswith('important/app.conf') for n in names), names\nassert any(n.endswith('important/data.txt') for n in names), names\n",
    }
    tasks.append(
        task(
            f"sys_backup_restore_{i:03d}",
            "backup_restore",
            "Create a robust `backup.sh` for the `important/` directory. It should create timestamped compressed archives under `backups/`, refuse to continue on errors, and print what it wrote. Avoid absolute machine-specific paths.",
            files,
            [{"name": "backup check", "cmd": "python3 check_backup.py"}],
        )
    )

log_samples = [
    ("ssh brute force", "sshd[100]: Failed password for invalid user admin from 203.0.113.4 port 51234 ssh2"),
    ("oom kill", "kernel: Out of memory: Killed process 4444 (python3) total-vm:123456kB"),
    ("nginx 5xx", "nginx: 10.0.0.4 - - \"GET /api HTTP/1.1\" 502 173"),
    ("disk full", "app: write failed: No space left on device"),
    ("permission denied", "backup: cannot open /srv/data: Permission denied"),
]
for i in range(1, 21):
    label, line = log_samples[(i - 1) % len(log_samples)]
    files = {
        "incident.log": "\n".join([line, "app: normal heartbeat ok", line.replace("203.0.113.4", "198.51.100.9")]) + "\n",
        "check_incident.py": "from pathlib import Path\np=Path('incident_report.md')\nassert p.exists()\ns=p.read_text().lower()\nfor term in ['summary','evidence','impact','next actions']:\n    assert term in s, term\nassert len(s) > 300\n",
    }
    tasks.append(
        task(
            f"sys_incident_triage_{i:03d}",
            "incident_triage",
            f"Triage `incident.log` for a likely {label} incident. Write `incident_report.md` with summary, evidence, impact, immediate containment, and follow-up prevention. Do not invent facts not present in the log.",
            files,
            [{"name": "incident report", "cmd": "python3 check_incident.py"}],
        )
    )

for i in range(1, 16):
    files = {
        "myapp.service": "[Unit]\nDescription=Example app\n[Service]\nExecStart=/usr/bin/python3 /opt/myapp/app.py\nUser=root\n[Install]\nWantedBy=multi-user.target\n",
        "check_unit.py": "from pathlib import Path\ns=Path('myapp.service').read_text()\nfor term in ['NoNewPrivileges=true','PrivateTmp=true','ProtectSystem=strict','Restart=on-failure']:\n    assert term in s, term\nassert 'User=root' not in s\n",
    }
    tasks.append(
        task(
            f"sys_systemd_harden_{i:03d}",
            "systemd_hardening",
            "Harden `myapp.service` for a long-running Python service. Avoid root, add restart behavior and practical sandboxing directives without breaking ExecStart. Keep the unit file syntactically plausible.",
            files,
            [{"name": "unit hardening", "cmd": "python3 check_unit.py"}],
        )
    )

for i in range(1, 16):
    files = {
        "nginx.conf": "server {\n    listen 80;\n    server_name example.local;\n    location / { proxy_pass http://127.0.0.1:9000; }\n}\n",
        "check_nginx.py": "from pathlib import Path\ns=Path('nginx.conf').read_text().lower()\nfor term in ['proxy_set_header host','x-forwarded-for','client_max_body_size','location /health']:\n    assert term in s, term\n",
    }
    tasks.append(
        task(
            f"sys_reverse_proxy_{i:03d}",
            "reverse_proxy_config",
            "Improve this nginx reverse-proxy config for a small internal app. Preserve the upstream, add standard proxy headers, a lightweight `/health` response, and a reasonable request body limit.",
            files,
            [{"name": "nginx config check", "cmd": "python3 check_nginx.py"}],
        )
    )

for i in range(1, 16):
    files = {
        "deploy.sh": "#!/usr/bin/env bash\nset -e\ncd /srv/example\ndocker compose pull\ndocker compose up -d\n",
        "check_deploy.py": "from pathlib import Path\ns=Path('deploy.sh').read_text()\nfor term in ['set -euo pipefail','docker compose config','trap','rollback']:\n    assert term in s, term\n",
    }
    tasks.append(
        task(
            f"sys_deploy_script_{i:03d}",
            "deployment_automation",
            "Make `deploy.sh` safer for Docker Compose deployment. Add validation, logging, error trapping, and a simple rollback path. Keep it understandable and avoid requiring external services.",
            files,
            [{"name": "deploy script check", "cmd": "python3 check_deploy.py"}],
        )
    )

for i in range(1, 16):
    files = {
        "sample_ss.txt": "LISTEN 0 4096 0.0.0.0:22 0.0.0.0:* users:((\"sshd\",pid=777,fd=3))\nLISTEN 0 4096 127.0.0.1:5432 0.0.0.0:* users:((\"postgres\",pid=888,fd=5))\nLISTEN 0 4096 0.0.0.0:8080 0.0.0.0:* users:((\"dev-server\",pid=999,fd=8))\n",
        "check_ports.py": "import json, subprocess, sys\nsubprocess.check_call([sys.executable, 'analyze_ports.py', 'sample_ss.txt', 'ports.json'])\nd=json.load(open('ports.json'))\nassert any(x['port']==22 and x['exposure']=='public' for x in d)\nassert any(x['port']==5432 and x['exposure']=='loopback' for x in d)\nassert any(x['port']==8080 and x['risk'] in ('medium','high') for x in d)\n",
    }
    tasks.append(
        task(
            f"sys_port_analysis_{i:03d}",
            "security_scan_parser",
            "Write `analyze_ports.py` to parse `ss -ltnp`-style output and emit JSON records with port, bind address, process, exposure (`public` or `loopback`), and risk. Treat public dev/admin ports as at least medium risk.",
            files,
            [{"name": "port parser", "cmd": "python3 check_ports.py"}],
        )
    )

out = Path("sysadmin_tasks.json")
out.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
print(f"wrote {len(tasks)} tasks to {out}")
