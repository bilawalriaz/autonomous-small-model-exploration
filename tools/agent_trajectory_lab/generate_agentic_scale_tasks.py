#!/usr/bin/env python3
"""Generate parameterized agentic trajectory tasks at scale.

The corpus is intentionally verifier-backed and split into seven behavior
families:

1. tool-format tasks
2. file-operation tasks
3. shell/repo-inspection tasks
4. failure-recovery tasks
5. summarisation/state-compaction tasks
6. choose-the-right-tool routing tasks
7. multi-step mini-agent tasks

By default this writes 700 templates and 7,000 task instances, with 100
templates per family and 10 variants per template.
"""

from __future__ import annotations

import argparse
import collections
import json
import textwrap
from pathlib import Path


FAMILIES = [
    "tool_format",
    "file_operations",
    "shell_repo_inspection",
    "failure_recovery",
    "summarisation_state_compaction",
    "tool_routing",
    "multi_step_mini_agent",
]


def dedent(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def task(task_id: str, family: str, template_id: str, variant_id: int, prompt: str, files: dict[str, str], verifiers: list[dict]) -> dict:
    return {
        "id": task_id,
        "family": family,
        "template_id": template_id,
        "variant_id": variant_id,
        "prompt": prompt,
        "files": files,
        "verifiers": verifiers,
    }


def py_verifier(code: str) -> dict:
    return {"name": "task verifier", "cmd": "python3 verify_task.py", "timeout": 120}


def build_tool_format(template_n: int, variant_n: int) -> dict:
    keys = ["action", "target", "priority", "evidence"]
    required = keys[: 2 + (template_n % 3)]
    prompt = (
        "Read `request.txt` and produce `tool_call.json`. The JSON must contain "
        f"exactly these top-level keys in this order: {', '.join(required)}. "
        "Use only facts present in the request. Do not wrap the JSON in markdown."
    )
    request = f"Please inspect service-{variant_n} and restart it only if health is degraded. Priority P{template_n % 4}."
    files = {
        "request.txt": request + "\n",
        "verify_task.py": dedent(
            f"""
            import json
            from pathlib import Path
            path = Path('tool_call.json')
            assert path.exists(), 'missing tool_call.json'
            data = json.loads(path.read_text())
            assert list(data.keys()) == {required!r}, list(data.keys())
            assert 'service-{variant_n}' in json.dumps(data)
            assert 'markdown' not in path.read_text().lower()
            """
        ),
    }
    return task(
        f"agt_tool_format_{template_n:03d}_v{variant_n:02d}",
        "tool_format",
        f"agt_tool_format_{template_n:03d}",
        variant_n,
        prompt,
        files,
        [py_verifier("")],
    )


def build_file_operations(template_n: int, variant_n: int) -> dict:
    service = f"svc_{template_n:03d}_{variant_n:02d}"
    prompt = (
        "Update this small config repository. Rename the service from the old name "
        "to the new name everywhere, add a README changelog entry, and preserve valid JSON/YAML."
    )
    files = {
        "config/service.json": json.dumps({"name": "old-service", "port": 7000 + variant_n, "enabled": True}, indent=2) + "\n",
        "deploy.yaml": f"service: old-service\nreplicas: {1 + template_n % 3}\n",
        "README.md": "# Service\n\nExisting deployment notes.\n",
        "verify_task.py": dedent(
            f"""
            import json
            from pathlib import Path
            cfg = json.loads(Path('config/service.json').read_text())
            assert cfg['name'] == {service!r}
            assert {service!r} in Path('deploy.yaml').read_text()
            readme = Path('README.md').read_text().lower()
            assert 'changelog' in readme and {service!r} in readme
            for path in Path('.').rglob('*'):
                if path.is_file() and path.name != 'verify_task.py':
                    assert 'old-service' not in path.read_text(errors='ignore'), path
            """
        ),
    }
    return task(
        f"agt_file_ops_{template_n:03d}_v{variant_n:02d}",
        "file_operations",
        f"agt_file_ops_{template_n:03d}",
        variant_n,
        f"{prompt} New service name: `{service}`.",
        files,
        [py_verifier("")],
    )


def build_shell_repo_inspection(template_n: int, variant_n: int) -> dict:
    vulnerable = template_n % 2 == 0
    files = {
        "app/main.py": "import os\nSECRET='dev-only'\nprint('hello')\n" if vulnerable else "def main():\n    print('hello')\n",
        "app/utils.py": "def add(a,b): return a+b\n",
        "tests/test_utils.py": "from app.utils import add\n\ndef test_add():\n    assert add(1,2)==3\n",
        "requirements.txt": "flask==3.0.0\npytest==8.0.0\n",
        "verify_task.py": dedent(
            f"""
            import json
            from pathlib import Path
            data = json.loads(Path('repo_report.json').read_text())
            assert data['file_count'] >= 4
            assert 'python_files' in data and data['python_files'] >= 2
            assert data['has_tests'] is True
            assert data['hardcoded_secret_found'] is {str(vulnerable)}
            assert 'commands_run' in data and len(data['commands_run']) >= 2
            """
        ),
    }
    prompt = (
        "Inspect this repository using shell commands. Write `repo_report.json` with "
        "file_count, python_files, has_tests, hardcoded_secret_found, and commands_run. "
        "Do not edit application files."
    )
    return task(
        f"agt_repo_inspect_{template_n:03d}_v{variant_n:02d}",
        "shell_repo_inspection",
        f"agt_repo_inspect_{template_n:03d}",
        variant_n,
        prompt,
        files,
        [py_verifier("")],
    )


def build_failure_recovery(template_n: int, variant_n: int) -> dict:
    mode = template_n % 4
    bug = [
        "return nums[len(nums)//2]",
        "return sum(nums) / len(nums)",
        "return sorted(set(nums))[0]",
        "return ','.join(items)",
    ][mode]
    fixed_assert = [
        "assert median([3,1,2]) == 2\nassert median([4,1,2,3]) == 2.5",
        "assert average([]) == 0\nassert average([2,4]) == 3",
        "assert smallest([3,1,1,2]) == 1",
        "assert csv(['a','b']) == 'a,b'\nassert csv([]) == ''",
    ][mode]
    func = ["median", "average", "smallest", "csv"][mode]
    args = ["nums", "nums", "nums", "items"][mode]
    files = {
        "buggy.py": f"def {func}({args}):\n    {bug}\n",
        "test_buggy.py": f"from buggy import {func}\n\n\ndef test_behavior():\n    {fixed_assert.replace(chr(10), chr(10) + '    ')}\n",
        "verify_task.py": "import subprocess, sys\nsubprocess.check_call([sys.executable, 'pytest.py'])\n",
    }
    prompt = (
        "Fix the failing behavior in `buggy.py`. Run the local tests, diagnose any failure, "
        "and keep the implementation small. Do not edit `test_buggy.py`."
    )
    return task(
        f"agt_failure_recovery_{template_n:03d}_v{variant_n:02d}",
        "failure_recovery",
        f"agt_failure_recovery_{template_n:03d}",
        variant_n,
        prompt,
        files,
        [py_verifier("")],
    )


def build_summarisation(template_n: int, variant_n: int) -> dict:
    incident = f"INC-{template_n:03d}-{variant_n:02d}"
    notes = "\n".join(
        [
            f"{incident} status: investigating latency spike",
            "Fact: database CPU reached 91 percent for 7 minutes",
            "Fact: no data loss observed",
            "Decision: defer index migration until off-peak",
            "Next: add dashboard annotation and notify support",
        ]
    )
    files = {
        "session_notes.md": notes + "\n",
        "verify_task.py": dedent(
            f"""
            from pathlib import Path
            text = Path('state_summary.md').read_text().lower()
            for term in ['{incident.lower()}', 'decisions', 'next actions', 'open questions']:
                assert term in text, term
            assert 'current state' in text or 'status' in text
            assert 'data loss observed' in text
            assert 'resolved' not in text
            assert len(text.split()) <= 180
            """
        ),
    }
    prompt = (
        "Compress `session_notes.md` into `state_summary.md` for a future agent handoff. "
        "Preserve facts, decisions, next actions, and open questions. Include a `Current State` "
        "or `Status` section. Do not claim resolution."
    )
    return task(
        f"agt_state_compact_{template_n:03d}_v{variant_n:02d}",
        "summarisation_state_compaction",
        f"agt_state_compact_{template_n:03d}",
        variant_n,
        prompt,
        files,
        [py_verifier("")],
    )


def build_tool_routing(template_n: int, variant_n: int) -> dict:
    route = ["read_file", "terminal", "patch", "search_files"][template_n % 4]
    files = {
        "routing_request.txt": f"Need action for variant {variant_n}: choose the safest tool. Expected route: {route}.\n",
        "verify_task.py": dedent(
            f"""
            import json
            from pathlib import Path
            data = json.loads(Path('route_decision.json').read_text())
            assert data['chosen_tool'] == {route!r}, data
            assert data['why'] and len(data['why']) > 20
            assert data['unsafe_alternatives']
            """
        ),
    }
    prompt = (
        "Read `routing_request.txt` and write `route_decision.json` with chosen_tool, why, "
        "and unsafe_alternatives. Do not perform the routed action; this task is only about choosing the right tool."
    )
    return task(
        f"agt_tool_routing_{template_n:03d}_v{variant_n:02d}",
        "tool_routing",
        f"agt_tool_routing_{template_n:03d}",
        variant_n,
        prompt,
        files,
        [py_verifier("")],
    )


def build_mini_agent(template_n: int, variant_n: int) -> dict:
    port = 8800 + variant_n
    files = {
        "service.py": f"PORT={port}\nDEBUG=True\n\ndef status():\n    return 'ok'\n",
        "deploy.toml": f"name='mini-{template_n:03d}'\nport={port}\ndebug=true\n",
        "README.md": "# Mini Service\n",
        "verify_task.py": dedent(
            """
            from pathlib import Path
            svc = Path('service.py').read_text()
            dep = Path('deploy.toml').read_text().lower()
            readme = Path('README.md').read_text().lower()
            assert 'DEBUG=False' in svc
            assert 'debug=false' in dep
            assert 'runbook' in readme and 'rollback' in readme and 'verify' in readme
            assert Path('ops_checklist.md').exists()
            assert 'health' in Path('ops_checklist.md').read_text().lower()
            """
        ),
    }
    prompt = (
        "Act as a mini deployment agent. Inspect the files, disable debug mode consistently, "
        "add an `ops_checklist.md`, and update the README with runbook, verify, and rollback notes. "
        "Run verification before finishing."
    )
    return task(
        f"agt_mini_agent_{template_n:03d}_v{variant_n:02d}",
        "multi_step_mini_agent",
        f"agt_mini_agent_{template_n:03d}",
        variant_n,
        prompt,
        files,
        [py_verifier("")],
    )


BUILDERS = {
    "tool_format": build_tool_format,
    "file_operations": build_file_operations,
    "shell_repo_inspection": build_shell_repo_inspection,
    "failure_recovery": build_failure_recovery,
    "summarisation_state_compaction": build_summarisation,
    "tool_routing": build_tool_routing,
    "multi_step_mini_agent": build_mini_agent,
}


def write_shards(tasks: list[dict], out_dir: Path, prefix: str, shards: int) -> list[Path]:
    paths = []
    for shard in range(shards):
        shard_tasks = tasks[shard::shards]
        path = out_dir / f"{prefix}_shard_{shard:02d}.json"
        path.write_text(json.dumps(shard_tasks, indent=2), encoding="utf-8")
        paths.append(path)
    return paths


def interleave_by_family(tasks: list[dict]) -> list[dict]:
    grouped = collections.defaultdict(list)
    for item in tasks:
        grouped[item["family"]].append(item)

    ordered = []
    while any(grouped.values()):
        for family in FAMILIES:
            if grouped[family]:
                ordered.append(grouped[family].pop(0))
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--templates-per-family", type=int, default=100)
    parser.add_argument("--variants-per-template", type=int, default=10)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--prefix", default="agentic_scale_tasks")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    templates = []
    instances = []
    for family in FAMILIES:
        builder = BUILDERS[family]
        for template_n in range(1, args.templates_per_family + 1):
            template_id = f"agt_{family}_{template_n:03d}"
            templates.append({"template_id": template_id, "family": family, "variants": args.variants_per_template})
            for variant_n in range(1, args.variants_per_template + 1):
                instances.append(builder(template_n, variant_n))

    template_path = out_dir / f"{args.prefix}_templates.json"
    instance_path = out_dir / f"{args.prefix}.json"
    balanced_path = out_dir / f"{args.prefix}_balanced.json"
    summary_path = out_dir / f"{args.prefix}.summary.json"
    template_path.write_text(json.dumps(templates, indent=2), encoding="utf-8")
    instance_path.write_text(json.dumps(instances, indent=2), encoding="utf-8")
    balanced = interleave_by_family(instances)
    balanced_path.write_text(json.dumps(balanced, indent=2), encoding="utf-8")
    shard_paths = write_shards(instances, out_dir, args.prefix, args.shards)
    balanced_shard_paths = write_shards(balanced, out_dir, f"{args.prefix}_balanced", args.shards)

    summary = {
        "families": FAMILIES,
        "templates": len(templates),
        "instances": len(instances),
        "templates_per_family": args.templates_per_family,
        "variants_per_template": args.variants_per_template,
        "shards": len(shard_paths),
        "shard_paths": [str(p) for p in shard_paths],
        "balanced_shard_paths": [str(p) for p in balanced_shard_paths],
        "instance_path": str(instance_path),
        "balanced_instance_path": str(balanced_path),
        "template_path": str(template_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
