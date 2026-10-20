#!/usr/bin/env python3
"""Command-line helper for the FL Platform repository.

Usage:
    python skills/skill.py list
    python skills/skill.py specs
    python skills/skill.py config
    python skills/skill.py check
    python skills/skill.py lint
    python skills/skill.py test
    python skills/skill.py dashboard [options]
    python skills/skill.py server [options]
    python skills/skill.py client [options]
    python skills/skill.py compose [config|up|down|ps]
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = ROOT / "specs"
CONFIG_FILE = ROOT / "config.yaml"

SKILLS = {
    "list": "List available FL Platform commands",
    "specs": "List Markdown specs and flag documents over 60 lines",
    "config": "Validate and summarize the experiment configuration",
    "check": "Check runtime entry points, modules, and configuration",
    "lint": "Parse all repository Python files for syntax errors",
    "test": "Discover and run unittest tests",
    "dashboard": "Launch the dashboard through run.py",
    "server": "Launch the FL server through run.py",
    "client": "Launch an FL client through run.py",
    "compose": "Validate Compose configuration or manage services",
}


def cmd_list(_args: list[str]) -> None:
    print("=== FL Platform commands ===")
    for name, description in SKILLS.items():
        print(f"  {name:<12} {description}")


def cmd_specs(_args: list[str]) -> None:
    print("=== specs/ status ===")
    files = sorted(SPECS_DIR.rglob("*.md")) if SPECS_DIR.exists() else []
    if not files:
        print("  No Markdown documents found.")
        return
    for file in files:
        count = len(file.read_text(encoding="utf-8").splitlines())
        warning = " [WARN: over 60 lines]" if count > 60 else ""
        print(f"  {file.relative_to(ROOT).as_posix():<40} {count:>3} lines{warning}")


def _load_config():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install dependencies with `pip install -r requirements.txt`.") from exc
    if not CONFIG_FILE.exists():
        raise RuntimeError("config.yaml was not found.")
    try:
        data = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Could not parse config.yaml: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("The top-level config.yaml value must be a YAML object.")
    return data


def cmd_config(_args: list[str]) -> None:
    config = _load_config()
    required = {"data", "server", "client", "system", "tracking"}
    missing = sorted(required - config.keys())
    if missing:
        raise RuntimeError("Missing required configuration groups: " + ", ".join(missing))
    invalid = sorted(key for key in required if not isinstance(config[key], dict))
    if invalid:
        raise RuntimeError("Configuration groups must be YAML objects: " + ", ".join(invalid))
    data, server, client = config["data"], config["server"], config["client"]
    print("=== FL experiment configuration ===")
    print(f"  task       : {config.get('task_id', '(unset)')}")
    print(f"  seed       : {config.get('seed', '(unset)')}")
    print(f"  model      : {config.get('model', '(unset)')}")
    print(f"  dataset    : {data.get('dataset', '(unset)')}")
    print(f"  clients    : {data.get('num_of_clients', '(unset)')}")
    print(f"  split      : {data.get('split_type', '(unset)')}")
    print(f"  rounds     : {server.get('rounds', '(unset)')}")
    print(f"  strategy   : {server.get('aggregation_strategy', '(unset)')}")
    print(f"  local train: {client.get('local_epoch', '(unset)')} epochs, "
          f"{client.get('optimizer', '(unset)')} lr={client.get('lr', '(unset)')}")


def cmd_check(_args: list[str]) -> None:
    print("=== Repository check ===")
    required_files = ["run.py", "server.py", "client.py", "config.yaml", "docker-compose.yml"]
    missing = [name for name in required_files if not (ROOT / name).is_file()]
    if missing:
        print("  [ERROR] Missing required files: " + ", ".join(missing))
    else:
        print("  [OK] Entry points and configuration files exist")
    implementation_files = [
        "fl_platform/dashboard/app.py",
        "fl_platform/coordinator.py",
        "fl_platform/deploy/fl_server.py",
        "fl_platform/deploy/fl_client.py",
        "fl_platform/security.py",
        "fl_platform/models/model_registry.py",
        "fl_platform/tracking.py",
    ]
    absent = [name for name in implementation_files if not (ROOT / name).is_file()]
    if absent:
        print("  [ERROR] Missing runtime modules: " + ", ".join(absent))
    else:
        print("  [OK] Dashboard, coordinator, server, client, security, model, and tracking modules exist")
    try:
        _load_config()
        print("  [OK] config.yaml parses and required groups exist")
    except RuntimeError as exc:
        print(f"  [ERROR] {exc}")
        raise SystemExit(1) from exc
    if missing or absent:
        raise SystemExit(1)


def cmd_lint(_args: list[str]) -> None:
    files = [ROOT / name for name in ("run.py", "server.py", "client.py")]
    for directory in ("skills", "tests", "fl_platform", "scripts"):
        path = ROOT / directory
        if path.is_dir():
            files.extend(path.rglob("*.py"))
    files = sorted(set(path for path in files if path.is_file()))
    errors = []
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    if errors:
        print("[ERROR] Python syntax errors:")
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"[OK] Python syntax check passed ({len(files)} files)")


def cmd_test(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="skill.py test")
    parser.add_argument("--start-dir", default="tests", help="unittest discovery directory")
    opts = parser.parse_args(args)
    start_dir = ROOT / opts.start_dir
    if not start_dir.exists():
        print(f"Test directory not found: {opts.start_dir}")
        print("Skipping tests. Add project tests and run this command again.")
        return
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(start_dir)], cwd=ROOT
    )
    raise SystemExit(result.returncode)


def cmd_launch(mode: str, args: list[str]) -> None:
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), mode, *args], cwd=ROOT)
    raise SystemExit(result.returncode)


def cmd_compose(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="skill.py compose")
    parser.add_argument("action", nargs="?", default="config", choices=["config", "up", "down", "ps"])
    parser.add_argument("--build", action="store_true", help="build images when running up")
    opts = parser.parse_args(args)
    command = ["docker", "compose"]
    if opts.action == "config":
        command.append("config")
    elif opts.action == "up":
        command.append("up")
        if opts.build:
            command.append("--build")
    elif opts.action == "down":
        command.append("down")
    else:
        command.append("ps")
    result = subprocess.run(command, cwd=ROOT)
    raise SystemExit(result.returncode)


DISPATCH = {
    "list": cmd_list,
    "specs": cmd_specs,
    "config": cmd_config,
    "check": cmd_check,
    "lint": cmd_lint,
    "test": cmd_test,
    "dashboard": lambda args: cmd_launch("dashboard", args),
    "server": lambda args: cmd_launch("server", args),
    "client": lambda args: cmd_launch("client", args),
    "compose": cmd_compose,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in DISPATCH:
        print(__doc__)
        print("Available commands: " + ", ".join(DISPATCH))
        raise SystemExit(1)
    try:
        DISPATCH[sys.argv[1]](sys.argv[2:])
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
