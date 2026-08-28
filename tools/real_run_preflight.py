#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed readiness check for matched LMCache-Ascend connector runs.

This command launches no service and produces no performance result. It writes a
small provenance record that a later real-online runner can consume.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import ast
import json
import os
import re
import socket
import subprocess
import sys

REQUIRED_COMPONENTS = {
    "scheduler": "vllm.kv_connector.scheduler.v1",
    "worker": "vllm.kv_connector.worker.v1",
    "telemetry": "vllm.kv_connector.telemetry.v1",
}
REQUIRED_IMPORTS = (
    "vllm",
    "vllm.engine.arg_utils",
    "lmcache",
    "lmcache_ascend",
    "torch_npu",
)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    evidence: Any


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _git_revision(path: Path) -> str | None:
    result = _run(["git", "rev-parse", "HEAD"], cwd=path)
    return result.stdout.strip() if result.returncode == 0 else None


def _target_lmcache_version(project_root: Path) -> str | None:
    source = project_root / "lmcache_ascend" / "__init__.py"
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "LMCACHE_UPSTREAM_TAG"
                for target in node.targets
            ):
                value = ast.literal_eval(node.value)
                return value.removeprefix("v") if isinstance(value, str) else None
    return None


def _last_json_object(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _probe_environment(project_root: Path, core_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    roots = [str(core_root), str(project_root)]
    if existing:
        roots.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def _python_probe(python: Path, project_root: Path, core_root: Path) -> dict[str, Any]:
    probe = """
import importlib
import importlib.metadata
import json

modules = {}
for name in (
    "vllm",
    "vllm.engine.arg_utils",
    "lmcache",
    "lmcache_ascend",
    "torch_npu",
):
    try:
        module = importlib.import_module(name)
        modules[name] = {"ok": True, "file": getattr(module, "__file__", None)}
    except Exception as exc:
        modules[name] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
versions = {}
for distribution in ("vllm", "lmcache", "lmcache-ascend", "torch-npu"):
    try:
        versions[distribution] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        versions[distribution] = None
print(json.dumps({"modules": modules, "versions": versions}, sort_keys=True))
"""
    result = _run(
        [str(python), "-c", probe],
        cwd=core_root,
        env=_probe_environment(project_root, core_root),
    )
    if result.returncode != 0:
        return {
            "probe_ok": False,
            "error": result.stderr.strip() or result.stdout.strip(),
        }
    payload = _last_json_object(result.stdout)
    if payload is None:
        return {"probe_ok": False, "error": "probe returned non-JSON output"}
    payload["probe_ok"] = True
    return payload


def _cli_probe(python: Path, project_root: Path, core_root: Path) -> dict[str, Any]:
    result = _run(
        [str(python), "-m", "vllm.entrypoints.cli.main", "--help"],
        cwd=core_root,
        env=_probe_environment(project_root, core_root),
    )
    return {
        "ok": result.returncode == 0 and "serve" in result.stdout,
        "returncode": result.returncode,
        "error": result.stderr.strip() or None,
    }


def _source_is_under(source: str | None, root: Path) -> bool:
    if source is None:
        return False
    try:
        Path(source).resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _manifest_probe(project_root: Path) -> dict[str, Any]:
    path = project_root / "lmcache_ascend_bundle" / "extension-bundle-v1.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "path": str(path), "error": str(exc)}
    components = {
        item.get("component_id"): set(item.get("contracts", []))
        for item in payload.get("components", [])
    }
    missing = [
        component
        for component, contract in REQUIRED_COMPONENTS.items()
        if contract not in components.get(component, set())
    ]
    return {
        "ok": payload.get("bundle_id") == "vllm-hust.lmcache-ascend" and not missing,
        "path": str(path.resolve()),
        "bundle_id": payload.get("bundle_id"),
        "missing_roles": missing,
    }


def _npu_probe(expected_devices: int) -> dict[str, Any]:
    result = _run(["npu-smi", "info", "-l"])
    ids = sorted(
        {int(value) for value in re.findall(r"NPU ID\s*:\s*(\d+)", result.stdout)}
    )
    return {
        "ok": result.returncode == 0 and len(ids) >= expected_devices,
        "expected_devices": expected_devices,
        "device_ids": ids,
        "error": result.stderr.strip() if result.returncode else None,
    }


def _ports_probe(ports: list[int]) -> dict[str, Any]:
    unavailable: list[int] = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                candidate.bind(("127.0.0.1", port))
            except OSError:
                unavailable.append(port)
    return {"ok": not unavailable, "ports": ports, "unavailable": unavailable}


def build_record(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve()
    core_root = args.core_root.resolve()
    model = args.model.resolve()
    output_dir = args.output_dir.resolve()
    # Preserve a virtual environment's interpreter path. Resolving the symlink
    # to its base interpreter would silently discard pyvenv.cfg discovery.
    python = args.python.absolute()
    output_is_new = not output_dir.exists()
    python_probe = (
        _python_probe(python, project_root, core_root)
        if python.is_file()
        else {
            "probe_ok": False,
            "error": "python executable does not exist",
        }
    )
    cli_probe = (
        _cli_probe(python, project_root, core_root)
        if python.is_file() and project_root.is_dir() and core_root.is_dir()
        else {"ok": False, "error": "python or source root is missing"}
    )
    target_version = _target_lmcache_version(project_root)
    installed_version = python_probe.get("versions", {}).get("lmcache")
    imports = python_probe.get("modules", {})
    manifest_probe = _manifest_probe(project_root)
    npu_probe = _npu_probe(args.expected_devices)
    ports_probe = _ports_probe(args.ports)
    project_revision = _git_revision(project_root)
    core_revision = _git_revision(core_root)
    checks = [
        Check("project_root", project_root.is_dir(), str(project_root)),
        Check("core_root", core_root.is_dir(), str(core_root)),
        Check("model", model.is_dir(), str(model)),
        Check("output_directory_new", output_is_new, str(output_dir)),
        Check("python", python.is_file(), str(python)),
        Check(
            "required_imports",
            python_probe.get("probe_ok", False)
            and all(imports.get(name, {}).get("ok") for name in REQUIRED_IMPORTS),
            python_probe,
        ),
        Check(
            "vllm_source",
            _source_is_under(imports.get("vllm", {}).get("file"), core_root),
            {
                "required_root": str(core_root),
                "imported_file": imports.get("vllm", {}).get("file"),
            },
        ),
        Check(
            "lmcache_ascend_source",
            _source_is_under(
                imports.get("lmcache_ascend", {}).get("file"), project_root
            ),
            {
                "required_root": str(project_root),
                "imported_file": imports.get("lmcache_ascend", {}).get("file"),
            },
        ),
        Check("vllm_cli", cli_probe["ok"], cli_probe),
        Check(
            "lmcache_version",
            target_version is not None and installed_version == target_version,
            {"required": target_version, "installed": installed_version},
        ),
        Check("bundle_manifest", manifest_probe["ok"], manifest_probe),
        Check("npu_inventory", npu_probe["ok"], npu_probe),
        Check("ports", ports_probe["ok"], ports_probe),
        Check(
            "revisions",
            project_revision is not None and core_revision is not None,
            {
                "lmcache_ascend": project_revision,
                "vllm": core_revision,
            },
        ),
    ]
    record = {
        "provenance_label": "preflight-only",
        "ready_for_real_online": all(check.passed for check in checks),
        "launch_performed": False,
        "checks": [asdict(check) for check in checks],
        "required_matrix": ["legacy", "typed", "rollback-to-legacy"],
        "configuration_invariants": {
            "legacy": "kv_connector_module_path only; no extension manifest",
            "typed": "typed selection plus admitted split-role manifest",
            "rollback-to-legacy": (
                "fresh process; legacy config; no typed selection or manifest"
            ),
        },
    }
    if output_is_new:
        output_dir.mkdir(parents=True, mode=0o700)
        (output_dir / "preflight.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--expected-devices", type=int, default=8)
    parser.add_argument("--ports", type=int, nargs="+", default=[8000, 8100])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    record = build_record(parse_args(argv))
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["ready_for_real_online"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
