# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from pathlib import Path
import argparse
import importlib.util
import json
import sys

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "tools" / "real_run_preflight.py"


def _module():
    spec = importlib.util.spec_from_file_location("real_run_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_target_version_is_read_without_importing_package(tmp_path: Path) -> None:
    module = _module()
    package = tmp_path / "lmcache_ascend"
    package.mkdir()
    (package / "__init__.py").write_text(
        'raise RuntimeError("must not import")\nLMCACHE_UPSTREAM_TAG = "v0.4.4"\n',
        encoding="utf-8",
    )
    assert module._target_lmcache_version(tmp_path) == "0.4.4"


def test_manifest_probe_requires_all_three_runtime_roles(tmp_path: Path) -> None:
    module = _module()
    bundle = tmp_path / "lmcache_ascend_bundle"
    bundle.mkdir()
    manifest = {
        "bundle_id": "vllm-hust.lmcache-ascend",
        "components": [
            {"component_id": key, "contracts": [contract]}
            for key, contract in module.REQUIRED_COMPONENTS.items()
        ],
    }
    (bundle / "extension-bundle-v1.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert module._manifest_probe(tmp_path)["ok"] is True
    manifest["components"].pop()
    (bundle / "extension-bundle-v1.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    probe = module._manifest_probe(tmp_path)
    assert probe["ok"] is False
    assert probe["missing_roles"] == ["telemetry"]


def test_probe_parser_ignores_runtime_logs_before_final_json() -> None:
    module = _module()
    assert module._last_json_object('INFO initialized\n{"ok": true}\n') == {"ok": True}
    assert module._last_json_object("INFO only\n") is None


def test_runtime_sources_must_match_selected_repositories(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "selected"
    source = root / "package" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    outside = tmp_path / "installed" / "package" / "__init__.py"
    outside.parent.mkdir(parents=True)
    outside.write_text("", encoding="utf-8")
    assert module._source_is_under(str(source), root)
    assert not module._source_is_under(str(outside), root)
    assert not module._source_is_under(None, root)


def test_cli_probe_requires_runnable_service_command(
    monkeypatch, tmp_path: Path
) -> None:
    module = _module()
    monkeypatch.setattr(
        module,
        "_run",
        lambda command, cwd=None, env=None: module.subprocess.CompletedProcess(
            command, 0, stdout="commands: serve\n", stderr=""
        ),
    )
    assert module._cli_probe(Path("/python"), tmp_path, tmp_path)["ok"]
    monkeypatch.setattr(
        module,
        "_run",
        lambda command, cwd=None, env=None: module.subprocess.CompletedProcess(
            command, 1, stdout="", stderr="runtime mismatch"
        ),
    )
    blocked = module._cli_probe(Path("/python"), tmp_path, tmp_path)
    assert not blocked["ok"]
    assert blocked["error"] == "runtime mismatch"


def test_preflight_records_blockers_without_launching(
    monkeypatch, tmp_path: Path
) -> None:
    module = _module()
    project = tmp_path / "project"
    core = tmp_path / "core"
    model = tmp_path / "model"
    for path in (project, core, model):
        path.mkdir()
    package = project / "lmcache_ascend"
    package.mkdir()
    (package / "__init__.py").write_text(
        'LMCACHE_UPSTREAM_TAG = "v0.4.4"\n', encoding="utf-8"
    )
    bundle = project / "lmcache_ascend_bundle"
    bundle.mkdir()
    (bundle / "extension-bundle-v1.json").write_text(
        json.dumps({"bundle_id": "wrong", "components": []}), encoding="utf-8"
    )
    monkeypatch.setattr(
        module,
        "_python_probe",
        lambda _python, project_root, core_root: {
            "probe_ok": True,
            "modules": {name: {"ok": False} for name in module.REQUIRED_IMPORTS},
            "versions": {"lmcache": None},
        },
    )
    monkeypatch.setattr(
        module,
        "_npu_probe",
        lambda expected: {
            "ok": True,
            "expected_devices": expected,
            "device_ids": list(range(expected)),
        },
    )
    monkeypatch.setattr(
        module, "_ports_probe", lambda ports: {"ok": True, "ports": ports}
    )
    monkeypatch.setattr(module, "_git_revision", lambda path: "a" * 40)
    monkeypatch.setattr(
        module,
        "_cli_probe",
        lambda python, project_root, core_root: {"ok": True},
    )
    output = tmp_path / "evidence"
    args = argparse.Namespace(
        project_root=project,
        core_root=core,
        model=model,
        output_dir=output,
        python=Path(__file__),
        expected_devices=8,
        ports=[8000, 8100],
    )
    record = module.build_record(args)
    assert record["provenance_label"] == "preflight-only"
    assert record["launch_performed"] is False
    assert record["ready_for_real_online"] is False
    assert json.loads((output / "preflight.json").read_text()) == record


def test_existing_output_directory_is_never_overwritten(
    monkeypatch, tmp_path: Path
) -> None:
    module = _module()
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_text("owned", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_python_probe",
        lambda _python, project_root, core_root: {"probe_ok": False},
    )
    monkeypatch.setattr(
        module,
        "_cli_probe",
        lambda python, project_root, core_root: {"ok": False},
    )
    monkeypatch.setattr(module, "_npu_probe", lambda expected: {"ok": False})
    monkeypatch.setattr(module, "_ports_probe", lambda ports: {"ok": False})
    monkeypatch.setattr(module, "_git_revision", lambda path: None)
    args = argparse.Namespace(
        project_root=tmp_path,
        core_root=tmp_path,
        model=tmp_path,
        output_dir=output,
        python=Path(__file__),
        expected_devices=8,
        ports=[8000],
    )
    record = module.build_record(args)
    output_check = next(
        check for check in record["checks"] if check["name"] == "output_directory_new"
    )
    assert output_check["passed"] is False
    assert sentinel.read_text(encoding="utf-8") == "owned"
    assert not (output / "preflight.json").exists()


def test_virtual_environment_interpreter_path_is_not_resolved(
    monkeypatch, tmp_path: Path
) -> None:
    module = _module()
    real_python = tmp_path / "base-python"
    real_python.write_text("", encoding="utf-8")
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    venv_python = venv / "python"
    venv_python.symlink_to(real_python)
    observed: list[Path] = []

    def python_probe(path: Path, project_root: Path, core_root: Path):
        observed.append(path)
        return {"probe_ok": False}

    monkeypatch.setattr(module, "_python_probe", python_probe)
    monkeypatch.setattr(
        module,
        "_cli_probe",
        lambda python, project_root, core_root: {"ok": False},
    )
    monkeypatch.setattr(module, "_npu_probe", lambda expected: {"ok": False})
    monkeypatch.setattr(module, "_ports_probe", lambda ports: {"ok": False})
    monkeypatch.setattr(module, "_git_revision", lambda path: None)
    args = argparse.Namespace(
        project_root=tmp_path,
        core_root=tmp_path,
        model=tmp_path,
        output_dir=tmp_path / "evidence",
        python=venv_python,
        expected_devices=8,
        ports=[8000],
    )
    module.build_record(args)
    assert observed == [venv_python.absolute()]
