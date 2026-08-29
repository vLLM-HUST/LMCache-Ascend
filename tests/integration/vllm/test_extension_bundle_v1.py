# SPDX-License-Identifier: Apache-2.0
import ast
import importlib
import json
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "lmcache_ascend_bundle" / "extension-bundle-v1.json"
CONNECTOR = (
    ROOT / "lmcache_ascend" / "integration" / "vllm" / "lmcache_ascend_connector_v1.py"
)


def test_manifest_declares_provider_roles_and_permissions() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    components = {item["component_id"]: item for item in payload["components"]}

    assert payload["bundle_id"] == "vllm-hust.lmcache-ascend"
    assert set(components) == {"scheduler", "worker", "telemetry"}
    assert components["telemetry"]["execution_planes"] == ["api"]
    assert components["telemetry"]["permissions"] == []
    assert "device_access" not in components["scheduler"]["permissions"]
    assert "device_access" in components["worker"]["permissions"]


def test_dynamic_connector_accepts_typed_factory_third_argument() -> None:
    tree = ast.parse(CONNECTOR.read_text(encoding="utf-8"))
    connector_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LMCacheAscendConnectorV1Dynamic"
    )
    initializer = next(
        node
        for node in connector_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assert [argument.arg for argument in initializer.args.args][-1] == (
        "kv_cache_config"
    )


def test_telemetry_import_does_not_import_runtime_connector() -> None:
    connector_module = "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
    sys.modules.pop(connector_module, None)
    module = importlib.import_module("lmcache_ascend_bundle.telemetry")

    assert connector_module not in sys.modules
    assert module.LMCacheAscendTelemetryProvider.build_kv_connector_stats() is None


def test_manifest_passes_vllm_hust_bundle_schema_when_available() -> None:
    manifest_module = pytest.importorskip("vllm.plugins.manifest")
    bundle = manifest_module.load_extension_bundle_manifest(MANIFEST)
    assert len(bundle.components) == 3


def test_manifest_requires_explicit_permission_admission_when_available() -> None:
    contracts = pytest.importorskip("vllm.plugins.contracts")
    startup = pytest.importorskip("vllm.plugins.startup")

    with pytest.raises(startup.ExtensionBundleAdmissionError):
        startup.resolve_extension_startup((MANIFEST,), allowed_permissions=())
    resolution = startup.resolve_extension_startup(
        (MANIFEST,),
        allowed_permissions=tuple(contracts.ComponentPermission),
    )
    assert len(resolution.snapshot.components) == 3


def test_lightweight_bundle_and_manifest_are_wheel_package_data() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = project["tool"]["setuptools"]["package-data"]

    assert package_data["lmcache_ascend_bundle"] == ["extension-bundle-v1.json"]
    assert project["project"]["entry-points"]["vllm.extension_bundles"] == {
        "vllm-hust.lmcache-ascend": "lmcache_ascend_bundle"
    }
