# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest


def _import_and_patch_vllm_connector():
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")

    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
        LMCacheConnectorV1,
    )

    lmcache_ascend = pytest.importorskip("lmcache_ascend")
    lmcache_ascend._patch_vllm_v1_adapter()
    return LMCacheConnectorV1


def _make_adapter(adapter_mod, *, store_async, kv_role, lmcache_engine):
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.store_async = store_async
    adapter.kv_role = kv_role
    adapter._manager = SimpleNamespace(lmcache_engine=lmcache_engine)
    return adapter


def test_lmcache_connector_delegates_preemptions_after_ascend_patch():
    """Ascend patches the outer vLLM connector to delegate preemptions."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = MagicMock()

    metadata = adapter_mod.LMCacheAscendConnectorMetadata(
        preempted_req_ids={"req-1", "req-2"}
    )
    connector.handle_preemptions(metadata)

    connector._lmcache_engine.handle_preemptions.assert_called_once_with(metadata)


def test_lmcache_connector_preemption_patch_handles_no_inner_impl():
    """The Ascend patch should tolerate inner implementations without a hook."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = object()

    connector.handle_preemptions(MagicMock())


def test_ascend_metadata_preserves_lmcache_fields_and_preemptions():
    """HUST preemption IDs travel with the regular LMCache metadata."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    request = MagicMock()
    base_metadata = adapter_mod.LMCacheConnectorMetadata(requests=[request])

    metadata = adapter_mod._with_preempted_req_ids(
        base_metadata,
        {"req-1", "req-2"},
    )

    assert isinstance(metadata, adapter_mod.LMCacheConnectorMetadata)
    assert metadata.requests == [request]
    assert metadata.preempted_req_ids == {"req-1", "req-2"}


def test_ascend_adapter_drains_pending_stores_for_async_producer():
    """Async non-consumer workers must drain pending stores before reuse."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    lmcache_engine = MagicMock()
    lmcache_engine.wait_for_pending_stores.return_value = {"req-1"}
    adapter = _make_adapter(
        adapter_mod,
        store_async=True,
        kv_role="kv_both",
        lmcache_engine=lmcache_engine,
    )

    metadata = adapter_mod.LMCacheAscendConnectorMetadata(
        preempted_req_ids={"req-1", "req-2"}
    )
    adapter.handle_preemptions(metadata)

    expected_req_ids = {"req-1", "req-2"}
    lmcache_engine.wait_for_pending_stores.assert_called_once_with(expected_req_ids)
    assert lmcache_engine.lookup_unpin.call_count == 2


@pytest.mark.parametrize(
    ("store_async", "kv_role", "has_engine"),
    [
        (False, "kv_both", True),
        (True, "kv_consumer", True),
        (True, "kv_both", False),
    ],
)
def test_ascend_adapter_skips_preemption_drain_when_not_required(
    store_async, kv_role, has_engine
):
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    lmcache_engine = MagicMock() if has_engine else None
    adapter = _make_adapter(
        adapter_mod,
        store_async=store_async,
        kv_role=kv_role,
        lmcache_engine=lmcache_engine,
    )

    metadata = adapter_mod.LMCacheAscendConnectorMetadata(
        preempted_req_ids={"req-1"}
    )
    adapter.handle_preemptions(metadata)

    if has_engine:
        lmcache_engine.wait_for_pending_stores.assert_not_called()


def test_ascend_adapter_keeps_legacy_preemption_ids_compatible():
    """Older vLLM versions may still call the adapter with a request-ID set."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    lmcache_engine = MagicMock()
    adapter = _make_adapter(
        adapter_mod,
        store_async=False,
        kv_role="kv_both",
        lmcache_engine=lmcache_engine,
    )

    adapter.handle_preemptions({"legacy-req"})

    lmcache_engine.lookup_unpin.assert_called_once_with("legacy-req")
