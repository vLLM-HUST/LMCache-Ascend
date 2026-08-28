# SPDX-License-Identifier: Apache-2.0
"""API-plane telemetry contract without importing LMCache or NPU code."""

from typing import Any


class LMCacheAscendTelemetryProvider:
    """No-op codec matching the current LMCache connector telemetry surface."""

    @classmethod
    def build_kv_connector_stats(cls, data: dict[str, Any] | None = None) -> None:
        return None

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: Any,
        metric_types: dict[type[Any], type[Any]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        return None


__all__ = ["LMCacheAscendTelemetryProvider"]
