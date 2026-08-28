# LMCache-Ascend connector equivalence real-run gate

This runbook separates environment readiness from measured connector results.

## Evidence labels

- `preflight-only` validates the exact core and provider source roots, engine
  import chain, runnable vLLM CLI, LMCache version, bundle roles, model path,
  ports, NPU inventory, output uniqueness, and Git revisions. It launches no
  service and proves no connector behavior.
- `real-online` requires live vLLM and LMCache processes, real requests, raw
  telemetry, and cleanup of only the processes created by the runner.
- Unit tests, mocks, historical logs, and regenerated summaries are not a
  `real-online` rerun.

## Safe preflight

Use a controlled Python environment containing the exact vLLM-HUST,
LMCache-Ascend, LMCache, and torch-npu builds intended for the run:

```bash
python tools/real_run_preflight.py \
  --project-root /workspace/LMCache-Ascend \
  --core-root /workspace/vllm-hust \
  --model /workspace/models/Qwen3-8B \
  --python /controlled/env/bin/python \
  --output-dir /new/path/tied-to-the-run
```

The output directory must not already exist. The command writes only
`preflight.json`; it starts no server, installs no dependency, and never
terminates a process. A nonzero exit and `ready_for_real_online: false` mean the
environment is blocked, not that LMCache-Ascend failed a performance test.
Top-level namespace imports are insufficient: `vllm` and `lmcache_ascend` must
resolve under the selected checkouts, an engine-level vLLM import must pass, and
the service CLI must construct in the same controlled interpreter.

## Required matched matrix

Use the same core/provider commits, model, prompts, request order, devices,
ports, cache budget, CANN/torch-npu versions, and service implementation for:

1. legacy `kv_connector_module_path` configuration;
2. typed split-role bundle selection;
3. rollback to the legacy configuration on a fresh process start, with no
   typed selection or extension-manifest environment.

Retain raw requests, resolved configurations, scheduler/worker/API telemetry,
cache hit/miss and save/load records, preemption behavior, failures, shutdown
logs, device inventory, dependency versions, and exact revisions. Compare
outputs, cache decisions, TTFT, throughput, failure policy, pending-store
drain, monkey-patch behavior, and cleanup. A successful typed import or startup
does not satisfy rollback, lifecycle, accelerator, or performance equivalence.

## Current host observation

On 2026-08-29, host 112 exposed eight NPUs but the isolated core environment
lacked `lmcache`, `lmcache_ascend`, and `torch_npu`; the invoking user also
could not access the Docker daemon. Source checkouts alone are not a controlled
runtime. The preflight is therefore blocked before service launch.
