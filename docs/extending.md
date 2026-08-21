# Extending vastai-gpu-runner

## Adding a provider

Implement `CloudRunner` only for a provider that exposes the direct VM
lifecycle: search, create, boot, verify, deploy, setup, launch, poll,
download, and destroy. The base class supplies retry and machine-deduplication
orchestration.

The included `CloudRunner` implementations are `VastaiRunner` and
`LocalRunner`. A declarative batch service belongs behind the separate
`ManagedJobRunner` protocol instead; that provider owns the VM lifecycle.
Keep provider-specific SDK calls inside the provider module and implement the
signatures listed in [the API reference](api.md#runner).

## Driving a local run programmatically

`LocalRunner` runs the `CloudRunner` lifecycle as a local subprocess. The
inherited `run_full_cycle` method returns after worker launch, so callers must
poll, collect, and destroy in a `finally` block:

```python
import time
from pathlib import Path

from vastai_gpu_runner.providers.local import LocalRunner
from vastai_gpu_runner.types import DeploymentConfig

runner = LocalRunner(DeploymentConfig(worker_script="worker.sh"))
result = runner.run_full_cycle(
    files={"worker.sh": Path("worker.sh"), "input.json": Path("input.json")},
    local_output_dir=Path("outputs/local"),
    max_retries=1,
)
if not result.success or result.instance is None:
    raise SystemExit(result.error)
instance = result.instance

try:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        progress = runner.check_progress(instance)
        if progress.get("complete"):
            break
        if progress.get("worker_dead"):
            raise SystemExit(f"worker exited without DONE: {progress.get('log_tail')}")
        time.sleep(1)
    else:
        raise SystemExit("worker did not complete in time")
    runner.download_all_results(instance, Path("outputs/local"), critical_files={"DONE"})
finally:
    runner.destroy_instance(instance)
```

The worker runs with its current working directory set to the temporary
workspace. `LocalRunner` launches the script, manages the process, and removes
the workspace; it does not provide a container or a cloud service.

## Managed-job providers

Implement `ManagedJobRunner` for a declarative batch API. The implementation
must expose `provider_name`, `submit`, `get_status`, `list_tasks`, `cancel`,
and `delete`. Use `ManagedJobSpec`, `ManagedJobHandle`, and the status DTOs
from `vastai_gpu_runner.managed_jobs`; do not reuse `CloudRunner` instance
types.

`GcpBatchRunner` is the included implementation. New generic mounts use
`StorageMount`; `gcs_mounts` is a deprecated compatibility field. The
provider must map unsupported URI schemes to a clear submission error.

## Custom storage

Use `R2Sink` for Cloudflare R2/S3-compatible results and `GcsSink` for GCS
artifacts. `GcsSink` is constructed with `bucket_name`, optionally a client,
and an optional `chunk_size`:

```python
from vastai_gpu_runner.storage.gcs import GcsSink

sink = GcsSink(bucket_name="my-bucket")
uri = sink.upload_bytes("artifacts/result.json", b"{}")
```

GCS plain writes are create-only, CAS writes use object generations, and
`upload_atomic_json` is a staged write followed by final-key overwrite. It
does not perform an atomic rename or create a `DONE` marker.

## Custom workers

Subclass `BaseWorker` and implement `run_workload() -> int`. The base worker
handles GPU checks, preflight gates, result upload, and self-destruction.
Workers that customize `preflight_gates()` must return a list of zero-argument
callables returning `bool`.

## Batch orchestration

`BatchOrchestrator` is an abstract, consumer-owned state coordinator above
`CloudRunner`. Its constructor requires a `ProviderCleanupPolicy` in addition
to a runner factory and validated label prefix. Subclasses implement the unit
iteration, state persistence, payload, collection, failure classification, and
state-mutation hooks. See [the API reference](api.md#batch-orchestrator) for
the current constructor and hook names.
