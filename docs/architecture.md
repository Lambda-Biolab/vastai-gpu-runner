# Architecture

## Module layout

```text
vastai_gpu_runner/
    __init__.py           # Public API re-exports
    cli.py                # CLI: check, instances, estimate, cleanup, run, batch, r2-lifecycle
    types.py              # Provider, InstanceStatus, DeploymentConfig, CloudInstance
    runner.py             # CloudRunner ABC with run_full_cycle + download_all_results
    ssh.py                # ssh_cmd, scp_upload, scp_download
    state.py              # BatchState/ShardState + JobState/JobBatchState
    orchestrator.py       # check_budget
    cleanup_policy.py     # v4 candidate / verdict / ownership policy
    batch.py              # BatchOrchestrator ABC — deploy/poll/collect lifecycle
    providers/
        vastai.py         # VastaiRunner — Vast.ai marketplace implementation
        local.py          # LocalRunner — zero-cost local subprocess backend
        destroy.py        # Provider-agnostic destroy orchestration
    destroy_adapters/     # Per-provider destroy adapters
    managed_jobs/         # ManagedJobRunner implementations (declarative cloud batch)
        base.py           # ManagedJobRunner Protocol + DTOs
        gcp_batch.py      # GcpBatchRunner + FakeGcpBatchClient + FakeGcsClient
        state.py          # ManagedJobState JSON loader
    storage/
        r2.py             # R2Sink — S3-compatible storage with DONE markers
        r2_lifecycle.py   # R2 lifecycle operations
        gcs.py            # GcsSink — Google Cloud Storage artifact sink
    hybrid.py             # Hybrid local+cloud orchestration
    unit_lifecycle.py     # Unit-level lifecycle helpers
    worker/
        base.py           # BaseWorker ABC — template method lifecycle
        health.py         # check_gpu, check_r2_connectivity
    estimator/
        core.py           # GPU_SPEED_FACTOR, build_scaling_table, ScalingRow
        pricing.py        # query_vastai_pricing — live marketplace query
```

## Layered design

```text
┌─────────────────────────────────────────────────┐
│  CLI (cli.py)                                   │  User-facing commands
│  (check, instances, estimate, cleanup, run,      │
│   batch, r2-lifecycle)                          │
├─────────────────────────────────────────────────┤
│  BatchOrchestrator (batch.py)                   │  Deploy/poll/collect many units
├─────────────────────────────────────────────────┤
│  Orchestrator utils (orchestrator.py)           │  Budget check
├─────────────────────────────────────────────────┤
│  CloudRunner (runner.py)                        │  Provider-agnostic lifecycle
│  (VastaiRunner, LocalRunner)                     │
├─────────────────────────────────────────────────┤
│  ManagedJobRunner (managed_jobs/)               │  Declarative cloud batch
│  (GcpBatchRunner)                                │
├─────────────────────────────────────────────────┤
│  VastaiRunner (providers/vastai.py)             │  Vast.ai marketplace API
├─────────────────────────────────────────────────┤
│  SSH (ssh.py)                                   │  ssh_cmd, scp_upload/download
├─────────────────────────────────────────────────┤
│  Workers (worker/base.py)                       │  GPU-side execution
├─────────────────────────────────────────────────┤
│  Storage (storage/r2.py, storage/gcs.py)        │  Result persistence
├─────────────────────────────────────────────────┤
│  State (state.py)                               │  Crash recovery
└─────────────────────────────────────────────────┘
```

## Two parallel provider abstractions

The runtime has two distinct provider abstractions that are
**separate on purpose**:

| Abstraction | Implemented by | Use case |
|---|---|---|
| `CloudRunner` (`runner.py`) | `VastaiRunner`, `LocalRunner` | Direct VM SSH lifecycle (you run the VM, you SSH in) |
| `ManagedJobRunner` (`managed_jobs/base.py`) | `GcpBatchRunner` | Declarative cloud batch (the cloud platform owns the VM lifecycle) |

A consumer picks one explicitly via configuration. The two
abstractions do not share code or types. When adding a new provider,
decide which abstraction it implements — `CloudRunner` (for
VM-lifecycle-style providers) or `ManagedJobRunner` (for declarative
batch providers).

## Design decisions

### Ownership guard

`VastaiRunner` takes an `ownership` parameter (`OwnershipPolicy`).
`destroy_instance()` refuses to destroy instances not matching the
ownership's label prefix. This prevents cross-project accidents on
shared Vast.ai accounts (e.g. destroying a training run when cleaning
up an inference batch). The v0.4.0-recommended `ownership=` + `credentials=`
path is canonical; `allowed_images=frozenset(...)` is a deprecated
back-compat alias.

### Configurable R2Sink

Bucket and prefix are constructor params, not hardcoded. Projects subclass `R2Sink` with their own defaults:

```python
class MyR2Sink(R2Sink):
    def __init__(self):
        super().__init__(bucket="my-bucket", prefix="my-project/batches")
```

### Managed-job contract and GcsSink

`ManagedJobRunner` is a provider-neutral declarative job contract and is
deliberately separate from `CloudRunner`, whose providers own direct VM and
SSH lifecycle. `spec.name` is the managed-job idempotency key: a duplicate
provider name is a typed conflict, while cancel and delete tolerate
not-found responses.

`ManagedJobLifecycleState` distinguishes queued, pending, running, succeeded,
failed, cancelling, cancelled, and unknown. `ManagedJobTerminalState` remains
as a deprecated compatibility alias; it must not collapse in-progress states
into unknown. Provider failures are exposed through the typed managed-job
error hierarchy with the provider exception preserved as the cause.

`ManagedJobSpec.storage_mounts` carries a URI, mount path, and read-only
intent. The deprecated `gcs_mounts` field remains for migration. GCP Batch
maps supported `gs://` mounts to GCS volumes and rejects other schemes.

`ManagedJobState` persists neutral correlation metadata. Schema migration keeps
legacy `campaign_id` and `stage_id` values, so old state files remain loadable.

GcsSink mirrors the surface of R2Sink (upload / download / list) but
uses `google-cloud-storage`.

The GCS source-of-truth is the object set itself, not a DONE marker — there is
no GCS equivalent of the R2 DONE-marker sentinel. Plain writes are
create-only, CAS writes use generation preconditions, and the historical
`upload_atomic_json` API stages a temporary object before overwriting the
final key. It does not perform an atomic rename.

### Template method workers

`BaseWorker` uses the template method pattern. The `main()` sequence is fixed (pid -> gpu -> preflight -> workload -> upload -> self_destruct), but each step can be overridden. This ensures all workers get GPU health checks and self-destruct without reimplementing them.

### Atomic state persistence

`BatchState.save()` writes to a temp file then renames. This guarantees the state file is never corrupt — even if the process crashes mid-write, the previous state file is intact.

### Consumer-owned state, orchestrator-driven events

`BatchOrchestrator` never mutates the consumer's `BatchState` / `JobBatchState` directly. Instead, it drives events (`on_unit_deployed`, `on_unit_failed`, `on_unit_completed`, `on_unit_preempted`) and the consumer updates its own fields, status strings, and retry counters. This keeps status-string vocabulary, retry accounting, and persistence format entirely in the consumer — so Boltz-2 and OpenMM can share the orchestration loop without sharing a state schema.

### R2-first poll loop

When an R2 sink is configured, the poll loop checks `unit_is_done_in_r2` *before* any SSH call. This matters in two places: (1) healthy-path completion detection is cheaper and works even if the SSH channel is flaky, and (2) when silent-crash detection fires (`worker_dead`), we re-check R2 one more time before treating the unit as preempted — the worker may have uploaded results and then died between the first R2 check and the SSH probe. Without the re-check, successful-but-crashed workers get unnecessarily re-deployed.

### SSH hardening

All SSH commands use:

- `StrictHostKeyChecking=no` — Vast.ai IPs are ephemeral
- `UserKnownHostsFile=/dev/null` — no stale host key warnings
- `stdin=DEVNULL` — prevents stdin stealing (production incident from UTI-project)
- `ConnectTimeout` capped at 10s — fast fail on unreachable hosts

### Belt-and-suspenders destroy

`destroy_instance()` uses 4 mechanisms in sequence because Vast.ai instances sometimes resurrect after a single DELETE:

1. CLI `vastai destroy instance`
2. REST API PUT `state=stopped` (kills Docker pull on booting instances)
3. REST API DELETE (up to 3 retries)
4. Verify after 5s delay, re-destroy if resurrected
