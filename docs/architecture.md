# Architecture

## Package layout

```text
src/vastai_gpu_runner/
├── types.py                 # Direct-VM enums and dataclasses
├── runner.py                # CloudRunner direct-VM lifecycle
├── providers/
│   ├── vastai.py            # VastaiRunner and Vast.ai cleanup composition
│   ├── local.py             # LocalRunner subprocess backend
│   ├── destroy.py           # Generic destroy protocol
│   └── destroy_adapters/    # Provider destroy adapters
├── managed_jobs/
│   ├── base.py              # ManagedJobRunner protocol and DTOs
│   ├── gcp_batch.py         # GcpBatchRunner and in-memory fakes
│   ├── errors.py             # Typed managed-job errors
│   └── state.py              # ManagedJobState schema loader
├── storage/
│   ├── r2.py                # R2 artifacts and DONE-marker workflows
│   ├── r2_lifecycle.py      # R2 lifecycle administration
│   └── gcs.py               # GCS artifacts and generation preconditions
├── batch.py                 # BatchOrchestrator
├── cleanup_policy.py        # Ownership and cleanup-policy DTOs
├── state.py                 # Direct-VM batch state
├── worker/                  # GPU-side worker template and health checks
├── estimator/               # GPU scaling and Vast.ai pricing
├── cli.py                   # CLI composition root
└── ssh.py                   # Direct-VM SSH/SCP helpers
```

## Two parallel provider abstractions

The abstractions are deliberately separate:

| Abstraction | Implementation | Lifecycle |
|---|---|---|
| `CloudRunner` | `VastaiRunner`, `LocalRunner` | Direct VM lifecycle: the runner creates a VM, uses SSH or a local process, and destroys it. |
| `ManagedJobRunner` | `GcpBatchRunner` | Declarative batch: the cloud service owns the VM lifecycle. |

`VastaiRunner` is not a `ManagedJobRunner`. A consumer selects one abstraction
explicitly; their handles, statuses, and resource types are not interchangeable.

## Managed-job contract

`ManagedJobRunner` is a runtime-checkable provider-neutral protocol. It accepts
`ManagedJobSpec`, returns an opaque `ManagedJobHandle`, and exposes
`get_status` plus per-task `list_tasks` diagnostics. `submit` treats
`spec.name` as the provider idempotency key: a duplicate raises the typed
`ManagedJobConflictError`, not implicit success. `cancel` and `delete` are
behaviorally idempotent when the provider reports not-found.

`ManagedJobLifecycleState` is canonical and distinguishes `QUEUED`, `PENDING`,
`RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLING`, `CANCELLED`, and `UNKNOWN`.
Only `SUCCEEDED`, `FAILED`, `CANCELLED`, and `UNKNOWN` are terminal. The historical
`ManagedJobTerminalState` import is an alias to the canonical enum, so
in-progress states are not converted to `UNKNOWN`.

Typed errors include transient, permanent, conflict, and not-found outcomes.
GCP SDK exceptions are mapped to the stable error classes with the SDK error
retained as the exception cause.

`ManagedJobSpec.storage_mounts` is the typed generic mount contract:
`StorageMount(uri, mount_path, read_only=False)`. GCP Batch currently maps
`gs://` mounts to GCS volumes and rejects unsupported schemes. The deprecated
`gcs_mounts` field remains for migration compatibility.

`ManagedJobState` uses schema 2 and neutral `correlation_metadata`. Schema 0
and 1 files are migrated while legacy `campaign_id` and `stage_id` values are
preserved. Unknown or malformed state fails closed with `ManagedJobStateError`.

## Storage semantics

`R2Sink` and `GcsSink` are sibling storage implementations, not one shared
provider implementation. R2 workflows use DONE markers where the R2 contract
requires them. GCS completion is the expected object set itself; `GcsSink`
does not create or require a DONE marker.

For GCS, plain byte/file/text writes are create-only with
`if_generation_match=0`. `upload_cas_write` uses object generations for
compare-and-set updates and raises `GcsPreconditionFailed` on a generation
mismatch. The historical `upload_atomic_json` name means staged temporary
write followed by final-key overwrite. It does not claim atomic rename.

## Direct-VM lifecycle

`CloudRunner.run_full_cycle` performs the direct-VM gates through worker
launch. Callers poll `check_progress`, collect with `download_file` or
`download_all_results`, and destroy in a `finally` block. `BatchOrchestrator`
adds multi-unit deploy, polling, collection, state persistence, budget, and
cleanup-policy dispatch above that lifecycle.

## State and cleanup

Direct-VM `BatchState` and `JobBatchState` are consumer-owned JSON state
models. `BatchOrchestrator` receives a required `ProviderCleanupPolicy`; it
does not infer provider behavior from a runner. The policy lists
`InstanceCandidate` values and returns typed cleanup results, allowing
ownership refusal, already-gone, confirmed destruction, and unresolved
outcomes to remain distinct.

The package keeps the historical `vastai-gpu-runner` distribution and import
name for compatibility. Provider-neutral managed jobs were added alongside
the original Vast.ai direct-VM surface; they did not rename that package.
