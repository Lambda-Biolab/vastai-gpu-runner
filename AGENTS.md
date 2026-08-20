# AGENTS.md — vastai-gpu-runner

This is the primary instruction file for AI coding agents working on this project.
Read this file first. It supersedes any default behavior.

## Project Purpose

Cloud GPU orchestration framework for Vast.ai + GCP Batch:

- **CloudRunner** — direct VM lifecycle (Vast.ai, local) with retry and machine deduplication
- **ManagedJobRunner** — provider-neutral interface for declarative cloud batch workloads (GCP Batch backend)
- **R2Sink** / **GcsSink** — artifact sinks (Cloudflare R2 / GCS)
- **BaseWorker** — template method worker
- **BatchState** — atomic JSON persistence for crash-recoverable batch orchestration
- **CLI** — credential checks, instance listing, cost estimation, orphan cleanup, batch, run, r2-lifecycle

The distribution and import package retain the historical `vastai-gpu-runner`
and `vastai_gpu_runner` names. The current published version is `0.6.0`; the
managed-job contract stabilization is the recommended next `0.7.0` release,
not a published release.

## Architecture

```text
src/vastai_gpu_runner/
├── types.py              # Enums and dataclasses
├── runner.py             # CloudRunner ABC
├── ssh.py                # SSH/SCP utilities
├── state.py              # Batch + job state persistence
├── orchestrator.py       # Batch orchestration (check_budget)
├── cleanup_policy.py     # v4 candidate / verdict / ownership policy
├── cli.py                # CLI: check, instances, estimate, cleanup, run, batch, r2-lifecycle
├── providers/
│   ├── vastai.py         # Vast.ai implementation (CloudRunner)
│   ├── local.py          # Local subprocess backend (CloudRunner)
│   ├── destroy.py        # Provider-agnostic destroy orchestration
│   └── destroy_adapters/ # Per-provider destroy adapters
├── managed_jobs/         # ManagedJobRunner implementations (declarative cloud batch)
│   ├── base.py           # ManagedJobRunner Protocol + DTOs
│   ├── gcp_batch.py      # GcpBatchRunner + FakeGcpBatchClient + FakeGcsClient
│   └── state.py          # ManagedJobState JSON loader
├── storage/
│   ├── r2.py             # R2/S3 result storage
│   ├── r2_lifecycle.py   # R2 lifecycle operations
│   └── gcs.py            # GcsSink (GCS artifact sink)
├── hybrid.py             # Hybrid local+cloud orchestration
├── unit_lifecycle.py     # Unit-level lifecycle helpers
├── worker/base.py        # BaseWorker template method
├── worker/health.py      # GPU + R2 health checks
├── estimator/core.py     # Scaling tables, GPU speed factors
└── estimator/pricing.py  # Live Vast.ai pricing
```

## Two parallel provider abstractions

The runtime has two distinct provider abstractions that are
**separate on purpose**:

| Abstraction | Implemented by | Use case |
|---|---|---|
| `CloudRunner` (`runner.py`) | `VastaiRunner`, `LocalRunner` | Direct VM/process lifecycle (you run the VM or local worker) |
| `ManagedJobRunner` (`managed_jobs/base.py`) | `GcpBatchRunner` | Declarative cloud batch (the cloud platform owns the VM lifecycle) |

A consumer picks one explicitly via configuration. The two
abstractions do not share code or types. When adding a new
provider, decide which abstraction it implements — `CloudRunner`
(for VM-lifecycle-style providers) or `ManagedJobRunner` (for
declarative batch providers).

## Domain Rules

### `ManagedJobRunner` (declarative cloud batch)

- The Protocol lives in `managed_jobs/base.py`. Implementations
  MUST provide `provider_name`, `submit`, `get_status`, `list_tasks`,
  `cancel`, `delete`. The Protocol is `@runtime_checkable` so
  `isinstance(runner, ManagedJobRunner)` works.
- `spec.name` is the provider idempotency key. Duplicate names raise
  `ManagedJobConflictError`; they are not treated as implicit success.
- Lifecycle states are `QUEUED | PENDING | RUNNING | SUCCEEDED | FAILED |
  CANCELLING | CANCELLED | UNKNOWN` (`ManagedJobLifecycleState`). The
  historical `ManagedJobTerminalState` import remains an alias, and
  in-progress states are not silently converted to `UNKNOWN`.
- `cancel` and `delete` are behaviorally idempotent, including provider
  not-found responses.
- `ManagedJobState` uses neutral `correlation_metadata`; schema migration
  preserves legacy `campaign_id` / `stage_id` fields and old state files.
- `GcsSink` (GCS artifact sink) is a sibling of `R2Sink` — the
  GCS source-of-truth is the object set itself, not a DONE
  marker. Its historical `upload_atomic_json` name means staged temp
  write plus final overwrite; it does not claim atomic rename.
- `ManagedJobSpec.storage_mounts` is the typed generic mount contract.
  `gcs_mounts` remains a deprecated compatibility field for `gs://`
  mounts during migration.

### `CloudRunner` (direct VM lifecycle)

- `VastaiRunner` is the production Vast.ai implementation. The
  v4 ownership / credentials / label_prefix pattern is canonical;
  `allowed_images=frozenset(...)` is a deprecated back-compat alias.
- `LocalRunner` is the zero-cost CI backend — same `CloudRunner`
  interface, no cloud credentials.

## Dependencies

- **Vast.ai deployment**: `vastai` CLI (pip), `VASTAI_API_KEY`
- **R2 storage**: `R2_*` credentials in `~/.cloud-credentials`
- **GCP Batch + GCS** (optional, `gcp` extra): `google-cloud-batch`,
  `google-cloud-storage`, `google-cloud-logging`; `GOOGLE_*`
  auth via `gcloud auth application-default login`

## How to Add a New Provider

1. Decide which abstraction it implements: `CloudRunner` (VM
   lifecycle) or `ManagedJobRunner` (declarative batch).
2. Place the implementation in the matching subpackage:
   - `providers/<name>.py` for `CloudRunner` (mirroring
     `vastai.py` / `local.py`)
   - `managed_jobs/<name>.py` for `ManagedJobRunner` (mirroring
     `gcp_batch.py`)
3. Define config + result dataclasses
4. Implement the direct-VM lifecycle methods for `CloudRunner`, or the
   `submit` + `get_status` + `list_tasks` + `cancel` + `delete` methods for
   `ManagedJobRunner`, including the documented duplicate and idempotency
   semantics
5. Add tests in `tests/` using mocks (no real cloud APIs)
6. Add optional extras in `pyproject.toml`
7. Export from `__init__.py`

## Quality Assurance

```bash
make validate       # Full gate: ruff → pyright → complexipy → bandit → pytest
make quick_validate # Fast gate: ruff + pyright
make lint           # Check linting and formatting
make test           # Run tests only
```

CI workflow: `.github/workflows/ci.yml` (lint → type → complexity →
test, Python 3.11 + 3.12). The CI test job installs with
`uv sync --frozen --all-extras --group dev` so pyright can resolve
`google.cloud.batch_v1` / `storage` symbols.

## Other Governance Docs

- [CONTRIBUTING.md](CONTRIBUTING.md) — contributor workflow, code style
- [AGENT_REQUESTS.md](AGENT_REQUESTS.md) — escalation to humans
- [AGENT_LEARNINGS.md](AGENT_LEARNINGS.md) — pattern discovery
- [CHANGELOG.md](CHANGELOG.md) — version history
