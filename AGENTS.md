# AGENTS.md — vastai-gpu-runner

This is the primary instruction file for AI coding agents working on this project.
Read this file first. It supersedes any default behavior.

## Project Purpose

Cloud GPU orchestration framework for Vast.ai + GCP Batch:

- **CloudRunner** — provider-agnostic lifecycle (Vast.ai, local) with retry and machine deduplication
- **ManagedJobRunner** — provider-neutral interface for declarative cloud batch workloads (GCP Batch backend)
- **R2Sink** / **GcsSink** — artifact sinks (Cloudflare R2 / GCS)
- **BaseWorker** — template method worker
- **BatchState** — atomic JSON persistence for crash-recoverable batch orchestration
- **CLI** — credential checks, instance listing, cost estimation, orphan cleanup, batch, run, r2-lifecycle

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
│   └── destroy.py        # Provider-agnostic destroy orchestration
├── destroy_adapters/     # Per-provider destroy adapters
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
| `CloudRunner` (`runner.py`) | `VastaiRunner`, `LocalRunner` | Direct VM SSH lifecycle (you run the VM, you SSH in) |
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
- Idempotency is the caller's responsibility. The
  `managed_jobs/state.py` `ManagedJobState` loader is the canonical
  way to detect "already submitted" on resume.
- Terminal states are `SUCCEEDED | FAILED | CANCELLED | UNKNOWN`
  (the `ManagedJobTerminalState` enum). `UNKNOWN` is a fail-closed
  fallback for cloud-side status responses that don't map cleanly.
- `GcsSink` (GCS artifact sink) is a sibling of `R2Sink` — the
  GCS source-of-truth is the object set itself, not a DONE
  marker. Don't introduce DONE-marker logic to GcsSink.

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
4. Implement the runner class with `run()` (or `submit` + `get_status`),
   idempotency, logging
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
