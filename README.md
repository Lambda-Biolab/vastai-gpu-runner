# vastai-gpu-runner

[![Version](https://img.shields.io/badge/version-0.6.0-8A2BE2)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-58f4c2.svg)](https://www.python.org/)
[![CI](https://github.com/Lambda-Biolab/vastai-gpu-runner/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Lambda-Biolab/vastai-gpu-runner/actions/workflows/ci.yml)
[![Dependabot Updates](https://github.com/Lambda-Biolab/vastai-gpu-runner/actions/actions/dependabot-updates/badge.svg?branch=main)](https://github.com/Lambda-Biolab/vastai-gpu-runner/actions/workflows/dependabot/dependabot-updates)
[![CodeQL](https://github.com/Lambda-Biolab/vastai-gpu-runner/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/Lambda-Biolab/vastai-gpu-runner/actions/workflows/codeql.yml)
![vastai-gpu-runner — Cloud GPU batches for Vast.ai.](.github/social-preview.png)

Cloud GPU orchestration framework for [Vast.ai](https://vast.ai) — batch deployment, R2 storage, worker lifecycle, crash recovery.

## Features

- **CloudRunner** — provider-agnostic lifecycle (Vast.ai, GCP, local) with retry and machine deduplication.
- **VastaiRunner** — hardened Vast.ai deployment with quality filters and ownership guards. Implements `CloudRunner` (direct VM SSH lifecycle), **not** `ManagedJobRunner`.
- **LocalRunner** — zero-cost local/CI backend: the same lifecycle as a subprocess, no cloud credentials.
- **ManagedJobRunner** (Protocol) — provider-neutral interface for declarative cloud batch workloads (GCP Batch, AWS Batch, Azure Batch). The contract behind `GcpBatchRunner`. **Not** the contract behind `VastaiRunner` (that's `CloudRunner`).
- **GcpBatchRunner** — Google Cloud Batch backend implementing `ManagedJobRunner`. Submits jobs, polls status, downloads via the shared `ArtifactSink` interface.
- **GcsSink** — Google Cloud Storage artifact sink. Mirrors the surface of `R2Sink` (upload / download / list) but uses `google-cloud-storage`.
- **R2Sink** — Cloudflare R2 / S3-compatible result storage with DONE markers and parallel downloads.
- **BaseWorker** — template method worker: GPU check, preflight gates, self-destruct.
- **BatchState** — atomic JSON persistence for crash-recoverable batch orchestration.
- **Cost estimator** — GPU speed factors, live pricing, scaling tables.
- **CLI** — credential checks, instance listing, cost estimation, orphan cleanup, batch, run, r2-lifecycle.

## Installation

```bash
uv add "vastai-gpu-runner @ git+https://github.com/Lambda-Biolab/vastai-gpu-runner.git"
```

Requires Python >= 3.11, `vastai` CLI (`pip install vastai`), and R2 credentials in `~/.cloud-credentials`.

For GCP Batch + GCS Sink functionality, install with the `[gcp]` extra:

```bash
uv add "vastai-gpu-runner[gcp] @ git+https://github.com/Lambda-Biolab/vastai-gpu-runner.git"
# or equivalently: pip install vastai-gpu-runner[gcp]
```

The `[gcp]` extra provides `google-cloud-batch>=0.17.0`,
`google-cloud-storage>=3.0.0`, and `google-cloud-logging>=3.10.0`.
Install with `--all-extras` to get all optional backends.

## Quick start

```python
from vastai_gpu_runner.providers.vastai import VastaiRunner
from vastai_gpu_runner.types import DeploymentConfig, OwnershipPolicy

runner = VastaiRunner(
    DeploymentConfig(gpu_model="RTX_4090", max_cost_per_hour=0.35),
    docker_image="my-org/my-image:latest",
    ownership=OwnershipPolicy(label_prefix="myproject"),
)

result = runner.run_full_cycle(
    files={"worker.sh": script_path, "input.tar": data_path},
    local_output_dir=output_path,
    max_retries=3,
)
```

The v0.4.0-recommended `ownership=` + `credentials=` path is shown;
`allowed_images=frozenset(...)` is a deprecated back-compat alias.

See [docs/guide.md](docs/guide.md) for workers, batch state, R2 storage, and cost estimation examples.

## CLI

```bash
vastai-gpu-runner check                  # Verify Vast.ai + R2 credentials
vastai-gpu-runner instances              # List active instances
vastai-gpu-runner estimate -w 10         # Scaling table for 10h of GPU work
vastai-gpu-runner cleanup -l "myproject" # Destroy orphaned instances
vastai-gpu-runner run --provider local --file worker.sh --output outputs/local  # Run locally, no cloud credentials
vastai-gpu-runner batch --provider gcp-batch --gcs-bucket my-bucket --spec spec.json  # Submit a managed job
vastai-gpu-runner r2-lifecycle --bucket my-bucket --prefix outputs/  # R2 lifecycle operations
```

## Architecture

```text
vastai_gpu_runner/
├── types.py              # Enums and dataclasses
├── runner.py             # CloudRunner ABC
├── ssh.py                # SSH/SCP utilities
├── state.py              # Batch + job state persistence
├── orchestrator.py       # check_budget
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

See [docs/architecture.md](docs/architecture.md) for design decisions.

## Documentation

- [User guide](docs/guide.md) — deployment, workers, state, storage, estimation
- [Extending](docs/extending.md) — adding providers, custom storage, worker examples
- [API reference](docs/api.md) — all classes, methods, and parameters
- [Architecture](docs/architecture.md) — module layout and design decisions
- [Changelog](CHANGELOG.md)

## Development

```bash
git clone https://github.com/Lambda-Biolab/vastai-gpu-runner.git
cd vastai-gpu-runner
uv sync --all-extras        # include GCP extras so pyright can resolve google.cloud.*
uv run pytest              # 812 tests
uv run ruff check src/     # linting
uv run pyright src/ tests/ # type checking (includes tests)
```

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
