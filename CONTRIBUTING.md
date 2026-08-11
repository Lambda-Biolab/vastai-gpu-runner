# Contributing

This project uses standard Python tooling. See the project README for the
high-level overview and the Makefile for the canonical command surface.

## Commands

The Makefile is the single source of truth for build / lint / test commands:

```bash
make help        # list all targets
make test        # run unit tests
make lint        # run ruff + pyright
make format      # auto-format
make validate    # run the full pre-push gate (lint + typecheck + test)
```

`make help` prints the full list. There is no separate command cheatsheet —
if a target exists, the Makefile documents it.

## Setup

```bash
# Full setup with all extras (required for GCP Batch + GCS development)
make setup_dev

# Equivalent manual setup
uv sync --all-extras --group dev
```

The `[gcp]` extra provides `google-cloud-batch>=0.17.0`,
`google-cloud-storage>=3.0.0`, `google-cloud-logging>=3.10.0`. CI
installs with `uv sync --frozen --all-extras --group dev` so pyright
can resolve `google.cloud.batch_v1` / `storage` symbols. If you skip
the `[gcp]` extra, lint and type-check will not catch symbol errors
in `managed_jobs/` or `storage/gcs.py`.

## Testing

Tests live under `tests/`. Run them with `make test` (which uses `pytest`).
For mutation testing, see `make mutate`.

### Fake-GCP test infrastructure

The `managed_jobs/gcp_batch.py` module provides two in-memory
test doubles — `FakeGcpBatchClient` and `FakeGcsClient` — that
are reusable across multiple test modules. Tests can preset
status transitions via `FakeGcpBatchClient.statuses`; the fake GCS
client supports bucket/blob round-trip keyed by name.

Tests pinning the fake-GCP semantics live in `tests/test_fake_gcp_clients.py`.
The 14 GcpBatchRunner unit tests live in `tests/test_gcp_batch.py`.

### Real-Vast.ai stress tests

Production-scale stress tests (real Vast.ai + real R2, costs real
money) live in `tests/stress/test_real_vastai_stress.py`. They are
gated on `VASTAI_API_KEY` being set and run only via the opt-in
`stress-real.yml` workflow. They MUST NEVER run automatically on
PR push.

## Code style

Ruff is the formatter and linter (configured in `pyproject.toml`). Pyright
is the type checker. Both run via `make lint`. There are no additional
project-specific style rules beyond what ruff and pyright enforce.

## Pull requests

- Branch from `main`
- Keep commits signed and conventionally-prefixed (`feat:`, `fix:`, `ci:`, etc.)
- Run `make validate` locally before pushing
- Open the PR against `main`; CI will run the same gate
