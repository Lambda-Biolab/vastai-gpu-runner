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

## Testing

Tests live under `tests/`. Run them with `make test` (which uses `pytest`).
For mutation testing, see `make mutate`.

## Code style

Ruff is the formatter and linter (configured in `pyproject.toml`). Pyright
is the type checker. Both run via `make lint`. There are no additional
project-specific style rules beyond what ruff and pyright enforce.

## Pull requests

- Branch from `main`
- Keep commits signed and conventionally-prefixed (`feat:`, `fix:`, `ci:`, etc.)
- Run `make validate` locally before pushing
- Open the PR against `main`; CI will run the same gate
