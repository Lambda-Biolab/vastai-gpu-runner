# Changelog

## 0.4.0 (2026-07-27) — v4 cleanup-policy architecture

### Added

- **`cleanup_policy` module** (`src/vastai_gpu_runner/cleanup_policy.py`) —
  provider-agnostic DTOs for the v4 architecture: `InstanceCandidate`,
  `CleanupResult` (verdict + refusal + error), `OwnershipPolicy`,
  `ProviderCleanupPolicy`, and the `_repository` Docker/OCI reference
  grammar. Imports nothing from `providers/`.
- **`VastaiProviderConfig`** — frozen dataclass carrying the canonical
  ownership + credentials + label_prefix for Vast.ai composition.
  `__post_init__` validates the docker_image ownership invariant.
- **`VastaiRunner` v4 constructor** — `__init__` now requires
  `ownership: OwnershipPolicy | None` and `credentials: CredentialResolution | None`.
  `allowed_images=` is a deprecated back-compat alias; simultaneous
  `ownership=` + `allowed_images=` raises `ValueError`.
  `from_config(canonical)` classmethod preserves the v4 identity.
- **`list_vastai_instances(*, credentials)`** — credential-aware Vast.ai
  enumeration. AVAILABLE uses REST pagination with the canonical
  `credentials.key`; ABSENT uses ambient CLI enumeration; EXPLICITLY_DISABLED
  returns `[]` without any provider call. `VASTAI_TERMINAL_STATES` constant.
- **`verify_instance_ownership(instance_id, *, ownership)`** — tagged-enum
  `OwnershipVerification` (DISABLED / OWNED / ABSENT / REFUSED). The v2
  bool-returning helper is removed.
- **`build_vastai_cleanup_policy(*, ownership, credentials)`** — v4 factory
  wiring `ProviderCleanupPolicy.destroy_fn` (eligibility + adapter
  + CLI fallback + refusal translation) and `list_instances_fn`.
- **State schema migration** — `BatchState` and `JobBatchState` carry
  `label_scope: str = ""`, `requested_label_prefix: str = ""`,
  `schema_version: int = CURRENT_SCHEMA_VERSION`. `load_batch_state()`
  migrates schema 0 → 1 (strips 12-hex suffix from canonical legacy
  `label`, recovers `requested_label_prefix`, archives terminal-scope-less
  legacy state). `StateMigrationError` for unrecoverable state.
  `resolve_label_scope` reuses persisted scope or creates fresh; rejects
  drift. `validate_label_prefix` rejects empty/blank/padded before any
  provider call.
- **`BatchOrchestrator` v4 constructor** — requires `cleanup_policy:
  ProviderCleanupPolicy`. `_sweep_zombies` is policy-driven end-to-end
  with severity-by-outcome logging via the `_log_cleanup_outcome` helper
  (LEAKED=ERROR, UNKNOWN/CLI_ATTEMPTED/CREDENTIALS_DISABLED=WARNING,
  ALREADY_GONE=INFO, refusals=INFO, unrecognised=ERROR).
- **CLI composition roots** (`src/vastai_gpu_runner/cli.py`):
  - `batch` — composition root that loads (or creates) state via
    `load_batch_state`, resolves a unique `label_scope`, builds
    `VastaiProviderConfig` + `VastaiRunner.from_config` +
    `build_vastai_cleanup_policy`.
  - `cleanup` — full canonical scope required; `--allowed-images ""`
    is fail-closed; `--allow-adjacent-scopes` enables broad prefix
    matching (DANGEROUS, documented). Reports destroyed / already-gone
    / unresolved separately.
  - `instances` — uses `list_vastai_instances(credentials=...)` +
    `OwnershipPolicy.matches()` for the "Owned" column. The v2
    `img.split(":")[0] in image` substring match is removed.
  - `check` — uses `list_vastai_instances(credentials=read_vastai_api_key())`
    instead of the direct `vastai_cmd(["show", "instances", "--raw"])` parse.
- **`scripts/audit_caller_sites.sh`** — repository-wide audit for
  post-deletion invariants. Exits 0 when no actionable CODE reference
  remains for any of the deleted v3 / v2 symbols.

### Changed

- **`BatchOrchestrator.__init__`** stores `validate_label_prefix(self._label_prefix)`
  instead of accepting the bare string. Empty/whitespace/padded labels
  raise `ValueError` immediately.
- **`_sweep_zombies`** uses the exact delimited scope `f"{label_prefix}-"`
  so adjacent scopes like `f"{label_prefix}evil"` cannot match.
- **`VastaiRunner.destroy_instance`** is a single
  `destroy_vastai_instance(...)` adapter call; logs the typed
  `DestroyResult` for non-DESTROYED outcomes (no more silent returns).

### Removed (v3 → v4)

- `orchestrator.sweep_zombie_instances` (and helpers
  `_fetch_vastai_instances`, `_sweep_zombies_for_instances`, `_is_zombie`,
  `_r2_says_done`, `_destroy_zombie`, `_log_sweep_outcome`).
- `orchestrator.load_vastai_api_key` (was already removed in v3 step 7;
  v3 destroy adapter's `read_vastai_api_key()` replaces it).
- `providers.vastai._image_is_allowed` (v2 substring/prefix match).
- v3 destroy adapter's local `_repository` + `_is_image_allowed` +
  local `OwnershipVerification` enum + `verify_instance_ownership`
  (replaced by canonical versions in `cleanup_policy.py` and
  `providers/vastai.py`).
- v3 destroy adapter's `_cli_destroy_instance` + `_rest_destroy`
  (v4 adapter + v4 factory dispatch own the destroy path).
- `tests/test_orchestrator.py` (the file tested the v3 `_is_zombie`
  helper which no longer exists; equivalent coverage now lives in
  `tests/test_batch.py::TestZombieSweep` and
  `tests/integration/test_cleanup_policy_integration.py`).

### Tested

- 19 stress tests in `tests/stress/test_orchestrator_stress.py`
  (mock-based, real Vast.ai + SSH behaviours): large job with
  concurrent deploys, connection drops during poll, resume after
  kill mid-cycle, pre-v4 state resume, mixed failures (success +
  preempt + fatal), budget abort, concurrent max-parallel safety,
  zombie sweep during live run, state persistence + atomic write,
  v4 label scope helpers.
- 2 real Vast.ai stress tests in `tests/stress/test_real_vastai_stress.py`
  (opt-in via `VASTAI_API_KEY`): cheapest-RTX-3060 deploy + SSH +
  destroy end-to-end, and v4 `build_vastai_cleanup_policy` REST
  enumeration against real Vast.ai. **< $0.05 total cloud spend**.
- **623 tests pass** (was 404 at the start of the v4 work).
- All 4 CI gates green: ruff format/check, pyright strict,
  complexipy CC ≤ 10, pytest.

## 0.3.0 (2026-04-15)

### Added

- `BatchOrchestrator(..., max_parallel_collects: int = 1)` — opt-in concurrent
  finalisation of terminal units within a single poll cycle. Default preserves
  sequential semantics. Set >1 when many units complete around the same
  wall-clock time and the finalise step is I/O-bound (e.g. rsync over SSH).
  Bandwidth-constrained environments should leave it at 1 or 2.
- `BatchOrchestrator._classify_live_unit()` — pure classification half of the
  poll cycle (R2 → SSH → worker_dead re-check), no side effects. Returns
  `"terminal" | "running" | "preempted"`. The split makes `_poll_cycle_once`
  safe to finalise terminal units in a thread pool.

### Changed

- `_poll_cycle_once` now classifies all live units first, then handles
  preempted units serially, then finalises terminal units via
  `_finalise_terminal_units` (optional parallel). `_check_unit` is retained
  as a backwards-compat composition for single-unit callers and unit tests.
- `BatchOrchestrator.__init__` rejects `max_parallel_collects < 1` with
  `ValueError`.

## 0.2.0 (2026-04-14)

### Added

- `BatchOrchestrator[UnitT]` — generic template-method ABC above `CloudRunner`
  that coordinates many cloud GPU units in parallel. Handles resume, deploy,
  zombie sweep, poll with exponential backoff, R2-first completion, silent
  crash detection, retry cap, collect phase, cleanup. Consumers implement 14
  narrow hooks over their own `BatchState` / `JobBatchState` type; bug fixes
  land once and both shard-based and job-based workloads inherit them.
- 26 unit tests covering deploy/poll/resume/retry/collect/cleanup/run lifecycle.

## 0.1.0 (2026-04-12)

Initial extraction from [OralBiome-AMP](https://github.com/Lambda-Biolab/OralBiome-AMP).

### Added

- `CloudRunner` ABC with `run_full_cycle()` retry orchestration and machine deduplication
- `VastaiRunner` — Vast.ai marketplace implementation with quality filters, configurable ownership guard, belt-and-suspenders instance destruction
- `R2Sink` — S3-compatible storage with configurable bucket/prefix, DONE markers, parallel downloads (8 threads), DCD trajectory chunk support, upload script generation
- `BaseWorker` — template method worker lifecycle: GPU health check, preflight gates, self-destruct via Vast.ai REST API
- `BatchState`/`ShardState` — shard-based batch state with atomic JSON persistence
- `JobState`/`JobBatchState` — job-based batch state with cost tracking
- Orchestrator utilities: `sweep_zombie_instances`, `ensure_detached` (fork + setsid), `check_budget`, `poll_instance_progress`
- Cost estimator: `GPU_SPEED_FACTOR` (3090/4090/5090), `build_scaling_table`, live Vast.ai pricing, timing persistence
- SSH utilities: `ssh_cmd`, `scp_upload`, `scp_download` with hardened defaults
- CLI: `check` (credential verification), `instances` (listing with ownership), `estimate` (scaling tables), `cleanup` (orphan destruction)
- 68 unit tests, ruff + pyright clean
