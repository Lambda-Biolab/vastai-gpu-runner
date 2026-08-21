# Changelog

The published package is `0.6.0`. The contract-stabilization changes below
are the recommended contents of the next `0.7.0` release; `0.7.0` has not
been published.

Entries under released versions preserve the historical release record, even
where a later compatibility change superseded an example or constructor
name. Use [`docs/api.md`](docs/api.md) for the current signatures.

## Unreleased — managed-job contract stabilization + R2 lifecycle

### Added

- **Managed-job contract stabilization** — `ManagedJobLifecycleState` now
  distinguishes queued, pending, running, succeeded, failed, cancelling,
  cancelled, and unknown. `ManagedJobTerminalState` remains a compatibility
  alias. Added typed managed-job errors for transient, permanent, conflict,
  and not-found outcomes; GCP provider causes are preserved.
- **Generic storage mounts** — `StorageMount(uri, mount_path, read_only)` is
  the provider-neutral field on `ManagedJobSpec`. GCP maps `gs://` mounts and
  rejects unsupported schemes; deprecated `gcs_mounts` remains during
  migration.
- **Managed-job state schema 2** — neutral `correlation_metadata` replaces
  stage-specific correlation while loading and preserving old state files.

- **`storage/r2_lifecycle.py`** — provider-agnostic domain for managing
  one Cloudflare R2 bucket-lifecycle expiration rule per prefix.
  `R2AdminCredentials` (redacted `__repr__`) — accepts ONLY
  `R2_ADMIN_*` keys (worker-style `R2_*` keys are explicitly
  rejected to enforce credential separation between worker
  object-write credentials and bucket-policy admin credentials).
  `R2ExpirationPolicy` (validated in `__post_init__`),
  `R2LifecycleManager` with `plan_apply`, `plan_remove`, `apply`,
  `remove`. The PUT request is wrapped in
  ``LifecycleConfiguration={"Rules": [...]}`` per the boto3 / R2
  contract (verified via ``botocore.stub.Stubber`` contract tests).
  `apply` uses ``put_bucket_lifecycle_configuration``;
  `remove` uses ``delete_bucket_lifecycle`` when the managed rule
  is the only one remaining (R2 / S3 require at least one rule per
  configuration, so DELETE is the documented way to clear it).
  Deterministic managed-rule IDs
  (`vastai-gpu-runner-expire-<12-hex>`). Optimistic stale-plan
  detection via source-fingerprint comparison (an external writer
  can still race the subsequent PUT). Read-after-write verification
  compares the *complete* normalised rule collection against
  `plan.after_rules`. Typed exception hierarchy:
  `ValidationError`, `CredentialsError`, `AccessDeniedError`,
  `CollisionError`, `StalePlanError`, `VerificationError`,
  `RuleLimitError`.
- **`r2-lifecycle` CLI sub-application** (`cli_r2_lifecycle.py`,
  mounted as `vastai-gpu-runner r2-lifecycle ...`) — three commands:
  `show`, `apply`, `remove`. Required `--credentials-file`,
  no-default `--expire-after-days`, `--dry-run`, `--yes`, exit
  codes 1-9 for typed errors. Bucket-wide (root) prefix rejected.
- **`docs/architecture-r2-collection-handshake.md`** — proposed
  long-term bounded-teardown protocol. Explicitly NOT implemented
  in this change.
- **`tests/test_r2_lifecycle.py`** — 71 unit tests for the lifecycle
  domain (canonicalisation, deterministic IDs, plan/apply/remove,
  full-ruleset read-after-write, fresh-bucket 404 handling,
  access-denied, rule-limit, secret redaction, collision checks
  for both apply and remove).
- **`tests/test_r2_upload_scripts.py`** — 16 behavioural tests for
  the generated upload scripts: success path, exitcode failure,
  missing-exitcode failure (worker.exitcode absent is a failure),
  required-file failure, prediction failure omits DONE markers,
  DONE-marker failure, chunk flush failure, checkpoint mode
  stays best-effort.
- **`providers/local.py` (`LocalRunner`)** — subprocess `CloudRunner`
  backend for zero-cost local and CI runs. `search_offers` returns a
  single synthetic offer `{"machine_id": "local", "dph_total": 0.0}`;
  `create_instance` allocates a tempdir workspace; `verify_gpu` probes
  `nvidia-smi` but proceeds on failure (CPU-only OK); `deploy_files`
  copies payload files; `launch_worker` runs `bash worker.sh`;
  `check_progress` polls PID liveness + `DONE` marker;
  `destroy_instance` terminates the worker process and removes the
  workspace. No SSH, cloud credentials, or Docker; single job per
  runner. `build_local_cleanup_policy()` returns a `Provider.LOCAL`
  policy with no cross-process candidates (the owning runner cleans
  up). No changes to `runner.py`.
- **`Provider.LOCAL`** enum member (`"local"`) in `types.py`.
- **`run` CLI command** (`cli_run.py`) — `vastai-gpu-runner run
  --provider local --file worker.sh [--file INPUT...] --output OUTPUT`.
  Waits for a `DONE` marker (default timeout 300s, `--poll-interval`
  1s), downloads the workspace into `--output`, and destroys the
  runner in a `finally` block. Non-local `--provider` values are
  rejected.
- **Tests** — `tests/integration/test_local_runner_integration.py`
  (end-to-end worker run), `tests/test_local_runner.py` (lifecycle
  methods + cleanup policy), `tests/test_cli_run.py` (CLI `run`).
- **Docs** — README features/CLI, `docs/api.md` (Local Provider +
  `run` command), `docs/guide.md` (local dry run + lifecycle caveat),
  `docs/extending.md` (programmatic local run),
  `docs/architecture-v2.md` (LocalRunner implemented),
  `docs/roadmap.md` (item 1 implemented).

### Changed

- **Managed-job submission semantics** — `spec.name` is the provider
  idempotency key; duplicate names raise a typed conflict. `cancel` and
  `delete` are behaviorally idempotent when the provider reports not-found.
- **`GcsSink` contract** — create-only writes, generation-based CAS, and
  staged JSON writes are documented consistently. `upload_atomic_json` keeps
  its historical name but does not claim atomic rename; completion remains
  the expected object set with no DONE marker.
- **Version recommendation** — release these contract changes as `0.7.0`;
  the package name remains `vastai-gpu-runner`.

- **`BaseWorker.upload_results()`** — fixed `R2_FINAL_UPLOAD_TIMEOUT_SECONDS`
  lowered from 300s to 90s. `subprocess.TimeoutExpired` caught
  separately with a warning that explicitly notes teardown continues.
  Non-zero return codes now logged with truncated stderr/stdout
  (not as "R2 upload complete"). Transport failure does NOT change
  the workload exit code; `self_destruct()` still runs from
  `main()`'s `finally`. Module constant `R2_FINAL_UPLOAD_TIMEOUT_SECONDS`
  exported for downstream visibility. Docstring revised to
  acknowledge that the job uploader's `--done` mode may upload
  large output sets and is therefore a deliberate best-effort
  cutoff.
- **Generated shard uploader (`R2Sink.generate_upload_script`)** —
  `upload_prediction()` now fails closed: if any required file
  upload fails, the per-shard and global DONE markers are NOT
  published and the script exits non-zero. A local upload-failure
  sentinel (`<workspace>/upload_failures.log`) is persisted on
  failure so subsequent `--done` or no-arg invocations also refuse
  to publish the top-level shard DONE marker — a transient
  per-prediction upload failure cannot be masked by a later
  successful worker exit. The whole-shard positive completion
  marker (`<workspace>/shard_completed`) is written atomically by
  `upload_all()` on full success and required by `--done` as
  positive proof of completion; `upload_all()` also writes
  per-prediction completion markers for every subdirectory under
  `outputs/` and clears any stale `<no-arg>` failure entry so a
  retry after recovery can complete normally.
  `upload_done_marker()` and `upload_all()` no longer treat a
  missing `worker.exitcode` as success — absence is a failure
  because we cannot verify the workload outcome without it. The
  `worker.exitcode` upload is attempted *before* the `DONE`
  marker.
- **Generated job uploader (`R2Sink.generate_job_upload_script`)** —
  `_flush_large_file_chunk()` now returns `"uploaded" | "none" | "failed"`
  so callers can distinguish "no new bytes" from "chunk flush failed".
  `upload_all()` tracks required-file failures and omits `DONE` on
  any failure. `_upload_exitcode_job()` rejects missing exitcode.
  Checkpoint mode (`--checkpoint`) remains best-effort — failures
  are warnings, no `DONE` is published.
- **`docs/guide.md`** — new "R2 bucket lifecycle administration"
  section documenting the operator workflow, admin credentials
  separation, lifecycle as maximum-retention backstop, and the
  best-effort rsync fallback caveat.
- **`docs/api.md`** — new `R2 lifecycle` and `Worker upload bounds`
  sections, plus CLI sub-app reference and full exit-code table
  (0, 1, 2, 3, 4, 5, 6, 7, 8, 9).

## 0.6.0 (2026-08-09) — managed-jobs GCP Batch backend

### Added

- **`GcpBatchRunner`** (`src/vastai_gpu_runner/managed_jobs/gcp_batch.py`) —
  concrete implementation of `ManagedJobRunner` (protocol) backed by
  Google Cloud Batch. Submits jobs via `google-cloud-batch` v0.17+,
  polls job status via the Batch v1 API, downloads log + output
  artifacts via the shared `ArtifactSink` interface. Instantiated
  via `GcpBatchRunner(project_id=..., location=..., gcs_bucket=...)`.
- **`GcsSink`** (`src/vastai_gpu_runner/storage/gcs.py`) — Google
  Cloud Storage artifact sink. Mirrors the surface of `R2Sink`
  (upload / download / list) but uses `google-cloud-storage`
  v3.0+. No DONE-marker / sentinel logic — the GCS source-of-truth
  is the object set itself, not a marker file.
- **`FakeGcpBatchClient`** + **`FakeGcsClient`** (private to
  `managed_jobs/gcp_batch.py`) — in-memory stand-ins for
  `batch_v1.BatchServiceClient` and `google.cloud.storage.Client`
  for unit-test isolation. Tests can preset status transitions
  via `FakeGcpBatchClient.statuses`; the fake GCS client supports
  bucket/blob round-trip keyed by name.
- **`tests/test_gcp_batch.py`** — 14 unit tests for `GcpBatchRunner`
  (manifest submission, status polling, terminal-status detection,
  cancellation, error paths via the stub `FakeGcpBatchClient`).
- **`tests/test_fake_gcp_clients.py`** — 3 unit tests that pin the
  semantics of `FakeGcsClient` (bucket lookup, blob round-trip,
  exists() / create_bucket() / bucket().exists() / bucket().blob())
  so the fakes don't silently drift from the subset of the Google
  SDK surface they claim to support.
- **`src/vastai_gpu_runner/managed_jobs/state.py`** — JSON state
  loader for `ManagedJobState` records with a fail-closed
  schema-version loader. Idempotency: the canonical way to detect
  "already submitted" on resume.

### Changed

- **`pyproject.toml`** — new `[gcp]` extra provides
  `google-cloud-batch>=0.17.0`, `google-cloud-storage>=3.0.0`,
  `google-cloud-logging>=3.10.0`. CI workflow now installs with
  `uv sync --frozen --all-extras --group dev` so pyright can
  resolve `google.cloud.batch_v1` / `storage` symbols.
- **`src/vastai_gpu_runner/managed_jobs/__init__.py`** — exports
  `ManagedJobRunner`, `ManagedJobSpec`, `ManagedJobHandle`,
  `ManagedJobStatus`, `ManagedTaskStatus`, `ManagedJobTerminalState`.

### Note

This release ships the GCP Batch backend. The `ManagedJobRunner`
**protocol** (the contract) landed in v0.5.0; v0.6.0 is the first
concrete backend (`GcpBatchRunner`) plus the `GcsSink` it depends
on. `VastaiRunner` does **not** implement `ManagedJobRunner` — it
implements the orthogonal `CloudRunner` interface (direct VM SSH
lifecycle). The two abstractions do not share code or types; a
consumer selects one explicitly via configuration.

## 0.5.0 (2026-08-05) — managed-jobs protocol

### Added

- **`ManagedJobRunner` Protocol** (`src/vastai_gpu_runner/managed_jobs/base.py`) —
  provider-neutral interface for declarative cloud-job providers
  (where the cloud platform owns the underlying VM lifecycle —
  e.g. GCP Batch, AWS Batch, Azure Batch). Methods:
  - `provider_name` (property) — short provider identifier
  - `submit(spec: ManagedJobSpec) -> ManagedJobHandle`
  - `get_status(handle: ManagedJobHandle) -> ManagedJobStatus`
  - `list_tasks(handle: ManagedJobHandle) -> Iterable[ManagedTaskStatus]`
  - `cancel(handle: ManagedJobHandle) -> None`
  - `delete(handle: ManagedJobHandle) -> None`
- **Provider-neutral DTOs** (`managed_jobs/base.py`):
  - `ManagedJobSpec` — name, task_count, parallelism, image,
    command, environment, labels, gcs_mounts, timeout_seconds,
    retry_on_preempt, region
  - `ManagedJobHandle` — opaque (provider, resource_name, location)
  - `ManagedJobStatus` — handle, state, task_count, message,
    create_time, update_time
  - `ManagedTaskStatus` — per-task snapshot (task_index, state,
    exit_code, message)
  - `ManagedJobTerminalState` enum — `SUCCEEDED | FAILED |
    CANCELLED | UNKNOWN`
- **`tests/test_managed_jobs.py`** — 12 unit tests pinning the
  Provider-neutral DTO behavior (to_dict round-trip, default
  factories, frozen-dataclass semantics).

### Note

`VastaiRunner` does **not** implement `ManagedJobRunner` — it
implements the orthogonal `CloudRunner` interface (direct VM SSH
lifecycle). The two abstractions are deliberately separate: a
consumer selects one explicitly via configuration.

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
