# Changelog

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
