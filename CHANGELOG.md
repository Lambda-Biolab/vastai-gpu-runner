# Changelog

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
