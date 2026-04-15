# Architecture

## Module layout

```
vastai_gpu_runner/
    __init__.py           # Public API re-exports
    cli.py                # CLI: check, instances, estimate, cleanup
    types.py              # Provider, InstanceStatus, DeploymentConfig, CloudInstance
    runner.py             # CloudRunner ABC with run_full_cycle + download_all_results
    ssh.py                # ssh_cmd, scp_upload, scp_download
    state.py              # BatchState/ShardState + JobState/JobBatchState
    orchestrator.py       # sweep_zombies, ensure_detached, check_budget
    batch.py              # BatchOrchestrator ABC — deploy/poll/collect lifecycle
    providers/
        vastai.py         # VastaiRunner — Vast.ai marketplace implementation
    storage/
        r2.py             # R2Sink — S3-compatible storage with DONE markers
    worker/
        base.py           # BaseWorker ABC — template method lifecycle
        health.py         # check_gpu, check_r2_connectivity
    estimator/
        core.py           # GPU_SPEED_FACTOR, build_scaling_table, ScalingRow
        pricing.py        # query_vastai_pricing — live marketplace query
```

## Layered design

```
┌─────────────────────────────────────────────────┐
│  CLI (cli.py)                                   │  User-facing commands
├─────────────────────────────────────────────────┤
│  BatchOrchestrator (batch.py)                   │  Deploy/poll/collect many units
├─────────────────────────────────────────────────┤
│  Orchestrator utils (orchestrator.py)           │  Zombie sweep, detach, budget
├─────────────────────────────────────────────────┤
│  CloudRunner (runner.py)                        │  Provider-agnostic lifecycle
├─────────────────────────────────────────────────┤
│  VastaiRunner (providers/vastai.py)             │  Vast.ai marketplace API
├─────────────────────────────────────────────────┤
│  SSH (ssh.py)                                   │  ssh_cmd, scp_upload/download
├─────────────────────────────────────────────────┤
│  Workers (worker/base.py)                       │  GPU-side execution
├─────────────────────────────────────────────────┤
│  Storage (storage/r2.py)                        │  Result persistence
├─────────────────────────────────────────────────┤
│  State (state.py)                               │  Crash recovery
└─────────────────────────────────────────────────┘
```

## Design decisions

### Ownership guard

`VastaiRunner` takes an `allowed_images` parameter. `destroy_instance()` refuses to destroy instances running images not in this set. This prevents cross-project accidents on shared Vast.ai accounts (e.g. destroying a training run when cleaning up an inference batch).

### Configurable R2Sink

Bucket and prefix are constructor params, not hardcoded. Projects subclass `R2Sink` with their own defaults:

```python
class MyR2Sink(R2Sink):
    def __init__(self):
        super().__init__(bucket="my-bucket", prefix="my-project/batches")
```

### Template method workers

`BaseWorker` uses the template method pattern. The `main()` sequence is fixed (pid -> gpu -> preflight -> workload -> upload -> self_destruct), but each step can be overridden. This ensures all workers get GPU health checks and self-destruct without reimplementing them.

### Atomic state persistence

`BatchState.save()` writes to a temp file then renames. This guarantees the state file is never corrupt — even if the process crashes mid-write, the previous state file is intact.

### Consumer-owned state, orchestrator-driven events

`BatchOrchestrator` never mutates the consumer's `BatchState` / `JobBatchState` directly. Instead, it drives events (`on_unit_deployed`, `on_unit_failed`, `on_unit_completed`, `on_unit_preempted`) and the consumer updates its own fields, status strings, and retry counters. This keeps status-string vocabulary, retry accounting, and persistence format entirely in the consumer — so Boltz-2 and OpenMM can share the orchestration loop without sharing a state schema.

### R2-first poll loop

When an R2 sink is configured, the poll loop checks `unit_is_done_in_r2` *before* any SSH call. This matters in two places: (1) healthy-path completion detection is cheaper and works even if the SSH channel is flaky, and (2) when silent-crash detection fires (`worker_dead`), we re-check R2 one more time before treating the unit as preempted — the worker may have uploaded results and then died between the first R2 check and the SSH probe. Without the re-check, successful-but-crashed workers get unnecessarily re-deployed.

### SSH hardening

All SSH commands use:
- `StrictHostKeyChecking=no` — Vast.ai IPs are ephemeral
- `UserKnownHostsFile=/dev/null` — no stale host key warnings
- `stdin=DEVNULL` — prevents stdin stealing (production incident from UTI-project)
- `ConnectTimeout` capped at 10s — fast fail on unreachable hosts

### Belt-and-suspenders destroy

`destroy_instance()` uses 4 mechanisms in sequence because Vast.ai instances sometimes resurrect after a single DELETE:
1. CLI `vastai destroy instance`
2. REST API PUT `state=stopped` (kills Docker pull on booting instances)
3. REST API DELETE (up to 3 retries)
4. Verify after 5s delay, re-destroy if resurrected
