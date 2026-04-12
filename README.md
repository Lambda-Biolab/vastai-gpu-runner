# vastai-gpu-runner

Cloud GPU orchestration framework for [Vast.ai](https://vast.ai) — battle-tested infrastructure for deploying GPU workloads with crash recovery, R2 storage, and autonomous workers.

Extracted from [OralBiome-AMP](https://github.com/Lambda-Biolab/OralBiome-AMP)'s cloud deployment infrastructure after 100+ production deployments across Boltz-2 structure prediction and OpenMM molecular dynamics workloads.

## Features

- **CloudRunner ABC** — provider-agnostic lifecycle: search offers, create instance, boot, verify GPU, deploy files, launch worker, poll, download, destroy
- **VastaiRunner** — hardened Vast.ai implementation with quality filters (reliability > 0.995, bandwidth > 800 Mbps), ownership guards, and belt-and-suspenders instance destruction
- **R2Sink** — S3-compatible result storage (Cloudflare R2, AWS S3, MinIO) with DONE markers, incremental uploads, parallel downloads, and DCD chunk support
- **BaseWorker** — template method worker lifecycle: PID file, GPU health check, preflight gates, workload execution, result upload, self-destruct
- **BatchState / JobState** — atomic JSON state persistence for crash recovery (sharded batches and 1:1 job-to-instance mappings)
- **Orchestrator utilities** — zombie instance sweep, SSH disconnect survival (fork + setsid), budget ceiling enforcement
- **Cost estimator** — GPU speed factors, live Vast.ai pricing, scaling tables for cost/time tradeoff analysis
- **CLI tools** — credential verification, instance listing, cost estimation, orphan cleanup

## Installation

```bash
# From GitHub (recommended for projects using uv)
uv add "vastai-gpu-runner @ git+https://github.com/antomicblitz/vastai-gpu-runner.git"

# Or with pip
pip install "vastai-gpu-runner @ git+https://github.com/antomicblitz/vastai-gpu-runner.git"
```

### Requirements

- Python >= 3.11
- `vastai` CLI tool (`pip install vastai`) for Vast.ai API access
- `~/.cloud-credentials` file with R2 credentials (for R2Sink):
  ```bash
  export R2_ENDPOINT="https://your-account.r2.cloudflarestorage.com"
  export R2_ACCESS_KEY_ID="your-key"
  export R2_SECRET_ACCESS_KEY="your-secret"
  ```

## CLI

The package installs a `vastai-gpu-runner` command with four subcommands:

```bash
vastai-gpu-runner check                  # Verify Vast.ai API key + R2 credentials
vastai-gpu-runner instances              # List active instances with ownership info
vastai-gpu-runner estimate -w 10         # Cost/time scaling table for 10h of GPU work
vastai-gpu-runner cleanup -l "myproject" # Destroy orphaned instances by label prefix
```

### `check` — Verify credentials

```bash
$ vastai-gpu-runner check
Vast.ai CLI
  OK — API key valid, 0 instance(s)
R2 Storage
  OK — R2 reachable

All checks passed.
```

### `instances` — List active instances

```bash
$ vastai-gpu-runner instances --allowed-images "my-org/my-image:latest"
          3 Active Instance(s)
┏━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ ID       ┃ GPU        ┃ Status  ┃ Label               ┃  $/hr ┃ Owned ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 34771283 │ RTX 3090   │ running │ myproject-batch-001  │ $0.15 │  yes  │
│ 34772251 │ RTX 4090   │ running │ myproject-batch-002  │ $0.32 │  yes  │
│ 34773100 │ RTX 3090   │ running │ other-project-xyz    │ $0.14 │  no   │
└──────────┴────────────┴─────────┴─────────────────────┴───────┴───────┘
```

### `estimate` — Cost/time scaling table

```bash
$ vastai-gpu-runner estimate --work-hours 10 --gpus "0,4,8,16"
       Scaling table (1 local RTX_4090 + N cloud GPUs)
┏━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Cloud GPUs ┃ GPU type ┃ Wall time ┃ Est. cost ┃ Notes      ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━┩
│  0 (local) │ RTX_4090 │   10h 00m │     $0.00 │ local only │
│          4 │ RTX_3090 │    2h 27m │     $1.47 │            │
│          8 │ RTX_3090 │    1h 23m │     $1.68 │            │
│         16 │ RTX_3090 │       45m │     $1.80 │            │
└────────────┴──────────┴───────────┴───────────┴────────────┘
```

Use `--live` for real-time Vast.ai marketplace pricing, `--json` for machine-readable output.

### `cleanup` — Destroy orphaned instances

```bash
$ vastai-gpu-runner cleanup --label "myproject-batch" --dry-run
Found 2 instance(s) matching 'myproject-batch':
  34771283: RTX 3090 status=stopped label=myproject-batch-001
  34772251: RTX 4090 status=exited  label=myproject-batch-002

Dry run — no instances destroyed.
```

Remove `--dry-run` to actually destroy. Always prompts for confirmation.

## Quick start

### Deploy a GPU workload

```python
from vastai_gpu_runner.providers.vastai import VastaiRunner
from vastai_gpu_runner.types import DeploymentConfig

config = DeploymentConfig(
    gpu_model="RTX_4090",
    max_cost_per_hour=0.35,
    min_reliability=0.995,
    workspace_dir="/workspace/my_job",
)

runner = VastaiRunner(
    config,
    docker_image="my-org/my-gpu-image:latest",
    allowed_images=frozenset({"my-org/my-gpu-image:latest"}),
)

# Full lifecycle with retry
result = runner.run_full_cycle(
    files={"worker.sh": local_script_path, "input.tar": local_data_path},
    local_output_dir=output_path,
    max_retries=3,
)

if result.success:
    print(f"Deployed on instance {result.instance.instance_id}")
```

### Build a custom worker

Workers use the template method pattern. Override `run_workload()` for your GPU task — everything else (GPU health, R2 gate, PID file, self-destruct) is handled automatically.

```python
from pathlib import Path
from vastai_gpu_runner.worker.base import BaseWorker

class TrainingWorker(BaseWorker):
    def run_workload(self) -> int:
        """Run your GPU workload. Return 0 on success."""
        import subprocess
        result = subprocess.run(
            ["python", "train.py", "--epochs", "10"],
            timeout=3600,
            check=False,
        )
        return result.returncode

# On the cloud instance:
worker = TrainingWorker(workspace=Path("/workspace/training"))
exit_code = worker.main()
```

#### Worker lifecycle

`BaseWorker.main()` executes this sequence:

```
1. write_pid()          — Write worker.pid for process detection
2. check_gpu()          — nvidia-smi temp/ECC check (abort if GPU unhealthy)
3. preflight_gates()    — R2 connectivity + custom gates (abort if any fail)
4. run_workload()       — YOUR CODE — return 0 for success
5. upload_results()     — Call r2_upload.py --done (if script exists)
6. self_destruct()      — DELETE via Vast.ai REST API (if env vars set)
```

Override any step by defining the method in your subclass:

```python
class MyWorker(BaseWorker):
    def preflight_gates(self):
        """Add custom preflight checks."""
        return [self._check_r2, self._check_weights]

    def _check_weights(self) -> bool:
        """Download model weights if not present."""
        weights = self.workspace / "model.pt"
        if weights.exists():
            return True
        # ... download logic ...
        return weights.exists()

    def run_workload(self) -> int:
        # ... your GPU work ...
        return 0

    def upload_results(self):
        """Custom upload with progress tracking."""
        # ... your upload logic ...
```

### Track batch state

Two state models for different workload patterns:

**Sharded batches** (N items split across M GPUs):
```python
from vastai_gpu_runner.state import BatchState, ShardState

state = BatchState(
    batch_id="my-batch-001",
    num_gpus=4,
    shards=[ShardState(shard_id=i, item_ids=[...]) for i in range(4)],
)
state.save(Path("batch_state.json"))  # Atomic write (tmp + rename)

# Resume after crash
state = BatchState.load(Path("batch_state.json"))
print(f"Active: {len(state.active_shards)}, Failed: {len(state.failed_shards)}")
```

**Job-based batches** (1 job = 1 GPU instance):
```python
from vastai_gpu_runner.state import JobBatchState, JobState

state = JobBatchState(
    batch_id="md-batch-001",
    jobs=[JobState(job_name=f"sim_{i}", cost_per_hour=0.15) for i in range(10)],
)
state.save(Path("md_batch_state.json"))
print(f"Total cost: ${state.total_cost:.2f}")
```

### Use R2 storage

```python
from vastai_gpu_runner.storage.r2 import R2Sink

sink = R2Sink(bucket="my-bucket", prefix="project/batches")

# Check completion markers
sink.is_shard_done("batch-001", shard_id=0)
sink.prediction_exists("batch-001", "peptide_042")

# Download results (parallel, 8 threads)
files = sink.download_shard("batch-001", shard_id=0, local_dir=Path("./results"))

# Generate upload script for cloud workers
script = sink.generate_upload_script("batch-001", shard_id=0, workspace="/workspace")
```

### Estimate costs

```python
from vastai_gpu_runner.estimator.core import build_scaling_table, fallback_pricing
from vastai_gpu_runner.estimator.pricing import query_vastai_pricing

# Live pricing (requires vastai CLI)
pricing = query_vastai_pricing()

# Or static fallback (offline)
pricing = fallback_pricing()

rows = build_scaling_table(
    total_work_hours_base=10.0,  # 10 hours of work on RTX 4090
    cloud_gpu_counts=[0, 4, 8, 16],
    pricing=pricing,
)

for row in rows:
    print(f"{row.cloud_gpus} GPUs: {row.wall_time_human}, {row.cost_display}")
```

## Architecture

```
vastai_gpu_runner/
    __init__.py           # Public API re-exports
    cli.py                # CLI entry point (check, instances, estimate, cleanup)
    types.py              # Provider, InstanceStatus, DeploymentConfig, CloudInstance
    runner.py             # CloudRunner ABC with run_full_cycle + download_all_results
    ssh.py                # ssh_cmd, scp_upload, scp_download
    state.py              # BatchState/ShardState + JobState/JobBatchState (JSON persistence)
    orchestrator.py       # sweep_zombies, ensure_detached, check_budget, poll_instance_progress
    providers/
        vastai.py         # VastaiRunner — full Vast.ai marketplace implementation
    storage/
        r2.py             # R2Sink — S3-compatible storage with DONE markers
    worker/
        base.py           # BaseWorker ABC — template method worker lifecycle
        health.py         # check_gpu, check_r2_connectivity
    estimator/
        core.py           # GPU_SPEED_FACTOR, build_scaling_table, ScalingRow, EstimateResult
        pricing.py        # query_vastai_pricing — live marketplace query
```

### Key design decisions

- **Ownership guard**: `VastaiRunner` takes an `allowed_images` parameter. `destroy_instance()` refuses to destroy instances running images not in this set — prevents cross-project accidents on shared Vast.ai accounts.
- **R2Sink is configurable**: bucket and prefix are constructor params, not hardcoded. Works with any S3-compatible storage.
- **BaseWorker uses template method**: subclass and override `run_workload()`. GPU health, R2 gate, PID file, self-destruct are handled automatically.
- **Atomic state persistence**: `BatchState.save()` writes to a temp file then renames — no corrupt state on crash.
- **SSH stdin redirection**: all SSH commands use `stdin=DEVNULL` to prevent stdin stealing (learned from production incidents).
- **Belt-and-suspenders destroy**: `destroy_instance()` uses CLI destroy + REST API stop + DELETE + verify + retry — Vast.ai instances sometimes resurrect after DELETE.

### Extending with a new provider

Implement the `CloudRunner` interface:

```python
from vastai_gpu_runner.runner import CloudRunner
from vastai_gpu_runner.types import CloudInstance, DeploymentConfig

class RunPodRunner(CloudRunner):
    def search_offers(self, **kwargs):
        """Query RunPod marketplace."""
        # ... RunPod API call ...
        return [{"id": "pod-123", "gpu_name": "RTX 4090", ...}]

    def create_instance(self, offer):
        """Create a RunPod pod."""
        # ... RunPod API call ...
        return CloudInstance(instance_id="pod-123", ...)

    def wait_for_boot(self, instance):
        """Wait for pod to reach running state."""
        # ... poll RunPod API ...
        return True

    # ... implement remaining methods ...
```

The `run_full_cycle()` method in `CloudRunner` handles retry logic, machine deduplication, and error recovery — your provider just implements the individual steps.

### Extending R2Sink for custom storage

Subclass `R2Sink` to add project-specific defaults or methods:

```python
from vastai_gpu_runner.storage.r2 import R2Sink as R2SinkBase

class MyR2Sink(R2SinkBase):
    def __init__(self):
        super().__init__(bucket="my-bucket", prefix="my-project/batches")

    def upload_model(self, batch_id, model_path):
        """Custom upload for trained models."""
        key = f"{self.prefix}/{batch_id}/model.pt"
        self._client.upload_file(str(model_path), self.bucket, key)
```

## API reference

### Types (`vastai_gpu_runner.types`)

| Class | Description |
|-------|-------------|
| `Provider` | Enum: `VASTAI`, `RUNPOD` |
| `InstanceStatus` | Enum: `CREATING`, `BOOTING`, `RUNNING`, `FAILED`, `DESTROYED` |
| `DeploymentConfig` | GPU model, cost limits, timeouts, workspace, reliability thresholds |
| `CloudInstance` | Instance metadata: ID, SSH host/port, GPU model, cost, status |
| `DeploymentResult` | Success flag, instance reference, error message, output files |

### Runner (`vastai_gpu_runner.runner`)

| Method | Description |
|--------|-------------|
| `CloudRunner.search_offers()` | Search marketplace for GPU offers |
| `CloudRunner.create_instance(offer)` | Create instance from an offer |
| `CloudRunner.wait_for_boot(instance)` | Wait for running status |
| `CloudRunner.verify_gpu(instance)` | Verify GPU via nvidia-smi |
| `CloudRunner.deploy_files(instance, files)` | Upload files via SCP |
| `CloudRunner.setup_environment(instance)` | Install deps (micromamba/conda) |
| `CloudRunner.launch_worker(instance)` | Start worker process |
| `CloudRunner.check_progress(instance)` | Check DONE file / PID liveness |
| `CloudRunner.destroy_instance(instance)` | Tear down instance |
| `CloudRunner.run_full_cycle(files, output_dir)` | Full lifecycle with retry |
| `CloudRunner.download_all_results(instance, dir)` | Bulk rsync download |

### SSH (`vastai_gpu_runner.ssh`)

| Function | Description |
|----------|-------------|
| `ssh_cmd(instance, command)` | Execute command via SSH, returns `(rc, stdout)` |
| `scp_upload(instance, local, remote)` | Upload file via SCP |
| `scp_download(instance, remote, local)` | Download file via SCP |

### State (`vastai_gpu_runner.state`)

| Class | Description |
|-------|-------------|
| `ShardState` | Per-shard state: ID, instance, status, items, cost, retries |
| `BatchState` | Batch of shards with `save()`/`load()` and status properties |
| `JobState` | Per-job state: name, instance, status, cost calculation |
| `JobBatchState` | Batch of jobs with `save()`/`load()` and cost totals |

### Worker (`vastai_gpu_runner.worker`)

| Class/Function | Description |
|----------------|-------------|
| `BaseWorker` | Abstract worker with template method `main()` |
| `BaseWorker.run_workload()` | Override this — your GPU code goes here |
| `BaseWorker.preflight_gates()` | Override to add custom pre-checks |
| `check_gpu(min_memory_mib, max_temp_c)` | GPU health via nvidia-smi |
| `check_r2_connectivity(workspace)` | R2 reachability via upload script |

### Orchestrator (`vastai_gpu_runner.orchestrator`)

| Function | Description |
|----------|-------------|
| `sweep_zombie_instances(live_runners, label_prefix)` | Destroy orphaned instances |
| `ensure_detached(log_path, pid_path)` | Fork + setsid for SSH survival |
| `check_budget(spent, ceiling)` | Budget enforcement with warning at 80% |
| `poll_instance_progress(instance, workspace)` | 3-layer check: DONE / PID / log |
| `load_vastai_api_key()` | Read API key from standard paths |

### Estimator (`vastai_gpu_runner.estimator`)

| Class/Function | Description |
|----------------|-------------|
| `GPU_SPEED_FACTOR` | Dict: RTX_3090=0.77, RTX_4090=1.0, RTX_5090=1.43 |
| `PriceSummary` | GPU pricing snapshot: min/max/median $/hr, availability |
| `ScalingRow` | One row: GPU count, wall time, cost range, feasibility |
| `EstimateResult` | Full result with scaling table + Rich table rendering |
| `build_scaling_table(work_hours, gpu_counts, pricing)` | Compute scaling table |
| `query_vastai_pricing()` | Live marketplace price query |
| `fallback_pricing()` | Static fallback prices (offline mode) |
| `record_timing(path, workload, **metrics)` | Persist benchmarks to JSONL |
| `load_calibration(path, workload)` | Load benchmark records |

## Development

```bash
git clone https://github.com/antomicblitz/vastai-gpu-runner.git
cd vastai-gpu-runner
uv sync
uv run pytest          # 68 tests
uv run ruff check src/ # linting
uv run pyright src/    # type checking
```

## Changelog

### 0.1.0 (2026-04-12)

- Initial extraction from OralBiome-AMP
- `CloudRunner` ABC with `run_full_cycle()` retry orchestration
- `VastaiRunner` with quality filters, ownership guard, belt-and-suspenders destroy
- `R2Sink` with configurable bucket/prefix, DONE markers, parallel downloads, DCD chunk support
- `BaseWorker` template method: GPU check, preflight gates, self-destruct
- `BatchState`/`ShardState` + `JobState`/`JobBatchState` with atomic JSON persistence
- Orchestrator utilities: zombie sweep, detach, budget ceiling
- Cost estimator: GPU speed factors, live pricing, scaling tables
- CLI: `check`, `instances`, `estimate`, `cleanup`
- 68 tests, ruff + pyright clean

## License

MIT
