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

## Installation

```bash
# From GitHub
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

```python
from pathlib import Path
from vastai_gpu_runner.worker.base import BaseWorker

class MyWorker(BaseWorker):
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
worker = MyWorker(workspace=Path("/workspace/training"))
exit_code = worker.main()
# Lifecycle: write_pid -> check_gpu -> check_r2 -> run_workload -> upload -> self_destruct
```

### Track batch state

```python
from vastai_gpu_runner.state import BatchState, ShardState

# Create batch
state = BatchState(
    batch_id="my-batch-001",
    num_gpus=4,
    shards=[ShardState(shard_id=i, item_ids=[...]) for i in range(4)],
)

# Persist (atomic write)
state.save(Path("batch_state.json"))

# Resume after crash
state = BatchState.load(Path("batch_state.json"))
for shard in state.failed_shards:
    print(f"Shard {shard.shard_id} needs re-deploy: {shard.failure_reason}")
```

### Estimate costs

```python
from vastai_gpu_runner.estimator.core import build_scaling_table, fallback_pricing

pricing = fallback_pricing()  # or query_vastai_pricing() for live prices
rows = build_scaling_table(
    total_work_hours_base=10.0,  # 10 hours of work on RTX 4090
    cloud_gpu_counts=[0, 4, 8, 16],
    pricing=pricing,
)

for row in rows:
    print(f"{row.cloud_gpus} GPUs: {row.wall_time_human}, {row.cost_display}")
# 0 GPUs: 10h 00m, $0.00
# 4 GPUs: 2h 03m, $2.63-$5.25
# 8 GPUs: 1h 05m, $2.79-$5.57
# 16 GPUs: 0h 34m, $2.88-$5.75
```

## Architecture

```
vastai_gpu_runner/
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

## Development

```bash
git clone https://github.com/antomicblitz/vastai-gpu-runner.git
cd vastai-gpu-runner
uv sync
uv run pytest          # 68 tests
uv run ruff check src/ # linting
uv run pyright src/    # type checking
```

## License

MIT
