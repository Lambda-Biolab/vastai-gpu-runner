# User guide

## Deploy a GPU workload

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

## Build a custom worker

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

### Worker lifecycle

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

## Track batch state

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

## Use R2 storage

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

## Estimate costs

```python
from vastai_gpu_runner.estimator.core import build_scaling_table, fallback_pricing
from vastai_gpu_runner.estimator.pricing import query_vastai_pricing

# Live pricing (requires vastai CLI)
pricing = query_vastai_pricing()

# Or static fallback (offline)
pricing = fallback_pricing()

rows = build_scaling_table(
    total_work_hours_base=10.0,
    cloud_gpu_counts=[0, 4, 8, 16],
    pricing=pricing,
)

for row in rows:
    print(f"{row.cloud_gpus} GPUs: {row.wall_time_human}, {row.cost_display}")
```
