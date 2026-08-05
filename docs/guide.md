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

# Deploys through launch; poll/download/destroy are separate (see below)
result = runner.run_full_cycle(
    files={"worker.sh": local_script_path, "input.tar": local_data_path},
    local_output_dir=output_path,
    max_retries=3,
)

if result.success:
    print(f"Deployed on instance {result.instance.instance_id}")
```text

## Run a worker locally (dry run)

The `run` command executes a worker as a local subprocess — no cloud credentials, SSH, or Docker:

```bash
vastai-gpu-runner run --provider local --file worker.sh --file input.json --output outputs/local
```text

It copies the payload into a temp workspace, waits for a `DONE` marker, downloads the results into `--output`, then removes the workspace. Only `--provider local` is supported for now. Tuning options: `--worker-script`, `--timeout`, `--poll-interval`, `--verbose`.

**Lifecycle caveat**: programmatic callers of `run_full_cycle` get the instance back after the worker *launches* — the method does not wait for completion. Poll `check_progress` for the `DONE` marker (or worker death), collect with `download_all_results`, and call `destroy_instance` in a `finally`; the CLI `run` command does all three. See [docs/extending.md](extending.md) for the worked programmatic example.

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
```text

### Worker lifecycle

`BaseWorker.main()` executes this sequence:

```text
1. write_pid()          — Write worker.pid for process detection
2. check_gpu()          — nvidia-smi temp/ECC check (abort if GPU unhealthy)
3. preflight_gates()    — R2 connectivity + custom gates (abort if any fail)
4. run_workload()       — YOUR CODE — return 0 for success
5. upload_results()     — Call r2_upload.py --done (if script exists)
6. self_destruct()      — DELETE via Vast.ai REST API (if env vars set)
```text

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
```text

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
```text

**Job-based batches** (1 job = 1 GPU instance):

```python
from vastai_gpu_runner.state import JobBatchState, JobState

state = JobBatchState(
    batch_id="md-batch-001",
    jobs=[JobState(job_name=f"sim_{i}", cost_per_hour=0.15) for i in range(10)],
)
state.save(Path("md_batch_state.json"))
print(f"Total cost: ${state.total_cost:.2f}")
```text

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
```text

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
```text

## R2 bucket lifecycle administration

The `r2-lifecycle` CLI sub-application manages one Cloudflare R2
bucket-lifecycle expiration rule per prefix. It is a **destructive
bucket-policy mutation** that applies to **every object matching
the configured prefix**, regardless of how those objects were
uploaded or otherwise managed. Activating the rule makes all
existing objects older than the retention window eligible for
deletion; removing the rule does NOT restore any objects already
expired by it. The rule never activates implicitly during package
install, `R2Sink` construction, batch startup, or worker execution
— only the operator CLI can apply it.

### Separate admin credentials

Lifecycle administration requires bucket-policy write authority.
Worker credentials should not be reused for this — workers only need
object-level read/write. Create a separate credentials file:

```bash
# ~/.cloud-credentials.r2-admin
export R2_ADMIN_ENDPOINT="https://<accountid>.r2.cloudflarestorage.com"
export R2_ADMIN_ACCESS_KEY_ID="<admin-access-key>"
export R2_ADMIN_SECRET_ACCESS_KEY="<admin-secret-key>"
```text

The CLI **requires** `R2_ADMIN_*` keys — the worker-style `R2_*`
keys are explicitly rejected so the worker's object-write
credentials cannot be passed off as bucket-administration
credentials. This enforces least-privilege separation between
the two roles.

### `show` — inspect the managed rule

```bash
vastai-gpu-runner r2-lifecycle show \
    --bucket my-bucket \
    --prefix project/batches/ \
    --credentials-file ~/.cloud-credentials.r2-admin
```text

Prints the current managed-rule state and the bucket's source
fingerprint. Non-mutating, non-interactive.

### `apply` — set the retention rule

```bash
vastai-gpu-runner r2-lifecycle apply \
    --bucket my-bucket \
    --prefix project/batches/ \
    --credentials-file ~/.cloud-credentials.r2-admin \
    --expire-after-days 30 \
    --dry-run          # see the plan first
vastai-gpu-runner r2-lifecycle apply \
    --bucket my-bucket \
    --prefix project/batches/ \
    --credentials-file ~/.cloud-credentials.r2-admin \
    --expire-after-days 30 \
    --yes              # mutate; requires --yes when stdin is not a TTY
```text

The CLI prompts for confirmation unless `--yes` is supplied on a
non-interactive stdin. After applying, the CLI re-reads the bucket
configuration to verify the rule was actually written. Unrelated
rules on the bucket are preserved verbatim.

Existing objects older than the chosen retention may become eligible
for deletion after activation. Removal of the rule does *not*
restore objects that were already expired by it.

### `remove` — drop the managed rule

```bash
vastai-gpu-runner r2-lifecycle remove \
    --bucket my-bucket \
    --prefix project/batches/ \
    --credentials-file ~/.cloud-credentials.r2-admin \
    --dry-run
vastai-gpu-runner r2-lifecycle remove \
    --bucket my-bucket \
    --prefix project/batches/ \
    --credentials-file ~/.cloud-credentials.r2-admin \
    --yes
```text

The managed rule is identified by a deterministic rule ID derived
from the bucket and prefix. Removing the rule does not affect
unrelated lifecycle rules on the bucket.

### `R2Sink.cleanup_batch()` is independent

`R2Sink.cleanup_batch(batch_id)` immediately deletes all R2 objects
under one batch prefix. It does *not* consult or replace the
lifecycle configuration. Use it for one-off cleanup; use
`r2-lifecycle apply` for ongoing retention policy.

### Rsync recovery is best-effort

The orchestrator's rsync fallback for instances whose R2 DONE marker
is missing remains best-effort. Polling currently starts only after
the parallel deployment phase completes, so a fast worker can finish
before the first poll tick. See
`docs/architecture-r2-collection-handshake.md` for the planned
longer-term bounded-teardown protocol.
