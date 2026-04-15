# Extending vastai-gpu-runner

## Adding a new cloud provider

Implement the `CloudRunner` interface. The base class provides `run_full_cycle()` which handles retry logic, machine deduplication, and error recovery — your provider just implements the individual lifecycle steps.

```python
from vastai_gpu_runner.runner import CloudRunner
from vastai_gpu_runner.types import CloudInstance, DeploymentConfig, InstanceStatus, Provider

class RunPodRunner(CloudRunner):
    def search_offers(self, **kwargs):
        """Query RunPod marketplace for GPU pods."""
        import runpod
        pods = runpod.get_gpus()
        return [{"id": p["id"], "gpu_name": p["displayName"], ...} for p in pods]

    def create_instance(self, offer):
        """Create a RunPod pod from an offer."""
        import runpod
        pod = runpod.create_pod(
            name="my-job",
            image_name=self.docker_image,
            gpu_type_id=offer["id"],
        )
        return CloudInstance(
            provider=Provider.RUNPOD,
            instance_id=pod["id"],
            gpu_model=offer["gpu_name"],
            status=InstanceStatus.CREATING,
        )

    def wait_for_boot(self, instance):
        """Wait for pod to reach RUNNING state."""
        import runpod, time
        deadline = time.time() + self.config.boot_timeout_seconds
        while time.time() < deadline:
            pod = runpod.get_pod(instance.instance_id)
            if pod["desiredStatus"] == "RUNNING" and pod.get("runtime"):
                instance.ssh_host = pod["runtime"]["ports"][0]["ip"]
                instance.ssh_port = int(pod["runtime"]["ports"][0]["publicPort"])
                instance.status = InstanceStatus.RUNNING
                return True
            time.sleep(5)
        return False

    def verify_gpu(self, instance):
        """Verify GPU via SSH."""
        from vastai_gpu_runner.ssh import ssh_cmd
        rc, output = ssh_cmd(instance, "nvidia-smi")
        return rc == 0

    def deploy_files(self, instance, files):
        """Upload files via SCP."""
        from vastai_gpu_runner.ssh import scp_upload
        for remote_name, local_path in files.items():
            if not scp_upload(instance, local_path, f"/workspace/{remote_name}"):
                return False
        return True

    def setup_environment(self, instance):
        """RunPod images are pre-configured — skip setup."""
        return True

    def launch_worker(self, instance):
        """Launch worker via SSH."""
        from vastai_gpu_runner.ssh import ssh_cmd
        rc, _ = ssh_cmd(instance, "cd /workspace && nohup bash worker.sh > worker.log 2>&1 &")
        return rc == 0

    def check_progress(self, instance):
        """Check for DONE file."""
        from vastai_gpu_runner.ssh import ssh_cmd
        rc, _ = ssh_cmd(instance, "test -f /workspace/DONE")
        return {"running": rc != 0, "complete": rc == 0}

    def destroy_instance(self, instance):
        """Terminate RunPod pod."""
        import runpod
        runpod.terminate_pod(instance.instance_id)
        instance.status = InstanceStatus.DESTROYED
        return True
```

Usage is identical to `VastaiRunner`:

```python
runner = RunPodRunner(DeploymentConfig(gpu_model="RTX_4090"))
result = runner.run_full_cycle(files, output_dir, max_retries=3)
```

## Custom storage backend

Subclass `R2Sink` to add project-specific defaults or methods:

```python
from vastai_gpu_runner.storage.r2 import R2Sink as R2SinkBase

class MyProjectSink(R2SinkBase):
    """R2 sink with project defaults."""

    def __init__(self):
        super().__init__(bucket="my-bucket", prefix="my-project/batches")

    def upload_model(self, batch_id, model_path):
        """Custom: upload a trained model checkpoint."""
        key = f"{self.prefix}/{batch_id}/model.pt"
        self._client.upload_file(str(model_path), self.bucket, key)

    def download_tensorboard(self, batch_id, local_dir):
        """Custom: download TensorBoard logs."""
        return self.download_job(batch_id, "tensorboard", local_dir)
```

For a completely different storage backend (e.g. GCS, plain S3), you'd implement the same interface from scratch rather than subclassing — `R2Sink` is tightly coupled to boto3's S3 API.

## Custom worker with extra preflight gates

```python
from pathlib import Path
from vastai_gpu_runner.worker.base import BaseWorker

class InferenceWorker(BaseWorker):
    """Worker that downloads a model before running inference."""

    def __init__(self, workspace: Path, model_url: str):
        super().__init__(workspace, min_gpu_memory_mib=20000)
        self.model_url = model_url

    def preflight_gates(self):
        return [self._check_r2, self._download_model]

    def _download_model(self) -> bool:
        """Download model weights if not cached."""
        model_path = self.workspace / "model.pt"
        if model_path.exists():
            return True
        import subprocess
        result = subprocess.run(
            ["wget", "-q", "-O", str(model_path), self.model_url],
            timeout=600, check=False,
        )
        return result.returncode == 0 and model_path.exists()

    def run_workload(self) -> int:
        import subprocess
        return subprocess.run(
            ["python", "infer.py", "--model", "model.pt", "--input", "data/"],
            timeout=7200, check=False,
        ).returncode

    def upload_results(self):
        """Upload predictions directory to R2."""
        import subprocess
        r2_script = self.workspace / "r2_upload.py"
        if r2_script.exists():
            subprocess.run(
                ["python", str(r2_script), "--done"],
                timeout=300, check=False,
            )
```

## Building a batch orchestrator

`BatchOrchestrator[UnitT]` is the template-method ABC above `CloudRunner`. Subclass it when you need to deploy many GPU units in parallel with crash-recovery checkpoints, R2-first polling, preemption handling, and per-unit retry accounting. The ABC handles the lifecycle loop; you provide the domain hooks over your own batch-state type (`ShardState` or `JobState`).

```python
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from vastai_gpu_runner import BatchOrchestrator, CloudRunner
from vastai_gpu_runner.providers.vastai import VastaiRunner
from vastai_gpu_runner.state import BatchState, ShardState
from vastai_gpu_runner.storage.r2 import R2Sink
from vastai_gpu_runner.types import CloudInstance, DeploymentConfig


class MyShardOrchestrator(BatchOrchestrator[ShardState]):
    """Orchestrator over a shard-based BatchState."""

    def __init__(
        self,
        state: BatchState,
        state_path: Path,
        output_dir: Path,
        shard_inputs: dict[int, dict[str, Path]],
        r2_sink: R2Sink,
    ) -> None:
        self._state = state
        self._state_path = state_path
        self._output_dir = output_dir
        self._shard_inputs = shard_inputs
        super().__init__(
            runner_factory=lambda: VastaiRunner(DeploymentConfig()),
            label_prefix=f"myproject-{state.batch_id[:8]}",
            r2_sink=r2_sink,
            r2_batch_id=state.batch_id,
            budget_usd=50.0,
            max_retries=2,
        )

    # -- State iteration ---------------------------------------------------

    def iter_pending_units(self) -> Iterable[ShardState]:
        return list(self._state.pending_shards)

    def iter_active_units(self) -> Iterable[ShardState]:
        return list(self._state.active_shards)

    def iter_failed_units(self) -> Iterable[ShardState]:
        return list(self._state.failed_shards)

    def iter_completed_units(self) -> Iterable[ShardState]:
        return list(self._state.downloaded_shards)

    def save_state(self) -> None:
        self._state.save(self._state_path)

    def unit_key(self, unit: ShardState) -> str:
        return str(unit.shard_id)

    def unit_label(self, unit: ShardState) -> str:
        return f"shard-{unit.shard_id}"

    # -- Domain hooks ------------------------------------------------------

    def build_unit_payload(self, unit: ShardState) -> dict[str, Path]:
        return self._shard_inputs[unit.shard_id]

    def reconstruct_instance(self, unit: ShardState) -> CloudInstance:
        return CloudInstance(
            instance_id=unit.instance_id,
            ssh_host=unit.ssh_host,
            ssh_port=unit.ssh_port,
            cost_per_hour=unit.cost_per_hour,
        )

    def collect_unit_results(self, unit: ShardState, instance: CloudInstance) -> bool:
        local = self._output_dir / f"shard_{unit.shard_id}"
        # Consumer decides whether to use rsync, R2, or a hybrid.
        return bool(self._r2_sink and self._r2_sink.download_shard(
            self._state.batch_id, unit.shard_id, local,
        ))

    def unit_is_done_in_r2(self, unit: ShardState) -> bool:
        if not self._r2_sink:
            return False
        return self._r2_sink.is_shard_done(self._state.batch_id, unit.shard_id)

    def classify_failure(self, unit: ShardState, error: str) -> Literal["retry", "fatal"]:
        if "input file missing" in error or "preflight" in error:
            return "fatal"
        return "retry"

    # -- Event callbacks ---------------------------------------------------

    def on_unit_deployed(self, unit: ShardState, instance: CloudInstance) -> None:
        unit.instance_id = instance.instance_id
        unit.ssh_host = instance.ssh_host
        unit.ssh_port = instance.ssh_port
        unit.cost_per_hour = instance.cost_per_hour
        unit.status = "deployed"
        self.save_state()

    def on_unit_failed(self, unit: ShardState, reason: str) -> None:
        unit.status = "failed"
        unit.failure_reason = reason
        unit.retry_count += 1
        self.save_state()

    def on_unit_completed(self, unit: ShardState) -> None:
        unit.status = "downloaded"
        self.save_state()

    def on_unit_preempted(self, unit: ShardState) -> None:
        unit.instance_id = ""
        unit.status = "pending"
        unit.retry_count += 1
        self.save_state()
```

Call `orch.run()` to drive the full lifecycle. Everything else — parallel deploys, exponential-backoff polling, zombie sweeps, budget enforcement, R2-first completion checks, silent-crash detection, `max_retries` enforcement — is inherited.

The same pattern works for job-based workloads: substitute `JobState` for `ShardState`, iterate over `JobBatchState.pending_jobs` / `.active_jobs` / `.completed_jobs`, and map `on_unit_completed` to `status="completed"` or `"downloaded"` to match your persistence format.
