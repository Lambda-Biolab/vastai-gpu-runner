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
