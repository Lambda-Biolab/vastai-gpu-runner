"""Vast.ai marketplace runner implementation.

Implements the CloudRunner interface for Vast.ai's GPU marketplace.
Encodes all UTI-project deployment lessons as default behavior.

Requires: ``pip install vastai`` or the ``vastai`` CLI tool.

Usage::

    from vastai_gpu_runner.providers.vastai import VastaiRunner

    runner = VastaiRunner(allowed_images=frozenset({"my/image:latest"}))
    result = runner.run_full_cycle(files, output_dir)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from pathlib import Path

from vastai_gpu_runner.runner import CloudRunner
from vastai_gpu_runner.ssh import scp_download, scp_upload, ssh_cmd
from vastai_gpu_runner.types import (
    CloudInstance,
    DeploymentConfig,
    InstanceStatus,
    Provider,
)

logger = logging.getLogger(__name__)

# Vast.ai GPU name mapping
GPU_NAME_MAP: dict[str, str] = {
    "RTX_3090": "RTX 3090",
    "RTX_4090": "RTX 4090",
    "RTX_5090": "RTX 5090",
}

# Default Docker image (bare CUDA runtime)
DEFAULT_IMAGE = "nvidia/cuda:12.4.0-devel-ubuntu22.04"

# Minimum GPU VRAM in MiB
MIN_GPU_VRAM_MIB = 20_000


def _get_image_cuda_version(image: str) -> str:
    """Extract required CUDA version from a Docker image.

    Tries ``docker inspect`` labels first, falls back to parsing the image
    tag (e.g. ``cuda:12.4.1`` -> ``"12.4"``).

    Args:
        image: Docker image name with tag.

    Returns:
        CUDA major.minor version string (e.g. ``"12.4"``).
    """
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                '{{index .Config.Labels "cuda_version"}}',
                image,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        version = result.stdout.strip()
        if version and version != "<no value>":
            parts = version.split(".")[:2]
            return ".".join(parts)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: parse tag string for cuda version pattern
    match = re.search(r"cuda[:\-](\d+\.\d+)", image)
    if match:
        return match.group(1)
    return "12.4"


def vastai_cmd(args: list[str], *, timeout: int = 30) -> str:
    """Run a vastai CLI command.

    Args:
        args: Command arguments (after 'vastai').
        timeout: Command timeout in seconds.

    Returns:
        stdout text.

    Raises:
        RuntimeError: If command fails.
    """
    cmd = ["vastai", *args]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            msg = f"vastai {' '.join(args)} failed: {result.stderr.strip()}"
            raise RuntimeError(msg)
        return result.stdout.strip()
    except FileNotFoundError as exc:
        msg = "vastai CLI not installed. Install with: pip install vastai"
        raise RuntimeError(msg) from exc
    except subprocess.TimeoutExpired as exc:
        msg = f"vastai {' '.join(args)} timed out after {timeout}s"
        raise RuntimeError(msg) from exc


def verify_instance_ownership(
    instance_id: str,
    *,
    allowed_images: frozenset[str] | None = None,
) -> bool:
    """Check that a Vast.ai instance belongs to the caller before destruction.

    Queries the Vast.ai API for the instance details and verifies the Docker
    image is in the allowed set.  This prevents accidental deletion of
    instances belonging to other projects sharing the same Vast.ai account.

    Args:
        instance_id: Vast.ai instance ID.
        allowed_images: Set of Docker image names considered safe to destroy.
            If None, ownership check is skipped (all instances allowed).

    Returns:
        True if the instance is safe to destroy.
    """
    if allowed_images is None:
        return True

    try:
        raw = vastai_cmd(["show", "instances", "--raw"], timeout=15)
        instances = json.loads(raw)
    except (RuntimeError, json.JSONDecodeError) as exc:
        logger.warning(
            "Cannot verify ownership of instance %s (API error: %s) — refusing to destroy",
            instance_id,
            exc,
        )
        return False

    inst = _find_instance(instances, instance_id)
    if inst is None:
        logger.info("Instance %s not found in account (already destroyed?)", instance_id)
        return True
    return _image_is_allowed(inst, instance_id, allowed_images)


def _find_instance(
    instances: list[dict[str, object]],
    instance_id: str,
) -> dict[str, object] | None:
    """Find one instance by ID. Returns None if not present."""
    for inst in instances:
        if str(inst.get("id")) == str(instance_id):
            return inst
    return None


def _image_is_allowed(
    inst: dict[str, object],
    instance_id: str,
    allowed_images: frozenset[str],
) -> bool:
    """Return True if the instance's image matches the allowlist."""
    image = str(inst.get("image_uuid", ""))
    label = str(inst.get("label", ""))
    if image in allowed_images:
        return True
    if any(img.split(":")[0] in image for img in allowed_images):
        return True
    logger.error(
        "BLOCKED: instance %s belongs to another project (image=%s, label=%s). Will NOT destroy.",
        instance_id,
        image,
        label,
    )
    return False


class VastaiRunner(CloudRunner):
    """Vast.ai marketplace runner with hardened deployment.

    Args:
        config: Deployment configuration.
        allowed_images: Docker images owned by this project.
            destroy_instance() refuses to destroy instances running
            images not in this set. Pass None to skip ownership checks.
        docker_image: Docker image to use for new instances.
        min_gpu_vram_mib: Minimum GPU VRAM required (default 20 GB).
    """

    def __init__(
        self,
        config: DeploymentConfig | None = None,
        *,
        allowed_images: frozenset[str] | None = None,
        docker_image: str = DEFAULT_IMAGE,
        min_gpu_vram_mib: int = MIN_GPU_VRAM_MIB,
        setup_commands: list[str] | None = None,
    ) -> None:
        """Initialize Vast.ai runner with deployment config and safety guards."""
        super().__init__(config)
        self.allowed_images = allowed_images
        self.docker_image = docker_image
        self.min_gpu_vram_mib = min_gpu_vram_mib
        self._setup_commands = setup_commands or []

    def search_offers(self, **kwargs: object) -> list[dict[str, object]]:
        """Search Vast.ai marketplace for matching GPU offers."""
        docker_img = str(kwargs.get("docker_image", self.docker_image))
        gpu_name = GPU_NAME_MAP.get(self.config.gpu_model, self.config.gpu_model)
        cuda_ver = _get_image_cuda_version(docker_img)
        logger.info("Filtering Vast.ai offers for CUDA >= %s (from image)", cuda_ver)
        query = (
            f'gpu_name="{gpu_name}" '
            f"num_gpus=1 "
            f"rentable=true "
            f"cuda_max_good>={cuda_ver} "
            f"dph<={self.config.max_cost_per_hour} "
            f"inet_down>={self.config.min_network_mbps} "
            f"reliability>={self.config.min_reliability}"
        )

        try:
            output = vastai_cmd(
                ["search", "offers", query, "--order", "dph", "--limit", "20", "--raw"],
                timeout=30,
            )
            offers: list[dict[str, object]] = json.loads(output)
            logger.info("Found %d Vast.ai offers for %s", len(offers), gpu_name)
            return offers
        except (RuntimeError, json.JSONDecodeError) as exc:
            logger.error("Failed to search Vast.ai offers: %s", exc)
            return []

    def create_instance(self, offer: dict[str, object]) -> CloudInstance:
        """Create a Vast.ai instance from an offer."""
        offer_id = str(offer.get("id", ""))
        label = f"gpu-runner-{int(time.time()) % 100000}"

        try:
            output = vastai_cmd(
                [
                    "create",
                    "instance",
                    offer_id,
                    "--image",
                    self.docker_image,
                    "--disk",
                    str(self.config.min_disk_gb),
                    "--label",
                    label,
                    "--raw",
                ],
                timeout=30,
            )

            data = json.loads(output)
            instance_id = str(data.get("new_contract", data.get("id", offer_id)))

            return CloudInstance(
                provider=Provider.VASTAI,
                instance_id=instance_id,
                gpu_model=str(offer.get("gpu_name", self.config.gpu_model)),
                cost_per_hour=float(str(offer.get("dph_total", 0.0))),
                status=InstanceStatus.CREATING,
                label=label,
            )
        except (RuntimeError, json.JSONDecodeError, KeyError) as exc:
            msg = f"Failed to create Vast.ai instance: {exc}"
            raise RuntimeError(msg) from exc

    def wait_for_boot(self, instance: CloudInstance) -> bool:
        """Wait for Vast.ai instance to reach 'running' status."""
        deadline = time.time() + self.config.boot_timeout_seconds
        instance.status = InstanceStatus.BOOTING

        while time.time() < deadline:
            try:
                output = vastai_cmd(
                    ["show", "instance", instance.instance_id, "--raw"],
                    timeout=15,
                )
                data = json.loads(output)
                status = data.get("actual_status", "")

                if status == "running":
                    ssh_host = data.get("ssh_host", "")
                    ssh_port = int(data.get("ssh_port", 22))
                    if ssh_host:
                        instance.ssh_host = ssh_host
                        instance.ssh_port = ssh_port
                        instance.status = InstanceStatus.RUNNING
                        logger.info(
                            "Instance %s is running (SSH: %s:%d)",
                            instance.instance_id,
                            ssh_host,
                            ssh_port,
                        )
                        return True

            except (RuntimeError, json.JSONDecodeError):
                pass

            time.sleep(5)

        logger.warning(
            "Instance %s stuck in boot after %ds",
            instance.instance_id,
            self.config.boot_timeout_seconds,
        )
        # Caller (_try_one_offer) now owns the cleanup path: it calls
        # capture_deploy_failure_diagnostics BEFORE destroy_instance so
        # subclasses can pull ``vastai logs`` / ssh diagnostics while
        # the instance still exists. Previously we destroyed here
        # inline, which erased the container before diagnostics could
        # run and made boot-timeout failures unobservable.
        instance.status = InstanceStatus.FAILED
        return False

    def verify_gpu(self, instance: CloudInstance) -> bool:
        """Verify GPU is accessible and has sufficient VRAM."""
        deadline = time.time() + self.config.gpu_verify_timeout

        while time.time() < deadline:
            rc, output = ssh_cmd(
                instance,
                "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits",
            )
            if rc == 0:
                try:
                    parts = output.strip().split("\n")[0].split(",")
                    mem_used = int(parts[0].strip())
                    mem_total = int(parts[1].strip())
                    if mem_total >= self.min_gpu_vram_mib:
                        logger.info(
                            "GPU verified: %d/%d MiB (used/total)",
                            mem_used,
                            mem_total,
                        )
                        return True
                    logger.warning(
                        "GPU VRAM too low: %d MiB < %d MiB required",
                        mem_total,
                        self.min_gpu_vram_mib,
                    )
                    return False
                except (ValueError, IndexError):
                    pass
            time.sleep(3)

        logger.warning("GPU verification failed for instance %s", instance.instance_id)
        return False

    def deploy_files(
        self,
        instance: CloudInstance,
        files: dict[str, Path],
    ) -> bool:
        """Upload files via SCP."""
        ssh_cmd(instance, f"mkdir -p {self.config.workspace_dir}")

        for remote_name, local_path in files.items():
            if not local_path.exists():
                logger.warning("Local file not found: %s", local_path)
                continue

            remote_path = f"{self.config.workspace_dir}/{remote_name}"
            if not scp_upload(instance, local_path, remote_path):
                return False

        return True

    def setup_environment(self, instance: CloudInstance) -> bool:
        """Run environment setup commands on the instance.

        If ``setup_commands`` was provided at construction, runs those.
        Otherwise, if ``conda_env_spec`` is set in the config, installs
        micromamba + creates a conda environment. If neither is set,
        skips setup entirely (assumes Docker image is ready).

        Override this method for fully custom setup logic.
        """
        commands = self._setup_commands
        if not commands and not self.config.conda_env_spec:
            logger.info("No setup commands or conda_env_spec — skipping environment setup")
            return True

        if not commands:
            # Default: micromamba + conda env
            commands = [
                "apt-get update -qq && apt-get install -y -qq bzip2 ca-certificates",
                "curl -kL -o /tmp/mm.tar.bz2 "
                '"https://micro.mamba.pm/api/micromamba/linux-64/latest"',
                "mkdir -p /opt/micromamba"
                " && tar -xjf /tmp/mm.tar.bz2 -C /opt/micromamba --strip-components=1",
                "/opt/micromamba/bin/micromamba create -y -n env"
                f" -c conda-forge {self.config.conda_env_spec}",
            ]

        for cmd in commands:
            rc, output = ssh_cmd(instance, cmd, timeout=600)
            if rc != 0:
                logger.error("Setup command failed: %s -> %s", cmd[:50], output[:200])
                return False
            logger.debug("Setup OK: %s", cmd[:50])

        logger.info("Environment setup complete on %s", instance.instance_id)
        return True

    def launch_worker(self, instance: CloudInstance) -> bool:
        """Launch the worker script on the instance."""
        ws = self.config.workspace_dir
        worker_script = self.config.worker_script

        # Check for duplicate workers
        rc, output = ssh_cmd(instance, f"pgrep -f {worker_script}")
        if rc == 0 and output.strip():
            logger.warning("Worker already running on %s — skipping launch", instance.instance_id)
            return True

        launch_cmd = f"cd {ws} && nohup bash {worker_script} > {ws}/worker.log 2>&1 &"

        rc, _ = ssh_cmd(instance, launch_cmd, timeout=30)
        if rc != 0:
            logger.error("Worker launch failed on %s", instance.instance_id)
            return False

        time.sleep(5)
        rc, output = ssh_cmd(instance, f"pgrep -f {worker_script}")
        if rc != 0:
            logger.error("Worker process not found after launch on %s", instance.instance_id)
            return False

        logger.info("Worker launched on %s", instance.instance_id)
        return True

    def check_progress(self, instance: CloudInstance) -> dict[str, object]:
        """Check worker progress via DONE file and PID liveness."""
        ws = self.config.workspace_dir

        rc, _ = ssh_cmd(instance, f"test -f {ws}/DONE")
        if rc == 0:
            return {"running": False, "complete": True}

        # Check if worker PID is alive (detects silent preemption)
        rc_pid, pid_str = ssh_cmd(instance, f"cat {ws}/worker.pid 2>/dev/null", timeout=5)
        if rc_pid == 0 and pid_str.strip().isdigit():
            rc_alive, _ = ssh_cmd(instance, f"kill -0 {pid_str.strip()} 2>/dev/null", timeout=5)
            if rc_alive != 0:
                logger.warning(
                    "Worker PID %s is dead on %s but no DONE file — silent crash",
                    pid_str.strip(),
                    instance.instance_id,
                )
                return {
                    "running": False,
                    "complete": False,
                    "worker_dead": True,
                    "log_tail": f"Worker PID {pid_str.strip()} dead, no DONE file",
                }

        rc, output = ssh_cmd(instance, f"tail -3 {ws}/worker.log", timeout=10)
        return {
            "running": True,
            "complete": False,
            "log_tail": output,
        }

    def list_remote_files(self, instance: CloudInstance) -> list[str]:
        """List all files in workspace."""
        ws = self.config.workspace_dir
        rc, output = ssh_cmd(instance, f"ls -1 {ws}/", timeout=10)
        if rc != 0:
            return []
        return [f.strip() for f in output.splitlines() if f.strip()]

    def download_file(
        self,
        instance: CloudInstance,
        remote_name: str,
        local_path: Path,
    ) -> bool:
        """Download a single file via SCP."""
        remote_path = f"{self.config.workspace_dir}/{remote_name}"
        return scp_download(instance, remote_path, local_path)

    def capture_deploy_failure_diagnostics(
        self,
        instance: CloudInstance,
        error: str,
        attempt: int,
    ) -> None:
        """Pull ``vastai logs`` + SSH dmesg/nvidia-smi before destroy.

        Vast.ai does not retain container logs after ``destroy_instance``
        (``vastai logs <id>`` returns 404 on the underlying docker
        container). This is our one chance to capture why a deploy gate
        failed. Saves to ``batch_diagnostics/deploy__{unit_or_id}_{ts}.log``
        under the current working directory — mirrors the layout used by
        ``BatchOrchestrator.capture_preempt_diagnostics``.

        Always swallows exceptions; a diagnostic capture must NEVER block
        the destroy that follows.
        """
        try:
            diag_dir = Path.cwd() / "batch_diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            iid = instance.instance_id or "unknown"
            out_path = diag_dir / f"deploy__{iid}_{timestamp}.log"

            sections: list[str] = [
                f"# deploy-failure diagnostics for instance {iid}",
                f"# attempt: {attempt}",
                f"# error: {error}",
                f"# ssh: {instance.ssh_user}@{instance.ssh_host}:{instance.ssh_port}",
                f"# captured_at: {timestamp}",
            ]

            # vastai-level container logs (fetched from Vast's log storage,
            # which holds content for some seconds after container stop).
            try:
                vlogs = vastai_cmd(["logs", iid], timeout=30)
                sections.extend(["", "## vastai logs ##", vlogs])
            except Exception as exc:
                sections.extend(["", "## vastai logs FAILED ##", str(exc)])

            # SSH-level diagnostics: workspace worker.log if it exists,
            # plus dmesg tail + nvidia-smi for kernel/driver state.
            ws = self.config.workspace_dir
            rc, output = ssh_cmd(
                instance,
                (
                    f"cat {ws}/worker.log 2>/dev/null; "
                    f"echo '---DMESG---'; dmesg -T 2>/dev/null | tail -50; "
                    f"echo '---NVIDIA-SMI---'; nvidia-smi 2>&1 | head -30; "
                    f"echo '---DF---'; df -h {ws} 2>/dev/null || df -h"
                ),
                timeout=20,
            )
            sections.extend(
                [
                    "",
                    f"## ssh diagnostics (rc={rc}) ##",
                    output or "(empty)",
                ]
            )

            out_path.write_text("\n".join(sections) + "\n")
            logger.info(
                "Deploy-failure diagnostics captured (%d sections) → %s",
                len(sections),
                out_path,
            )
        except Exception as exc:
            logger.warning(
                "capture_deploy_failure_diagnostics swallowed exception: %s",
                exc,
            )

    def destroy_instance(self, instance: CloudInstance) -> bool:
        """Destroy a Vast.ai instance (with ownership safety guard).

        Verifies the instance belongs to this project before destruction
        if ``allowed_images`` was set on the runner.
        """
        if not verify_instance_ownership(instance.instance_id, allowed_images=self.allowed_images):
            logger.error(
                "REFUSED to destroy instance %s — ownership check failed.",
                instance.instance_id,
            )
            return False

        try:
            vastai_cmd(["destroy", "instance", instance.instance_id], timeout=15)
            instance.status = InstanceStatus.DESTROYED
            logger.info("Destroyed instance %s", instance.instance_id)
        except RuntimeError as exc:
            logger.error("CLI destroy failed for %s: %s", instance.instance_id, exc)

        # Belt-and-suspenders: REST stop -> DELETE -> verify -> retry.
        self._rest_destroy(instance)
        instance.status = InstanceStatus.DESTROYED
        return True

    def _rest_destroy(self, instance: CloudInstance) -> None:
        """Force-destroy via REST API (handles stuck/resurrected instances).

        Four-step flow: read api key → force-stop → DELETE up to 3 times →
        verify and re-destroy if resurrected.
        """
        api_key = _read_vastai_api_key()
        if not api_key:
            return
        try:
            hdrs = {"Authorization": f"Bearer {api_key}"}
            _rest_stop(instance.instance_id, hdrs)
            _rest_delete_with_retries(instance.instance_id, hdrs)
            _rest_verify_and_redestroy(instance.instance_id, hdrs)
        except Exception as exc:
            logger.warning("REST destroy failed for %s: %s", instance.instance_id, exc)


def _read_vastai_api_key() -> str:
    """Read the Vast.ai API key from the standard config paths."""
    from pathlib import Path as _Path

    for kp in (
        _Path("~/.config/vastai/vast_api_key").expanduser(),
        _Path("~/.vast_api_key").expanduser(),
    ):
        if kp.exists():
            return kp.read_text().strip()
    return ""


def _rest_stop(instance_id: str, hdrs: dict[str, str]) -> None:
    """Force-stop an instance via the Vast.ai REST API (kills stuck Docker pulls)."""
    import requests

    base = "https://console.vast.ai/api/v0/instances"
    requests.put(
        f"{base}/{instance_id}/",
        headers={**hdrs, "Content-Type": "application/json"},
        json={"state": "stopped"},
        timeout=10,
    )
    time.sleep(2)


def _rest_delete_with_retries(instance_id: str, hdrs: dict[str, str]) -> None:
    """DELETE an instance up to 3 times, pausing 3s between attempts."""
    import requests

    base = "https://console.vast.ai/api/v0/instances"
    for del_attempt in range(3):
        resp = requests.delete(f"{base}/{instance_id}/", headers=hdrs, timeout=15)
        if resp.status_code in (200, 204, 404):
            logger.info("REST DELETE %s: %d", instance_id, resp.status_code)
            return
        logger.warning(
            "REST DELETE %s returned %d (attempt %d/3)",
            instance_id,
            resp.status_code,
            del_attempt + 1,
        )
        if del_attempt < 2:
            time.sleep(3)


def _rest_verify_and_redestroy(instance_id: str, hdrs: dict[str, str]) -> None:
    """Wait 5s, then re-destroy if the instance was resurrected."""
    import requests

    base = "https://console.vast.ai/api/v0/instances"
    time.sleep(5)
    verify = requests.get(f"{base}/{instance_id}/", headers=hdrs, timeout=10)
    if verify.status_code != 200:
        return
    vstatus = verify.json().get("actual_status", "")
    if vstatus in ("", "destroyed"):
        return
    logger.warning(
        "Instance %s resurrected as '%s' — sending stop+delete again",
        instance_id,
        vstatus,
    )
    _rest_stop(instance_id, hdrs)
    time.sleep(3)
    requests.delete(f"{base}/{instance_id}/", headers=hdrs, timeout=10)
