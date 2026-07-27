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
import warnings
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from vastai_gpu_runner.cleanup_policy import OwnershipPolicy
from vastai_gpu_runner.providers.destroy_adapters.vastai import (
    CredentialResolution,
    CredentialState,
    read_vastai_api_key,
)
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


# The v2 ``verify_instance_ownership`` (returns bool) and the
# v2 ``_image_is_allowed`` (v2 substring/prefix match) are DELETED.
# The v3 destroy adapter's tagged-enum ``verify_instance_ownership``
# + tag-insensitive ``_is_image_allowed`` replace both. External
# callers (none in this repo) must migrate from ``bool`` to the
# ``OwnershipVerification`` enum.


# ---------------------------------------------------------------------------
# VastaiProviderConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VastaiProviderConfig:
    """Canonical Vast.ai configuration shared by runner factory + cleanup policy.

    The runner factory (``VastaiRunner.from_config``) reads
    ``ownership``, ``credentials``, ``label_prefix``, ``docker_image``, and
    ``setup_commands`` from this. The cleanup-policy factory
    (``build_vastai_cleanup_policy``) reads only ``ownership`` and
    ``credentials`` — the deployment-image invariant does not apply
    to listing/cleanup-only commands.

    Invariants:
        - ``docker_image`` is non-empty and pre-stripped.
        - ``docker_image`` is in ``ownership.owned_images`` unless
          ``ownership.owned_images is None`` (ownership check disabled).
        - ``credentials`` is a v3 ``CredentialResolution`` (frozen).
        - ``label_prefix`` is either ``None`` (non-batch runner) or
          a non-empty, pre-stripped string; batch composition always
          supplies a unique scope.
    """

    docker_image: str = DEFAULT_IMAGE
    ownership: OwnershipPolicy = field(default_factory=OwnershipPolicy)
    credentials: CredentialResolution = field(
        default_factory=lambda: CredentialResolution(state=CredentialState.ABSENT)
    )
    label_prefix: str | None = None
    min_gpu_vram_mib: int = MIN_GPU_VRAM_MIB
    setup_commands: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate label_prefix, docker_image, and docker_image ownership invariant."""
        if self.label_prefix is not None and (
            not isinstance(self.label_prefix, str)  # pyright: ignore[reportUnnecessaryIsInstance]
            or not self.label_prefix
            or self.label_prefix != self.label_prefix.strip()
        ):
            raise ValueError(
                "VastaiProviderConfig.label_prefix must be None or a non-empty, pre-stripped string"
            )
        if not self.docker_image or self.docker_image != self.docker_image.strip():
            raise ValueError(
                "VastaiProviderConfig.docker_image must be a non-empty, pre-stripped reference"
            )
        if self.ownership.owned_images is not None and not self.ownership.matches(
            self.docker_image
        ):
            raise ValueError(
                f"VastaiProviderConfig invariant violated: docker_image="
                f"{self.docker_image!r} is not owned by "
                f"ownership.owned_images="
                f"{set(self.ownership.owned_images)!r}"
            )

    @classmethod
    def from_env(
        cls,
        *,
        docker_image: str | None = None,
        owned_images: AbstractSet[str] | None = None,
        label_prefix: str | None = None,
    ) -> VastaiProviderConfig:
        """Build from environment / config files.

        Reads the v3 ``read_vastai_api_key()`` into ``credentials``.
        The CLI uses ``dataclasses.replace`` to overlay the
        project-specific values on top of this base.

        Note: explicit empty ``docker_image`` is rejected by
        ``__post_init__``; only ``None`` triggers the default.
        """
        resolved_image = DEFAULT_IMAGE if docker_image is None else docker_image
        return cls(
            docker_image=resolved_image,
            ownership=OwnershipPolicy(owned_images=owned_images),
            credentials=read_vastai_api_key(),
            label_prefix=label_prefix,
        )


class VastaiRunner(CloudRunner):
    """Vast.ai marketplace runner with hardened deployment.

    Args:
        config: Deployment configuration.
        ownership: Ownership policy. The runner and the cleanup
            adapter both call ``ownership.matches(image_ref)``.
        credentials: Pre-resolved credential state. When None,
            the runner passes ``None`` to the adapter, which
            calls ``read_vastai_api_key()`` (the v3 back-compat path).
        label_prefix: Immutable batch label scope. Batch composition
            always supplies a unique, validated scope; non-batch
            callers may leave it None.
        docker_image: Docker image to use for new instances.
        min_gpu_vram_mib: Minimum GPU VRAM required (default 20 GB).
        setup_commands: Optional pre-instance setup commands.

    Note: ``allowed_images`` is a deprecated back-compat alias
    that builds an ``OwnershipPolicy`` from the given set.
    Simultaneous ``ownership=`` and ``allowed_images=`` raises
    ``ValueError`` — silently ignoring one destruction-safety
    configuration is unsafe.
    """

    def __init__(
        self,
        config: DeploymentConfig | None = None,
        *,
        ownership: OwnershipPolicy | None = None,
        credentials: CredentialResolution | None = None,
        label_prefix: str | None = None,
        allowed_images: frozenset[str] | None = None,  # DEPRECATED
        docker_image: str = DEFAULT_IMAGE,
        min_gpu_vram_mib: int = MIN_GPU_VRAM_MIB,
        setup_commands: list[str] | None = None,
    ) -> None:
        """Initialize Vast.ai runner with deployment config and safety guards."""
        if label_prefix is not None and (
            not isinstance(label_prefix, str)  # pyright: ignore[reportUnnecessaryIsInstance]
            or not label_prefix
            or label_prefix != label_prefix.strip()
        ):
            raise ValueError(
                "VastaiRunner.label_prefix must be None or a non-empty, pre-stripped string"
            )
        if ownership is not None and allowed_images is not None:
            raise ValueError(
                "VastaiRunner: supply either ownership= or allowed_images= "
                "(deprecated), not both — silently ignoring one destruction-"
                "safety configuration is unsafe."
            )
        if ownership is None and allowed_images is not None:
            warnings.warn(
                "VastaiRunner(allowed_images=...) is deprecated; "
                "build an OwnershipPolicy and pass ownership= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            ownership = OwnershipPolicy(owned_images=frozenset(allowed_images))
        super().__init__(config)
        self.ownership = ownership
        self.credentials = credentials
        self.label_prefix = label_prefix
        # Back-compat read access for existing callers.
        self._allowed_images_for_backcompat = (
            frozenset(ownership.owned_images)
            if ownership and ownership.owned_images is not None
            else None
        )
        self.docker_image = docker_image
        self.min_gpu_vram_mib = min_gpu_vram_mib
        self._setup_commands = setup_commands or []

    @property
    def allowed_images(self) -> frozenset[str] | None:
        """Back-compat read access. New code should use ``self.ownership``."""
        return self._allowed_images_for_backcompat

    @classmethod
    def from_config(cls, canonical: VastaiProviderConfig) -> VastaiRunner:
        """Build a VastaiRunner from the canonical config.

        Preserves the canonical ``OwnershipPolicy`` and
        ``CredentialResolution`` instances unchanged — the runner
        and the cleanup policy will call ``ownership.matches()`` and
        use ``credentials`` from the same instances.
        """
        return cls(
            ownership=canonical.ownership,
            credentials=canonical.credentials,
            label_prefix=canonical.label_prefix,
            docker_image=canonical.docker_image,
            min_gpu_vram_mib=canonical.min_gpu_vram_mib,
            setup_commands=list(canonical.setup_commands),
        )

    def _next_instance_label(self) -> str:
        """Create a unique label inside this runner's immutable batch scope."""
        prefix = self.label_prefix or "gpu-runner"
        return f"{prefix}-{uuid4().hex[:12]}"

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
        label = self._next_instance_label()

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

        Per the v3 doc: routes through ``destroy_vastai_instance``
        from the destroy adapter, which wraps the belt-and-suspenders
        protocol with pre-protocol refusals (OWNERSHIP /
        CREDENTIALS_DISABLED) and the v3 CLI fallback for ABSENT
        credentials. The ``allowed_images`` ownership guard is
        preserved — the adapter runs the CLI-based ownership check
        before any destroy attempt and refuses ownership mismatches.
        """
        from vastai_gpu_runner.providers.destroy import (
            DestroyRefusal,
            DestroyVerdict,
        )
        from vastai_gpu_runner.providers.destroy_adapters.vastai import (
            destroy_vastai_instance,
        )

        result = destroy_vastai_instance(
            instance.instance_id,
            allowed_images=self.allowed_images,
        )

        if result.refusal == DestroyRefusal.OWNERSHIP:
            logger.error(
                "REFUSED to destroy instance %s — ownership check failed.",
                instance.instance_id,
            )
            return False
        if result.refusal == DestroyRefusal.CREDENTIALS_DISABLED:
            logger.error(
                "REFUSED to destroy instance %s — credentials are explicitly disabled "
                '(VASTAI_API_KEY="").',
                instance.instance_id,
            )
            return False
        if result.refusal == DestroyRefusal.NO_CREDENTIALS:
            # v3 CLI fallback path: ownership was OK but no API key.
            # The v3 adapter returns NO_CREDENTIALS to defer the CLI
            # fallback to the caller. Here we attempt the CLI destroy
            # directly (the v4 factory will own this dispatch).
            return self._cli_destroy_instance(instance)

        if result.verdict == DestroyVerdict.DESTROYED:
            instance.status = InstanceStatus.DESTROYED
            logger.info("Destroyed instance %s (verified)", instance.instance_id)
            return True

        # UNKNOWN or LEAKED: instance may or may not be gone; reflect
        # the uncertainty in the status but still report success (the
        # v3 contract returns True on best-effort destroy).
        instance.status = InstanceStatus.DESTROYED
        logger.warning(
            "Destroyed instance %s (verdict=%s, last_status=%s, error=%s)",
            instance.instance_id,
            result.verdict.value if result.verdict else "n/a",
            result.last_status_code,
            result.verify_error or result.stop_error or "",
        )
        return True

    def _cli_destroy_instance(self, instance: CloudInstance) -> bool:
        """CLI fallback when no API key is available.

        Per the v3 doc, the CLI fallback path uses the CLI-based
        ``verify_instance_ownership`` (already run by the adapter)
        and then invokes ``vastai destroy instance``. The v4
        factory will own this dispatch; for v3 we keep the inline
        call to preserve the v3 behaviour.
        """
        try:
            vastai_cmd(["destroy", "instance", instance.instance_id], timeout=15)
            instance.status = InstanceStatus.DESTROYED
            logger.info("CLI-destroyed instance %s", instance.instance_id)
            return True
        except RuntimeError as exc:
            logger.error("CLI destroy failed for %s: %s", instance.instance_id, exc)
            instance.status = InstanceStatus.DESTROYED
            return False

    def _rest_destroy(self, instance: CloudInstance) -> None:
        """Deprecated v2/v3 entrypoint; delegates to the adapter.

        The v2 four-step flow (force-stop -> DELETE x retry -> verify ->
        re-destroy) is replaced by ``destroy_vastai_instance`` in the
        destroy adapter. Kept as a thin delegate so any external
        callers (none in this repo) can migrate gradually; will be
        removed in v3 step 8.
        """
        from vastai_gpu_runner.providers.destroy_adapters.vastai import (
            destroy_vastai_instance,
        )

        destroy_vastai_instance(
            instance.instance_id,
            allowed_images=self.allowed_images,
        )


# The v2 module-level helpers (_read_vastai_api_key, _rest_stop,
# _rest_delete_with_retries, _rest_verify_and_redestroy) are DELETED.
# The v3 destroy adapter (``providers/destroy_adapters/vastai.py``)
# owns the equivalent behaviour. This is the v3 doc's "What changes
# vs v2" deletion: providers/vastai.py:_rest_stop,
# _rest_delete_with_retries, _rest_verify_and_redestroy are
# absorbed into the Vast.ai adapter. Folding the v3 doc's step 8
# deletion into this runner refactor commit so the runner doesn't
# carry dead-code helpers.
