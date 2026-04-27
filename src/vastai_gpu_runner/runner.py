"""Abstract base class for cloud GPU runners.

``CloudRunner`` defines the provider-agnostic lifecycle interface:
search → create → boot → verify → deploy → setup → launch → poll → download → destroy.

``run_full_cycle`` orchestrates the full lifecycle with retry logic and
machine deduplication across threads.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
from typing import TYPE_CHECKING

from vastai_gpu_runner.types import CloudInstance, DeploymentConfig, DeploymentResult

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)


class CloudRunner:
    """Abstract base class for cloud GPU runners.

    Subclasses must implement the provider-specific methods:
    ``search_offers``, ``create_instance``, ``wait_for_boot``,
    ``verify_gpu``, ``deploy_files``, ``setup_environment``,
    ``launch_worker``, ``check_progress``, and ``destroy_instance``.

    ``run_full_cycle`` orchestrates the full lifecycle with retry logic.
    ``download_all_results`` provides rsync-based bulk download.
    """

    def __init__(self, config: DeploymentConfig | None = None) -> None:
        """Initialize with optional deployment config."""
        self.config = config or DeploymentConfig()

    # -- Provider-specific methods (override in subclasses) ----------------

    def search_offers(self, **kwargs: object) -> list[dict[str, object]]:
        """Search for GPU offers matching the deployment config."""
        return []

    def create_instance(self, offer: dict[str, object]) -> CloudInstance:
        """Create a cloud instance from an offer."""
        raise NotImplementedError

    def wait_for_boot(self, instance: CloudInstance) -> bool:
        """Wait for an instance to reach running status."""
        raise NotImplementedError

    def verify_gpu(self, instance: CloudInstance) -> bool:
        """Verify GPU is functional on the instance."""
        raise NotImplementedError

    def deploy_files(
        self,
        instance: CloudInstance,
        files: dict[str, Path],
    ) -> bool:
        """Upload files to the cloud instance."""
        raise NotImplementedError

    def setup_environment(self, instance: CloudInstance) -> bool:
        """Set up the execution environment on the instance."""
        raise NotImplementedError

    def launch_worker(self, instance: CloudInstance) -> bool:
        """Launch the worker process on the instance."""
        raise NotImplementedError

    def check_progress(self, instance: CloudInstance) -> dict[str, object]:
        """Check worker progress on the instance."""
        raise NotImplementedError

    def list_remote_files(self, instance: CloudInstance) -> list[str]:
        """List files in the remote workspace."""
        return []

    def download_file(
        self,
        instance: CloudInstance,
        remote_name: str,
        local_path: Path,
    ) -> bool:
        """Download a single file from the instance."""
        raise NotImplementedError

    def destroy_instance(self, instance: CloudInstance) -> bool:
        """Destroy a cloud instance."""
        raise NotImplementedError

    # -- Diagnostic hooks (override in subclasses) -------------------------

    def capture_deploy_failure_diagnostics(
        self,
        instance: CloudInstance,
        error: str,
        attempt: int,
    ) -> None:
        """Hook: capture diagnostics on a deploy-phase failure before destroy.

        Called from ``run_full_cycle`` / ``_try_one_offer`` immediately
        before ``destroy_instance`` when a gate in ``_run_gate_chain``
        fails (boot timeout, GPU verify failure, file deploy failure,
        environment setup failure, worker launch failure) OR an
        exception escapes the gate chain.

        Providers like Vast.ai do NOT retain container logs after the
        instance is destroyed — ``vastai logs <id>`` returns 404 on the
        underlying docker container. SSH is still up at this hook point
        (the boot/verify gates passed at least partially), which gives
        the subclass its one chance to pull ``worker.log``, ``dmesg``,
        ``nvidia-smi``, or provider-level container logs before destroy.

        Default implementation: no-op. Override in subclasses (see
        ``vastai_gpu_runner.providers.vastai.VastaiRunner`` for an
        example that SSH-cats the workspace log tail).

        Swallows every exception — a diagnostic capture MUST NEVER
        block destroy, because a leaked instance keeps burning dollars.

        Args:
            instance: The cloud instance that failed to deploy.
            error: Human-readable error message from the failed gate.
            attempt: Zero-indexed attempt number (for logging + filenames).
        """
        _ = instance, error, attempt  # default no-op

    # -- Orchestrated lifecycle --------------------------------------------

    def run_full_cycle(
        self,
        files: dict[str, Path],
        local_output_dir: Path,
        *,
        max_retries: int = 3,
        offers: list[dict[str, object]] | None = None,
        used_machine_ids: set[str] | None = None,
        machine_lock: threading.Lock | object | None = None,
    ) -> DeploymentResult:
        """Run the full deploy -> boot -> verify -> launch -> poll -> download cycle.

        Retries up to *max_retries* times on transient failures (boot timeout,
        GPU verification failure, worker crash).  Each attempt picks the next
        cheapest offer from the pre-filtered list.

        Args:
            files: Mapping of remote relative path -> local absolute path.
            local_output_dir: Where to download results.
            max_retries: Maximum deployment attempts.
            offers: Pre-fetched offer list (avoids re-querying marketplace).
            used_machine_ids: Shared set of machine IDs already claimed by
                other threads.  Updated under *machine_lock* on success.
            machine_lock: Lock protecting *used_machine_ids*.

        Returns:
            DeploymentResult with success status, instance, and any error.
        """
        del local_output_dir  # reserved for future use
        if offers is None:
            offers = self.search_offers()
        if not offers:
            return DeploymentResult(success=False, error="No GPU offers available")

        last_error = ""
        for attempt in range(max_retries):
            if attempt >= len(offers):
                break
            offer = offers[attempt]
            machine_id = str(offer.get("machine_id", ""))
            # Atomically claim the machine_id BEFORE deploying. The previous
            # claim-on-success pattern raced when many parallel threads
            # started together: each read ``used_machine_ids`` (still empty)
            # at offer-selection time and all picked offers[0], landing
            # multiple instances on the same physical host. Two co-located
            # workers fight for the GPU → boot-timeout / GPU-verify failures.
            # The fix is a tentative claim under the lock now, with a
            # release on deploy failure so the host comes back into the
            # pool for retry.
            if not self._try_claim_machine(machine_id, used_machine_ids, machine_lock):
                continue
            result, error = self._try_one_offer(offer, files, attempt)
            if result is not None:
                return result
            # Deploy failed — release the tentative claim so future attempts
            # (this thread's next iteration or another thread's next pick)
            # can re-use the machine.
            self._release_machine(machine_id, used_machine_ids, machine_lock)
            last_error = error

        return DeploymentResult(success=False, error=last_error)

    def _try_one_offer(
        self,
        offer: dict[str, object],
        files: dict[str, Path],
        attempt: int,
    ) -> tuple[DeploymentResult | None, str]:
        """Run the lifecycle on one offer. Returns (result, error).

        ``result`` is a successful ``DeploymentResult`` or ``None`` if the
        attempt failed (caller should try the next offer). ``error`` is the
        human-readable failure reason to record when ``result`` is ``None``.

        On any gate failure (or exception escape) we call
        ``capture_deploy_failure_diagnostics`` *before* destroying the
        instance so subclasses can pull ``worker.log`` / provider logs
        while SSH is still up and the container still exists.
        """
        instance: CloudInstance | None = None
        try:
            instance = self.create_instance(offer)
            error = self._run_gate_chain(instance, files, attempt)
            if error:
                with contextlib.suppress(Exception):
                    self.capture_deploy_failure_diagnostics(instance, error, attempt)
                self.destroy_instance(instance)
                return None, error
            return DeploymentResult(success=True, instance=instance), ""
        except Exception as exc:
            err = f"Attempt {attempt + 1} failed: {exc}"
            logger.warning("Deployment attempt %d failed: %s", attempt + 1, exc)
            if instance:
                with contextlib.suppress(Exception):
                    self.capture_deploy_failure_diagnostics(instance, err, attempt)
                with contextlib.suppress(Exception):
                    self.destroy_instance(instance)
            return None, err

    def _run_gate_chain(
        self,
        instance: CloudInstance,
        files: dict[str, Path],
        attempt: int,
    ) -> str:
        """Run boot -> verify -> deploy -> setup -> launch gates in order.

        Returns empty string on success, or the first failing gate's error.
        """
        gates: list[tuple[Callable[[], bool], str]] = [
            (lambda: self.wait_for_boot(instance), "Boot timeout"),
            (lambda: self.verify_gpu(instance), "GPU verification failed"),
            (lambda: self.deploy_files(instance, files), "File deploy failed"),
            (lambda: self.setup_environment(instance), "Environment setup failed"),
            (lambda: self.launch_worker(instance), "Worker launch failed"),
        ]
        for gate, label in gates:
            if not gate():
                return f"{label} (attempt {attempt + 1})"
        return ""

    @staticmethod
    def _try_claim_machine(
        machine_id: str,
        used_machine_ids: set[str] | None,
        machine_lock: threading.Lock | object | None,
    ) -> bool:
        """Atomically claim a machine_id under the shared lock.

        Returns ``True`` if the claim succeeded (machine was free and is
        now reserved by this caller), ``False`` if the machine was already
        taken by a parallel thread. ``None`` set / lock means there is
        no shared coordination, so the claim trivially succeeds.

        Pair with :func:`_release_machine` on deploy failure so the host
        rejoins the pool for retry.
        """
        if used_machine_ids is None or not machine_id:
            return True
        if machine_lock is not None and hasattr(machine_lock, "__enter__"):
            with machine_lock:  # type: ignore[union-attr]
                if machine_id in used_machine_ids:
                    return False
                used_machine_ids.add(machine_id)
                return True
        if machine_id in used_machine_ids:
            return False
        used_machine_ids.add(machine_id)
        return True

    @staticmethod
    def _release_machine(
        machine_id: str,
        used_machine_ids: set[str] | None,
        machine_lock: threading.Lock | object | None,
    ) -> None:
        """Release a tentative machine_id claim after a failed deploy."""
        if used_machine_ids is None or not machine_id:
            return
        if machine_lock is not None and hasattr(machine_lock, "__enter__"):
            with machine_lock:  # type: ignore[union-attr]
                used_machine_ids.discard(machine_id)
        else:
            used_machine_ids.discard(machine_id)

    @staticmethod
    def _claim_machine(
        machine_id: str,
        used_machine_ids: set[str] | None,
        machine_lock: threading.Lock | object | None,
    ) -> None:
        """Backwards-compatible alias for the post-deploy claim path.

        Kept for callers that still want the claim-on-success pattern;
        the in-tree :meth:`run_full_cycle` now uses the atomic
        :meth:`_try_claim_machine` + :meth:`_release_machine` pair to
        prevent same-machine collisions across parallel deploys.
        """
        if used_machine_ids is None:
            return
        if machine_lock is not None and hasattr(machine_lock, "__enter__"):
            with machine_lock:  # type: ignore[union-attr]
                used_machine_ids.add(machine_id)
        else:
            used_machine_ids.add(machine_id)

    def download_all_results(
        self,
        instance: CloudInstance,
        local_dir: Path,
        *,
        remote_subdir: str = "",
        critical_files: set[str] | None = None,
    ) -> list[str]:
        """Download all results from the instance via rsync.

        Args:
            instance: Running cloud instance.
            local_dir: Local directory to download into.
            remote_subdir: Subdirectory within workspace to download from.
            critical_files: Set of filenames that must exist after download.

        Returns:
            List of downloaded file paths (empty on failure).
        """
        local_dir.mkdir(parents=True, exist_ok=True)
        ws = self.config.workspace_dir
        remote_path = f"{ws}/{remote_subdir}/" if remote_subdir else f"{ws}/"

        rsync_cmd = [
            "rsync",
            "-avz",
            "--timeout=120",
            "-e",
            f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"-p {instance.ssh_port}",
            f"{instance.ssh_user}@{instance.ssh_host}:{remote_path}",
            f"{local_dir}/",
        ]

        try:
            result = subprocess.run(
                rsync_cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min for large trajectories
                stdin=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode not in (0, 23, 24):
                # 23 = partial transfer, 24 = vanished source files
                logger.error("rsync failed (rc=%d): %s", result.returncode, result.stderr[:500])
                return []
        except subprocess.TimeoutExpired:
            logger.error("rsync timeout downloading from %s", instance.instance_id)
            return []

        # Collect downloaded files
        downloaded = [str(p.relative_to(local_dir)) for p in local_dir.rglob("*") if p.is_file()]

        # Check critical files
        if critical_files:
            from pathlib import Path as _Path

            missing = critical_files - {_Path(f).name for f in downloaded}
            if missing:
                logger.warning(
                    "Missing critical files after download: %s",
                    ", ".join(sorted(missing)),
                )
                return []

        logger.info(
            "Downloaded %d files from %s to %s",
            len(downloaded),
            instance.instance_id,
            local_dir,
        )
        return downloaded
