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

            # Skip machines already claimed by other threads
            if used_machine_ids and machine_id in used_machine_ids:
                continue

            instance: CloudInstance | None = None
            try:
                instance = self.create_instance(offer)
                if not self.wait_for_boot(instance):
                    last_error = f"Boot timeout (attempt {attempt + 1})"
                    self.destroy_instance(instance)
                    continue

                if not self.verify_gpu(instance):
                    last_error = f"GPU verification failed (attempt {attempt + 1})"
                    self.destroy_instance(instance)
                    continue

                if not self.deploy_files(instance, files):
                    last_error = f"File deploy failed (attempt {attempt + 1})"
                    self.destroy_instance(instance)
                    continue

                if not self.setup_environment(instance):
                    last_error = f"Environment setup failed (attempt {attempt + 1})"
                    self.destroy_instance(instance)
                    continue

                if not self.launch_worker(instance):
                    last_error = f"Worker launch failed (attempt {attempt + 1})"
                    self.destroy_instance(instance)
                    continue

                # Register machine to prevent other threads from using it
                if used_machine_ids is not None and machine_lock is not None:
                    if hasattr(machine_lock, "__enter__"):
                        with machine_lock:  # type: ignore[union-attr]
                            used_machine_ids.add(machine_id)
                    else:
                        used_machine_ids.add(machine_id)

                return DeploymentResult(success=True, instance=instance)

            except Exception as exc:
                last_error = f"Attempt {attempt + 1} failed: {exc}"
                logger.warning("Deployment attempt %d failed: %s", attempt + 1, exc)
                if instance:
                    with contextlib.suppress(Exception):
                        self.destroy_instance(instance)

        return DeploymentResult(success=False, error=last_error)

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
