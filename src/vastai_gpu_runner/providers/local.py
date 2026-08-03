"""Local subprocess provider for zero-cost development and CI runs."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

from vastai_gpu_runner.cleanup_policy import (
    CleanupResult,
    CleanupVerdict,
    InstanceCandidate,
    ProviderCleanupPolicy,
)
from vastai_gpu_runner.runner import CloudRunner
from vastai_gpu_runner.types import CloudInstance, DeploymentConfig, InstanceStatus, Provider

logger = logging.getLogger(__name__)


class LocalRunner(CloudRunner):
    """Execute the CloudRunner lifecycle on the current host.

    The local provider uses one temporary workspace and one subprocess for a
    worker. It does not use SSH, cloud credentials, Docker, or GPU-specific
    setup. The inherited ``CloudRunner.run_full_cycle`` still owns retry and
    machine-deduplication behavior; this class supplies the local lifecycle
    operations only.
    """

    def __init__(self, config: DeploymentConfig | None = None) -> None:
        """Initialize a local runner with an optional deployment config."""
        super().__init__(config)
        self._workspaces: dict[str, Path] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    def search_offers(self, **kwargs: object) -> list[dict[str, object]]:
        """Return the single synthetic local execution offer."""
        del kwargs
        return [{"machine_id": "local", "dph_total": 0.0}]

    def create_instance(self, offer: Mapping[str, object]) -> CloudInstance:
        """Allocate a temporary workspace representing the local instance."""
        del offer
        instance_id = "local"
        if instance_id in self._workspaces:
            raise RuntimeError("LocalRunner already has an active instance")
        workspace = Path(tempfile.mkdtemp(prefix="vastai-local-"))
        self._workspaces[instance_id] = workspace
        return CloudInstance(
            provider=Provider.LOCAL,
            instance_id=instance_id,
            status=InstanceStatus.CREATING,
            ssh_host="localhost",
        )

    def wait_for_boot(self, instance: CloudInstance) -> bool:
        """Mark the local instance as ready immediately."""
        instance.status = InstanceStatus.RUNNING
        return True

    def verify_gpu(self, instance: CloudInstance) -> bool:
        """Probe ``nvidia-smi`` but allow CPU-only local execution."""
        del instance
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("nvidia-smi unavailable; continuing with CPU execution: %s", exc)
            return True
        if result.returncode != 0:
            logger.warning("nvidia-smi failed; continuing with CPU execution")
        else:
            logger.info("Local GPU verified with nvidia-smi")
        return True

    def deploy_files(
        self,
        instance: CloudInstance,
        files: dict[str, Path],
    ) -> bool:
        """Copy payload files into the local temporary workspace."""
        workspace = self._workspace(instance)
        for remote_name, local_path in files.items():
            if not local_path.exists():
                logger.warning("Local file not found: %s", local_path)
                continue
            try:
                destination = self._workspace_path(instance, remote_name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(local_path, destination)
            except (OSError, ValueError) as exc:
                logger.error("Failed to copy %s into %s: %s", local_path, workspace, exc)
                return False
        return True

    def setup_environment(self, instance: CloudInstance) -> bool:
        """Skip environment setup because execution is already local."""
        del instance
        return True

    def launch_worker(self, instance: CloudInstance) -> bool:
        """Launch the configured worker script in the local workspace."""
        workspace = self._workspace(instance)
        existing = self._processes.get(instance.instance_id)
        if existing is not None and existing.poll() is None:
            logger.warning("Worker already running in %s", workspace)
            return True

        worker_path = self._workspace_path(instance, self.config.worker_script)
        if not worker_path.is_file():
            logger.error("Worker script not found: %s", worker_path)
            return False

        try:
            with (workspace / "worker.log").open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    ["bash", self.config.worker_script],
                    cwd=workspace,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            (workspace / "worker.pid").write_text(str(process.pid), encoding="utf-8")
        except OSError as exc:
            logger.error("Local worker launch failed: %s", exc)
            return False

        self._processes[instance.instance_id] = process
        logger.info("Local worker launched in %s", workspace)
        return True

    def check_progress(self, instance: CloudInstance) -> dict[str, object]:
        """Report DONE-marker completion or local worker liveness."""
        workspace = self._workspace(instance)
        if (workspace / "DONE").is_file():
            return {"running": False, "complete": True}

        process = self._processes.get(instance.instance_id)
        started = process is not None or (workspace / "worker.pid").is_file()
        alive = process is not None and process.poll() is None
        if process is None and started:
            alive = self._pid_is_alive(workspace / "worker.pid")
        if started and not alive:
            logger.warning("Local worker exited without a DONE marker")
            return {
                "running": False,
                "complete": False,
                "worker_dead": True,
                "log_tail": self._log_tail(workspace),
            }
        return {
            "running": True,
            "complete": False,
            "log_tail": self._log_tail(workspace),
        }

    def list_remote_files(self, instance: CloudInstance) -> list[str]:
        """List files in the local workspace as relative POSIX paths."""
        workspace = self._workspace(instance)
        return sorted(
            str(path.relative_to(workspace)) for path in workspace.rglob("*") if path.is_file()
        )

    def download_file(
        self,
        instance: CloudInstance,
        remote_name: str,
        local_path: Path,
    ) -> bool:
        """Copy one file from the local workspace to a destination path."""
        try:
            source = self._workspace_path(instance, remote_name)
            if not source.is_file():
                return False
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, local_path)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to copy local result %s: %s", remote_name, exc)
            return False
        return True

    def download_all_results(
        self,
        instance: CloudInstance,
        local_dir: Path,
        *,
        remote_subdir: str = "",
        critical_files: set[str] | None = None,
    ) -> list[str]:
        """Copy local workspace results without invoking rsync or SSH."""
        workspace = self._workspace(instance)
        if remote_subdir:
            source_dir = self._workspace_path(instance, remote_subdir)
        else:
            source_dir = workspace
        if not source_dir.is_dir():
            return []
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[str] = []
        for source in sorted(source_dir.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(workspace)
            destination = local_dir / relative
            if self.download_file(instance, str(relative), destination):
                downloaded.append(str(destination.relative_to(local_dir)))
        if critical_files and downloaded:
            downloaded = self._filter_critical(downloaded, critical_files)
        return downloaded

    @staticmethod
    def _filter_critical(downloaded: list[str], critical_files: set[str]) -> list[str]:
        """Return only downloads that satisfy every critical-file requirement."""
        downloaded_names = {Path(path).name for path in downloaded}
        if critical_files - downloaded_names:
            return []
        return downloaded

    def destroy_instance(self, instance: CloudInstance) -> bool:
        """Stop the local worker and remove its temporary workspace."""
        process = self._processes.pop(instance.instance_id, None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

        workspace = self._workspaces.pop(instance.instance_id, None)
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)
        instance.status = InstanceStatus.DESTROYED
        return True

    def _workspace(self, instance: CloudInstance) -> Path:
        """Return the workspace for an instance or raise a clear error."""
        try:
            return self._workspaces[instance.instance_id]
        except KeyError as exc:
            raise RuntimeError(f"Unknown local instance: {instance.instance_id!r}") from exc

    def _workspace_path(self, instance: CloudInstance, name: str) -> Path:
        """Resolve a relative workspace path without allowing traversal."""
        relative = Path(name)
        if not name or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"path must be a relative workspace path: {name!r}")
        workspace = self._workspace(instance).resolve()
        resolved = (workspace / relative).resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError(f"path must be a relative workspace path: {name!r}")
        return resolved

    @staticmethod
    def _log_tail(workspace: Path) -> str:
        """Read the final three lines of the worker log."""
        log_path = workspace / "worker.log"
        if not log_path.is_file():
            return ""
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-3:])

    @staticmethod
    def _pid_is_alive(pid_path: Path) -> bool:
        """Return whether the PID in a worker marker is still running."""
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
        except (OSError, ValueError):
            return False
        return True


def build_local_cleanup_policy() -> ProviderCleanupPolicy:
    """Build the cleanup policy for local processes and workspaces.

    Local instances are owned by the process that created them, so there is no
    safe cross-process enumeration for a zombie sweep. Live-runner cleanup is
    handled by ``LocalRunner.destroy_instance``.
    """

    def _list_instances() -> list[InstanceCandidate]:
        return []

    def _destroy(candidate: InstanceCandidate) -> CleanupResult:
        del candidate
        return CleanupResult(verdict=CleanupVerdict.DESTROYED)

    return ProviderCleanupPolicy(
        provider=Provider.LOCAL,
        list_instances_fn=_list_instances,
        destroy_fn=_destroy,
    )


__all__ = ["LocalRunner", "build_local_cleanup_policy"]
