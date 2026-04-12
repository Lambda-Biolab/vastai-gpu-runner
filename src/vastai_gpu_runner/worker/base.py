"""Base worker class for cloud GPU instances.

Template method pattern: the ``main()`` method orchestrates the worker
lifecycle (PID file -> GPU check -> preflight gates -> workload -> upload
-> self-destruct). Subclasses override ``run_workload()`` and optionally
``preflight_gates()`` and ``upload_results()``.

Usage::

    class MyWorker(BaseWorker):
        def run_workload(self) -> int:
            # Run prediction / simulation / training
            return 0  # exit code


    worker = MyWorker(workspace=Path("/workspace/my_workload"))
    sys.exit(worker.main())
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from vastai_gpu_runner.worker.health import check_gpu, check_r2_connectivity

logger = logging.getLogger(__name__)


class BaseWorker(ABC):
    """Abstract base worker for cloud GPU instances.

    Args:
        workspace: Worker workspace directory.
        min_gpu_memory_mib: Minimum GPU VRAM (0 = skip check).
        max_gpu_temp_c: Maximum GPU temperature.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        min_gpu_memory_mib: int = 0,
        max_gpu_temp_c: int = 90,
    ) -> None:
        """Initialize worker with workspace and GPU health thresholds."""
        self.workspace = workspace
        self.min_gpu_memory_mib = min_gpu_memory_mib
        self.max_gpu_temp_c = max_gpu_temp_c

    def main(self) -> int:
        """Template method — orchestrate the full worker lifecycle.

        Returns:
            Exit code (0 = success, 1 = workload failure, 3 = preflight gate).
        """
        os.chdir(self.workspace)
        self.write_pid()

        if not check_gpu(
            min_memory_mib=self.min_gpu_memory_mib,
            max_temp_c=self.max_gpu_temp_c,
        ):
            self._write_exit(1)
            return 1

        for gate in self.preflight_gates():
            if not gate():
                gate_name = getattr(gate, "__name__", str(gate))
                logger.error("Preflight gate failed: %s", gate_name)
                self._write_exit(3)
                return 3

        exit_code = self.run_workload()

        self._write_exit(exit_code)
        self._write_completed(exit_code == 0)

        if exit_code == 0:
            self.upload_results()

        self.self_destruct()
        return exit_code

    # -- Hooks (override in subclasses) ------------------------------------

    def preflight_gates(self) -> list[Callable[[], bool]]:
        """Return a list of preflight gate functions.

        Each gate returns True to proceed, False to abort. Override to
        add workload-specific gates (e.g. weight download, platform check).
        """
        return [self._check_r2]

    @abstractmethod
    def run_workload(self) -> int:
        """Execute the GPU workload. Returns exit code (0 = success)."""

    def upload_results(self) -> None:
        """Upload results to R2 after successful completion.

        Default: calls ``r2_upload.py --done`` if the script exists.
        Override for custom upload logic.
        """
        r2_script = self.workspace / "r2_upload.py"
        if r2_script.exists():
            try:
                subprocess.run(
                    [sys.executable, str(r2_script), "--done"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                logger.info("R2 upload complete")
            except Exception as exc:
                logger.warning("R2 upload failed: %s", exc)

    # -- Built-in operations -----------------------------------------------

    def write_pid(self) -> None:
        """Write PID file for process detection."""
        (self.workspace / "worker.pid").write_text(str(os.getpid()))

    def self_destruct(self) -> None:
        """Self-destruct via Vast.ai REST API.

        Reads ``VASTAI_INSTANCE_ID`` and ``VASTAI_API_KEY`` from environment
        (injected by the orchestrator at SSH launch time). Does nothing if
        the env vars are not set.
        """
        instance_id = os.environ.get("VASTAI_INSTANCE_ID", "")
        api_key = os.environ.get("VASTAI_API_KEY", "")
        if not instance_id or not api_key:
            logger.info("No VASTAI env vars — skipping self-destruct")
            return

        try:
            url = f"https://console.vast.ai/api/v0/instances/{instance_id}/?api_key={api_key}"
            req = urllib.request.Request(url, method="DELETE")  # noqa: S310
            urllib.request.urlopen(req, timeout=15)  # noqa: S310
            logger.info("Self-destruct: instance %s destroyed", instance_id)
        except Exception as exc:
            logger.warning("Self-destruct failed for instance %s: %s", instance_id, exc)

    # -- Internal helpers --------------------------------------------------

    def _check_r2(self) -> bool:
        """R2 connectivity gate (used in preflight_gates)."""
        return check_r2_connectivity(self.workspace)

    def _write_exit(self, code: int) -> None:
        """Write worker.exitcode file."""
        (self.workspace / "worker.exitcode").write_text(str(code))

    def _write_completed(self, success: bool) -> None:
        """Write worker.completed marker."""
        (self.workspace / "worker.completed").write_text("1" if success else "0")
        if success:
            (self.workspace / "DONE").write_text("")
