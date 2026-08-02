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


# Fixed upper bound for the final R2 upload call.
#
# The worker invokes ``r2_upload.py --done`` from ``upload_results()``
# after a successful workload. This call must complete promptly even
# when R2 is rate-limiting or briefly unavailable, so the worker can
# transition to ``self_destruct()`` without delaying instance teardown
# for the prior default of 300 seconds.
#
# Rationale for 90s (bounded-teardown trade-off): in the shard
# ``--done`` path, only the small ``worker.exitcode`` and ``DONE``
# marker are uploaded — well within 90s.
#
# SCOPE: This timeout applies ONLY to ``BaseWorker.upload_results()``
# which calls the SHARD uploader's ``--done`` mode. Job-based
# workers run their upload scripts outside this path and do NOT
# inherit the 90s ceiling. Subclasses that override
# ``upload_results()`` to call a different script or a different
# mode must set their own timeout; ``R2_FINAL_UPLOAD_TIMEOUT_SECONDS``
# is the marker-only budget.
#
# Operators with very large final-output workloads should override
# ``upload_results()`` and rely on the orchestrator's rsync fallback
# (or wait for the long-term bounded-teardown protocol in
# ``docs/architecture-r2-collection-handshake.md``).
#
# Not publicly configurable in v1.
R2_FINAL_UPLOAD_TIMEOUT_SECONDS = 90


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

        Self-destruct always runs on exit, including when ``run_workload`` or
        any earlier step raises an uncaught exception. Without this
        guarantee, a subclass bug would leak a running Vast.ai instance and
        keep billing indefinitely.
        """
        try:
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

            return exit_code
        except Exception:
            logger.exception("Unhandled exception in worker main()")
            self._write_exit(1)
            self._write_completed(success=False)
            return 1
        finally:
            self.self_destruct()

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

        Behaviour:
            - Subprocess is bounded by ``R2_FINAL_UPLOAD_TIMEOUT_SECONDS``.
            - ``subprocess.TimeoutExpired`` is caught separately and logged
              as a warning; teardown (``self_destruct``) continues.
            - Non-zero return codes are logged with truncated stderr/stdout.
            - Success (``returncode == 0``) is logged explicitly.
            - Transport failure does NOT change the workload exit code.
            - ``self_destruct()`` always runs from ``main()`` regardless
              of upload outcome.
        """
        r2_script = self.workspace / "r2_upload.py"
        if not r2_script.exists():
            return

        cmd = [sys.executable, str(r2_script), "--done"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=R2_FINAL_UPLOAD_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "R2 final upload exceeded %ds timeout — continuing teardown "
                "(self_destruct will still run). Rsync fallback applies if "
                "the orchestrator reaches collect phase.",
                R2_FINAL_UPLOAD_TIMEOUT_SECONDS,
            )
            return
        except Exception as exc:
            # Catch-all for launch / filesystem / unexpected errors.
            # Do NOT raise — the workload already succeeded and the
            # self_destruct() in main()'s finally block must execute.
            logger.warning("R2 upload launch failed: %s", exc)
            return

        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-200:]
            stdout_tail = (result.stdout or "")[-200:]
            logger.warning(
                "R2 final upload returned non-zero (rc=%d). Teardown continues.",
                result.returncode,
            )
            if stderr_tail:
                logger.warning("R2 stderr (tail): %s", stderr_tail)
            if stdout_tail:
                logger.warning("R2 stdout (tail): %s", stdout_tail)
            return

        logger.info("R2 upload complete")

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
            from urllib.parse import urlparse

            # Vast.ai's destroy-instance endpoint. We embed the API
            # key in the query string (matches the v0 REST API). The
            # endpoint host is hardcoded so the URL can only point at
            # console.vast.ai; we parse and assert this before the
            # network call so a malicious env var can't redirect to
            # an attacker-controlled host.
            endpoint_host = "console.vast.ai"
            path = f"/api/v0/instances/{instance_id}/"
            query = f"api_key={api_key}"
            url = f"https://{endpoint_host}{path}?{query}"
            parsed = urlparse(url)
            if parsed.hostname != endpoint_host:
                msg = f"refusing to call non-Vast.ai host: {parsed.hostname!r}"
                raise ValueError(msg)
            # bandit B310 / ruff S310: host is verified above
            # (parsed.hostname == endpoint_host). The request can
            # only ever hit console.vast.ai regardless of
            # instance_id/api_key contents.
            req = urllib.request.Request(url, method="DELETE")  # nosec B310  # noqa: S310
            urllib.request.urlopen(req, timeout=15)  # nosec B310  # noqa: S310
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
