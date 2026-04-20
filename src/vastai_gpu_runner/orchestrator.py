"""Shared orchestration patterns for cloud GPU batch workloads.

Extracts the common lifecycle patterns from batch orchestrators:
- ``sweep_zombie_instances``: destroy orphaned instances
- ``ensure_detached``: fork + setsid for SSH disconnect survival
- ``check_budget``: cost ceiling enforcement
- ``load_vastai_api_key``: read API key from standard paths
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from vastai_gpu_runner.providers.vastai import vastai_cmd
from vastai_gpu_runner.ssh import ssh_cmd

if TYPE_CHECKING:
    from vastai_gpu_runner.runner import CloudRunner
    from vastai_gpu_runner.storage.r2 import R2Sink
    from vastai_gpu_runner.types import CloudInstance

logger = logging.getLogger(__name__)


def load_vastai_api_key() -> str:
    """Read the Vast.ai API key from standard file paths.

    Checks ``~/.config/vastai/vast_api_key`` and ``~/.vast_api_key``.

    Returns:
        API key string, or empty string if not found.
    """
    key_paths = [
        Path("~/.config/vastai/vast_api_key").expanduser(),
        Path("~/.vast_api_key").expanduser(),
    ]
    for kp in key_paths:
        if kp.exists():
            return kp.read_text().strip()
    return ""


def sweep_zombie_instances(
    live_runners: dict[int, tuple[CloudRunner, CloudInstance]],
    *,
    label_prefix: str,
    r2_sink: R2Sink | None = None,
    r2_batch_id: str = "",
) -> int:
    """Destroy Vast.ai instances not tracked by live_runners.

    Vast.ai sometimes resurrects destroyed instances as 'stopped' after
    boot timeout or GPU verification failure. This sweep catches them.

    Only sweeps instances whose label starts with ``label_prefix`` to
    avoid cross-orchestrator kills.

    Args:
        live_runners: Map of shard/job index -> (runner, instance) for
            actively tracked instances.
        label_prefix: Label prefix filter (e.g. ``"myproject-boltz2-abc123"``).
        r2_sink: Optional R2 sink for checking DONE markers before destroying
            stopped instances that may have completed.
        r2_batch_id: Batch ID for R2 DONE marker checks.

    Returns:
        Number of zombies destroyed.
    """
    try:
        raw = vastai_cmd(["show", "instances", "--raw"], timeout=15)
        instances = json.loads(raw)
    except Exception:
        return 0

    tracked_ids = {inst.instance_id for _, (_, inst) in live_runners.items()}
    killed = 0
    for inst in instances:
        if _is_zombie(inst, label_prefix, tracked_ids, r2_sink, r2_batch_id):
            _destroy_zombie(str(inst.get("id", "")), inst.get("cur_state", ""), tracked_ids)
            killed += 1
    if killed:
        logger.info("Zombie sweep: destroyed %d instance(s)", killed)
    return killed


# Minimum age before a tracked instance can be destroyed as a zombie.
# Vast.ai's ``cur_state`` briefly transitions through ``stopped`` while a
# container boots and its long-running worker process attaches. Destroying
# during that window kills healthy instances.  Five minutes comfortably
# exceeds the longest observed boot-to-worker transition (~2 min).
_ZOMBIE_GRACE_SECONDS = 300.0


def _is_zombie(
    inst: dict[str, object],
    label_prefix: str,
    tracked_ids: set[str],
    r2_sink: R2Sink | None,
    r2_batch_id: str,
) -> bool:
    """Classify whether an instance should be destroyed by the sweep."""
    iid = str(inst.get("id", ""))
    label = str(inst.get("label", ""))
    status = str(inst.get("cur_state", ""))

    if not label.startswith(label_prefix):
        return False
    if iid in tracked_ids and status == "running":
        return False
    if iid in tracked_ids and _within_grace_period(inst):
        logger.debug("Zombie sweep: %s stopped but within grace period — skipping", iid)
        return False
    if _r2_says_done(iid, label, label_prefix, status, tracked_ids, r2_sink, r2_batch_id):
        return False
    return status in ("stopped", "exited") or iid not in tracked_ids


def _within_grace_period(inst: dict[str, object]) -> bool:
    """Return True if ``inst`` is younger than ``_ZOMBIE_GRACE_SECONDS``.

    Uses Vast.ai's ``start_date`` unix timestamp. Returns False when the
    field is missing or unparseable so the caller falls through to its
    normal classification.
    """
    start_date = inst.get("start_date")
    if not isinstance(start_date, (int, float)):
        return False
    return (time.time() - float(start_date)) < _ZOMBIE_GRACE_SECONDS


def _r2_says_done(
    iid: str,
    label: str,
    label_prefix: str,
    status: str,
    tracked_ids: set[str],
    r2_sink: R2Sink | None,
    r2_batch_id: str,
) -> bool:
    """Return True if R2 has a DONE marker that should spare a stopped tracked instance."""
    if iid not in tracked_ids or status not in ("stopped", "exited"):
        return False
    if r2_sink is None or not r2_batch_id:
        return False
    try:
        job_name = label.replace(label_prefix + "-", "", 1) if label_prefix in label else ""
        if job_name and r2_sink.is_job_done(r2_batch_id, job_name):
            logger.info("Zombie sweep: %s is stopped but R2 DONE — skipping", iid)
            return True
    except Exception:
        logger.debug("R2 check failed for %s — proceeding with destroy", iid)
    return False


def _destroy_zombie(iid: str, status: str, tracked_ids: set[str]) -> None:
    """Destroy one zombie instance via REST API (preferred) or CLI fallback."""
    logger.info(
        "Zombie sweep: destroying %s (status=%s, tracked=%s)",
        iid,
        status,
        iid in tracked_ids,
    )
    try:
        api_key = load_vastai_api_key()
        if api_key:
            _destroy_via_rest(iid, api_key)
        else:
            vastai_cmd(["destroy", "instance", iid], timeout=15)
    except Exception as exc:
        logger.warning("Zombie sweep: failed to destroy %s: %s", iid, exc)


def _destroy_via_rest(iid: str, api_key: str) -> None:
    """Stop-then-delete an instance via the Vast.ai REST API."""
    import requests

    base = "https://console.vast.ai/api/v0/instances"
    hdrs = {"Authorization": f"Bearer {api_key}"}
    requests.put(
        f"{base}/{iid}/",
        headers={**hdrs, "Content-Type": "application/json"},
        json={"state": "stopped"},
        timeout=10,
    )
    time.sleep(1)
    requests.delete(f"{base}/{iid}/", headers=hdrs, timeout=10)


def ensure_detached(
    log_path: Path,
    pid_path: Path,
    *,
    detach_env_var: str = "_GPU_RUNNER_DETACHED",
) -> None:
    """Fork + setsid so the orchestrator survives SSH disconnect.

    No-op if already detached or inside tmux/screen.

    Args:
        log_path: Path for the detached process log file.
        pid_path: Path for the detached process PID file.
        detach_env_var: Environment variable name used to detect
            re-entry after fork.
    """
    import sys

    if os.environ.get(detach_env_var) == "1":
        return
    if os.environ.get("TMUX") or os.environ.get("STY"):
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, *sys.argv]
    env = {**os.environ, detach_env_var: "1"}

    logger.info("Detaching — batch survives SSH disconnect.")
    logger.info("  Log:    %s", log_path)
    logger.info("  PID:    %s", pid_path)

    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    os.setsid()
    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    pid_path.write_text(str(os.getpid()))
    os.execve(sys.executable, argv, env)  # noqa: S606


def check_budget(spent: float, ceiling: float) -> bool:
    """Check if cloud spend is within budget.

    Args:
        spent: Total spent so far in USD.
        ceiling: Budget ceiling in USD.

    Returns:
        True if within budget, False if over.
    """
    if spent >= ceiling:
        logger.error("BUDGET EXCEEDED: $%.2f >= $%.2f ceiling", spent, ceiling)
        return False
    if spent >= ceiling * 0.8:
        pct = spent / ceiling * 100
        logger.warning("BUDGET WARNING: $%.2f (%.0f%% of $%.2f)", spent, pct, ceiling)
    return True


def poll_instance_progress(
    instance: CloudInstance,
    workspace: str,
) -> dict[str, object]:
    """Check worker progress via DONE file, PID liveness, and log tail.

    Three-layer check:
    1. DONE file exists -> complete
    2. PID file exists but process dead -> worker crashed
    3. Otherwise -> still running, return log tail

    Args:
        instance: Cloud instance to check.
        workspace: Remote workspace path.

    Returns:
        Dict with keys: running, complete, worker_dead (optional), log_tail.
    """
    # Layer 1: DONE file
    rc, _ = ssh_cmd(instance, f"test -f {workspace}/DONE")
    if rc == 0:
        return {"running": False, "complete": True}

    # Layer 2: PID liveness
    rc_pid, pid_str = ssh_cmd(instance, f"cat {workspace}/worker.pid 2>/dev/null", timeout=5)
    if rc_pid == 0 and pid_str.strip().isdigit():
        rc_alive, _ = ssh_cmd(instance, f"kill -0 {pid_str.strip()} 2>/dev/null", timeout=5)
        if rc_alive != 0:
            logger.warning(
                "Worker PID %s is dead on %s but no DONE file",
                pid_str.strip(),
                instance.instance_id,
            )
            return {
                "running": False,
                "complete": False,
                "worker_dead": True,
                "log_tail": f"Worker PID {pid_str.strip()} dead, no DONE file",
            }

    # Layer 3: Log tail
    rc, output = ssh_cmd(instance, f"tail -3 {workspace}/worker.log", timeout=10)
    return {
        "running": True,
        "complete": False,
        "log_tail": output,
    }
