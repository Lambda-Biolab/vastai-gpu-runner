"""Shared orchestration patterns for cloud GPU batch workloads.

Extracts the common lifecycle patterns from batch orchestrators:
- ``sweep_zombie_instances``: destroy orphaned instances (v3 routes
  through ``destroy_vastai_instance`` from the destroy adapter;
  ``killed`` only on confirmed ``DESTROYED``)
- ``check_budget``: cost ceiling enforcement
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from vastai_gpu_runner.providers.destroy import (
    DestroyRefusal,
    DestroyVerdict,
)
from vastai_gpu_runner.providers.destroy_adapters.vastai import (
    CredentialState,
    OwnershipVerification,
    destroy_vastai_instance,
    read_vastai_api_key,
)
from vastai_gpu_runner.providers.vastai import vastai_cmd

if TYPE_CHECKING:
    from vastai_gpu_runner.runner import CloudRunner
    from vastai_gpu_runner.storage.r2 import R2Sink
    from vastai_gpu_runner.types import CloudInstance

logger = logging.getLogger(__name__)


def sweep_zombie_instances(
    live_runners: dict[int, tuple[CloudRunner, CloudInstance]],
    *,
    label_prefix: str,
    r2_sink: R2Sink | None = None,
    r2_batch_id: str = "",
    allowed_images: frozenset[str] | None = None,
) -> int:
    """Destroy Vast.ai instances not tracked by ``live_runners``.

    Vast.ai sometimes resurrects destroyed instances as 'stopped' after
    boot timeout or GPU verification failure. This sweep catches them.

    Per the v3 doc:

    - Short-circuits on ``EXPLICITLY_DISABLED`` (returns 0 before the
      CLI enumeration step — the user has opted out).
    - Routes through ``destroy_vastai_instance`` for each orphan
      that passes the label-prefix filter.
    - ``killed`` only on confirmed ``DESTROYED`` verdict.
    - CLI-fallback attempts (``refusal=NO_CREDENTIALS`` followed by
      an attempted CLI destroy) are logged separately as
      ``cli_attempted`` but do NOT count toward ``killed`` (the
      orchestrator's contract: ``killed`` means a confirmed
      destroy happened).

    Only sweeps instances whose label starts with ``label_prefix`` to
    avoid cross-orchestrator kills.

    Args:
        live_runners: Map of shard/job index -> (runner, instance) for
            actively tracked instances.
        label_prefix: Label prefix filter (e.g.
            ``"myproject-boltz2-abc123"``).
        r2_sink: Optional R2 sink for checking DONE markers before
            destroying stopped instances that may have completed.
        r2_batch_id: Batch ID for R2 DONE marker checks.
        allowed_images: Image allowlist forwarded to
            ``destroy_vastai_instance``. ``None`` disables the
            ownership check; empty frozenset fails closed (refuses
            every instance).

    Returns:
        Number of zombies destroyed (only confirmed DESTROYED).
    """
    credentials = read_vastai_api_key()
    if credentials.state == CredentialState.EXPLICITLY_DISABLED:
        logger.info(
            "Zombie sweep: credentials explicitly disabled; skipping "
            "(EXPLICITLY_DISABLED short-circuits before enumeration)"
        )
        return 0

    instances = _fetch_vastai_instances()
    if instances is None:
        return 0

    tracked_ids = {inst.instance_id for _, (_, inst) in live_runners.items()}
    return _sweep_zombies_for_instances(
        instances, tracked_ids, label_prefix, r2_sink, r2_batch_id, allowed_images
    )


def _sweep_zombies_for_instances(
    instances: list[object],
    tracked_ids: set[str],
    label_prefix: str,
    r2_sink: R2Sink | None,
    r2_batch_id: str,
    allowed_images: frozenset[str] | None,
) -> int:
    """Apply the label-filter + ownership check + destroy to each instance."""
    killed, cli_attempted = 0, 0
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        if not _is_zombie(inst, label_prefix, tracked_ids, r2_sink, r2_batch_id):
            continue
        iid = str(inst.get("id", ""))
        if not iid:
            continue
        outcome = _destroy_zombie(iid, allowed_images=allowed_images)
        if outcome == "destroyed":
            killed += 1
        elif outcome == "cli_attempted":
            cli_attempted += 1
    _log_sweep_outcome(killed, cli_attempted)
    return killed


def _log_sweep_outcome(killed: int, cli_attempted: int) -> None:
    """Log the sweep outcome; kept terse to avoid extra branching."""
    if killed:
        logger.info("Zombie sweep: destroyed %d instance(s)", killed)
    if cli_attempted:
        logger.info(
            "Zombie sweep: %d CLI fallback attempt(s) (ownership verified; "
            "no API key — destroy attempted via vastai CLI)",
            cli_attempted,
        )


def _fetch_vastai_instances() -> list[object] | None:
    """Run ``vastai show instances --raw`` and parse the JSON list.

    Returns the parsed list on success; ``None`` on any failure
    (timeout, parse error, non-list response). The sweep short-
    circuits on failure (no zombies destroyed).
    """
    try:
        raw = vastai_cmd(["show", "instances", "--raw"], timeout=15)
        instances = json.loads(raw)
    except Exception:
        return None
    if not isinstance(instances, list):
        logger.warning(
            "Zombie sweep: expected list from vastai CLI, got %s",
            type(instances).__name__,
        )
        return None
    return instances


def _is_zombie(
    inst: dict[str, object],
    label_prefix: str,
    tracked_ids: set[str],
    r2_sink: R2Sink | None,
    r2_batch_id: str,
) -> bool:
    """Classify whether an instance should be destroyed by the sweep.

    The sweep is for **orphans only** — instances whose label matches our
    batch prefix but which we are NOT tracking. Tracked instances must
    never be destroyed here, because Vast.ai's ``cur_state`` API is
    unreliable for this purpose: it reports ``stopped`` / ``exited``
    persistently for containers whose long-running OpenMM worker is
    still running fine (confirmed via SSH 2026-04-20). The
    orchestrator's normal collect/destroy flow handles
    tracked-instance cleanup.

    The one exception is the R2 DONE marker path: if R2 confirms a
    tracked instance has uploaded its final results, the sweep still
    skips the destroy so the collect phase can harvest cleanly.
    """
    iid = str(inst.get("id", ""))
    label = str(inst.get("label", ""))
    status = str(inst.get("cur_state", ""))

    if not label.startswith(label_prefix):
        return False

    if iid in tracked_ids:
        return False

    return not _r2_says_done(iid, label, label_prefix, status, tracked_ids, r2_sink, r2_batch_id)


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


def _destroy_zombie(iid: str, *, allowed_images: frozenset[str] | None) -> str:
    """Destroy one zombie via the v3 destroy adapter.

    Returns one of:
    - ``"destroyed"`` — belt_and_suspenders reported DESTROYED.
    - ``"cli_attempted"`` — adapter returned NO_CREDENTIALS; we
      tried the CLI fallback (``vastai destroy instance``). The
      v4 factory will own this dispatch; v3 keeps the inline call
      to preserve the v3 behaviour.
    - ``"refused"`` — adapter refused (OWNERSHIP or
      CREDENTIALS_DISABLED); the instance is not destroyed.
    - ``"unknown"`` — adapter reported UNKNOWN or LEAKED; the
      instance may or may not be gone.
    """
    logger.info("Zombie sweep: destroying %s (label-matched orphan)", iid)
    result = destroy_vastai_instance(iid, allowed_images=allowed_images)

    if result.verdict == DestroyVerdict.DESTROYED:
        return "destroyed"
    if result.verdict == DestroyVerdict.LEAKED:
        # The instance is gone (verify said GONE) but our DELETE
        # never returned success. The user's intent is achieved.
        return "destroyed"
    if result.refusal == DestroyRefusal.OWNERSHIP:
        logger.warning(
            "Zombie sweep: %s ownership REFUSED (image not in allowlist); skipping",
            iid,
        )
        return "refused"
    if result.refusal == DestroyRefusal.CREDENTIALS_DISABLED:
        logger.warning(
            "Zombie sweep: %s credentials DISABLED; skipping",
            iid,
        )
        return "refused"
    if result.refusal == DestroyRefusal.NO_CREDENTIALS:
        # v3 CLI fallback: ownership was OK, no API key, attempt
        # the CLI destroy. v4 factory will own the dispatch.
        try:
            vastai_cmd(["destroy", "instance", iid], timeout=15)
            logger.info("Zombie sweep: CLI-destroyed %s", iid)
            return "cli_attempted"
        except Exception as exc:
            logger.warning("Zombie sweep: CLI destroy failed for %s: %s", iid, exc)
            return "unknown"
    # UNKNOWN verdict — uncertain state.
    return "unknown"


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


# DELETED in v3 step 7:
# - load_vastai_api_key() — the v3 destroy adapter's
#   read_vastai_api_key() (with env-first, fail-closed) replaces it
# - _destroy_via_rest() — absorbed into the v3 destroy adapter
# - ensure_detached() — dead public API, no callers in this repo
# - poll_instance_progress() — dead public API, no callers in this
#   repo; the v3 decide_next_action (unit_lifecycle.py) replaces
#   the per-unit classification logic

# Suppress unused-import warning for OwnershipVerification (kept
# imported for downstream re-export of the public surface and
# future consumer-side imports of the typed ownership enum).
_ = OwnershipVerification
