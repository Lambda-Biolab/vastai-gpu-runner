"""Per-cycle unit action decision tree.

Owns the R2-first → SSH → worker_dead re-check observation sequence and
returns a tagged ``UnitAction`` enum value. No side effects: destroyers,
collectors, and lock acquisition all stay on the orchestrator.

Extracted from ``BatchOrchestrator._classify_live_unit`` in v3 of the
architecture (see ``docs/architecture-v3.md``). The orchestrator's
``_poll_cycle_once`` and ``_check_unit`` paths both route through
``decide_next_action`` so their logic cannot diverge.

Public surface:

- ``Action`` — the high-level outcome enum
- ``PreemptCause`` — why the worker died (stable contract)
- ``ProgressSnapshot`` — normalised view of ``runner.check_progress``
- ``Continue`` / ``Complete`` / ``Preempt`` — frozen plan dataclasses
- ``UnitAction`` — tagged union alias
- ``decide_next_action`` — the single public entry point
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Protocol, TypeVar

UnitT = TypeVar("UnitT")
logger = logging.getLogger(__name__)


class Action(StrEnum):
    """High-level unit outcome from the decision tree."""

    CONTINUE = "continue"
    COMPLETE = "complete"
    PREEMPT = "preempt"


class PreemptCause(StrEnum):
    """Why the decision module believes the worker died.

    The enum is a stable contract: ``classify_failure`` and other
    consumer-side hooks may eventually branch on the cause. Display
    text belongs in ``Preempt.detail`` (free-form, for humans only).
    """

    WORKER_DIED = "worker_died"  # PID alive check failed, no DONE file
    # Reserved for future use:
    # BOOT_TIMEOUT_RETRY_EXHAUSTED = "boot_timeout_retry_exhausted"
    # SSH_LOST = "ssh_lost"
    # GPU_OOM = "gpu_oom"


@dataclass(frozen=True)
class ProgressSnapshot:
    """Normalised view of the runner's ``check_progress`` response.

    Missing or contradictory fields default to the conservative read
    (both flags clear on contradiction). Missing or non-dict envelope
    fails closed (returns a snapshot with all defaults).

    Constructor accepts a raw ``check_progress`` dict and tolerates
    any malformed input — this is the boundary that prevents malformed
    provider data from leaking into the decision tree.
    """

    complete: bool = False
    worker_dead: bool = False
    log_tail: str = ""

    @classmethod
    def from_raw(cls, raw: Any) -> ProgressSnapshot:
        """Build a snapshot from an arbitrary runner response.

        ``None`` and non-dict inputs return the all-defaults snapshot.
        String-coercible truthy fields become ``True``; everything else
        ``False``. ``log_tail`` is taken as a string when present.
        """
        if not isinstance(raw, dict):
            return cls()
        complete = bool(raw.get("complete"))
        worker_dead = bool(raw.get("worker_dead"))
        log_tail_raw = raw.get("log_tail", "")
        log_tail = log_tail_raw if isinstance(log_tail_raw, str) else ""
        return cls(complete=complete, worker_dead=worker_dead, log_tail=log_tail)


@dataclass(frozen=True)
class Continue:
    """Plan: keep polling — no observed completion or preemption."""

    action: ClassVar[Action] = Action.CONTINUE


@dataclass(frozen=True)
class Complete:
    """Plan: this unit has finished — R2 done or SSH reports complete."""

    action: ClassVar[Action] = Action.COMPLETE


@dataclass(frozen=True)
class Preempt:
    """Plan: the worker died before uploading — treat as instance loss."""

    action: ClassVar[Action] = Action.PREEMPT
    cause: PreemptCause
    detail: str | None = None


UnitAction = Continue | Complete | Preempt


class CloudRunnerLike(Protocol):
    def check_progress(self, instance: object) -> dict[str, object]: ...


def decide_next_action(
    unit: UnitT,
    runner: CloudRunnerLike,
    instance: object,
    is_done_in_r2: Callable[[UnitT], bool],
) -> UnitAction:
    """Classify a live unit and return the next action.

    Performs a localised observation sequence: R2 first, then SSH,
    then R2 again on worker_dead. The two R2 reads are in immediate
    succession inside this function — no orchestrator code can
    interleave between them.

    Exception policy: R2 raises → log + continue to SSH;
    ``check_progress`` raises → Continue; worker_dead re-check raises
    → Continue (don't destroy on unknown final-upload status).
    """
    try:
        if is_done_in_r2(unit):
            return Complete()
    except Exception as exc:
        logger.warning("decide_next_action: initial R2 check raised %s", exc)
    try:
        progress = ProgressSnapshot.from_raw(runner.check_progress(instance))
    except Exception as exc:
        logger.warning("decide_next_action: check_progress raised %s", exc)
        return Continue()
    if progress.complete:
        return Complete()
    if progress.worker_dead:
        try:
            if is_done_in_r2(unit):
                return Complete()
        except Exception as exc:
            logger.warning("decide_next_action: worker_dead R2 re-check raised %s", exc)
            return Continue()
        return Preempt(cause=PreemptCause.WORKER_DIED, detail=progress.log_tail)
    return Continue()


__all__ = [
    "Action",
    "Complete",
    "Continue",
    "Preempt",
    "PreemptCause",
    "ProgressSnapshot",
    "UnitAction",
    "decide_next_action",
]
