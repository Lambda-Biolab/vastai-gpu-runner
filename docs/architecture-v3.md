# Architecture v3 (target)

This doc describes the **target architecture** after the unit-lifecycle decision tree and belt-and-suspenders destroy refactor land. For the current-state architecture (today's code) see
[`architecture.md`](architecture.md). For the next-step target see
[`architecture-v2.md`](architecture-v2.md). For the architectural review that
motivated this doc see the [HTML report][review].

[review]: file:///tmp/architecture-review.Jxytgw/vastai-gpu-runner-review.html

## What changes vs v2

In one paragraph: the unit-lifecycle decision tree (R2 → SSH → worker_dead re-check → action) moves out of `BatchOrchestrator` into a new `unit_lifecycle` deep module exposing one function `decide_next_action(unit, runner, instance, is_done_in_r2) -> UnitAction`, where `UnitAction` is a tagged union of three frozen dataclasses (`Continue | Complete | Preempt`). The four-step belt-and-suspenders destroy protocol (stop → DELETE×retry → verify → re-destroy, with a second verification after resurrection cleanup) moves out of `VastaiRunner._rest_destroy` and the standalone helpers in `providers/vastai.py` into a new `providers/destroy` deep module exposing `belt_and_suspenders(stop_fn, delete_fn, verify_fn, *, policy: DestroyPolicy) -> DestroyResult`. The destroy protocol returns a typed `DestroyResult` (verdict + error context + attempt count), not a boolean; the request that started the refactor ("hide the timing constants in the protocol module") is reversed — the Vast.ai-discovered timings live in a `DestroyPolicy` so the loop in the generic module is provider-neutral. Two latent bugs disappear with the move: the orchestrator's zombie-sweep `_destroy_via_rest` (orchestrator.py:173) is a simplified copy that cannot handle resurrected instances and silently treats HTTP errors as "instance gone"; and the `BatchOrchestrator._check_unit` dispatch is mixed with state mutations, lock acquisition, and side effects across three methods. Both are corrected by the new module boundaries.

Diff vs v2:

- **+** `src/vastai_gpu_runner/unit_lifecycle.py` — `decide_next_action`, `Action` enum, `Continue | Complete | Preempt` plan types, `PreemptCause` enum, `ProgressSnapshot` dataclass
- **+** `src/vastai_gpu_runner/providers/destroy.py` — `DestroyResult` / `DestroyVerdict` dataclasses, `DestroyPolicy` dataclass, `belt_and_suspenders()` function
- **+** `src/vastai_gpu_runner/providers/destroy_adapters/vastai.py` — Vast.ai REST callbacks (`stop_fn`, `delete_fn`, `verify_fn`), `destroy_vastai_instance`, `read_vastai_api_key` (env-first precedence)
- **~** `BatchOrchestrator._check_unit` becomes a thin wrapper (one deprecation cycle only — the API reference documents it as inherited lifecycle surface) that delegates to `decide_next_action` and a shared dispatch helper. `_classify_live_unit` is **deleted**, not wrapped.
- **~** `BatchOrchestrator._poll_cycle_once` and `_check_unit` share a single dispatch helper so their logic cannot diverge
- **~** `BatchOrchestrator._sweep_zombies` routes through `destroy_vastai_instance` (with CLI fallback when no API key) and increments `killed` on confirmed destroy only
- **~** `VastaiRunner._rest_destroy` delegates to `destroy_vastai_instance` with the `allowed_images` ownership guard preserved
- **—** `orchestrator.py:_destroy_via_rest` (simplified copy, latent bug) **deleted**
- **—** `orchestrator.py:poll_instance_progress` (dead public API, no callers anywhere) **deleted**
- **—** `orchestrator.py:ensure_detached` (dead public API, no callers anywhere) **deleted**
- **—** `orchestrator.py:load_vastai_api_key` (byte-for-byte duplicate of `_read_vastai_api_key`) **deleted**; `_read_vastai_api_key` promoted to public `read_vastai_api_key()` with env-first precedence and fail-closed semantics
- **—** `providers/vastai.py:_rest_stop`, `_rest_delete_with_retries`, `_rest_verify_and_redestroy` **deleted**; absorbed into the Vast.ai adapter
- **—** `BatchOrchestrator._classify_live_unit` **deleted** (not wrapped); its tests move to `test_unit_lifecycle.py`
- **—** No changes to `runner.py` ABC signature
- **—** No changes to `state.py`, `storage/r2.py`, `worker/base.py`, `estimator/`

## Module taxonomy

The v3 doc adds two new deep modules around the existing ABCs. The `CloudRunner` ABC is unchanged; the new modules sit at the same layer as the existing provider modules.

### Existing ABCs (unchanged)

`CloudRunner` (Lane A provider lifecycle) and `BatchOrchestrator` (multi-unit orchestration) keep their public surface. The refactor narrows the implementation behind those ABCs.

### New: `unit_lifecycle` — owns the per-cycle decision tree

Owns: the decision rules (R2-first → SSH → worker_dead re-check → action mapping), the `Action` enum, the three plan dataclasses, the `PreemptCause` enum, the `ProgressSnapshot` normaliser.

Does **not** own: side effects (destroy, collect, capture_preempt_diagnostics, on_unit_*), lock acquisition, parallel fan-out, consumer hooks. Those stay on the orchestrator.

### New: `providers/destroy` — owns the belt-and-suspenders destroy protocol

Owns: the loop shape (stop → DELETE×retry → verify → re-destroy, with second verification after resurrection cleanup). The generic loop takes a `DestroyPolicy` for timing/retry constants so the loop itself is policy-agnostic. Returns a typed `DestroyResult`, never a boolean.

Does **not** own: REST URLs, API key paths, image-ownership guard, timing constants. The constants live in the policy (supplied by the adapter); the URLs and credentials live in the adapter.

### New: `providers/destroy_adapters/vastai.py` — Vast.ai REST callbacks

Owns: the three Vast.ai REST endpoints (`PUT state=stopped`, `DELETE`, `GET for verify`), the `read_vastai_api_key` call with env-first precedence and fail-closed semantics, the `allowed_images` ownership guard with `is not None` semantics, the `DestroyPolicy` constants (the Vast.ai-discovered 5s verify delay, 3s retry sleep, 3 max delete attempts).

The RunPod adapter lands separately when `RunPodRunner` ships (roadmap item 2). It will register its own `stop_fn` / `delete_fn` / `verify_fn` callbacks and its own `DestroyPolicy` (different retry timing may be appropriate for RunPod's API).

## Layered design (v3)

```
┌─────────────────────────────────────────────────┐
│  CLI (cli.py)                                   │  User-facing commands
├─────────────────────────────────────────────────┤
│  BatchOrchestrator (batch.py)                   │  Phase loop + side-effect dispatchers
│    └── calls unit_lifecycle.decide_next_action  │  (per-unit decision is delegated)
├─────────────────────────────────────────────────┤
│  unit_lifecycle (unit_lifecycle.py)             │  NEW: decision tree, no side effects
│    └── Action enum + Continue/Complete/Preempt  │
│    └── PreemptCause enum + ProgressSnapshot     │
├─────────────────────────────────────────────────┤
│  Orchestrator utils (orchestrator.py)           │  Zombie sweep routes through destroy
│    └── sweep_zombies (delegates to destroy)     │
├─────────────────────────────────────────────────┤
│  providers/destroy (providers/destroy.py)       │  NEW: belt-and-suspenders protocol
│    └── belt_and_suspenders → DestroyResult      │
├─────────────────────────────────────────────────┤
│  providers/destroy_adapters/vastai.py           │  NEW: Vast.ai REST callbacks + policy
│  providers/destroy_adapters/runpod.py (future)  │  [roadmap item 2]
├─────────────────────────────────────────────────┤
│  CloudRunner (runner.py) ── Lane A ABC          │  Provider-agnostic lifecycle
├──────────────┬──────────────┬───────────────────┤
│ VastaiRunner │ RunPodRunner │ LocalRunner       │  Lane A implementations
├──────────────┴──────────────┴───────────────────┤
│  SSH (ssh.py)      — used by Vast.ai, RunPod    │
│  subprocess        — used by Local              │
├─────────────────────────────────────────────────┤
│  Workers (worker/base.py)                       │  GPU-side execution
│    └── imports inference/ (Lane C, optional)    │
├─────────────────────────────────────────────────┤
│  Storage (storage/r2.py)                        │  Result persistence
├─────────────────────────────────────────────────┤
│  State (state.py)                               │  Crash recovery
└─────────────────────────────────────────────────┘
```

## `unit_lifecycle` shape

Single public entry point, single tagged union, three frozen dataclasses, plus a structured preempt cause and a snapshot normaliser.

```python
# src/vastai_gpu_runner/unit_lifecycle.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, ClassVar, Protocol, TypeVar

UnitT = TypeVar("UnitT")
logger = logging.getLogger(__name__)


class Action(StrEnum):
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

    The decision module never inspects the raw dict. Callers
    (``decide_next_action``) pass the snapshot through ``normalise``
    first, which coerces missing or malformed fields to safe
    defaults. Missing or contradictory fields never trigger a
    destructive action — the conservative result is "still running".
    """
    complete: bool = False
    worker_dead: bool = False
    log_tail: str = ""

    @classmethod
    def normalise(cls, raw: dict[str, object] | None) -> "ProgressSnapshot":
        if not isinstance(raw, dict):
            logger.warning("check_progress returned non-mapping: %r", raw)
            return cls()
        complete = raw.get("complete")
        worker_dead = raw.get("worker_dead")
        log_tail = raw.get("log_tail")
        if not isinstance(complete, bool):
            logger.warning("check_progress.complete is not bool: %r", complete)
            complete = False
        if not isinstance(worker_dead, bool):
            logger.warning("check_progress.worker_dead is not bool: %r", worker_dead)
            worker_dead = False
        if complete and worker_dead:
            # Contradictory — prefer the conservative read.
            logger.warning("check_progress has complete=True and worker_dead=True")
            worker_dead = False
        if not isinstance(log_tail, str):
            log_tail = ""
        return cls(complete=bool(complete), worker_dead=bool(worker_dead), log_tail=log_tail)


@dataclass(frozen=True)
class Continue:
    """The unit is still running. No action this cycle."""
    action: ClassVar[Action] = Action.CONTINUE


@dataclass(frozen=True)
class Complete:
    """The unit has finished (R2 confirms DONE OR SSH reports complete).
    Orchestrator should finalise: collect + destroy."""
    action: ClassVar[Action] = Action.COMPLETE


@dataclass(frozen=True)
class Preempt:
    """The worker died silently. Orchestrator should capture diagnostics,
    destroy the instance, and invoke the consumer's ``classify_failure``
    hook to decide retry-vs-fatal.
    """
    action: ClassVar[Action] = Action.PREEMPT
    cause: PreemptCause
    detail: str | None = None


UnitAction = Continue | Complete | Preempt


class CloudRunnerLike(Protocol):
    """Structural type for the runner dependency. The real CloudRunner
    satisfies this; tests pass a tiny fake."""
    def check_progress(self, instance: object) -> dict[str, object]: ...


def decide_next_action(
    unit: UnitT,
    runner: CloudRunnerLike,
    instance: object,
    is_done_in_r2: Callable[[UnitT], bool],
) -> UnitAction:
    """Classify a live unit and return the next action.

    Performs a *localised observation sequence*: R2 first, then SSH,
    then R2 again on worker_dead (the between-check upload case).
    The two R2 reads are in immediate succession inside this
    function — no orchestrator code can interleave between them —
    so the SSH verdict and the R2 verdict cannot disagree about a
    checkpoint that landed between the two reads.

    Exception policy:

    * ``is_done_in_r2(unit)`` raises on the first call → log, treat
      as not-done, continue to SSH.
    * ``runner.check_progress(instance)`` raises → log, treat as
      "still running" (transient SSH flakiness — the next poll cycle
      retries).
    * ``is_done_in_r2(unit)`` raises on the worker_dead re-check →
      return ``Continue``. Destroying while final-upload status is
      unknown risks losing recoverable results.
    """
    # Phase 1: R2-first check.
    try:
        if is_done_in_r2(unit):
            return Complete()
    except Exception as exc:
        logger.warning(
            "decide_next_action: initial R2 check raised %s — continuing to SSH",
            exc,
        )

    # Phase 2: SSH check (with exception handler).
    try:
        progress = ProgressSnapshot.normalise(runner.check_progress(instance))
    except Exception as exc:
        logger.warning(
            "decide_next_action: check_progress raised %s — treating as running",
            exc,
        )
        return Continue()

    if progress.complete:
        return Complete()

    if progress.worker_dead:
        # Re-check R2 once more — worker may have uploaded between
        # the SSH check and now. If this raises, fall back to
        # Continue (don't destroy on unknown final-upload status).
        try:
            if is_done_in_r2(unit):
                return Complete()
        except Exception as exc:
            logger.warning(
                "decide_next_action: worker_dead R2 re-check raised %s — "
                "returning Continue to avoid losing recoverable results",
                exc,
            )
            return Continue()
        return Preempt(cause=PreemptCause.WORKER_DIED, detail=progress.log_tail)

    return Continue()
```

Why per-unit, per-cycle (not per-run). v15's `decide()` is called once before the lifecycle starts and the plan is terminal. The vastai-gpu-runner decision is repeated every poll cycle (default 30s) for every live unit, so the plan types are *transient actions* ("what now?"), not *terminal plans* ("what's the verdict?"). Per-unit keeps the function's interface small; the orchestrator iterates and fans out the parallel finalise.

Why the `is_done_in_r2` callable. The current `_classify_live_unit` is a method on `BatchOrchestrator` and calls `self.unit_is_done_in_r2(unit)` — the consumer's hook. To make `decide_next_action` a module-level function (not a method), we pass the hook as a callable. The orchestrator does `self.unit_is_done_in_r2` in the call site. This decouples the decision from the orchestrator without forcing the consumer to implement a method on the new module.

Why the protocol type for `runner`. Injecting the full `CloudRunner` class would couple the decision module to the `runner.py` hierarchy. A structural protocol with one method (`check_progress`) lets tests pass a tiny stub without dragging in the rest of the ABC.

Why `ProgressSnapshot` normaliser. The current `_classify_live_unit` does `progress.get("complete")` and `progress.get("worker_dead")` — both default to falsy if missing. A missing `complete` key, a `None` value, or a contradictory `complete=True, worker_dead=True` all silently propagate to the wrong verdict. The normaliser guards each coercion and logs every malformed response.

Why `PreemptCause` not a free string. `classify_failure` (the consumer's hook) may eventually branch on the cause. A `str` reason field becomes an unstable policy API the moment two consumers want to dispatch differently. The enum is the stable contract; `detail` carries the human-readable context.

## `providers/destroy` shape

Single public entry point, three callbacks per provider, structured `DestroyResult` return type, policy-supplied timings.

```python
# src/vastai_gpu_runner/providers/destroy.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DestroyVerdict(StrEnum):
    """What the destroy attempt actually achieved.

    Never collapse these into a bool. The Vast.ai dashboard API
    returns 4xx and 5xx on transient issues; treating those as
    "destroyed" silently leaks running instances.
    """
    DESTROYED = "destroyed"  # confirmed gone on first verification
    LEAKED = "leaked"        # verified PRESENT after resurrection cleanup
    UNKNOWN = "unknown"      # verification was uncertain (auth, network, etc.)


@dataclass(frozen=True)
class DestroyPolicy:
    """Provider-specific retry and timing policy.

    The Vast.ai-discovered values (5s verify delay, 3s retry sleep,
    3 max delete attempts) live in the Vast.ai adapter, not here.
    The generic module owns the loop shape; the policy owns the
    numbers. RunPod may need different values.
    """
    verify_delay_s: float
    retry_delay_s: float
    max_delete_attempts: int
    verify_after_resurrection: bool = True


@dataclass(frozen=True)
class DestroyResult:
    """Outcome of a belt-and-suspenders destroy attempt.

    Carries the verdict plus enough context for the caller to
    surface unresolved leaks to the user. ``stop_error`` is None
    unless the best-effort stop call failed; ``last_status_code``
    is the HTTP status of the most recent verify call (or None);
    ``attempts`` is the total number of DELETE attempts made.
    """
    verdict: DestroyVerdict
    attempts: int = 0
    stop_error: str | None = None
    last_status_code: int | None = None


class StopFn(Protocol):
    """Best-effort stop (kills stuck Docker pulls). Must not raise."""
    def __call__(self) -> None: ...


class DeleteFn(Protocol):
    """DELETE the instance. Returns True on 2xx OR 404 (idempotent)."""
    def __call__(self) -> bool: ...


class VerifyFn(Protocol):
    """Verify the instance is gone.

    Returns a ``DestroyVerdict``:
      * ``DESTROYED`` when the instance is confirmed gone (404,
        explicit ``actual_status == "destroyed"``, empty status).
      * ``PRESENT`` when the instance is still alive.
      * ``UNKNOWN`` for any other HTTP status (401, 403, 429, 5xx)
        or any exception. May also set ``last_status_code`` via
        the wrapped return (see adapter).
    """
    def __call__(self) -> DestroyVerdict: ...


def belt_and_suspenders(
    *,
    stop_fn: StopFn,
    delete_fn: DeleteFn,
    verify_fn: VerifyFn,
    policy: DestroyPolicy,
) -> DestroyResult:
    """Four-step belt-and-suspenders destroy with structured result.

    1. ``stop_fn()`` — best-effort. Its error is recorded in
       ``DestroyResult.stop_error`` but does not block the
       subsequent DELETE call. A stop failure alone never causes
       a return; we always try DELETE.
    2. ``delete_fn()`` retried up to ``policy.max_delete_attempts``
       times with ``policy.retry_delay_s`` between attempts.
    3. After ``policy.verify_delay_s``, ``verify_fn()`` checks
       whether the instance is gone.
    4. If the first verification is ``PRESENT``, run stop + delete
       once more (no retries) and verify again. If the second
       verification is still ``PRESENT``, return ``LEAKED``; if
       ``DESTROYED``, return ``DESTROYED``; if ``UNKNOWN``, return
       ``UNKNOWN``.

    Returns ``DestroyResult`` with verdict + error context + attempt
    count. Never raises from the protocol layer.
    """
    attempts = 0
    stop_error: str | None = None

    # Phase 1: best-effort stop. Must not block DELETE.
    try:
        stop_fn()
    except Exception as exc:
        stop_error = f"{type(exc).__name__}: {exc}"
        logger.warning("belt_and_suspenders: stop_fn raised %s — continuing", stop_error)

    # Phase 2: DELETE with retries.
    for attempt in range(policy.max_delete_attempts):
        attempts += 1
        try:
            if delete_fn():
                break
        except Exception as exc:
            logger.warning(
                "belt_and_suspenders: delete_fn raised %s (attempt %d/%d)",
                exc, attempt + 1, policy.max_delete_attempts,
            )
        if attempt < policy.max_delete_attempts - 1:
            time.sleep(policy.retry_delay_s)

    # Phase 3: first verification.
    time.sleep(policy.verify_delay_s)
    try:
        verdict = verify_fn()
    except Exception as exc:
        logger.warning("belt_and_suspenders: verify_fn raised %s — returning UNKNOWN", exc)
        return DestroyResult(
            verdict=DestroyVerdict.UNKNOWN,
            attempts=attempts,
            stop_error=stop_error,
        )

    if verdict == DestroyVerdict.DESTROYED:
        return DestroyResult(
            verdict=DestroyVerdict.DESTROYED,
            attempts=attempts,
            stop_error=stop_error,
        )
    if verdict == DestroyVerdict.UNKNOWN:
        return DestroyResult(
            verdict=DestroyVerdict.UNKNOWN,
            attempts=attempts,
            stop_error=stop_error,
        )

    # Phase 4: resurrection cleanup, then re-verify.
    if not policy.verify_after_resurrection:
        return DestroyResult(
            verdict=DestroyVerdict.LEAKED,
            attempts=attempts,
            stop_error=stop_error,
        )
    try:
        stop_fn()
        time.sleep(policy.retry_delay_s)
        attempts += 1
        delete_fn()
    except Exception as exc:
        logger.warning("belt_and_suspenders: resurrection cleanup raised %s", exc)
        return DestroyResult(
            verdict=DestroyVerdict.UNKNOWN,
            attempts=attempts,
            stop_error=stop_error,
        )

    # Phase 5: second verification.
    time.sleep(policy.verify_delay_s)
    try:
        verdict = verify_fn()
    except Exception as exc:
        logger.warning("belt_and_suspenders: second verify_fn raised %s — UNKNOWN", exc)
        return DestroyResult(
            verdict=DestroyVerdict.UNKNOWN,
            attempts=attempts,
            stop_error=stop_error,
        )

    if verdict == DestroyVerdict.DESTROYED:
        return DestroyResult(
            verdict=DestroyVerdict.DESTROYED,
            attempts=attempts,
            stop_error=stop_error,
        )
    if verdict == DestroyVerdict.PRESENT:
        return DestroyResult(
            verdict=DestroyVerdict.LEAKED,
            attempts=attempts,
            stop_error=stop_error,
        )
    return DestroyResult(
        verdict=DestroyVerdict.UNKNOWN,
        attempts=attempts,
        stop_error=stop_error,
    )
```

Why callback-based, not subclassed. v14's `checkpoint.py` is a single concrete module (no adapters) because the checkpoint format is provider-agnostic. The destroy protocol is provider-agnostic in its *loop* but provider-specific in its *endpoints* and *timing policy*. By accepting `stop_fn`, `delete_fn`, `verify_fn` callbacks plus a `DestroyPolicy`, the protocol stays portable across Vast.ai, RunPod, and any future adapter without forcing a class hierarchy.

Why typed `DestroyResult`, not bool. The current `_rest_verify_and_redestroy` returns `None` on the happy path and "logs a warning" on the resurrection path. The caller cannot distinguish "destroyed" from "leaked" from "verify was uncertain." A 401 or 500 from the dashboard API is silently treated as "destroyed." The structured `DestroyResult` makes these outcomes distinguishable.

Why `stop_fn` is best-effort. A stop timeout or 401 must not prevent the more important DELETE attempt. The stop call's purpose is to kill stuck Docker pulls; if it fails, DELETE is still the durable teardown. Recording the error in the result preserves observability without breaking the protocol.

Why `DestroyPolicy` is adapter-supplied. The 5s verify delay and 3s retry sleep are Vast.ai production discoveries. RunPod may need different values (faster retry, no resurrection check). Hardcoding them in the generic module would bake Vast.ai assumptions into the protocol. The policy is the place for that knowledge.

## Vast.ai adapter shape

```python
# src/vastai_gpu_runner/providers/destroy_adapters/vastai.py
from __future__ import annotations

import logging
import os
from pathlib import Path

import requests

from vastai_gpu_runner.providers.destroy import (
    DestroyPolicy,
    DestroyResult,
    DestroyVerdict,
    belt_and_suspenders,
)
from vastai_gpu_runner.providers.vastai import verify_instance_ownership

logger = logging.getLogger(__name__)

BASE_URL = "https://console.vast.ai/api/v0/instances"

# Vast.ai-discovered production values. Different providers may need
# different timings — register their own DestroyPolicy.
VASTAI_POLICY = DestroyPolicy(
    verify_delay_s=5.0,
    retry_delay_s=3.0,
    max_delete_attempts=3,
    verify_after_resurrection=True,
)


def read_vastai_api_key() -> str:
    """Resolve the Vast.ai API key with env-first precedence.

    Order:
      1. ``VASTAI_API_KEY`` environment variable. A present-but-empty
         value fails closed (returns empty string, refusing to fall
         through to the file — a stale ``~/.cloud-credentials`` would
         silently select the wrong key).
      2. ``~/.config/vastai/vast_api_key`` file.
      3. ``~/.vast_api_key`` file.

    Returns:
        Non-empty API key, or empty string if no source is configured.
    """
    env_key = os.environ.get("VASTAI_API_KEY")
    if env_key is not None:
        if env_key.strip() == "":
            logger.warning(
                "VASTAI_API_KEY is set but empty — ignoring and NOT "
                "falling through to file credentials (fail-closed)."
            )
            return ""
        return env_key.strip()
    for path in (
        Path("~/.config/vastai/vast_api_key").expanduser(),
        Path("~/.vast_api_key").expanduser(),
    ):
        if path.exists():
            return path.read_text().strip()
    return ""


def vastai_stop(instance_id: str, hdrs: dict[str, str]) -> None:
    """Best-effort stop. Raises on transport failures (caught by
    ``belt_and_suspenders``)."""
    requests.put(
        f"{BASE_URL}/{instance_id}/",
        headers={**hdrs, "Content-Type": "application/json"},
        json={"state": "stopped"},
        timeout=10,
    )


def vastai_delete(instance_id: str, hdrs: dict[str, str]) -> bool:
    """DELETE. Returns True on 2xx OR 404 (already gone)."""
    resp = requests.delete(f"{BASE_URL}/{instance_id}/", headers=hdrs, timeout=15)
    return resp.status_code in (200, 204, 404)


def vastai_verify(instance_id: str, hdrs: dict[str, str]) -> DestroyVerdict:
    """Verify the instance is gone.

    404 → DESTROYED. 200 with ``actual_status in ("", "destroyed")``
    → DESTROYED. 200 with any other status → PRESENT. Any other
    HTTP status (401, 403, 429, 5xx) or any exception → UNKNOWN.
    """
    try:
        verify = requests.get(f"{BASE_URL}/{instance_id}/", headers=hdrs, timeout=10)
    except Exception as exc:
        logger.warning("vastai_verify: GET raised %s — UNKNOWN", exc)
        return DestroyVerdict.UNKNOWN
    if verify.status_code == 404:
        return DestroyVerdict.DESTROYED
    if verify.status_code != 200:
        logger.warning(
            "vastai_verify: GET %s returned %d — UNKNOWN",
            instance_id, verify.status_code,
        )
        return DestroyVerdict.UNKNOWN
    try:
        vstatus = verify.json().get("actual_status", "")
    except Exception as exc:
        logger.warning("vastai_verify: JSON parse raised %s — UNKNOWN", exc)
        return DestroyVerdict.UNKNOWN
    if vstatus in ("", "destroyed"):
        return DestroyVerdict.DESTROYED
    return DestroyVerdict.PRESENT


def destroy_vastai_instance(
    instance_id: str,
    *,
    allowed_images: set[str] | None = None,
) -> DestroyResult:
    """Belt-and-suspenders destroy with the ownership guard.

    The ownership guard is applied first: if ``allowed_images`` is
    not None (note: an empty set *rejects every image*, not skips
    the guard) and the instance's image is not in the set, return
    ``DestroyResult(verdict=DESTROYED, ...)`` with a logged
    refusal — no API call is made. This protects against
    cross-project accidents on shared Vast.ai accounts.

    If no API key is configured, returns ``DestroyResult(verdict=UNKNOWN)``
    and the caller (the zombie sweep) is responsible for falling
    back to the CLI ``vastai destroy instance`` path.
    """
    if allowed_images is not None and not verify_instance_ownership(
        instance_id, allowed_images=allowed_images
    ):
        logger.error(
            "REFUSED to destroy instance %s — ownership check failed.",
            instance_id,
        )
        return DestroyResult(verdict=DestroyVerdict.UNKNOWN)
    api_key = read_vastai_api_key()
    if not api_key:
        return DestroyResult(verdict=DestroyVerdict.UNKNOWN)
    hdrs = {"Authorization": f"Bearer {api_key}"}
    return belt_and_suspenders(
        stop_fn=lambda: vastai_stop(instance_id, hdrs),
        delete_fn=lambda: vastai_delete(instance_id, hdrs),
        verify_fn=lambda: vastai_verify(instance_id, hdrs),
        policy=VASTAI_POLICY,
    )
```

The existing `VastaiRunner._rest_destroy` collapses into `destroy_vastai_instance(instance_id, allowed_images=self.allowed_images)`. The `allowed_images` ownership guard is preserved — it lives at the adapter layer, not in the protocol module, because the guard is Vast.ai-specific (the other providers do not currently have an image-allowlist feature).

The `is not None` guard (not `if allowed_images and ...`) is the safety-critical fix. An empty `set()` passed as `allowed_images` rejects every image, which is the correct conservative behaviour; the previous `if allowed_images and ...` shape would silently skip the guard on an empty set and allow cross-project deletion.

## Zombie sweep integration

The orchestrator's `_sweep_zombies` routes through `destroy_vastai_instance` and falls back to the CLI when no API key is configured. The `killed` counter is incremented on confirmed destroy only — unresolved leaks are surfaced, not hidden.

```python
# In BatchOrchestrator._sweep_zombies (after the refactor):

def _sweep_zombies(self) -> int:
    """Destroy Vast.ai instances not tracked by live_runners.

    Each candidate instance is destroyed via the belt-and-suspenders
    adapter. If no API key is configured, the sweep falls back to
    the CLI ``vastai destroy instance`` path. The ``killed`` counter
    is incremented only on confirmed destroy (DestroyResult.verdict
    == DESTROYED). Unresolved leaks (LEAKED, UNKNOWN) are logged
    so the user can investigate.
    """
    # ... enumerate candidates via api_call (existing logic) ...
    killed = 0
    for iid in zombie_candidates:
        result = destroy_vastai_instance(
            iid,
            allowed_images=self._allowed_images,
        )
        if result.verdict == DestroyVerdict.DESTROYED:
            killed += 1
        elif result.verdict == DestroyVerdict.LEAKED:
            logger.error(
                "Zombie sweep: instance %s could not be destroyed after "
                "resurrection cleanup — operator intervention required",
                iid,
            )
        elif result.verdict == DestroyVerdict.UNKNOWN:
            if not read_vastai_api_key():
                # Fall back to CLI when no API key is configured.
                try:
                    vastai_cmd(["destroy", "instance", iid], timeout=15)
                    killed += 1
                except Exception as exc:
                    logger.warning(
                        "Zombie sweep: CLI fallback failed for %s: %s",
                        iid, exc,
                    )
            else:
                logger.warning(
                    "Zombie sweep: instance %s destroy result was UNKNOWN "
                    "(stop_error=%s, last_status=%s, attempts=%d)",
                    iid, result.stop_error, result.last_status_code, result.attempts,
                )
    if killed:
        logger.info("Zombie sweep: confirmed-destroyed %d instance(s)", killed)
    return killed
```

## Thin-wrapper plan

The v15 precedent was *deletion*, not wrapper retention. This v3 follows v15's discipline.

```python
# In BatchOrchestrator, after the refactor:

def _check_unit(
    self,
    runner: CloudRunner,
    instance: CloudInstance,
    unit: UnitT,
) -> Literal["completed", "running", "preempted", "failed"]:
    """Deprecated. Will be removed in the next minor release.

    The docstring on the previous version said: "Kept for
    backwards-compat with direct callers (unit tests, consumers
    that prefer synchronous single-unit polling)." The two
    call sites (this method and ``_poll_cycle_once``) now share
    a single dispatch helper so their logic cannot diverge. This
    thin wrapper is retained for one deprecation cycle only.
    """
    unit_key = self.unit_key(unit)
    return self._dispatch_unit(runner, instance, unit, unit_key)


def _dispatch_unit(
    self,
    runner: CloudRunner,
    instance: CloudInstance,
    unit: UnitT,
    unit_key: str,
) -> Literal["completed", "running", "preempted", "failed"]:
    """Single dispatch helper for both ``_check_unit`` and
    ``_poll_cycle_once``. They cannot diverge because they call
    this and only this.
    """
    from vastai_gpu_runner.unit_lifecycle import (
        decide_next_action,
        Continue, Complete, Preempt,
    )
    action = decide_next_action(
        unit, runner, instance, self.unit_is_done_in_r2,
    )
    match action:
        case Continue():
            return "running"
        case Complete():
            return self._finalise_completed(runner, instance, unit, unit_key)
        case Preempt(cause=cause, detail=detail):
            with contextlib.suppress(Exception):
                self.capture_preempt_diagnostics(runner, instance, unit)
            with contextlib.suppress(Exception):
                runner.destroy_instance(instance)
            with self._state_lock:
                self._handle_instance_loss(
                    unit, unit_key, _format_preempt_reason(cause, detail),
                )
            return "preempted"
```

`_classify_live_unit` is **deleted** (not wrapped). Its tests move to `test_unit_lifecycle.py` against the new `decide_next_action` interface. The v15 deletion is the right precedent — the wrapper added no value over a direct delegate, and tests that mock the dispatcher at the string level can be re-pointed at the new module.

The match statement is the v15-faithful shape: the runner is a thin dispatcher that matches on the plan type. The deprecation warning on `_check_unit` is emitted via `warnings.warn(..., DeprecationWarning, stacklevel=2)` so static analysers and unit tests can flag it.

## ABC changes required

**None.** `CloudRunner` keeps its public interface. The new `unit_lifecycle` module does not subclass the runner; it takes a `runner` parameter via protocol. The new `providers/destroy` module takes callbacks plus a `DestroyPolicy`; `VastaiRunner` still calls `destroy_vastai_instance` from its own `destroy_instance` method.

The RunPod adapter does not exist yet. When `RunPodRunner` ships (roadmap item 2), it gets a sibling `providers/destroy_adapters/runpod.py` with the same callback shape and its own `DestroyPolicy`. No ABC change.

## Glossary of new terms

| Term | Meaning |
|---|---|
| `Action` | StrEnum; one of `CONTINUE`, `COMPLETE`, `PREEMPT`. The action the orchestrator should take based on the decision. |
| `UnitAction` | The tagged union `Continue | Complete | Preempt`. The return type of `decide_next_action`. |
| `Continue` | Plan dataclass; the unit is still running, no action this cycle. |
| `Complete` | Plan dataclass; the unit has finished (R2 or SSH confirms). Orchestrator should finalise. |
| `Preempt` | Plan dataclass; the worker died silently. Orchestrator should capture diagnostics, destroy, and invoke consumer's `classify_failure`. Carries `cause: PreemptCause` and optional `detail: str`. |
| `PreemptCause` | StrEnum; the reason the worker died. Currently `WORKER_DIED`; reserved for `BOOT_TIMEOUT_RETRY_EXHAUSTED`, `SSH_LOST`, `GPU_OOM`. |
| `ProgressSnapshot` | Frozen dataclass normalising the runner's `check_progress` response before classification. Missing or malformed fields default to the conservative read. |
| `decide_next_action` | The single public function in `unit_lifecycle.py`. Performs a localised observation sequence (R2 → SSH → R2-on-worker_dead), returns the action. |
| `DestroyResult` | Frozen dataclass; the outcome of a belt-and-suspenders destroy attempt. Carries `verdict: DestroyVerdict`, `attempts`, `stop_error`, `last_status_code`. |
| `DestroyVerdict` | StrEnum; `DESTROYED`, `LEAKED`, or `UNKNOWN`. Never collapses into a bool. |
| `DestroyPolicy` | Frozen dataclass; provider-specific retry/timing policy (`verify_delay_s`, `retry_delay_s`, `max_delete_attempts`, `verify_after_resurrection`). |
| `belt_and_suspenders` | The single public function in `providers/destroy.py`. Four-step destroy loop with second verification after resurrection cleanup. Returns `DestroyResult`. |
| `StopFn` / `DeleteFn` / `VerifyFn` | The callback protocols for the destroy loop. `verify_fn` returns `DestroyVerdict` (not bool). |
| `destroy_vastai_instance` | The Vast.ai adapter for the belt-and-suspenders protocol. Lives in `providers/destroy_adapters/vastai.py`. |
| `read_vastai_api_key` | The Vast.ai credential loader with env-first precedence and fail-closed semantics. |

## Resolved design decisions

These were open questions in the v3 draft; they are resolved in this revision.

### Credential precedence (Vast.ai)

`read_vastai_api_key()` resolves in this order:

1. `VASTAI_API_KEY` environment variable. A present-but-empty value fails closed (returns empty string, refuses to fall through to file credentials).
2. `~/.config/vastai/vast_api_key` file.
3. `~/.vast_api_key` file.

A present-empty env var is not the same as an absent one. A stale `~/.cloud-credentials` file would silently select the wrong key; the fail-closed semantics prevent this.

Multi-provider credential unification is deferred until `RunPodRunner` lands.

### Destroy callback return values

`verify_fn` returns `DestroyVerdict` (an enum), not a bool. The protocol interprets:
- `DESTROYED` → confirmed gone, return `DestroyResult(verdict=DESTROYED)`.
- `PRESENT` → resurrection detected, run stop + delete + re-verify.
- `UNKNOWN` → cannot determine (auth error, transient outage, JSON parse failure). Return `DestroyResult(verdict=UNKNOWN)` with `last_status_code` if available.

`delete_fn` returns `bool` (success or no — idempotent on 404). `stop_fn` returns `None` and must not raise (the protocol wraps it in try/except).

### `Preempt` reason

`Preempt` carries `cause: PreemptCause` (enum) and optional `detail: str | None`. No free-form `reason: str` field. The enum is the stable contract; `detail` is for humans only.

### Context module location

No `CONTEXT.md` is created. The glossary stays in `architecture-v3.md`. Future docs that need the terms link here. If the project later wants a single glossary file, this section is the seed.

## Critical test cases

The following test cases must be enumerated in the implementation PR. The design doc lists them so the implementation reviewer can verify the test surface is complete.

### `unit_lifecycle` tests

- **Malformed progress response.** `check_progress` returns `{}`, `{"running": False}`, `{"complete": "true"}` (string, not bool), `{"complete": True, "worker_dead": True}` (contradictory), or a non-dict (None, list). Each coerces to `Continue` (conservative). The contradictory case logs a warning and downgrades to `running` (not preempted).
- **Initial R2 raises.** `is_done_in_r2` raises on the first call. The logs the exception and continues to SSH. SSH verdict is honoured.
- **Worker_dead R2 re-check raises.** `is_done_in_r2` raises on the second call (after `worker_dead`). Returns `Continue` (not `Preempt`) — destroying on unknown final-upload status risks losing recoverable results.
- **`check_progress` raises.** Returns `Continue` (transient SSH flakiness).
- **Happy path: R2 done.** `is_done_in_r2` returns True on the first call. Returns `Complete`. (SSH not consulted.)
- **Happy path: SSH complete.** R2 not done, `check_progress` returns `{"complete": True}`. Returns `Complete`.
- **Happy path: worker dead, R2 re-check done.** Returns `Complete`.
- **Happy path: worker dead, R2 re-check not done.** Returns `Preempt(cause=WORKER_DIED, detail=log_tail)`.

### `providers/destroy` tests

- **Happy delete.** `stop_fn` succeeds, `delete_fn` returns True on first attempt, `verify_fn` returns `DESTROYED`. Result: `DestroyResult(verdict=DESTROYED, attempts=1)`.
- **Delete retry.** `delete_fn` returns False twice, True on the third. `verify_fn` returns `DESTROYED`. Result: `DestroyResult(verdict=DESTROYED, attempts=3)`.
- **Resurrection cleanup succeeds.** First verify returns `PRESENT`. Run stop + delete + re-verify. Second verify returns `DESTROYED`. Result: `DestroyResult(verdict=DESTROYED, attempts=2)`.
- **Resurrection cleanup fails.** First verify returns `PRESENT`. Re-verify still returns `PRESENT`. Result: `DestroyResult(verdict=LEAKED, attempts=2)`.
- **Unknown verification.** `verify_fn` returns `UNKNOWN` on first call. Result: `DestroyResult(verdict=UNKNOWN)`. No resurrection cleanup.
- **Stop failure.** `stop_fn` raises. The error is recorded in `stop_error`; the DELETE still runs. Result: `DestroyResult(verdict=DESTROYED, stop_error="...")`.
- **Unknown after resurrection.** First verify returns `PRESENT`. Resurrection cleanup completes. Second verify returns `UNKNOWN`. Result: `DestroyResult(verdict=UNKNOWN)`.

### Adapter tests

- **Empty allowlist.** `allowed_images=set()` (empty, not None). Instance image is any value. Returns `DestroyResult(verdict=UNKNOWN)` with refusal logged. **This is the safety-critical test for the blocker fix.**
- **None allowlist.** `allowed_images=None`. No guard applied. Proceeds to API call.
- **Matching allowlist.** Instance image is in the set. Proceeds.
- **Non-matching allowlist.** Instance image is not in the set. Returns `DestroyResult(verdict=UNKNOWN)` with refusal logged.
- **`VASTAI_API_KEY` env empty.** `read_vastai_api_key()` returns empty string and does not fall through to file credentials. Fail-closed.
- **`VASTAI_API_KEY` env set.** `read_vastai_api_key()` returns the env value.
- **Vast.ai verify 404.** Returns `DestroyVerdict.DESTROYED`.
- **Vast.ai verify 200 with `actual_status == "destroyed"`.** Returns `DestroyVerdict.DESTROYED`.
- **Vast.ai verify 200 with `actual_status == "running"`.** Returns `DestroyVerdict.PRESENT`.
- **Vast.ai verify 401 / 403 / 429 / 500.** Returns `DestroyVerdict.UNKNOWN`.
- **Vast.ai verify JSON parse failure.** Returns `DestroyVerdict.UNKNOWN`.

## Migration checklist

For the implementer (one PR, one PR-number):

1. Delete `orchestrator.py:poll_instance_progress` and `orchestrator.py:ensure_detached`. Update `CHANGELOG.md`, `docs/api.md`, `docs/architecture.md` to reflect the deletion.
2. Delete `orchestrator.py:load_vastai_api_key`. Add `read_vastai_api_key()` to `providers/destroy_adapters/vastai.py` with env-first precedence and fail-closed semantics. Remove the old `_read_vastai_api_key` from `providers/vastai.py` (the new module owns it).
3. New file `src/vastai_gpu_runner/providers/destroy.py` (the protocol module: `DestroyResult`, `DestroyVerdict`, `DestroyPolicy`, `belt_and_suspenders`).
4. New file `src/vastai_gpu_runner/providers/destroy_adapters/vastai.py` (the Vast.ai adapter: `vastai_stop`, `vastai_delete`, `vastai_verify`, `destroy_vastai_instance`, `read_vastai_api_key`, `VASTAI_POLICY`).
5. Update `VastaiRunner._rest_destroy` to delegate to `destroy_vastai_instance`. Delete the standalone helpers `_rest_stop`, `_rest_delete_with_retries`, `_rest_verify_and_redestroy` from `providers/vastai.py`.
6. Update `orchestrator.py:_sweep_zombies` to route through `destroy_vastai_instance` with CLI fallback when no API key is configured. Increment `killed` on confirmed destroy only. Delete `orchestrator.py:_destroy_via_rest`.
7. New file `src/vastai_gpu_runner/unit_lifecycle.py` (the decision module: `Action`, `PreemptCause`, `ProgressSnapshot`, `Continue` / `Complete` / `Preempt`, `decide_next_action`).
8. Update `BatchOrchestrator._poll_cycle_once` and `_check_unit` to use `_dispatch_unit` (single dispatch helper). Delete `_classify_live_unit`. Mark `_check_unit` deprecated.
9. New tests `tests/test_unit_lifecycle.py`, `tests/test_destroy.py`, `tests/test_destroy_adapters_vastai.py`. Update `tests/test_batch.py` and `tests/test_batch_orchestrator.py`.
10. Update `CHANGELOG.md` with the v3 changes.

## Review process

This design is published as a PR for review before implementation begins. Reviewers (code owner + ChatGPT-with-GitHub-plugin) should focus on:

- **Seam placement.** Is the `unit_lifecycle` ↔ `BatchOrchestrator` seam at the right boundary? Is the `providers/destroy` ↔ `providers/destroy_adapters/vastai.py` split clean?
- **Typed destroy result.** Does `DestroyResult` + `DestroyVerdict` cover every case the reviewer named (gone / already-gone / present / unknown)?
- **Safety behaviour.** Does the empty-allowlist test (`allowed_images=set()` rejects all images) catch the regression that the previous draft introduced?
- **R2 exception policy.** Are the three exception paths (initial R2, check_progress, worker_dead re-check) all handled the way the reviewer asked for?
- **`Preempt` shape.** Is `Preempt(cause: PreemptCause, detail: str | None)` the right shape, or should `cause` carry more structure?
- **Thin-wrapper retention.** Is removing `_classify_live_unit` and keeping `_check_unit` for one deprecation cycle the right balance, or should both be removed in the same PR?
- **Credential precedence.** Does the env-first + fail-closed semantics match the project's intent for Vast.ai credentials?
- **Biolab-runners precedent.** The biolab-runners v15 PR ([#88](https://github.com/Lambda-Biolab/biolab-runners/pull/88)) and v14 PR ([#82](https://github.com/Lambda-Biolab/biolab-runners/pull/82)) are the precedent. Does this design match v15's discipline (single localised observation sequence, frozen dataclass plans, ClassVar action enum) and v14's discipline (one protocol module, one adapter per provider, callbacks + policy over class hierarchy)?

Discussion thread: the PR comments.
