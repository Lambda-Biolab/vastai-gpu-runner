# Architecture v3 (target)

This doc describes the **target architecture** after the unit-lifecycle decision tree and belt-and-suspenders destroy refactor land. For the current-state architecture (today's code) see
[`architecture.md`](architecture.md). For the next-step target see
[`architecture-v2.md`](architecture-v2.md). For the architectural review that
motivated this doc see the [HTML report][review].

[review]: file:///tmp/architecture-review.Jxytgw/vastai-gpu-runner-review.html

## What changes vs v2

In one paragraph: the unit-lifecycle decision tree (R2 → SSH → worker_dead re-check → action) moves out of `BatchOrchestrator` into a new `unit_lifecycle` deep module exposing one function `decide_next_action(unit, runner, instance, is_done_in_r2) -> UnitAction`, where `UnitAction` is a tagged union of three frozen dataclasses (`Continue | Complete | Preempt`). The four-step belt-and-suspenders destroy protocol moves out of `VastaiRunner._rest_destroy` and the standalone helpers in `providers/vastai.py` into a new `providers/destroy` deep module exposing `belt_and_suspenders`. The destroy protocol has two distinct enums: `VerifyVerdict` (the verifier's observation: `GONE | PRESENT | UNKNOWN`) and `DestroyVerdict` (the protocol's outcome: `DESTROYED | LEAKED | UNKNOWN`). The `VerifyFn` callback returns a `VerifyResult` carrying `verdict` + `status_code` + `error`. The protocol returns a `DestroyResult` carrying `verdict` (protocol outcome) or `refusal` (pre-protocol outcome: `OWNERSHIP | NO_CREDENTIALS | CREDENTIALS_DISABLED`), but never both, plus `verify_error` to preserve the most recent verifier error context. This split makes the CLI fallback in the zombie sweep safe: a refusal for `OWNERSHIP` cannot be bypassed by falling back to the CLI, and a refusal for `CREDENTIALS_DISABLED` (operator set `VASTAI_API_KEY=""`) cannot be bypassed by the CLI re-enabling file credentials. The destroy module owns the loop shape; provider-specific timing policy lives in a `DestroyPolicy` dataclass supplied by the adapter. Three latent bugs are corrected: the orchestrator's `_destroy_via_rest` silently treats HTTP errors as "instance gone," the current `VastaiRunner.destroy_instance` ignores the destroy result and unconditionally returns `True`, and the fail-closed credential semantics were bypassed by the CLI fallback in the zombie sweep.

Diff vs v2:

- **+** `src/vastai_gpu_runner/unit_lifecycle.py` — `decide_next_action`, `Action` enum, `Continue | Complete | Preempt` plan types, `PreemptCause` enum, `ProgressSnapshot` dataclass
- **+** `src/vastai_gpu_runner/providers/destroy.py` — `VerifyVerdict` + `DestroyVerdict` + `DestroyRefusal` (incl. `CREDENTIALS_DISABLED`) enums, `VerifyResult` + `DestroyResult` (with `verify_error` field + refusal invariants) + `DestroyPolicy` dataclasses, `belt_and_suspenders()` function
- **+** `src/vastai_gpu_runner/providers/destroy_adapters/vastai.py` — `CredentialState` + `CredentialResolution` types (with `__post_init__` invariants + blank-file / read-error handling), `read_vastai_api_key` (three-state env-first precedence), `verify_instance_ownership_rest` (REST-based ownership using same headers as destroy, parses nested `instances.image_uuid`), Vast.ai REST callbacks (`stop_fn`, `delete_fn`, `verify_fn`, the latter parses nested `instances.actual_status`), `destroy_vastai_instance`, `VASTAI_POLICY`
- **~** `BatchOrchestrator.__init__` adds `allowed_images: AbstractSet[str] | None = None` (normalised to `frozenset` internally; matches the existing `VastaiRunner.allowed_images: frozenset[str] | None` contract) so the zombie sweep can apply the same image allowlist as the runner
- **~** `BatchOrchestrator._poll_cycle_once` keeps its two-stage shape (classify-all → preemptions → parallel finalise) using the shared `decide_next_action` primitive
- **~** `BatchOrchestrator._check_unit` becomes a thin wrapper (one deprecation cycle only — the API reference documents it as inherited lifecycle surface) that delegates to `decide_next_action`. `_classify_live_unit` is **deleted**, not wrapped.
- **~** `BatchOrchestrator._sweep_zombies` routes through `destroy_vastai_instance` and branches on `DestroyResult.verdict` vs `DestroyResult.refusal`; CLI fallback only fires for `NO_CREDENTIALS`, never for `OWNERSHIP`
- **~** `VastaiRunner.destroy_instance` returns `True` only when `DestroyResult.verdict == DESTROYED`. LEAKED and UNKNOWN return `False` with logged error/warning. `InstanceStatus.DESTROYED` is set only on confirmed destruction.
- **~** `VastaiRunner._rest_destroy` delegates to `destroy_vastai_instance` with the `allowed_images` ownership guard preserved
- **—** `orchestrator.py:_destroy_via_rest` (simplified copy, latent bug) **deleted**
- **—** `orchestrator.py:poll_instance_progress` (dead public API, no callers anywhere) **deleted**
- **—** `orchestrator.py:ensure_detached` (dead public API, no callers anywhere) **deleted**
- **—** `orchestrator.py:load_vastai_api_key` (byte-for-byte duplicate of `_read_vastai_api_key`) **deleted**; credential loading moves to `providers/destroy_adapters/vastai.py` with env-first precedence and fail-closed semantics
- **—** `providers/vastai.py:_rest_stop`, `_rest_delete_with_retries`, `_rest_verify_and_redestroy` **deleted**; absorbed into the Vast.ai adapter
- **—** `BatchOrchestrator._classify_live_unit` **deleted** (not wrapped); its tests move to `test_unit_lifecycle.py`
- **—** No changes to `runner.py` ABC signature, `storage/r2.py`, `worker/base.py`, `estimator/`

## Module taxonomy

The v3 doc adds two new deep modules around the existing ABCs. The `CloudRunner` ABC is unchanged; the new modules sit at the same layer as the existing provider modules.

### Existing ABCs (unchanged)

`CloudRunner` (Lane A provider lifecycle) and `BatchOrchestrator` (multi-unit orchestration) keep their public surface. The refactor narrows the implementation behind those ABCs.

### New: `unit_lifecycle` — owns the per-cycle decision tree

Owns: the decision rules (R2-first → SSH → worker_dead re-check → action mapping), the `Action` enum, the three plan dataclasses, the `PreemptCause` enum, the `ProgressSnapshot` normaliser.

Does **not** own: side effects (destroy, collect, capture_preempt_diagnostics, on_unit_*), lock acquisition, parallel fan-out, consumer hooks. Those stay on the orchestrator.

### New: `providers/destroy` — owns the belt-and-suspenders destroy protocol

Owns: the loop shape (stop → DELETE×retry → verify → re-destroy, with second verification after resurrection cleanup). The generic loop takes a `DestroyPolicy` for timing/retry constants so the loop itself is policy-agnostic. Returns a typed `DestroyResult` with `verdict` (protocol outcome) or `refusal` (pre-protocol outcome), never both.

Does **not** own: REST URLs, API key paths, image-ownership guard, timing constants. The constants live in the policy (supplied by the adapter); the URLs and credentials live in the adapter.

### New: `providers/destroy_adapters/vastai.py` — Vast.ai REST callbacks

Owns: the three Vast.ai REST endpoints (`PUT state=stopped`, `DELETE`, `GET for verify`), the `CredentialState` and `CredentialResolution` types, the `read_vastai_api_key` call (three-state: `AVAILABLE` / `ABSENT` / `EXPLICITLY_DISABLED`), the `verify_instance_ownership_rest` function (REST-based ownership using the same auth headers as destroy), the `allowed_images` ownership guard with `is not None` semantics, the `DestroyPolicy` constants (the Vast.ai-discovered 5s verify delay, 3s retry sleep, 3 max delete attempts), the `destroy_vastai_instance` function that wraps the protocol with pre-protocol refusals (`OWNERSHIP` / `NO_CREDENTIALS` / `CREDENTIALS_DISABLED`).

The RunPod adapter lands separately when `RunPodRunner` ships (roadmap item 2). It will register its own `stop_fn` / `delete_fn` / `verify_fn` callbacks and its own `DestroyPolicy` (different retry timing may be appropriate for RunPod's API).

## Layered design (v3)

```
┌─────────────────────────────────────────────────┐
│  CLI (cli.py)                                   │  User-facing commands
├─────────────────────────────────────────────────┤
│  BatchOrchestrator (batch.py)                   │  Phase loop + side-effect dispatchers
│    └── calls unit_lifecycle.decide_next_action  │  (per-unit decision is delegated)
│    └── _poll_cycle_once: two-stage shape        │
│        (classify-all → preemptions → parallel)  │
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
│    └── VerifyVerdict + DestroyVerdict + Refusal │
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
    destructive action — the conservative result is "still running":
    both ``complete`` and ``worker_dead`` are set to False on
    contradiction.
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
            # Contradictory. Do not act on this — neither flag is
            # trustworthy. Return "still running" with both flags
            # cleared (the conservative read).
            logger.warning("check_progress has complete=True and worker_dead=True")
            complete = False
            worker_dead = False
        if not isinstance(log_tail, str):
            log_tail = ""
        return cls(complete=complete, worker_dead=worker_dead, log_tail=log_tail)


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

Why `ProgressSnapshot` normaliser. The current `_classify_live_unit` does `progress.get("complete")` and `progress.get("worker_dead")` — both default to falsy if missing. A missing `complete` key, a `None` value, or a contradictory `complete=True, worker_dead=True` all silently propagate to the wrong verdict. The normaliser guards each coercion and logs every malformed response. Contradictory flags downgrades to "still running" (both flags clear) so no destructive action fires on the malformed input.

Why `PreemptCause` not a free string. `classify_failure` (the consumer's hook) may eventually branch on the cause. A `str` reason field becomes an unstable policy API the moment two consumers want to dispatch differently. The enum is the stable contract; `detail` carries the human-readable context.

## `providers/destroy` shape

Two distinct enums for two distinct concerns, plus a structured `DestroyResult` whose `verdict` and `refusal` fields are mutually exclusive.

```python
# src/vastai_gpu_runner/providers/destroy.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


class VerifyVerdict(StrEnum):
    """The verifier's observation at one instant in time.

    The verifier does not know about the protocol's outcome — it
    only answers: "is the instance present at this URL right now?"
    The protocol translates this into ``DestroyVerdict``.
    """
    GONE = "gone"       # 404 OR 200 + instances.actual_status == "destroyed"
    PRESENT = "present" # 200 + instances.actual_status != "destroyed" (incl. empty,
                        # missing, non-string, or missing/malformed envelope —
                        # the instance is still listed; we cannot prove it is gone)
    UNKNOWN = "unknown" # any other HTTP status, JSON parse failure, network error


class DestroyVerdict(StrEnum):
    """The protocol's final outcome of a belt-and-suspenders attempt.

    Never collapse this into a bool. The Vast.ai dashboard API
    returns 4xx and 5xx on transient issues; treating those as
    "destroyed" silently leaks running instances.
    """
    DESTROYED = "destroyed"  # confirmed gone on first verification
    LEAKED = "leaked"        # verified PRESENT after resurrection cleanup
    UNKNOWN = "unknown"      # verification was uncertain at any point


class DestroyRefusal(StrEnum):
    """Pre-protocol refusals — the belt-and-suspenders loop did not run.

    The zombie sweep uses these to decide whether to fall back to
    the CLI:

    * ``OWNERSHIP`` forbids the CLI fallback (the image allowlist
      is the destruction-safety boundary regardless of credentials).
    * ``NO_CREDENTIALS`` permits the CLI fallback (the CLI may
      have its own auth independent of the Python loader).
    * ``CREDENTIALS_DISABLED`` forbids the CLI fallback: the
      operator has explicitly set ``VASTAI_API_KEY=""`` to disable
      credentials, and the CLI fallback would silently re-enable
      them via ``~/.cloud-credentials``. The instance is left
      for explicit operator action.
    """
    OWNERSHIP = "ownership"
    NO_CREDENTIALS = "no_credentials"
    CREDENTIALS_DISABLED = "credentials_disabled"


@dataclass(frozen=True)
class VerifyResult:
    """The verifier's observation plus HTTP context.

    ``status_code`` is the HTTP status of the verify call (or
    None if the verifier never made one). ``error`` carries the
    exception message for transport failures or JSON parse
    failures; None on success.
    """
    verdict: VerifyVerdict
    status_code: int | None = None
    error: str | None = None


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

    def __post_init__(self) -> None:
        if self.verify_delay_s < 0:
            raise ValueError(
                f"DestroyPolicy.verify_delay_s must be >= 0, got {self.verify_delay_s}"
            )
        if self.retry_delay_s < 0:
            raise ValueError(
                f"DestroyPolicy.retry_delay_s must be >= 0, got {self.retry_delay_s}"
            )
        if self.max_delete_attempts < 1:
            raise ValueError(
                f"DestroyPolicy.max_delete_attempts must be >= 1, got {self.max_delete_attempts}"
            )


@dataclass(frozen=True)
class DestroyResult:
    """Outcome of a belt-and-suspenders destroy attempt.

    Exactly one of ``verdict`` and ``refusal`` is set. When
    ``refusal`` is set, the protocol never ran and the
    ``attempts`` / ``stop_error`` / ``last_status_code`` /
    ``verify_error`` fields are all None or 0. The invariant
    is enforced in ``__post_init__``.

    * ``verdict`` set → the protocol ran; result is one of
      DESTROYED / LEAKED / UNKNOWN.
    * ``refusal`` set → the protocol did not run; one of
      OWNERSHIP / NO_CREDENTIALS / CREDENTIALS_DISABLED applies.

    The zombie sweep branches on which field is set. CLI fallback
    is only permitted for ``refusal == NO_CREDENTIALS``.
    ``CREDENTIALS_DISABLED`` forbids the CLI fallback because the
    operator has explicitly disabled credentials.
    """
    verdict: DestroyVerdict | None = None
    refusal: DestroyRefusal | None = None
    attempts: int = 0
    stop_error: str | None = None
    last_status_code: int | None = None
    verify_error: str | None = None

    def __post_init__(self) -> None:
        if (self.verdict is None) == (self.refusal is None):
            raise ValueError(
                "DestroyResult: exactly one of verdict or refusal must be set, "
                f"got verdict={self.verdict} refusal={self.refusal}"
            )
        if self.refusal is not None:
            # Protocol never ran — no execution context to carry.
            if self.attempts != 0:
                raise ValueError(
                    f"DestroyResult.refusal={self.refusal} but attempts={self.attempts} "
                    "(protocol never ran; attempts must be 0)"
                )
            if self.stop_error is not None:
                raise ValueError(
                    f"DestroyResult.refusal={self.refusal} but stop_error is set "
                    "(protocol never ran; stop_error must be None)"
                )
            if self.last_status_code is not None:
                raise ValueError(
                    f"DestroyResult.refusal={self.refusal} but last_status_code is set "
                    "(protocol never ran; last_status_code must be None)"
                )
            if self.verify_error is not None:
                raise ValueError(
                    f"DestroyResult.refusal={self.refusal} but verify_error is set "
                    "(protocol never ran; verify_error must be None)"
                )
        else:
            # Protocol ran. ``belt_and_suspenders`` increments
            # ``attempts`` before the first DELETE in phase 2,
            # so every protocol-produced result has
            # ``attempts >= 1``. Independently constructed
            # results with ``verdict`` set and ``attempts=0``
            # are not produced by the protocol and are
            # rejected as a contract violation.
            if self.attempts < 1:
                raise ValueError(
                    f"DestroyResult.verdict={self.verdict} but attempts={self.attempts} "
                    "(protocol always increments attempts to >=1 before the first DELETE)"
                )


class StopFn(Protocol):
    """Best-effort stop (kills stuck Docker pulls).

    May raise on transport failures (network errors, HTTP
    4xx/5xx from the dashboard). The protocol catches the
    exception and records it in
    ``DestroyResult.stop_error``; DELETE still runs.

    The previous wording ("Must not raise") was incorrect: the
    protocol's try/except block in phase 1 explicitly handles
    stop failures by recording them and continuing. HTTP
    failures must be raised so the protocol can record them;
    a silent success on a 401 would mean the stop never
    happened but the protocol would proceed to DELETE without
    a recorded error.
    """
    def __call__(self) -> None: ...


class DeleteFn(Protocol):
    """DELETE the instance. Returns True on 2xx OR 404 (idempotent)."""
    def __call__(self) -> bool: ...


class VerifyFn(Protocol):
    """Verify the instance is gone. Returns a ``VerifyResult`` carrying
    the verifier's observation and HTTP context."""
    def __call__(self) -> VerifyResult: ...


def belt_and_suspenders(
    *,
    stop_fn: StopFn,
    delete_fn: DeleteFn,
    verify_fn: VerifyFn,
    policy: DestroyPolicy,
) -> DestroyResult:
    """Four-step belt-and-suspenders destroy with structured result.

    Returns a ``DestroyResult`` with ``verdict`` set (never
    ``refusal``; refusals are produced by the adapter or caller
    before invoking this function).

    1. ``stop_fn()`` — best-effort. Its error is recorded in
       ``DestroyResult.stop_error`` but does not block the
       subsequent DELETE call. A stop failure alone never causes
       a return; we always try DELETE.
    2. ``delete_fn()`` retried up to ``policy.max_delete_attempts``
       times with ``policy.retry_delay_s`` between attempts.
    3. After ``policy.verify_delay_s``, ``verify_fn()`` checks
       whether the instance is gone.
    4. If the first verification is ``PRESENT``, run stop (best-
       effort, separately from delete) + delete (separately from
       stop) + re-verify. If the second verification is still
       ``PRESENT``, return ``LEAKED``; if ``DESTROYED``, return
       ``DESTROYED``; if ``UNKNOWN``, return ``UNKNOWN``.

    Never raises from the protocol layer.
    """
    attempts = 0
    stop_error: str | None = None
    last_status_code: int | None = None
    verify_error: str | None = None

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
        first_verify = verify_fn()
    except Exception as exc:
        verify_error = f"{type(exc).__name__}: {exc}"
        logger.warning("belt_and_suspenders: verify_fn raised %s — UNKNOWN", verify_error)
        return DestroyResult(
            verdict=DestroyVerdict.UNKNOWN,
            attempts=attempts,
            stop_error=stop_error,
            verify_error=verify_error,
        )
    last_status_code = first_verify.status_code
    verify_error = first_verify.error

    if first_verify.verdict == VerifyVerdict.GONE:
        return DestroyResult(
            verdict=DestroyVerdict.DESTROYED,
            attempts=attempts,
            stop_error=stop_error,
            last_status_code=last_status_code,
            verify_error=verify_error,
        )
    if first_verify.verdict == VerifyVerdict.UNKNOWN:
        return DestroyResult(
            verdict=DestroyVerdict.UNKNOWN,
            attempts=attempts,
            stop_error=stop_error,
            last_status_code=last_status_code,
            verify_error=verify_error,
        )

    # Phase 4: resurrection cleanup. Stop is best-effort; delete
    # is mandatory. They are independent try/excepts so a stop
    # failure does not block the second DELETE.
    if not policy.verify_after_resurrection:
        return DestroyResult(
            verdict=DestroyVerdict.LEAKED,
            attempts=attempts,
            stop_error=stop_error,
            last_status_code=last_status_code,
            verify_error=verify_error,
        )
    try:
        stop_fn()
    except Exception as exc:
        logger.warning(
            "belt_and_suspenders: resurrection stop_fn raised %s — continuing",
            exc,
        )
        if stop_error is None:
            stop_error = f"resurrection: {type(exc).__name__}: {exc}"
    time.sleep(policy.retry_delay_s)
    attempts += 1
    try:
        delete_fn()
    except Exception as exc:
        logger.warning("belt_and_suspenders: resurrection delete_fn raised %s", exc)
        return DestroyResult(
            verdict=DestroyVerdict.UNKNOWN,
            attempts=attempts,
            stop_error=stop_error,
            last_status_code=last_status_code,
            verify_error=verify_error,
        )

    # Phase 5: second verification.
    time.sleep(policy.verify_delay_s)
    try:
        second_verify = verify_fn()
    except Exception as exc:
        verify_error = f"{type(exc).__name__}: {exc}"
        logger.warning("belt_and_suspenders: second verify_fn raised %s — UNKNOWN", verify_error)
        return DestroyResult(
            verdict=DestroyVerdict.UNKNOWN,
            attempts=attempts,
            stop_error=stop_error,
            last_status_code=last_status_code,
            verify_error=verify_error,
        )
    last_status_code = second_verify.status_code
    verify_error = second_verify.error

    if second_verify.verdict == VerifyVerdict.GONE:
        return DestroyResult(
            verdict=DestroyVerdict.DESTROYED,
            attempts=attempts,
            stop_error=stop_error,
            last_status_code=last_status_code,
            verify_error=verify_error,
        )
    if second_verify.verdict == VerifyVerdict.PRESENT:
        return DestroyResult(
            verdict=DestroyVerdict.LEAKED,
            attempts=attempts,
            stop_error=stop_error,
            last_status_code=last_status_code,
            verify_error=verify_error,
        )
    return DestroyResult(
        verdict=DestroyVerdict.UNKNOWN,
        attempts=attempts,
        stop_error=stop_error,
        last_status_code=last_status_code,
        verify_error=verify_error,
    )
```

Why callback-based, not subclassed. v14's `checkpoint.py` is a single concrete module (no adapters) because the checkpoint format is provider-agnostic. The destroy protocol is provider-agnostic in its *loop* but provider-specific in its *endpoints* and *timing policy*. By accepting `stop_fn`, `delete_fn`, `verify_fn` callbacks plus a `DestroyPolicy`, the protocol stays portable across Vast.ai, RunPod, and any future adapter without forcing a class hierarchy.

Why `VerifyVerdict` and `DestroyVerdict` are separate. The verifier answers "is the instance present right now?"; the protocol answers "did we destroy it?" Collapsing them into one enum forces the protocol to run a `verify_fn` interpretation on every observation, which is exactly the bug that made the previous design return `DestroyVerdict.UNKNOWN` (then "destroyed") on HTTP errors. The split lets the producer of each verdict be unambiguous.

Why `DestroyRefusal` is separate from `DestroyVerdict`. The CLI fallback only makes sense for `NO_CREDENTIALS` (no API key to do the work). It is a safety violation for `OWNERSHIP` (the image allowlist rejected the instance). Putting both into one enum would force the zombie sweep to inspect string values to decide fallback, which is the kind of fragile dispatch the typed shape is meant to prevent.

Why typed `DestroyResult`, not bool. The current `_rest_verify_and_redestroy` returns `None` on the happy path and "logs a warning" on the resurrection path. The caller cannot distinguish "destroyed" from "leaked" from "verify was uncertain." A 401 or 500 from the dashboard API is silently treated as "destroyed." The structured `DestroyResult` makes these outcomes distinguishable.

Why `stop_fn` is best-effort. A stop timeout or 401 must not prevent the more important DELETE attempt. The stop call's purpose is to kill stuck Docker pulls; if it fails, DELETE is still the durable teardown. Recording the error in the result preserves observability without breaking the protocol.

Why `DestroyPolicy` is adapter-supplied. The 5s verify delay and 3s retry sleep are Vast.ai production discoveries. RunPod may need different values (faster retry, no resurrection check). Hardcoding them in the generic module would bake Vast.ai assumptions into the protocol. The policy is the place for that knowledge.

Why `DestroyPolicy.__post_init__` validates. A negative sleep value would silently violate the "never raises" claim on the protocol. Zero delete attempts would silently skip deletion. The invariants are enforced at construction so the protocol can stay naive.

## Vast.ai adapter shape

```python
# src/vastai_gpu_runner/providers/destroy_adapters/vastai.py
from __future__ import annotations

import logging
import os
from collections.abc import AbstractSet
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import requests

from vastai_gpu_runner.providers.destroy import (
    DestroyPolicy,
    DestroyRefusal,
    DestroyResult,
    DestroyVerdict,
    VerifyResult,
    VerifyVerdict,
    belt_and_suspenders,
)

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


class CredentialState(StrEnum):
    """Result of resolving Vast.ai credentials.

    The three states are distinct because they imply different
    behaviours downstream:

    * ``AVAILABLE``: a key was found. Use it for REST.
    * ``ABSENT``: no credentials configured anywhere. The CLI
      may have its own auth; the Python loader has none.
    * ``EXPLICITLY_DISABLED``: ``VASTAI_API_KEY`` is set but
      empty. The operator has explicitly chosen not to use any
      credentials. The CLI fallback is forbidden because the
      CLI would silently re-enable file credentials that this
      env var was meant to disable.
    """
    AVAILABLE = "available"
    ABSENT = "absent"
    EXPLICITLY_DISABLED = "explicitly_disabled"


@dataclass(frozen=True)
class CredentialResolution:
    """The outcome of credential lookup.

    Invariants (enforced in ``__post_init__``):

    * When ``state == AVAILABLE``, ``key`` is a non-empty
      string and is already stripped (no leading/trailing
      whitespace).
    * When ``state != AVAILABLE``, ``key`` is exactly ``""``.

    The "already stripped" rule prevents a caller from
    constructing ``CredentialResolution(state=AVAILABLE,
    key=" real-key ")`` and accidentally passing a
    whitespace-padded key to a downstream check that compares
    against a stripped expected value. The loader itself
    always strips before construction, so this rule
    primarily protects direct construction.

    Callers must branch on ``state`` rather than on the key's
    truthiness — ``EXPLICITLY_DISABLED`` and ``ABSENT`` both
    have an empty key but different semantics.
    """
    state: CredentialState
    key: str = ""

    def __post_init__(self) -> None:
        if self.state == CredentialState.AVAILABLE:
            if not self.key:
                raise ValueError(
                    f"CredentialResolution: state=AVAILABLE but key is empty: {self.key!r}"
                )
            if not self.key.strip():
                raise ValueError(
                    f"CredentialResolution: state=AVAILABLE but key is "
                    f"whitespace-only: {self.key!r}"
                )
            if self.key != self.key.strip():
                raise ValueError(
                    f"CredentialResolution: state=AVAILABLE but key has "
                    f"leading/trailing whitespace (must be pre-stripped): {self.key!r}"
                )
        else:
            if self.key:
                raise ValueError(
                    f"CredentialResolution: state={self.state} but key is "
                    f"set: {self.key!r}"
                )


def read_vastai_api_key() -> CredentialResolution:
    """Resolve the Vast.ai API key with env-first precedence.

    Order:
      1. ``VASTAI_API_KEY`` environment variable. A present-but-empty
         value is ``EXPLICITLY_DISABLED``: the operator has turned
         off credentials and the CLI fallback must not re-enable
         them via ``~/.cloud-credentials``.
      2. ``~/.config/vastai/vast_api_key`` file.
      3. ``~/.vast_api_key`` file.

    File contents:
      * Blank or whitespace-only file → ``ABSENT`` with a warning
        (the file exists but is effectively empty; treat as
        no-credentials rather than as an empty key).
      * ``OSError`` on read (permission denied, transient I/O
        error, etc.) → ``ABSENT`` with a warning. File-read
        errors do not escape from this function. The
        ``CredentialResolution`` invariant would otherwise
        reject an empty key with ``state=AVAILABLE`` and
        surface the failure as a hard ``ValueError`` inside
        ``destroy_vastai_instance``.

    Returns:
        ``CredentialResolution`` satisfying the dataclass
        invariants. Callers must branch on ``state`` rather
        than on the key's truthiness.
    """
    env_key = os.environ.get("VASTAI_API_KEY")
    if env_key is not None:
        if env_key.strip() == "":
            logger.warning(
                "VASTAI_API_KEY is set but empty — EXPLICITLY_DISABLED. "
                "Credentials disabled; CLI fallback forbidden."
            )
            return CredentialResolution(state=CredentialState.EXPLICITLY_DISABLED)
        return CredentialResolution(
            state=CredentialState.AVAILABLE, key=env_key.strip(),
        )
    for path in (
        Path("~/.config/vastai/vast_api_key").expanduser(),
        Path("~/.vast_api_key").expanduser(),
    ):
        if not path.exists():
            continue
        try:
            raw = path.read_text()
        except OSError as exc:
            logger.warning(
                "read_vastai_api_key: could not read %s: %s — treating as ABSENT",
                path, exc,
            )
            continue
        stripped = raw.strip()
        if not stripped:
            logger.warning(
                "read_vastai_api_key: %s exists but is blank — treating as ABSENT",
                path,
            )
            continue
        return CredentialResolution(
            state=CredentialState.AVAILABLE, key=stripped,
        )
    return CredentialResolution(state=CredentialState.ABSENT)


def _repository(ref: str) -> str:
    """Strip the tag and digest from a Docker image reference.

    Returns the bare repository name suitable for equality
    comparison against an allowlist entry. A tag is only
    stripped when the final colon appears after the final
    slash (so ``registry:5000/myorg/app:1.0`` parses to
    ``registry:5000/myorg/app``, not ``registry``). A digest
    (``@sha256:...``) is always stripped. Empty or malformed
    references return ``""``.

    Examples::

        >>> _repository("myorg/app:1.0")
        'myorg/app'
        >>> _repository("registry:5000/myorg/app:1.0")
        'registry:5000/myorg/app'
        >>> _repository("myorg/app@sha256:abcdef...")
        'myorg/app'
        >>> _repository("myorg/app")
        'myorg/app'
        >>> _repository("")  # malformed
        ''
    """
    if not ref:
        return ""
    without_digest = ref.split("@", 1)[0]
    last_slash = without_digest.rfind("/")
    last_colon = without_digest.rfind(":")
    if last_colon > last_slash:
        return without_digest[:last_colon]
    return without_digest


def _image_is_allowed(image: str, allowed_images: AbstractSet[str]) -> bool:
    """Check whether the instance's image matches the allowlist.

    Matching rules:

    1. **Exact reference match** — the image string equals an
       allowlist entry character-for-character. Covers
       ``myorg/app:1.0`` against allowlist entry
       ``myorg/app:1.0``.

    2. **Tag-insensitive repository match** — the bare
       repository of the image equals the bare repository of
       an allowlist entry. Covers ``myorg/app:latest`` against
       allowlist entry ``myorg/app:1.0``: both reduce to
       ``myorg/app``. This is the practical policy for
       ``"any tag of myorg/app is OK"`` allowlists.

    Rules that are explicitly NOT supported (would be
    ownership-safety defects):

    * **Substring/prefix match** — ``myorg/app:1.0`` must
      not match ``myorg/app-malicious:latest`` (a
      privilege-escalation risk).
    * **Registry-port-as-prefix** — ``registry:5000/myorg/app:1.0``
      must not produce prefix ``registry`` and match every
      image starting with ``registry``.

    Returns ``False`` for empty or malformed image strings
    (fail-closed).
    """
    if not image:
        return False
    if image in allowed_images:
        return True
    image_repo = _repository(image)
    if not image_repo:
        return False
    return any(
        image_repo == _repository(allowed)
        for allowed in allowed_images
    )


def verify_instance_ownership_rest(
    instance_id: str,
    allowed_images: AbstractSet[str],
    hdrs: dict[str, str],
) -> bool:
    """Ownership check via REST, using the same headers as destroy.

    This is the ownership check used by the REST path. The CLI
    fallback uses the existing CLI-based ``verify_instance_ownership``
    in ``providers/vastai.py`` (separate auth context).

    Returns ``True`` when the instance's image is in the allowlist
    or when the instance does not exist (404). Returns ``False``
    on every other failure (network error, non-200 status, JSON
    parse failure, missing `instances` envelope, missing
    `image_uuid` field, or the image genuinely not in the
    allowlist). **Fail-closed**: any uncertainty in the lookup
    rejects the destroy.

    The Vast.ai ``GET /api/v0/instances/{id}/`` response wraps the
    instance under an ``instances`` object, and the Docker image
    field is ``image_uuid`` (not ``image``). Both are required
    for a correct parse.
    """
    try:
        resp = requests.get(f"{BASE_URL}/{instance_id}/", headers=hdrs, timeout=10)
    except Exception as exc:
        logger.warning("verify_instance_ownership_rest: GET raised %s", exc)
        return False  # fail-closed on network failure
    if resp.status_code == 404:
        return True  # instance is gone; ownership is moot
    if resp.status_code != 200:
        logger.warning(
            "verify_instance_ownership_rest: GET %s returned %d",
            instance_id, resp.status_code,
        )
        return False
    try:
        payload = resp.json()
    except Exception as exc:
        logger.warning("verify_instance_ownership_rest: JSON parse raised %s", exc)
        return False
    if not isinstance(payload, dict):
        logger.warning(
            "verify_instance_ownership_rest: response is not a dict: %r",
            type(payload).__name__,
        )
        return False
    instance = payload.get("instances")
    if not isinstance(instance, dict):
        logger.warning(
            "verify_instance_ownership_rest: missing or non-dict 'instances' "
            "envelope in response",
        )
        return False
    image = instance.get("image_uuid")
    if not isinstance(image, str):
        logger.warning(
            "verify_instance_ownership_rest: image_uuid missing or non-string: %r",
            image,
        )
        return False
    return _image_is_allowed(image, allowed_images)


def vastai_stop(instance_id: str, hdrs: dict[str, str]) -> None:
    """Best-effort stop. Raises on transport failures and HTTP
    4xx/5xx so the protocol records them in
    ``DestroyResult.stop_error``.

    ``requests.put`` does not raise on HTTP error statuses; the
    response must be inspected explicitly. A 401, 403, 429, or
    5xx response is a real stop failure — the stop never
    happened, but a silent return would let DELETE proceed
    without a recorded error.
    """
    resp = requests.put(
        f"{BASE_URL}/{instance_id}/",
        headers={**hdrs, "Content-Type": "application/json"},
        json={"state": "stopped"},
        timeout=10,
    )
    if resp.status_code not in (200, 204):
        raise RuntimeError(
            f"vastai_stop: PUT {instance_id} returned {resp.status_code}: "
            f"{resp.text[:200]}"
        )


def vastai_delete(instance_id: str, hdrs: dict[str, str]) -> bool:
    """DELETE. Returns True on 2xx OR 404 (already gone)."""
    resp = requests.delete(f"{BASE_URL}/{instance_id}/", headers=hdrs, timeout=15)
    return resp.status_code in (200, 204, 404)


def vastai_verify(instance_id: str, hdrs: dict[str, str]) -> VerifyResult:
    """Verify the instance is gone.

    Returns a ``VerifyResult`` with the verifier's observation plus
    HTTP context. The mapping:

    * 404 → ``GONE`` (instance is not in the dashboard).
    * 200 with nested ``instances.actual_status == "destroyed"`` → ``GONE``.
    * 200 with nested ``instances.actual_status`` empty, missing,
      non-string, or any other value → ``PRESENT`` (the instance
      is still listed; we cannot prove it is gone).
    * Any other HTTP status (401, 403, 429, 5xx) → ``UNKNOWN``.
    * Any exception (network, parse) → ``UNKNOWN``.
    * 200 with missing/non-dict ``instances`` envelope → ``PRESENT``
      (the response is well-formed JSON but doesn't carry the
      status field; we cannot prove the instance is gone).

    The empty-status case is intentionally NOT ``GONE``: the
    Vast.ai dashboard API may return empty status for transient
    states (just-booted, just-stopped, error). Only an explicit
    "destroyed" status is affirmative evidence.

    The Vast.ai ``GET /api/v0/instances/{id}/`` response wraps the
    instance under an ``instances`` object, and the status field
    is ``actual_status``. Both the envelope and the nested field
    are required for a correct parse.
    """
    try:
        verify = requests.get(f"{BASE_URL}/{instance_id}/", headers=hdrs, timeout=10)
    except Exception as exc:
        logger.warning("vastai_verify: GET raised %s — UNKNOWN", exc)
        return VerifyResult(verdict=VerifyVerdict.UNKNOWN, error=str(exc))
    if verify.status_code == 404:
        return VerifyResult(verdict=VerifyVerdict.GONE, status_code=404)
    if verify.status_code != 200:
        logger.warning(
            "vastai_verify: GET %s returned %d — UNKNOWN",
            instance_id, verify.status_code,
        )
        return VerifyResult(
            verdict=VerifyVerdict.UNKNOWN,
            status_code=verify.status_code,
            error=f"http {verify.status_code}",
        )
    try:
        payload = verify.json()
    except Exception as exc:
        logger.warning("vastai_verify: JSON parse raised %s — UNKNOWN", exc)
        return VerifyResult(
            verdict=VerifyVerdict.UNKNOWN,
            status_code=200,
            error=str(exc),
        )
    if not isinstance(payload, dict):
        logger.warning(
            "vastai_verify: response is not a dict: %r — UNKNOWN",
            type(payload).__name__,
        )
        return VerifyResult(
            verdict=VerifyVerdict.UNKNOWN,
            status_code=200,
            error="non-dict response",
        )
    instance = payload.get("instances")
    if not isinstance(instance, dict):
        # Well-formed JSON but missing the `instances` envelope.
        # We cannot prove the instance is gone; treat as PRESENT
        # so the protocol does not falsely report DESTROYED.
        logger.warning(
            "vastai_verify: missing or non-dict 'instances' envelope — PRESENT",
        )
        return VerifyResult(verdict=VerifyVerdict.PRESENT, status_code=200)
    raw_status = instance.get("actual_status")
    if not isinstance(raw_status, str):
        # Empty, missing, or non-string: the instance is still listed
        # and we cannot prove it is gone.
        logger.warning(
            "vastai_verify: actual_status missing or non-string: %r — PRESENT",
            raw_status,
        )
        return VerifyResult(verdict=VerifyVerdict.PRESENT, status_code=200)
    if raw_status == "destroyed":
        return VerifyResult(verdict=VerifyVerdict.GONE, status_code=200)
    # Any other non-"destroyed" status (running, stopped, exited, ...):
    # the instance is still listed. We cannot prove it is gone.
    return VerifyResult(verdict=VerifyVerdict.PRESENT, status_code=200)


def destroy_vastai_instance(
    instance_id: str,
    *,
    allowed_images: AbstractSet[str] | None = None,
) -> DestroyResult:
    """Belt-and-suspenders destroy with the ownership guard.

    The ownership guard is applied first via REST (using the same
    headers as the destroy). If the image is not in the allowlist,
    return ``DestroyResult(refusal=OWNERSHIP)``. No API call to
    delete is made.

    Credential resolution distinguishes three states:

    * ``AVAILABLE``: a key was found. Build the REST headers and
      run the protocol.
    * ``ABSENT``: no credentials configured. Return ``refusal=
      NO_CREDENTIALS``. The zombie sweep may try the CLI fallback
      (the CLI may have its own auth).
    * ``EXPLICITLY_DISABLED``: the operator has set
      ``VASTAI_API_KEY=""`` to disable credentials. Return
      ``refusal=CREDENTIALS_DISABLED``. The CLI fallback is
      forbidden because the CLI would silently re-enable file
      credentials that this env var was meant to disable.

    The runner's ``destroy_instance`` path treats every refusal
    as a hard failure (returns False), including
    ``CREDENTIALS_DISABLED`` and ``NO_CREDENTIALS``. The CLI
    fallback is *only* available in the zombie sweep.
    """
    resolution = read_vastai_api_key()
    if resolution.state == CredentialState.EXPLICITLY_DISABLED:
        logger.error(
            "destroy_vastai_instance: %s — credentials disabled "
            "via VASTAI_API_KEY=''. Refusing.",
            instance_id,
        )
        return DestroyResult(refusal=DestroyRefusal.CREDENTIALS_DISABLED)
    if resolution.state == CredentialState.ABSENT:
        logger.error(
            "destroy_vastai_instance: %s — no credentials configured",
            instance_id,
        )
        return DestroyResult(refusal=DestroyRefusal.NO_CREDENTIALS)
    hdrs = {"Authorization": f"Bearer {resolution.key}"}
    if allowed_images is not None and not verify_instance_ownership_rest(
        instance_id, allowed_images, hdrs,
    ):
        logger.error(
            "REFUSED to destroy instance %s — ownership check failed.",
            instance_id,
        )
        return DestroyResult(refusal=DestroyRefusal.OWNERSHIP)
    return belt_and_suspenders(
        stop_fn=lambda: vastai_stop(instance_id, hdrs),
        delete_fn=lambda: vastai_delete(instance_id, hdrs),
        verify_fn=lambda: vastai_verify(instance_id, hdrs),
        policy=VASTAI_POLICY,
    )
```

The `is not None` guard (not `if allowed_images and ...`) is the safety-critical fix. An empty `set()` passed as `allowed_images` rejects every image, which is the correct conservative behaviour.

The three-state credential model is the second safety-critical fix. The previous design's `read_vastai_api_key()` returned `""` for both ABSENT and EXPLICITLY_DISABLED; the adapter flattened both to `NO_CREDENTIALS`; the CLI fallback then used file credentials that the explicit empty env var was meant to disable. The new design makes the operator's intent explicit: setting `VASTAI_API_KEY=""` is a hard stop, not a "try the CLI fallback."

The ownership check now uses REST (when REST is available) so that the same auth identity applies to the ownership check and the destroy. The CLI-based `verify_instance_ownership` in `providers/vastai.py` is retained for the CLI fallback path (which uses the CLI's auth, not the REST key).

## `VastaiRunner.destroy_instance` integration

The runner's `destroy_instance` is the unit-tracked path. It must use the typed `DestroyResult` directly — not just call the protocol and discard the output.

```python
# In providers/vastai.py — VastaiRunner.destroy_instance, after the refactor:

def destroy_instance(self, instance: CloudInstance) -> bool:
    """Destroy a Vast.ai instance (with ownership safety guard).

    Returns True only when the belt-and-suspenders protocol confirmed
    destruction. Returns False on refusal (ownership, missing
    credentials, credentials disabled), leakage (resurrected after
    cleanup), or unknown verification — each with an explanatory
    log message. ``InstanceStatus.DESTROYED`` is set only on
    confirmed destruction.

    The adapter is called once; the result drives the boolean
    return. The runner does not pre-resolve the API key (the
    adapter owns credential resolution and the refusal contract).
    """
    result = destroy_vastai_instance(
        instance.instance_id, allowed_images=self.allowed_images,
    )
    if result.refusal == DestroyRefusal.OWNERSHIP:
        logger.error(
            "VastaiRunner.destroy_instance: %s refused by ownership guard",
            instance.instance_id,
        )
        return False
    if result.refusal == DestroyRefusal.NO_CREDENTIALS:
        logger.error(
            "VastaiRunner.destroy_instance: %s — no API key",
            instance.instance_id,
        )
        return False
    if result.refusal == DestroyRefusal.CREDENTIALS_DISABLED:
        logger.error(
            "VastaiRunner.destroy_instance: %s — credentials disabled "
            "via VASTAI_API_KEY=''",
            instance.instance_id,
        )
        return False
    if result.verdict == DestroyVerdict.DESTROYED:
        instance.status = InstanceStatus.DESTROYED
        logger.info("Destroyed instance %s", instance.instance_id)
        return True
    if result.verdict == DestroyVerdict.LEAKED:
        logger.error(
            "VastaiRunner.destroy_instance: %s LEAKED after resurrection "
            "cleanup — operator intervention required "
            "(attempts=%d, last_status=%s, last_error=%s, stop_error=%s)",
            instance.instance_id,
            result.attempts, result.last_status_code,
            result.verify_error, result.stop_error,
        )
        return False
    logger.warning(
        "VastaiRunner.destroy_instance: %s destroy result UNKNOWN "
        "(attempts=%d, last_status=%s, last_error=%s, stop_error=%s)",
        instance.instance_id,
        result.attempts, result.last_status_code,
        result.verify_error, result.stop_error,
    )
    return False
```

The current code's "CLI destroy + REST + always return True" pattern is replaced. The CLI fallback is *only* available in the zombie sweep (the orphan path), and only for `NO_CREDENTIALS`. The runner's `destroy_instance` is the unit-tracked path and treats every refusal as a hard failure. Adapter is called once; the result drives the boolean return.

## Zombie sweep integration

The zombie sweep lives in `orchestrator.py:sweep_zombie_instances` as today. The function gains an `allowed_images: AbstractSet[str] | None` keyword argument and routes through `destroy_vastai_instance`. It branches on the `DestroyResult.verdict` vs `DestroyResult.refusal` distinction.

* `DestroyResult.verdict == DESTROYED` → `killed` increments.
* `DestroyResult.verdict == LEAKED` / `UNKNOWN` → logged with full error context.
* `DestroyResult.refusal == OWNERSHIP` → skipped (no CLI fallback, ever).
* `DestroyResult.refusal == CREDENTIALS_DISABLED` → skipped (CLI fallback forbidden — it would silently re-enable file credentials).
* `DestroyResult.refusal == NO_CREDENTIALS` → CLI fallback is opportunistic; tracked in `cli_attempted` (separate from `killed`).

`BatchOrchestrator._sweep_zombies` is a thin delegate that calls `sweep_zombie_instances` with `allowed_images=self._allowed_images`. This keeps the public zombie-cleanup surface in `orchestrator.py` (where the function lives today) and the orchestrator's role minimal.

```python
# In orchestrator.py — sweep_zombie_instances, after the refactor:
#
# Existing import: ``from vastai_gpu_runner.providers.vastai import vastai_cmd``.
# New import:        ``from vastai_gpu_runner.providers.vastai import
#                     verify_instance_ownership`` (the existing CLI-backed
#                     ownership check, used in the CLI fallback path).
# New import:        ``from collections.abc import AbstractSet``.
# New import:        ``from vastai_gpu_runner.providers.destroy import
#                     DestroyRefusal, DestroyVerdict``.
# New import:        ``from vastai_gpu_runner.providers.destroy_adapters.vastai
#                     import destroy_vastai_instance``.

def sweep_zombie_instances(
    live_runners: dict[int, tuple[CloudRunner, CloudInstance]],
    *,
    label_prefix: str,
    allowed_images: AbstractSet[str] | None = None,
    r2_sink: R2Sink | None = None,
    r2_batch_id: str = "",
) -> int:
    """Destroy Vast.ai instances not tracked by live_runners.

    Each candidate instance is destroyed via the belt-and-suspenders
    adapter. The image allowlist (``allowed_images``) is the
    destruction-safety boundary; the batch-label prefix is the
    orphan-detection boundary. The two boundaries are independent:
    the label prefix determines *which* instances are candidates;
    the image allowlist determines *whether* each candidate may
    be destroyed.

    Short-circuit: if the operator has set ``VASTAI_API_KEY=""``
    (``CredentialState.EXPLICITLY_DISABLED``), the function
    returns 0 immediately with a logged warning. The sweep
    would otherwise enumerate instances through the CLI
    (``vastai_cmd(["show", "instances", "--raw"])``) which uses
    file credentials — the very credentials the operator
    meant to disable. Returning early is the consistent
    behaviour: the operator's intent is "no credentials
    anywhere, including enumeration." Orphans must be cleaned
    up by the operator manually after re-enabling
    credentials.

    The CLI fallback is opportunistic: it runs only for
    ``DestroyResult.refusal == NO_CREDENTIALS`` and is recorded
    in a separate counter (``cli_attempted``). The CLI fallback
    MUST run ``verify_instance_ownership`` (CLI-backed) before
    invoking the CLI destroy — without this check, the CLI
    path bypasses the allowlist on the REST-disabled code path.
    Ownership and
    credentials-disabled refusals are never bypassed by the CLI.
    The ``killed`` counter is incremented only on confirmed
    destroy (``verdict == DESTROYED``).
    """
    # Short-circuit on operator-disabled credentials. The
    # enumeration step would otherwise use file credentials
    # via the CLI, contradicting the operator's intent.
    resolution = read_vastai_api_key()
    if resolution.state == CredentialState.EXPLICITLY_DISABLED:
        logger.warning(
            "Zombie sweep: VASTAI_API_KEY='' — EXPLICITLY_DISABLED. "
            "Skipping enumeration and orphan cleanup. Operator "
            "must re-enable credentials or clean up manually."
        )
        return 0
    # ... existing enumeration via vastai_cmd (unchanged) ...
    killed = 0
    cli_attempted = 0
    for iid in zombie_candidates:
        result = destroy_vastai_instance(
            iid, allowed_images=allowed_images,
        )
        if result.verdict == DestroyVerdict.DESTROYED:
            killed += 1
        elif result.verdict == DestroyVerdict.LEAKED:
            logger.error(
                "Zombie sweep: instance %s LEAKED after resurrection cleanup "
                "— operator intervention required (attempts=%d, "
                "last_status=%s, last_error=%s, stop_error=%s)",
                iid, result.attempts, result.last_status_code,
                result.verify_error, result.stop_error,
            )
        elif result.verdict == DestroyVerdict.UNKNOWN:
            logger.warning(
                "Zombie sweep: instance %s destroy result UNKNOWN "
                "(attempts=%d, last_status=%s, last_error=%s, stop_error=%s)",
                iid, result.attempts, result.last_status_code,
                result.verify_error, result.stop_error,
            )
        elif result.refusal == DestroyRefusal.OWNERSHIP:
            logger.info(
                "Zombie sweep: skipped %s (ownership refusal). "
                "The label-prefix selection and the image allowlist "
                "disagree — reconcile before next sweep.",
                iid,
            )
        elif result.refusal == DestroyRefusal.CREDENTIALS_DISABLED:
            # CLI fallback is forbidden. The operator explicitly
            # disabled credentials via VASTAI_API_KEY=''; the CLI
            # would silently re-enable file credentials that the
            # env var was meant to disable. Operator must act.
            logger.warning(
                "Zombie sweep: skipped %s (credentials disabled via "
                "VASTAI_API_KEY=''). CLI fallback forbidden — "
                "operator action required.",
                iid,
            )
        elif result.refusal == DestroyRefusal.NO_CREDENTIALS:
            # CLI fallback permitted — the CLI may have its own auth
            # independent of the Python loader. The ownership guard
            # MUST run before the CLI destroy. The CLI-based
            # ``verify_instance_ownership`` queries through the CLI's
            # own authentication context and checks the instance's
            # ``image_uuid`` against the allowlist. Without this,
            # the CLI fallback would bypass the allowlist on the
            # REST-disabled path.
            if allowed_images is not None and not verify_instance_ownership(
                iid, allowed_images=frozenset(allowed_images),
            ):
                logger.error(
                    "Zombie sweep: CLI ownership check refused destroy of %s "
                    "(allowed_images has %d entries). Skipping.",
                    iid, len(allowed_images),
                )
                continue
            # Ownership passes (or no allowlist was configured).
            # The killed counter does not increment because we
            # cannot verify the result through the CLI without
            # significant work.
            try:
                vastai_cmd(["destroy", "instance", iid], timeout=15)
                cli_attempted += 1
                logger.warning(
                    "Zombie sweep: CLI fallback for %s — verify manually. "
                    "CLI success does not count as confirmed destroy.",
                    iid,
                )
            except Exception as exc:
                logger.warning(
                    "Zombie sweep: CLI fallback failed for %s: %s",
                    iid, exc,
                )
    if killed:
        logger.info("Zombie sweep: confirmed-destroyed %d instance(s)", killed)
    if cli_attempted:
        logger.warning(
            "Zombie sweep: %d CLI fallback(s) attempted — verify manually",
            cli_attempted,
        )
    return killed
```

```python
# In BatchOrchestrator, after the refactor:

def __init__(
    self,
    *,
    runner_factory: RunnerFactory,
    label_prefix: str,
    allowed_images: AbstractSet[str] | None = None,
    workspace_dir: str = "/workspace",
    r2_sink: R2Sink | None = None,
    r2_batch_id: str = "",
    budget_usd: float = 0.0,
    max_retries: int = 2,
    max_parallel_deploys: int = 16,
    max_parallel_collects: int = 1,
    poll_interval_seconds: int = 30,
    zombie_sweep_every_n_cycles: int = 5,
    poll_timeout_seconds: float = 0.0,
) -> None:
    # ... existing assignments ...
    self._allowed_images = (
        frozenset(allowed_images) if allowed_images is not None else None
    )
    # ... rest unchanged ...

def _sweep_zombies(self) -> int:
    """Thin delegate to ``sweep_zombie_instances``.

    Carries the orchestrator's ``allowed_images`` config into the
    shared sweep function. The actual cleanup logic lives in
    ``orchestrator.py:sweep_zombie_instances``; the orchestrator
    just adapts the signature.
    """
    with self._state_lock:
        live_map: dict[int, tuple[CloudRunner, CloudInstance]] = {
            i: (entry[0], entry[1]) for i, entry in enumerate(self._live_runners.values())
        }
    try:
        return sweep_zombie_instances(
            live_map,
            label_prefix=self._label_prefix,
            allowed_images=self._allowed_images,
            r2_sink=self._r2_sink,
            r2_batch_id=self._r2_batch_id,
        )
    except Exception as exc:
        logger.warning("Zombie sweep failed: %s", exc)
        return 0
```

The `allowed_images` field is intentionally distinct from the label-prefix boundary. The label prefix identifies orphans (instances we created but lost track of); the image allowlist is the safety guard against cross-project deletion. Both must agree for the CLI fallback to be safe — if they disagree, the operator has misconfigured something and the instance is logged but not destroyed.

The fresh `allowed_images` is normalised to `frozenset` at construction so downstream calls (which all use `AbstractSet[str]`) work identically whether the caller passed a `set`, `frozenset`, or any other `AbstractSet` implementation. This matches the existing `VastaiRunner.allowed_images` contract, which is `frozenset[str] | None`.

**Known limitation (tracked in [#19](https://github.com/Lambda-Biolab/vastai-gpu-runner/issues/19)).** Adding `allowed_images` directly to the provider-neutral `BatchOrchestrator` weakens its "no provider coupling" principle and creates a second independently configured copy of the runner's ownership policy. The longer-term shape is a `destroy_zombie` callback or a `ProviderCleanupPolicy` dataclass. See issue #19 for the proposal, the open questions, and the acceptance criteria. Issue #19 also notes that Option A (per-runner `destroy_zombie` method) needs more design work because a zombie candidate is by definition not in the live-runner map; Option B (`ProviderCleanupPolicy` dataclass) is the more directly implementable proposal. The Option B polish (single canonical provider config shared between runner factory and cleanup policy, rather than re-constructing runners on every destroy) is also tracked in #19.

## Two-stage poll loop

The shared primitive is `decide_next_action` (pure). The orchestrator's poll loop keeps the existing two-stage structure: classify all units, handle preemptions, then parallel finalise terminals. There is no single synchronous dispatcher that loses the parallelisation.

```python
# In BatchOrchestrator, after the refactor:

def _poll_cycle_once(self) -> bool:
    """One sweep over live units. Returns True if any unit made progress.

    Stage 1 (classify all): invoke ``decide_next_action`` for each
    live unit. No side effects. The actions are collected for stage 2+
    3 to act on.

    Stage 2 (preemptions serial): for each ``Preempt`` action, capture
    diagnostics + destroy + instance-loss bookkeeping. Serial because
    these are cheap but touch shared state.

    Stage 3 (finalise terminals parallel): for each ``Complete``
    action, gather into the batch and call ``_finalise_terminal_units``
    which fans out via ``max_parallel_collects``.
    """
    from vastai_gpu_runner.unit_lifecycle import (
        decide_next_action,
        Continue, Complete, Preempt,
    )

    # Stage 1: classify all units (pure, no side effects).
    classified: list[tuple[
        str, "UnitAction", CloudRunner, CloudInstance, UnitT,
    ]] = []
    for unit_key in list(self._live_runners.keys()):
        entry = self._live_runners.get(unit_key)
        if entry is None:
            continue
        runner, instance, unit = entry
        action = decide_next_action(
            unit, runner, instance, self.unit_is_done_in_r2,
        )
        classified.append((unit_key, action, runner, instance, unit))

    # Stage 2: handle preemptions (serial).
    preempted_count = 0
    for unit_key, action, runner, instance, unit in classified:
        if isinstance(action, Preempt):
            self._handle_preempt(
                runner, instance, unit, unit_key, action.cause, action.detail,
            )
            preempted_count += 1

    # Stage 3: collect terminals (parallel via _finalise_terminal_units).
    terminals = [
        (unit_key, runner, instance, unit)
        for unit_key, action, runner, instance, unit in classified
        if isinstance(action, Complete)
    ]
    if terminals:
        self._finalise_terminal_units(terminals)

    return bool(terminals or preempted_count)


def _handle_preempt(
    self,
    runner: CloudRunner,
    instance: CloudInstance,
    unit: UnitT,
    unit_key: str,
    cause: PreemptCause,
    detail: str | None,
) -> None:
    """Preempt a single unit: capture diagnostics, destroy, mark loss."""
    with contextlib.suppress(Exception):
        self.capture_preempt_diagnostics(runner, instance, unit)
    with contextlib.suppress(Exception):
        runner.destroy_instance(instance)
    reason = f"{cause.value}" + (f": {detail}" if detail else "")
    with self._state_lock:
        self._handle_instance_loss(unit, unit_key, reason)
```

The `_check_unit` thin wrapper (kept for one deprecation cycle) uses the same primitive directly:

```python
def _check_unit(
    self,
    runner: CloudRunner,
    instance: CloudInstance,
    unit: UnitT,
) -> Literal["completed", "running", "preempted", "failed"]:
    """Deprecated. Will be removed in the next minor release.

    Kept for one deprecation cycle because the API reference
    documents it as inherited lifecycle surface. Uses the same
    ``decide_next_action`` primitive as the main poll loop.
    """
    warnings.warn(
        "_check_unit is deprecated; use unit_lifecycle.decide_next_action "
        "directly. Will be removed in the next minor release.",
        DeprecationWarning,
        stacklevel=2,
    )
    unit_key = self.unit_key(unit)
    from vastai_gpu_runner.unit_lifecycle import (
        decide_next_action,
        Continue, Complete, Preempt,
    )
    action = decide_next_action(
        unit, runner, instance, self.unit_is_done_in_r2,
    )
    if isinstance(action, Continue):
        return "running"
    if isinstance(action, Preempt):
        self._handle_preempt(
            runner, instance, unit, unit_key, action.cause, action.detail,
        )
        return "preempted"
    return self._finalise_completed(runner, instance, unit, unit_key)
```

`_classify_live_unit` is **deleted** (not wrapped). Its tests move to `test_unit_lifecycle.py` against the new `decide_next_action` interface.

## ABC changes required

**None.** `CloudRunner` keeps its public interface. The new `unit_lifecycle` module does not subclass the runner; it takes a `runner` parameter via protocol. The new `providers/destroy` module takes callbacks plus a `DestroyPolicy`; `VastaiRunner` still calls `destroy_vastai_instance` from its own `destroy_instance` method.

The new `BatchOrchestrator` constructor parameter `allowed_images: AbstractSet[str] | None = None` is additive (default `None` preserves existing behaviour for callers that did not set it). At construction the value is normalised to `frozenset` for stable hashing and immutability, matching `VastaiRunner.allowed_images`.

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
| `ProgressSnapshot` | Frozen dataclass normalising the runner's `check_progress` response before classification. Missing or malformed fields default to the conservative read. Contradictory fields clear both flags. |
| `decide_next_action` | The single public function in `unit_lifecycle.py`. Performs a localised observation sequence (R2 → SSH → R2-on-worker_dead), returns the action. |
| `VerifyVerdict` | StrEnum; the verifier's observation at one instant. `GONE | PRESENT | UNKNOWN`. |
| `VerifyResult` | Frozen dataclass; the verifier's output. Carries `verdict: VerifyVerdict`, `status_code: int | None`, `error: str | None`. |
| `DestroyVerdict` | StrEnum; the protocol's outcome. `DESTROYED | LEAKED | UNKNOWN`. |
| `DestroyRefusal` | StrEnum; pre-protocol refusals. `OWNERSHIP | NO_CREDENTIALS | CREDENTIALS_DISABLED`. |
| `DestroyResult` | Frozen dataclass; the protocol's output. Exactly one of `verdict: DestroyVerdict | None` and `refusal: DestroyRefusal | None` is set. Carries `attempts`, `stop_error`, `last_status_code`, `verify_error`. When `refusal` is set, `attempts == 0` and the other fields are `None`. |
| `CredentialState` | StrEnum; the result of Vast.ai credential lookup. `AVAILABLE | ABSENT | EXPLICITLY_DISABLED`. Used to distinguish a missing key from an operator-disabled key. |
| `CredentialResolution` | Frozen dataclass; carries `state: CredentialState` and `key: str` (set only when `state == AVAILABLE`). |
| `verify_error` | Field on `DestroyResult`; the most recent `verify_fn` error message (or `None`). Distinct from `stop_error` (best-effort stop failures). |
| `DestroyPolicy` | Frozen dataclass; provider-specific retry/timing policy with `__post_init__` invariants. |
| `belt_and_suspenders` | The single public function in `providers/destroy.py`. Four-step destroy loop with second verification after resurrection cleanup. Returns `DestroyResult` with `verdict` set. |
| `StopFn` / `DeleteFn` / `VerifyFn` | The callback protocols for the destroy loop. `VerifyFn` returns `VerifyResult` (not enum). |
| `destroy_vastai_instance` | The Vast.ai adapter. Wraps the protocol with pre-protocol refusals (ownership, no credentials). |
| `read_vastai_api_key` | The Vast.ai credential loader with env-first precedence and fail-closed semantics. |
| `VASTAI_POLICY` | The `DestroyPolicy` constant for Vast.ai (5s verify delay, 3s retry sleep, 3 max delete attempts). |
| `allowed_images` | New `BatchOrchestrator` constructor parameter. The image allowlist for the destroy-safety boundary; the label prefix remains the orphan-detection boundary. |

## Resolved design decisions

These were open questions in the v3 draft; they are resolved in this revision.

### Credential precedence (Vast.ai)

`read_vastai_api_key()` returns a `CredentialResolution` with three distinct states:

* `CredentialState.AVAILABLE`: a key was found (env-first, then `~/.config/vastai/vast_api_key`, then `~/.vast_api_key`).
* `CredentialState.ABSENT`: no credentials configured anywhere.
* `CredentialState.EXPLICITLY_DISABLED`: `VASTAI_API_KEY` is set but empty. The operator has explicitly turned off credentials.

The adapter translates the state into a `DestroyResult.refusal`:

* `AVAILABLE` → run the belt-and-suspenders protocol.
* `ABSENT` → `DestroyResult(refusal=NO_CREDENTIALS)`. CLI fallback is permitted (the CLI may have its own auth).
* `EXPLICITLY_DISABLED` → `DestroyResult(refusal=CREDENTIALS_DISABLED)`. CLI fallback is forbidden — the CLI would silently re-enable file credentials that the env var was meant to disable.

The previous design's `read_vastai_api_key()` returned `""` for both `ABSENT` and `EXPLICITLY_DISABLED`; the adapter flattened both to `NO_CREDENTIALS`; the CLI fallback then used file credentials that the explicit empty env var was meant to disable. The new design closes that loop: the operator's intent is now an explicit state the CLI fallback can read.

Multi-provider credential unification is deferred until `RunPodRunner` lands.

### Destroy callback return types

`verify_fn` returns `VerifyResult` (carrying `verdict: VerifyVerdict`, `status_code: int | None`, `error: str | None`). The protocol interprets:
- `verdict == GONE` → first verified gone, return `DestroyResult(verdict=DESTROYED)`.
- `verdict == PRESENT` → resurrection detected, run stop + delete + re-verify.
- `verdict == UNKNOWN` → cannot determine. Return `DestroyResult(verdict=UNKNOWN)` with `last_status_code` and `error` preserved.

`delete_fn` returns `bool` (success or no — idempotent on 404). `stop_fn` returns `None` and may raise; the protocol records the exception in `stop_error` and continues to DELETE. (The previous wording "must not raise" was incorrect: the protocol's `try/except` in phase 1 explicitly handles stop failures by recording them and proceeding. A silent return on a 401/429/500 would let DELETE proceed without a recorded error. The `vastai_stop` adapter now inspects the response status and raises `RuntimeError` on non-2xx so the protocol sees the failure.)

### `Preempt` reason

`Preempt` carries `cause: PreemptCause` (enum) and optional `detail: str | None`. No free-form `reason: str` field. The enum is the stable contract; `detail` is for humans only.

### Context module location

No `CONTEXT.md` is created. The glossary stays in `architecture-v3.md`. Future docs that need the terms link here. If the project later wants a single glossary file, this section is the seed.

### Ownership boundary

The image allowlist (`allowed_images`) is the destruction-safety boundary. The batch-label prefix is the orphan-detection boundary. They are independent and complementary:

- The label prefix identifies *which* instances are orphan candidates (instances we created but lost track of).
- The image allowlist determines *whether* each candidate may be destroyed.

The zombie sweep passes both to the adapter. The adapter's `OWNERSHIP` refusal fires when the image allowlist rejects the instance. The CLI fallback is only permitted for `NO_CREDENTIALS`; ownership refusals are never bypassed.

The `allowed_images` parameter is added to `BatchOrchestrator.__init__` with a default of `None` to preserve existing behaviour for callers that did not set it.

### CLI fallback honesty

The CLI `vastai destroy instance <id>` command returns 0 even if the instance becomes a zombie. The CLI fallback is therefore opportunistic: it runs only for `DestroyResult.refusal == NO_CREDENTIALS`, and its attempts are recorded in a separate counter (`cli_attempted`), not the `killed` counter. The `killed` counter increments only on `DestroyResult.verdict == DESTROYED` (confirmed by the belt-and-suspenders protocol).

## Critical test cases

The following test cases must be enumerated in the implementation PR. The design doc lists them so the implementation reviewer can verify the test surface is complete.

### `unit_lifecycle` tests

- **Malformed progress response.** `check_progress` returns `{}`, `{"running": False}`, `{"complete": "true"}` (string, not bool), `{"worker_dead": 1}` (int, not bool), or a non-dict (None, list). Each coerces to `Continue` (conservative).
- **Contradictory progress response.** `{"complete": True, "worker_dead": True}`. Both flags clear; returns `Continue`. The test asserts `not complete and not worker_dead` on the resulting snapshot.
- **Initial R2 raises.** `is_done_in_r2` raises on the first call. Logs the exception and continues to SSH. SSH verdict is honoured.
- **Worker_dead R2 re-check raises.** `is_done_in_r2` raises on the second call (after `worker_dead`). Returns `Continue` (not `Preempt`) — destroying on unknown final-upload status risks losing recoverable results.
- **`check_progress` raises.** Returns `Continue` (transient SSH flakiness).
- **Happy path: R2 done.** `is_done_in_r2` returns True on the first call. Returns `Complete`. SSH not consulted.
- **Happy path: SSH complete.** R2 not done, `check_progress` returns `{"complete": True}`. Returns `Complete`.
- **Happy path: worker dead, R2 re-check done.** Returns `Complete`.
- **Happy path: worker dead, R2 re-check not done.** Returns `Preempt(cause=WORKER_DIED, detail=log_tail)`.

### `providers/destroy` tests

- **Happy delete.** `stop_fn` succeeds, `delete_fn` returns True on first attempt, `verify_fn` returns `VerifyResult(verdict=GONE)`. Result: `DestroyResult(verdict=DESTROYED, attempts=1)`.
- **Delete retry.** `delete_fn` returns False twice, True on the third. `verify_fn` returns `VerifyResult(verdict=GONE)`. Result: `DestroyResult(verdict=DESTROYED, attempts=3)`.
- **Resurrection cleanup succeeds.** First verify returns `VerifyResult(verdict=PRESENT)`. Run stop + delete + re-verify. Second verify returns `VerifyResult(verdict=GONE)`. Result: `DestroyResult(verdict=DESTROYED, attempts=2)`.
- **Resurrection cleanup fails.** First verify returns `VERIFY(PRESENT)`. Re-verify still returns `VERIFY(PRESENT)`. Result: `DestroyResult(verdict=LEAKED, attempts=2)`.
- **Unknown verification.** `verify_fn` returns `VerifyResult(verdict=UNKNOWN)` on first call. Result: `DestroyResult(verdict=UNKNOWN)`. No resurrection cleanup.
- **`verify_error` preserved from verify_fn.** `verify_fn` returns `VerifyResult(verdict=UNKNOWN, error="http 500")`. Result: `DestroyResult(verdict=UNKNOWN, verify_error="http 500", last_status_code=500)`. The error context is propagated to the caller.
- **Verify_fn raises.** `verify_fn()` raises an exception. Result: `DestroyResult(verdict=UNKNOWN, verify_error="<ExceptionClass>: <message>")`. The exception is caught and converted to a structured error.
- **Stop failure.** `stop_fn` raises. The error is recorded in `stop_error`; the DELETE still runs. Result: `DestroyResult(verdict=DESTROYED, stop_error="...")`.
- **Resurrection stop failure.** First verify returns `PRESENT`. The second stop raises. The second delete still runs. Result: `DestroyResult(verdict=DESTROYED, attempts=2, stop_error="resurrection: ...")` (or `LEAKED` if the second verify still returns `PRESENT`).
- **Unknown after resurrection.** First verify returns `PRESENT`. Resurrection cleanup completes. Second verify returns `UNKNOWN`. Result: `DestroyResult(verdict=UNKNOWN)`.
- **DestroyPolicy invariants.** `DestroyPolicy(verify_delay_s=-1)` raises `ValueError`. `DestroyPolicy(retry_delay_s=-1)` raises `ValueError`. `DestroyPolicy(max_delete_attempts=0)` raises `ValueError`.
- **DestroyResult invariants.**
  - `DestroyResult()` (neither field) raises `ValueError`.
  - `DestroyResult(verdict=DESTROYED, refusal=NO_CREDENTIALS)` (both fields) raises `ValueError`.
  - `DestroyResult(verdict=DESTROYED, attempts=0)` raises `ValueError` (verdict set, `attempts >= 1` is required — the protocol always increments before the first DELETE).
  - `DestroyResult(verdict=LEAKED, attempts=0)` raises `ValueError` (same rule).
  - `DestroyResult(verdict=UNKNOWN, attempts=0)` raises `ValueError` (same rule).
  - `DestroyResult(verdict=DESTROYED, attempts=1)` is valid.
  - `DestroyResult(refusal=NO_CREDENTIALS, attempts=1)` raises `ValueError` (refusal set, attempts must be 0).
  - `DestroyResult(refusal=NO_CREDENTIALS, last_status_code=500)` raises `ValueError` (refusal set, last_status_code must be None).
  - `DestroyResult(refusal=NO_CREDENTIALS, verify_error="...")` raises `ValueError` (refusal set, verify_error must be None).
  - `DestroyResult(refusal=NO_CREDENTIALS, stop_error="...")` raises `ValueError` (refusal set, stop_error must be None).

### Adapter tests

- **Empty allowlist.** `allowed_images=set()` (empty, not None). Instance image is any value. Returns `DestroyResult(refusal=OWNERSHIP)` with refusal logged. **This is the safety-critical test for the empty-allowlist fix.**
- **None allowlist.** `allowed_images=None`. No guard applied. Proceeds to API call.
- **`frozenset` allowlist.** `allowed_images=frozenset({"image-1"})`. Works identically to a `set`. The orchestrator's `BatchOrchestrator` normalises to `frozenset`; the adapter's public type is `AbstractSet[str]`.
- **Matching allowlist.** Instance image is in the set. Proceeds.
- **Non-matching allowlist.** Instance image is not in the set. Returns `DestroyResult(refusal=OWNERSHIP)` with refusal logged.
- **REST ownership check uses same headers as destroy.** When a REST key is available, `verify_instance_ownership_rest` is called with the same `hdrs` used for `vastai_stop`/`vastai_delete`/`vastai_verify`. The CLI-based `verify_instance_ownership` is only called in the CLI fallback path. (Test: spy on the requests call to assert the auth header is shared.)

**Image-matching contract tests** (CONTRACT TESTS for the image-allowlist safety fix). The matching policy is exact-reference OR tag-insensitive repository equality. Substring / prefix / registry-port-as-prefix matches are forbidden (ownership-safety defects).

- **`myorg/app:1.0` in allowlist, instance `myorg/app:1.0`.** Returns `True` (exact reference match).
- **`myorg/app:1.0` in allowlist, instance `myorg/app:latest`.** Returns `True` (tag-insensitive repository match: both reduce to `myorg/app`).
- **`myorg/app:1.0` in allowlist, instance `myorg/app@sha256:abc...`.** Returns `True` (tag-insensitive repository match after digest strip).
- **`myorg/app:1.0` in allowlist, instance `myorg/app-malicious:latest`.** Returns **`False`**. **This is the safety-critical test for the prefix-defect fix.** A naive prefix match would accept this image.
- **`myorg/app:1.0` in allowlist, instance `myorg/app-malicious`.** Returns `False` (no tag, repository is `myorg/app-malicious`, not `myorg/app`).
- **`registry:5000/myorg/app:1.0` in allowlist, instance `registry:5000/myorg/app:1.0`.** Returns `True`.
- **`registry:5000/myorg/app:1.0` in allowlist, instance `registry:6000/myorg/app:1.0`.** Returns `False` (different registry host;port).
- **`registry:5000/myorg/app:1.0` in allowlist, instance `registry-malicious/myorg/app:1.0`.** Returns `False` (different registry; this is the safety-critical test that a naive "prefix = part before first colon" implementation would have accepted because `registry` is a prefix of `registry-malicious`).
- **`myorg/app:1.0` in allowlist, instance `myorg/other:1.0`.** Returns `False` (different repository).
- **Empty image string.** Returns `False` (fail-closed).
- **Empty allowlist entry.** Returns `False` (the `if not image` guard).

**`vastai_stop` HTTP error tests** (CONTRACT TESTS for the HTTP-failure fix).

- **`vastai_stop` on 401 response.** Raises `RuntimeError` with the status code. The protocol records it in `stop_error`; DELETE still runs.
- **`vastai_stop` on 403 response.** Same.
- **`vastai_stop` on 429 response.** Same.
- **`vastai_stop` on 500 response.** Same.
- **`vastai_stop` on 200 response.** Returns `None` (no exception).
- **`vastai_stop` on network error.** Raises the underlying exception; the protocol records it.

**`CredentialResolution` strict-strip test** (CONTRACT TEST for the strict-strip invariant).

- `CredentialResolution(state=AVAILABLE, key=" real-key ")` raises `ValueError` (leading/trailing whitespace).

**Schema contract tests.** All ownership and verification tests use the production-shaped Vast.ai response payload, NOT a flat mock:

```python
def _make_response(
    *,
    status_code: int = 200,
    payload: dict | None = None,
) -> dict:
    """Build a production-shaped Vast.ai response payload."""
    if payload is None:
        payload = {
            "instances": {
                "id": 123,
                "image_uuid": "nvidia/cuda:12.4.0-devel-ubuntu22.04",
                "actual_status": "running",
            }
        }
    return {"status_code": status_code, "json": payload}
```

The following tests assert the **nested** `instances.image_uuid` and `instances.actual_status` fields:

- **REST ownership 200 with nested `image_uuid` matching the allowlist.** Returns `True`.
- **REST ownership 200 with nested `image_uuid` not matching.** Returns `False`.
- **REST ownership 200 with missing `instances` envelope.** Returns `False` (fail-closed).
- **REST ownership 200 with non-dict `instances`.** Returns `False`.
- **REST ownership 200 with missing `image_uuid`.** Returns `False`.
- **REST ownership 200 with non-string `image_uuid`.** Returns `False`.
- **Vast.ai verify 200 with nested `instances.actual_status == "destroyed"`.** Returns `VerifyResult(verdict=GONE, status_code=200)`. **This is the safety-critical test for the schema fix.**
- **Vast.ai verify 200 with nested `instances.actual_status == "running"`.** Returns `VerifyResult(verdict=PRESENT, status_code=200)`.
- **Vast.ai verify 200 with nested empty `instances.actual_status`.** Returns `VerifyResult(verdict=PRESENT, status_code=200)`.
- **Vast.ai verify 200 with missing `instances` envelope.** Returns `VerifyResult(verdict=PRESENT, status_code=200)` — the response is well-formed JSON but doesn't carry the status; we cannot prove the instance is gone.
- **Vast.ai verify 200 with non-dict `instances` (e.g. a list or string).** Returns `VerifyResult(verdict=PRESENT, status_code=200)`.
- **Vast.ai verify 200 with `instances.actual_status` missing.** Returns `VerifyResult(verdict=PRESENT, status_code=200)`.
- **Vast.ai verify 200 with non-string `instances.actual_status` (e.g. None).** Returns `VerifyResult(verdict=PRESENT, status_code=200)`.

> Do not write tests with flat payloads like `{"image": ...}` or `{"actual_status": ...}` — those do not match the production API and would pass against the wrong (top-level) parser. The mock helper above enforces the production shape.

- **Vast.ai verify 404.** Returns `VerifyResult(verdict=GONE, status_code=404)`.
- **Vast.ai verify 401 / 403 / 429 / 500.** Returns `VerifyResult(verdict=UNKNOWN, status_code=X, error="http X")`.
- **Vast.ai verify JSON parse failure.** Returns `VerifyResult(verdict=UNKNOWN, status_code=200, error=...)`.
- **Vast.ai verify network error.** Returns `VerifyResult(verdict=UNKNOWN, error=...)`.

**Credential loader tests.** All `read_vastai_api_key` tests are direct unit tests, not integration tests:

- **`VASTAI_API_KEY` env unset, no file.** Returns `CredentialResolution(state=CredentialState.ABSENT, key="")`. The adapter returns `DestroyResult(refusal=NO_CREDENTIALS)`.
- **`VASTAI_API_KEY` env empty.** Returns `CredentialResolution(state=CredentialState.EXPLICITLY_DISABLED, key="")`. The adapter returns `DestroyResult(refusal=CREDENTIALS_DISABLED)`. **This is the safety-critical test for the credential-state blocker fix.** The file credentials are NEVER consulted.
- **`VASTAI_API_KEY` env unset, file present.** Returns `CredentialResolution(state=CredentialState.AVAILABLE, key="<file content>")`. Protocol runs.
- **`VASTAI_API_KEY` env set.** Returns `CredentialResolution(state=CredentialState.AVAILABLE, key="<env value>")`. Protocol runs.
- **Blank file contents (`~/.config/vastai/vast_api_key` exists but is empty).** Returns `CredentialResolution(state=CredentialState.ABSENT, key="")` with a logged warning. **This is the safety-critical test for the blank-file fix.**
- **Whitespace-only file contents.** Returns `CredentialResolution(state=CredentialState.ABSENT, key="")` with a logged warning.
- **File read error (permission denied, transient I/O).** Returns `CredentialResolution(state=CredentialState.ABSENT, key="")` with a logged warning. The `OSError` does not escape from `read_vastai_api_key`. The adapter does not raise.

**CredentialResolution invariant tests:**

- `CredentialResolution(state=AVAILABLE, key="real-key")` is valid.
- `CredentialResolution(state=AVAILABLE, key="")` raises `ValueError` (AVAILABLE requires non-empty key).
- `CredentialResolution(state=AVAILABLE, key="   ")` raises `ValueError` (whitespace-only is not non-empty).
- `CredentialResolution(state=ABSENT, key="")` is valid.
- `CredentialResolution(state=ABSENT, key="leak")` raises `ValueError` (non-AVAILABLE must have empty key).
- `CredentialResolution(state=EXPLICITLY_DISABLED, key="leak")` raises `ValueError`.

### Runner integration tests

- **`VastaiRunner.destroy_instance` with `DestroyResult(verdict=DESTROYED)`.** Returns `True`. `InstanceStatus.DESTROYED` is set.
- **`VastaiRunner.destroy_instance` with `DestroyResult(verdict=LEAKED)`.** Returns `False`. `InstanceStatus` is unchanged. Error logged. The `verify_error` from the protocol is included in the log.
- **`VastaiRunner.destroy_instance` with `DestroyResult(verdict=UNKNOWN)`.** Returns `False`. Warning logged. The `verify_error` and `last_status_code` are included in the log.
- **`VastaiRunner.destroy_instance` with `DestroyResult(refusal=OWNERSHIP)`.** Returns `False`. Error logged.
- **`VastaiRunner.destroy_instance` with `DestroyResult(refusal=NO_CREDENTIALS)`.** Returns `False`. Error logged. (No CLI fallback in the runner path.)
- **`VastaiRunner.destroy_instance` with `DestroyResult(refusal=CREDENTIALS_DISABLED)`.** Returns `False`. Error logged. (No CLI fallback.)
- **`VastaiRunner.destroy_instance` is called once.** The runner does not pre-resolve the API key (the adapter owns credential resolution). Test: spy on `read_vastai_api_key` to assert it's called exactly once per `destroy_instance` call.

### Zombie sweep integration tests

- **Sweep with `allowed_images=set()` and orphan candidate.** `destroy_vastai_instance` returns `OWNERSHIP`. The sweep skips the instance. CLI fallback does NOT fire. `killed` does not increment.
- **Sweep with `allowed_images=None` and orphan candidate.** Protocol runs. `DESTROYED` increments `killed`. `LEAKED` and `UNKNOWN` are logged.
- **Sweep with `VASTAI_API_KEY=""` (EXPLICITLY_DISABLED), no candidates enumerated.** The sweep's short-circuit fires at the top of the function: `read_vastai_api_key()` returns `EXPLICITLY_DISABLED`, the sweep logs the warning, and returns 0 BEFORE the CLI enumeration step. Test: spy on `vastai_cmd(["show", "instances", "--raw"])` to assert it is NOT called. The CLI (which would otherwise use file credentials) is never invoked. **This is the safety-critical test for the EXPLICITLY_DISABLED-scope fix.**
- **Sweep with `CREDENTIALS_DISABLED` after the short-circuit (e.g. via the adapter on a non-existent orphan).** The adapter's `CREDENTIALS_DISABLED` refusal is still produced for the per-orphan path; the sweep logs the skip and does not invoke the CLI. `killed` does not increment; `cli_attempted` does not increment. **This is the safety-critical test for the credential-state blocker fix.** (The short-circuit prevents the per-orphan path from being reached in practice; this test exists to document the precedence.)
- **Sweep with `NO_CREDENTIALS` and orphan candidate (empty allowlist).** `destroy_vastai_instance` returns `NO_CREDENTIALS`. The sweep runs the CLI-based `verify_instance_ownership` first; the empty allowlist rejects; the sweep logs the refusal and does NOT invoke the CLI. `killed` does not increment; `cli_attempted` does not increment. **This is the safety-critical test for the CLI-fallback ownership fix.**
- **Sweep with `NO_CREDENTIALS` and orphan candidate (non-matching allowlist).** Same as above: the CLI-based ownership check rejects; the CLI is not invoked. Test: spy on `vastai_cmd` to assert it is NOT called.
- **Sweep with `NO_CREDENTIALS` and orphan candidate (matching allowlist).** The CLI-based ownership check passes; the CLI is invoked; `cli_attempted` increments.
- **Sweep with `NO_CREDENTIALS` and orphan candidate (CLI ownership lookup fails, e.g. transient CLI error).** The sweep logs the failure and does NOT invoke the CLI. `cli_attempted` does not increment. (The CLI ownership check itself returns `False` on lookup failure — fail-closed.)
- **Sweep with `allowed_images=None` and `NO_CREDENTIALS` (no allowlist, REST path is no-credentials).** The sweep invokes the CLI fallback directly (no ownership check to run). `cli_attempted` increments.
- **Sweep with `OWNERSHIP` and `NO_CREDENTIALS` simultaneously (impossible in practice but documents the precedence).** `OWNERSHIP` takes precedence. CLI fallback does NOT fire.
- **Sweep with `CREDENTIALS_DISABLED` and `NO_CREDENTIALS` simultaneously (impossible in practice but documents the precedence).** `CREDENTIALS_DISABLED` takes precedence. CLI fallback does NOT fire.

## Migration checklist

For the implementer (one PR, one PR-number):

1. Delete `orchestrator.py:poll_instance_progress` and `orchestrator.py:ensure_detached`. Update `CHANGELOG.md`, `docs/api.md`, `docs/architecture.md` to reflect the deletion.
2. Delete `orchestrator.py:load_vastai_api_key`. Add `CredentialState` and `CredentialResolution` to `providers/destroy_adapters/vastai.py`. Rewrite `read_vastai_api_key()` to return `CredentialResolution` with three states. Remove the old `_read_vastai_api_key` from `providers/vastai.py`.
3. Add `allowed_images: AbstractSet[str] | None = None` to `BatchOrchestrator.__init__` (normalised to `frozenset` internally). Pass it to `sweep_zombie_instances` as `allowed_images=allowed_images` (no `*`-args forward).
4. New file `src/vastai_gpu_runner/providers/destroy.py` (the protocol module: `VerifyVerdict`, `DestroyVerdict`, `DestroyRefusal`, `VerifyResult`, `DestroyResult`, `DestroyPolicy`, `belt_and_suspenders`).
5. New file `src/vastai_gpu_runner/providers/destroy_adapters/vastai.py` (the Vast.ai adapter: `CredentialState`, `CredentialResolution`, `read_vastai_api_key`, `_image_is_allowed`, `verify_instance_ownership_rest`, `vastai_stop`, `vastai_delete`, `vastai_verify`, `destroy_vastai_instance`, `VASTAI_POLICY`).
6. Update `VastaiRunner._rest_destroy` (and the public `destroy_instance`) to delegate to `destroy_vastai_instance`. The public `destroy_instance` returns `True` only on `verdict=DESTROYED`. The runner calls the adapter exactly once; the adapter owns credential resolution. Delete the standalone helpers `_rest_stop`, `_rest_delete_with_retries`, `_rest_verify_and_redestroy` from `providers/vastai.py`. The CLI-based `verify_instance_ownership` is retained for the CLI fallback path (separate auth context).
7. Update `orchestrator.py:sweep_zombie_instances` to take `allowed_images: AbstractSet[str] | None`, route through `destroy_vastai_instance`, and branch on `verdict` vs `refusal`. CLI fallback only for `NO_CREDENTIALS`; `CREDENTIALS_DISABLED` and `OWNERSHIP` are never bypassed. Track `cli_attempted` separately from `killed`. Delete `orchestrator.py:_destroy_via_rest`. `BatchOrchestrator._sweep_zombies` becomes a thin delegate that passes `self._allowed_images` to `sweep_zombie_instances`.
8. New file `src/vastai_gpu_runner/unit_lifecycle.py` (the decision module: `Action`, `PreemptCause`, `ProgressSnapshot`, `Continue` / `Complete` / `Preempt`, `decide_next_action`).
9. Update `BatchOrchestrator._poll_cycle_once` to use the two-stage shape (classify-all → preemptions → parallel finalise). Add `_handle_preempt`. Update `BatchOrchestrator._check_unit` to use the same primitive. Delete `_classify_live_unit`. Mark `_check_unit` deprecated.
10. New tests `tests/test_unit_lifecycle.py`, `tests/test_destroy.py`, `tests/test_destroy_adapters_vastai.py`, `tests/test_zombie_sweep.py`. Update `tests/test_batch.py` and `tests/test_batch_orchestrator.py`. The critical regression tests for this revision: empty `actual_status` returns `PRESENT`, `VASTAI_API_KEY=""` returns `CREDENTIALS_DISABLED` and the CLI fallback does not fire, `verify_error` is propagated from `verify_fn` to `DestroyResult`, `DestroyResult` invariants reject refusal with non-zero `attempts` / `last_status_code` / `stop_error` / `verify_error`.
11. Update `CHANGELOG.md` with the v3 changes.
12. (Already done in this design PR.) The tracking issue is [#19](https://github.com/Lambda-Biolab/vastai-gpu-runner/issues/19) "Replace `BatchOrchestrator.allowed_images` with a zombie-destroy callback or provider cleanup policy." The issue documents the two candidate shapes, the open questions (CLI fallback location, runner factory reconciliation, `_handle_preempt` interaction, `LocalRunner` accommodation), and the acceptance criteria. Do not block the v3 implementation on this; revisit before the second adapter ships.

## Review process

This design is published as a PR for review before implementation begins. Reviewers (code owner + ChatGPT-with-GitHub-plugin) should focus on:

- **CredentialState vs DestroyRefusal.** Does the three-state credential model (`AVAILABLE` / `ABSENT` / `EXPLICITLY_DISABLED`) correctly distinguish "no credentials" from "operator-disabled credentials," and does the CLI fallback only fire for `ABSENT`?
- **VerifyVerdict vs DestroyVerdict split.** Does the two-enum split correctly separate the verifier's observation from the protocol's outcome? Are the names right?
- **DestroyRefusal.** Does the `OWNERSHIP` / `NO_CREDENTIALS` / `CREDENTIALS_DISABLED` split correctly distinguish the three failure modes for the CLI fallback decision?
- **Safety behaviour.** Does the empty-allowlist test (`allowed_images=set()` rejects all images) catch the regression that the original draft introduced? Does the `CREDENTIALS_DISABLED` test catch the credential-state regression?
- **Empty-status classification.** Does the `vastai_verify` 200 + empty/missing/non-string `actual_status` correctly return `PRESENT`, not `GONE`?
- **R2 exception policy.** Are the three exception paths (initial R2, check_progress, worker_dead re-check) all handled the way the reviewer asked for?
- **`Preempt` shape.** Is `Preempt(cause: PreemptCause, detail: str | None)` the right shape, or should `cause` carry more structure?
- **Thin-wrapper retention.** Is removing `_classify_live_unit` and keeping `_check_unit` for one deprecation cycle the right balance, or should both be removed in the same PR?
- **Two-stage poll loop.** Does the new `_poll_cycle_once` preserve the parallel-finalisation behaviour of the previous design?
- **VastaiRunner destroy mapping.** Does the `DestroyResult` → boolean return mapping correctly distinguish DESTROYED from LEAKED / UNKNOWN / OWNERSHIP / NO_CREDENTIALS / CREDENTIALS_DISABLED?
- **CLI fallback honesty.** Is the `cli_attempted` counter separate from `killed` the right way to encode the opportunistic nature of the CLI fallback?
- **Credential precedence.** Does the env-first + three-state semantics match the project's intent for Vast.ai credentials? Does the `CREDENTIALS_DISABLED` refusal correctly propagate to the zombie sweep and prevent CLI fallback?
- **REST ownership uses same auth identity.** When REST is available, does `verify_instance_ownership_rest` use the same `hdrs` as `destroy_vastai_instance`?
- **Schema correctness.** Do the adapter tests use the production-shaped nested `instances.image_uuid` / `instances.actual_status` payload (not flat `image` / `actual_status` at the top level)? Do the implementation tests assert the nested parse?
- **CLI fallback ownership.** Does the zombie sweep's `NO_CREDENTIALS` branch call `verify_instance_ownership` (CLI-backed) before invoking the CLI destroy, with all four ownership outcomes tested (empty allowlist, non-matching, matching, lookup failure)?
- **`verify_error` preservation.** Does the `verify_error` field on `DestroyResult` correctly carry the most recent `verify_fn` error through the protocol?
- **`DestroyResult` invariants.** Do the `__post_init__` checks correctly reject `refusal`-with-non-zero-`attempts` / `last_status_code` / `stop_error` / `verify_error`?
- **`CredentialResolution` invariants.** Do the `__post_init__` checks reject `AVAILABLE` with empty/whitespace key, and `non-AVAILABLE` with non-empty key?
- **Blank-file / read-error handling.** Does `read_vastai_api_key` return `ABSENT` (with warning) for blank files and `OSError` on read, rather than `AVAILABLE` with empty key or escaping exceptions?
- **Biolab-runners precedent.** The biolab-runners v15 PR ([#88](https://github.com/Lambda-Biolab/biolab-runners/pull/88)) and v14 PR ([#82](https://github.com/Lambda-Biolab/biolab-runners/pull/82)) are the precedent. Does this design match v15's discipline (localised observation sequence, frozen dataclass plans, ClassVar action enum) and v14's discipline (one protocol module, one adapter per provider, callbacks + policy over class hierarchy)?

Discussion thread: the PR comments.
