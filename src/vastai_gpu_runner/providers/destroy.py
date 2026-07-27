"""Belt-and-suspenders destroy protocol — generic loop + typed result.

The loop shape is provider-agnostic: ``stop -> DELETE x retry -> verify ->
re-destroy`` (with a second verification after the resurrection cleanup
window). The provider supplies three callbacks (``stop_fn``,
``delete_fn``, ``verify_fn``) and a ``DestroyPolicy`` with timing/retry
constants; the loop itself never knows the provider, the URL, the API
key, or the image-ownership rules.

The Vast.ai adapter (``providers/destroy_adapters/vastai.py``) wires
the REST callbacks plus the Vast.ai-discovered policy.

Design contract: returns a ``DestroyResult`` with exactly one of
``verdict`` (protocol ran) or ``refusal`` (pre-protocol outcome
— e.g. ownership denied, no credentials). ``verdict`` and
``refusal`` are mutually exclusive; their respective field
invariants are checked in ``DestroyResult.__post_init__``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class VerifyVerdict(StrEnum):
    """What a single ``verify_fn`` call reported."""

    GONE = "gone"  # 404 OR 200 + instances.actual_status == "destroyed"
    PRESENT = (
        "present"  # 200 + instances.actual_status != "destroyed" (incl. empty/missing/non-string)
    )
    UNKNOWN = "unknown"  # other HTTP status, parse failure, network error


class DestroyVerdict(StrEnum):
    """Outcome after the protocol ran."""

    DESTROYED = "destroyed"  # verified gone after at least one DELETE
    LEAKED = "leaked"  # verify says gone but DELETE never returned success
    UNKNOWN = "unknown"  # protocol ran but the final state is unclear


class DestroyRefusal(StrEnum):
    """Pre-protocol refusal — the protocol never ran."""

    OWNERSHIP = "ownership"  # image allowlist rejected the instance
    NO_CREDENTIALS = "no_credentials"  # no API key (CLI fallback permitted)
    CREDENTIALS_DISABLED = "credentials_disabled"  # VASTAI_API_KEY="" (no fallback)


@dataclass(frozen=True)
class DestroyResult:
    """Typed result of a destroy attempt.

    Exactly one of ``verdict`` or ``refusal`` is set. Field-level
    invariants (checked in ``__post_init__``):

    - ``refusal is not None`` → protocol never ran. ``attempts == 0``;
      no stop/verify errors; no last status code.
    - ``verdict is not None`` → protocol ran. ``attempts >= 1`` for
      DESTROYED/LEAKED; ``attempts >= 0`` for UNKNOWN (the protocol
      may have failed at the very first step).
    """

    verdict: DestroyVerdict | None = None
    refusal: DestroyRefusal | None = None
    attempts: int = 0
    stop_error: str | None = None
    last_status_code: int | None = None
    verify_error: str | None = None

    def __post_init__(self) -> None:
        """Validate mutually-exclusive verdict/refusal + per-state invariants."""
        self._check_one_of_verdict_refusal()
        if self.refusal is not None:
            self._check_refusal_invariants()
        elif self.verdict == DestroyVerdict.UNKNOWN:
            # UNKNOWN allows attempts=0 (protocol failed at the first step).
            pass
        else:
            self._check_verdict_invariants()

    def _check_one_of_verdict_refusal(self) -> None:
        if (self.verdict is None) == (self.refusal is None):
            raise ValueError("DestroyResult must have exactly one of verdict or refusal")

    def _check_refusal_invariants(self) -> None:
        if self.attempts != 0:
            raise ValueError("refusal requires attempts == 0")
        if self.stop_error is not None:
            raise ValueError("refusal requires stop_error is None")
        if self.last_status_code is not None:
            raise ValueError("refusal requires last_status_code is None")
        if self.verify_error is not None:
            raise ValueError("refusal requires verify_error is None")

    def _check_verdict_invariants(self) -> None:
        # DESTROYED or LEAKED — protocol ran; attempts >= 1
        if self.attempts < 1:
            raise ValueError("verdict requires attempts >= 1")


@dataclass(frozen=True)
class DestroyPolicy:
    """Timing + retry constants for the protocol. Provider-specific.

    Verified against the Vast.ai API empirically (see
    ``providers/destroy_adapters/vastai.py:VASTAI_POLICY``). A
    different provider may need a different ``verify_delay_s``,
    ``retry_delay_s``, or ``max_delete_attempts``.
    """

    verify_delay_s: float
    retry_delay_s: float
    max_delete_attempts: int
    verify_after_resurrection: bool = True

    def __post_init__(self) -> None:
        """Validate timing constants are non-negative and attempts >= 1."""
        if self.verify_delay_s < 0:
            raise ValueError("verify_delay_s must be non-negative")
        if self.retry_delay_s < 0:
            raise ValueError("retry_delay_s must be non-negative")
        if self.max_delete_attempts < 1:
            raise ValueError("max_delete_attempts must be >= 1")


# Callback types -------------------------------------------------------------
# Each callback raises on infrastructure failure. The loop logs + reports
# UNKNOWN with the error message rather than letting the exception escape.
# All callbacks are expected to return quickly (no long blocking I/O).
# ``verify_fn`` returns a ``VerifyVerdict``; ``stop_fn`` and ``delete_fn``
# return the HTTP status code (or None for non-HTTP transport like CLI).


VerifyFn = Callable[[], "VerifyResult"]
StopFn = Callable[[], "int | None"]
DeleteFn = Callable[[], "int | None"]


@dataclass(frozen=True)
class VerifyResult:
    """Wrapper for ``verify_fn`` output (verdict + diagnostics)."""

    verdict: VerifyVerdict
    status_code: int | None = None
    error: str | None = None


def belt_and_suspenders(
    stop_fn: StopFn,
    delete_fn: DeleteFn,
    verify_fn: VerifyFn,
    *,
    policy: DestroyPolicy,
) -> DestroyResult:
    """Run the belt-and-suspenders protocol. Returns a typed result.

    Sequence (per the v3 doc):

    1. ``stop_fn`` — set state=stopped
    2. ``verify_delay_s`` seconds
    3. ``delete_fn`` up to ``policy.max_delete_attempts`` times
       (with ``retry_delay_s`` between attempts)
    4. ``verify_fn`` — confirm GONE
    5. If the verify says GONE but DELETE never returned success, the
       verdict is ``LEAKED`` (delete-fail-immune by design — the
       instance is gone, that's what we wanted)
    6. If ``policy.verify_after_resurrection``, sleep
       ``verify_delay_s`` and verify again, since the instance can
       resurrect during the verify window
    7. Any unexpected callback exception becomes ``verdict=UNKNOWN``
       with the exception's repr in ``verify_error`` /
       ``stop_error`` / ``last_status_code=None``

    The loop never raises. The provider's API quirks live in the
    callbacks; the protocol's shape lives here.
    """
    last_status, stop_error = _step_stop(stop_fn)
    if stop_error is not None:
        return DestroyResult(
            verdict=DestroyVerdict.UNKNOWN,
            attempts=0,
            stop_error=stop_error,
        )

    if policy.verify_delay_s > 0:
        time.sleep(policy.verify_delay_s)

    attempts, last_status = _step_delete(delete_fn, policy, last_status)

    try:
        verify = verify_fn()
    except Exception as exc:
        logger.warning("belt_and_suspenders: verify_fn raised %s", exc)
        return DestroyResult(
            verdict=DestroyVerdict.UNKNOWN,
            attempts=attempts,
            stop_error=stop_error,
            last_status_code=last_status,
            verify_error=f"{type(exc).__name__}: {exc}",
        )

    if verify.verdict == VerifyVerdict.GONE:
        return _verdict_after_gone(attempts, stop_error, last_status)
    if verify.verdict == VerifyVerdict.UNKNOWN:
        return DestroyResult(
            verdict=DestroyVerdict.UNKNOWN,
            attempts=attempts,
            stop_error=stop_error,
            last_status_code=last_status,
            verify_error=verify.error,
        )
    return _verdict_after_present(verify_fn, policy, attempts, stop_error, last_status, verify)


def _step_stop(stop_fn: StopFn) -> tuple[int | None, str | None]:
    """Run stop_fn; return (last_status, error). error is non-None on raise."""
    try:
        return stop_fn(), None
    except Exception as exc:
        logger.warning("belt_and_suspenders: stop_fn raised %s", exc)
        return None, f"{type(exc).__name__}: {exc}"


def _step_delete(
    delete_fn: DeleteFn, policy: DestroyPolicy, last_status: int | None
) -> tuple[int, int | None]:
    """Run delete_fn up to max_delete_attempts; return (attempts, last_status)."""
    attempts = 0
    for attempt in range(1, policy.max_delete_attempts + 1):
        try:
            status = delete_fn()
        except Exception as exc:
            logger.warning(
                "belt_and_suspenders: delete_fn attempt %d raised %s",
                attempt,
                exc,
            )
            last_status = None
            continue
        attempts = attempt
        if status is not None:
            last_status = status
        if status in (200, 204):
            break
        if attempt < policy.max_delete_attempts and policy.retry_delay_s > 0:
            time.sleep(policy.retry_delay_s)
    return attempts, last_status


def _verdict_after_gone(
    attempts: int,
    stop_error: str | None,
    last_status: int | None,
) -> DestroyResult:
    """verify_fn said GONE: DESTROYED if DELETE returned 2xx, else LEAKED."""
    if last_status in (200, 204):
        return DestroyResult(
            verdict=DestroyVerdict.DESTROYED,
            attempts=attempts,
            stop_error=stop_error,
            last_status_code=last_status,
        )
    return DestroyResult(
        verdict=DestroyVerdict.LEAKED,
        attempts=attempts,
        stop_error=stop_error,
        last_status_code=last_status,
    )


def _verdict_after_present(
    verify_fn: VerifyFn,
    policy: DestroyPolicy,
    attempts: int,
    stop_error: str | None,
    last_status: int | None,
    verify: VerifyResult,
) -> DestroyResult:
    """verify_fn said PRESENT: resurrection check; otherwise UNKNOWN."""
    if not policy.verify_after_resurrection:
        return _result_unknown(attempts, stop_error, last_status, verify.error)
    if policy.verify_delay_s > 0:
        time.sleep(policy.verify_delay_s)
    try:
        second = verify_fn()
    except Exception as exc:
        logger.warning("belt_and_suspenders: second verify_fn raised %s", exc)
        return DestroyResult(
            verdict=DestroyVerdict.UNKNOWN,
            attempts=attempts,
            stop_error=stop_error,
            last_status_code=last_status,
            verify_error=f"{type(exc).__name__}: {exc}",
        )
    if second.verdict == VerifyVerdict.GONE:
        return DestroyResult(
            verdict=DestroyVerdict.DESTROYED,
            attempts=attempts,
            stop_error=stop_error,
            last_status_code=last_status,
        )
    return _result_unknown(attempts, stop_error, last_status, second.error or verify.error)


def _result_unknown(
    attempts: int,
    stop_error: str | None,
    last_status: int | None,
    verify_error: str | None,
) -> DestroyResult:
    return DestroyResult(
        verdict=DestroyVerdict.UNKNOWN,
        attempts=attempts,
        stop_error=stop_error,
        last_status_code=last_status,
        verify_error=verify_error or "verify returned PRESENT",
    )


__all__ = [
    "DeleteFn",
    "DestroyPolicy",
    "DestroyRefusal",
    "DestroyResult",
    "DestroyVerdict",
    "StopFn",
    "VerifyFn",
    "VerifyResult",
    "VerifyVerdict",
    "belt_and_suspenders",
]
