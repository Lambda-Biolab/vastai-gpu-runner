# pyright: reportPrivateUsage=warning, reportMissingParameterType=warning, reportUnusedFunction=false, reportUnusedClass=false
"""Tests for providers.destroy — belt-and-suspenders protocol."""

from __future__ import annotations

import pytest

from vastai_gpu_runner.providers.destroy import (
    DeleteFn,
    DestroyPolicy,
    DestroyRefusal,
    DestroyResult,
    DestroyVerdict,
    StopFn,
    VerifyFn,
    VerifyResult,
    VerifyVerdict,
    belt_and_suspenders,
)


class TestDestroyResultInvariants:
    def test_verdict_does_not_require_refusal(self) -> None:
        r = DestroyResult(verdict=DestroyVerdict.DESTROYED, attempts=1)
        assert r.verdict == DestroyVerdict.DESTROYED
        assert r.refusal is None

    def test_refusal_does_not_require_verdict(self) -> None:
        r = DestroyResult(refusal=DestroyRefusal.OWNERSHIP)
        assert r.refusal == DestroyRefusal.OWNERSHIP
        assert r.verdict is None

    def test_both_none_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one of verdict or refusal"):
            DestroyResult()

    def test_both_set_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one of verdict or refusal"):
            DestroyResult(
                verdict=DestroyVerdict.DESTROYED,
                refusal=DestroyRefusal.OWNERSHIP,
                attempts=1,
            )

    def test_refusal_with_attempts_raises(self) -> None:
        with pytest.raises(ValueError, match="attempts == 0"):
            DestroyResult(refusal=DestroyRefusal.OWNERSHIP, attempts=1)

    def test_refusal_with_stop_error_raises(self) -> None:
        with pytest.raises(ValueError, match="stop_error"):
            DestroyResult(refusal=DestroyRefusal.OWNERSHIP, stop_error="boom")

    def test_refusal_with_last_status_raises(self) -> None:
        with pytest.raises(ValueError, match="last_status_code"):
            DestroyResult(refusal=DestroyRefusal.OWNERSHIP, last_status_code=503)

    def test_refusal_with_verify_error_raises(self) -> None:
        with pytest.raises(ValueError, match="verify_error"):
            DestroyResult(refusal=DestroyRefusal.OWNERSHIP, verify_error="boom")

    def test_destroyed_with_attempts_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="attempts >= 1"):
            DestroyResult(verdict=DestroyVerdict.DESTROYED, attempts=0)

    def test_leaked_with_attempts_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="attempts >= 1"):
            DestroyResult(verdict=DestroyVerdict.LEAKED, attempts=0)

    def test_unknown_with_attempts_zero_allowed(self) -> None:
        """UNKNOWN verdict allows attempts=0 (protocol failed at the first step)."""
        r = DestroyResult(verdict=DestroyVerdict.UNKNOWN, attempts=0)
        assert r.attempts == 0
        assert r.verdict == DestroyVerdict.UNKNOWN


class TestDestroyPolicyInvariants:
    def test_valid_policy(self) -> None:
        p = DestroyPolicy(verify_delay_s=5.0, retry_delay_s=3.0, max_delete_attempts=3)
        assert p.verify_delay_s == 5.0
        assert p.retry_delay_s == 3.0
        assert p.max_delete_attempts == 3
        assert p.verify_after_resurrection is True

    def test_negative_verify_delay_raises(self) -> None:
        with pytest.raises(ValueError, match="verify_delay_s"):
            DestroyPolicy(verify_delay_s=-1, retry_delay_s=0, max_delete_attempts=3)

    def test_negative_retry_delay_raises(self) -> None:
        with pytest.raises(ValueError, match="retry_delay_s"):
            DestroyPolicy(verify_delay_s=0, retry_delay_s=-1, max_delete_attempts=3)

    def test_zero_max_delete_raises(self) -> None:
        with pytest.raises(ValueError, match="max_delete_attempts"):
            DestroyPolicy(verify_delay_s=0, retry_delay_s=0, max_delete_attempts=0)


def _make_callbacks(
    stop_result: int | None = 200,
    delete_results: tuple[int | None, ...] = (200,),
    verify_result: VerifyResult | None = None,
) -> tuple[StopFn, DeleteFn, VerifyFn, dict[str, int]]:
    """Build stop/delete/verify callbacks with call counters.

    ``delete_results`` is the sequence of status codes returned for each
    delete attempt (the i-th attempt gets the i-th element; subsequent
    attempts reuse the last element).
    """
    if verify_result is None:
        verify_result = VerifyResult(VerifyVerdict.GONE)
    counters = {"stop": 0, "delete": 0, "verify": 0}

    def stop_fn() -> int | None:
        counters["stop"] += 1
        return stop_result

    def delete_fn() -> int | None:
        counters["delete"] += 1
        idx = min(counters["delete"] - 1, len(delete_results) - 1)
        return delete_results[idx]

    def verify_fn() -> VerifyResult:
        counters["verify"] += 1
        return verify_result

    return stop_fn, delete_fn, verify_fn, counters


class TestBeltAndSuspendersHappyPath:
    def test_destroyed(self) -> None:
        p = DestroyPolicy(verify_delay_s=0, retry_delay_s=0, max_delete_attempts=3)
        stop, delete, verify, _ = _make_callbacks(
            delete_results=(200,),
            verify_result=VerifyResult(VerifyVerdict.GONE),
        )
        result = belt_and_suspenders(stop, delete, verify, policy=p)
        assert result.verdict == DestroyVerdict.DESTROYED
        assert result.attempts == 1
        assert result.last_status_code == 200

    def test_leaked_when_verify_gone_but_delete_failed(self) -> None:
        """DELETE returned 503 but verify reports GONE -> LEAKED.

        The instance is gone (the user's intent is achieved); the leak
        is a status-code leak, not a real instance leak. We don't
        re-destroy.
        """
        p = DestroyPolicy(verify_delay_s=0, retry_delay_s=0, max_delete_attempts=3)
        stop, delete, verify, _ = _make_callbacks(
            delete_results=(503,),
            verify_result=VerifyResult(VerifyVerdict.GONE),
        )
        result = belt_and_suspenders(stop, delete, verify, policy=p)
        assert result.verdict == DestroyVerdict.LEAKED
        assert result.last_status_code == 503

    def test_present_means_unknown(self) -> None:
        p = DestroyPolicy(verify_delay_s=0, retry_delay_s=0, max_delete_attempts=3)
        stop, delete, verify, _ = _make_callbacks(
            verify_result=VerifyResult(VerifyVerdict.PRESENT, error="still there"),
        )
        result = belt_and_suspenders(stop, delete, verify, policy=p)
        assert result.verdict == DestroyVerdict.UNKNOWN
        assert result.verify_error == "still there"


class TestBeltAndSuspendersRetries:
    def test_retries_until_success(self) -> None:
        p = DestroyPolicy(verify_delay_s=0, retry_delay_s=0, max_delete_attempts=3)
        stop, delete, verify, counters = _make_callbacks(
            delete_results=(503, 503, 200),
            verify_result=VerifyResult(VerifyVerdict.GONE),
        )
        result = belt_and_suspenders(stop, delete, verify, policy=p)
        assert result.verdict == DestroyVerdict.DESTROYED
        assert counters["delete"] == 3
        assert result.attempts == 3

    def test_exhausts_max_attempts(self) -> None:
        p = DestroyPolicy(verify_delay_s=0, retry_delay_s=0, max_delete_attempts=2)
        stop, delete, verify, counters = _make_callbacks(
            delete_results=(503, 503),
            verify_result=VerifyResult(VerifyVerdict.GONE),
        )
        result = belt_and_suspenders(stop, delete, verify, policy=p)
        # verify GONE but last_status_code=503 -> LEAKED (not DESTROYED).
        # The instance is gone (the user's intent is achieved); the leak
        # is the status-code leak.
        assert result.verdict == DestroyVerdict.LEAKED
        assert counters["delete"] == 2
        assert result.attempts == 2
        assert result.last_status_code == 503


class TestBeltAndSuspendersExceptionBoundary:
    def test_stop_raises_returns_unknown(self) -> None:
        p = DestroyPolicy(verify_delay_s=0, retry_delay_s=0, max_delete_attempts=3)

        def stop() -> int | None:
            raise RuntimeError("network down")

        def delete() -> int | None:
            return 200

        def verify() -> VerifyResult:
            return VerifyResult(VerifyVerdict.GONE)

        result = belt_and_suspenders(stop, delete, verify, policy=p)
        assert result.verdict == DestroyVerdict.UNKNOWN
        assert result.attempts == 0
        assert "network down" in (result.stop_error or "")

    def test_verify_raises_returns_unknown(self) -> None:
        p = DestroyPolicy(verify_delay_s=0, retry_delay_s=0, max_delete_attempts=3)

        def verify() -> VerifyResult:
            raise RuntimeError("API timeout")

        result = belt_and_suspenders(lambda: 200, lambda: 200, verify, policy=p)
        assert result.verdict == DestroyVerdict.UNKNOWN
        assert "API timeout" in (result.verify_error or "")

    def test_delete_raises_does_not_break_loop(self) -> None:
        """delete_fn may raise on a flaky attempt; the loop continues."""
        p = DestroyPolicy(verify_delay_s=0, retry_delay_s=0, max_delete_attempts=3)
        counter = {"n": 0}

        def delete() -> int | None:
            counter["n"] += 1
            if counter["n"] < 3:
                raise RuntimeError("flaky")
            return 200

        result = belt_and_suspenders(
            lambda: 200, delete, lambda: VerifyResult(VerifyVerdict.GONE), policy=p
        )
        assert result.verdict == DestroyVerdict.DESTROYED
        assert counter["n"] == 3


class TestBeltAndSuspendersResurrection:
    def test_second_verify_confirms_gone(self) -> None:
        """First verify says PRESENT; second verify (after delay) says GONE."""
        p = DestroyPolicy(
            verify_delay_s=0,
            retry_delay_s=0,
            max_delete_attempts=3,
            verify_after_resurrection=True,
        )
        results = iter(
            [
                VerifyResult(VerifyVerdict.PRESENT, error="still"),
                VerifyResult(VerifyVerdict.GONE),
            ]
        )

        def verify() -> VerifyResult:
            return next(results)

        result = belt_and_suspenders(lambda: 200, lambda: 200, verify, policy=p)
        assert result.verdict == DestroyVerdict.DESTROYED

    def test_no_resurrection_check_when_disabled(self) -> None:
        """verify_after_resurrection=False skips the second verify."""
        p = DestroyPolicy(
            verify_delay_s=0,
            retry_delay_s=0,
            max_delete_attempts=3,
            verify_after_resurrection=False,
        )
        verify_calls = {"n": 0}

        def verify() -> VerifyResult:
            verify_calls["n"] += 1
            return VerifyResult(VerifyVerdict.PRESENT, error="still")

        result = belt_and_suspenders(lambda: 200, lambda: 200, verify, policy=p)
        assert result.verdict == DestroyVerdict.UNKNOWN
        assert verify_calls["n"] == 1

    def test_unknown_verify_returns_unknown(self) -> None:
        p = DestroyPolicy(verify_delay_s=0, retry_delay_s=0, max_delete_attempts=3)
        stop, delete, verify, _ = _make_callbacks(
            verify_result=VerifyResult(VerifyVerdict.UNKNOWN, error="parse fail"),
        )
        result = belt_and_suspenders(stop, delete, verify, policy=p)
        assert result.verdict == DestroyVerdict.UNKNOWN
        assert "parse fail" in (result.verify_error or "")
