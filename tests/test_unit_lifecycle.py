# pyright: reportPrivateUsage=warning, reportMissingParameterType=warning
"""Tests for unit_lifecycle — per-cycle decision tree."""

from __future__ import annotations

import logging

import pytest

from vastai_gpu_runner.unit_lifecycle import (
    Action,
    Complete,
    Continue,
    Preempt,
    PreemptCause,
    ProgressSnapshot,
    decide_next_action,
)


class TestProgressSnapshot:
    def test_from_raw_none(self) -> None:
        ps = ProgressSnapshot.from_raw(None)
        assert ps.complete is False
        assert ps.worker_dead is False
        assert ps.log_tail == ""

    def test_from_raw_non_dict(self) -> None:
        for raw in ("string", 42, [1, 2, 3], True):
            ps = ProgressSnapshot.from_raw(raw)
            assert ps.complete is False
            assert ps.worker_dead is False
            assert ps.log_tail == ""

    def test_from_raw_truthy_coercion(self) -> None:
        ps = ProgressSnapshot.from_raw({"complete": "yes", "worker_dead": 1, "log_tail": "tail"})
        assert ps.complete is True
        assert ps.worker_dead is True
        assert ps.log_tail == "tail"

    def test_from_raw_non_string_log_tail(self) -> None:
        ps = ProgressSnapshot.from_raw({"log_tail": 42})
        assert ps.log_tail == ""

    def test_from_raw_missing_fields(self) -> None:
        ps = ProgressSnapshot.from_raw({})
        assert ps.complete is False
        assert ps.worker_dead is False
        assert ps.log_tail == ""


class _FakeRunner:
    def __init__(self, progress: object) -> None:
        self._progress = progress

    def check_progress(self, instance: object) -> dict[str, object]:
        return self._progress  # type: ignore[return-value]


class _RaisingRunner:
    def check_progress(self, instance: object) -> dict[str, object]:
        raise RuntimeError("ssh timeout")


class TestDecideNextAction:
    def test_r2_done_returns_complete(self) -> None:
        action = decide_next_action(
            unit="u1",
            runner=_FakeRunner({"complete": False, "worker_dead": False}),
            instance="i1",
            is_done_in_r2=lambda u: True,
        )
        assert isinstance(action, Complete)
        assert action.action == Action.COMPLETE

    def test_ssh_complete_returns_complete(self) -> None:
        action = decide_next_action(
            unit="u1",
            runner=_FakeRunner({"complete": True, "worker_dead": False}),
            instance="i1",
            is_done_in_r2=lambda u: False,
        )
        assert isinstance(action, Complete)

    def test_worker_dead_with_r2_recheck_done_returns_complete(self) -> None:
        counter = {"n": 0}

        def is_done(u: object) -> bool:
            counter["n"] += 1
            return counter["n"] >= 2  # 2nd call (the re-check) returns True

        action = decide_next_action(
            unit="u1",
            runner=_FakeRunner({"complete": False, "worker_dead": True, "log_tail": "x"}),
            instance="i1",
            is_done_in_r2=is_done,
        )
        assert isinstance(action, Complete)
        assert counter["n"] == 2

    def test_worker_dead_with_r2_recheck_missed_returns_preempt(self) -> None:
        action = decide_next_action(
            unit="u1",
            runner=_FakeRunner({"complete": False, "worker_dead": True, "log_tail": "fatal"}),
            instance="i1",
            is_done_in_r2=lambda u: False,
        )
        assert isinstance(action, Preempt)
        assert action.action == Action.PREEMPT
        assert action.cause == PreemptCause.WORKER_DIED
        assert action.detail == "fatal"

    def test_ssh_empty_returns_continue(self) -> None:
        action = decide_next_action(
            unit="u1",
            runner=_FakeRunner({}),
            instance="i1",
            is_done_in_r2=lambda u: False,
        )
        assert isinstance(action, Continue)

    def test_ssh_raises_returns_continue(self) -> None:
        action = decide_next_action(
            unit="u1",
            runner=_RaisingRunner(),
            instance="i1",
            is_done_in_r2=lambda u: False,
        )
        assert isinstance(action, Continue)

    def test_r2_raises_falls_through_to_ssh(self, caplog: pytest.LogCaptureFixture) -> None:
        def is_done(u: object) -> bool:
            raise RuntimeError("r2 timeout")

        with caplog.at_level(logging.WARNING, logger="vastai_gpu_runner.unit_lifecycle"):
            action = decide_next_action(
                unit="u1",
                runner=_FakeRunner({"complete": True, "worker_dead": False}),
                instance="i1",
                is_done_in_r2=is_done,
            )
        assert isinstance(action, Complete)
        assert any("R2 check raised" in r.message for r in caplog.records)

    def test_ssh_non_dict_returns_continue(self) -> None:
        action = decide_next_action(
            unit="u1",
            runner=_FakeRunner("not a dict"),  # type: ignore[arg-type]
            instance="i1",
            is_done_in_r2=lambda u: False,
        )
        assert isinstance(action, Continue)

    def test_ssh_raises_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="vastai_gpu_runner.unit_lifecycle"):
            decide_next_action(
                unit="u1",
                runner=_RaisingRunner(),
                instance="i1",
                is_done_in_r2=lambda u: False,
            )
        assert any("check_progress raised" in r.message for r in caplog.records)

    def test_worker_dead_r2_recheck_raises_returns_continue(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def is_done(u: object) -> bool:
            raise RuntimeError("r2 recheck timeout")

        with caplog.at_level(logging.WARNING, logger="vastai_gpu_runner.unit_lifecycle"):
            action = decide_next_action(
                unit="u1",
                runner=_FakeRunner({"complete": False, "worker_dead": True}),
                instance="i1",
                is_done_in_r2=is_done,
            )
        assert isinstance(action, Continue)
        assert any("R2 re-check raised" in r.message for r in caplog.records)


class TestPlanDataclasses:
    def test_continue_action_classvar(self) -> None:
        c = Continue()
        assert c.action == Action.CONTINUE

    def test_complete_action_classvar(self) -> None:
        c = Complete()
        assert c.action == Action.COMPLETE

    def test_preempt_action_classvar(self) -> None:
        p = Preempt(cause=PreemptCause.WORKER_DIED, detail="x")
        assert p.action == Action.PREEMPT
        assert p.cause == PreemptCause.WORKER_DIED
        assert p.detail == "x"

    def test_preempt_detail_optional(self) -> None:
        p = Preempt(cause=PreemptCause.WORKER_DIED)
        assert p.detail is None


class TestActionEnum:
    def test_values(self) -> None:
        assert Action.CONTINUE.value == "continue"
        assert Action.COMPLETE.value == "complete"
        assert Action.PREEMPT.value == "preempt"

    def test_preempt_cause_values(self) -> None:
        assert PreemptCause.WORKER_DIED.value == "worker_died"
