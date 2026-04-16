"""Comprehensive tests for BatchOrchestrator ABC concrete methods.

Covers the full lifecycle: instance-loss state machine, resume, deploy,
poll/classify, finalise, collect (R2 recovery), cleanup, backoff, and
budget guards.

Uses ``ConcreteOrchestrator`` — a minimal in-memory subclass that records
every abstract-method call so tests can assert on event sequences without
any real I/O or cloud provider calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from vastai_gpu_runner.batch import BatchOrchestrator, FailureVerdict
from vastai_gpu_runner.runner import CloudRunner
from vastai_gpu_runner.types import CloudInstance, DeploymentResult

if TYPE_CHECKING:
    from collections.abc import Iterable


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _instance(
    instance_id: str = "inst-1", *, host: str = "1.2.3.4", port: int = 22
) -> CloudInstance:
    return CloudInstance(
        instance_id=instance_id,
        ssh_host=host,
        ssh_port=port,
        cost_per_hour=0.40,
    )


def _ok_result(instance_id: str = "inst-1") -> DeploymentResult:
    return DeploymentResult(success=True, instance=_instance(instance_id))


def _fail_result(error: str = "boot timeout") -> DeploymentResult:
    return DeploymentResult(success=False, error=error)


def _mock_runner(
    *,
    deploy_result: DeploymentResult | None = None,
    progress: dict[str, object] | None = None,
    raise_progress: Exception | None = None,
) -> CloudRunner:
    r = MagicMock(spec=CloudRunner)
    if deploy_result is not None:
        r.run_full_cycle = MagicMock(return_value=deploy_result)
    if raise_progress is not None:
        r.check_progress = MagicMock(side_effect=raise_progress)
    else:
        r.check_progress = MagicMock(return_value=progress or {"running": True, "complete": False})
    r.destroy_instance = MagicMock(return_value=True)
    return r


# ---------------------------------------------------------------------------
# ConcreteOrchestrator — in-memory test double
# ---------------------------------------------------------------------------


@dataclass
class Unit:
    """Minimal BatchUnit-compatible dataclass."""

    key: str
    instance_id: str = ""
    ssh_host: str = ""
    ssh_port: int = 0
    cost_per_hour: float = 0.0
    status: str = "pending"
    retry_count: int = 0
    failure_reason: str = ""
    # Test-controlled flags
    done_in_r2: bool = False
    collect_ok: bool = True
    failure_verdict: FailureVerdict = "retry"
    # Call recorder
    events: list[str] = field(default_factory=list)


class ConcreteOrchestrator(BatchOrchestrator[Unit]):
    """Minimal concrete orchestrator for unit tests.

    All abstract methods are implemented with simple in-memory logic.
    Every event method appends a string to ``unit.events`` so tests can
    assert on the exact call sequence without side-effects.
    """

    def __init__(self, units: list[Unit], **kwargs: object) -> None:
        self.units = units
        self.saves: int = 0
        self.payload_calls: list[str] = []
        self.collect_calls: list[str] = []
        super().__init__(**kwargs)  # type: ignore[arg-type]

    # -- iter hooks ----------------------------------------------------------

    def iter_pending_units(self) -> Iterable[Unit]:
        return [u for u in self.units if u.status == "pending"]

    def iter_active_units(self) -> Iterable[Unit]:
        return [u for u in self.units if u.status in ("deployed", "running")]

    def iter_failed_units(self) -> Iterable[Unit]:
        return [u for u in self.units if u.status == "failed"]

    def iter_completed_units(self) -> Iterable[Unit]:
        return [u for u in self.units if u.status == "downloaded"]

    # -- identity hooks ------------------------------------------------------

    def unit_key(self, unit: Unit) -> str:
        return unit.key

    def unit_label(self, unit: Unit) -> str:
        return unit.key

    # -- payload / collect / classify ----------------------------------------

    def build_unit_payload(self, unit: Unit) -> dict[str, Path]:
        self.payload_calls.append(unit.key)
        return {}

    def reconstruct_instance(self, unit: Unit) -> CloudInstance:
        return _instance(unit.instance_id, host=unit.ssh_host, port=unit.ssh_port)

    def collect_unit_results(self, unit: Unit, instance: CloudInstance) -> bool:
        self.collect_calls.append(unit.key)
        return unit.collect_ok

    def unit_is_done_in_r2(self, unit: Unit) -> bool:
        return unit.done_in_r2

    def classify_failure(self, unit: Unit, error: str) -> FailureVerdict:
        return unit.failure_verdict

    # -- event hooks (state mutations) ---------------------------------------

    def save_state(self) -> None:
        self.saves += 1

    def on_unit_deployed(self, unit: Unit, instance: CloudInstance) -> None:
        unit.instance_id = instance.instance_id
        unit.ssh_host = instance.ssh_host
        unit.ssh_port = instance.ssh_port
        unit.cost_per_hour = instance.cost_per_hour
        unit.status = "deployed"
        unit.events.append("deployed")
        self.save_state()

    def on_unit_failed(self, unit: Unit, reason: str) -> None:
        unit.status = "failed"
        unit.failure_reason = reason
        unit.retry_count += 1
        unit.events.append(f"failed:{reason}")
        self.save_state()

    def on_unit_completed(self, unit: Unit) -> None:
        unit.status = "downloaded"
        unit.events.append("completed")
        self.save_state()

    def on_unit_preempted(self, unit: Unit) -> None:
        unit.instance_id = ""
        unit.status = "pending"
        unit.retry_count += 1
        unit.events.append("preempted")
        self.save_state()


def _orch(units: list[Unit] | None = None, **kwargs: object) -> ConcreteOrchestrator:
    """Factory for ConcreteOrchestrator with safe defaults (kwargs override defaults)."""
    runner = _mock_runner(deploy_result=_ok_result())
    defaults: dict[str, object] = {"label_prefix": "test"}
    defaults.update(kwargs)
    return ConcreteOrchestrator(
        units=units or [],
        runner_factory=lambda: runner,
        **defaults,
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_max_parallel_collects_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_parallel_collects"):
            _orch(max_parallel_collects=0)

    def test_max_parallel_collects_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_parallel_collects"):
            _orch(max_parallel_collects=-1)

    def test_default_max_parallel_collects_is_one(self) -> None:
        o = _orch()
        assert o._max_parallel_collects == 1

    def test_budget_default_zero_means_unlimited(self) -> None:
        o = _orch()
        assert o._budget_usd == 0.0


# ---------------------------------------------------------------------------
# Instance-loss state machine (_handle_instance_loss)
# ---------------------------------------------------------------------------


class TestHandleInstanceLoss:
    def _make(
        self, retry_count: int = 0, max_retries: int = 2, verdict: FailureVerdict = "retry"
    ) -> tuple[ConcreteOrchestrator, Unit]:
        unit = Unit(
            key="u1",
            status="deployed",
            instance_id="i1",
            retry_count=retry_count,
            failure_verdict=verdict,
        )
        o = _orch([unit], max_retries=max_retries)
        o._live_runners[unit.key] = (_mock_runner(), _instance("i1"), unit)
        return o, unit

    def test_retryable_calls_on_unit_preempted(self) -> None:
        o, unit = self._make(retry_count=0, max_retries=2, verdict="retry")
        result = o._handle_instance_loss(unit, unit.key, "ssh timeout")
        assert result is True
        assert "preempted" in unit.events
        assert unit.status == "pending"

    def test_retries_exhausted_calls_on_unit_failed(self) -> None:
        o, unit = self._make(retry_count=2, max_retries=2)
        result = o._handle_instance_loss(unit, unit.key, "crash")
        assert result is False
        assert unit.status == "failed"
        assert "retries exhausted" in unit.failure_reason
        assert not any(e == "preempted" for e in unit.events)

    def test_fatal_verdict_calls_on_unit_failed(self) -> None:
        o, unit = self._make(retry_count=0, max_retries=5, verdict="fatal")
        result = o._handle_instance_loss(unit, unit.key, "bad input")
        assert result is False
        assert unit.status == "failed"
        assert "fatal" in unit.failure_reason

    def test_removes_from_live_runners(self) -> None:
        o, unit = self._make(retry_count=0)
        assert unit.key in o._live_runners
        o._handle_instance_loss(unit, unit.key, "preempt")
        assert unit.key not in o._live_runners

    def test_removes_from_live_runners_even_on_fatal(self) -> None:
        o, unit = self._make(retry_count=0, verdict="fatal")
        o._handle_instance_loss(unit, unit.key, "fatal crash")
        assert unit.key not in o._live_runners

    def test_retries_exhausted_does_not_call_classify_failure(self) -> None:
        """When retry_count >= max_retries, classify_failure must not be consulted."""
        o, unit = self._make(retry_count=2, max_retries=2)
        calls: list[str] = []

        def _spy_classify(u: Unit, err: str) -> FailureVerdict:
            calls.append(err)
            return "retry"

        o.classify_failure = _spy_classify  # type: ignore[method-assign]
        o._handle_instance_loss(unit, unit.key, "crash")
        assert calls == []


# ---------------------------------------------------------------------------
# Resume phase (_resume_from_state)
# ---------------------------------------------------------------------------


class TestResumeFromState:
    def test_reconnects_active_unit(self) -> None:
        unit = Unit(
            key="u1", status="running", instance_id="i1", ssh_host="10.0.0.1", ssh_port=2222
        )
        o = _orch([unit])
        o._resume_from_state()
        assert "u1" in o._live_runners
        _, inst, _ = o._live_runners["u1"]
        assert inst.instance_id == "i1"
        assert unit.status == "running"  # unchanged

    def test_skips_unit_without_instance_id(self) -> None:
        unit = Unit(key="u1", status="deployed", instance_id="")
        o = _orch([unit])
        o._resume_from_state()
        assert not o._live_runners
        assert unit.status == "deployed"  # unchanged — not preempted

    def test_reconstruct_failure_calls_on_unit_preempted(self) -> None:
        unit = Unit(key="u1", status="running", instance_id="dead-i1")
        o = _orch([unit])

        def _boom(u: Unit) -> CloudInstance:
            raise RuntimeError("host unreachable")

        o.reconstruct_instance = _boom  # type: ignore[method-assign]
        o._resume_from_state()
        assert "preempted" in unit.events
        assert unit.status == "pending"
        assert "u1" not in o._live_runners

    def test_skips_pending_units(self) -> None:
        """iter_active_units excludes 'pending' — nothing should be reconnected."""
        unit = Unit(key="u1", status="pending", instance_id="i1")
        o = _orch([unit])
        o._resume_from_state()
        assert not o._live_runners

    def test_multiple_active_units_all_reconnected(self) -> None:
        units = [
            Unit(key=f"u{i}", status="running", instance_id=f"i{i}", ssh_host="h", ssh_port=22)
            for i in range(3)
        ]
        o = _orch(units)
        o._resume_from_state()
        assert len(o._live_runners) == 3


# ---------------------------------------------------------------------------
# Deploy phase
# ---------------------------------------------------------------------------


class TestDeployPhase:
    def test_deploys_pending_units(self) -> None:
        unit = Unit(key="u1")
        runner = _mock_runner(deploy_result=_ok_result("i-42"))
        o = ConcreteOrchestrator([unit], runner_factory=lambda: runner, label_prefix="test")
        o._deploy_phase()
        assert unit.status == "deployed"
        assert unit.instance_id == "i-42"
        assert "u1" in o._live_runners

    def test_skips_r2_done_units(self) -> None:
        unit = Unit(key="u1", done_in_r2=True)
        runner = _mock_runner(deploy_result=_ok_result())
        o = ConcreteOrchestrator([unit], runner_factory=lambda: runner, label_prefix="test")
        o._deploy_phase()
        assert unit.status == "pending"
        assert o.payload_calls == []
        assert not o._live_runners

    def test_no_pending_returns_immediately(self) -> None:
        unit = Unit(key="u1", status="downloaded")
        runner = _mock_runner(deploy_result=_ok_result())
        o = ConcreteOrchestrator([unit], runner_factory=lambda: runner, label_prefix="test")
        o._deploy_phase()
        assert not o._live_runners
        assert o.payload_calls == []

    def test_budget_exceeded_marks_all_failed(self) -> None:
        units = [Unit(key=f"u{i}") for i in range(3)]
        o = _orch(units, budget_usd=10.0)
        with patch("vastai_gpu_runner.batch.check_budget", return_value=False):
            o._deploy_phase()
        assert all(u.status == "failed" for u in units)
        assert all("budget" in u.failure_reason for u in units)


class TestDeployOne:
    def test_success_adds_to_live_runners_and_fires_deployed(self) -> None:
        unit = Unit(key="u1")
        runner = _mock_runner(deploy_result=_ok_result("i-ok"))
        o = ConcreteOrchestrator([unit], runner_factory=lambda: runner, label_prefix="test")
        ok = o._deploy_one(unit)
        assert ok is True
        assert unit.status == "deployed"
        assert "deployed" in unit.events
        assert "u1" in o._live_runners

    def test_failure_result_fires_on_unit_failed(self) -> None:
        unit = Unit(key="u1")
        runner = _mock_runner(deploy_result=_fail_result("GPU unavailable"))
        o = ConcreteOrchestrator([unit], runner_factory=lambda: runner, label_prefix="test")
        ok = o._deploy_one(unit)
        assert ok is False
        assert unit.status == "failed"
        assert "GPU unavailable" in unit.failure_reason

    def test_run_full_cycle_raises_fires_on_unit_failed(self) -> None:
        unit = Unit(key="u1")
        runner = MagicMock(spec=CloudRunner)
        runner.run_full_cycle = MagicMock(side_effect=RuntimeError("network error"))
        runner.destroy_instance = MagicMock()
        o = ConcreteOrchestrator([unit], runner_factory=lambda: runner, label_prefix="test")
        ok = o._deploy_one(unit)
        assert ok is False
        assert unit.status == "failed"
        assert "network error" in unit.failure_reason

    def test_success_only_when_result_instance_not_none(self) -> None:
        """DeploymentResult.success=True but instance=None must still fail."""
        unit = Unit(key="u1")
        runner = MagicMock(spec=CloudRunner)
        runner.run_full_cycle = MagicMock(
            return_value=DeploymentResult(success=True, instance=None, error="no inst")
        )
        runner.destroy_instance = MagicMock()
        o = ConcreteOrchestrator([unit], runner_factory=lambda: runner, label_prefix="test")
        ok = o._deploy_one(unit)
        assert ok is False
        assert unit.status == "failed"


class TestDeployBudgetOk:
    def test_no_budget_always_true(self) -> None:
        o = _orch(budget_usd=0.0)
        assert o._deploy_budget_ok([]) is True

    def test_within_budget_true(self) -> None:
        o = _orch(budget_usd=100.0)
        with patch("vastai_gpu_runner.batch.check_budget", return_value=True):
            assert o._deploy_budget_ok([]) is True

    def test_exceeded_marks_pending_failed(self) -> None:
        units = [Unit(key="u1"), Unit(key="u2")]
        o = _orch(units, budget_usd=5.0)
        with patch("vastai_gpu_runner.batch.check_budget", return_value=False):
            result = o._deploy_budget_ok(units)
        assert result is False
        assert all(u.status == "failed" for u in units)


# ---------------------------------------------------------------------------
# Poll / classify (_classify_live_unit)
# ---------------------------------------------------------------------------


class TestClassifyLiveUnit:
    def _setup(self, unit: Unit, runner: CloudRunner) -> ConcreteOrchestrator:
        o = _orch([unit])
        inst = _instance(unit.instance_id or "i1")
        o._live_runners[unit.key] = (runner, inst, unit)
        return o

    def test_r2_done_returns_terminal(self) -> None:
        unit = Unit(key="u1", done_in_r2=True, status="deployed")
        runner = _mock_runner()
        o = self._setup(unit, runner)
        verdict = o._classify_live_unit(runner, _instance(), unit)
        assert verdict == "terminal"
        runner.check_progress.assert_not_called()

    def test_ssh_complete_returns_terminal(self) -> None:
        unit = Unit(key="u1", status="deployed")
        runner = _mock_runner(progress={"complete": True, "running": False})
        o = self._setup(unit, runner)
        verdict = o._classify_live_unit(runner, _instance(), unit)
        assert verdict == "terminal"

    def test_worker_dead_r2_miss_returns_preempted(self) -> None:
        unit = Unit(key="u1", status="deployed", done_in_r2=False)
        runner = _mock_runner(progress={"complete": False, "worker_dead": True})
        o = self._setup(unit, runner)
        verdict = o._classify_live_unit(runner, _instance(), unit)
        assert verdict == "preempted"

    def test_worker_dead_r2_hit_returns_terminal(self) -> None:
        """R2 re-check after worker_dead: if results landed in R2, recover."""
        unit = Unit(key="u1", status="deployed")
        runner = _mock_runner(progress={"complete": False, "worker_dead": True})
        o = self._setup(unit, runner)
        # First call (before check_progress) → False; second (after worker_dead) → True
        r2_calls = [False, True]
        o.unit_is_done_in_r2 = lambda _u: r2_calls.pop(0)  # type: ignore[method-assign]
        verdict = o._classify_live_unit(runner, _instance(), unit)
        assert verdict == "terminal"

    def test_running_stays_running(self) -> None:
        unit = Unit(key="u1", status="deployed")
        runner = _mock_runner(progress={"running": True, "complete": False})
        o = self._setup(unit, runner)
        verdict = o._classify_live_unit(runner, _instance(), unit)
        assert verdict == "running"

    def test_check_progress_exception_stays_running(self) -> None:
        unit = Unit(key="u1", status="deployed")
        runner = _mock_runner(raise_progress=OSError("SSH refused"))
        o = self._setup(unit, runner)
        verdict = o._classify_live_unit(runner, _instance(), unit)
        assert verdict == "running"

    def test_classify_is_pure_no_state_mutations(self) -> None:
        """classify must never mutate unit state or call destroy_instance."""
        unit = Unit(key="u1", status="deployed")
        runner = _mock_runner(progress={"complete": True})
        o = self._setup(unit, runner)
        o._classify_live_unit(runner, _instance(), unit)
        # Unit state unchanged, destroy never called
        assert unit.status == "deployed"
        runner.destroy_instance.assert_not_called()
        assert unit.key in o._live_runners


# ---------------------------------------------------------------------------
# Finalise (_finalise_completed)
# ---------------------------------------------------------------------------


class TestFinaliseCompleted:
    def _setup(self, unit: Unit) -> tuple[ConcreteOrchestrator, CloudRunner, CloudInstance]:
        o = _orch([unit])
        runner = _mock_runner()
        inst = _instance(unit.instance_id or "i1")
        o._live_runners[unit.key] = (runner, inst, unit)
        return o, runner, inst

    def test_collect_success_calls_on_completed_and_destroy(self) -> None:
        unit = Unit(key="u1", status="deployed", collect_ok=True)
        o, runner, inst = self._setup(unit)
        result = o._finalise_completed(runner, inst, unit, unit.key)
        assert result == "completed"
        assert unit.status == "downloaded"
        assert "completed" in unit.events
        runner.destroy_instance.assert_called_once_with(inst)
        assert unit.key not in o._live_runners

    def test_collect_failure_calls_on_failed_and_destroy(self) -> None:
        unit = Unit(key="u1", status="deployed", collect_ok=False)
        o, runner, inst = self._setup(unit)
        result = o._finalise_completed(runner, inst, unit, unit.key)
        assert result == "failed"
        assert unit.status == "failed"
        runner.destroy_instance.assert_called_once()

    def test_collect_exception_calls_on_failed_and_destroy(self) -> None:
        unit = Unit(key="u1", status="deployed")
        o, runner, inst = self._setup(unit)

        def _boom(u: Unit, i: CloudInstance) -> bool:
            raise RuntimeError("rsync error")

        o.collect_unit_results = _boom  # type: ignore[method-assign]
        result = o._finalise_completed(runner, inst, unit, unit.key)
        assert result == "failed"
        assert unit.status == "failed"
        runner.destroy_instance.assert_called_once()

    def test_removes_from_live_runners(self) -> None:
        unit = Unit(key="u1", status="deployed", collect_ok=True)
        o, runner, inst = self._setup(unit)
        assert unit.key in o._live_runners
        o._finalise_completed(runner, inst, unit, unit.key)
        assert unit.key not in o._live_runners


# ---------------------------------------------------------------------------
# Collect phase (_collect_phase) — R2 recovery for failed units
# ---------------------------------------------------------------------------


class TestCollectPhase:
    def test_recovers_failed_unit_with_r2_done(self) -> None:
        unit = Unit(key="u1", status="failed", done_in_r2=True, collect_ok=True)
        o = _orch([unit], r2_sink=MagicMock())
        o._collect_phase()
        assert unit.status == "downloaded"
        assert "u1" in o.collect_calls

    def test_skips_failed_unit_without_r2(self) -> None:
        unit = Unit(key="u1", status="failed", done_in_r2=False)
        o = _orch([unit], r2_sink=MagicMock())
        o._collect_phase()
        assert unit.status == "failed"
        assert o.collect_calls == []

    def test_noop_without_r2_sink(self) -> None:
        unit = Unit(key="u1", status="failed", done_in_r2=True)
        o = _orch([unit], r2_sink=None)
        o._collect_phase()
        assert unit.status == "failed"
        assert o.collect_calls == []

    def test_collect_returns_false_does_not_mark_completed(self) -> None:
        unit = Unit(key="u1", status="failed", done_in_r2=True, collect_ok=False)
        o = _orch([unit], r2_sink=MagicMock())
        o._collect_phase()
        # collect was called, but collect_unit_results returned False
        assert unit.key in o.collect_calls
        assert unit.status == "failed"  # not promoted to downloaded

    def test_collect_exception_does_not_propagate(self) -> None:
        unit = Unit(key="u1", status="failed", done_in_r2=True)
        o = _orch([unit], r2_sink=MagicMock())

        def _boom(u: Unit, i: CloudInstance) -> bool:
            raise RuntimeError("disk full")

        o.collect_unit_results = _boom  # type: ignore[method-assign]
        # Should not raise — exceptions are swallowed with a warning log
        o._collect_phase()
        assert unit.status == "failed"


# ---------------------------------------------------------------------------
# Cleanup phase (_cleanup_phase)
# ---------------------------------------------------------------------------


class TestCleanupPhase:
    def test_destroys_leftover_instances(self) -> None:
        unit = Unit(key="u1", status="deployed", instance_id="i1")
        o = _orch([unit])
        runner = _mock_runner()
        inst = _instance("i1")
        o._live_runners[unit.key] = (runner, inst, unit)
        with patch("vastai_gpu_runner.batch.sweep_zombie_instances", return_value=0):
            o._cleanup_phase()
        runner.destroy_instance.assert_called_once_with(inst)

    def test_clears_live_runners(self) -> None:
        unit = Unit(key="u1", status="deployed", instance_id="i1")
        o = _orch([unit])
        o._live_runners[unit.key] = (_mock_runner(), _instance("i1"), unit)
        with patch("vastai_gpu_runner.batch.sweep_zombie_instances", return_value=0):
            o._cleanup_phase()
        assert o._live_runners == {}

    def test_destroy_exception_does_not_propagate(self) -> None:
        unit = Unit(key="u1", status="deployed", instance_id="i1")
        o = _orch([unit])
        runner = MagicMock(spec=CloudRunner)
        runner.destroy_instance = MagicMock(side_effect=RuntimeError("API error"))
        o._live_runners[unit.key] = (runner, _instance("i1"), unit)
        with patch("vastai_gpu_runner.batch.sweep_zombie_instances", return_value=0):
            o._cleanup_phase()  # must not raise
        assert o._live_runners == {}

    def test_noop_with_empty_live_runners(self) -> None:
        o = _orch()
        with patch("vastai_gpu_runner.batch.sweep_zombie_instances", return_value=0):
            o._cleanup_phase()  # no runners → no-op
        assert o._live_runners == {}


# ---------------------------------------------------------------------------
# Zombie sweep (_sweep_zombies)
# ---------------------------------------------------------------------------


class TestSweepZombies:
    def test_delegates_to_sweep_zombie_instances(self) -> None:
        o = _orch(label_prefix="myproject-boltz2-abc", r2_batch_id="batch-xyz")
        with patch("vastai_gpu_runner.batch.sweep_zombie_instances", return_value=3) as mock_fn:
            killed = o._sweep_zombies()
        assert killed == 3
        _, kwargs = mock_fn.call_args
        assert kwargs["label_prefix"] == "myproject-boltz2-abc"
        assert kwargs["r2_batch_id"] == "batch-xyz"

    def test_exception_returns_zero(self) -> None:
        o = _orch()
        with patch(
            "vastai_gpu_runner.batch.sweep_zombie_instances", side_effect=RuntimeError("api down")
        ):
            killed = o._sweep_zombies()
        assert killed == 0


# ---------------------------------------------------------------------------
# Backoff (_advance_poll_interval)
# ---------------------------------------------------------------------------


class TestAdvancePollInterval:
    def test_progress_resets_to_min_of_base_and_five(self) -> None:
        with patch("vastai_gpu_runner.batch.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.time = MagicMock(return_value=0.0)
            result = ConcreteOrchestrator._advance_poll_interval(
                any_progress=True, cur_interval=60, base=30, max_interval=60
            )
        assert result == 5  # min(base=30, 5)
        mock_time.sleep.assert_not_called()

    def test_progress_resets_to_base_when_base_lt_five(self) -> None:
        with patch("vastai_gpu_runner.batch.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.time = MagicMock(return_value=0.0)
            result = ConcreteOrchestrator._advance_poll_interval(
                any_progress=True, cur_interval=30, base=3, max_interval=10
            )
        assert result == 3  # min(base=3, 5)
        mock_time.sleep.assert_not_called()

    def test_no_progress_sleeps_and_doubles(self) -> None:
        with patch("vastai_gpu_runner.batch.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.time = MagicMock(return_value=0.0)
            result = ConcreteOrchestrator._advance_poll_interval(
                any_progress=False, cur_interval=10, base=10, max_interval=60
            )
        mock_time.sleep.assert_called_once_with(10)
        assert result == 20  # doubled

    def test_no_progress_caps_at_max_interval(self) -> None:
        with patch("vastai_gpu_runner.batch.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.time = MagicMock(return_value=0.0)
            result = ConcreteOrchestrator._advance_poll_interval(
                any_progress=False, cur_interval=40, base=30, max_interval=60
            )
        assert result == 60  # 40*2=80 capped to 60


# ---------------------------------------------------------------------------
# Budget check (_poll_budget_ok / _deploy_budget_ok)
# ---------------------------------------------------------------------------


class TestBudgetChecks:
    def test_poll_budget_no_budget_always_true(self) -> None:
        o = _orch(budget_usd=0.0)
        assert o._poll_budget_ok() is True

    def test_poll_budget_within_ceiling_true(self) -> None:
        o = _orch(budget_usd=100.0)
        with patch("vastai_gpu_runner.batch.check_budget", return_value=True):
            assert o._poll_budget_ok() is True

    def test_poll_budget_exceeded_false(self) -> None:
        o = _orch(budget_usd=5.0)
        with patch("vastai_gpu_runner.batch.check_budget", return_value=False):
            assert o._poll_budget_ok() is False

    def test_deploy_budget_no_budget_always_true(self) -> None:
        o = _orch(budget_usd=0.0)
        assert o._deploy_budget_ok([]) is True

    def test_deploy_budget_within_ceiling_true(self) -> None:
        o = _orch(budget_usd=100.0)
        with patch("vastai_gpu_runner.batch.check_budget", return_value=True):
            assert o._deploy_budget_ok([]) is True


# ---------------------------------------------------------------------------
# Full run() lifecycle (integration-level smoke tests)
# ---------------------------------------------------------------------------


class TestRunLifecycle:
    def test_happy_path_deploy_then_complete(self) -> None:
        unit = Unit(key="u1")
        runner = _mock_runner(
            deploy_result=_ok_result("i1"),
            progress={"complete": True, "running": False},
        )
        o = ConcreteOrchestrator(
            [unit],
            runner_factory=lambda: runner,
            label_prefix="test",
            poll_interval_seconds=1,
        )
        with patch("vastai_gpu_runner.batch.sweep_zombie_instances", return_value=0):
            o.run()
        assert unit.status == "downloaded"
        assert "deployed" in unit.events
        assert "completed" in unit.events

    def test_no_units_is_noop(self) -> None:
        o = _orch()
        with patch("vastai_gpu_runner.batch.sweep_zombie_instances", return_value=0):
            o.run()
        assert o._live_runners == {}

    def test_deploy_failure_leaves_unit_failed(self) -> None:
        unit = Unit(key="u1")
        runner = _mock_runner(deploy_result=_fail_result("no offers"))
        o = ConcreteOrchestrator([unit], runner_factory=lambda: runner, label_prefix="test")
        with patch("vastai_gpu_runner.batch.sweep_zombie_instances", return_value=0):
            o.run()
        assert unit.status == "failed"
        assert "no offers" in unit.failure_reason

    def test_phases_called_in_order(self) -> None:
        """run() must call phases in the documented order."""
        calls: list[str] = []
        o = _orch()
        o._resume_from_state = lambda: calls.append("resume")  # type: ignore[method-assign]
        o._deploy_phase = lambda: calls.append("deploy")  # type: ignore[method-assign]
        o._sweep_zombies = lambda: calls.append("sweep") or 0  # type: ignore[method-assign]
        o._poll_phase = lambda: calls.append("poll")  # type: ignore[method-assign]
        o._collect_phase = lambda: calls.append("collect")  # type: ignore[method-assign]
        o._cleanup_phase = lambda: calls.append("cleanup")  # type: ignore[method-assign]
        o.run()
        assert calls == ["resume", "deploy", "sweep", "poll", "collect", "cleanup"]


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------


def test_abc_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        BatchOrchestrator(  # type: ignore[abstract]
            runner_factory=lambda: CloudRunner(),
            label_prefix="test",
        )
