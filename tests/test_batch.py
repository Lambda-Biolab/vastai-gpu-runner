"""Tests for BatchOrchestrator ABC.

Uses a mock CloudRunner and a concrete in-memory orchestrator subclass.
No real Vast.ai calls, no SSH. Covers:

- deploy phase success and failure
- poll phase: R2-first, SSH fallback, silent-crash detection
- resume from active units (reconstruct live runners)
- retry cap → fatal after retries exhausted
- collect phase: R2 recovery for failed-but-uploaded units
- cleanup phase: destroys leftover instances
- full run() lifecycle end-to-end
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
# Fake unit + concrete orchestrator
# ---------------------------------------------------------------------------


@dataclass
class FakeUnit:
    """In-memory unit for tests. Mirrors ShardState/JobState shape."""

    key: str
    instance_id: str = ""
    ssh_host: str = ""
    ssh_port: int = 0
    cost_per_hour: float = 0.0
    status: str = "pending"
    retry_count: int = 0
    failure_reason: str = ""
    done_in_r2: bool = False
    collect_result: bool = True
    events: list[str] = field(default_factory=list)


class FakeOrchestrator(BatchOrchestrator[FakeUnit]):
    """Test orchestrator that owns a list of FakeUnits and records events."""

    def __init__(self, units: list[FakeUnit], **kwargs: object) -> None:
        self.units = units
        self.state_saves = 0
        self.payload_builds: list[str] = []
        self.collect_calls: list[str] = []
        super().__init__(**kwargs)  # type: ignore[arg-type]

    def iter_pending_units(self) -> Iterable[FakeUnit]:
        return [u for u in self.units if u.status == "pending"]

    def iter_active_units(self) -> Iterable[FakeUnit]:
        return [u for u in self.units if u.status in ("deployed", "running")]

    def iter_failed_units(self) -> Iterable[FakeUnit]:
        return [u for u in self.units if u.status == "failed"]

    def iter_completed_units(self) -> Iterable[FakeUnit]:
        return [u for u in self.units if u.status == "downloaded"]

    def save_state(self) -> None:
        self.state_saves += 1

    def unit_key(self, unit: FakeUnit) -> str:
        return unit.key

    def unit_label(self, unit: FakeUnit) -> str:
        return unit.key

    def build_unit_payload(self, unit: FakeUnit) -> dict[str, Path]:
        self.payload_builds.append(unit.key)
        return {"input.txt": Path("/tmp/fake")}

    def reconstruct_instance(self, unit: FakeUnit) -> CloudInstance:
        return CloudInstance(
            instance_id=unit.instance_id,
            ssh_host=unit.ssh_host,
            ssh_port=unit.ssh_port,
        )

    def collect_unit_results(self, unit: FakeUnit, instance: CloudInstance) -> bool:
        del instance
        self.collect_calls.append(unit.key)
        return unit.collect_result

    def unit_is_done_in_r2(self, unit: FakeUnit) -> bool:
        return unit.done_in_r2

    def classify_failure(self, unit: FakeUnit, error: str) -> FailureVerdict:
        del unit, error
        return "retry"

    def on_unit_deployed(self, unit: FakeUnit, instance: CloudInstance) -> None:
        unit.instance_id = instance.instance_id
        unit.ssh_host = instance.ssh_host
        unit.ssh_port = instance.ssh_port
        unit.cost_per_hour = instance.cost_per_hour
        unit.status = "deployed"
        unit.events.append("deployed")
        self.save_state()

    def on_unit_failed(self, unit: FakeUnit, reason: str) -> None:
        unit.status = "failed"
        unit.failure_reason = reason
        unit.retry_count += 1
        unit.events.append(f"failed:{reason}")
        self.save_state()

    def on_unit_completed(self, unit: FakeUnit) -> None:
        unit.status = "downloaded"
        unit.events.append("completed")
        self.save_state()

    def on_unit_preempted(self, unit: FakeUnit) -> None:
        unit.instance_id = ""
        unit.status = "pending"
        unit.retry_count += 1
        unit.events.append("preempted")
        self.save_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_deploy(instance_id: str = "inst-1") -> DeploymentResult:
    return DeploymentResult(
        success=True,
        instance=CloudInstance(
            instance_id=instance_id,
            ssh_host="1.2.3.4",
            ssh_port=22,
            cost_per_hour=0.5,
        ),
    )


def _fail_deploy(error: str = "boot timeout") -> DeploymentResult:
    return DeploymentResult(success=False, error=error)


def _mock_runner_factory(
    *,
    deploy_result: DeploymentResult,
    progress: dict[str, object] | None = None,
) -> object:
    """Build a factory returning mock CloudRunners with controlled behaviour."""

    def factory() -> CloudRunner:
        r = CloudRunner()
        r.run_full_cycle = MagicMock(return_value=deploy_result)  # type: ignore[method-assign]
        r.check_progress = MagicMock(  # type: ignore[method-assign]
            return_value=progress or {"running": True, "complete": False}
        )
        r.destroy_instance = MagicMock(return_value=True)  # type: ignore[method-assign]
        return r

    return factory


# ---------------------------------------------------------------------------
# Deploy phase
# ---------------------------------------------------------------------------


class TestDeployPhase:
    def test_deploy_success_sets_live_runner_and_event(self) -> None:
        unit = FakeUnit(key="u1")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy("inst-42")),
            label_prefix="test",
        )
        orch._deploy_phase()

        assert unit.status == "deployed"
        assert unit.instance_id == "inst-42"
        assert unit.events == ["deployed"]
        assert "u1" in [orch.unit_key(u) for _, _, u in orch._live_runners.values()]

    def test_deploy_failure_marks_failed(self) -> None:
        unit = FakeUnit(key="u1")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_fail_deploy("no offers")),
            label_prefix="test",
        )
        orch._deploy_phase()

        assert unit.status == "failed"
        assert "no offers" in unit.failure_reason
        assert unit.retry_count == 1
        assert not orch._live_runners

    def test_deploy_skips_units_already_done_in_r2(self) -> None:
        unit = FakeUnit(key="u1", done_in_r2=True)
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        orch._deploy_phase()

        assert unit.status == "pending"  # unchanged
        assert unit.events == []
        assert orch.payload_builds == []  # no deploy attempted

    def test_deploy_budget_exceeded_fails_all(self) -> None:
        unit1 = FakeUnit(key="u1")
        unit2 = FakeUnit(key="u2")
        orch = FakeOrchestrator(
            units=[unit1, unit2],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
            budget_usd=10.0,
        )
        with patch("vastai_gpu_runner.batch.check_budget", return_value=False):
            orch._deploy_phase()

        assert unit1.status == "failed"
        assert unit2.status == "failed"
        assert "budget" in unit1.failure_reason
        assert "budget" in unit2.failure_reason

    def test_deploy_parallel_many_units(self) -> None:
        units = [FakeUnit(key=f"u{i}") for i in range(5)]
        orch = FakeOrchestrator(
            units=units,
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
            max_parallel_deploys=4,
        )
        orch._deploy_phase()

        assert all(u.status == "deployed" for u in units)
        assert len(orch._live_runners) == 5


# ---------------------------------------------------------------------------
# Poll phase: _check_unit
# ---------------------------------------------------------------------------


class TestCheckUnit:
    def test_r2_done_short_circuits_to_completed(self) -> None:
        unit = FakeUnit(key="u1", done_in_r2=True, status="deployed")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        runner = MagicMock(spec=CloudRunner)
        runner.destroy_instance = MagicMock(return_value=True)
        instance = CloudInstance(instance_id="i1")
        orch._live_runners[unit.key] = (runner, instance, unit)

        verdict = orch._check_unit(runner, instance, unit)

        assert verdict == "completed"
        assert unit.status == "downloaded"
        runner.check_progress.assert_not_called()
        runner.destroy_instance.assert_called_once()

    def test_ssh_complete_collects_results(self) -> None:
        unit = FakeUnit(key="u1", status="deployed")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        runner = MagicMock(spec=CloudRunner)
        runner.check_progress = MagicMock(return_value={"complete": True, "running": False})
        runner.destroy_instance = MagicMock(return_value=True)
        instance = CloudInstance(instance_id="i1")
        orch._live_runners[unit.key] = (runner, instance, unit)

        verdict = orch._check_unit(runner, instance, unit)

        assert verdict == "completed"
        assert unit.status == "downloaded"
        assert "u1" in orch.collect_calls

    def test_collect_failure_marks_unit_failed(self) -> None:
        unit = FakeUnit(key="u1", status="deployed", collect_result=False)
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        runner = MagicMock(spec=CloudRunner)
        runner.check_progress = MagicMock(return_value={"complete": True, "running": False})
        runner.destroy_instance = MagicMock(return_value=True)
        instance = CloudInstance(instance_id="i1")
        orch._live_runners[unit.key] = (runner, instance, unit)

        verdict = orch._check_unit(runner, instance, unit)

        assert verdict == "failed"
        assert unit.status == "failed"

    def test_silent_worker_crash_triggers_instance_loss(self) -> None:
        unit = FakeUnit(key="u1", status="deployed", instance_id="i1")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        runner = MagicMock(spec=CloudRunner)
        runner.check_progress = MagicMock(
            return_value={
                "complete": False,
                "running": False,
                "worker_dead": True,
            }
        )
        runner.destroy_instance = MagicMock(return_value=True)
        instance = CloudInstance(instance_id="i1")
        orch._live_runners[unit.key] = (runner, instance, unit)

        verdict = orch._check_unit(runner, instance, unit)

        assert verdict == "preempted"
        assert unit.status == "pending"
        assert unit.instance_id == ""
        runner.destroy_instance.assert_called_once()

    def test_silent_crash_but_r2_done_recovers(self) -> None:
        """Worker dead + R2 DONE between checks → treat as completed."""
        unit = FakeUnit(key="u1", status="deployed")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        runner = MagicMock(spec=CloudRunner)
        # First R2 check: False. Second (after worker_dead): True.
        r2_check_calls = [False, True]
        orch.unit_is_done_in_r2 = lambda _u: r2_check_calls.pop(0)  # type: ignore[method-assign]
        runner.check_progress = MagicMock(
            return_value={"complete": False, "running": False, "worker_dead": True}
        )
        runner.destroy_instance = MagicMock(return_value=True)
        instance = CloudInstance(instance_id="i1")
        orch._live_runners[unit.key] = (runner, instance, unit)

        verdict = orch._check_unit(runner, instance, unit)

        assert verdict == "completed"
        assert unit.status == "downloaded"

    def test_running_keeps_unit_live(self) -> None:
        unit = FakeUnit(key="u1", status="deployed")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        runner = MagicMock(spec=CloudRunner)
        runner.check_progress = MagicMock(return_value={"running": True, "complete": False})
        runner.destroy_instance = MagicMock(return_value=True)
        instance = CloudInstance(instance_id="i1")
        orch._live_runners[unit.key] = (runner, instance, unit)

        verdict = orch._check_unit(runner, instance, unit)

        assert verdict == "running"
        assert unit.status == "deployed"
        assert unit.key in orch._live_runners


# ---------------------------------------------------------------------------
# Retry cap
# ---------------------------------------------------------------------------


class TestRetryCap:
    def test_retry_exhausted_marks_fatal(self) -> None:
        unit = FakeUnit(key="u1", status="deployed", retry_count=2, instance_id="i1")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
            max_retries=2,
        )
        orch._live_runners[unit.key] = (
            MagicMock(spec=CloudRunner),
            CloudInstance(instance_id="i1"),
            unit,
        )

        redeploy_ok = orch._handle_instance_loss(unit, unit.key, "crash")

        assert redeploy_ok is False
        assert unit.status == "failed"
        assert "retries exhausted" in unit.failure_reason

    def test_under_cap_allows_redeploy(self) -> None:
        unit = FakeUnit(key="u1", status="deployed", retry_count=0, instance_id="i1")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
            max_retries=2,
        )
        orch._live_runners[unit.key] = (
            MagicMock(spec=CloudRunner),
            CloudInstance(instance_id="i1"),
            unit,
        )

        redeploy_ok = orch._handle_instance_loss(unit, unit.key, "crash")

        assert redeploy_ok is True
        assert unit.status == "pending"
        assert unit.retry_count == 1

    def test_fatal_classification_skips_redeploy(self) -> None:
        unit = FakeUnit(key="u1", status="deployed", retry_count=0)
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        orch._live_runners[unit.key] = (
            MagicMock(spec=CloudRunner),
            CloudInstance(instance_id="i1"),
            unit,
        )
        orch.classify_failure = lambda u, e: "fatal"  # type: ignore[method-assign,return-value]

        redeploy_ok = orch._handle_instance_loss(unit, unit.key, "bad input")

        assert redeploy_ok is False
        assert unit.status == "failed"
        assert "fatal" in unit.failure_reason


# ---------------------------------------------------------------------------
# Resume from state
# ---------------------------------------------------------------------------


class TestResume:
    def test_resume_reconnects_active_units(self) -> None:
        unit = FakeUnit(
            key="u1", status="running", instance_id="i1", ssh_host="1.2.3.4", ssh_port=22
        )
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        orch._resume_from_state()

        assert "u1" in orch._live_runners
        _, instance, _ = orch._live_runners["u1"]
        assert instance.instance_id == "i1"
        assert unit.status == "running"  # unchanged

    def test_resume_no_instance_id_skipped(self) -> None:
        unit = FakeUnit(key="u1", status="deployed", instance_id="")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        orch._resume_from_state()
        assert not orch._live_runners

    def test_resume_reconstruct_failure_preempts(self) -> None:
        unit = FakeUnit(key="u1", status="deployed", instance_id="i1")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )

        def _boom(u: FakeUnit) -> CloudInstance:
            raise RuntimeError("dead host")

        orch.reconstruct_instance = _boom  # type: ignore[method-assign]
        orch._resume_from_state()

        assert unit.status == "pending"
        assert "preempted" in unit.events


# ---------------------------------------------------------------------------
# Collect + cleanup phases
# ---------------------------------------------------------------------------


class TestCollectAndCleanup:
    def test_collect_phase_r2_recovery(self) -> None:
        unit = FakeUnit(key="u1", status="failed", done_in_r2=True)
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
            r2_sink=MagicMock(),
        )
        orch._collect_phase()

        assert unit.status == "downloaded"
        assert "u1" in orch.collect_calls

    def test_collect_phase_no_r2_sink_is_noop(self) -> None:
        unit = FakeUnit(key="u1", status="failed", done_in_r2=True)
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
            r2_sink=None,
        )
        orch._collect_phase()

        assert unit.status == "failed"
        assert orch.collect_calls == []

    def test_cleanup_destroys_leftover_instances(self) -> None:
        unit = FakeUnit(key="u1", status="deployed", instance_id="i1")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        runner = MagicMock(spec=CloudRunner)
        runner.destroy_instance = MagicMock(return_value=True)
        instance = CloudInstance(instance_id="i1")
        orch._live_runners[unit.key] = (runner, instance, unit)

        with patch("vastai_gpu_runner.batch.sweep_zombie_instances", return_value=0):
            orch._cleanup_phase()

        runner.destroy_instance.assert_called_once_with(instance)
        assert not orch._live_runners


# ---------------------------------------------------------------------------
# Full run() lifecycle
# ---------------------------------------------------------------------------


class TestRunLifecycle:
    def test_run_happy_path_deploy_then_complete(self) -> None:
        unit = FakeUnit(key="u1")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(
                deploy_result=_ok_deploy("i1"),
                progress={"complete": True, "running": False},
            ),
            label_prefix="test-happy",
            poll_interval_seconds=1,
        )

        with patch("vastai_gpu_runner.batch.sweep_zombie_instances", return_value=0):
            orch.run()

        assert unit.status == "downloaded"
        assert "deployed" in unit.events
        assert "completed" in unit.events

    def test_run_no_pending_units_is_noop(self) -> None:
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test-empty",
        )

        with patch("vastai_gpu_runner.batch.sweep_zombie_instances", return_value=0):
            orch.run()

        assert not orch._live_runners

    def test_run_deploy_failure_propagates_to_failed_state(self) -> None:
        unit = FakeUnit(key="u1")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_fail_deploy("boom")),
            label_prefix="test-fail",
        )

        with patch("vastai_gpu_runner.batch.sweep_zombie_instances", return_value=0):
            orch.run()

        assert unit.status == "failed"
        assert "boom" in unit.failure_reason


# ---------------------------------------------------------------------------
# Zombie sweep delegation
# ---------------------------------------------------------------------------


class TestZombieSweep:
    def test_sweep_delegates_with_label_prefix(self) -> None:
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="oralamp-boltz2-abc",
            r2_batch_id="batch-123",
        )
        with patch("vastai_gpu_runner.batch.sweep_zombie_instances", return_value=2) as mock_sweep:
            killed = orch._sweep_zombies()

        assert killed == 2
        _, kwargs = mock_sweep.call_args
        assert kwargs["label_prefix"] == "oralamp-boltz2-abc"
        assert kwargs["r2_batch_id"] == "batch-123"

    def test_sweep_failure_returns_zero(self) -> None:
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        with patch(
            "vastai_gpu_runner.batch.sweep_zombie_instances",
            side_effect=RuntimeError("api down"),
        ):
            killed = orch._sweep_zombies()
        assert killed == 0


# ---------------------------------------------------------------------------
# Parallel collect (max_parallel_collects)
# ---------------------------------------------------------------------------


class TestParallelCollect:
    """Tests for the ``max_parallel_collects`` constructor arg + split hooks.

    The hook exists so consumers with slow I/O-bound finalise steps (e.g.
    rsync over SSH) can opt into concurrent collection when many units
    complete around the same wall-clock time. Default (1) preserves
    sequential semantics; tests below pin both paths.
    """

    def test_default_is_one_sequential(self) -> None:
        """Default constructor: max_parallel_collects == 1."""
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        assert orch._max_parallel_collects == 1

    def test_invalid_value_raises(self) -> None:
        """max_parallel_collects < 1 is rejected at construction."""
        with pytest.raises(ValueError, match="max_parallel_collects"):
            FakeOrchestrator(
                units=[],
                runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
                label_prefix="test",
                max_parallel_collects=0,
            )

    def test_classify_live_unit_is_pure_no_side_effects(self) -> None:
        """_classify_live_unit must not mutate state or call destroy_instance.

        This is the contract that makes parallel finalise safe: the classify
        half is pure, so we can batch multiple units' classifications before
        touching any state.
        """
        unit = FakeUnit(key="u1", status="deployed")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        runner = MagicMock(spec=CloudRunner)
        runner.check_progress = MagicMock(return_value={"complete": True, "running": False})
        runner.destroy_instance = MagicMock(return_value=True)
        instance = CloudInstance(instance_id="i1")
        orch._live_runners[unit.key] = (runner, instance, unit)

        verdict = orch._classify_live_unit(runner, instance, unit)

        assert verdict == "terminal"
        # No side effects: unit still "deployed", no collect called, no destroy.
        assert unit.status == "deployed"
        assert "u1" not in orch.collect_calls
        runner.destroy_instance.assert_not_called()
        assert unit.key in orch._live_runners

    def test_classify_returns_preempted_without_destroying(self) -> None:
        """Classify reports preempted but does NOT destroy — that's phase B."""
        unit = FakeUnit(key="u1", status="deployed", instance_id="i1")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        runner = MagicMock(spec=CloudRunner)
        runner.check_progress = MagicMock(
            return_value={"complete": False, "running": False, "worker_dead": True}
        )
        runner.destroy_instance = MagicMock(return_value=True)
        instance = CloudInstance(instance_id="i1")
        orch._live_runners[unit.key] = (runner, instance, unit)

        verdict = orch._classify_live_unit(runner, instance, unit)

        assert verdict == "preempted"
        runner.destroy_instance.assert_not_called()  # destroy is phase B, not classify
        assert unit.status == "deployed"  # state mutation is phase B too

    def test_poll_cycle_finalises_multiple_terminal_units_in_parallel(self) -> None:
        """3 units, all terminal in one cycle, max_parallel_collects=3.

        All 3 must be finalised, all 3 collect_calls recorded, all 3 status
        == "downloaded", all 3 destroy_instance calls made, no units left in
        live_runners.
        """
        units = [FakeUnit(key=f"u{i}", status="deployed") for i in range(3)]
        orch = FakeOrchestrator(
            units=units,
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
            max_parallel_collects=3,
        )
        runners: list[MagicMock] = []
        for u in units:
            runner = MagicMock(spec=CloudRunner)
            runner.check_progress = MagicMock(return_value={"complete": True, "running": False})
            runner.destroy_instance = MagicMock(return_value=True)
            instance = CloudInstance(instance_id=f"i-{u.key}")
            orch._live_runners[u.key] = (runner, instance, u)
            runners.append(runner)

        any_progress = orch._poll_cycle_once()

        assert any_progress is True
        assert sorted(orch.collect_calls) == ["u0", "u1", "u2"]
        assert all(u.status == "downloaded" for u in units)
        for runner in runners:
            runner.destroy_instance.assert_called_once()
        assert not orch._live_runners  # all removed

    def test_poll_cycle_sequential_when_max_parallel_is_one(self) -> None:
        """max_parallel_collects=1 path: same 3 units, serial finalise.

        Same observable outcome as the parallel path — the split doesn't
        change semantics, only concurrency.
        """
        units = [FakeUnit(key=f"u{i}", status="deployed") for i in range(3)]
        orch = FakeOrchestrator(
            units=units,
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
            max_parallel_collects=1,
        )
        for u in units:
            runner = MagicMock(spec=CloudRunner)
            runner.check_progress = MagicMock(return_value={"complete": True, "running": False})
            runner.destroy_instance = MagicMock(return_value=True)
            instance = CloudInstance(instance_id=f"i-{u.key}")
            orch._live_runners[u.key] = (runner, instance, u)

        any_progress = orch._poll_cycle_once()

        assert any_progress is True
        assert orch.collect_calls == ["u0", "u1", "u2"]  # strict order
        assert all(u.status == "downloaded" for u in units)
        assert not orch._live_runners


# ---------------------------------------------------------------------------
# Sanity: ABC cannot be instantiated
# ---------------------------------------------------------------------------


def test_abc_cannot_instantiate() -> None:
    with pytest.raises(TypeError):
        BatchOrchestrator(  # type: ignore[abstract]
            runner_factory=lambda: CloudRunner(),
            label_prefix="test",
        )
