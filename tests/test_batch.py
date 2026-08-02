# pyright: reportPrivateUsage=warning, reportMissingParameterType=warning, reportUnusedFunction=false, reportUnusedClass=false
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
- v4 cleanup_policy wiring: severity-by-outcome logging, label
  delimiter safety, provider decoupling (no module imports of
  providers/*).
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from vastai_gpu_runner.batch import BatchOrchestrator, FailureVerdict, RunnerFactory
from vastai_gpu_runner.cleanup_policy import (
    CleanupRefusal,
    CleanupResult,
    CleanupVerdict,
    InstanceCandidate,
    Provider,
    ProviderCleanupPolicy,
)
from vastai_gpu_runner.runner import CloudRunner
from vastai_gpu_runner.storage.r2 import R2Sink
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


def _noop_runner_factory() -> CloudRunner:
    """Build a factory returning a single MagicMock CloudRunner.

    Default runner_factory for tests that don't need to exercise
    the runner.
    """
    runner = MagicMock()
    runner.run_full_cycle = MagicMock(
        return_value=MagicMock(
            success=True,
            instance=CloudInstance(instance_id="noop"),
        )
    )
    runner.check_progress = MagicMock(return_value={"running": True, "complete": False})
    runner.destroy_instance = MagicMock(return_value=True)
    return runner


class FakeOrchestrator(BatchOrchestrator[FakeUnit]):
    """Test orchestrator that owns a list of FakeUnits and records events."""

    def __init__(
        self,
        units: list[FakeUnit],
        cleanup_policy: ProviderCleanupPolicy | None = None,
        *,
        runner_factory: RunnerFactory = _noop_runner_factory,
        label_prefix: str = "test",
        workspace_dir: str = "/tmp/test",
        r2_sink: R2Sink | None = None,
        r2_batch_id: str = "test-batch",
        budget_usd: float = 0.0,
        max_retries: int = 2,
        max_parallel_deploys: int = 1,
        max_parallel_collects: int = 1,
        poll_interval_seconds: int = 30,
        zombie_sweep_every_n_cycles: int = 5,
        poll_timeout_seconds: float = 0.0,
    ) -> None:
        self.units = units
        self.state_saves = 0
        self.payload_builds: list[str] = []
        self.collect_calls: list[str] = []
        super().__init__(
            runner_factory=runner_factory,
            label_prefix=label_prefix,
            cleanup_policy=cleanup_policy or _noop_cleanup_policy(),
            workspace_dir=workspace_dir,
            r2_sink=r2_sink,
            r2_batch_id=r2_batch_id,
            budget_usd=budget_usd,
            max_retries=max_retries,
            max_parallel_deploys=max_parallel_deploys,
            max_parallel_collects=max_parallel_collects,
            poll_interval_seconds=poll_interval_seconds,
            zombie_sweep_every_n_cycles=zombie_sweep_every_n_cycles,
            poll_timeout_seconds=poll_timeout_seconds,
        )

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
) -> RunnerFactory:
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


def _noop_cleanup_policy() -> ProviderCleanupPolicy:
    """Default policy for tests that don't exercise zombie sweep."""

    def _list() -> list[InstanceCandidate]:
        return []

    def _destroy(candidate: InstanceCandidate) -> CleanupResult:
        return CleanupResult(verdict=CleanupVerdict.DESTROYED)

    return ProviderCleanupPolicy(
        provider=Provider.VASTAI,
        list_instances_fn=_list,
        destroy_fn=_destroy,
    )


def _recording_cleanup_policy(
    *,
    candidates: list[InstanceCandidate],
    destroy_responses: dict[str, CleanupResult] | None = None,
    list_returns: list[InstanceCandidate] | None = None,
) -> ProviderCleanupPolicy:
    """Policy that exposes the candidates fed to ``list_instances`` and
    the cleanup results dispatched to ``destroy``. Use
    ``destroy_responses`` keyed by ``candidate.instance_id`` to drive
    per-candidate outcomes (defaults to ``DESTROYED``). Use
    ``list_returns`` to override what ``list_instances`` returns on
    subsequent calls (defaults to ``candidates`` for every call).
    """
    responses = destroy_responses or {}
    seen_candidates: list[InstanceCandidate] = []
    list_invocations = 0
    destroy_invocations: list[InstanceCandidate] = []

    def _list() -> list[InstanceCandidate]:
        nonlocal list_invocations
        list_invocations += 1
        return list(list_returns if list_returns is not None else candidates)

    def _destroy(candidate: InstanceCandidate) -> CleanupResult:
        seen_candidates.append(candidate)
        destroy_invocations.append(candidate)
        return responses.get(
            candidate.instance_id,
            CleanupResult(verdict=CleanupVerdict.DESTROYED),
        )

    policy = ProviderCleanupPolicy(
        provider=Provider.VASTAI,
        list_instances_fn=_list,
        destroy_fn=_destroy,
    )
    policy.__dict__["_test_list_invocations"] = lambda: list_invocations  # type: ignore[attr-defined]
    policy.__dict__["_test_destroy_invocations"] = lambda: list(destroy_invocations)  # type: ignore[attr-defined]
    return policy


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
# capture_preempt_diagnostics — silent worker crash forensics (#164)
#
# Before the fix: a "worker died silently" verdict fired destroy_instance
# without any log capture, so the orchestrator only recorded the single
# "worker died silently" log line. 3 of 5 instances in a 2026-04-20 MD
# batch crashed this way ~60-180 s after Deploy success, and there was
# no diagnostic trail to investigate why. After the fix: the default
# implementation pulls /workspace/worker.log (+ worker.exitcode +
# dmesg tail) via SSH and writes it to
# batch_diagnostics/{unit}_{instance}_{timestamp}.log before destroy.
# A raise inside the capture must not block destroy — leaked instances
# keep burning dollars.
# ---------------------------------------------------------------------------


class TestCapturePreemptDiagnostics:
    """Preempted path captures worker.log before destroy, swallows errors."""

    @staticmethod
    def _preempted_poll() -> dict[str, object]:
        return {"complete": False, "running": False, "worker_dead": True}

    def _orch_with_preempted_unit(
        self,
    ) -> tuple[FakeOrchestrator, MagicMock, CloudInstance, FakeUnit]:
        unit = FakeUnit(key="u1", status="deployed", instance_id="i1")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        runner = MagicMock(spec=CloudRunner)
        runner.check_progress = MagicMock(return_value=self._preempted_poll())
        runner.destroy_instance = MagicMock(return_value=True)
        instance = CloudInstance(
            instance_id="i1",
            ssh_host="1.2.3.4",
            ssh_port=22,
            ssh_user="root",
        )
        orch._live_runners[unit.key] = (runner, instance, unit)
        return orch, runner, instance, unit

    def test_capture_called_before_destroy_on_preempted(self) -> None:
        """capture_preempt_diagnostics fires before destroy_instance."""
        orch, runner, instance, unit = self._orch_with_preempted_unit()

        call_order: list[str] = []
        orch.capture_preempt_diagnostics = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda *_a, **_k: call_order.append("capture")
        )
        runner.destroy_instance = MagicMock(
            side_effect=lambda *_a, **_k: call_order.append("destroy")
        )

        verdict = orch._check_unit(runner, instance, unit)

        assert verdict == "preempted"
        assert call_order == ["capture", "destroy"]
        orch.capture_preempt_diagnostics.assert_called_once_with(runner, instance, unit)

    def test_capture_exception_does_not_block_destroy(self) -> None:
        """A raising capture must not prevent the instance from being destroyed."""
        orch, runner, instance, unit = self._orch_with_preempted_unit()
        orch.capture_preempt_diagnostics = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("SSH network is on fire")
        )

        verdict = orch._check_unit(runner, instance, unit)

        assert verdict == "preempted"
        runner.destroy_instance.assert_called_once()
        assert unit.status == "pending"  # instance-loss bookkeeping still happened

    def test_default_capture_writes_diagnostics_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default implementation SSH-cats worker.log into batch_diagnostics/."""
        orch, runner, instance, unit = self._orch_with_preempted_unit()
        monkeypatch.chdir(tmp_path)

        def fake_ssh(_instance: CloudInstance, cmd: str, *, timeout: int = 30) -> tuple[int, str]:
            del timeout
            if "worker.log" in cmd and "exitcode" not in cmd:
                return 0, "Starting production\nCUDA error: driver initialization failed\n"
            if "worker.exitcode" in cmd:
                return 0, "1"
            if "dmesg" in cmd:
                return 0, "[12345.678] nvidia-smi: GPU has fallen off the bus"
            return 1, ""

        with patch("vastai_gpu_runner.ssh.ssh_cmd", side_effect=fake_ssh):
            orch._check_unit(runner, instance, unit)

        diag_dir = tmp_path / "batch_diagnostics"
        assert diag_dir.is_dir()
        logs = list(diag_dir.glob("u1_i1_*.log"))
        assert len(logs) == 1
        content = logs[0].read_text()
        assert "worker.log" in content
        assert "CUDA error" in content
        assert "exitcode" in content
        assert "GPU has fallen off the bus" in content

    def test_successful_completion_does_not_capture(self) -> None:
        """Happy path (complete=True) must never trigger diagnostic capture."""
        unit = FakeUnit(key="u1", status="deployed", instance_id="i1", collect_result=True)
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="test",
        )
        orch.capture_preempt_diagnostics = MagicMock()  # type: ignore[method-assign]
        runner = MagicMock(spec=CloudRunner)
        runner.check_progress = MagicMock(return_value={"complete": True, "running": False})
        runner.destroy_instance = MagicMock(return_value=True)
        instance = CloudInstance(instance_id="i1")
        orch._live_runners[unit.key] = (runner, instance, unit)

        orch._check_unit(runner, instance, unit)

        orch.capture_preempt_diagnostics.assert_not_called()

    def test_poll_cycle_once_also_captures_on_preempted(self) -> None:
        """The parallel-poll path (``_poll_cycle_once``) must also capture."""
        orch, runner, instance, unit = self._orch_with_preempted_unit()
        capture_mock = MagicMock()
        orch.capture_preempt_diagnostics = capture_mock  # type: ignore[method-assign]

        orch._poll_cycle_once()

        capture_mock.assert_called_once_with(runner, instance, unit)
        runner.destroy_instance.assert_called_once()


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

        orch.run()

        assert not orch._live_runners

    def test_run_deploy_failure_propagates_to_failed_state(self) -> None:
        unit = FakeUnit(key="u1")
        orch = FakeOrchestrator(
            units=[unit],
            runner_factory=_mock_runner_factory(deploy_result=_fail_deploy("boom")),
            label_prefix="test-fail",
        )

        orch.run()

        assert unit.status == "failed"
        assert "boom" in unit.failure_reason


# ---------------------------------------------------------------------------
# Zombie sweep delegation
# ---------------------------------------------------------------------------


class TestZombieSweep:
    """The v4 sweep delegates to ``cleanup_policy`` end-to-end.

    Each test pins a specific aspect of the policy-driven contract:
    label-scope filtering (delimiter safety), tracked-ID exclusion,
    per-candidate destroy dispatch, severity-by-outcome logging,
    continue-on-exception behaviour, and provider decoupling (no
    module-level provider imports).
    """

    @staticmethod
    def _candidate(iid: str, label: str, state: str = "running") -> InstanceCandidate:
        return InstanceCandidate(
            provider=Provider.VASTAI,
            instance_id=iid,
            label=label,
            state=state,
            image_uuid="img-uuid",
            gpu_model="RTX 4090",
            cost_per_hour=0.4,
            started_at=0.0,
        )

    def test_sweep_calls_list_instances_once(self) -> None:
        """policy.list_instances() is called exactly once per sweep."""
        policy = _recording_cleanup_policy(candidates=[])
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="prod",
            cleanup_policy=policy,
        )
        orch._sweep_zombies()
        assert policy._test_list_invocations() == 1  # type: ignore[attr-defined]

    def test_sweep_filters_by_delimited_scope(self) -> None:
        """Only candidates with ``label.startswith(f"{label_prefix}-")`` are destroyed.

        Adjacent scopes like ``f"{label_prefix}evil"`` cannot match.
        """
        keep = self._candidate("i1", "prod-3f9a1b2c4d5e")
        evil = self._candidate("i2", "prodevil-abcdef012345")
        sibling = self._candidate("i3", "other-abcdef012345")
        policy = _recording_cleanup_policy(candidates=[keep, evil, sibling])
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="prod",
            cleanup_policy=policy,
        )

        killed = orch._sweep_zombies()

        destroyed_ids = {
            c.instance_id
            for c in policy._test_destroy_invocations()  # type: ignore[attr-defined]
        }
        assert destroyed_ids == {"i1"}
        assert killed == 1

    def test_sweep_excludes_tracked_instance_ids(self) -> None:
        """Candidates whose ``instance_id`` is in ``_live_runners`` are skipped."""
        tracked = self._candidate("i1", "prod-3f9a1b2c4d5e")
        orphan = self._candidate("i2", "prod-abcdef012345")
        policy = _recording_cleanup_policy(candidates=[tracked, orphan])
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="prod",
            cleanup_policy=policy,
        )
        runner = MagicMock(spec=CloudRunner)
        instance = CloudInstance(instance_id="i1")
        orch._live_runners["u1"] = (runner, instance, FakeUnit(key="u1"))

        killed = orch._sweep_zombies()

        destroyed_ids = {
            c.instance_id
            for c in policy._test_destroy_invocations()  # type: ignore[attr-defined]
        }
        assert destroyed_ids == {"i2"}
        assert killed == 1

    def test_sweep_counts_only_destroyed(self) -> None:
        """``ALREADY_GONE`` is not counted as a kill (no destroy happened)."""
        c1 = self._candidate("i1", "prod-3f9a1b2c4d5e")
        c2 = self._candidate("i2", "prod-abcdef012345")
        c3 = self._candidate("i3", "prod-123456789abc")
        policy = _recording_cleanup_policy(
            candidates=[c1, c2, c3],
            destroy_responses={
                "i1": CleanupResult(verdict=CleanupVerdict.DESTROYED),
                "i2": CleanupResult(verdict=CleanupVerdict.ALREADY_GONE),
                "i3": CleanupResult(verdict=CleanupVerdict.DESTROYED),
            },
        )
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="prod",
            cleanup_policy=policy,
        )
        killed = orch._sweep_zombies()
        assert killed == 2

    def test_sweep_logs_leaked_at_error(self, caplog: pytest.LogCaptureFixture) -> None:
        c1 = self._candidate("i1", "prod-3f9a1b2c4d5e")
        policy = _recording_cleanup_policy(
            candidates=[c1],
            destroy_responses={
                "i1": CleanupResult(
                    verdict=CleanupVerdict.LEAKED,
                    error="API still has it after destroy",
                ),
            },
        )
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="prod",
            cleanup_policy=policy,
        )
        with caplog.at_level(logging.ERROR, logger="vastai_gpu_runner.batch"):
            orch._sweep_zombies()
        leaked_records = [r for r in caplog.records if "LEAKED" in r.getMessage()]
        assert leaked_records
        assert leaked_records[0].levelno == logging.ERROR

    def test_sweep_logs_unknown_at_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        c1 = self._candidate("i1", "prod-3f9a1b2c4d5e")
        policy = _recording_cleanup_policy(
            candidates=[c1],
            destroy_responses={
                "i1": CleanupResult(
                    verdict=CleanupVerdict.UNKNOWN,
                    error="boom",
                ),
            },
        )
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="prod",
            cleanup_policy=policy,
        )
        with caplog.at_level(logging.WARNING, logger="vastai_gpu_runner.batch"):
            orch._sweep_zombies()
        unknown_records = [r for r in caplog.records if "UNKNOWN" in r.getMessage()]
        assert unknown_records
        assert unknown_records[0].levelno == logging.WARNING

    def test_sweep_logs_cli_attempted_at_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        c1 = self._candidate("i1", "prod-3f9a1b2c4d5e")
        policy = _recording_cleanup_policy(
            candidates=[c1],
            destroy_responses={
                "i1": CleanupResult(
                    verdict=CleanupVerdict.CLI_ATTEMPTED,
                    error="vastai destroy instance i1 — exit 0",
                ),
            },
        )
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="prod",
            cleanup_policy=policy,
        )
        with caplog.at_level(logging.WARNING, logger="vastai_gpu_runner.batch"):
            orch._sweep_zombies()
        records = [r for r in caplog.records if "CLI fallback attempted" in r.getMessage()]
        assert records
        assert records[0].levelno == logging.WARNING

    def test_sweep_logs_already_gone_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        c1 = self._candidate("i1", "prod-3f9a1b2c4d5e")
        policy = _recording_cleanup_policy(
            candidates=[c1],
            destroy_responses={
                "i1": CleanupResult(verdict=CleanupVerdict.ALREADY_GONE),
            },
        )
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="prod",
            cleanup_policy=policy,
        )
        with caplog.at_level(logging.INFO, logger="vastai_gpu_runner.batch"):
            orch._sweep_zombies()
        records = [r for r in caplog.records if "already gone" in r.getMessage()]
        assert records
        assert records[0].levelno == logging.INFO

    def test_sweep_logs_credentials_disabled_at_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        c1 = self._candidate("i1", "prod-3f9a1b2c4d5e")
        policy = _recording_cleanup_policy(
            candidates=[c1],
            destroy_responses={
                "i1": CleanupResult(
                    refusal=CleanupRefusal.CREDENTIALS_DISABLED,
                    error="VASTAI_API_KEY=''",
                ),
            },
        )
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="prod",
            cleanup_policy=policy,
        )
        with caplog.at_level(logging.WARNING, logger="vastai_gpu_runner.batch"):
            orch._sweep_zombies()
        records = [r for r in caplog.records if "credentials disabled" in r.getMessage()]
        assert records
        assert records[0].levelno == logging.WARNING

    def test_sweep_logs_unexpected_no_credentials_at_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        c1 = self._candidate("i1", "prod-3f9a1b2c4d5e")
        policy = _recording_cleanup_policy(
            candidates=[c1],
            destroy_responses={
                "i1": CleanupResult(
                    refusal=CleanupRefusal.NO_CREDENTIALS,
                    error="bypass",
                ),
            },
        )
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="prod",
            cleanup_policy=policy,
        )
        with caplog.at_level(logging.WARNING, logger="vastai_gpu_runner.batch"):
            orch._sweep_zombies()
        records = [r for r in caplog.records if "NO_CREDENTIALS" in r.getMessage()]
        assert records
        assert records[0].levelno == logging.WARNING

    def test_sweep_logs_ownership_refusal_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        c1 = self._candidate("i1", "prod-3f9a1b2c4d5e")
        policy = _recording_cleanup_policy(
            candidates=[c1],
            destroy_responses={
                "i1": CleanupResult(
                    refusal=CleanupRefusal.OWNERSHIP,
                    error="image mismatch",
                ),
            },
        )
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="prod",
            cleanup_policy=policy,
        )
        with caplog.at_level(logging.INFO, logger="vastai_gpu_runner.batch"):
            orch._sweep_zombies()
        records = [r for r in caplog.records if "refused (ownership)" in r.getMessage()]
        assert records
        assert records[0].levelno == logging.INFO

    def test_sweep_logs_unrecognized_outcome_at_error(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        c1 = self._candidate("i1", "prod-3f9a1b2c4d5e")
        # Defensive branch: construct a CleanupResult with a verdict
        # value that's not a CleanupVerdict enum member. The orchestrator
        # must log this at ERROR.
        bogus_result = object.__new__(CleanupResult)
        bogus_result.__dict__.update(
            verdict="totally-unrecognized",
            refusal=None,
            error="?",
        )
        policy = _recording_cleanup_policy(
            candidates=[c1],
            destroy_responses={"i1": bogus_result},
        )
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="prod",
            cleanup_policy=policy,
        )
        with caplog.at_level(logging.ERROR, logger="vastai_gpu_runner.batch"):
            orch._sweep_zombies()
        records = [r for r in caplog.records if "unrecognized cleanup outcome" in r.getMessage()]
        assert records
        assert records[0].levelno == logging.ERROR

    def test_sweep_continues_on_destroy_fn_exception(self) -> None:
        """A raising ``destroy_fn`` is contained by the policy and logged.

        The orchestrator's loop continues: subsequent candidates are still
        processed.
        """

        def _list() -> list[InstanceCandidate]:
            return [
                self._candidate("i1", "prod-3f9a1b2c4d5e"),
                self._candidate("i2", "prod-abcdef012345"),
            ]

        def _destroy(candidate: InstanceCandidate) -> CleanupResult:
            if candidate.instance_id == "i1":
                raise RuntimeError("boom")
            return CleanupResult(verdict=CleanupVerdict.DESTROYED)

        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=_list,
            destroy_fn=_destroy,
        )
        orch = FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="prod",
            cleanup_policy=policy,
        )
        # The policy boundary converts the exception into UNKNOWN; the
        # orchestrator logs WARNING and counts the second DESTROYED.
        killed = orch._sweep_zombies()
        assert killed == 1

    def test_sweep_does_not_import_provider_modules(self) -> None:
        """``batch._sweep_zombies`` source contains no ``providers.*`` references.

        Keeps the orchestrator decoupled from any provider module so
        test fixtures and consumer subclasses can import batch without
        pulling provider SDKs. ``vastai_gpu_runner.cleanup_policy`` is
        a local module — not a provider module — so its import path
        is fine.
        """
        source = inspect.getsource(BatchOrchestrator._sweep_zombies)
        assert "providers." not in source
        assert "destroy_vastai_instance" not in source
        assert "verify_instance_ownership" not in source


# ---------------------------------------------------------------------------
# label_prefix validation
# ---------------------------------------------------------------------------


class TestLabelPrefixValidation:
    """``__init__`` enforces ``validate_label_prefix`` BEFORE any provider call."""

    def test_accepts_non_empty_pre_stripped(self) -> None:
        FakeOrchestrator(
            units=[],
            runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
            label_prefix="prod",
        )

    @pytest.mark.parametrize("bad", ["", " ", "  padded  "])
    def test_rejects_empty_blank_padded(self, bad: str) -> None:
        with pytest.raises(ValueError, match="label_prefix"):
            FakeOrchestrator(
                units=[],
                runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
                label_prefix=bad,
            )

    def test_rejects_none(self) -> None:
        with pytest.raises(ValueError, match="label_prefix"):
            FakeOrchestrator(  # type: ignore[arg-type]
                units=[],
                runner_factory=_mock_runner_factory(deploy_result=_ok_deploy()),
                label_prefix=None,  # type: ignore[arg-type]
            )


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
