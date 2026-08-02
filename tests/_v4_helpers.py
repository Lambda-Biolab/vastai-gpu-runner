# pyright: reportPrivateUsage=warning, reportMissingParameterType=warning
"""Shared test helpers for v4 integration tests.

The ``FakeOrchestrator`` and ``_noop_cleanup_policy`` / ``_recording_cleanup_policy``
helpers are defined here so both ``tests/test_batch.py`` and the
v4 integration tests can share them without circular imports.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

from vastai_gpu_runner.batch import BatchOrchestrator, FailureVerdict
from vastai_gpu_runner.cleanup_policy import (
    CleanupResult,
    CleanupVerdict,
    InstanceCandidate,
    Provider,
    ProviderCleanupPolicy,
)
from vastai_gpu_runner.types import CloudInstance


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
    """Minimal concrete orchestrator for v4 tests.

    All abstract methods are implemented with simple in-memory logic.
    The default cleanup policy is a no-op; v4 integration tests that
    need to drive zombie sweep pass their own ``cleanup_policy``.
    """

    def __init__(
        self,
        units: list[FakeUnit],
        cleanup_policy: ProviderCleanupPolicy | None = None,
        **kwargs: object,
    ) -> None:
        self.units = units
        self.state_saves = 0
        self.payload_builds: list[str] = []
        self.collect_calls: list[str] = []
        super().__init__(  # type: ignore[arg-type]
            cleanup_policy=cleanup_policy or _noop_cleanup_policy(),
            **kwargs,
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


def _noop_cleanup_policy() -> ProviderCleanupPolicy:
    """No-op policy that returns DESTROYED for every candidate."""

    def _list() -> list[InstanceCandidate]:
        return []

    def _destroy(candidate: InstanceCandidate) -> CleanupResult:
        return CleanupResult(verdict=CleanupVerdict.DESTROYED)

    return ProviderCleanupPolicy(
        provider=Provider.VASTAI,
        list_instances_fn=_list,
        destroy_fn=_destroy,
    )


def _mock_runner_factory(
    *,
    deploy_result: object | None = None,
) -> object:
    """Build a factory returning mock CloudRunners with controlled behaviour."""

    from vastai_gpu_runner.types import DeploymentResult

    if deploy_result is None:
        deploy_result = DeploymentResult(
            success=True,
            instance=CloudInstance(
                instance_id="inst-1",
                ssh_host="1.2.3.4",
                ssh_port=22,
                cost_per_hour=0.5,
            ),
        )

    def factory() -> object:
        r = MagicMock()
        r.run_full_cycle = MagicMock(return_value=deploy_result)  # type: ignore[method-assign]
        r.check_progress = MagicMock(  # type: ignore[method-assign]
            return_value={"running": True, "complete": False}
        )
        r.destroy_instance = MagicMock(return_value=True)  # type: ignore[method-assign]
        return r

    return factory
