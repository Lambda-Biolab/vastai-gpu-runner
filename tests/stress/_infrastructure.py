# pyright: reportPrivateUsage=warning, reportMissingParameterType=warning
"""End-to-end stress test infrastructure for the v4 BatchOrchestrator.

These tests use a concrete ``StressOrchestrator`` (a real
``BatchOrchestrator[StressUnit]`` subclass) driving a
``StressCloudRunner`` (a real ``CloudRunner`` subclass) with
configurable failure modes. The lifecycle methods (deploy, poll,
collect, destroy) are exercised against in-memory state machines
that simulate real Vast.ai + SSH behaviour including:

- Transient SSH connection drops (configurable count)
- Worker preemption (worker dies mid-cycle)
- Deploy failures (no GPU offers, boot timeout, GPU verify fail)
- Per-shard random failure (some succeed, some fail fatally)
- Zombie sweep behaviour during a live run

The test orchestrator drives the FULL ``BatchOrchestrator.run()``
lifecycle end-to-end (deploy → poll → collect → cleanup + zombie
sweep) so any regression in the v4 orchestration state machine
surfaces immediately.

The fixtures here are deliberately realistic: the mocks exercise
the v4 lifecycle boundaries the way a real consumer (Boltz-2,
OpenMM) would, not just the contract surface. This means the tests
catch regressions in:

- ``_deploy_phase`` thread-safety + offer pool bookkeeping
- ``_poll_phase`` R2-first + SSH fallback ordering
- ``_collect_phase`` R2 recovery for failed-but-uploaded units
- ``_cleanup_phase`` + zombie sweep severity logging
- ``_sweep_zombies`` policy-driven dispatch with delimited scope
- ``BatchState`` / ``JobBatchState`` schema persistence + resume
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from vastai_gpu_runner.batch import BatchOrchestrator, FailureVerdict
from vastai_gpu_runner.cleanup_policy import (
    CleanupResult,
    CleanupVerdict,
    InstanceCandidate,
    Provider,
    ProviderCleanupPolicy,
)
from vastai_gpu_runner.runner import CloudRunner
from vastai_gpu_runner.state import BatchState, ShardState
from vastai_gpu_runner.types import CloudInstance, DeploymentResult

# ---------------------------------------------------------------------------
# Stress unit (ShardState-mirroring)
# ---------------------------------------------------------------------------


@dataclass
class StressUnit:
    """In-memory unit for stress tests. Mirrors ``ShardState``."""

    shard_id: int
    instance_id: str = ""
    status: str = "pending"
    ssh_host: str = ""
    ssh_port: int = 0
    cost_per_hour: float = 0.0
    retry_count: int = 0
    failure_reason: str = ""
    items_expected: int = 1
    items_completed: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    done_in_r2: bool = False
    # Test-controlled behaviour flag — set after construction.
    outcome: str = "succeed"  # succeed | fail_fatal | preempted | slow

    def __post_init__(self) -> None:
        if self.start_time == 0.0:
            self.start_time = time.time()


# ---------------------------------------------------------------------------
# Concrete BatchOrchestrator subclass
# ---------------------------------------------------------------------------


class StressOrchestrator(BatchOrchestrator[StressUnit]):
    """Real ``BatchOrchestrator[StressUnit]`` for stress scenarios.

    Implements every abstract method with the minimal logic a
    Boltz-2/OpenMM consumer would need: identity hooks, payload
    upload (no-op for stress), instance reconstruction from
    persisted fields, R2-first completion check, and event
    handlers that mutate the in-memory unit state.
    """

    def __init__(
        self,
        state: BatchState,
        *,
        runner_factory: object,
        cleanup_policy: ProviderCleanupPolicy,
        **kwargs: object,
    ) -> None:
        # In-memory mirror of state.shards so the orchestrator sees
        # the same units the consumer's persistence layer would.
        self._units_by_id: dict[int, StressUnit] = {
            s.shard_id: StressUnit(
                shard_id=s.shard_id,
                instance_id=s.instance_id,
                status=s.status,
                ssh_host=s.ssh_host,
                ssh_port=s.ssh_port,
                cost_per_hour=s.cost_per_hour,
                retry_count=s.retry_count,
                failure_reason=s.failure_reason,
                items_expected=s.items_expected,
                items_completed=s.items_completed,
                start_time=s.start_time,
                end_time=s.end_time,
                done_in_r2=False,
                outcome=getattr(s, "outcome", "succeed"),
            )
            for s in state.shards
        }
        # outcome is configured separately by the test after
        # construction. Build a parallel map.
        self._outcomes: dict[int, str] = {}
        super().__init__(
            runner_factory=runner_factory,  # type: ignore[arg-type]
            cleanup_policy=cleanup_policy,
            **kwargs,
        )
        self._state = state
        self._state_saves: int = 0

    # -- iter hooks ---------------------------------------------------------

    def iter_pending_units(self) -> Iterable[StressUnit]:
        return [u for u in self._units_by_id.values() if u.status == "pending"]

    def iter_active_units(self) -> Iterable[StressUnit]:
        return [u for u in self._units_by_id.values() if u.status in ("deployed", "running")]

    def iter_failed_units(self) -> Iterable[StressUnit]:
        return [u for u in self._units_by_id.values() if u.status == "failed"]

    def iter_completed_units(self) -> Iterable[StressUnit]:
        return [u for u in self._units_by_id.values() if u.status == "downloaded"]

    # -- identity / payload -------------------------------------------------

    def unit_key(self, unit: StressUnit) -> str:
        return f"shard-{unit.shard_id}"

    def unit_label(self, unit: StressUnit) -> str:
        return f"shard-{unit.shard_id}"

    def build_unit_payload(self, unit: StressUnit) -> dict[str, Path]:
        # No real payload — the runner ignores the file contents
        # in our test mode (run_full_cycle short-circuits on
        # files=None in stress mode).
        return {}

    def reconstruct_instance(self, unit: StressUnit) -> CloudInstance:
        return CloudInstance(
            instance_id=unit.instance_id,
            ssh_host=unit.ssh_host,
            ssh_port=unit.ssh_port,
            cost_per_hour=unit.cost_per_hour,
        )

    def collect_unit_results(self, unit: StressUnit, instance: CloudInstance) -> bool:
        del instance
        return unit.status not in ("failed", "destroyed")

    def unit_is_done_in_r2(self, unit: StressUnit) -> bool:
        return unit.done_in_r2

    def classify_failure(self, unit: StressUnit, error: str) -> FailureVerdict:
        del unit, error
        return "retry"

    # -- state save ---------------------------------------------------------

    def save_state(self) -> None:
        # Persist in-memory units to BatchState.shards.
        new_shards: list[ShardState] = []
        for u in self._units_by_id.values():
            new_shards.append(
                ShardState(
                    shard_id=u.shard_id,
                    instance_id=u.instance_id,
                    status=u.status,
                    ssh_host=u.ssh_host,
                    ssh_port=u.ssh_port,
                    cost_per_hour=u.cost_per_hour,
                    retry_count=u.retry_count,
                    failure_reason=u.failure_reason,
                    items_expected=u.items_expected,
                    items_completed=u.items_completed,
                    start_time=u.start_time,
                    end_time=u.end_time,
                )
            )
        self._state.shards = new_shards
        self._state.save(self._state_path) if self._state_path else None
        self._state_saves += 1

    def configure_unit_outcome(self, shard_id: int, outcome: str) -> None:
        """Configure the per-shard lifecycle outcome for stress testing."""
        self._outcomes[shard_id] = outcome
        if shard_id in self._units_by_id:
            self._units_by_id[shard_id].outcome = outcome

    @property
    def state_saves(self) -> int:
        return self._state_saves

    def set_state_path(self, path: Path) -> None:
        self._state_path = path

    _state_path: Path | None = None

    # -- event hooks --------------------------------------------------------

    def on_unit_deployed(self, unit: StressUnit, instance: CloudInstance) -> None:
        unit.instance_id = instance.instance_id
        unit.ssh_host = instance.ssh_host
        unit.ssh_port = instance.ssh_port
        unit.cost_per_hour = instance.cost_per_hour
        unit.status = "deployed"
        self.save_state()

    def on_unit_failed(self, unit: StressUnit, reason: str) -> None:
        unit.status = "failed"
        unit.failure_reason = reason
        unit.retry_count += 1
        self.save_state()

    def on_unit_completed(self, unit: StressUnit) -> None:
        unit.status = "downloaded"
        unit.end_time = time.time()
        self.save_state()

    def on_unit_preempted(self, unit: StressUnit) -> None:
        unit.instance_id = ""
        unit.status = "pending"
        unit.retry_count += 1
        self.save_state()


# ---------------------------------------------------------------------------
# Stress CloudRunner
# ---------------------------------------------------------------------------


@dataclass
class StressBehavior:
    """Per-shard behaviour configuration for ``StressCloudRunner``."""

    # Deploy behaviour: when ``deploy_success=False``, ``run_full_cycle``
    # returns ``DeploymentResult(success=False, error=...)`` after the
    # first attempt. When ``deploy_success=False`` AND ``deploy_raises``
    # is set, the runner raises instead (mirrors a real boot timeout
    # that escapes ``run_full_cycle``).
    deploy_success: bool = True
    deploy_error: str = "boot timeout"
    deploy_raises: Exception | None = None
    # SSH failures before the runner stops raising (simulates a flaky
    # network during poll). Each call to ``check_progress`` counts down;
    # 0 = no failures.
    ssh_failures_remaining: int = 0
    # When ``preempt_after_polls > 0``, the runner reports the worker
    # as dead (``worker_dead: True``) after that many successful
    # polls. The orchestrator routes this to ``_handle_preempted_unit``
    # which destroys + marks the unit for retry.
    preempt_after_polls: int = 0
    # Number of polls that report a successful ``complete=True``. The
    # stress orchestrator treats this as the unit's terminal state.
    complete_after_polls: int = 2


class StressCloudRunner(CloudRunner):
    """In-memory ``CloudRunner`` with configurable failure modes.

    Each instance carries a ``StressBehavior`` describing its
    deploy/poll/collect lifecycle. The factory wires this behaviour
    into a ``StressOrchestrator`` via a closure.

    Thread-safe: each method holds a lock to serialise state
    mutations; this lets multiple deploy threads race in
    ``_deploy_phase`` without corrupting the deploy counter.
    """

    _id_counter: int = 0
    _id_lock: threading.Lock = threading.Lock()

    def __init__(self, behavior: StressBehavior) -> None:
        super().__init__()
        self.behavior = behavior
        self.deploy_count: int = 0
        self.poll_count: int = 0
        self.destroy_count: int = 0
        self.complete_count: int = 0
        self._lock = threading.Lock()
        self._instance: CloudInstance | None = None
        with self._id_lock:
            StressCloudRunner._id_counter += 1
            self._runner_id = f"runner-{StressCloudRunner._id_counter}"

    def search_offers(self) -> list[dict[str, object]]:
        """Return a single deterministic cheap offer."""
        # Mimic a real Vast.ai RTX 3060 offer (cheapest available).
        return [
            {
                "id": 99000001,
                "machine_id": 99000001,
                "gpu_name": "RTX 3060",
                "gpu_mem": 12.0,
                "num_gpus": 1,
                "dph_total": 0.05,
                "geolocation": "Test",
                "cuda_max_good": 12.0,
                "inet_up": 1000.0,
                "inet_down": 1000.0,
                "storage_cost": 0.0,
            }
        ]

    def run_full_cycle(
        self,
        files: dict[str, Path],
        local_output_dir: Path,
        *,
        max_retries: int = 3,
        offers: list[dict[str, object]] | None = None,
        used_machine_ids: set[str] | None = None,
        machine_lock: threading.Lock | object | None = None,
    ) -> DeploymentResult:
        """Synchronous in-memory deploy.

        The real ``CloudRunner.run_full_cycle`` invokes boot / verify /
        launch / poll / collect; in stress mode we short-circuit
        the gate chain and return a synthetic ``DeploymentResult``.
        """
        del files, local_output_dir, max_retries, offers, used_machine_ids, machine_lock
        with self._lock:
            self.deploy_count += 1
            if self.behavior.deploy_raises is not None:
                raise self.behavior.deploy_raises
            if not self.behavior.deploy_success:
                return DeploymentResult(
                    success=False,
                    error=self.behavior.deploy_error,
                )
        # Build a fresh CloudInstance. The instance_id is the runner
        # id so the orchestrator's per-unit tracking is unique.
        instance = CloudInstance(
            instance_id=self._runner_id,
            ssh_host=f"{self._runner_id}.test.local",
            ssh_port=22,
            cost_per_hour=0.05,
        )
        with self._lock:
            self._instance = instance
        return DeploymentResult(success=True, instance=instance)

    def check_progress(self, instance: CloudInstance) -> dict[str, object]:
        """In-memory poll that simulates SSH failures + completion.

        Returns ``worker_dead: True`` when the configured
        ``preempt_after_polls`` threshold is reached, so the
        orchestrator's ``decide_next_action`` routes the unit to
        ``_handle_preempted_unit`` (destroy + retry).
        """
        del instance
        with self._lock:
            self.poll_count += 1
            polls_done = self.poll_count
            if self.behavior.ssh_failures_remaining > 0:
                self.behavior.ssh_failures_remaining -= 1
                # Mirror a real ``RuntimeError("ssh: connection reset")``
                # the orchestrator logs and treats as a transient retry.
                raise RuntimeError("ssh: connection reset by peer")
            if (
                self.behavior.preempt_after_polls > 0
                and polls_done >= self.behavior.preempt_after_polls
            ):
                # The orchestrator's ProgressSnapshot reads
                # ``worker_dead`` from the response, so we MUST
                # set it to True for the unit to be preempted.
                return {"worker_dead": True, "complete": False}
            if polls_done >= self.behavior.complete_after_polls:
                self.complete_count += 1
                return {"complete": True}
        return {"running": True, "complete": False}

    def destroy_instance(self, instance: CloudInstance) -> bool:
        del instance
        with self._lock:
            self.destroy_count += 1
        return True

    def list_remote_files(self, instance: CloudInstance) -> list[str]:
        del instance
        return ["result.txt"] if self.complete_count > 0 else []

    def download_file(
        self,
        instance: CloudInstance,
        remote_name: str,
        local_path: Path,
    ) -> bool:
        del instance, remote_name
        # Touch the local file so the orchestrator's collect phase
        # succeeds.
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(f"stress-test-result from {self._runner_id}\n")
        return True


# ---------------------------------------------------------------------------
# Runner factory
# ---------------------------------------------------------------------------


def make_runner_factory(
    behaviors: Sequence[StressBehavior],
) -> object:
    """Build a runner factory that hands out a fresh runner per call.

    Each call to the returned factory creates a runner with the
    next behaviour in the list (cycling when the list is exhausted).
    This lets a stress test distribute failure modes across shards.
    """
    it = iter(behaviors)

    def factory() -> CloudRunner:
        try:
            return StressCloudRunner(next(it))
        except StopIteration:
            return StressCloudRunner(StressBehavior())

    return factory


# ---------------------------------------------------------------------------
# Cleanup-policy factory
# ---------------------------------------------------------------------------


def noop_cleanup_policy() -> ProviderCleanupPolicy:
    """A cleanup policy that returns no candidates.

    Stress tests that don't exercise zombie sweep use this; tests
    that DO exercise zombie sweep construct a real policy from
    `tests._v4_helpers._noop_cleanup_policy` (or use the in-line
    recorder pattern below).
    """

    def _list() -> list[InstanceCandidate]:
        return []

    def _destroy(candidate: InstanceCandidate) -> CleanupResult:
        return CleanupResult(verdict=CleanupVerdict.DESTROYED)

    return ProviderCleanupPolicy(
        provider=Provider.VASTAI,
        list_instances_fn=_list,
        destroy_fn=_destroy,
    )


# ---------------------------------------------------------------------------
# BatchState builder
# ---------------------------------------------------------------------------


def build_batch_state(
    *,
    num_shards: int,
    label_scope: str = "stress-3f9a1b2c4d5e",
    requested_label_prefix: str = "stress",
    state_path: Path | None = None,
) -> tuple[BatchState, Path]:
    """Build a fresh ``BatchState`` with ``num_shards`` pending shards."""
    state = BatchState(
        batch_id="stress",
        label_scope=label_scope,
        requested_label_prefix=requested_label_prefix,
        schema_version=1,
        shards=[
            ShardState(shard_id=i, status="pending", items_expected=1) for i in range(num_shards)
        ],
    )
    if state_path is None:
        state_path = Path("/tmp/stress_state.json")
    state.save(state_path)
    return state, state_path


def load_batch_state_v4(state_path: Path) -> BatchState:
    """Load a state file via the v4 ``load_batch_state`` boundary."""
    from vastai_gpu_runner.state import load_batch_state

    loaded = load_batch_state(state_path, state_cls=BatchState)
    assert loaded is not None
    return loaded


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def write_pre_v4_state_file(
    path: Path,
    *,
    num_shards: int,
    canonical_label: str = "stress-3f9a1b2c4d5e",
    shard_status: str = "deployed",
) -> None:
    """Write a real pre-v4 (schema_version 0) state JSON fixture.

    The ``label`` value follows the canonical
    ``f"{prefix}-<12 lowercase hex>"`` pattern so
    ``_migrate_pre_v4`` recovers the requested prefix + scope.
    """
    fixture = {
        "label": canonical_label,
        "batch_id": "stress",
        "num_gpus": num_shards,
        "created_at": time.time(),
        "shards": [
            {
                "shard_id": i,
                "instance_id": f"legacy-inst-{i}",
                "status": shard_status,
                "items_expected": 1,
                "items_completed": 0,
                "start_time": time.time(),
            }
            for i in range(num_shards)
        ],
    }
    path.write_text(json.dumps(fixture))


def apply_random_outcomes(
    orchestrator: StressOrchestrator,
    *,
    success_rate: float = 0.7,
    preempt_rate: float = 0.15,
    fatal_rate: float = 0.10,
    slow_rate: float = 0.05,
    seed: int | None = None,
) -> None:
    """Assign per-shard outcomes by weighted random draw.

    Used by mixed-failure scenarios. ``success_rate + preempt_rate +
    fatal_rate + slow_rate`` should sum to 1.0.
    """
    # Pseudo-random is fine here — the outcome distribution is a
    # deterministic test fixture, not a security primitive. The
    # deterministic seed (when provided) makes the test
    # reproducible.
    rng = random.Random(seed)  # noqa: S311
    total = success_rate + preempt_rate + fatal_rate + slow_rate
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"outcome rates must sum to 1.0, got {total}")
    for unit in orchestrator._units_by_id.values():
        roll = rng.random()
        if roll < success_rate:
            outcome = "succeed"
        elif roll < success_rate + preempt_rate:
            outcome = "preempted"
        elif roll < success_rate + preempt_rate + fatal_rate:
            outcome = "fail_fatal"
        else:
            outcome = "slow"
        orchestrator.configure_unit_outcome(unit.shard_id, outcome)


# ---------------------------------------------------------------------------
# Behaviour translation
# ---------------------------------------------------------------------------


def behavior_for_outcome(outcome: str, *, poll_interval_seconds: int = 0) -> StressBehavior:
    """Translate a per-shard outcome string into a ``StressBehavior``."""
    if outcome == "succeed":
        # Default: succeeds after 2 polls.
        return StressBehavior(complete_after_polls=2)
    if outcome == "preempted":
        # First poll succeeds, second reports the worker dead.
        return StressBehavior(complete_after_polls=99, preempt_after_polls=2)
    if outcome == "fail_fatal":
        # Deploy fails outright — return a non-success
        # ``DeploymentResult`` (no exception), mirroring a real
        # boot timeout that ``_try_one_offer`` records as a
        # failure but doesn't raise out of ``run_full_cycle``.
        return StressBehavior(deploy_success=False, deploy_error="boot timeout")
    if outcome == "slow":
        # 1 SSH drop, then completes.
        return StressBehavior(ssh_failures_remaining=1, complete_after_polls=3)
    raise ValueError(f"unknown outcome: {outcome}")


def run_behaviors_for(orchestrator: StressOrchestrator) -> list[StressBehavior]:
    """Build a per-shard behaviour list, one entry per pending unit."""
    return [
        behavior_for_outcome(orchestrator._outcomes.get(u.shard_id, "succeed"))
        for u in orchestrator._units_by_id.values()
    ]


# ---------------------------------------------------------------------------
# Convenience assertion helpers
# ---------------------------------------------------------------------------


def count_by_status(orchestrator: StressOrchestrator) -> dict[str, int]:
    """Count units by status; mirrors BatchState properties."""
    counts: dict[str, int] = {}
    for u in orchestrator._units_by_id.values():
        counts[u.status] = counts.get(u.status, 0) + 1
    return counts


def assert_no_thread_leaks(threads_before: set[int]) -> None:
    """Sanity check: every thread spawned during the test has exited.

    The orchestrator's ``_deploy_phase`` uses a
    ``ThreadPoolExecutor``; if a worker thread leaks (e.g. an
    unhandled exception in a future), the test will see its
    ident in ``threading.enumerate()``. This guard catches that
    class of regression.
    """
    live = {t.ident for t in threading.enumerate() if t is not threading.main_thread()}
    leaked = live - threads_before
    assert not leaked, f"thread leak detected: {leaked}"
