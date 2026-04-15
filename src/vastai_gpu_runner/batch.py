"""Batch orchestrator for cloud GPU workloads.

``BatchOrchestrator`` is the template-method ABC that sits one layer above
``CloudRunner``. Where ``CloudRunner`` handles the lifecycle of a *single*
instance (search → create → boot → verify → deploy → launch → poll →
download → destroy), ``BatchOrchestrator`` coordinates the lifecycle of
*many* units in parallel — deploying, polling, classifying failures,
handling instance loss, and resuming from a crash-recovery checkpoint.

A "unit" is either a shard (``ShardState``, N items per GPU) or a job
(``JobState``, 1 item per GPU). Consumers choose the shape by implementing
the ``iter_*`` hook methods over their own batch-state type.

Design principles:

1. **No provider coupling.** The orchestrator talks to a ``CloudRunner``
   factory exclusively. Vast.ai, RunPod, or a mock all work the same.

2. **Consumer owns state.** The orchestrator drives *events*
   (``on_unit_deployed``, ``on_unit_failed``, ``on_unit_completed``,
   ``on_unit_preempted``); the consumer updates its own ``BatchState`` /
   ``JobBatchState`` object and persists it via ``save_state``. Status
   strings, field names, and retry accounting all live in consumer code.

3. **Hooks are narrow.** The orchestrator asks the consumer:
   - What units are pending / active / completed? (``iter_*``)
   - What files does this unit need? (``build_unit_payload``)
   - Can I reconstruct a live instance on resume? (``reconstruct_instance``)
   - Where do I download results? (``collect_unit_results``)
   - Is this unit already done in R2? (``unit_is_done_in_r2``)
   - How do I label it in logs? (``unit_label``)
   - Is this failure retryable? (``classify_failure``)

4. **Drift-free symmetry.** Boltz-2 and OpenMM share exactly the same
   orchestration loop via this class. Bug fixes land once.

Usage sketch::

    class MyOrch(BatchOrchestrator[ShardState]):
        def __init__(self, state: BatchState, ...):
            super().__init__(...)
            self._state = state

        def iter_pending_units(self):
            return list(self._state.pending_shards)

        def on_unit_deployed(self, unit, instance):
            unit.instance_id = instance.instance_id
            unit.ssh_host = instance.ssh_host
            unit.ssh_port = instance.ssh_port
            unit.cost_per_hour = instance.cost_per_hour
            unit.status = "deployed"
            self.save_state()

        # ... etc

The ABC does NOT handle: weight uploads, local GPU hybrid mode, shard input
preparation. Those live in the consumer. The ABC handles only the cloud
lifecycle loop.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeVar

from vastai_gpu_runner.orchestrator import check_budget, sweep_zombie_instances

if TYPE_CHECKING:
    from pathlib import Path

    from vastai_gpu_runner.runner import CloudRunner
    from vastai_gpu_runner.storage.r2 import R2Sink
    from vastai_gpu_runner.types import CloudInstance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class BatchUnit(Protocol):
    """Structural type for one unit of work in a batch.

    Both ``state.ShardState`` and ``state.JobState`` satisfy this protocol.
    """

    instance_id: str
    ssh_host: str
    ssh_port: int
    cost_per_hour: float
    status: str
    retry_count: int


RunnerFactory = Callable[[], "CloudRunner"]
"""Factory returning a fresh ``CloudRunner`` for one deploy attempt."""


UnitT = TypeVar("UnitT", bound=BatchUnit)

FailureVerdict = Literal["retry", "fatal"]


# ---------------------------------------------------------------------------
# BatchOrchestrator ABC
# ---------------------------------------------------------------------------


class BatchOrchestrator(ABC, Generic[UnitT]):
    """Abstract batch orchestrator for cloud GPU workloads.

    See module docstring for design overview. Generic over unit type
    (``ShardState`` or ``JobState``).
    """

    def __init__(
        self,
        *,
        runner_factory: RunnerFactory,
        label_prefix: str,
        workspace_dir: str = "/workspace",
        r2_sink: R2Sink | None = None,
        r2_batch_id: str = "",
        budget_usd: float = 0.0,
        max_retries: int = 2,
        max_parallel_deploys: int = 16,
        poll_interval_seconds: int = 30,
        zombie_sweep_every_n_cycles: int = 5,
        poll_timeout_seconds: float = 0.0,
    ) -> None:
        """Initialise orchestrator state. See class docstring for argument meanings."""
        self._runner_factory = runner_factory
        self._label_prefix = label_prefix
        self._workspace_dir = workspace_dir
        self._r2_sink = r2_sink
        self._r2_batch_id = r2_batch_id
        self._budget_usd = budget_usd
        self._max_retries = max_retries
        self._max_parallel_deploys = max_parallel_deploys
        self._poll_interval_seconds = poll_interval_seconds
        self._zombie_sweep_every_n_cycles = zombie_sweep_every_n_cycles
        self._poll_timeout_seconds = poll_timeout_seconds

        # Live runner registry: unit_key → (runner, instance, unit)
        self._live_runners: dict[str, tuple[CloudRunner, CloudInstance, UnitT]] = {}
        self._state_lock = threading.Lock()
        self._used_machine_ids: set[str] = set()

    # -- Domain hooks (subclasses override) --------------------------------

    @abstractmethod
    def iter_pending_units(self) -> Iterable[UnitT]:
        """Units that have not yet been deployed (status == pending)."""

    @abstractmethod
    def iter_active_units(self) -> Iterable[UnitT]:
        """Units that are currently deployed or running (need polling)."""

    @abstractmethod
    def iter_failed_units(self) -> Iterable[UnitT]:
        """Units that failed and may be eligible for re-deploy."""

    @abstractmethod
    def iter_completed_units(self) -> Iterable[UnitT]:
        """Units whose results have been downloaded (terminal, skip)."""

    @abstractmethod
    def save_state(self) -> None:
        """Persist batch state atomically. Called after every event."""

    @abstractmethod
    def unit_key(self, unit: UnitT) -> str:
        """Stable unique identifier for this unit (shard_id, job_name, ...)."""

    @abstractmethod
    def unit_label(self, unit: UnitT) -> str:
        """Human-readable identifier for logs."""

    @abstractmethod
    def build_unit_payload(self, unit: UnitT) -> dict[str, Path]:
        """Files to upload for this unit. Maps remote relative path → local path."""

    @abstractmethod
    def reconstruct_instance(self, unit: UnitT) -> CloudInstance:
        """Rebuild a ``CloudInstance`` from persisted unit fields (for resume)."""

    @abstractmethod
    def collect_unit_results(self, unit: UnitT, instance: CloudInstance) -> bool:
        """Download artifacts for one completed unit. True on success."""

    @abstractmethod
    def unit_is_done_in_r2(self, unit: UnitT) -> bool:
        """Whether this unit's results already exist in R2. False if no R2 sink."""

    @abstractmethod
    def classify_failure(self, unit: UnitT, error: str) -> FailureVerdict:
        """Classify a failure as retryable or fatal."""

    # -- State-mutation events (subclasses override) -----------------------

    @abstractmethod
    def on_unit_deployed(self, unit: UnitT, instance: CloudInstance) -> None:
        """Consumer: set instance_id/ssh_*/cost_per_hour/status='deployed'; save_state."""

    @abstractmethod
    def on_unit_failed(self, unit: UnitT, reason: str) -> None:
        """Consumer: set status='failed', failure_reason, bump retry_count; save_state."""

    @abstractmethod
    def on_unit_completed(self, unit: UnitT) -> None:
        """Consumer: set status='downloaded' after collect_unit_results succeeded; save_state."""

    @abstractmethod
    def on_unit_preempted(self, unit: UnitT) -> None:
        """Consumer: reset instance_id='', status='pending' for redeploy; save_state."""

    # -- Concrete lifecycle (inherited) ------------------------------------

    def run(self) -> None:
        """Run the batch end-to-end: resume → deploy → poll → collect → cleanup."""
        logger.info("BatchOrchestrator: starting batch %s", self._label_prefix)
        self._resume_from_state()
        self._deploy_phase()
        self._sweep_zombies()
        self._poll_phase()
        self._collect_phase()
        self._cleanup_phase()
        logger.info("BatchOrchestrator: batch %s complete", self._label_prefix)

    # -- Phase 1: resume ---------------------------------------------------

    def _resume_from_state(self) -> None:
        """Reconstruct live-runner map from already-active units."""
        for unit in self.iter_active_units():
            if not unit.instance_id:
                continue
            try:
                instance = self.reconstruct_instance(unit)
                runner = self._runner_factory()
                self._live_runners[self.unit_key(unit)] = (runner, instance, unit)
                logger.info(
                    "Resume: reconnected to %s on instance %s",
                    self.unit_label(unit),
                    unit.instance_id,
                )
            except Exception as exc:
                logger.warning(
                    "Resume: failed to reconnect %s: %s — will re-deploy",
                    self.unit_label(unit),
                    exc,
                )
                self.on_unit_preempted(unit)

    # -- Phase 2: deploy ---------------------------------------------------

    def _deploy_phase(self) -> None:
        """Parallel deploy of pending units via ThreadPoolExecutor."""
        pending = [u for u in self.iter_pending_units() if not self.unit_is_done_in_r2(u)]
        if not pending:
            logger.info("Deploy phase: no pending units")
            return

        if self._budget_usd > 0 and not check_budget(0.0, self._budget_usd):
            logger.error("Deploy phase: budget exceeded before deploy")
            for unit in pending:
                self.on_unit_failed(unit, "budget exceeded")
            return

        max_workers = min(self._max_parallel_deploys, len(pending))
        logger.info(
            "Deploy phase: %d unit(s) across %d worker(s)",
            len(pending),
            max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self._deploy_one, u): u for u in pending}
            for fut in as_completed(futures):
                unit = futures[fut]
                try:
                    ok = fut.result()
                except Exception as exc:
                    logger.exception("Deploy %s raised: %s", self.unit_label(unit), exc)
                    with self._state_lock:
                        self.on_unit_failed(unit, f"deploy exception: {exc}")
                    continue
                if ok:
                    logger.info("Deploy %s: success", self.unit_label(unit))
                else:
                    logger.warning("Deploy %s: failed", self.unit_label(unit))

    def _deploy_one(self, unit: UnitT) -> bool:
        """Deploy a single unit. Thread-safe via ``self._state_lock``."""
        runner = self._runner_factory()
        files = self.build_unit_payload(unit)

        try:
            # CloudRunner.run_full_cycle does boot, verify, deploy, setup, launch.
            result = runner.run_full_cycle(
                files=files,
                local_output_dir=self._workspace_local_for(unit),
                max_retries=3,
                used_machine_ids=self._used_machine_ids,
                machine_lock=self._state_lock,
            )
        except Exception as exc:
            with self._state_lock:
                self.on_unit_failed(unit, f"run_full_cycle exception: {exc}")
            return False

        with self._state_lock:
            if result.success and result.instance is not None:
                self._live_runners[self.unit_key(unit)] = (runner, result.instance, unit)
                self.on_unit_deployed(unit, result.instance)
                return True
            self.on_unit_failed(unit, result.error or "deploy failed")
            return False

    def _workspace_local_for(self, unit: UnitT) -> Path:
        """Default local workspace path. Override in subclass for custom layout."""
        from pathlib import Path as _Path

        return _Path.cwd() / "outputs" / self.unit_label(unit)

    # -- Phase 3: poll -----------------------------------------------------

    def _poll_phase(self) -> None:
        """Poll until all live units terminal. Exponential backoff + R2-first."""
        if not self._live_runners:
            logger.info("Poll phase: no live runners")
            return

        base = self._poll_interval_seconds
        cur_interval = min(base, 5)
        max_interval = min(base * 2, 60)
        deadline = (
            time.time() + self._poll_timeout_seconds
            if self._poll_timeout_seconds > 0
            else float("inf")
        )
        cycle = 0

        while self._live_runners and time.time() < deadline:
            cycle += 1

            if cycle % self._zombie_sweep_every_n_cycles == 0:
                self._sweep_zombies()

            if self._budget_usd > 0 and not check_budget(
                self._estimate_current_spend(),
                self._budget_usd,
            ):
                logger.error("Poll phase: budget exceeded — aborting")
                break

            any_made_progress = False
            for unit_key in list(self._live_runners.keys()):
                entry = self._live_runners.get(unit_key)
                if entry is None:
                    continue
                runner, instance, unit = entry

                verdict = self._check_unit(runner, instance, unit)
                if verdict == "completed":
                    any_made_progress = True
                elif verdict == "preempted":
                    any_made_progress = True

            if any_made_progress:
                cur_interval = min(base, 5)
            else:
                time.sleep(cur_interval)
                cur_interval = min(cur_interval * 2, max_interval)

        if time.time() >= deadline and self._live_runners:
            logger.warning(
                "Poll phase: timeout after %.0fs with %d live units",
                self._poll_timeout_seconds,
                len(self._live_runners),
            )

    def _check_unit(
        self,
        runner: CloudRunner,
        instance: CloudInstance,
        unit: UnitT,
    ) -> Literal["completed", "running", "preempted", "failed"]:
        """Single-unit poll cycle. R2-first, then SSH fallback + rescue."""
        unit_key = self.unit_key(unit)

        # Layer 1: R2 DONE marker (works even if SSH is flaky)
        if self.unit_is_done_in_r2(unit):
            return self._finalise_completed(runner, instance, unit, unit_key)

        # Layer 2: SSH progress check
        try:
            progress = runner.check_progress(instance)
        except Exception as exc:
            logger.warning(
                "Poll %s: check_progress raised %s — treating as running",
                self.unit_label(unit),
                exc,
            )
            return "running"

        if progress.get("complete"):
            return self._finalise_completed(runner, instance, unit, unit_key)

        # Layer 3: silent worker crash detection
        if progress.get("worker_dead"):
            # Re-check R2 once more — worker may have uploaded results
            # between the check above and now.
            if self.unit_is_done_in_r2(unit):
                return self._finalise_completed(runner, instance, unit, unit_key)
            logger.warning(
                "Poll %s: worker dead on %s — handling as instance loss",
                self.unit_label(unit),
                unit.instance_id,
            )
            with contextlib.suppress(Exception):
                runner.destroy_instance(instance)
            with self._state_lock:
                self._handle_instance_loss(unit, unit_key, "worker died silently")
            return "preempted"

        return "running"

    def _finalise_completed(
        self,
        runner: CloudRunner,
        instance: CloudInstance,
        unit: UnitT,
        unit_key: str,
    ) -> Literal["completed", "failed"]:
        """Download results, call on_unit_completed, destroy instance."""
        ok = False
        try:
            ok = self.collect_unit_results(unit, instance)
        except Exception as exc:
            logger.exception(
                "Collect %s raised: %s",
                self.unit_label(unit),
                exc,
            )
        with self._state_lock:
            if ok:
                self.on_unit_completed(unit)
            else:
                self.on_unit_failed(unit, "collect_unit_results failed")
            self._live_runners.pop(unit_key, None)
        with contextlib.suppress(Exception):
            runner.destroy_instance(instance)
        return "completed" if ok else "failed"

    # -- Phase 4: collect any stragglers (R2 recovery for failed units) ----

    def _collect_phase(self) -> None:
        """After poll, try R2 recovery for units that failed with uploaded results."""
        if self._r2_sink is None:
            return
        for unit in list(self.iter_failed_units()):
            if not self.unit_is_done_in_r2(unit):
                continue
            logger.info(
                "R2 recovery: %s is done in R2 — collecting",
                self.unit_label(unit),
            )
            try:
                instance = self.reconstruct_instance(unit)
            except Exception:
                # No live instance — consumer's collect_unit_results must
                # handle this path (e.g. download from R2 directly).
                from vastai_gpu_runner.types import CloudInstance as _CloudInstance

                instance = _CloudInstance()
            try:
                if self.collect_unit_results(unit, instance):
                    with self._state_lock:
                        self.on_unit_completed(unit)
            except Exception as exc:
                logger.warning(
                    "R2 recovery collect failed for %s: %s",
                    self.unit_label(unit),
                    exc,
                )

    # -- Phase 5: cleanup --------------------------------------------------

    def _cleanup_phase(self) -> None:
        """Destroy any leftover live instances. Final zombie sweep."""
        for runner, instance, unit in list(self._live_runners.values()):
            logger.warning(
                "Cleanup: destroying leftover instance for %s",
                self.unit_label(unit),
            )
            with contextlib.suppress(Exception):
                runner.destroy_instance(instance)
        self._live_runners.clear()
        self._sweep_zombies()

    # -- Instance loss handling --------------------------------------------

    def _handle_instance_loss(
        self,
        unit: UnitT,
        unit_key: str,
        reason: str,
    ) -> bool:
        """Mark unit preempted and remove from live map. Caller must hold lock.

        Returns True if the unit is eligible for re-deploy (retry_count
        under cap), False if it should be marked fatally failed. The
        actual re-deploy happens in the next deploy_phase iteration
        (not implemented as an in-poll redeploy in this ABC — the
        consumer can call ``_deploy_phase`` again if desired).
        """
        self._live_runners.pop(unit_key, None)
        if unit.retry_count >= self._max_retries:
            self.on_unit_failed(unit, f"{reason} (retries exhausted)")
            return False
        verdict = self.classify_failure(unit, reason)
        if verdict == "fatal":
            self.on_unit_failed(unit, f"{reason} (fatal)")
            return False
        self.on_unit_preempted(unit)
        return True

    # -- Zombie sweep + budget --------------------------------------------

    def _sweep_zombies(self) -> int:
        """Destroy Vast.ai instances not tracked by live_runners."""
        with self._state_lock:
            live_map: dict[int, tuple[CloudRunner, CloudInstance]] = {
                i: (entry[0], entry[1]) for i, entry in enumerate(self._live_runners.values())
            }
        try:
            return sweep_zombie_instances(
                live_map,
                label_prefix=self._label_prefix,
                r2_sink=self._r2_sink,
                r2_batch_id=self._r2_batch_id,
            )
        except Exception as exc:
            logger.warning("Zombie sweep failed: %s", exc)
            return 0

    def _estimate_current_spend(self) -> float:
        """Estimate current batch spend. Override for provider-specific tracking."""
        # Default: no spend tracking. Subclasses can override to sum
        # (now - start_time) * cost_per_hour across all units.
        return 0.0
