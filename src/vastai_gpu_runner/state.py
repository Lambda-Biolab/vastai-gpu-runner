"""Persistent batch state classes for cloud orchestration.

These dataclasses are serialized to JSON for crash recovery. The orchestrator
writes state after each lifecycle phase; a new process can resume by reading
the state file — skipping downloaded shards, re-polling active ones, and
re-deploying failed ones.

Two models:
- ``ShardState`` / ``BatchState``: N items split across M shards (1 shard = 1 GPU).
  Used for batch prediction workloads.
- ``JobState`` / ``JobBatchState``: 1 job = 1 instance (no sharding).
  Used for long-running single-GPU workloads like MD simulation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

MAX_SHARD_RETRIES = 2  # Max re-deploys per shard on preemption


# ---------------------------------------------------------------------------
# Shard-based batch state (N items → M shards → M GPUs)
# ---------------------------------------------------------------------------


@dataclass
class ShardState:
    """Serialisable state for one cloud shard.

    Status flow: ``pending`` -> ``deployed`` -> ``running`` ->
    ``downloaded`` -> ``destroyed`` | ``failed``
    """

    shard_id: int
    instance_id: str = ""
    provider: str = ""
    ssh_host: str = ""
    ssh_port: int = 0
    cost_per_hour: float = 0.0
    status: str = "pending"
    items_expected: int = 0
    items_completed: int = 0
    item_ids: list[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    failure_reason: str = ""
    retry_count: int = 0


@dataclass
class BatchState:
    """Persistent batch state written to disk after each lifecycle phase.

    Enables resume-on-crash: a new orchestrator reads this file, skips
    ``downloaded`` shards, polls ``deployed``/``running`` shards, and
    re-deploys ``failed`` shards.
    """

    batch_id: str = ""
    label: str = ""
    num_gpus: int = 0
    shards: list[ShardState] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Atomically write state to disk (write tmp + rename)."""
        self.updated_at = time.time()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> BatchState:
        """Load state from disk."""
        data = json.loads(path.read_text())
        shards = [ShardState(**s) for s in data.pop("shards", [])]
        return cls(**data, shards=shards)

    @property
    def active_shards(self) -> list[ShardState]:
        """Shards that are deployed or running (need polling)."""
        return [s for s in self.shards if s.status in ("deployed", "running")]

    @property
    def failed_shards(self) -> list[ShardState]:
        """Shards that failed and can be re-deployed."""
        return [s for s in self.shards if s.status == "failed"]

    @property
    def downloaded_shards(self) -> list[ShardState]:
        """Shards whose results have been downloaded."""
        return [s for s in self.shards if s.status in ("downloaded", "destroyed")]

    @property
    def pending_shards(self) -> list[ShardState]:
        """Shards not yet deployed."""
        return [s for s in self.shards if s.status == "pending"]


# ---------------------------------------------------------------------------
# Job-based batch state (1 job = 1 instance)
# ---------------------------------------------------------------------------


@dataclass
class JobState:
    """State for one cloud job (1 job = 1 GPU instance).

    Status flow: ``pending`` -> ``deploying`` -> ``running`` ->
    ``completed`` -> ``downloaded`` | ``failed``
    """

    job_name: str
    status: str = "pending"
    instance_id: str = ""
    ssh_host: str = ""
    ssh_port: int = 0
    machine_id: str = ""
    error: str = ""
    submit_time: str = ""
    complete_time: str = ""
    cost_per_hour: float = 0.0
    retry_count: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        """Estimate cost based on elapsed time."""
        if not self.submit_time:
            return 0.0
        start = datetime.fromisoformat(self.submit_time)
        end = (
            datetime.fromisoformat(self.complete_time)
            if self.complete_time
            else datetime.now(tz=UTC)
        )
        hours = (end - start).total_seconds() / 3600
        return hours * self.cost_per_hour


@dataclass
class JobBatchState:
    """Persistent batch state for job-based workloads."""

    batch_id: str = ""
    jobs: list[JobState] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save state atomically (tmp + rename)."""
        self.updated_at = datetime.now(tz=UTC).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> JobBatchState:
        """Load state from JSON."""
        data = json.loads(path.read_text())
        jobs = [JobState(**j) for j in data.pop("jobs", [])]
        state = cls(**data)
        state.jobs = jobs
        return state

    @property
    def pending_jobs(self) -> list[JobState]:
        """Jobs not yet deployed."""
        return [j for j in self.jobs if j.status == "pending"]

    @property
    def active_jobs(self) -> list[JobState]:
        """Jobs that are deploying or running."""
        return [j for j in self.jobs if j.status in ("deploying", "running")]

    @property
    def completed_jobs(self) -> list[JobState]:
        """Jobs completed but not downloaded."""
        return [j for j in self.jobs if j.status == "completed"]

    @property
    def total_cost(self) -> float:
        """Total estimated spend."""
        return sum(j.cost_usd for j in self.jobs)
