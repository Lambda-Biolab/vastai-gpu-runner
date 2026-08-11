"""Provider-neutral managed-job types and ABC.

A managed-job provider is a cloud service that owns the underlying
VM lifecycle (GCP Batch, AWS Batch, Azure Batch, etc.). Consumers
submit a :class:`ManagedJobSpec` describing the desired unit of work,
receive a :class:`ManagedJobHandle` with the cloud resource name, and
poll :meth:`ManagedJobRunner.get_status` until the job reaches a
terminal state. :meth:`ManagedJobRunner.list_tasks` exposes per-task
diagnostics when the provider supports it.

The ABC is intentionally minimal — provider-specific fields live on
subclasses — so a fake implementation can be used in tests without
touching the real cloud SDKs.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class ManagedJobTerminalState(enum.Enum):
    """Normalized terminal states for managed-job providers."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


def _empty_str_tuple() -> tuple[str, ...]:
    return ()


@dataclass(frozen=True)
class ManagedJobSpec:
    """Provider-neutral description of a unit of managed-job work.

    Providers map this onto their native job representation. The
    ``labels`` mapping is opaque to the provider-neutral layer; it is
    carried through to status, log, and state payloads so consumers
    can correlate jobs across stages.
    """

    name: str
    task_count: int = 1
    parallelism: int = 1
    image: str = ""
    command: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)
    gcs_mounts: tuple[str, ...] = ()
    timeout_seconds: int | None = None
    retry_on_preempt: bool = True
    region: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize the spec into a JSON-safe dictionary."""
        return {
            "name": self.name,
            "task_count": self.task_count,
            "parallelism": self.parallelism,
            "image": self.image,
            "command": list(self.command),
            "environment": dict(self.environment),
            "labels": dict(self.labels),
            "gcs_mounts": list(self.gcs_mounts),
            "timeout_seconds": self.timeout_seconds,
            "retry_on_preempt": self.retry_on_preempt,
            "region": self.region,
        }


@dataclass(frozen=True)
class ManagedJobHandle:
    """Opaque handle to a managed-job instance returned by the provider."""

    provider: str
    resource_name: str
    location: str = ""


@dataclass(frozen=True)
class ManagedTaskStatus:
    """Per-task status snapshot returned by :meth:`ManagedJobRunner.list_tasks`."""

    task_index: int
    state: str
    exit_code: int | None = None
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize the task status to a JSON-safe dictionary."""
        return {
            "task_index": self.task_index,
            "state": self.state,
            "exit_code": self.exit_code,
            "message": self.message,
        }


@dataclass(frozen=True)
class ManagedJobStatus:
    """Snapshot of a managed-job's state at a point in time."""

    handle: ManagedJobHandle
    state: ManagedJobTerminalState
    succeeded_tasks: int = 0
    failed_tasks: int = 0
    total_tasks: int = 0
    message: str = ""
    raw_events: tuple[str, ...] = field(default_factory=_empty_str_tuple)

    def to_dict(self) -> dict[str, object]:
        """Serialize the job status to a JSON-safe dictionary."""
        return {
            "provider": self.handle.provider,
            "resource_name": self.handle.resource_name,
            "location": self.handle.location,
            "state": self.state.value,
            "succeeded_tasks": self.succeeded_tasks,
            "failed_tasks": self.failed_tasks,
            "total_tasks": self.total_tasks,
            "message": self.message,
            "raw_events": tuple(self.raw_events),
        }


@runtime_checkable
class ManagedJobRunner(Protocol):
    """Provider-neutral interface for declarative cloud-job providers.

    Implementations must be safe to use from multiple threads; the
    Activin-E pipeline calls ``submit`` once per stage attempt and
    polls ``get_status`` from a background loop.
    """

    @property
    def provider_name(self) -> str:
        """Short provider identifier (e.g. ``"gcp-batch"``)."""
        ...

    def submit(self, spec: ManagedJobSpec) -> ManagedJobHandle:
        """Submit ``spec`` and return the cloud resource handle.

        Idempotency is the caller's responsibility: providers may
        reject duplicate submissions with a name collision. The
        :class:`~vastai_gpu_runner.managed_jobs.state.ManagedJobState`
        loader is the canonical way to detect "already submitted" on
        resume.
        """
        ...

    def get_status(self, handle: ManagedJobHandle) -> ManagedJobStatus:
        """Return the latest status snapshot for ``handle``."""
        ...

    def list_tasks(self, handle: ManagedJobHandle) -> Iterable[ManagedTaskStatus]:
        """Return per-task status, when the provider supports it."""
        ...

    def cancel(self, handle: ManagedJobHandle) -> None:
        """Request cancellation. Idempotent."""
        ...

    def delete(self, handle: ManagedJobHandle) -> None:
        """Permanently remove the job. Idempotent."""
        ...
