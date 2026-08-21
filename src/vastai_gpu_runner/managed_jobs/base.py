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

Resource fields are provider-neutral typed dataclasses
(:class:`BootDisk`, :class:`GpuAccelerator`, :class:`MachineResource`,
:class:`ServiceAccount`, :class:`NetworkConfig`). Providers map them
onto their native representations; consumers compose specs without
touching cloud-specific types.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class ManagedJobLifecycleState(enum.Enum):
    """Normalized lifecycle states returned by managed-job providers."""

    QUEUED = "queued"
    PENDING = "pending"
    RUNNING = "running"

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        """Return whether this state represents a completed lifecycle."""
        return self in {
            self.SUCCEEDED,
            self.FAILED,
            self.CANCELLED,
            self.UNKNOWN,
        }


# Kept as an alias because consumers imported this name before in-progress
# states were represented explicitly.
ManagedJobTerminalState = ManagedJobLifecycleState


def _empty_str_tuple() -> tuple[str, ...]:
    return ()


@dataclass(frozen=True)
class BootDisk:
    """Provider-neutral boot disk descriptor.

    Attributes:
        image: Optional VM image override. Empty string = use the default.
        size_gb: Boot disk size in GB. Provider picks a default when 0.
        type_: Disk type / family hint (e.g. ``"pd-ssd"``). Empty = default.
    """

    image: str = ""
    size_gb: int = 0
    type_: str = ""


@dataclass(frozen=True)
class GpuAccelerator:
    """Provider-neutral GPU accelerator descriptor.

    Attributes:
        type_: Accelerator model identifier (e.g. ``"nvidia-tesla-a100"``).
        count: Number of GPUs to attach.
        driver_version: Optional driver version override. Empty = default.
        install_gpu_drivers: When True, the provider installs GPU drivers.
    """

    type_: str = ""
    count: int = 0
    driver_version: str = ""
    install_gpu_drivers: bool = False


@dataclass(frozen=True)
class MachineResource:
    """Provider-neutral VM shape + accelerators + boot disk.

    Used to populate the provider's allocation policy (GCP Batch
    ``AllocationPolicy.InstancePolicy``). All fields are optional;
    unset fields fall back to the provider's defaults.

    Attributes:
        machine_type: VM shape, e.g. ``"n1-standard-4"``. Empty = auto.
        boot_disk: Optional boot disk override.
        accelerators: GPUs to attach. Empty tuple = none.
        min_cpu_platform: Minimum CPU platform hint. Empty = provider default.
    """

    machine_type: str = ""
    boot_disk: BootDisk | None = None
    accelerators: tuple[GpuAccelerator, ...] = ()
    min_cpu_platform: str = ""


@dataclass(frozen=True)
class ComputeResource:
    """Provider-neutral per-task compute limits.

    Maps to GCP Batch ``TaskSpec.compute_resource`` (CPU in millicores,
    memory in MiB, boot disk in MiB). All fields are optional; unset
    fields fall back to the provider's defaults.

    Attributes:
        cpu_milli: CPU in millicores (1000 = 1 vCPU).
        memory_mib: Memory in MiB.
        boot_disk_mib: Boot disk in MiB.
    """

    cpu_milli: int = 0
    memory_mib: int = 0
    boot_disk_mib: int = 0


@dataclass(frozen=True)
class ServiceAccount:
    """Provider-neutral service account descriptor.

    Attributes:
        email: Service account email. Empty = use the default.
        scopes: OAuth scopes to attach. Empty tuple = default scopes.
    """

    email: str = ""
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class NetworkConfig:
    """Provider-neutral network configuration.

    Attributes:
        network: Network resource URI (full or short). Empty = default.
        subnetwork: Subnetwork resource URI. Empty = default.
        no_external_ip_address: When True, the VM has no external IP.
    """

    network: str = ""
    subnetwork: str = ""
    no_external_ip_address: bool = False


@dataclass(frozen=True)
class StorageMount:
    """Provider-neutral storage mount declaration."""

    uri: str
    mount_path: str
    read_only: bool = False


@dataclass(frozen=True)
class ManagedJobSpec:
    """Provider-neutral description of a unit of managed-job work.

    Providers map this onto their native job representation. The
    ``labels`` mapping is opaque to the provider-neutral layer; it is
    carried through to status, log, and state payloads so consumers
    can correlate jobs across stages.

    Resource fields are typed (:class:`MachineResource`,
    :class:`ComputeResource`, :class:`ServiceAccount`,
    :class:`NetworkConfig`). The provider-specific ``environment``
    mapping is a free-form ``str→str`` dict for ad-hoc key/value
    transport (kept for compatibility; new code should prefer typed
    fields).

    Attributes:
        storage_mounts: Typed storage mounts. Providers may support
            different URI schemes; unsupported schemes fail at submit time.
        gcs_mounts: Deprecated GCS-only compatibility field. New callers
            should use ``storage_mounts``.
        allowed_locations: Allowed regions/zones for the job
            (e.g. ``("regions/us-central1",)``). Validated against
            ``region`` at submit time when non-empty.
        spot: When True, the job runs on Spot/preemptible VMs.
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
    machine_resource: MachineResource | None = None
    compute_resource: ComputeResource | None = None
    service_account: ServiceAccount | None = None
    network: NetworkConfig | None = None
    allowed_locations: tuple[str, ...] = ()
    spot: bool = False
    storage_mounts: tuple[StorageMount, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize the spec into a JSON-safe dictionary."""
        machine = self.machine_resource
        compute = self.compute_resource
        sa = self.service_account
        net = self.network
        return {
            "name": self.name,
            "task_count": self.task_count,
            "parallelism": self.parallelism,
            "image": self.image,
            "command": list(self.command),
            "environment": dict(self.environment),
            "labels": dict(self.labels),
            "gcs_mounts": list(self.gcs_mounts),
            "storage_mounts": [
                {
                    "uri": mount.uri,
                    "mount_path": mount.mount_path,
                    "read_only": mount.read_only,
                }
                for mount in self.storage_mounts
            ],
            "timeout_seconds": self.timeout_seconds,
            "retry_on_preempt": self.retry_on_preempt,
            "region": self.region,
            "machine_resource": (
                {
                    "machine_type": machine.machine_type,
                    "boot_disk": (
                        {
                            "image": machine.boot_disk.image,
                            "size_gb": machine.boot_disk.size_gb,
                            "type": machine.boot_disk.type_,
                        }
                        if machine.boot_disk is not None
                        else None
                    ),
                    "accelerators": [
                        {
                            "type": acc.type_,
                            "count": acc.count,
                            "driver_version": acc.driver_version,
                            "install_gpu_drivers": acc.install_gpu_drivers,
                        }
                        for acc in machine.accelerators
                    ],
                    "min_cpu_platform": machine.min_cpu_platform,
                }
                if machine is not None
                else None
            ),
            "compute_resource": (
                {
                    "cpu_milli": compute.cpu_milli,
                    "memory_mib": compute.memory_mib,
                    "boot_disk_mib": compute.boot_disk_mib,
                }
                if compute is not None
                else None
            ),
            "service_account": (
                {"email": sa.email, "scopes": list(sa.scopes)} if sa is not None else None
            ),
            "network": (
                {
                    "network": net.network,
                    "subnetwork": net.subnetwork,
                    "no_external_ip_address": net.no_external_ip_address,
                }
                if net is not None
                else None
            ),
            "allowed_locations": list(self.allowed_locations),
            "spot": self.spot,
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
    state: str | ManagedJobLifecycleState
    exit_code: int | None = None
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize the task status to a JSON-safe dictionary."""
        return {
            "task_index": self.task_index,
            "state": self.state.value if isinstance(self.state, enum.Enum) else self.state,
            "exit_code": self.exit_code,
            "message": self.message,
        }


@dataclass(frozen=True)
class ManagedJobStatus:
    """Snapshot of a managed-job's state at a point in time."""

    handle: ManagedJobHandle
    state: ManagedJobLifecycleState
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
            "raw_events": list(self.raw_events),
        }


@runtime_checkable
class ManagedJobRunner(Protocol):
    """Provider-neutral interface for declarative cloud-job providers.

    Implementations must be safe to use from multiple threads; the
    A controller calls ``submit`` once per job attempt and polls
    ``get_status`` from a background loop.
    """

    @property
    def provider_name(self) -> str:
        """Short provider identifier (e.g. ``"gcp-batch"``)."""
        ...

    def submit(self, spec: ManagedJobSpec) -> ManagedJobHandle:
        """Submit ``spec`` and return the cloud resource handle.

        ``spec.name`` is the idempotency key. A duplicate provider name
        raises ``ManagedJobConflictError``; it is not implicit success.
        State persistence can still be used by callers to avoid a retry
        after a successful submit.
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
