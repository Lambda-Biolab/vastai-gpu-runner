"""Provider-neutral managed-job orchestration.

Complements the direct-VM :class:`~vastai_gpu_runner.runner.CloudRunner`
abstraction. ``ManagedJobRunner`` is for declarative cloud services
such as GCP Batch where the cloud platform owns the underlying VM
lifecycle. The two abstractions do not share code or types; a consumer
selects one explicitly via configuration.

Managed jobs are described by a :class:`ManagedJobSpec`, polled via
:class:`ManagedJobStatus`, and recorded as a :class:`ManagedJobState`
JSON file with a fail-closed schema-version loader.
"""

from __future__ import annotations

from vastai_gpu_runner.managed_jobs.base import (
    BootDisk,
    ComputeResource,
    GpuAccelerator,
    MachineResource,
    ManagedJobHandle,
    ManagedJobLifecycleState,
    ManagedJobRunner,
    ManagedJobSpec,
    ManagedJobStatus,
    ManagedJobTerminalState,
    ManagedTaskStatus,
    NetworkConfig,
    ServiceAccount,
    StorageMount,
)
from vastai_gpu_runner.managed_jobs.errors import (
    ManagedJobAlreadyExistsError,
    ManagedJobConflictError,
    ManagedJobError,
    ManagedJobNotFoundError,
    ManagedJobPermanentError,
    ManagedJobTransientError,
)
from vastai_gpu_runner.managed_jobs.gcp_batch import GcpBatchRunner
from vastai_gpu_runner.managed_jobs.state import (
    CURRENT_MANAGED_JOB_SCHEMA,
    ManagedJobState,
    ManagedJobStateError,
    load_managed_job_state,
    load_or_none,
    save_managed_job_state,
)

__all__ = [
    "CURRENT_MANAGED_JOB_SCHEMA",
    "BootDisk",
    "ComputeResource",
    "GcpBatchRunner",
    "GpuAccelerator",
    "MachineResource",
    "ManagedJobAlreadyExistsError",
    "ManagedJobConflictError",
    "ManagedJobError",
    "ManagedJobHandle",
    "ManagedJobLifecycleState",
    "ManagedJobNotFoundError",
    "ManagedJobPermanentError",
    "ManagedJobRunner",
    "ManagedJobSpec",
    "ManagedJobState",
    "ManagedJobStateError",
    "ManagedJobStatus",
    "ManagedJobTerminalState",
    "ManagedJobTransientError",
    "ManagedTaskStatus",
    "NetworkConfig",
    "ServiceAccount",
    "StorageMount",
    "load_managed_job_state",
    "load_or_none",
    "save_managed_job_state",
]
