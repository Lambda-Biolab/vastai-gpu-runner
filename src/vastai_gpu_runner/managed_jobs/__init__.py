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
    ManagedJobHandle,
    ManagedJobRunner,
    ManagedJobSpec,
    ManagedJobStatus,
    ManagedJobTerminalState,
    ManagedTaskStatus,
)
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
    "ManagedJobHandle",
    "ManagedJobRunner",
    "ManagedJobSpec",
    "ManagedJobState",
    "ManagedJobStateError",
    "ManagedJobStatus",
    "ManagedJobTerminalState",
    "ManagedTaskStatus",
    "load_managed_job_state",
    "load_or_none",
    "save_managed_job_state",
]
