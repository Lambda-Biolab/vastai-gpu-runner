"""Cloud GPU orchestration framework for Vast.ai + GCP Batch.

Provides battle-tested infrastructure for deploying GPU workloads to
Vast.ai marketplace instances with R2 storage, crash recovery, and
worker lifecycle management. The v0.6.0 release adds a parallel
``ManagedJobRunner`` abstraction for declarative cloud batch workloads
(GCP Batch) with a Google Cloud Storage artifact sink.

Public API::

    # Types and interfaces
    from vastai_gpu_runner import (
        CloudRunner,
        DeploymentConfig,
        CloudInstance,
        DeploymentResult,
        Provider,
        InstanceStatus,
    )

    # Vast.ai provider (implements CloudRunner)
    from vastai_gpu_runner.providers.vastai import VastaiRunner

    # Local provider (implements CloudRunner; zero-cost CI backend)
    from vastai_gpu_runner.providers.local import LocalRunner

    # R2 storage
    from vastai_gpu_runner.storage.r2 import R2Sink

    # Managed jobs (declarative cloud batch — separate from CloudRunner)
    from vastai_gpu_runner.managed_jobs import (
        ManagedJobRunner,
        ManagedJobSpec,
        ManagedJobHandle,
        ManagedJobStatus,
        ManagedJobTerminalState,
        GcpBatchRunner,  # provided gcp extra
    )

    # GCS storage (provided gcp extra)
    from vastai_gpu_runner.storage.gcs import GcsSink

    # Batch state
    from vastai_gpu_runner.state import BatchState, ShardState, JobState, JobBatchState

    # Worker framework
    from vastai_gpu_runner.worker.base import BaseWorker

    # Estimator
    from vastai_gpu_runner.estimator.core import (
        GPU_SPEED_FACTOR,
        ScalingRow,
        EstimateResult,
        PriceSummary,
    )
"""

from vastai_gpu_runner.batch import BatchOrchestrator, BatchUnit, FailureVerdict
from vastai_gpu_runner.runner import CloudRunner
from vastai_gpu_runner.types import (
    CloudInstance,
    ComputeMode,
    DeploymentConfig,
    DeploymentResult,
    InstanceStatus,
    Provider,
)

__all__ = [
    "BatchOrchestrator",
    "BatchUnit",
    "CloudInstance",
    "CloudRunner",
    "ComputeMode",
    "DeploymentConfig",
    "DeploymentResult",
    "FailureVerdict",
    "InstanceStatus",
    "Provider",
]
