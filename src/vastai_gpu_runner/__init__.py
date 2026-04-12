"""Cloud GPU orchestration framework for Vast.ai.

Provides battle-tested infrastructure for deploying GPU workloads to
Vast.ai marketplace instances with R2 storage, crash recovery, and
worker lifecycle management.

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

    # Vast.ai provider
    from vastai_gpu_runner.providers.vastai import VastaiRunner

    # R2 storage
    from vastai_gpu_runner.storage.r2 import R2Sink

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

from vastai_gpu_runner.runner import CloudRunner
from vastai_gpu_runner.types import (
    CloudInstance,
    DeploymentConfig,
    DeploymentResult,
    InstanceStatus,
    Provider,
)

__all__ = [
    "CloudInstance",
    "CloudRunner",
    "DeploymentConfig",
    "DeploymentResult",
    "InstanceStatus",
    "Provider",
]
