"""Core types for cloud GPU orchestration.

Defines the shared enums and dataclasses used by all providers and
orchestrators:

- ``Provider`` / ``InstanceStatus`` — state and provider enums
- ``DeploymentConfig`` — per-provider configuration
- ``CloudInstance`` — represents a running cloud GPU instance
- ``DeploymentResult`` — outcome of a deployment attempt
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Provider(enum.Enum):
    """Cloud GPU provider."""

    VASTAI = "vastai"
    RUNPOD = "runpod"


class ComputeMode(enum.Enum):
    """Where to run GPU workloads."""

    LOCAL = "L"
    CLOUD = "C"
    HYBRID = "H"


class InstanceStatus(enum.Enum):
    """Lifecycle state of a cloud instance."""

    CREATING = "creating"
    BOOTING = "booting"
    RUNNING = "running"
    FAILED = "failed"
    DESTROYED = "destroyed"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DeploymentConfig:
    """Configuration for cloud GPU deployments."""

    gpu_model: str = "RTX_4090"
    max_cost_per_hour: float = 0.45
    boot_timeout_seconds: int = 300
    gpu_verify_timeout: int = 120
    min_disk_gb: int = 40
    min_network_mbps: int = 800
    min_reliability: float = 0.995
    worker_script: str = "worker.sh"
    workspace_dir: str = "/workspace"
    conda_env_spec: str = ""
    upload_checkpoint: bool = False
    download_checkpoint: bool = False


@dataclass
class CloudInstance:
    """Represents a running cloud GPU instance."""

    provider: Provider = Provider.VASTAI
    instance_id: str = ""
    gpu_model: str = ""
    cost_per_hour: float = 0.0
    status: InstanceStatus = InstanceStatus.CREATING
    label: str = ""
    ssh_host: str = ""
    ssh_port: int = 22
    ssh_user: str = "root"


@dataclass
class DeploymentResult:
    """Outcome of a cloud deployment attempt."""

    success: bool = False
    instance: CloudInstance | None = None
    error: str = ""
    output_files: list[str] = field(default_factory=list)
