"""Shared test fixtures for vastai-gpu-runner."""

from __future__ import annotations

import pytest

from vastai_gpu_runner.types import CloudInstance, DeploymentConfig, Provider


@pytest.fixture
def config() -> DeploymentConfig:
    """Default deployment config for tests."""
    return DeploymentConfig(
        gpu_model="RTX_4090",
        max_cost_per_hour=0.45,
        boot_timeout_seconds=10,
        workspace_dir="/workspace/test",
    )


@pytest.fixture
def instance() -> CloudInstance:
    """A running cloud instance for tests."""
    return CloudInstance(
        provider=Provider.VASTAI,
        instance_id="12345",
        gpu_model="RTX 4090",
        cost_per_hour=0.32,
        ssh_host="ssh5.vast.ai",
        ssh_port=22022,
        ssh_user="root",
    )
