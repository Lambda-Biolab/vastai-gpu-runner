"""Tests for types module — enums and dataclasses."""

from __future__ import annotations

from vastai_gpu_runner.types import (
    CloudInstance,
    DeploymentConfig,
    DeploymentResult,
    InstanceStatus,
    Provider,
)


class TestProvider:
    def test_vastai_value(self) -> None:
        assert Provider.VASTAI.value == "vastai"

    def test_runpod_value(self) -> None:
        assert Provider.RUNPOD.value == "runpod"


class TestInstanceStatus:
    def test_lifecycle_values(self) -> None:
        assert InstanceStatus.CREATING.value == "creating"
        assert InstanceStatus.RUNNING.value == "running"
        assert InstanceStatus.DESTROYED.value == "destroyed"


class TestDeploymentConfig:
    def test_defaults(self) -> None:
        config = DeploymentConfig()
        assert config.gpu_model == "RTX_4090"
        assert config.max_cost_per_hour == 0.45
        assert config.min_reliability == 0.995
        assert config.workspace_dir == "/workspace"

    def test_custom_values(self) -> None:
        config = DeploymentConfig(gpu_model="RTX_3090", max_cost_per_hour=0.15)
        assert config.gpu_model == "RTX_3090"
        assert config.max_cost_per_hour == 0.15


class TestCloudInstance:
    def test_defaults(self) -> None:
        inst = CloudInstance()
        assert inst.provider == Provider.VASTAI
        assert inst.ssh_port == 22
        assert inst.ssh_user == "root"

    def test_mutable_status(self) -> None:
        inst = CloudInstance()
        inst.status = InstanceStatus.RUNNING
        assert inst.status == InstanceStatus.RUNNING


class TestDeploymentResult:
    def test_failure_default(self) -> None:
        result = DeploymentResult()
        assert result.success is False
        assert result.instance is None
        assert result.output_files == []

    def test_success(self) -> None:
        inst = CloudInstance(instance_id="123")
        result = DeploymentResult(success=True, instance=inst)
        assert result.success is True
        assert result.instance is not None
