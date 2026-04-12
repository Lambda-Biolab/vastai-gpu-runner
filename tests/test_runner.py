"""Tests for CloudRunner base class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from vastai_gpu_runner.runner import CloudRunner
from vastai_gpu_runner.types import CloudInstance, DeploymentConfig


class TestCloudRunner:
    def test_default_config(self) -> None:
        runner = CloudRunner()
        assert runner.config.gpu_model == "RTX_4090"

    def test_custom_config(self) -> None:
        config = DeploymentConfig(gpu_model="RTX_3090")
        runner = CloudRunner(config)
        assert runner.config.gpu_model == "RTX_3090"

    def test_search_offers_default_empty(self) -> None:
        runner = CloudRunner()
        assert runner.search_offers() == []

    def test_run_full_cycle_no_offers(self) -> None:
        runner = CloudRunner()
        result = runner.run_full_cycle({}, Path("/tmp/test"))
        assert result.success is False
        assert "No GPU offers" in result.error

    def test_run_full_cycle_with_mock_provider(self) -> None:
        """Full cycle succeeds when all steps return True."""
        runner = CloudRunner()
        inst = CloudInstance(instance_id="mock-1")

        runner.create_instance = MagicMock(return_value=inst)
        runner.wait_for_boot = MagicMock(return_value=True)
        runner.verify_gpu = MagicMock(return_value=True)
        runner.deploy_files = MagicMock(return_value=True)
        runner.setup_environment = MagicMock(return_value=True)
        runner.launch_worker = MagicMock(return_value=True)

        offers = [{"id": "1", "machine_id": "m1"}]
        result = runner.run_full_cycle({}, Path("/tmp/test"), offers=offers)
        assert result.success is True
        assert result.instance is not None
        assert result.instance.instance_id == "mock-1"

    def test_run_full_cycle_retry_on_boot_failure(self) -> None:
        """Retries with next offer when boot fails."""
        runner = CloudRunner()
        inst1 = CloudInstance(instance_id="fail-1")
        inst2 = CloudInstance(instance_id="succeed-2")

        runner.create_instance = MagicMock(side_effect=[inst1, inst2])
        runner.wait_for_boot = MagicMock(side_effect=[False, True])
        runner.verify_gpu = MagicMock(return_value=True)
        runner.deploy_files = MagicMock(return_value=True)
        runner.setup_environment = MagicMock(return_value=True)
        runner.launch_worker = MagicMock(return_value=True)
        runner.destroy_instance = MagicMock(return_value=True)

        offers = [
            {"id": "1", "machine_id": "m1"},
            {"id": "2", "machine_id": "m2"},
        ]
        result = runner.run_full_cycle({}, Path("/tmp/test"), offers=offers)
        assert result.success is True
        runner.destroy_instance.assert_called_once_with(inst1)

    def test_machine_dedup(self) -> None:
        """Skips machines already claimed by other threads."""
        runner = CloudRunner()
        import threading

        used = {"m1"}
        lock = threading.Lock()

        inst = CloudInstance(instance_id="ok")
        runner.create_instance = MagicMock(return_value=inst)
        runner.wait_for_boot = MagicMock(return_value=True)
        runner.verify_gpu = MagicMock(return_value=True)
        runner.deploy_files = MagicMock(return_value=True)
        runner.setup_environment = MagicMock(return_value=True)
        runner.launch_worker = MagicMock(return_value=True)

        offers = [
            {"id": "1", "machine_id": "m1"},  # should be skipped
            {"id": "2", "machine_id": "m2"},
        ]
        result = runner.run_full_cycle(
            {},
            Path("/tmp/test"),
            offers=offers,
            used_machine_ids=used,
            machine_lock=lock,
        )
        assert result.success is True
        assert "m2" in used
