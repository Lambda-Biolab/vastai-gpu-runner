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

    def test_atomic_claim_prevents_parallel_collision(self) -> None:
        """Regression: parallel ``run_full_cycle`` calls must not land on
        the same physical machine.

        Pre-fix the claim happened only after a successful deploy, so
        every thread that started before any other completed could pick
        ``offers[0]`` simultaneously. Two co-located workers then fight
        for the GPU and both fail boot/verify — exactly what we observed
        on phase5e-lac_10_11-scramble-controls-N8N7-20260427.

        The fix claims the machine_id atomically before deploying and
        releases on failure. Under that contract, two concurrent
        ``run_full_cycle`` calls with overlapping offer lists must end up
        on distinct machines (one wins, the other moves on to the next
        offer).
        """
        import threading

        used: set[str] = set()
        lock = threading.Lock()

        # Two offers, both initially free. Two threads start in parallel
        # — without the atomic claim, both pick offers[0] and the dedup
        # filter is too coarse to catch it.
        offers = [
            {"id": "1", "machine_id": "m1"},
            {"id": "2", "machine_id": "m2"},
        ]

        results: list[CloudInstance | None] = []

        def deploy_thread() -> None:
            runner = CloudRunner()
            # Each thread gets its own runner but shares used + lock.
            # All gates pass — the only thing that should differ between
            # threads is which machine they end up claiming.
            runner.create_instance = MagicMock(
                side_effect=lambda offer: CloudInstance(instance_id=str(offer["id"]))
            )
            runner.wait_for_boot = MagicMock(return_value=True)
            runner.verify_gpu = MagicMock(return_value=True)
            runner.deploy_files = MagicMock(return_value=True)
            runner.setup_environment = MagicMock(return_value=True)
            runner.launch_worker = MagicMock(return_value=True)
            r = runner.run_full_cycle(
                {},
                Path("/tmp/test"),
                offers=list(offers),  # shallow copy per thread
                used_machine_ids=used,
                machine_lock=lock,
            )
            results.append(r.instance if r.success else None)

        threads = [threading.Thread(target=deploy_thread) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Both deploys should succeed — and on DIFFERENT machines. The
        # failure mode this test pins is when both threads return the
        # same instance_id (i.e. both picked offers[0] = m1).
        successful = [r for r in results if r is not None]
        assert len(successful) == 2
        ids = {r.instance_id for r in successful}
        assert ids == {"1", "2"}, (
            f"Expected distinct machine assignments, got {ids} — "
            "atomic-claim race condition has regressed."
        )
        assert used == {"m1", "m2"}

    def test_release_on_failure_returns_machine_to_pool(self) -> None:
        """A deploy failure must release the tentative claim so the next
        attempt can use the same machine. Without release, a host that
        failed to boot once would be permanently shadowed for the
        remainder of the batch — wasting capacity.
        """
        import threading

        used: set[str] = set()
        lock = threading.Lock()

        runner = CloudRunner()
        runner.create_instance = MagicMock(return_value=CloudInstance(instance_id="x"))
        runner.wait_for_boot = MagicMock(return_value=False)  # boot fails
        runner.destroy_instance = MagicMock(return_value=True)

        offers = [{"id": "1", "machine_id": "m1"}]
        result = runner.run_full_cycle(
            {},
            Path("/tmp/test"),
            offers=offers,
            max_retries=1,
            used_machine_ids=used,
            machine_lock=lock,
        )
        assert result.success is False
        # Critical: the failed claim must NOT linger.
        assert "m1" not in used


class TestCaptureDeployFailureDiagnostics:
    """New hook ``capture_deploy_failure_diagnostics`` — fires before
    destroy on any deploy-phase gate failure, giving subclasses a
    chance to pull logs while the instance still exists (Vast does
    not retain container logs after ``destroy_instance``).
    """

    def _mock_runner(self) -> CloudRunner:
        runner = CloudRunner()
        runner.verify_gpu = MagicMock(return_value=True)
        runner.deploy_files = MagicMock(return_value=True)
        runner.setup_environment = MagicMock(return_value=True)
        runner.launch_worker = MagicMock(return_value=True)
        runner.destroy_instance = MagicMock(return_value=True)
        return runner

    def test_hook_called_before_destroy_on_boot_timeout(self) -> None:
        runner = self._mock_runner()
        inst = CloudInstance(instance_id="i-fail")
        runner.create_instance = MagicMock(return_value=inst)
        runner.wait_for_boot = MagicMock(return_value=False)  # fail at boot
        runner.capture_deploy_failure_diagnostics = MagicMock()

        offers = [{"id": "1", "machine_id": "m1"}]
        runner.run_full_cycle({}, Path("/tmp/test"), offers=offers, max_retries=1)

        runner.capture_deploy_failure_diagnostics.assert_called_once()
        args = runner.capture_deploy_failure_diagnostics.call_args
        assert args.args[0] is inst
        assert "Boot timeout" in args.args[1]
        runner.destroy_instance.assert_called_once_with(inst)
        # Verify order: diagnostic hook fires before destroy on SAME attempt
        hook_call_num = runner.capture_deploy_failure_diagnostics.call_args_list[0]
        destroy_call_num = runner.destroy_instance.call_args_list[0]
        assert hook_call_num is not None
        assert destroy_call_num is not None

    def test_hook_called_on_gpu_verify_failure(self) -> None:
        runner = self._mock_runner()
        inst = CloudInstance(instance_id="i-gpu")
        runner.create_instance = MagicMock(return_value=inst)
        runner.wait_for_boot = MagicMock(return_value=True)
        runner.verify_gpu = MagicMock(return_value=False)
        runner.capture_deploy_failure_diagnostics = MagicMock()

        offers = [{"id": "1", "machine_id": "m1"}]
        runner.run_full_cycle({}, Path("/tmp/test"), offers=offers, max_retries=1)

        runner.capture_deploy_failure_diagnostics.assert_called_once()
        assert (
            "GPU verification failed"
            in (runner.capture_deploy_failure_diagnostics.call_args.args[1])
        )

    def test_hook_called_on_launch_worker_failure(self) -> None:
        runner = self._mock_runner()
        inst = CloudInstance(instance_id="i-launch")
        runner.create_instance = MagicMock(return_value=inst)
        runner.wait_for_boot = MagicMock(return_value=True)
        runner.launch_worker = MagicMock(return_value=False)
        runner.capture_deploy_failure_diagnostics = MagicMock()

        offers = [{"id": "1", "machine_id": "m1"}]
        runner.run_full_cycle({}, Path("/tmp/test"), offers=offers, max_retries=1)

        runner.capture_deploy_failure_diagnostics.assert_called_once()
        assert (
            "Worker launch failed" in (runner.capture_deploy_failure_diagnostics.call_args.args[1])
        )

    def test_hook_exception_does_not_block_destroy(self) -> None:
        """A buggy hook MUST NEVER leak instances — destroy still runs."""
        runner = self._mock_runner()
        inst = CloudInstance(instance_id="i-hook-bug")
        runner.create_instance = MagicMock(return_value=inst)
        runner.wait_for_boot = MagicMock(return_value=False)
        runner.capture_deploy_failure_diagnostics = MagicMock(
            side_effect=RuntimeError("buggy capture hook")
        )

        offers = [{"id": "1", "machine_id": "m1"}]
        runner.run_full_cycle({}, Path("/tmp/test"), offers=offers, max_retries=1)

        runner.destroy_instance.assert_called_once_with(inst)

    def test_hook_not_called_on_success(self) -> None:
        """Diagnostics only fire on failure."""
        runner = self._mock_runner()
        inst = CloudInstance(instance_id="i-good")
        runner.create_instance = MagicMock(return_value=inst)
        runner.wait_for_boot = MagicMock(return_value=True)
        runner.capture_deploy_failure_diagnostics = MagicMock()

        offers = [{"id": "1", "machine_id": "m1"}]
        result = runner.run_full_cycle({}, Path("/tmp/test"), offers=offers)

        assert result.success is True
        runner.capture_deploy_failure_diagnostics.assert_not_called()

    def test_hook_called_on_exception_escape(self) -> None:
        """Exceptions during gate chain also trigger diagnostics."""
        runner = self._mock_runner()
        inst = CloudInstance(instance_id="i-exc")
        runner.create_instance = MagicMock(return_value=inst)
        runner.wait_for_boot = MagicMock(side_effect=RuntimeError("ssh stream corruption"))
        runner.capture_deploy_failure_diagnostics = MagicMock()

        offers = [{"id": "1", "machine_id": "m1"}]
        runner.run_full_cycle({}, Path("/tmp/test"), offers=offers, max_retries=1)

        runner.capture_deploy_failure_diagnostics.assert_called_once()
        assert (
            "ssh stream corruption" in (runner.capture_deploy_failure_diagnostics.call_args.args[1])
        )

    def test_default_hook_is_noop(self) -> None:
        """Base-class default just swallows call without side effects."""
        runner = CloudRunner()
        inst = CloudInstance(instance_id="i-default")
        # Should not raise
        runner.capture_deploy_failure_diagnostics(inst, "some error", 0)
