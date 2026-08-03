# pyright: reportPrivateUsage=warning, reportMissingParameterType=warning, reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnusedFunction=false, reportUnusedClass=false
"""Behavioral tests for the local subprocess runner."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

import vastai_gpu_runner.providers.local as local_module
from vastai_gpu_runner.cleanup_policy import (
    CleanupRefusal,
    CleanupVerdict,
    InstanceCandidate,
)
from vastai_gpu_runner.providers.local import LocalRunner, build_local_cleanup_policy
from vastai_gpu_runner.types import CloudInstance, DeploymentConfig, InstanceStatus, Provider


class _FakeProcess:
    def __init__(self, *, pid: int = 123, returncode: int | None = None) -> None:
        self.pid = pid
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode if self.returncode is not None else 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def _instance() -> CloudInstance:
    return CloudInstance(provider=Provider.LOCAL, instance_id="local")


class TestLocalRunner:
    def test_search_and_create_use_a_synthetic_local_offer(self, tmp_path: Path) -> None:
        runner = LocalRunner(DeploymentConfig(workspace_dir=str(tmp_path)))

        offers = runner.search_offers()
        instance = runner.create_instance(offers[0])

        assert offers == [{"machine_id": "local", "dph_total": 0.0}]
        assert instance.provider == Provider.LOCAL
        assert instance.instance_id == "local"
        assert instance.ssh_host == "localhost"
        assert runner._workspace(instance).is_dir()

        assert runner.destroy_instance(instance) is True

    def test_wait_for_boot_marks_local_instance_running(self) -> None:
        runner = LocalRunner()
        instance = _instance()

        assert runner.wait_for_boot(instance) is True
        assert instance.status == InstanceStatus.RUNNING

    def test_verify_gpu_passes_when_nvidia_smi_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = LocalRunner()
        instance = _instance()
        unavailable = Mock(side_effect=FileNotFoundError("nvidia-smi"))
        monkeypatch.setattr(local_module.subprocess, "run", unavailable)

        assert runner.verify_gpu(instance) is True

    def test_file_operations_round_trip_through_local_workspace(self, tmp_path: Path) -> None:
        runner = LocalRunner()
        instance = runner.create_instance({"machine_id": "local"})
        source = tmp_path / "input.txt"
        source.write_text("payload\n")
        destination = tmp_path / "results" / "input.txt"

        try:
            assert runner.deploy_files(instance, {"nested/input.txt": source}) is True
            assert runner.list_remote_files(instance) == ["nested/input.txt"]
            assert runner.download_file(instance, "nested/input.txt", destination) is True
            assert destination.read_text() == "payload\n"
        finally:
            runner.destroy_instance(instance)

    def test_deploy_files_skips_missing_sources(self, tmp_path: Path) -> None:
        runner = LocalRunner()
        instance = runner.create_instance({"machine_id": "local"})

        try:
            assert runner.deploy_files(instance, {"missing.txt": tmp_path / "missing.txt"}) is True
            assert runner.list_remote_files(instance) == []
        finally:
            runner.destroy_instance(instance)

    def test_launch_and_progress_report_worker_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = LocalRunner()
        instance = runner.create_instance({"machine_id": "local"})
        process = _FakeProcess()
        popen_calls: list[tuple[object, dict[str, object]]] = []

        def fake_popen(command: object, **kwargs: object) -> _FakeProcess:
            popen_calls.append((command, kwargs))
            return process

        monkeypatch.setattr(local_module.subprocess, "Popen", fake_popen)
        try:
            worker = runner._workspace(instance) / "worker.sh"
            worker.write_text("#!/usr/bin/env bash\n")

            assert runner.launch_worker(instance) is True
            assert popen_calls[0][0] == ["bash", "worker.sh"]
            assert (runner._workspace(instance) / "worker.pid").read_text() == "123"
            assert runner.check_progress(instance) == {
                "running": True,
                "complete": False,
                "log_tail": "",
            }

            (runner._workspace(instance) / "DONE").touch()
            progress = runner.check_progress(instance)
            assert progress["complete"] is True
            assert progress["running"] is False
        finally:
            runner.destroy_instance(instance)

    def test_progress_reports_dead_worker_without_done(self) -> None:
        runner = LocalRunner()
        instance = runner.create_instance({"machine_id": "local"})
        process = _FakeProcess(returncode=1)
        runner._processes[instance.instance_id] = process  # type: ignore[assignment]
        workspace = runner._workspace(instance)
        (workspace / "worker.pid").write_text("123")
        (workspace / "worker.log").write_text("last line\n")

        try:
            assert runner.check_progress(instance) == {
                "running": False,
                "complete": False,
                "worker_dead": True,
                "log_tail": "last line",
            }
        finally:
            runner.destroy_instance(instance)

    def test_destroy_is_idempotent_and_stops_worker(self) -> None:
        runner = LocalRunner()
        instance = runner.create_instance({"machine_id": "local"})
        process = _FakeProcess()
        runner._processes[instance.instance_id] = process  # type: ignore[assignment]
        workspace = runner._workspace(instance)
        (workspace / "output.txt").write_text("done")

        assert runner.destroy_instance(instance) is True
        assert process.terminated is True
        assert instance.status == InstanceStatus.DESTROYED
        assert not workspace.exists()
        assert runner.destroy_instance(instance) is True

    def test_cleanup_policy_is_local_and_has_no_cloud_candidates(self) -> None:
        policy = build_local_cleanup_policy()

        assert policy.provider == Provider.LOCAL
        assert policy.list_instances() == []
        result = policy.destroy(
            InstanceCandidate(
                provider=Provider.LOCAL,
                instance_id="local",
                label="local",
                state="running",
            )
        )
        assert result.verdict == CleanupVerdict.DESTROYED

        mismatch = policy.destroy(
            InstanceCandidate(
                provider=Provider.VASTAI,
                instance_id="vast",
                label="local",
                state="running",
            )
        )
        assert mismatch.refusal == CleanupRefusal.PROVIDER_MISMATCH

    def test_local_runner_rejects_workspace_path_traversal(self, tmp_path: Path) -> None:
        runner = LocalRunner()
        instance = runner.create_instance({"machine_id": "local"})
        source = tmp_path / "input.txt"
        source.write_text("payload")

        try:
            assert (
                runner.deploy_files(
                    instance,
                    {"../outside.txt": source},
                )
                is False
            )
            assert (
                runner.download_file(instance, "../outside.txt", tmp_path / "outside.txt") is False
            )
        finally:
            runner.destroy_instance(instance)


def test_verify_gpu_tolerates_nonzero_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = LocalRunner()
    result = subprocess.CompletedProcess(args=["nvidia-smi"], returncode=1)
    monkeypatch.setattr(local_module.subprocess, "run", Mock(return_value=result))

    assert runner.verify_gpu(_instance()) is True


def test_local_runner_duplicate_create_rejects(tmp_path: Path) -> None:
    runner = LocalRunner()
    instance = runner.create_instance({"machine_id": "local"})
    try:
        with pytest.raises(RuntimeError, match="already has an active instance"):
            runner.create_instance({"machine_id": "local"})
    finally:
        runner.destroy_instance(instance)


def test_local_runner_rejects_workspace_lookup_without_create() -> None:
    runner = LocalRunner()
    with pytest.raises(RuntimeError, match="Unknown local instance"):
        runner._workspace(_instance())


def test_local_runner_launch_fails_for_missing_worker(tmp_path: Path) -> None:
    runner = LocalRunner()
    instance = runner.create_instance({"machine_id": "local"})
    try:
        assert runner.launch_worker(instance) is False  # no worker.sh deployed
    finally:
        runner.destroy_instance(instance)


def test_local_runner_progress_when_launch_not_yet_attempted(tmp_path: Path) -> None:
    runner = LocalRunner()
    instance = runner.create_instance({"machine_id": "local"})
    try:
        progress = runner.check_progress(instance)
        assert progress == {
            "running": True,
            "complete": False,
            "log_tail": "",
        }
    finally:
        runner.destroy_instance(instance)


def test_local_runner_download_file_round_trip(tmp_path: Path) -> None:
    runner = LocalRunner()
    instance = runner.create_instance({"machine_id": "local"})
    destination = tmp_path / "copy.bin"
    try:
        (runner._workspace(instance) / "blob.bin").write_text("payload")
        assert runner.download_file(instance, "blob.bin", destination) is True
        assert destination.read_text() == "payload"
        missing_destination = tmp_path / "missing.bin"
        assert runner.download_file(instance, "nope.bin", missing_destination) is False
        assert not missing_destination.exists()
    finally:
        runner.destroy_instance(instance)


def test_local_runner_download_file_rejects_traversal(tmp_path: Path) -> None:
    runner = LocalRunner()
    instance = runner.create_instance({"machine_id": "local"})
    try:
        assert runner.download_file(instance, "nested/../escape", tmp_path / "out") is False
        with pytest.raises(ValueError, match="relative workspace path"):
            runner._workspace_path(instance, "/absolute")
    finally:
        runner.destroy_instance(instance)


def test_local_runner_download_all_results_handles_subdir_and_missing(tmp_path: Path) -> None:
    runner = LocalRunner()
    instance = runner.create_instance({"machine_id": "local"})
    try:
        sub = runner._workspace(instance) / "nested"
        sub.mkdir()
        (sub / "data.txt").write_text("x")
        assert runner.download_all_results(
            instance,
            tmp_path / "out",
            remote_subdir="nested",
            critical_files={"data.txt"},
        )
        assert (
            runner.download_all_results(
                instance,
                tmp_path / "missing",
                remote_subdir="does-not-exist",
            )
            == []
        )
    finally:
        runner.destroy_instance(instance)


def test_local_runner_download_all_results_missing_critical(tmp_path: Path) -> None:
    runner = LocalRunner()
    instance = runner.create_instance({"machine_id": "local"})
    try:
        (runner._workspace(instance) / "ok.txt").write_text("ok")
        assert (
            runner.download_all_results(
                instance,
                tmp_path / "out",
                critical_files={"never.txt"},
            )
            == []
        )
    finally:
        runner.destroy_instance(instance)
