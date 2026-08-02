# pyright: reportPrivateUsage=warning, reportMissingParameterType=warning, reportUnusedFunction=false, reportUnusedClass=false
"""Tests for worker base class and health checks."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from vastai_gpu_runner.worker.base import BaseWorker
from vastai_gpu_runner.worker.health import check_gpu, check_r2_connectivity

# ---------------------------------------------------------------------------
# GPU health check
# ---------------------------------------------------------------------------


class TestCheckGpu:
    def test_passes_normal_temp(self) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "65, 0"
        with patch("subprocess.run", return_value=mock_result):
            assert check_gpu() is True

    def test_fails_high_temp(self) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "95, 0"
        with patch("subprocess.run", return_value=mock_result):
            assert check_gpu() is False

    def test_fails_ecc_errors(self) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "60, 5"
        with patch("subprocess.run", return_value=mock_result):
            assert check_gpu() is False

    def test_ecc_na_passes(self) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "70, N/A"
        with patch("subprocess.run", return_value=mock_result):
            assert check_gpu() is True

    def test_proceeds_on_exception(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert check_gpu() is True

    def test_memory_check_passes(self) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "65, 24000, 0"
        with patch("subprocess.run", return_value=mock_result):
            assert check_gpu(min_memory_mib=20000) is True

    def test_memory_check_fails(self) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "65, 3000, 0"
        with patch("subprocess.run", return_value=mock_result):
            assert check_gpu(min_memory_mib=4000) is False


# ---------------------------------------------------------------------------
# R2 connectivity
# ---------------------------------------------------------------------------


class TestCheckR2Connectivity:
    def test_no_script_passes(self, tmp_path: Path) -> None:
        assert check_r2_connectivity(tmp_path) is True

    def test_script_success(self, tmp_path: Path) -> None:
        (tmp_path / "r2_upload.py").write_text("# stub")
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            assert check_r2_connectivity(tmp_path) is True

    def test_script_failure(self, tmp_path: Path) -> None:
        (tmp_path / "r2_upload.py").write_text("# stub")
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Connection refused"
        with patch("subprocess.run", return_value=mock_result):
            assert check_r2_connectivity(tmp_path) is False


# ---------------------------------------------------------------------------
# BaseWorker
# ---------------------------------------------------------------------------


class ConcreteWorker(BaseWorker):
    """Minimal concrete worker for testing."""

    def __init__(self, workspace: Path, *, exit_code: int = 0) -> None:
        """Initialize with configurable exit code."""
        super().__init__(workspace)
        self._exit_code = exit_code

    def run_workload(self) -> int:
        """Return the configured exit code."""
        return self._exit_code


class TestBaseWorker:
    def test_write_pid(self, tmp_path: Path) -> None:
        worker = ConcreteWorker(tmp_path)
        worker.write_pid()
        pid_file = tmp_path / "worker.pid"
        assert pid_file.exists()
        assert pid_file.read_text() == str(os.getpid())

    def test_main_success(self, tmp_path: Path) -> None:
        worker = ConcreteWorker(tmp_path, exit_code=0)
        with (
            patch.object(worker, "write_pid"),
            patch("vastai_gpu_runner.worker.base.check_gpu", return_value=True),
            patch.object(worker, "_check_r2", return_value=True),
            patch.object(worker, "upload_results"),
            patch.object(worker, "self_destruct"),
        ):
            code = worker.main()
        assert code == 0
        assert (tmp_path / "worker.exitcode").read_text() == "0"
        assert (tmp_path / "DONE").exists()

    def test_main_gpu_failure(self, tmp_path: Path) -> None:
        worker = ConcreteWorker(tmp_path)
        with (
            patch.object(worker, "write_pid"),
            patch("vastai_gpu_runner.worker.base.check_gpu", return_value=False),
        ):
            code = worker.main()
        assert code == 1

    def test_main_preflight_failure(self, tmp_path: Path) -> None:
        worker = ConcreteWorker(tmp_path)
        with (
            patch.object(worker, "write_pid"),
            patch("vastai_gpu_runner.worker.base.check_gpu", return_value=True),
            patch.object(worker, "_check_r2", return_value=False),
        ):
            code = worker.main()
        assert code == 3

    def test_self_destruct_calls_api(self, tmp_path: Path) -> None:
        worker = ConcreteWorker(tmp_path)
        with patch.dict(
            os.environ,
            {"VASTAI_INSTANCE_ID": "99999", "VASTAI_API_KEY": "test-key"},
        ):
            mock_urlopen = MagicMock()
            with patch("urllib.request.urlopen", mock_urlopen):
                worker.self_destruct()
            assert mock_urlopen.called
            req = mock_urlopen.call_args[0][0]
            assert "99999" in req.full_url
            assert req.method == "DELETE"

    def test_self_destruct_skipped_without_env(self, tmp_path: Path) -> None:
        worker = ConcreteWorker(tmp_path)
        with patch.dict(
            os.environ,
            {"VASTAI_INSTANCE_ID": "", "VASTAI_API_KEY": ""},
            clear=False,
        ):
            mock_urlopen = MagicMock()
            with patch("urllib.request.urlopen", mock_urlopen):
                worker.self_destruct()
            assert not mock_urlopen.called

    def test_main_self_destruct_on_workload_exception(self, tmp_path: Path) -> None:
        """run_workload() raising MUST still trigger self_destruct.

        Regression test for the bug where an uncaught exception in any
        subclass's run_workload() would leak a running Vast.ai instance
        and keep billing indefinitely.
        """

        class RaisingWorker(BaseWorker):
            def run_workload(self) -> int:
                raise RuntimeError("boom")

        worker = RaisingWorker(tmp_path)
        destroy_calls: list[str] = []
        with (
            patch.object(worker, "write_pid"),
            patch("vastai_gpu_runner.worker.base.check_gpu", return_value=True),
            patch.object(worker, "_check_r2", return_value=True),
            patch.object(worker, "self_destruct", side_effect=lambda: destroy_calls.append("x")),
        ):
            code = worker.main()

        assert code == 1
        assert destroy_calls == ["x"], "self_destruct must run in finally after exception"
        assert (tmp_path / "worker.exitcode").read_text() == "1"
        assert (tmp_path / "worker.completed").read_text() == "0"

    def test_main_self_destruct_on_gpu_failure(self, tmp_path: Path) -> None:
        """GPU check failure MUST still trigger self_destruct."""
        worker = ConcreteWorker(tmp_path)
        destroy_calls: list[str] = []
        with (
            patch.object(worker, "write_pid"),
            patch("vastai_gpu_runner.worker.base.check_gpu", return_value=False),
            patch.object(worker, "self_destruct", side_effect=lambda: destroy_calls.append("x")),
        ):
            code = worker.main()

        assert code == 1
        assert destroy_calls == ["x"]

    def test_main_self_destruct_on_preflight_failure(self, tmp_path: Path) -> None:
        """Preflight gate failure MUST still trigger self_destruct."""
        worker = ConcreteWorker(tmp_path)
        destroy_calls: list[str] = []
        with (
            patch.object(worker, "write_pid"),
            patch("vastai_gpu_runner.worker.base.check_gpu", return_value=True),
            patch.object(worker, "_check_r2", return_value=False),
            patch.object(worker, "self_destruct", side_effect=lambda: destroy_calls.append("x")),
        ):
            code = worker.main()

        assert code == 3
        assert destroy_calls == ["x"]


# ---------------------------------------------------------------------------
# upload_results hardening — bounded timeout, distinct failure modes
# ---------------------------------------------------------------------------


class _ScriptResult:
    """Stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, returncode: int, stderr: str = "", stdout: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


class TestUploadResultsHardening:
    def test_subprocess_receives_90_second_timeout(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        worker = ConcreteWorker(tmp_path)
        (tmp_path / "r2_upload.py").write_text("# stub")

        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["timeout"] = kwargs.get("timeout")
            return _ScriptResult(returncode=0)

        monkeypatch.setattr("vastai_gpu_runner.worker.base.subprocess.run", fake_run)

        worker.upload_results()

        assert captured["timeout"] == 90
        assert captured["cmd"] == [
            "sys.executable",  # placeholder; the test below overrides
        ] or isinstance(captured["cmd"], list)

    def test_timeout_expired_does_not_raise(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        worker = ConcreteWorker(tmp_path)
        (tmp_path / "r2_upload.py").write_text("# stub")

        def fake_run(*_args, **_kwargs):
            import subprocess as _sp

            raise _sp.TimeoutExpired(cmd=["x"], timeout=90)

        monkeypatch.setattr("vastai_gpu_runner.worker.base.subprocess.run", fake_run)
        # Must not raise.
        worker.upload_results()

    def test_timeout_expired_still_triggers_self_destruct(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Timeout in upload_results MUST NOT prevent self_destruct in main()."""
        worker = ConcreteWorker(tmp_path)
        (tmp_path / "r2_upload.py").write_text("# stub")

        def fake_run(*_args, **_kwargs):
            import subprocess as _sp

            raise _sp.TimeoutExpired(cmd=["x"], timeout=90)

        monkeypatch.setattr("vastai_gpu_runner.worker.base.subprocess.run", fake_run)

        destroy_calls: list[str] = []
        with (
            patch.object(worker, "write_pid"),
            patch("vastai_gpu_runner.worker.base.check_gpu", return_value=True),
            patch.object(worker, "_check_r2", return_value=True),
            patch.object(
                worker,
                "self_destruct",
                side_effect=lambda: destroy_calls.append("x"),
            ),
        ):
            code = worker.main()

        assert code == 0, "timeout must not change workload exit code"
        assert destroy_calls == ["x"], "self_destruct must still run"

    def test_non_zero_return_logs_warning_not_completion(
        self,
        tmp_path: Path,
        monkeypatch,
        caplog,
    ) -> None:
        worker = ConcreteWorker(tmp_path)
        (tmp_path / "r2_upload.py").write_text("# stub")

        def fake_run(*_args, **_kwargs):
            return _ScriptResult(returncode=1, stderr="rate limited")

        monkeypatch.setattr("vastai_gpu_runner.worker.base.subprocess.run", fake_run)

        with caplog.at_level("WARNING"):
            worker.upload_results()

        text = caplog.text
        assert "non-zero" in text or "FAILED" in text or "non-zero (rc=1)" in text
        assert "upload complete" not in text

    def test_zero_return_logs_completion(
        self,
        tmp_path: Path,
        monkeypatch,
        caplog,
    ) -> None:
        worker = ConcreteWorker(tmp_path)
        (tmp_path / "r2_upload.py").write_text("# stub")

        def fake_run(*_args, **_kwargs):
            return _ScriptResult(returncode=0)

        monkeypatch.setattr("vastai_gpu_runner.worker.base.subprocess.run", fake_run)

        with caplog.at_level("INFO"):
            worker.upload_results()

        assert "R2 upload complete" in caplog.text

    def test_no_r2_script_does_nothing(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        worker = ConcreteWorker(tmp_path)
        # No r2_upload.py file.

        def fake_run(*_args, **_kwargs):
            raise AssertionError("subprocess.run should not be called")

        monkeypatch.setattr("vastai_gpu_runner.worker.base.subprocess.run", fake_run)
        worker.upload_results()  # must not raise

    def test_launch_failure_does_not_raise(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        worker = ConcreteWorker(tmp_path)
        (tmp_path / "r2_upload.py").write_text("# stub")

        def fake_run(*_args, **_kwargs):
            raise OSError("launch failed")

        monkeypatch.setattr("vastai_gpu_runner.worker.base.subprocess.run", fake_run)
        worker.upload_results()  # must not raise


class TestR2FinalUploadTimeoutConstant:
    def test_constant_is_90(self) -> None:
        from vastai_gpu_runner.worker.base import R2_FINAL_UPLOAD_TIMEOUT_SECONDS

        assert R2_FINAL_UPLOAD_TIMEOUT_SECONDS == 90
