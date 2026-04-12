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
