"""Tests for SSH utility functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from vastai_gpu_runner.ssh import scp_download, scp_upload, ssh_cmd
from vastai_gpu_runner.types import CloudInstance


class TestSshCmd:
    def test_success(self) -> None:
        inst = CloudInstance(ssh_host="test.host", ssh_port=22022)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "hello\n"
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            rc, output = ssh_cmd(inst, "echo hello")
        assert rc == 0
        assert output == "hello"
        # Verify SSH options
        args = mock_run.call_args[0][0]
        assert "-p" in args
        assert "22022" in args
        assert "StrictHostKeyChecking=no" in " ".join(args)

    def test_timeout_returns_minus_one(self) -> None:
        inst = CloudInstance(ssh_host="test.host", ssh_port=22)
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ssh", 30)):
            rc, output = ssh_cmd(inst, "sleep 100")
        assert rc == -1
        assert "timeout" in output.lower()

    def test_ssh_not_found(self) -> None:
        inst = CloudInstance(ssh_host="test.host")
        with patch("subprocess.run", side_effect=FileNotFoundError):
            rc, _output = ssh_cmd(inst, "ls")
        assert rc == -1


class TestScpUpload:
    def test_success(self, tmp_path: Path) -> None:
        inst = CloudInstance(ssh_host="test.host", ssh_port=22022)
        local_file = tmp_path / "test.txt"
        local_file.write_text("data")
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            assert scp_upload(inst, local_file, "/remote/test.txt") is True

    def test_failure(self, tmp_path: Path) -> None:
        inst = CloudInstance(ssh_host="test.host", ssh_port=22022)
        local_file = tmp_path / "test.txt"
        local_file.write_text("data")
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Permission denied"
        with patch("subprocess.run", return_value=mock_result):
            assert scp_upload(inst, local_file, "/remote/test.txt") is False


class TestScpDownload:
    def test_success(self, tmp_path: Path) -> None:
        inst = CloudInstance(ssh_host="test.host", ssh_port=22022)
        local_file = tmp_path / "downloaded.txt"
        mock_result = MagicMock()
        mock_result.returncode = 0

        def side_effect(*args, **kwargs):
            local_file.write_text("downloaded")
            return mock_result

        with patch("subprocess.run", side_effect=side_effect):
            assert scp_download(inst, "/remote/file.txt", local_file) is True
        assert local_file.exists()
