"""Behavioral tests for the local execution CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from vastai_gpu_runner.cli import app

FIXTURE_WORKER = Path(__file__).parent / "fixtures" / "local_runner" / "worker.sh"
CLI_RUNNER = CliRunner()


def test_run_local_command_executes_and_collects_worker_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"

    result = CLI_RUNNER.invoke(
        app,
        [
            "run",
            "--provider",
            "local",
            "--file",
            str(FIXTURE_WORKER),
            "--output",
            str(output_dir),
            "--poll-interval",
            "0.02",
            "--timeout",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Local worker completed" in result.output
    assert (output_dir / "DONE").is_file()
    assert (output_dir / "worker.exitcode").read_text() == "0\n"


def test_run_rejects_unsupported_provider(tmp_path: Path) -> None:
    result = CLI_RUNNER.invoke(
        app,
        [
            "run",
            "--provider",
            "vastai",
            "--file",
            str(FIXTURE_WORKER),
            "--output",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code != 0
    assert "currently only supports" in result.output.replace("\n", " ")


def test_run_rejects_duplicate_payload_filenames(tmp_path: Path) -> None:
    second = tmp_path / "worker.sh"
    second.write_text("echo")

    result = CLI_RUNNER.invoke(
        app,
        [
            "run",
            "--file",
            str(FIXTURE_WORKER),
            "--file",
            str(second),
            "--output",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code != 0
    assert "duplicate payload filename" in result.output.replace("\n", " ")


def test_run_rejects_traversal_worker_script(tmp_path: Path) -> None:
    result = CLI_RUNNER.invoke(
        app,
        [
            "run",
            "--file",
            str(FIXTURE_WORKER),
            "--output",
            str(tmp_path / "output"),
            "--worker-script",
            "../escape.sh",
        ],
    )

    assert result.exit_code != 0
    assert "worker script" in result.output.replace("\n", " ")


def test_run_rejects_worker_script_not_in_payload(tmp_path: Path) -> None:
    other = tmp_path / "other.sh"
    other.write_text("#!/usr/bin/env bash\n")

    result = CLI_RUNNER.invoke(
        app,
        [
            "run",
            "--file",
            str(other),
            "--output",
            str(tmp_path / "output"),
            "--worker-script",
            "worker.sh",
        ],
    )

    assert result.exit_code != 0
    assert "worker script" in result.output
    assert "--file" in result.output


def test_run_reports_timeout_when_worker_hangs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hang_worker = tmp_path / "worker.sh"
    hang_worker.write_text("#!/usr/bin/env bash\nsleep 5\n")
    output_dir = tmp_path / "output"

    class _HangingRunner:
        def __init__(self, _config: object) -> None:
            pass

        def run_full_cycle(self, **kwargs: object) -> object:
            from vastai_gpu_runner.types import (
                CloudInstance,
                DeploymentResult,
            )

            del kwargs
            return DeploymentResult(
                success=True,
                instance=CloudInstance(instance_id="local"),
            )

        def check_progress(self, instance: object) -> dict[str, object]:
            del instance
            return {"running": True, "complete": False, "log_tail": ""}

        def download_all_results(self, *args: object, **kwargs: object) -> list[str]:
            return []

        def destroy_instance(self, instance: object) -> bool:
            return True

    monkeypatch.setattr("vastai_gpu_runner.cli_run.LocalRunner", _HangingRunner)

    result = CLI_RUNNER.invoke(
        app,
        [
            "run",
            "--file",
            str(hang_worker),
            "--output",
            str(output_dir),
            "--timeout",
            "0.2",
            "--poll-interval",
            "0.05",
        ],
    )

    assert result.exit_code != 0
    assert "did not complete" in result.output.replace("\n", " ")


def test_run_reports_launch_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hang_worker = tmp_path / "worker.sh"
    hang_worker.write_text("#!/usr/bin/env bash\nsleep 5\n")
    output_dir = tmp_path / "output"

    class _FailingRunner:
        def __init__(self, _config: object) -> None:
            pass

        def run_full_cycle(self, **kwargs: object) -> object:
            from vastai_gpu_runner.types import DeploymentResult

            del kwargs
            return DeploymentResult(success=False, error="boot timeout")

    monkeypatch.setattr("vastai_gpu_runner.cli_run.LocalRunner", _FailingRunner)

    result = CLI_RUNNER.invoke(
        app,
        [
            "run",
            "--file",
            str(hang_worker),
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code != 0
    assert "launch failed" in result.output.replace("\n", " ")


def test_run_reports_no_files_after_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hang_worker = tmp_path / "worker.sh"
    hang_worker.write_text("#!/usr/bin/env bash\nsleep 5\n")
    output_dir = tmp_path / "output"

    class _EmptyRunner:
        def __init__(self, _config: object) -> None:
            pass

        def run_full_cycle(self, **kwargs: object) -> object:
            from vastai_gpu_runner.types import (
                CloudInstance,
                DeploymentResult,
            )

            del kwargs
            return DeploymentResult(
                success=True,
                instance=CloudInstance(instance_id="local"),
            )

        def check_progress(self, instance: object) -> dict[str, object]:
            del instance
            return {"running": False, "complete": True}

        def download_all_results(self, *args: object, **kwargs: object) -> list[str]:
            return []

        def destroy_instance(self, instance: object) -> bool:
            return True

    monkeypatch.setattr("vastai_gpu_runner.cli_run.LocalRunner", _EmptyRunner)

    result = CLI_RUNNER.invoke(
        app,
        [
            "run",
            "--file",
            str(hang_worker),
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code != 0
    assert "no downloadable files" in result.output.replace("\n", " ")
