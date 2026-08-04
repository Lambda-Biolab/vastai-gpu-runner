"""Behavioral tests for the local execution CLI.

Layered as:
1. Pure-function tests for the validation helpers (no CLI / Typer involved).
2. Thin CLI smoke tests that assert on exit codes only (Typer exit codes
   are stable: 2 for usage errors, 1 for runtime errors, 0 for success).
3. A single end-to-end CLI test exercising the real ``LocalRunner`` and a
   fixture ``worker.sh`` to cover the wiring contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from vastai_gpu_runner.cli import app
from vastai_gpu_runner.cli_run import (
    LocalRunValidationError,
    _build_payload_map,
    _validate_provider,
)

FIXTURE_WORKER = Path(__file__).parent / "fixtures" / "local_runner" / "worker.sh"
CLI_RUNNER = CliRunner()

# Typer exit codes (Click convention):
_TYPER_USAGE_ERROR = 2
_TYPER_RUNTIME_ERROR = 1
_TYPER_SUCCESS = 0


# ---------------------------------------------------------------------------
# Pure-function validation tests — stable across Typer / Rich versions.
# ---------------------------------------------------------------------------


class TestValidateProvider:
    def test_local_passes(self) -> None:
        _validate_provider("local")

    def test_local_is_case_insensitive(self) -> None:
        _validate_provider("LOCAL")

    def test_other_provider_raises_with_option(self) -> None:
        with pytest.raises(LocalRunValidationError) as excinfo:
            _validate_provider("vastai")
        assert excinfo.value.option == "--provider"
        assert "local" in str(excinfo.value)


class TestBuildPayloadMap:
    def test_returns_name_to_path_map(self, tmp_path: Path) -> None:
        worker = tmp_path / "worker.sh"
        worker.write_text("#!/usr/bin/env bash\n")
        other = tmp_path / "input.txt"
        other.write_text("data")
        result = _build_payload_map([worker, other], "worker.sh")
        assert result == {"worker.sh": worker, "input.txt": other}

    def test_empty_files_rejected_when_worker_missing(self) -> None:
        with pytest.raises(LocalRunValidationError) as excinfo:
            _build_payload_map(None, "worker.sh")
        assert excinfo.value.option == "--file"
        assert "worker.sh" in str(excinfo.value)

    def test_duplicate_filename_rejected(self, tmp_path: Path) -> None:
        a = tmp_path / "worker.sh"
        a.write_text("#!/usr/bin/env bash\n")
        b = tmp_path / "worker.sh"
        b.write_text("#!/usr/bin/env bash\n")
        with pytest.raises(LocalRunValidationError) as excinfo:
            _build_payload_map([a, b], "worker.sh")
        assert excinfo.value.option == "--file"
        assert "duplicate" in str(excinfo.value)

    def test_traversal_worker_script_rejected(self, tmp_path: Path) -> None:
        worker = tmp_path / "worker.sh"
        worker.write_text("#!/usr/bin/env bash\n")
        with pytest.raises(LocalRunValidationError) as excinfo:
            _build_payload_map([worker], "../escape.sh")
        assert excinfo.value.option == "--worker-script"

    def test_absolute_worker_script_rejected(self, tmp_path: Path) -> None:
        worker = tmp_path / "worker.sh"
        worker.write_text("#!/usr/bin/env bash\n")
        with pytest.raises(LocalRunValidationError) as excinfo:
            _build_payload_map([worker], "/abs/worker.sh")
        assert excinfo.value.option == "--worker-script"

    def test_empty_worker_script_rejected(self, tmp_path: Path) -> None:
        worker = tmp_path / "worker.sh"
        worker.write_text("#!/usr/bin/env bash\n")
        with pytest.raises(LocalRunValidationError) as excinfo:
            _build_payload_map([worker], "")
        assert excinfo.value.option == "--worker-script"


# ---------------------------------------------------------------------------
# CLI smoke tests — assert on exit codes (Typer / Click convention).
# ---------------------------------------------------------------------------


def test_run_local_command_executes_and_collects_worker_output(tmp_path: Path) -> None:
    """End-to-end CLI invokes the real LocalRunner against the fixture."""
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

    assert result.exit_code == _TYPER_SUCCESS, result.output
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
    assert result.exit_code == _TYPER_USAGE_ERROR


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
    assert result.exit_code == _TYPER_USAGE_ERROR


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
    assert result.exit_code == _TYPER_USAGE_ERROR


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
    assert result.exit_code == _TYPER_USAGE_ERROR


# ---------------------------------------------------------------------------
# CLI runtime-error smoke tests — exercise fake runners to verify exit codes.
# Substring assertions would couple the test to Typer's rendered output;
# exit codes are part of the CLI's public contract.
# ---------------------------------------------------------------------------


def _make_hanging_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    class _HangingRunner:
        def __init__(self, _config: object) -> None:
            pass

        def run_full_cycle(self, **kwargs: object) -> Any:
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


def test_run_reports_timeout_when_worker_hangs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hang_worker = tmp_path / "worker.sh"
    hang_worker.write_text("#!/usr/bin/env bash\nsleep 5\n")
    output_dir = tmp_path / "output"
    _make_hanging_runner(monkeypatch)

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
    assert result.exit_code == _TYPER_RUNTIME_ERROR


def test_run_reports_launch_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hang_worker = tmp_path / "worker.sh"
    hang_worker.write_text("#!/usr/bin/env bash\nsleep 5\n")
    output_dir = tmp_path / "output"

    class _FailingRunner:
        def __init__(self, _config: object) -> None:
            pass

        def run_full_cycle(self, **kwargs: object) -> Any:
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
    assert result.exit_code == _TYPER_RUNTIME_ERROR


def test_run_reports_no_files_after_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hang_worker = tmp_path / "worker.sh"
    hang_worker.write_text("#!/usr/bin/env bash\nsleep 5\n")
    output_dir = tmp_path / "output"

    class _EmptyRunner:
        def __init__(self, _config: object) -> None:
            pass

        def run_full_cycle(self, **kwargs: object) -> Any:
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
    assert result.exit_code == _TYPER_RUNTIME_ERROR
