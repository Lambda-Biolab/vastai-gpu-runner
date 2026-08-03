"""End-to-end tests for the local subprocess runner."""

from __future__ import annotations

import time
from pathlib import Path

from vastai_gpu_runner.providers.local import LocalRunner
from vastai_gpu_runner.types import DeploymentConfig, Provider

FIXTURE_WORKER = Path(__file__).parents[1] / "fixtures" / "local_runner" / "worker.sh"


def test_local_runner_executes_worker_and_collects_files(tmp_path: Path) -> None:
    runner = LocalRunner(DeploymentConfig(worker_script="worker.sh"))
    result = runner.run_full_cycle(
        files={"worker.sh": FIXTURE_WORKER},
        local_output_dir=tmp_path / "output",
        offers=[{"machine_id": "local", "dph_total": 0.0}],
    )

    assert result.success is True
    assert result.instance is not None
    assert result.instance.provider == Provider.LOCAL
    instance = result.instance

    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            progress = runner.check_progress(instance)
            if progress.get("complete"):
                break
            assert progress.get("worker_dead") is not True
            time.sleep(0.02)
        else:
            raise AssertionError("local worker did not complete before timeout")

        output_dir = tmp_path / "output"
        downloaded = runner.download_all_results(
            instance,
            output_dir,
            critical_files={"DONE", "worker.exitcode"},
        )
        assert {Path(path).name for path in downloaded} >= {"DONE", "worker.exitcode"}
        assert (output_dir / "worker.exitcode").read_text() == "0\n"
    finally:
        assert runner.destroy_instance(instance) is True
