"""Run a worker locally with no cloud credentials.

Usage:
    uv run python examples/local_runner.py path/to/worker.sh path/to/input.txt

The script launches ``worker.sh`` in a temporary workspace, polls for a DONE
marker, copies results into ``outputs/local``, and tears down the workspace
on exit. Reuse this as the canonical orchestration-layer glue for local
development and CI smoke tests.

Note: the worker script must produce a ``DONE`` marker on success (e.g. an
empty file at the workspace root) and write ``worker.pid`` if you want
silent-crash detection via ``check_progress``. See
``tests/fixtures/local_runner/worker.sh`` for a minimal reference.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from vastai_gpu_runner.providers.local import LocalRunner
from vastai_gpu_runner.types import DeploymentConfig


def main(argv: list[str]) -> int:
    """Drive a local run end-to-end and return a shell exit code."""
    if not argv:
        print("usage: local_runner.py <worker.sh> [extra file ...]", file=sys.stderr)
        return 2

    files = [Path(arg).resolve() for arg in argv]
    for path in files:
        if not path.is_file():
            print(f"missing payload file: {path}", file=sys.stderr)
            return 2

    file_map = {path.name: path for path in files}
    output_dir = Path("outputs/local").resolve()

    runner = LocalRunner(DeploymentConfig(worker_script=files[0].name))
    result = runner.run_full_cycle(
        files=file_map,
        local_output_dir=output_dir,
        max_retries=1,
    )
    if not result.success or result.instance is None:
        print(f"launch failed: {result.error}", file=sys.stderr)
        return 1

    instance = result.instance
    try:
        deadline = time.monotonic() + 300.0
        while time.monotonic() < deadline:
            progress = runner.check_progress(instance)
            if progress.get("complete"):
                break
            if progress.get("worker_dead"):
                print(
                    f"worker exited without DONE: {progress.get('log_tail', '')}",
                    file=sys.stderr,
                )
                return 1
            time.sleep(1.0)
        else:
            print("timed out waiting for DONE", file=sys.stderr)
            return 1

        downloaded = runner.download_all_results(
            instance,
            output_dir,
            critical_files={"DONE"},
        )
        if not downloaded:
            print("worker completed but produced no downloadable files", file=sys.stderr)
            return 1
        print(f"collected {len(downloaded)} file(s) into {output_dir}")
        return 0
    finally:
        runner.destroy_instance(instance)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
