"""CLI command for running a worker with the local provider."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from vastai_gpu_runner.providers.local import LocalRunner
from vastai_gpu_runner.types import CloudInstance, DeploymentConfig, Provider

logger = logging.getLogger(__name__)


def run(
    files: Annotated[
        list[Path] | None,
        typer.Option(
            "--file",
            "-f",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Input or worker file to copy into the local workspace (repeatable).",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Directory for files produced by the worker."),
    ] = Path("outputs/local"),
    provider: Annotated[
        str,
        typer.Option("--provider", help="Execution provider (currently only 'local')."),
    ] = Provider.LOCAL.value,
    worker_script: Annotated[
        str,
        typer.Option("--worker-script", help="Worker script filename in the payload."),
    ] = "worker.sh",
    timeout: Annotated[
        float,
        typer.Option("--timeout", min=0.1, help="Maximum seconds to wait for worker completion."),
    ] = 300.0,
    poll_interval: Annotated[
        float,
        typer.Option("--poll-interval", min=0.0, help="Seconds between progress checks."),
    ] = 1.0,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed logs.")] = False,
) -> None:
    """Run a worker locally without cloud credentials."""
    _validate_provider(provider)
    file_map = _build_payload_map(files, worker_script)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )
    console = Console()
    runner = LocalRunner(DeploymentConfig(worker_script=worker_script))
    result = runner.run_full_cycle(
        files=file_map,
        local_output_dir=output,
        max_retries=1,
    )
    if not result.success or result.instance is None:
        console.print(f"[red]Local worker launch failed:[/red] {result.error}")
        raise typer.Exit(1)

    instance = result.instance
    try:
        completed, detail = _wait_for_completion(
            runner,
            instance,
            timeout=timeout,
            poll_interval=poll_interval,
        )
        if not completed:
            console.print(f"[red]Local worker did not complete:[/red] {detail}")
            raise typer.Exit(1)
        output_files = runner.download_all_results(
            instance,
            output,
            critical_files={"DONE"},
        )
        if not output_files:
            console.print("[red]Local worker completed but produced no downloadable files.[/red]")
            raise typer.Exit(1)
        console.print(
            f"[green]Local worker completed[/green]; collected "
            f"{len(output_files)} file(s) in {output}"
        )
    finally:
        runner.destroy_instance(instance)


def _validate_provider(provider: str) -> None:
    """Reject provider values other than ``local`` until other backends ship."""
    if provider.lower() != Provider.LOCAL.value:
        raise typer.BadParameter(
            "the run command currently only supports --provider local",
            param_hint="--provider",
        )


def _build_payload_map(
    files: list[Path] | None,
    worker_script: str,
) -> dict[str, Path]:
    """Validate payload filenames and ensure the worker script is present."""
    script_path = Path(worker_script)
    if not worker_script or script_path.is_absolute() or ".." in script_path.parts:
        raise typer.BadParameter(
            "worker script must be a relative path inside the payload",
            param_hint="--worker-script",
        )
    payload = files or []
    file_map: dict[str, Path] = {}
    for path in payload:
        if path.name in file_map:
            raise typer.BadParameter(
                f"duplicate payload filename: {path.name}",
                param_hint="--file",
            )
        file_map[path.name] = path
    if worker_script not in file_map:
        raise typer.BadParameter(
            f"worker script {worker_script!r} must be supplied with --file",
            param_hint="--file",
        )
    return file_map


def _wait_for_completion(
    runner: LocalRunner,
    instance: CloudInstance,
    *,
    timeout: float,
    poll_interval: float,
) -> tuple[bool, str]:
    """Poll a local worker until completion, failure, or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        progress = runner.check_progress(instance)
        if progress.get("complete") is True:
            return True, ""
        if progress.get("worker_dead") is True:
            detail = str(progress.get("log_tail", "worker exited without DONE"))
            return False, detail
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
    return False, f"timed out after {timeout:g} seconds"


__all__ = ["run"]
