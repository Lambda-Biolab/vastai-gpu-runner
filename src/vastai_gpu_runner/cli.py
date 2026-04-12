"""CLI for vastai-gpu-runner — credential checks, instance management, cost estimation.

Usage::

    vastai-gpu-runner check       # Verify Vast.ai + R2 credentials
    vastai-gpu-runner instances    # List active instances with ownership info
    vastai-gpu-runner estimate     # Cost/time scaling table
    vastai-gpu-runner cleanup      # Destroy orphaned instances
"""

from __future__ import annotations

import json
import logging
from typing import Annotated

import typer

app = typer.Typer(
    name="vastai-gpu-runner",
    help="Cloud GPU orchestration for Vast.ai — credentials, instances, cost estimation.",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI output."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )


# ---------------------------------------------------------------------------
# check — verify credentials
# ---------------------------------------------------------------------------


@app.command()
def check(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show details")] = False,
) -> None:
    """Verify Vast.ai API key and R2 storage credentials.

    Checks:
    - Vast.ai CLI installed and API key valid (can list instances)
    - R2 credentials present and bucket reachable
    """
    _setup_logging(verbose)
    from rich.console import Console

    console = Console()
    all_ok = True

    # --- Vast.ai CLI ---
    console.print("[bold]Vast.ai CLI[/bold]")
    try:
        from vastai_gpu_runner.providers.vastai import vastai_cmd

        raw = vastai_cmd(["show", "instances", "--raw"], timeout=15)
        instances = json.loads(raw)
        console.print(f"  [green]OK[/green] — API key valid, {len(instances)} instance(s)")
    except RuntimeError as exc:
        console.print(f"  [red]FAIL[/red] — {exc}")
        all_ok = False
    except json.JSONDecodeError:
        console.print("  [red]FAIL[/red] — API returned invalid JSON")
        all_ok = False

    # --- R2 credentials ---
    console.print("[bold]R2 Storage[/bold]")
    try:
        from vastai_gpu_runner.storage.r2 import get_r2_client, load_r2_env

        env = load_r2_env()
        if not env.get("R2_ENDPOINT"):
            import os

            if os.environ.get("R2_ENDPOINT"):
                env["R2_ENDPOINT"] = os.environ["R2_ENDPOINT"]
            else:
                console.print("  [red]FAIL[/red] — R2_ENDPOINT not set in ~/.cloud-credentials")
                all_ok = False
                raise typer.Exit(1)

        client = get_r2_client()
        # Try a lightweight operation to verify connectivity
        client.list_objects_v2(Bucket="dv-results", MaxKeys=1)
        console.print("  [green]OK[/green] — R2 reachable")
        if verbose:
            console.print(f"    Endpoint: {env.get('R2_ENDPOINT', 'N/A')}")
    except Exception as exc:
        if "FAIL" not in str(exc):
            console.print(f"  [red]FAIL[/red] — {exc}")
        all_ok = False

    # --- Summary ---
    if all_ok:
        console.print("\n[bold green]All checks passed.[/bold green]")
    else:
        console.print("\n[bold red]Some checks failed.[/bold red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# instances — list active instances
# ---------------------------------------------------------------------------


@app.command()
def instances(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show full details")] = False,
    allowed_images: Annotated[
        str | None,
        typer.Option("--allowed-images", help="Comma-separated Docker images for ownership check"),
    ] = None,
) -> None:
    """List active Vast.ai instances with status and ownership info.

    Shows instance ID, GPU, status, label, cost/hr, and whether the
    instance belongs to your project (based on --allowed-images).
    """
    _setup_logging(verbose)
    from rich.console import Console
    from rich.table import Table

    console = Console()

    try:
        from vastai_gpu_runner.providers.vastai import vastai_cmd

        raw = vastai_cmd(["show", "instances", "--raw"], timeout=15)
        insts = json.loads(raw)
    except (RuntimeError, json.JSONDecodeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if not insts:
        console.print("No active instances.")
        return

    images_set = frozenset(allowed_images.split(",")) if allowed_images else None

    table = Table(title=f"{len(insts)} Active Instance(s)")
    table.add_column("ID", style="cyan")
    table.add_column("GPU")
    table.add_column("Status")
    table.add_column("Label")
    table.add_column("$/hr", justify="right", style="green")
    if images_set:
        table.add_column("Owned", justify="center")

    for inst in insts:
        iid = str(inst.get("id", ""))
        gpu = str(inst.get("gpu_name", ""))
        status = str(inst.get("actual_status", ""))
        label = str(inst.get("label", ""))
        cost = f"${float(inst.get('dph_total', 0)):.3f}"
        image = str(inst.get("image_uuid", ""))

        row = [iid, gpu, status, label, cost]
        if images_set:
            owned = any(img.split(":")[0] in image for img in images_set) or image in images_set
            row.append("[green]yes[/green]" if owned else "[red]no[/red]")
        table.add_row(*row)

    console.print(table)

    # Cost summary
    total_hourly = sum(float(i.get("dph_total", 0)) for i in insts)
    running = [i for i in insts if i.get("actual_status") == "running"]
    console.print(f"\nRunning: {len(running)}/{len(insts)}, Total: ${total_hourly:.2f}/hr")


# ---------------------------------------------------------------------------
# estimate — cost/time scaling table
# ---------------------------------------------------------------------------


@app.command()
def estimate(
    work_hours: Annotated[
        float, typer.Option("--work-hours", "-w", help="Total work in RTX 4090-equivalent hours")
    ],
    gpu_counts: Annotated[
        str, typer.Option("--gpus", "-g", help="Comma-separated cloud GPU counts")
    ] = "0,2,4,8,16",
    gpu_type: Annotated[
        str | None, typer.Option("--gpu-type", help="Preferred cloud GPU type")
    ] = None,
    live_pricing: Annotated[
        bool, typer.Option("--live/--no-live", help="Query live Vast.ai pricing")
    ] = False,
    output_json: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
) -> None:
    """Show cost/time scaling table for a GPU workload.

    Provide the total work in RTX 4090-equivalent hours and get a
    scaling table showing wall time and cost for different GPU counts.
    """
    _setup_logging()

    counts = [int(x.strip()) for x in gpu_counts.split(",")]

    if live_pricing:
        from vastai_gpu_runner.estimator.pricing import query_vastai_pricing

        pricing = query_vastai_pricing()
    else:
        from vastai_gpu_runner.estimator.core import fallback_pricing

        pricing = fallback_pricing()

    from vastai_gpu_runner.estimator.core import (
        EstimateResult,
        build_scaling_table,
    )

    rows = build_scaling_table(
        total_work_hours_base=work_hours,
        cloud_gpu_counts=counts,
        pricing=pricing,
        cloud_gpu_type=gpu_type,
    )

    result = EstimateResult(
        workload="custom",
        description=f"{work_hours}h of GPU work",
        num_items=0,
        pricing=pricing,
        scaling_table=rows,
    )

    if output_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        from rich.console import Console

        Console().print(result.to_rich_table())


# ---------------------------------------------------------------------------
# cleanup — destroy orphaned instances
# ---------------------------------------------------------------------------


@app.command()
def cleanup(
    label_prefix: Annotated[
        str, typer.Option("--label", "-l", help="Label prefix to match (e.g. 'myproject-')")
    ],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be destroyed without acting")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Destroy orphaned Vast.ai instances matching a label prefix.

    Only destroys instances whose label starts with the given prefix.
    Use --dry-run to preview without destroying.
    """
    _setup_logging(verbose)
    from rich.console import Console

    console = Console()

    try:
        from vastai_gpu_runner.providers.vastai import vastai_cmd

        raw = vastai_cmd(["show", "instances", "--raw"], timeout=15)
        insts = json.loads(raw)
    except (RuntimeError, json.JSONDecodeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    matches = [i for i in insts if str(i.get("label", "")).startswith(label_prefix)]

    if not matches:
        console.print(f"No instances matching label prefix '{label_prefix}'.")
        return

    console.print(f"Found {len(matches)} instance(s) matching '{label_prefix}':")
    for inst in matches:
        console.print(
            f"  {inst['id']}: {inst.get('gpu_name', '?')} "
            f"status={inst.get('actual_status', '?')} "
            f"label={inst.get('label', '')}"
        )

    if dry_run:
        console.print("\n[yellow]Dry run — no instances destroyed.[/yellow]")
        return

    if not typer.confirm(f"\nDestroy {len(matches)} instance(s)?"):
        console.print("Aborted.")
        raise typer.Exit(0)

    destroyed = 0
    for inst in matches:
        iid = str(inst["id"])
        try:
            vastai_cmd(["destroy", "instance", iid], timeout=15)
            console.print(f"  [green]Destroyed[/green] {iid}")
            destroyed += 1
        except RuntimeError as exc:
            console.print(f"  [red]Failed[/red] {iid}: {exc}")

    console.print(f"\nDestroyed {destroyed}/{len(matches)} instance(s).")


if __name__ == "__main__":
    app()
