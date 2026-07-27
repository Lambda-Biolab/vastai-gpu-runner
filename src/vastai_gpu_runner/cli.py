"""CLI for vastai-gpu-runner — credential checks, instance management, cost estimation.

Usage::

    vastai-gpu-runner check       # Verify Vast.ai + R2 credentials
    vastai-gpu-runner instances    # List active instances with ownership info
    vastai-gpu-runner estimate     # Cost/time scaling table
    vastai-gpu-runner cleanup      # Destroy orphaned instances

The v4 architecture routes every composition root through the canonical
v4 types — ``VastaiProviderConfig``, ``VastaiRunner.from_config``,
``BatchOrchestrator(cleanup_policy=...)``, and
``build_vastai_cleanup_policy``. The CLI is the one place that builds
those objects. Empty ``--allowed-images`` is **fail-closed** (empty
set, refuses every image), not opt-out.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from vastai_gpu_runner.cleanup_policy import (
    InstanceCandidate,
    OwnershipPolicy,
    ProviderCleanupPolicy,
)
from vastai_gpu_runner.providers.destroy_adapters.vastai import CredentialResolution

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
    vastai_ok = _check_vastai(console)
    r2_ok = _check_r2(console, verbose=verbose)

    if vastai_ok and r2_ok:
        console.print("\n[bold green]All checks passed.[/bold green]")
        return
    console.print("\n[bold red]Some checks failed.[/bold red]")
    raise typer.Exit(1)


def _check_vastai(console: object) -> bool:
    """Run the Vast.ai API key check via ``list_vastai_instances``.

    v4 composition: ``read_vastai_api_key()`` resolves the credential
    state, then ``list_vastai_instances(credentials=...)`` is the
    canonical enumeration. ``EXPLICITLY_DISABLED`` is reported as a
    valid (opt-out) state; ``ABSENT`` returns ``[]`` because ambient
    CLI fallback requires interactive shell context the check command
    does not have.
    """
    from vastai_gpu_runner.providers.destroy_adapters.vastai import (
        CredentialState,
        read_vastai_api_key,
    )
    from vastai_gpu_runner.providers.vastai import list_vastai_instances

    console.print("[bold]Vast.ai CLI[/bold]")  # type: ignore[attr-defined]
    credentials = read_vastai_api_key()
    if credentials.state == CredentialState.EXPLICITLY_DISABLED:
        console.print(  # type: ignore[attr-defined]
            "  [yellow]OK[/yellow] — credentials explicitly disabled",
        )
        return True
    try:
        instances = list_vastai_instances(credentials=credentials)
    except RuntimeError as exc:
        console.print(f"  [red]FAIL[/red] — {exc}")  # type: ignore[attr-defined]
        return False
    console.print(  # type: ignore[attr-defined]
        f"  [green]OK[/green] — API key valid, {len(instances)} instance(s)",
    )
    return True


def _check_r2(console: object, *, verbose: bool) -> bool:
    """Run the R2 credentials + connectivity check. Returns True on success."""
    console.print("[bold]R2 Storage[/bold]")  # type: ignore[attr-defined]
    env = _resolve_r2_endpoint(console)
    if env is None:
        return False
    try:
        from vastai_gpu_runner.storage.r2 import get_r2_client

        client = get_r2_client()
        client.list_objects_v2(Bucket="dv-results", MaxKeys=1)
    except Exception as exc:
        console.print(f"  [red]FAIL[/red] — {exc}")  # type: ignore[attr-defined]
        return False
    console.print("  [green]OK[/green] — R2 reachable")  # type: ignore[attr-defined]
    if verbose:
        console.print(  # type: ignore[attr-defined]
            f"    Endpoint: {env.get('R2_ENDPOINT', 'N/A')}",
        )
    return True


def _resolve_r2_endpoint(console: object) -> dict[str, str] | None:
    """Resolve R2 endpoint from config file, falling back to env vars."""
    import os

    from vastai_gpu_runner.storage.r2 import load_r2_env

    env = load_r2_env()
    if env.get("R2_ENDPOINT"):
        return env
    if os.environ.get("R2_ENDPOINT"):
        env["R2_ENDPOINT"] = os.environ["R2_ENDPOINT"]
        return env
    console.print(  # type: ignore[attr-defined]
        "  [red]FAIL[/red] — R2_ENDPOINT not set in ~/.cloud-credentials",
    )
    return None


# ---------------------------------------------------------------------------
# instances — list active instances
# ---------------------------------------------------------------------------


@app.command()
def instances(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show full details")] = False,
    allowed_images: Annotated[
        str | None,
        typer.Option(
            "--allowed-images",
            "--owned-images",  # alias
            help=(
                "Comma-separated Docker images owned by this project. "
                "Used for the 'Owned' column. Empty string → fail-closed "
                "(every instance shown as not owned)."
            ),
        ),
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

    from vastai_gpu_runner.cleanup_policy import OwnershipPolicy
    from vastai_gpu_runner.providers.destroy_adapters.vastai import (
        read_vastai_api_key,
    )
    from vastai_gpu_runner.providers.vastai import list_vastai_instances

    candidates = list_vastai_instances(credentials=read_vastai_api_key())
    if not candidates:
        console.print("No active instances.")
        return

    # Comma trimming matches the cleanup command. None → opt-out.
    if allowed_images is None:
        ownership = OwnershipPolicy()
    else:
        images = frozenset(item.strip() for item in allowed_images.split(",") if item.strip())
        ownership = OwnershipPolicy(owned_images=images)

    table = Table(title=f"{len(candidates)} Active Instance(s)")
    table.add_column("ID", style="cyan")
    table.add_column("GPU")
    table.add_column("Status")
    table.add_column("Label")
    table.add_column("$/hr", justify="right", style="green")
    table.add_column("Owned", justify="center")

    total_hourly = 0.0
    running = 0
    for c in candidates:
        owned = ownership.matches(c.ownership_key)
        total_hourly += c.cost_per_hour
        if c.state == "running":
            running += 1
        table.add_row(
            c.instance_id,
            c.gpu_model or "?",
            c.state or "?",
            c.label,
            f"${c.cost_per_hour:.3f}",
            "[green]yes[/green]" if owned else "[red]no[/red]",
        )

    console.print(table)
    console.print(f"\nRunning: {running}/{len(candidates)}, Total: ${total_hourly:.2f}/hr")


# ---------------------------------------------------------------------------
# estimate — cost/time scaling table
# ---------------------------------------------------------------------------


@app.command()
def estimate(
    work_hours: Annotated[
        float,
        typer.Option("--work-hours", "-w", help="Total work in RTX 4090-equivalent hours"),
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
    scope_arg: Annotated[
        str,
        typer.Option(
            "--label",
            "-l",
            help=(
                "Full canonical batch scope, e.g. prod-3f9a1b2c4d5e. "
                "By default matches only labels starting with `<scope>-`. "
                "With --allow-adjacent-scopes, matches any label that "
                "starts with the requested prefix (still requires the "
                "canonical 12-hex suffix in the scope argument)."
            ),
        ),
    ],
    adjacent_scopes: Annotated[
        bool,
        typer.Option(
            "--allow-adjacent-scopes",
            help=(
                "Match every label that starts with the requested scope "
                "without the `-` delimiter (e.g. a candidate labelled "
                "`prod-3f9a1b2c4d5eevil-...` would still match canonical "
                "scope `prod-3f9a1b2c4d5e`). DANGEROUS; only enable for "
                "intentional broad cleanup."
            ),
        ),
    ] = False,
    allowed_images: Annotated[
        str | None,
        typer.Option(
            "--allowed-images",  # canonical
            "--owned-images",  # alias
            help=(
                "Comma-separated Docker images owned by this project. "
                "Omit to opt out of the ownership check (DANGEROUS — "
                "every instance with the label prefix is destroyed). "
                "Pass an empty string to fail-closed (every instance "
                "is refused; use to test the wiring without risk)."
            ),
        ),
    ] = None,
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

    from vastai_gpu_runner.providers.destroy_adapters.vastai import (
        read_vastai_api_key,
    )
    from vastai_gpu_runner.providers.vastai import build_vastai_cleanup_policy

    console = Console()
    canonical_scope = _validate_cleanup_scope(scope_arg)
    ownership = _build_ownership_from_flag(allowed_images)
    cleanup_policy = build_vastai_cleanup_policy(
        ownership=ownership,
        credentials=read_vastai_api_key(),
    )

    # Manual cleanup defaults to one exact batch scope (`<scope>-`).
    # A broad requested-prefix match requires the dangerous opt-in.
    scope_prefix = canonical_scope + ("-" if not adjacent_scopes else "")
    candidates = cleanup_policy.list_instances()
    matches = [c for c in candidates if c.label.startswith(scope_prefix)]
    if not matches:
        console.print(f"No instances matching label scope '{canonical_scope}'.")
        return

    _print_matches(console, canonical_scope, matches)

    if dry_run:
        console.print("\n[yellow]Dry run — no instances destroyed.[/yellow]")
        return

    if not typer.confirm(f"\nDestroy {len(matches)} instance(s)?"):
        console.print("Aborted.")
        raise typer.Exit(0)

    counts = _destroy_matches(console, matches, cleanup_policy)
    console.print(
        f"\nDestroyed: {counts['destroyed']}; already gone: {counts['already_gone']}; "
        f"unresolved: {counts['unresolved']} (of {len(matches)} instance(s))."
    )


def _validate_cleanup_scope(scope_arg: str) -> str:
    """Validate --label; raise ``typer.BadParameter`` on failure."""
    from vastai_gpu_runner.state import validate_label_scope_for_cleanup

    try:
        return validate_label_scope_for_cleanup(scope_arg)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--label") from exc


def _build_ownership_from_flag(allowed_images: str | None) -> OwnershipPolicy:
    """Translate ``--allowed-images`` flag into an ``OwnershipPolicy``.

    ``None`` → opt-out (every image considered owned).
    Empty string → fail-closed (empty set, refuses every image).
    Comma-separated string → frozenset of trimmed non-empty entries.
    """
    if allowed_images is None:
        return OwnershipPolicy()
    images = frozenset(item.strip() for item in allowed_images.split(",") if item.strip())
    return OwnershipPolicy(owned_images=images)


def _print_matches(console: object, canonical_scope: str, matches: list[InstanceCandidate]) -> None:
    """Print the discovered match list before any confirmation prompt."""
    console.print(f"Found {len(matches)} instance(s) matching '{canonical_scope}':")  # type: ignore[attr-defined]
    for c in matches:
        console.print(  # type: ignore[attr-defined]
            f"  {c.instance_id}: {c.gpu_model or '?'} status={c.state or '?'} label={c.label}"
        )


def _destroy_matches(
    console: object,
    matches: list[InstanceCandidate],
    cleanup_policy: ProviderCleanupPolicy,
) -> dict[str, int]:
    """Destroy every match, printing each outcome; return aggregate counts."""
    from vastai_gpu_runner.cleanup_policy import CleanupVerdict

    destroyed = 0
    already_gone = 0
    unresolved = 0
    for c in matches:
        result = cleanup_policy.destroy(c)
        verdict = result.verdict
        if verdict == CleanupVerdict.DESTROYED:
            console.print(f"  [green]Destroyed[/green] {c.instance_id}")  # type: ignore[attr-defined]
            destroyed += 1
        elif verdict == CleanupVerdict.ALREADY_GONE:
            console.print(f"  [green]Already gone[/green] {c.instance_id}")  # type: ignore[attr-defined]
            already_gone += 1
        elif verdict is not None or result.refusal is not None:
            kind = (
                verdict.value if verdict is not None else result.refusal.value  # type: ignore[union-attr]
            )
            console.print(f"  [red]{kind}[/red] {c.instance_id}: {result.error}")  # type: ignore[attr-defined]
            unresolved += 1
        else:
            console.print(  # type: ignore[attr-defined]
                f"  [red]unknown_cleanup_outcome[/red] {c.instance_id}: "
                "cleanup returned no verdict or refusal"
            )
            unresolved += 1
    return {"destroyed": destroyed, "already_gone": already_gone, "unresolved": unresolved}


# ---------------------------------------------------------------------------
# batch — composition root (v4: state migration + canonical config)
# ---------------------------------------------------------------------------


@app.command()
def batch(
    state_path: Annotated[
        Path,
        typer.Option(help="Path to BatchState/JobBatchState JSON"),
    ],
    label: Annotated[str, typer.Option("--label", "-l")],
    image: Annotated[
        str,
        typer.Option("--image", help="Canonical Docker image owned by this project"),
    ],
    job_batch: Annotated[bool, typer.Option(help="Use JobBatchState (default BatchState)")] = False,
    max_parallel: Annotated[int, typer.Option("--max-parallel", "-p")] = 8,
    budget: Annotated[float, typer.Option("--budget")] = 0.0,
) -> None:
    """Build the v4 composition for a batch run.

    The v4 architecture assumes consumer projects subclass
    ``BatchOrchestrator`` (per the docstring of that ABC). This CLI
    command is the composition root: it validates the inputs, performs
    the state migration, builds the canonical ``VastaiProviderConfig``
    + cleanup policy, and emits a single JSON line describing the
    resolved scope + ownership + credentials so the consumer entry
    point can drive the run with the canonical objects. This keeps
    the CLI free of any consumer-specific ``iter_*`` / ``on_unit_*``
    logic while still exercising every v4 composition boundary.

    Exit code 0 on successful composition.
    """
    from vastai_gpu_runner.providers.vastai import (
        build_vastai_cleanup_policy,
    )
    from vastai_gpu_runner.state import (
        BatchState,
        JobBatchState,
    )

    state_cls = JobBatchState if job_batch else BatchState
    existing = _load_existing_state(state_path, state_cls=state_cls)
    persisted_scope, persisted_requested_prefix = _persisted_identity(existing)
    label_scope = _resolve_batch_label_scope(label, persisted_scope, persisted_requested_prefix)
    state = _build_or_update_state(existing, state_cls, state_path, label_scope, label)
    # ``state`` is typed ``object`` (state_cls is ``type``); narrow locally.
    state.save(state_path)  # type: ignore[attr-defined]

    config = _build_provider_config(image, label_scope)
    # ``config`` is typed ``object`` because the helper is import-decoupled.
    # Narrow via local aliases so the typed attribute accesses resolve.
    config_ownership: OwnershipPolicy = config.ownership  # type: ignore[attr-defined]
    config_credentials: CredentialResolution = config.credentials  # type: ignore[attr-defined]
    cleanup_policy = build_vastai_cleanup_policy(
        ownership=config_ownership,
        credentials=config_credentials,
    )

    summary = _build_batch_summary(
        label_scope=label_scope,
        requested_prefix=label,
        image=image,
        max_parallel=max_parallel,
        budget=budget,
        credentials=config_credentials,
        ownership=config_ownership,
        cleanup_provider=cleanup_policy.provider.name,
    )
    typer.echo(json.dumps(summary, indent=2))


def _load_existing_state(state_path: Path, *, state_cls: type) -> object:
    """Load existing state or raise ``typer.BadParameter`` on migration failure."""
    from vastai_gpu_runner.state import StateMigrationError, load_batch_state

    try:
        return load_batch_state(state_path, state_cls=state_cls)
    except StateMigrationError as exc:
        raise typer.BadParameter(str(exc), param_hint="--label") from exc


def _persisted_identity(existing: object) -> tuple[str | None, str | None]:
    """Extract ``(label_scope, requested_label_prefix)`` from a loaded state."""
    from vastai_gpu_runner.state import BatchState, JobBatchState

    if existing is None:
        return None, None
    if not isinstance(existing, (BatchState, JobBatchState)):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise typer.BadParameter(
            "persisted state has unexpected type; aborting",
            param_hint="--label",
        )
    return existing.label_scope, existing.requested_label_prefix


def _resolve_batch_label_scope(
    requested_prefix: str,
    persisted_scope: str | None,
    persisted_requested_prefix: str | None,
) -> str:
    """Resolve a fresh or persisted canonical scope; raise ``typer.BadParameter`` on drift."""
    from vastai_gpu_runner.state import (
        StateMigrationError,
        resolve_label_scope,
    )

    try:
        return resolve_label_scope(
            requested_prefix,
            persisted_scope,
            persisted_requested_prefix,
        )
    except (StateMigrationError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--label") from exc


def _build_or_update_state(
    existing: object,
    state_cls: type,
    state_path: Path,
    label_scope: str,
    requested_prefix: str,
) -> object:
    """Build a fresh state or overlay scope on an existing one.

    Returns ``object`` (rather than ``BatchState | JobBatchState``)
    because ``state_cls`` is ``type`` — the runtime branch decides
    which dataclass to construct, and pyright can't narrow ``type``
    through constructor dispatch.
    """
    """Build a fresh state or overlay scope on an existing one."""
    from vastai_gpu_runner.state import (
        CURRENT_SCHEMA_VERSION,
        BatchState,
        JobBatchState,
    )

    if existing is None:
        return state_cls(
            batch_id=state_path.stem,
            label_scope=label_scope,
            requested_label_prefix=requested_prefix,
            schema_version=CURRENT_SCHEMA_VERSION,
        )
    if not isinstance(existing, (BatchState, JobBatchState)):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise typer.BadParameter(
            "persisted state has unexpected type; aborting",
            param_hint="--label",
        )
    if isinstance(existing, JobBatchState):
        return JobBatchState(
            batch_id=existing.batch_id,
            jobs=existing.jobs,
            created_at=existing.created_at,
            updated_at=existing.updated_at,
            metadata=existing.metadata,
            label_scope=label_scope,
            requested_label_prefix=requested_prefix,
            schema_version=CURRENT_SCHEMA_VERSION,
        )
    return BatchState(
        batch_id=existing.batch_id,
        num_gpus=existing.num_gpus,
        shards=existing.shards,
        created_at=existing.created_at,
        updated_at=existing.updated_at,
        metadata=existing.metadata,
        label_scope=label_scope,
        requested_label_prefix=requested_prefix,
        schema_version=CURRENT_SCHEMA_VERSION,
    )


def _build_provider_config(image: str, label_scope: str) -> object:
    """Compose the canonical ``VastaiProviderConfig`` for the batch run.

    Returns ``object`` to keep the helper decoupled from the dataclass
    import path; callers use the public attribute surface.
    """
    from dataclasses import replace

    from vastai_gpu_runner.providers.vastai import VastaiProviderConfig

    base = VastaiProviderConfig.from_env()
    return replace(
        base,
        docker_image=image,
        ownership=OwnershipPolicy(owned_images=frozenset({image})),
        label_prefix=label_scope,
    )


def _build_batch_summary(
    *,
    label_scope: str,
    requested_prefix: str,
    image: str,
    max_parallel: int,
    budget: float,
    credentials: CredentialResolution,
    ownership: OwnershipPolicy,
    cleanup_provider: str,
) -> dict[str, object]:
    """Render the JSON summary the ``batch`` command prints."""
    from vastai_gpu_runner.state import CURRENT_SCHEMA_VERSION

    return {
        "label_scope": label_scope,
        "requested_label_prefix": requested_prefix,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "docker_image": image,
        "max_parallel": max_parallel,
        "budget_usd": budget,
        "credential_state": credentials.state.name,
        "owned_images_count": len(ownership.owned_images or ()),
        "cleanup_policy_provider": cleanup_provider,
    }


if __name__ == "__main__":
    app()
