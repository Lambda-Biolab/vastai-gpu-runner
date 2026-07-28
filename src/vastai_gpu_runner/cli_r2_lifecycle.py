"""Operator CLI for ``vastai_gpu_runner.storage.r2_lifecycle``.

Three sub-commands:
    - ``show``     — print the current lifecycle configuration for a
      bucket/prefix and the managed-rule status. Non-mutating.
    - ``apply``    — plan, confirm, and apply a one-rule expiration
      policy. Requires ``--expire-after-days``. Supports
      ``--dry-run`` and ``--yes``.
    - ``remove``   — plan, confirm, and remove the managed rule.
      Supports ``--dry-run`` and ``--yes``.

Safety invariants:

- Credentials are explicitly supplied via ``--credentials-file``. No
  environment-variable auto-activation.
- ``--expire-after-days`` has no default; retention must be explicit.
- Bucket-wide (root) prefixes are rejected.
- ``apply`` and ``remove`` refuse to mutate without an explicit
  confirmation, except when ``--yes`` is set on a non-interactive stdin.
- All credential-bearing output goes through ``_safe_console`` which
  strips secret material.

The CLI is a thin Typer composition layer over the ``R2LifecycleManager``
domain object. Domain logic stays in ``r2_lifecycle.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from vastai_gpu_runner.storage.r2_lifecycle import (
    R2AdminCredentials,
    R2ExpirationPolicy,
    R2LifecycleAccessDeniedError,
    R2LifecycleCollisionError,
    R2LifecycleCredentialsError,
    R2LifecycleError,
    R2LifecycleManager,
    R2LifecycleRuleLimitError,
    R2LifecycleStalePlanError,
    R2LifecycleValidationError,
    R2LifecycleVerificationError,
)

logger = logging.getLogger(__name__)


app = typer.Typer(
    name="r2-lifecycle",
    help=(
        "Administer Cloudflare R2 bucket lifecycle rules (one managed "
        "rule per prefix). Use 'show' before 'apply'. Always pass "
        "--credentials-file; worker credentials are never reused."
    ),
    no_args_is_help=True,
)


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI output."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )


def _safe_console():  # type: ignore[no-untyped-def]
    """Return a Rich console whose writers have no secret-bearing inputs."""
    from rich.console import Console

    return Console()


def _print_plan(plan: Any, *, dry_run: bool) -> None:
    """Print a human-readable plan summary."""
    console = _safe_console()
    console.print(f"  bucket        : {plan.bucket}")
    console.print(f"  canonical prefix: {plan.canonical_prefix}")
    console.print(f"  managed rule id: {plan.managed_rule_id}")
    if plan.expire_after_days is not None:
        console.print(f"  expire_after_days: {plan.expire_after_days}")
    console.print(f"  operation     : {plan.operation}")
    console.print(f"  before rules  : {len(plan.before_rules)}")
    console.print(f"  after rules   : {len(plan.after_rules)}")
    console.print(f"  no-op         : {plan.no_op}")
    if plan.warnings:
        console.print("\n  warnings:")
        for w in plan.warnings:
            console.print(f"    - {w}")
    if dry_run:
        console.print("\n  [yellow]dry-run: no PUT will be issued.[/yellow]")
    if plan.expire_after_days is not None:
        console.print(
            "\n  [yellow]note:[/yellow] existing objects older than "
            f"{plan.expire_after_days} days may become eligible for "
            "deletion after activation.",
        )


def _confirm(plan: Any, *, yes: bool) -> bool:
    """Prompt the operator; auto-approve iff ``--yes`` and stdin is non-tty."""
    if plan.no_op:
        return True
    if yes:
        return True
    if not sys.stdin.isatty():
        console = _safe_console()
        console.print(
            "\n  [red]refusing to mutate without confirmation.[/red] "
            "Re-run with --yes when stdin is not interactive.",
        )
        return False
    console = _safe_console()
    reply = typer.confirm(
        f"\nApply {plan.operation} for {plan.bucket}/{plan.canonical_prefix}?",
        default=False,
    )
    return bool(reply)


def _translate_error(exc: R2LifecycleError) -> tuple[str, int]:
    """Map a typed lifecycle error to a (message, exit_code) pair.

    Args:
        exc: The typed exception.

    Returns:
        (human-readable message, non-zero exit code).
    """
    if isinstance(exc, R2LifecycleValidationError):
        return f"validation: {exc}", 2
    if isinstance(exc, R2LifecycleCredentialsError):
        return f"credentials: {exc}", 3
    if isinstance(exc, R2LifecycleAccessDeniedError):
        return f"access denied: {exc}", 4
    if isinstance(exc, R2LifecycleCollisionError):
        return f"collision: {exc}", 5
    if isinstance(exc, R2LifecycleStalePlanError):
        return f"stale plan: {exc}", 6
    if isinstance(exc, R2LifecycleRuleLimitError):
        return f"rule limit: {exc}", 7
    if isinstance(exc, R2LifecycleVerificationError):
        return f"verification failed: {exc}", 8
    return f"lifecycle error: {exc}", 1


def _build_manager(credentials_file: Path) -> R2LifecycleManager:
    """Load admin credentials and instantiate the manager."""
    creds = R2AdminCredentials.from_file(credentials_file)
    return R2LifecycleManager(creds)


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@app.command()
def show(
    bucket: Annotated[str, typer.Option("--bucket", help="R2 bucket name")],
    prefix: Annotated[str, typer.Option("--prefix", help="Object key prefix")],
    credentials_file: Annotated[
        Path,
        typer.Option(
            "--credentials-file",
            dir_okay=False,
            help="Shell-export file containing R2_ADMIN_* variables",
        ),
    ],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Show the managed lifecycle rule for the given bucket/prefix.

    Non-mutating, non-interactive. Prints the current managed rule if
    present, plus the source fingerprint, then exits 0. Exits non-zero
    only on access / validation errors.
    """
    _setup_logging(verbose)
    console = _safe_console()
    try:
        manager = _build_manager(credentials_file)
        from vastai_gpu_runner.storage.r2_lifecycle import canonicalise_prefix, managed_rule_id

        canonical_prefix = canonicalise_prefix(prefix)
        rule_id = managed_rule_id(bucket, canonical_prefix)
        managed = manager.inspect_managed_rule(bucket, prefix)
        read = manager.read_lifecycle(bucket)
        console.print(f"bucket        : {bucket}")
        console.print(f"canonical prefix: {canonical_prefix}")
        console.print(f"managed rule id: {rule_id}")
        console.print(f"source fingerprint: {read.fingerprint}")
        console.print(f"rules on bucket : {len(read.rules)}")
        if managed is None:
            console.print("status: [yellow]not configured[/yellow]")
        else:
            console.print("status: [green]configured[/green]")
            console.print(f"  enabled: {managed.get('Status')}")
            expiration = managed.get("Expiration", {})
            console.print(f"  expiration days: {expiration.get('Days', '?')}")
    except R2LifecycleError as exc:
        msg, code = _translate_error(exc)
        console.print(f"[red]{msg}[/red]")
        raise typer.Exit(code) from None


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


@app.command()
def apply(
    bucket: Annotated[str, typer.Option("--bucket", help="R2 bucket name")],
    prefix: Annotated[str, typer.Option("--prefix", help="Object key prefix")],
    credentials_file: Annotated[
        Path,
        typer.Option(
            "--credentials-file",
            dir_okay=False,
            help="Shell-export file containing R2_ADMIN_* variables",
        ),
    ],
    expire_after_days: Annotated[
        int,
        typer.Option(
            "--expire-after-days",
            min=1,
            help="Retention in whole days (no default; required)",
        ),
    ],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Plan only; do not PUT")] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            help="Skip interactive confirmation (required when stdin is non-interactive)",
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Apply the managed expiration rule for the given bucket/prefix.

    Plans the change, prompts for confirmation, then PUTs the new
    lifecycle configuration and verifies the post-write state. The
    managed rule preserves all unrelated rules on the bucket.
    """
    _setup_logging(verbose)
    console = _safe_console()
    try:
        manager = _build_manager(credentials_file)
        policy = R2ExpirationPolicy(
            bucket=bucket,
            prefix=prefix,
            expire_after_days=expire_after_days,
        )
        plan = manager.plan_apply(policy)
        _print_plan(plan, dry_run=dry_run)
        if dry_run:
            return
        if not _confirm(plan, yes=yes):
            raise typer.Exit(9)
        result = manager.apply(plan)
        if result.no_op:
            console.print("\n[green]no-op[/green] — already in desired state.")
        else:
            console.print(
                f"\n[green]applied[/green] — {result.rules_count} rule(s) "
                "on bucket after read-after-write verification.",
            )
    except R2LifecycleError as exc:
        msg, code = _translate_error(exc)
        console.print(f"[red]{msg}[/red]")
        raise typer.Exit(code) from None


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


@app.command()
def remove(
    bucket: Annotated[str, typer.Option("--bucket", help="R2 bucket name")],
    prefix: Annotated[str, typer.Option("--prefix", help="Object key prefix")],
    credentials_file: Annotated[
        Path,
        typer.Option(
            "--credentials-file",
            dir_okay=False,
            help="Shell-export file containing R2_ADMIN_* variables",
        ),
    ],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Plan only; do not PUT")] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            help="Skip interactive confirmation (required when stdin is non-interactive)",
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Remove the managed expiration rule for the given bucket/prefix.

    Plans the change, prompts for confirmation, then PUTs the lifecycle
    configuration without the managed rule and verifies the post-write
    state. All unrelated rules are preserved.
    """
    _setup_logging(verbose)
    console = _safe_console()
    try:
        manager = _build_manager(credentials_file)
        plan = manager.plan_remove(bucket, prefix)
        _print_plan(plan, dry_run=dry_run)
        if dry_run:
            return
        if not _confirm(plan, yes=yes):
            raise typer.Exit(9)
        result = manager.remove(plan)
        if result.no_op:
            console.print("\n[green]no-op[/green] — managed rule already absent.")
        else:
            console.print(
                f"\n[green]removed[/green] — {result.rules_count} rule(s) "
                "remaining on bucket after read-after-write verification.",
            )
            console.print(
                "\n  [yellow]note:[/yellow] removing the rule does not "
                "restore objects that were already expired by it.",
            )
    except R2LifecycleError as exc:
        msg, code = _translate_error(exc)
        console.print(f"[red]{msg}[/red]")
        raise typer.Exit(code) from None


if __name__ == "__main__":
    app()
