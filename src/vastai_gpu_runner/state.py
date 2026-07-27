"""Persistent batch state classes for cloud orchestration.

These dataclasses are serialized to JSON for crash recovery. The orchestrator
writes state after each lifecycle phase; a new process can resume by reading
the state file — skipping downloaded shards, re-polling active ones, and
re-deploying failed ones.

Two models:
- ``ShardState`` / ``BatchState``: N items split across M shards (1 shard = 1 GPU).
  Used for batch prediction workloads.
- ``JobState`` / ``JobBatchState``: 1 job = 1 instance (no sharding).
  Used for long-running single-GPU workloads like MD simulation.

The v4 architecture adds:

- ``BatchState.label_scope`` and ``JobBatchState.label_scope`` — persisted
  canonical batch identity (``<prefix>-<12 lowercase hex>``) shared by the
  Vast.ai runner, instance labels, and orchestrator. Resume reloads this
  scope before composition; a drifted scope is rejected rather than silently
  re-scoped.
- ``BatchState.requested_label_prefix`` / ``JobBatchState.requested_label_prefix`` —
  the user-supplied prefix the scope was derived from. Stored so the
  resolver can detect prefix drift on resume.
- ``BatchState.schema_version`` / ``JobBatchState.schema_version`` — bumped
  from ``0`` (pre-v4) to ``CURRENT_SCHEMA_VERSION`` by ``load_batch_state``.
- ``StateMigrationError`` — raised on any unrecoverable persisted-state
  failure (read, parse, schema mismatch, migration, construction).
- ``load_batch_state`` — v4 loader; never silently returns ``None`` on a
  parse failure. Pass through ``load_or_none`` for legacy callers.
- ``resolve_label_scope`` — reuses a persisted scope or creates one for a
  fresh batch; rejects drift.
- ``validate_label_scope_for_cleanup`` — accepts a full canonical scope
  before manual cleanup.

Backward compatibility: ``BatchState.label`` and the legacy ``load_or_none``
behaviour remain in place; new code should use ``load_batch_state``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

logger = logging.getLogger(__name__)

MAX_SHARD_RETRIES = 2  # Max re-deploys per shard on preemption

# Default terminal statuses — consumers should override per their state machine.
_DEFAULT_SHARD_TERMINAL_STATUSES: frozenset[str] = frozenset({"downloaded", "destroyed", "failed"})
_DEFAULT_JOB_TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "downloaded", "failed"})

# ---------------------------------------------------------------------------
# v4 schema + label-scope helpers
# ---------------------------------------------------------------------------


CURRENT_SCHEMA_VERSION = 1
_VALID_SCHEMA_VERSIONS: frozenset[int] = frozenset({0, CURRENT_SCHEMA_VERSION})
_LABEL_SUFFIX_LEN = 12
TERMINAL_UNIT_STATUSES: frozenset[str] = frozenset(
    {"completed", "downloaded", "failed", "archived", "destroyed"}
)


class StateMigrationError(RuntimeError):
    """Raised when persisted batch state cannot be migrated to v4.

    The v3 ``load_or_none`` silently returned ``None`` on parse errors,
    which caused the CLI to silently re-scope existing batches. v4 funnels
    every persisted-state failure (read, parse, schema, migration,
    construction) through this exception so the CLI can surface a fatal
    error before composition rather than silently continuing with a
    fresh identity.
    """


def _validate_label_scope_shape(scope: str, requested: str) -> str:
    """Reject malformed ``f"{requested}-<12 hex>"`` scopes."""
    # Defensive runtime guards: callers may pass untyped dict values from JSON.
    if not isinstance(scope, str) or not isinstance(requested, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise StateMigrationError("label scope and request must be strings")
    expected_prefix = f"{requested}-"
    if not scope.startswith(expected_prefix):
        raise StateMigrationError(
            f"persisted label scope {scope!r} does not start with {expected_prefix!r}"
        )
    suffix = scope[len(expected_prefix) :]
    if len(suffix) != _LABEL_SUFFIX_LEN or not all(c in "0123456789abcdef" for c in suffix):
        raise StateMigrationError(f"persisted label scope {scope!r} has malformed suffix")
    return scope


def validate_label_prefix(label_prefix: str) -> str:
    """Return a safe label prefix or raise before provider enumeration.

    Account-wide cleanup is not expressible through an empty prefix.
    Empty, whitespace-only, or padded prefixes are rejected before any
    provider call. The CLI calls the same validator before composition,
    so invalid input fails before any enumeration or confirmation prompt.
    """
    if (
        not isinstance(label_prefix, str)  # pyright: ignore[reportUnnecessaryIsInstance]
        or not label_prefix
        or label_prefix != label_prefix.strip()
    ):
        raise ValueError("label_prefix must be non-empty, non-blank, and pre-stripped")
    return label_prefix


def _unit_status(unit: object) -> str | None:
    """Extract ``status`` from a unit dict; defensive for malformed input."""
    if not isinstance(unit, dict):
        return None
    status = unit.get("status")
    return status if isinstance(status, str) else None


def _all_units_terminal(units: list[object], terminal_statuses: frozenset[str]) -> bool:
    """True iff every unit's status is in the terminal set."""
    return all(_unit_status(u) in terminal_statuses for u in units)


def _pre_v4_collection_shapes_ok(data: dict) -> str | None:
    """Return an error string if ``shards``/``jobs`` are not lists/nulls, else None."""
    if not isinstance(data.get("shards"), (list, type(None))):
        return "shards must be a list or null in schema_version 0"
    if not isinstance(data.get("jobs"), (list, type(None))):
        return "jobs must be a list or null in schema_version 0"
    return None


def _recover_label_scope(effective_scope: str) -> str:
    """Strip the trailing 12-hex suffix to recover the requested prefix."""
    parts = effective_scope.rsplit("-", 1)
    if (
        len(parts) == 2
        and len(parts[1]) == _LABEL_SUFFIX_LEN
        and all(c in "0123456789abcdef" for c in parts[1])
    ):
        return parts[0]
    return effective_scope


def _legacy_identity_pair(
    *,
    effective_scope: str,
    legacy_label: str,
    label_scope: str,
    has_units: bool,
    all_terminal: bool,
) -> tuple[str, str] | str:
    """Compute ``(requested_label_prefix, label_scope)`` for a pre-v4 state.

    Returns a ``str`` error message on failure (and an empty
    identity); returns the identity tuple on success. Extracted
    from ``_migrate_pre_v4`` to keep the migration's cognitive
    complexity within the project gate.
    """
    if not effective_scope:
        if has_units and not all_terminal:
            return "nonterminal state lacks a recoverable label scope"
        return "", ""
    if label_scope and legacy_label and label_scope != legacy_label:
        return "pre-v4 state has both legacy label and label_scope; they disagree"
    return _recover_label_scope(effective_scope), effective_scope


def _migrate_pre_v4(data: dict, *, state_cls: type) -> tuple[dict, str | None]:
    """Migrate a pre-v4 (schema_version 0) state file to v4 in place.

    Recovers ``requested_label_prefix`` by removing the trailing
    12-lowercase-hex suffix from a canonical legacy ``label`` (e.g.
    ``prod-3f9a1b2c4d5e`` -> ``prod``). A nonterminal state without
    a recoverable scope, or with conflicting ``label``/``label_scope``,
    returns the data unchanged plus an error string so the caller
    can raise :class:`StateMigrationError`.

    Returns a tuple ``(migrated_data, error_message)``. Exactly one of
    ``error_message is None`` and ``migrated_data has been mutated`` holds
    in normal cases — but the caller is responsible for checking both.
    """
    legacy_label = data.get("label", "") or ""
    label_scope = data.get("label_scope", "") or ""
    if (legacy_label and not isinstance(legacy_label, str)) or (
        label_scope and not isinstance(label_scope, str)
    ):
        return data, "legacy label and label_scope must be strings when present"
    # Validate collection shapes BEFORE iterating: a non-list
    # ``shards``/``jobs`` (e.g. a dict) would coerce to a list of
    # keys under ``list(data.get("shards") or [])`` and silently
    # produce a truthy ``raw_units`` even though every entry is a
    # string, not a unit dict. The subsequent ``_unit_status``
    # pass would mark every entry ``None`` and the late type guard
    # at the bottom would discard the enumeration; the data flow
    # is sound but misleading. The shape guards now run first so
    # the failure mode is immediate and obvious.
    shape_error = _pre_v4_collection_shapes_ok(data)
    if shape_error:
        return data, shape_error
    raw_units = list(data.get("shards") or []) + list(data.get("jobs") or [])
    has_units = bool(raw_units)
    terminal_statuses = getattr(state_cls, "TERMINAL_STATUSES", TERMINAL_UNIT_STATUSES)
    all_terminal = has_units and _all_units_terminal(raw_units, terminal_statuses)
    effective_scope = label_scope or legacy_label
    pair_or_error = _legacy_identity_pair(
        effective_scope=effective_scope,
        legacy_label=legacy_label,
        label_scope=label_scope,
        has_units=has_units,
        all_terminal=all_terminal,
    )
    if isinstance(pair_or_error, str):
        return data, pair_or_error
    requested_label_prefix, migrated_label_scope = pair_or_error
    migrated = dict(data)
    migrated["label_scope"] = migrated_label_scope
    migrated["requested_label_prefix"] = requested_label_prefix
    migrated["schema_version"] = CURRENT_SCHEMA_VERSION
    migrated.pop("label", None)
    return migrated, None


def _read_and_parse_state(state_path: Path) -> dict:
    """Read + JSON-parse a state file. Raises ``StateMigrationError`` on failure."""
    try:
        raw = state_path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise StateMigrationError(f"could not read state: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StateMigrationError(f"state JSON is invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise StateMigrationError("state JSON root must be an object")
    return data


def _validate_persisted_schema(data: dict) -> int:
    """Validate the ``schema_version`` field and unit collection shapes."""
    schema_version = data.get("schema_version", 0)
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise StateMigrationError(f"schema_version must be an integer, got {schema_version!r}")
    if schema_version not in _VALID_SCHEMA_VERSIONS:
        raise StateMigrationError(
            f"unsupported schema_version {schema_version}; "
            f"expected one of {sorted(_VALID_SCHEMA_VERSIONS)}"
        )
    if not isinstance(data.get("shards"), (list, type(None))):
        raise StateMigrationError("persisted shards must be a list or null")
    if not isinstance(data.get("jobs"), (list, type(None))):
        raise StateMigrationError("persisted jobs must be a list or null")
    return schema_version


def _migrate_schema_zero(data: dict, *, state_cls: type) -> dict:
    """Run the pre-v4 migration or raise ``StateMigrationError``."""
    try:
        data, migration_error = _migrate_pre_v4(data, state_cls=state_cls)
    except (TypeError, ValueError, AttributeError) as exc:
        raise StateMigrationError(f"schema-0 migration failed: {exc}") from exc
    if migration_error:
        raise StateMigrationError(migration_error)
    return data


def _archive_scope_less_terminal_state(
    state_path: Path, raw_units: list[object], state_cls: type
) -> bool:
    """If ``raw_units`` is non-empty and all-terminal, archive and return True.

    Raises ``StateMigrationError`` on the nonterminal path; returns
    False when ``raw_units`` is empty.
    """
    if not raw_units:
        return False
    terminal_statuses = getattr(state_cls, "TERMINAL_STATUSES", TERMINAL_UNIT_STATUSES)
    all_terminal = _all_units_terminal(raw_units, terminal_statuses)
    if all_terminal:
        return _rename_to_archive(state_path)
    raise StateMigrationError("nonterminal state lacks a recoverable label scope")


def _rename_to_archive(state_path: Path) -> bool:
    """Rename ``state_path`` to a millisecond-stamped archive; verify rename."""
    archive_path = state_path.with_name(
        f"{state_path.stem}_archived_{int(time.time() * 1000)}{state_path.suffix}"
    )
    try:
        state_path.rename(archive_path)
    except (OSError, AttributeError, TypeError, ValueError) as exc:
        raise StateMigrationError(
            f"could not archive terminal scope-less legacy state: {exc}"
        ) from exc
    if state_path.exists():
        raise StateMigrationError(
            "failed to archive terminal scope-less legacy state; original file is still present"
        )
    return True


def _hydrate_nested_units(state: object, data: dict) -> object:
    """Replace ``shards``/``jobs`` raw dicts with proper dataclass instances."""
    # Use ``__dict__`` to bypass the strict attribute-access typing on the
    # generic ``state: object`` parameter: this function operates on a
    # runtime instance of either BatchState or JobBatchState.
    state_dict = state.__dict__
    if "shards" in state_dict and isinstance(data.get("shards"), list):
        try:
            state_dict["shards"] = [ShardState(**s) for s in data["shards"]]
        except (TypeError, ValueError) as exc:
            raise StateMigrationError(
                f"could not deserialize {type(state).__name__}.shards: {exc}"
            ) from exc
    if "jobs" in state_dict and isinstance(data.get("jobs"), list):
        try:
            state_dict["jobs"] = [JobState(**j) for j in data["jobs"]]
        except (TypeError, ValueError) as exc:
            raise StateMigrationError(
                f"could not deserialize {type(state).__name__}.jobs: {exc}"
            ) from exc
    return state


def load_batch_state(state_path: Path, *, state_cls: type) -> object:
    """Load a v4 BatchState/JobBatchState or raise ``StateMigrationError``.

    The legacy ``load_or_none`` helper always returns ``None`` on any
    parse error, which silently re-scopes existing batches. This loader
    raises :class:`StateMigrationError` instead whenever the persisted
    state cannot be migrated into a v4 identity. JSON decoding, schema
    mismatches, the pre-v4 migration, and dataclass construction all
    funnel through this boundary.

    Returns ``None`` only when the path does not exist or when a
    verified-terminal scope-less legacy state is archived on disk.
    """
    if not state_path.exists():
        return None
    data = _read_and_parse_state(state_path)
    schema_version = _validate_persisted_schema(data)
    if schema_version == 0:
        data = _migrate_schema_zero(data, state_cls=state_cls)
    raw_units = list(data.get("shards") or []) + list(data.get("jobs") or [])
    if not data.get("label_scope") and _archive_scope_less_terminal_state(
        state_path, raw_units, state_cls
    ):
        return None
    try:
        state = state_cls(**data)
    except (TypeError, ValueError) as exc:
        raise StateMigrationError(f"could not deserialize {state_cls.__name__}: {exc}") from exc
    return _hydrate_nested_units(state, data)


def resolve_label_scope(
    requested_prefix: str,
    persisted_scope: str | None,
    persisted_requested_prefix: str | None = None,
) -> str:
    """Reuse a persisted scope or create one for a new batch.

    Valid input shapes:

    - Genuinely new identity: ``persisted_scope is None`` and
      ``persisted_requested_prefix is None``; a new canonical scope is
      returned.
    - Valid modern identity: ``persisted_scope`` is a canonical
      ``f"{requested}-<12 hex>"`` scope; either no requested prefix
      was stored or the stored prefix matches the current request.
    - Legacy fallback: when ``persisted_scope`` is the legacy
      ``label`` value, a stored ``requested_label_prefix`` may be
      ``None`` and the resolver reuses the legacy value as the scope.

    Every other partial or inconsistent state pair raises
    :class:`StateMigrationError`; the resolver never silently re-scopes
    an existing batch.
    """
    from uuid import uuid4

    if persisted_scope is None and persisted_requested_prefix is None:
        return f"{validate_label_prefix(requested_prefix)}-{uuid4().hex[:_LABEL_SUFFIX_LEN]}"
    if persisted_scope is None or persisted_requested_prefix is None:
        raise StateMigrationError(
            "persisted batch label identity is partial; either both "
            "label_scope and requested_label_prefix are set, or neither is"
        )
    if not isinstance(persisted_scope, str) or not isinstance(persisted_requested_prefix, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise StateMigrationError("persisted label identity fields must be strings")
    requested = validate_label_prefix(requested_prefix)
    stored_request = validate_label_prefix(persisted_requested_prefix)
    if stored_request != requested:
        raise StateMigrationError("persisted batch prefix does not match the requested label")
    return _validate_label_scope_shape(persisted_scope, stored_request)


def validate_label_scope_for_cleanup(scope_arg: str) -> str:
    """Accept a full canonical scope or raise before manual cleanup."""
    validated = validate_label_prefix(scope_arg)
    suffix = validated.rsplit("-", 1)[-1]
    if len(suffix) != _LABEL_SUFFIX_LEN or any(c not in "0123456789abcdef" for c in suffix):
        raise ValueError(
            "cleanup --label requires a full canonical scope "
            "ending in 12 lowercase hex characters (e.g. prod-3f9a1b2c4d5e); "
            "use --allow-adjacent-scopes to opt into broader matching"
        )
    return validated


# ---------------------------------------------------------------------------
# Shard-based batch state (N items → M shards → M GPUs)
# ---------------------------------------------------------------------------


@dataclass
class ShardState:
    """Serialisable state for one cloud shard.

    Status flow: ``pending`` -> ``deployed`` -> ``running`` ->
    ``downloaded`` -> ``destroyed`` | ``failed``
    """

    shard_id: int
    instance_id: str = ""
    provider: str = ""
    ssh_host: str = ""
    ssh_port: int = 0
    cost_per_hour: float = 0.0
    status: str = "pending"
    items_expected: int = 0
    items_completed: int = 0
    item_ids: list[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    failure_reason: str = ""
    retry_count: int = 0


@dataclass
class BatchState:
    """Persistent batch state written to disk after each lifecycle phase.

    Enables resume-on-crash: a new orchestrator reads this file, skips
    ``downloaded`` shards, polls ``deployed``/``running`` shards, and
    re-deploys ``failed`` shards.
    """

    # Subclasses may override to customise "is this shard done?" for archive hygiene.
    # Default matches the ShardState status flow documented above.
    TERMINAL_STATUSES: ClassVar[frozenset[str]] = _DEFAULT_SHARD_TERMINAL_STATUSES

    batch_id: str = ""
    # v4 fields. ``label`` is the legacy pre-v4 field, retained for
    # migration only; ``label_scope`` / ``requested_label_prefix`` /
    # ``schema_version`` are the canonical v4 identity. ``load_or_none``
    # never sets ``label_scope`` from ``label`` — only ``load_batch_state``
    # does, via ``_migrate_pre_v4``.
    label: str = ""
    label_scope: str = ""
    requested_label_prefix: str = ""
    schema_version: int = CURRENT_SCHEMA_VERSION
    num_gpus: int = 0
    shards: list[ShardState] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Atomically write state to disk (write tmp + rename)."""
        self.updated_at = time.time()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> BatchState:
        """Load state from disk."""
        data = json.loads(path.read_text())
        shards = [ShardState(**s) for s in data.pop("shards", [])]
        return cls(**data, shards=shards)

    @classmethod
    def archive_if_all_terminal(cls, path: Path) -> None:
        """Rename ``path`` with a timestamp suffix if every shard is terminal.

        Prevents a finished batch from being re-loaded and re-polled on the
        next orchestrator run. Corrupt files are left alone — the resume
        loader (``load_or_none``) handles them.
        """
        if not path.exists():
            return
        try:
            old = cls.load(path)
        except (json.JSONDecodeError, KeyError, TypeError):
            return
        if not old.shards:
            return
        if not all(s.status in cls.TERMINAL_STATUSES for s in old.shards):
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        archive = path.with_name(f"{path.stem}_{ts}{path.suffix}")
        path.rename(archive)
        logger.info(
            "Archived terminal batch %s (%d shards) → %s",
            old.batch_id,
            len(old.shards),
            archive.name,
        )

    @classmethod
    def load_or_none(cls, path: Path) -> BatchState | None:
        """Load ``path`` if present and parseable, else return None.

        Corrupt state triggers a warning and a fresh start — never raises.
        """
        if not path.exists():
            return None
        try:
            return cls.load(path)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Corrupt %s — starting fresh: %s", path.name, exc)
            return None

    @property
    def active_shards(self) -> list[ShardState]:
        """Shards that are deployed or running (need polling)."""
        return [s for s in self.shards if s.status in ("deployed", "running")]

    @property
    def failed_shards(self) -> list[ShardState]:
        """Shards that failed and can be re-deployed."""
        return [s for s in self.shards if s.status == "failed"]

    @property
    def downloaded_shards(self) -> list[ShardState]:
        """Shards whose results have been downloaded."""
        return [s for s in self.shards if s.status in ("downloaded", "destroyed")]

    @property
    def pending_shards(self) -> list[ShardState]:
        """Shards not yet deployed."""
        return [s for s in self.shards if s.status == "pending"]


# ---------------------------------------------------------------------------
# Job-based batch state (1 job = 1 instance)
# ---------------------------------------------------------------------------


@dataclass
class JobState:
    """State for one cloud job (1 job = 1 GPU instance).

    Status flow: ``pending`` -> ``deploying`` -> ``running`` ->
    ``completed`` -> ``downloaded`` | ``failed``
    """

    job_name: str
    status: str = "pending"
    instance_id: str = ""
    ssh_host: str = ""
    ssh_port: int = 0
    machine_id: str = ""
    error: str = ""
    submit_time: str = ""
    complete_time: str = ""
    cost_per_hour: float = 0.0
    retry_count: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        """Estimate cost based on elapsed time."""
        if not self.submit_time:
            return 0.0
        start = datetime.fromisoformat(self.submit_time)
        end = (
            datetime.fromisoformat(self.complete_time)
            if self.complete_time
            else datetime.now(tz=UTC)
        )
        hours = (end - start).total_seconds() / 3600
        return hours * self.cost_per_hour


@dataclass
class JobBatchState:
    """Persistent batch state for job-based workloads."""

    # Subclasses may override to customise "is this job done?" for archive hygiene.
    # Default matches the JobState status flow documented above.
    TERMINAL_STATUSES: ClassVar[frozenset[str]] = _DEFAULT_JOB_TERMINAL_STATUSES

    batch_id: str = ""
    # v4 identity fields. See ``BatchState`` for migration semantics.
    label_scope: str = ""
    requested_label_prefix: str = ""
    schema_version: int = CURRENT_SCHEMA_VERSION
    jobs: list[JobState] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save state atomically (tmp + rename)."""
        self.updated_at = datetime.now(tz=UTC).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> JobBatchState:
        """Load state from JSON."""
        data = json.loads(path.read_text())
        jobs = [JobState(**j) for j in data.pop("jobs", [])]
        state = cls(**data)
        state.jobs = jobs
        return state

    @classmethod
    def archive_if_all_terminal(cls, path: Path) -> None:
        """Rename ``path`` with a timestamp suffix if every job is terminal.

        Prevents a finished batch from being re-loaded and re-polled on the
        next orchestrator run. Corrupt files are left alone — the resume
        loader (``load_or_none``) handles them.
        """
        if not path.exists():
            return
        try:
            old = cls.load(path)
        except (json.JSONDecodeError, KeyError, TypeError):
            return
        if not old.jobs:
            return
        if not all(j.status in cls.TERMINAL_STATUSES for j in old.jobs):
            return
        ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        archive = path.with_name(f"{path.stem}_{ts}{path.suffix}")
        path.rename(archive)
        logger.info(
            "Archived terminal batch %s (%d jobs) → %s",
            old.batch_id,
            len(old.jobs),
            archive.name,
        )

    @classmethod
    def load_or_none(cls, path: Path) -> JobBatchState | None:
        """Load ``path`` if present and parseable, else return None.

        Corrupt state triggers a warning and a fresh start — never raises.
        """
        if not path.exists():
            return None
        try:
            return cls.load(path)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Corrupt %s — starting fresh: %s", path.name, exc)
            return None

    @property
    def pending_jobs(self) -> list[JobState]:
        """Jobs not yet deployed."""
        return [j for j in self.jobs if j.status == "pending"]

    @property
    def active_jobs(self) -> list[JobState]:
        """Jobs that are deploying or running."""
        return [j for j in self.jobs if j.status in ("deploying", "running")]

    @property
    def completed_jobs(self) -> list[JobState]:
        """Jobs completed but not downloaded."""
        return [j for j in self.jobs if j.status == "completed"]

    @property
    def total_cost(self) -> float:
        """Total estimated spend."""
        return sum(j.cost_usd for j in self.jobs)
