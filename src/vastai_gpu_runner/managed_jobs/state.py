"""Versioned persistent state for managed-job orchestration.

Mirrors the v4 :mod:`vastai_gpu_runner.state` loader conventions:
schema-versioned, fail-closed on unknown or unsupported versions,
backward-compatible ``load_or_none`` for legacy callers.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CURRENT_MANAGED_JOB_SCHEMA = 2
_VALID_MANAGED_JOB_SCHEMA_VERSIONS: frozenset[int] = frozenset({0, 1, CURRENT_MANAGED_JOB_SCHEMA})


class ManagedJobStateError(RuntimeError):
    """Raised when managed-job state cannot be loaded or migrated."""


def _empty_object_dict() -> dict[str, Any]:
    return {}


def _normalise_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ManagedJobStateError(f"managed-job state {field_name} must be an object")
    return dict(value)


def _normalise_identifier(value: Any) -> str:
    return "" if value is None else str(value)


@dataclass
class ManagedJobState:
    """Persistent state for a managed-job attempt.

    One state file per (job attempt). The controller writes it
    after a successful :meth:`ManagedJobRunner.submit` and reads it
    on resume; a present state with a successful submit means the
    job is the canonical handle for this attempt and must not be
    resubmitted.
    """

    schema_version: int = CURRENT_MANAGED_JOB_SCHEMA
    provider: str = ""
    resource_name: str = ""
    location: str = ""
    task_count: int = 1
    state: str = "submitted"
    succeeded_tasks: int = 0
    failed_tasks: int = 0
    attempt: int = 0
    # Deprecated aliases retained so callers can load and resave old state
    # without losing the identifiers they used before schema 2.
    campaign_id: str = ""
    stage_id: str = ""
    extra: dict[str, Any] = field(default_factory=_empty_object_dict)
    correlation_metadata: dict[str, Any] = field(default_factory=_empty_object_dict)
    correlation: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Normalise correlation aliases and persisted object fields."""
        metadata = _normalise_mapping(self.correlation_metadata, "correlation_metadata")
        for key, value in _normalise_mapping(self.correlation, "correlation").items():
            metadata.setdefault(key, value)
        self.campaign_id = _normalise_identifier(self.campaign_id)
        self.stage_id = _normalise_identifier(self.stage_id)
        if self.campaign_id:
            metadata.setdefault("campaign_id", self.campaign_id)
        if self.stage_id:
            metadata.setdefault("stage_id", self.stage_id)
        self.correlation_metadata = metadata
        self.extra = _normalise_mapping(self.extra, "extra")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state to a JSON-safe dictionary."""
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "resource_name": self.resource_name,
            "location": self.location,
            "task_count": self.task_count,
            "state": self.state,
            "succeeded_tasks": self.succeeded_tasks,
            "failed_tasks": self.failed_tasks,
            "attempt": self.attempt,
            "correlation_metadata": dict(self.correlation_metadata),
            "campaign_id": self.campaign_id,
            "stage_id": self.stage_id,
            "extra": dict(self.extra),
        }


def load_managed_job_state(path: Path) -> ManagedJobState:
    """Load a managed-job state file; fail closed on unknown schema."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ManagedJobStateError(f"managed-job state not found: {path}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ManagedJobStateError(f"could not read managed-job state: {path}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManagedJobStateError(f"managed-job state is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ManagedJobStateError("managed-job state JSON root must be an object")
    return _load_payload(payload, path)


def _load_payload(payload: dict[str, Any], path: Path) -> ManagedJobState:
    """Validate, migrate, and construct a state from a parsed object."""
    try:
        payload = _migrate_payload(payload)
        return _build_state_from_payload(payload, path)
    except ManagedJobStateError:
        raise
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        raise ManagedJobStateError(f"managed-job state migration failed: {path}") from exc


def _migrate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Migrate a parsed payload and normalise all supported aliases."""
    version = payload.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ManagedJobStateError(f"managed-job state schema_version is invalid: {version!r}")
    if version not in _VALID_MANAGED_JOB_SCHEMA_VERSIONS:
        raise ManagedJobStateError(
            f"managed-job state schema_version {version!r} is not supported; "
            f"expected one of {sorted(_VALID_MANAGED_JOB_SCHEMA_VERSIONS)}"
        )

    if version == 0:
        logger.info("managed-job state: migrating v0 → v%d", CURRENT_MANAGED_JOB_SCHEMA)
        payload = _migrate_v0_to_v1(payload)
        version = 1
    if version == 1:
        logger.info("managed-job state: migrating v1 → v%d", CURRENT_MANAGED_JOB_SCHEMA)
        payload = _migrate_v1_to_v2(payload)
    return _normalise_correlation_fields(payload)


def load_or_none(path: Path) -> ManagedJobState | None:
    """Legacy load: return ``None`` if the state file is missing or unreadable."""
    try:
        return load_managed_job_state(path)
    except ManagedJobStateError:
        return None


def _migrate_v0_to_v1(payload: dict[str, Any]) -> dict[str, Any]:
    """Migrate a v0 managed-job state payload to the v1 shape."""
    payload["schema_version"] = 1
    payload.setdefault("task_count", 1)
    payload.setdefault("attempt", 0)
    payload.setdefault("succeeded_tasks", 0)
    payload.setdefault("failed_tasks", 0)
    payload.setdefault("extra", {})
    return payload


def _migrate_v1_to_v2(payload: dict[str, Any]) -> dict[str, Any]:
    """Move stage-specific correlation fields into neutral metadata."""
    payload = dict(payload)
    correlation = _normalise_mapping(payload.get("correlation_metadata"), "correlation_metadata")
    if payload.get("campaign_id"):
        correlation.setdefault("campaign_id", str(payload["campaign_id"]))
    if payload.get("stage_id"):
        correlation.setdefault("stage_id", str(payload["stage_id"]))
    payload["correlation_metadata"] = correlation
    payload["schema_version"] = CURRENT_MANAGED_JOB_SCHEMA
    return payload


def _normalise_correlation_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Backfill canonical metadata from every supported legacy alias."""
    payload = dict(payload)
    metadata = _normalise_mapping(payload.get("correlation_metadata"), "correlation_metadata")
    for key, value in _normalise_mapping(payload.get("correlation"), "correlation").items():
        metadata.setdefault(key, value)
    campaign_id = _normalise_identifier(payload.get("campaign_id"))
    stage_id = _normalise_identifier(payload.get("stage_id"))
    if campaign_id:
        metadata.setdefault("campaign_id", campaign_id)
    if stage_id:
        metadata.setdefault("stage_id", stage_id)
    payload["correlation_metadata"] = metadata
    payload["campaign_id"] = campaign_id
    payload["stage_id"] = stage_id
    return payload


def _build_state_from_payload(payload: dict[str, Any], path: Path) -> ManagedJobState:
    required = ("provider", "resource_name", "state")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ManagedJobStateError(
            f"managed-job state at {path} is missing required keys: {missing}"
        )
    return ManagedJobState(
        schema_version=CURRENT_MANAGED_JOB_SCHEMA,
        provider=str(payload["provider"]),
        resource_name=str(payload["resource_name"]),
        location=str(payload.get("location", "")),
        task_count=int(payload.get("task_count", 1)),
        state=str(payload["state"]),
        succeeded_tasks=int(payload.get("succeeded_tasks", 0)),
        failed_tasks=int(payload.get("failed_tasks", 0)),
        attempt=int(payload.get("attempt", 0)),
        correlation_metadata=_normalise_mapping(
            payload.get("correlation_metadata"), "correlation_metadata"
        ),
        campaign_id=_normalise_identifier(payload.get("campaign_id")),
        stage_id=_normalise_identifier(payload.get("stage_id")),
        extra=_normalise_mapping(payload.get("extra"), "extra"),
    )


def save_managed_job_state(state: ManagedJobState, path: Path) -> None:
    """Atomically write a managed-job state to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
