"""Versioned persistent state for managed-job orchestration.

Mirrors the v4 :mod:`vastai_gpu_runner.state` loader conventions:
schema-versioned, fail-closed on unknown or unsupported versions,
backward-compatible ``load_or_none`` for legacy callers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CURRENT_MANAGED_JOB_SCHEMA = 1
_VALID_MANAGED_JOB_SCHEMA_VERSIONS: frozenset[int] = frozenset({0, CURRENT_MANAGED_JOB_SCHEMA})


class ManagedJobStateError(RuntimeError):
    """Raised when managed-job state cannot be loaded or migrated."""


def _empty_object_dict() -> dict[str, Any]:
    return {}


@dataclass
class ManagedJobState:
    """Persistent state for a managed-job attempt.

    One state file per (stage, attempt). The orchestrator writes it
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
    campaign_id: str = ""
    stage_id: str = ""
    extra: dict[str, Any] = field(default_factory=_empty_object_dict)

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

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManagedJobStateError(f"managed-job state is not valid JSON: {path}") from exc

    version = payload.get("schema_version")
    if version not in _VALID_MANAGED_JOB_SCHEMA_VERSIONS:
        raise ManagedJobStateError(
            f"managed-job state schema_version {version!r} is not supported; "
            f"expected one of {sorted(_VALID_MANAGED_JOB_SCHEMA_VERSIONS)}"
        )

    if version == 0:
        logger.info("managed-job state: migrating v0 → v%d", CURRENT_MANAGED_JOB_SCHEMA)
        payload = _migrate_v0_to_v1(payload)

    return _build_state_from_payload(payload, path)


def load_or_none(path: Path) -> ManagedJobState | None:
    """Legacy load: return ``None`` if the state file is missing or unreadable."""
    try:
        return load_managed_job_state(path)
    except ManagedJobStateError:
        return None


def _migrate_v0_to_v1(payload: dict[str, Any]) -> dict[str, Any]:
    """Migrate a v0 managed-job state payload to the current schema."""
    payload["schema_version"] = CURRENT_MANAGED_JOB_SCHEMA
    payload.setdefault("task_count", 1)
    payload.setdefault("attempt", 0)
    payload.setdefault("succeeded_tasks", 0)
    payload.setdefault("failed_tasks", 0)
    payload.setdefault("extra", {})
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
        campaign_id=str(payload.get("campaign_id", "")),
        stage_id=str(payload.get("stage_id", "")),
        extra=dict(payload.get("extra", {})),
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
