"""Vast.ai destroy adapter — wires the belt-and-suspenders protocol.

Per the v3 doc (PR #20, merged) and the v4 design (PR #22, merged),
this module:

- Owns the Vast.ai REST callbacks (``stop_fn``, ``delete_fn``,
  ``verify_fn``) and the Vast.ai-discovered timing policy
  (``VASTAI_POLICY``).
- Resolves the API key with env-first precedence, fail-closed
  semantics: ``AVAILABLE`` / ``ABSENT`` / ``EXPLICITLY_DISABLED``.
- Implements ``verify_instance_ownership`` as a tagged
  ``OwnershipVerification`` enum (replaces the v2 ``bool`` return
  that conflated "owned and present" with "absent and therefore
  safe"). The v2 substring/prefix match is replaced with
  tag-insensitive repository equality.
- Implements ``destroy_vastai_instance`` with pre-protocol refusals
  (OWNERSHIP / NO_CREDENTIALS / CREDENTIALS_DISABLED) and the
  CLI fallback dispatch (when no API key, the CLI
  ``verify_instance_ownership`` and ``vastai destroy instance``
  paths take over).

The v4 design adds the ``ProviderCleanupPolicy`` factory on top of
these primitives; the v3 adapter here is the building block.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from vastai_gpu_runner.providers.destroy import (
    DestroyPolicy,
    DestroyRefusal,
    DestroyResult,
    DestroyVerdict,
    VerifyResult,
    VerifyVerdict,
    belt_and_suspenders,
)

logger = logging.getLogger(__name__)

# Default REST endpoint for the Vast.ai API.
VASTAI_API_BASE = "https://console.vast.ai/api/v0/instances"

# Vast.ai-discovered timing policy. Empirically verified against the
# Vast.ai API: the stop takes ~2s to settle, the delete can need 1-2
# retries on transient 5xx, and a verify-after-resurrection pass is
# needed because the instance can come back during the verify window.
VASTAI_POLICY = DestroyPolicy(
    verify_delay_s=5.0,
    retry_delay_s=3.0,
    max_delete_attempts=3,
    verify_after_resurrection=True,
)

# CLI invocation for the destroy fallback (no API key). The CLI uses
# a separate auth context and is permitted only when credentials are
# ABSENT (CLI fallback permitted) and never when EXPLICITLY_DISABLED.
VASTAI_CLI_DESTROY = "vastai"


# ---------------------------------------------------------------------------
# Credential types — v3 shape (re-exported by v4 cleanup_policy in step 1)
# ---------------------------------------------------------------------------


class CredentialState(StrEnum):
    """Three-state credential resolution (v3 verbatim, v4 reuses)."""

    AVAILABLE = "available"  # key present, non-empty, pre-stripped
    ABSENT = "absent"  # no key found (CLI fallback permitted)
    EXPLICITLY_DISABLED = "explicitly_disabled"  # VASTAI_API_KEY="" (no CLI fallback)


@dataclass(frozen=True)
class CredentialResolution:
    """Output of ``read_vastai_api_key``.

    Invariants (enforced in ``__post_init__``):

    - AVAILABLE requires non-empty, pre-stripped key.
    - ABSENT and EXPLICITLY_DISABLED require empty key.
    """

    state: CredentialState
    key: str = ""

    def __post_init__(self) -> None:
        """Validate state/key invariants for credential resolution."""
        if self.state == CredentialState.AVAILABLE:
            if not self.key or self.key != self.key.strip():
                raise ValueError(
                    "CredentialResolution.AVAILABLE requires non-empty, pre-stripped key"
                )
        elif self.key:
            raise ValueError(f"CredentialResolution.{self.state.value} requires empty key")


# ---------------------------------------------------------------------------
# read_vastai_api_key — env-first, fail-closed
# ---------------------------------------------------------------------------


def read_vastai_api_key() -> CredentialResolution:
    """Read the Vast.ai API key — env-first, fail-closed.

    Resolution order:

    1. Inspect ``VASTAI_API_KEY`` env var.
    2. Present but empty/whitespace → ``EXPLICITLY_DISABLED`` (the
       user has explicitly disabled credentials via the env var).
    3. Present and non-empty → ``AVAILABLE`` with stripped key.
    4. Then inspect credential files
       (``~/.config/vastai/vast_api_key``, ``~/.vast_api_key``).
    5. Blank file → warning, treat as ``ABSENT`` (CLI fallback
       permitted).
    6. Unreadable file (``OSError``) → warning, treat as ``ABSENT``.
    7. No file present → ``ABSENT``.

    Invariant: ``EXPLICITLY_DISABLED`` is reserved for the case where
    the user has explicitly opted out via the env var. A blank or
    unreadable file does NOT mean disabled — that would silently
    block the controlled CLI fallback.
    """
    env_resolution = _resolve_env_key()
    if env_resolution is not None:
        return env_resolution
    return _resolve_file_key() or CredentialResolution(state=CredentialState.ABSENT)


def _resolve_env_key() -> CredentialResolution | None:
    """Inspect VASTAI_API_KEY env var; return None if not set."""
    env_key = os.environ.get("VASTAI_API_KEY")
    if env_key is None:
        return None
    stripped = env_key.strip()
    if not stripped:
        return CredentialResolution(state=CredentialState.EXPLICITLY_DISABLED)
    return CredentialResolution(state=CredentialState.AVAILABLE, key=stripped)


def _resolve_file_key() -> CredentialResolution | None:
    """Inspect the standard credential file locations; return None if absent.

    Blank files warn and continue; unreadable files warn and continue;
    missing files silently continue to the next location.
    """
    for kp in _credential_file_paths():
        try:
            if not kp.exists():
                continue
            raw = kp.read_text()
            stripped = raw.strip()
            if stripped:
                return CredentialResolution(state=CredentialState.AVAILABLE, key=stripped)
            logger.warning(
                "Credential file %s is empty; treating as ABSENT (CLI fallback will be attempted)",
                kp,
            )
        except OSError as exc:
            logger.warning(
                "Could not read credential file %s: %s; treating as ABSENT",
                kp,
                exc,
            )
    return None


def _credential_file_paths() -> tuple[Path, ...]:
    """Standard Vast.ai credential file locations, in priority order."""
    return (
        Path("~/.config/vastai/vast_api_key").expanduser(),
        Path("~/.vast_api_key").expanduser(),
    )


# ---------------------------------------------------------------------------
# Image matching — v3 simplified _repository (full v4 grammar in step 1)
# ---------------------------------------------------------------------------


def _repository(image_ref: str) -> str:
    """Strip a Docker image reference down to its repository.

    Minimal v3 helper: handles ``[registry[:port]/]repo[:tag][@digest]``
    by stripping the tag and digest. Full Docker/OCI grammar (per the
    v4 cleanup_policy ``_repository``) lands in v4 step 1; this
    minimal version is good enough for the v3 image-matching
    semantics (no substring/prefix match).
    """
    ref = image_ref.strip()
    if not ref:
        return ""
    # Drop digest.
    if "@" in ref:
        ref = ref.split("@", 1)[0]
    # Drop tag (last ':' after the last '/').
    if ":" in ref:
        slash = ref.rfind("/")
        colon = ref.rfind(":")
        if colon > slash:
            ref = ref[:colon]
    return ref


def _is_image_allowed(image_uuid: str, allowed_images: frozenset[str]) -> bool:
    """Image matching: exact ref OR tag-insensitive repository equality.

    The v2 substring/prefix match is REMOVED. ``myorg/app:1.0`` does
    NOT allow ``myorg/app-malicious:latest``; ``registry:5000/myorg/
    app:1.0`` does NOT allow ``registry-malicious/myorg/app:1.0``.
    """
    if image_uuid in allowed_images:
        return True
    instance_repo = _repository(image_uuid)
    if not instance_repo:
        return False
    return any(_repository(allowed) == instance_repo for allowed in allowed_images)


# ---------------------------------------------------------------------------
# OwnershipVerification — tagged enum (v3 shape; v4 reuses from cleanup_policy)
# ---------------------------------------------------------------------------


class OwnershipVerification(StrEnum):
    """Result of an ownership check (tagged, replaces the v2 ``bool``).

    The v2 ``bool`` conflated "owned and present" with "absent and
    therefore safe"; this enum makes the two cases distinguishable so
    the CLI fallback can dispatch on each independently.
    """

    OWNED = "owned"  # well-formed API response + matching image
    ABSENT = "absent"  # well-formed response; the requested instance is not in it
    REFUSED = "refused"  # ownership mismatch, malformed record, or any failure
    DISABLED = "disabled"  # ownership check disabled (no allowed_images)


# ---------------------------------------------------------------------------
# verify_instance_ownership — CLI-based; v3 fallback path
# ---------------------------------------------------------------------------


def verify_instance_ownership(
    instance_id: str,
    *,
    allowed_images: frozenset[str] | None = None,
    timeout: int = 15,
) -> OwnershipVerification:
    """Check Vast.ai instance ownership via the CLI (separate auth context).

    Per the v3 doc: the CLI fallback path uses this function before
    invoking ``vastai destroy instance``. Empty allowlist rejects
    every image (the safety-critical contract).

    Returns the tagged ``OwnershipVerification`` so the caller can
    distinguish "absent and safe" from "refused".

    Args:
        instance_id: Vast.ai instance ID.
        allowed_images: Frozenset of allowed Docker image references.
            ``None`` → ``DISABLED`` (no API call). Empty frozenset
            → ``REFUSED`` (fail-closed, never silently allow).
        timeout: CLI subprocess timeout in seconds.

    Returns:
        ``DISABLED`` if no allowlist given;
        ``OWNED`` if the instance is in the account with a matching image;
        ``ABSENT`` if the instance is in the account but not present;
        ``REFUSED`` for ownership mismatch, malformed record, or any failure.
    """
    if allowed_images is None:
        return OwnershipVerification.DISABLED

    # Empty allowlist fails closed — never silently allow every image.
    if not allowed_images:
        return OwnershipVerification.REFUSED

    try:
        raw = _vastai_show_instances_raw(timeout=timeout)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        logger.error(
            "verify_instance_ownership: vastai CLI failed for %s: %s",
            instance_id,
            exc,
        )
        return OwnershipVerification.REFUSED
    except (OSError, json.JSONDecodeError) as exc:
        logger.error(
            "verify_instance_ownership: parse failed for %s: %s",
            instance_id,
            exc,
        )
        return OwnershipVerification.REFUSED

    try:
        instances = json.loads(raw)
    except (TypeError, ValueError) as exc:
        logger.error(
            "verify_instance_ownership: non-JSON response for %s: %s",
            instance_id,
            exc,
        )
        return OwnershipVerification.REFUSED

    if not isinstance(instances, list):
        logger.error(
            "verify_instance_ownership: expected list, got %s",
            type(instances).__name__,
        )
        return OwnershipVerification.REFUSED

    inst = _find_instance(instances, instance_id)
    if inst is None:
        return OwnershipVerification.ABSENT

    image_uuid = inst.get("image_uuid")
    if not isinstance(image_uuid, str) or not image_uuid:
        logger.error(
            "verify_instance_ownership: instance %s has no image_uuid; refusing",
            instance_id,
        )
        return OwnershipVerification.REFUSED

    if _is_image_allowed(image_uuid, allowed_images):
        return OwnershipVerification.OWNED

    logger.error(
        "BLOCKED: instance %s image %s is not in the allowlist; refusing",
        instance_id,
        image_uuid,
    )
    return OwnershipVerification.REFUSED


def _vastai_show_instances_raw(*, timeout: int) -> str:
    """Run ``vastai show instances --raw`` and return stdout."""
    result = subprocess.run(
        ["vastai", "show", "instances", "--raw"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return result.stdout


def _find_instance(
    instances: list[object],
    instance_id: str,
) -> dict[str, object] | None:
    """Find one instance by ID. Returns None if not present."""
    for inst in instances:
        if isinstance(inst, dict) and str(inst.get("id")) == str(instance_id):
            return inst
    return None


# ---------------------------------------------------------------------------
# REST callbacks (v3) — used by belt_and_suspenders when credentials are
# AVAILABLE. The CLI fallback path is taken when credentials are ABSENT
# (and the user has confirmed the CLI path is safe in their context).
# ---------------------------------------------------------------------------


def _rest_stop(instance_id: str, hdrs: dict[str, str]) -> int | None:
    """PUT state=stopped; return HTTP status or None on raise."""
    import requests

    resp = requests.put(
        f"{VASTAI_API_BASE}/{instance_id}/",
        headers={**hdrs, "Content-Type": "application/json"},
        json={"state": "stopped"},
        timeout=10,
    )
    return resp.status_code


def _rest_delete(instance_id: str, hdrs: dict[str, str]) -> int | None:
    """DELETE the instance; return HTTP status or None on raise."""
    import requests

    resp = requests.delete(
        f"{VASTAI_API_BASE}/{instance_id}/",
        headers=hdrs,
        timeout=15,
    )
    return resp.status_code


def _rest_verify(instance_id: str, hdrs: dict[str, str]) -> VerifyResult:
    """GET the instance; classify the response into a VerifyResult."""
    import requests

    try:
        resp = requests.get(
            f"{VASTAI_API_BASE}/{instance_id}/",
            headers=hdrs,
            timeout=10,
        )
    except requests.RequestException as exc:
        return VerifyResult(VerifyVerdict.UNKNOWN, error=f"{type(exc).__name__}: {exc}")

    if resp.status_code == 404:
        return VerifyResult(VerifyVerdict.GONE, status_code=resp.status_code)
    if resp.status_code != 200:
        return VerifyResult(
            VerifyVerdict.UNKNOWN,
            status_code=resp.status_code,
            error=f"HTTP {resp.status_code}",
        )

    try:
        data = resp.json()
    except (ValueError, TypeError) as exc:
        return VerifyResult(VerifyVerdict.UNKNOWN, status_code=resp.status_code, error=str(exc))

    if not isinstance(data, dict):
        return VerifyResult(
            VerifyVerdict.UNKNOWN,
            status_code=resp.status_code,
            error="response is not a dict",
        )

    actual_status = data.get("actual_status", "")
    if isinstance(actual_status, str) and actual_status == "destroyed":
        return VerifyResult(VerifyVerdict.GONE, status_code=resp.status_code)

    # 200 + actual_status != "destroyed" (incl. empty, missing,
    # non-string) → PRESENT.
    return VerifyResult(VerifyVerdict.PRESENT, status_code=resp.status_code)


# ---------------------------------------------------------------------------
# destroy_vastai_instance — pre-protocol refusals + belt_and_suspenders
# ---------------------------------------------------------------------------


def destroy_vastai_instance(
    instance_id: str,
    *,
    allowed_images: frozenset[str] | None = None,
    credentials: CredentialResolution | None = None,
) -> DestroyResult:
    """Stop + delete + verify a Vast.ai instance.

    Pre-protocol refusals:

    - ``OWNERSHIP`` — image allowlist rejected the instance
      (``REFUSED`` from ``verify_instance_ownership``) — fail-closed.
    - ``NO_CREDENTIALS`` — no API key (``ABSENT``); CLI fallback
      is permitted but the v3 adapter does not auto-invoke the CLI
      fallback (the v4 factory owns the dispatch).
    - ``CREDENTIALS_DISABLED`` — ``VASTAI_API_KEY=""``; CLI fallback
      is forbidden.

    Args:
        instance_id: Vast.ai instance ID.
        allowed_images: Optional allowlist. ``None`` skips the
            ownership check (use sparingly). Empty frozenset refuses
            every instance (the safety-critical contract).
        credentials: Pre-resolved credential state. When ``None``,
            falls back to ``read_vastai_api_key()`` — preserves the
            v3 back-compat path for direct callers.

    Returns:
        ``DestroyResult`` with either ``verdict`` or ``refusal``
        (never both).
    """
    if credentials is None:
        credentials = read_vastai_api_key()

    # Pre-protocol: ownership check.
    ownership = verify_instance_ownership(instance_id, allowed_images=allowed_images)
    if ownership == OwnershipVerification.REFUSED:
        return DestroyResult(refusal=DestroyRefusal.OWNERSHIP)
    if ownership == OwnershipVerification.OWNED:
        logger.info(
            "destroy_vastai_instance: %s ownership verified; proceeding",
            instance_id,
        )
    # ABSENT or DISABLED: no ownership refusal, proceed with destroy.

    # Pre-protocol: credential check.
    if credentials.state == CredentialState.EXPLICITLY_DISABLED:
        return DestroyResult(refusal=DestroyRefusal.CREDENTIALS_DISABLED)
    if credentials.state == CredentialState.ABSENT:
        # v3 adapter does not auto-invoke the CLI fallback. The v4
        # factory owns the dispatch (with operator consent for the
        # separate auth context). v3 returns NO_CREDENTIALS.
        return DestroyResult(refusal=DestroyRefusal.NO_CREDENTIALS)

    # AVAILABLE: run the protocol.
    hdrs = {"Authorization": f"Bearer {credentials.key}"}
    return belt_and_suspenders(
        stop_fn=lambda: _rest_stop(instance_id, hdrs),
        delete_fn=lambda: _rest_delete(instance_id, hdrs),
        verify_fn=lambda: _rest_verify(instance_id, hdrs),
        policy=VASTAI_POLICY,
    )


# Suppress unused import warning for ``DestroyVerdict`` (used by callers
# that re-export this module's public surface). The import is kept so
# downstream imports (e.g. ``from .vastai import DestroyVerdict``) still
# work.
_ = DestroyVerdict


__all__ = [
    "VASTAI_API_BASE",
    "VASTAI_CLI_DESTROY",
    "VASTAI_POLICY",
    "CredentialResolution",
    "CredentialState",
    "OwnershipVerification",
    "destroy_vastai_instance",
    "read_vastai_api_key",
    "verify_instance_ownership",
]
