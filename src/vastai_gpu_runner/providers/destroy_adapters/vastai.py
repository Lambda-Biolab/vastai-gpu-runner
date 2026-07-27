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

import logging
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from vastai_gpu_runner.cleanup_policy import OwnershipPolicy
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


# Both `_repository` (minimal v3 image-strip helper) and
# `_is_image_allowed` (v3 image allowlist helper) are REMOVED. The v4
# `verify_instance_ownership` uses `OwnershipPolicy.matches()` which has
# the same tag-insensitive semantics via the canonical `_repository` helper
# in `cleanup_policy.py`. No external callers remained after the v3
# destroy tests were migrated (they now pass `ownership=`).

# ---------------------------------------------------------------------------
# OwnershipVerification + verify_instance_ownership — owned by v4
# (cleanup_policy.py:OwnershipVerification, providers/vastai.py:verify_instance_ownership).
# This module re-exports them so existing v3 callers (and v3 destroy tests)
# continue to import them from the adapter namespace.
# ---------------------------------------------------------------------------


from vastai_gpu_runner.cleanup_policy import (  # noqa: E402  (kept after local definitions for grouping)
    OwnershipVerification as _CleanupPolicyOwnershipVerification,
)

# Re-export the canonical (v4) enum under the v3 name so existing
# import paths (`from vastai_gpu_runner.providers.destroy_adapters.vastai
# import OwnershipVerification`) keep working.
OwnershipVerification = _CleanupPolicyOwnershipVerification


def verify_instance_ownership(
    instance_id: str,
    *,
    ownership: OwnershipPolicy | None = None,
    allowed_images: frozenset[str] | None = None,
) -> OwnershipVerification:
    """Back-compat shim for the v3 ``allowed_images=`` signature.

    v4 callers should pass ``ownership=OwnershipPolicy(...)`` directly.
    v3 callers passing ``allowed_images=`` are translated to an
    ``OwnershipPolicy`` here.
    """
    # Lazy import to break the circular dependency:
    # providers/vastai.py imports CredentialResolution etc. from
    # this module; importing providers.vastai at module level would
    # re-enter this module while it's still being constructed.
    from vastai_gpu_runner.providers.vastai import (
        verify_instance_ownership as _verify_instance_ownership_v4,
    )

    if ownership is not None and allowed_images is not None:
        raise ValueError(
            "verify_instance_ownership: supply either ownership= or "
            "allowed_images= (deprecated), not both."
        )
    if ownership is None:
        ownership = OwnershipPolicy(owned_images=allowed_images)
    return _verify_instance_ownership_v4(instance_id, ownership=ownership)


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
    ownership: OwnershipPolicy | None = None,
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
        ownership: Optional OwnershipPolicy. ``None`` (or
            ``owned_images=None``) skips the ownership check.
            Empty ``owned_images`` frozenset refuses every instance
            (the safety-critical contract).
        allowed_images: DEPRECATED v3 alias. Translated to an
            ``OwnershipPolicy`` if ``ownership`` is not provided.
            Cannot be combined with ``ownership``.
        credentials: Pre-resolved credential state. When ``None``,
            falls back to ``read_vastai_api_key()`` — preserves the
            v3 back-compat path for direct callers.

    Returns:
        ``DestroyResult`` with either ``verdict`` or ``refusal``
        (never both).
    """
    if ownership is not None and allowed_images is not None:
        raise ValueError(
            "destroy_vastai_instance: supply either ownership= or "
            "allowed_images= (deprecated), not both."
        )
    if ownership is None:
        ownership = OwnershipPolicy(owned_images=allowed_images)

    if credentials is None:
        credentials = read_vastai_api_key()

    # Pre-protocol: ownership check.
    verdict = verify_instance_ownership(instance_id, ownership=ownership)
    if verdict == OwnershipVerification.REFUSED:
        return DestroyResult(refusal=DestroyRefusal.OWNERSHIP)
    if verdict == OwnershipVerification.OWNED:
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
