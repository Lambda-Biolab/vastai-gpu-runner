"""Vast.ai marketplace runner implementation.

Implements the CloudRunner interface for Vast.ai's GPU marketplace.
Encodes all UTI-project deployment lessons as default behavior.

Requires: ``pip install vastai`` or the ``vastai`` CLI tool.

Usage::

    from vastai_gpu_runner.providers.vastai import VastaiRunner

    runner = VastaiRunner(allowed_images=frozenset({"my/image:latest"}))
    result = runner.run_full_cycle(files, output_dir)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
import warnings
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import requests

from vastai_gpu_runner.cleanup_policy import (
    CleanupRefusal,
    CleanupResult,
    CleanupVerdict,
    InstanceCandidate,
    OwnershipPolicy,
    OwnershipVerification,
    ProviderCleanupPolicy,
    normalize_instance_id,
)
from vastai_gpu_runner.providers.destroy import (
    DestroyRefusal,
    DestroyResult,
    DestroyVerdict,
)
from vastai_gpu_runner.providers.destroy_adapters.vastai import (
    CredentialResolution,
    CredentialState,
    destroy_vastai_instance,
    read_vastai_api_key,
)
from vastai_gpu_runner.runner import CloudRunner
from vastai_gpu_runner.ssh import scp_download, scp_upload, ssh_cmd
from vastai_gpu_runner.types import (
    CloudInstance,
    DeploymentConfig,
    InstanceStatus,
    Provider,
)

logger = logging.getLogger(__name__)

# Vast.ai GPU name mapping
GPU_NAME_MAP: dict[str, str] = {
    "RTX_3090": "RTX 3090",
    "RTX_4090": "RTX 4090",
    "RTX_5090": "RTX 5090",
}

# Default Docker image (bare CUDA runtime)
DEFAULT_IMAGE = "nvidia/cuda:12.4.0-devel-ubuntu22.04"

# Minimum GPU VRAM in MiB
MIN_GPU_VRAM_MIB = 20_000


def _get_image_cuda_version(image: str) -> str:
    """Extract required CUDA version from a Docker image.

    Tries ``docker inspect`` labels first, falls back to parsing the image
    tag (e.g. ``cuda:12.4.1`` -> ``"12.4"``).

    Args:
        image: Docker image name with tag.

    Returns:
        CUDA major.minor version string (e.g. ``"12.4"``).
    """
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                '{{index .Config.Labels "cuda_version"}}',
                image,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        version = result.stdout.strip()
        if version and version != "<no value>":
            parts = version.split(".")[:2]
            return ".".join(parts)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: parse tag string for cuda version pattern
    match = re.search(r"cuda[:\-](\d+\.\d+)", image)
    if match:
        return match.group(1)
    return "12.4"


def vastai_cmd(args: list[str], *, timeout: int = 30) -> str:
    """Run a vastai CLI command.

    Args:
        args: Command arguments (after 'vastai').
        timeout: Command timeout in seconds.

    Returns:
        stdout text.

    Raises:
        RuntimeError: If command fails.
    """
    cmd = ["vastai", *args]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            msg = f"vastai {' '.join(args)} failed: {result.stderr.strip()}"
            raise RuntimeError(msg)
        return result.stdout.strip()
    except FileNotFoundError as exc:
        msg = "vastai CLI not installed. Install with: pip install vastai"
        raise RuntimeError(msg) from exc
    except subprocess.TimeoutExpired as exc:
        msg = f"vastai {' '.join(args)} timed out after {timeout}s"
        raise RuntimeError(msg) from exc


# v4 migration notes (deleted):
# - v2 ``verify_instance_ownership`` (returns bool) → tagged
#   ``OwnershipVerification`` enum (in providers/vastai.py).
# - v2 ``_image_is_allowed`` (substring/prefix match) →
#   ``cleanup_policy.OwnershipPolicy.matches()``.
# - v2 ``sweep_zombie_instances`` (in orchestrator.py) →
#   ``ProviderCleanupPolicy`` driven from batch._sweep_zombies.
# - v3 destroy adapter's local ``_image_is_allowed`` / ``_repository``
#   deleted in v4 step 3c.


# ---------------------------------------------------------------------------
# list_vastai_instances — credential-aware enumeration (v4)
# ---------------------------------------------------------------------------

# States that are already-destroyed or otherwise terminal. Anything
# not in this set is processed by the cleanup policy. Using a negative
# allowlist (terminal-skip) is conservative: new Vast.ai states that
# are not yet terminal are processed, not silently skipped.
VASTAI_TERMINAL_STATES: frozenset[str] = frozenset({"destroyed"})

# Official `vastai show instances` REST endpoint; keyset pages are
# capped at 25.
VASTAI_INSTANCES_URL = "https://console.vast.ai/api/v1/instances"
VASTAI_INSTANCES_PAGE_SIZE = 25
VASTAI_MAX_INSTANCE_PAGES = 1_000


def _validate_rest_page(payload: object, seen_tokens: set[str]) -> str | None:
    """Validate one page of REST instances; return next_token or None."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"Vast.ai REST instances response must be an object, got {type(payload).__name__}"
        )
    page = payload.get("instances")
    if not isinstance(page, list):
        raise ValueError("Vast.ai REST instances response has no list-valued 'instances'")

    raw_next_token = payload.get("next_token")
    if raw_next_token is None:
        return None
    if not isinstance(raw_next_token, str) or not raw_next_token or raw_next_token in seen_tokens:
        raise ValueError("Vast.ai REST instances response has an invalid pagination token")
    return raw_next_token


def _list_vastai_instances_rest(api_key: str) -> list[object]:
    """Enumerate all instances through REST using one explicit API key."""
    records: list[object] = []
    after_token: str | None = None
    seen_tokens: set[str] = set()

    for _page_number in range(VASTAI_MAX_INSTANCE_PAGES):
        params: dict[str, int | str] = {"limit": VASTAI_INSTANCES_PAGE_SIZE}
        if after_token is not None:
            params["after_token"] = after_token
        response = requests.get(
            VASTAI_INSTANCES_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        next_token = _validate_rest_page(payload, seen_tokens)
        page = payload.get("instances", [])  # type: ignore[union-attr]
        records.extend(page)
        if next_token is None:
            return records
        seen_tokens.add(next_token)
        after_token = next_token

    raise ValueError(
        f"Vast.ai REST instances pagination exceeded {VASTAI_MAX_INSTANCE_PAGES} pages"
    )


def _list_vastai_instances_cli() -> list[object]:
    """Enumerate instances through the ambient Vast.ai CLI context."""
    raw = vastai_cmd(["show", "instances", "--raw"], timeout=15)
    instances: object = json.loads(raw)
    if not isinstance(instances, list):
        raise ValueError(
            f"Vast.ai CLI instances response must be a list, got {type(instances).__name__}"
        )
    return instances


def _string_or_empty(value: object) -> str:
    """Normalize nullable provider strings without creating literal ``"None"``."""
    return value if isinstance(value, str) else ""


def _float_or_zero(value: object) -> float:
    """Normalize nullable provider numerics to ``0.0`` (non-negative finite).

    Booleans are explicitly rejected (per the InstanceCandidate
    invariant): ``True`` would otherwise coerce to ``1.0`` and
    silently shadow a real numeric value. Other non-numeric values
    return ``0.0`` rather than raising; the InstanceCandidate
    invariant then enforces the non-negative finite contract.
    """
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)) and value == value:  # not NaN
        return float(value)
    return 0.0


def _build_vastai_candidate(
    inst: dict[str, object],
    canonical_id: str,
) -> InstanceCandidate:
    """Build one InstanceCandidate from a validated raw record.

    Centralised here so the per-record try/except block in
    ``list_vastai_instances`` stays short and the
    InstanceCandidate invariant is enforced in one place.
    """
    image_uuid = _string_or_empty(inst.get("image_uuid"))
    return InstanceCandidate(
        provider=Provider.VASTAI,
        instance_id=canonical_id,
        image_uuid=image_uuid,
        ownership_key=image_uuid,  # Vast.ai: image_uuid is the ownership key
        gpu_model=_string_or_empty(inst.get("gpu_name")),
        cost_per_hour=_float_or_zero(inst.get("dph_total")),
        label=_string_or_empty(inst.get("label")),
        state=_string_or_empty(inst.get("actual_status")),
        started_at=_float_or_zero(inst.get("start_date")),
    )


def _append_record(
    candidates: list[InstanceCandidate],
    seen_candidate_ids: set[str],
    inst: object,
) -> bool:
    """Append one record as an InstanceCandidate if valid.

    Returns False if the caller should abort the enumeration
    (duplicate canonical IDs discard everything). Logs warnings
    for skipped records. Other malformed-field errors are caught
    by the per-record ``try`` block in the caller.
    """
    if not isinstance(inst, dict):
        logger.warning("Skipping non-object instance record: %r", inst)
        return True
    canonical_id = normalize_instance_id(inst.get("id"))
    if canonical_id is None:
        logger.warning(
            "Skipping instance with invalid ID: %r",
            {k: inst.get(k) for k in ("label", "image_uuid", "actual_status")},
        )
        return True
    if canonical_id in seen_candidate_ids:
        logger.error(
            "list_vastai_instances: duplicate canonical instance ID %s; "
            "discarding the entire enumeration",
            canonical_id,
        )
        return False
    seen_candidate_ids.add(canonical_id)
    try:
        candidates.append(_build_vastai_candidate(inst, canonical_id))
    except (TypeError, ValueError) as exc:
        logger.warning("Skipping malformed instance: %s", exc)
    return True


def _enumerate_provider_instances(
    credentials: CredentialResolution,
) -> list[object]:
    """Pick REST or CLI enumeration based on the credential snapshot."""
    if credentials.state == CredentialState.AVAILABLE:
        return _list_vastai_instances_rest(credentials.key)
    return _list_vastai_instances_cli()


def list_vastai_instances(
    *,
    credentials: CredentialResolution,
) -> list[InstanceCandidate]:
    """Enumerate Vast.ai instances using the canonical credential snapshot.

    ``AVAILABLE`` credentials use the paginated REST endpoint with
    exactly ``credentials.key``. ``ABSENT`` credentials use the
    ambient CLI context because the CLI ownership verifier and CLI
    destroy fallback use that same context. ``EXPLICITLY_DISABLED``
    returns ``[]`` without contacting either provider interface.

    Records whose shared ID normalizer returns ``None`` are skipped
    (logged). A failure to enumerate returns an empty list.

    Validation guards:
        - ``inst`` must be a dict (otherwise skip).
        - ``normalize_instance_id(inst.get("id"))`` must return a
          canonical ID; this rejects ``None``, ``bool``, non-``str`` /
          non-``int`` values, empty strings, and blank strings.
        - Valid IDs are canonicalised by the shared normalizer, so
          padded strings and integers match verifier semantics.
        - Duplicate canonical IDs, whether within one page or across
          REST pages, discard the entire enumeration before any
          candidate can reach label filtering or destruction.
        - Nullable or non-string ``image_uuid``, ``label``, and
          ``actual_status`` values normalize to ``""``; an empty label
          cannot match a non-empty scope and an empty state is refused
          by the factory.
        - Other malformed fields are caught by the per-record
          ``try`` block.

    The factory passes the same immutable ``CredentialResolution``
    snapshot later to the destroy adapter, so enumeration and REST
    destruction cannot silently use different accounts.
    """
    if credentials.state == CredentialState.EXPLICITLY_DISABLED:
        logger.warning(
            "Zombie sweep disabled: VASTAI_API_KEY is explicitly empty; "
            "provider enumeration was not attempted."
        )
        return []

    try:
        instances = _enumerate_provider_instances(credentials)
    except Exception as exc:
        source = "REST" if credentials.state == CredentialState.AVAILABLE else "CLI"
        logger.error(
            "list_vastai_instances: %s enumeration failed: %s",
            source,
            exc,
        )
        return []

    candidates: list[InstanceCandidate] = []
    seen_candidate_ids: set[str] = set()
    for inst in instances:
        if not _append_record(candidates, seen_candidate_ids, inst):
            return []
    return candidates


# ---------------------------------------------------------------------------
# verify_instance_ownership — CLI-side ownership check (v4)
# ---------------------------------------------------------------------------


def _validate_verifier_record(
    raw_record: object,
    instance_id: str,
    seen_ids: set[str],
) -> tuple[str, str] | None:
    """Validate one record from the verifier CLI response.

    Returns ``(canonical_id, image_uuid)`` on success or ``None``
    on any failure (the caller should ``REFUSED`` and abort).
    Defensive: malformed record → None.
    """
    if not isinstance(raw_record, dict):
        logger.error(
            "REFUSING: instance %s cannot be verified — "
            "response contains a non-object record: %r. "
            "Destroy refused.",
            instance_id,
            raw_record,
        )
        return None
    canonical_id = normalize_instance_id(raw_record.get("id"))
    if canonical_id is None:
        logger.error(
            "REFUSING: instance %s cannot be verified — "
            "record has missing/null/invalid id: %r. Destroy refused.",
            instance_id,
            raw_record,
        )
        return None
    if canonical_id in seen_ids:
        logger.error(
            "REFUSING: instance %s cannot be verified — "
            "response contains duplicate canonical ID %s. Destroy refused.",
            instance_id,
            canonical_id,
        )
        return None
    seen_ids.add(canonical_id)
    image_uuid = raw_record.get("image_uuid")
    if not isinstance(image_uuid, str):
        logger.error(
            "REFUSING: instance %s cannot be verified — "
            "record has non-string/null image_uuid: %r. Destroy refused.",
            instance_id,
            raw_record,
        )
        return None
    return canonical_id, image_uuid


def _eval_ownership_match(
    normalised_records: list[tuple[str, str]],
    instance_id: str,
    canonical_target: str,
    ownership: OwnershipPolicy,
) -> OwnershipVerification:
    """Locate the target record in the validated list and evaluate."""
    for record_id, image_uuid in normalised_records:
        if record_id == canonical_target:
            if ownership.matches(image_uuid):
                return OwnershipVerification.OWNED
            logger.error(
                "BLOCKED: instance %s belongs to another project (image=%s). Will NOT destroy.",
                instance_id,
                image_uuid,
            )
            return OwnershipVerification.REFUSED
    logger.info(
        "Instance %s not found in account (already destroyed?)",
        instance_id,
    )
    return OwnershipVerification.ABSENT


def _fetch_instances_for_verification(
    instance_id: str,
) -> list[object] | None:
    """Fetch + parse the CLI instances response for verification.

    Returns the parsed list on success, ``None`` on any
    subprocess / JSON / type-shape failure (the caller
    translates ``None`` to ``REFUSED``).
    """
    try:
        raw = vastai_cmd(["show", "instances", "--raw"], timeout=15)
        instances = json.loads(raw)
    except (RuntimeError, json.JSONDecodeError) as exc:
        logger.error(
            "REFUSING: cannot verify ownership of instance %s "
            "(API error: %s) — destroy refused. Resolve the API "
            "response and retry.",
            instance_id,
            exc,
        )
        return None
    if not isinstance(instances, list):
        logger.error(
            "REFUSING: cannot verify ownership of instance %s — "
            "response is not a list (got %s). Destroy refused.",
            instance_id,
            type(instances).__name__,
        )
        return None
    return instances


def _normalise_response(
    instances: list[object],
    instance_id: str,
) -> list[tuple[str, str]] | None:
    """Validate and normalise every record in the response.

    Returns the list of (canonical_id, image_uuid) tuples on
    success, ``None`` on any malformed-record / duplicate-id
    failure (the caller translates ``None`` to ``REFUSED``).
    """
    normalised: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for raw_record in instances:
        validated = _validate_verifier_record(raw_record, instance_id, seen_ids)
        if validated is None:
            return None
        normalised.append(validated)
    return normalised


def verify_instance_ownership(
    instance_id: str,
    *,
    ownership: OwnershipPolicy,
) -> OwnershipVerification:
    """CLI-side ownership check for a single instance (v4 factory).

    This is the CLI-auth verification used by the v4 factory's
    CLI fallback path (separate auth context from the REST path).
    Returns a tagged ``OwnershipVerification`` so the caller can
    distinguish four operationally different outcomes that v2
    conflated into ``bool``:

    - ``OWNED``    → instance exists and is owned; CLI destroy may
                     proceed.
    - ``ABSENT``   → instance is not in the API response (already
                     destroyed). The verifier has proved absence
                     from a fully well-formed response and the
                     caller can choose the desired verdict for
                     "already gone" without invoking destruction.
    - ``REFUSED``  → instance exists but is unowned, OR the API
                     response was so malformed that absence cannot
                     be proved, OR any other verifier-level failure.
                     Refusal means "do not destroy". Failure is
                     fail-closed.
    - ``DISABLED`` → ownership checking is disabled (no policy).
                     This is a short-circuit that bypasses the API
                     call entirely.

    Implementation notes (carefully ordered):

    1. **One short-circuit up front** (DISABLED): skip the API call
       entirely when ``owned_images is None``.
    2. **One canonical form for the requested ID**: ``normalize_instance_id``.
       Padded, numeric, and string IDs reduce to the same canonical
       form before comparison.
    3. **One outermost ``except Exception`` boundary**: subprocess
       failures, JSON parsing, response-shape problems, or even
       downstream ``ownership.matches`` failures cannot escape the
       function. (They're translated to ``REFUSED``.)
    4. **Validate-and-normalize the entire response first**, then
       locate and evaluate the requested record. A single malformed
       entry anywhere in the list fails the whole check
       (conservatively — we cannot prove absence of one ID when
       another entry is unreadable). Duplicate canonical IDs are
       also refused because conflicting records cannot prove
       ownership deterministically. We do NOT short-circuit on
       the first match, because a malformed record *after* a match
       would otherwise be ignored.
    5. **Failures are logged at ERROR** level so the operator sees
       the destroy has been refused (not "ambiguous state").

    The function never returns ``True`` / ``False`` — the contract
    is encoded in the enum so callers must explicitly handle each
    case. v2's conflated bool cannot represent "instance is owned
    AND present" separately from "instance is already absent".
    """
    if ownership.owned_images is None:
        return OwnershipVerification.DISABLED

    try:
        canonical_target = normalize_instance_id(instance_id)
        if canonical_target is None:
            logger.error(
                "REFUSING: requested instance_id %r is not a valid ID — refusing to destroy",
                instance_id,
            )
            return OwnershipVerification.REFUSED

        instances = _fetch_instances_for_verification(instance_id)
        if instances is None:
            return OwnershipVerification.REFUSED

        normalised = _normalise_response(instances, instance_id)
        if normalised is None:
            return OwnershipVerification.REFUSED

        return _eval_ownership_match(normalised, instance_id, canonical_target, ownership)
    except Exception as exc:
        logger.error(
            "REFUSING: cannot verify ownership of instance %s — "
            "unexpected error: %s. Destroy refused.",
            instance_id,
            exc,
        )
        return OwnershipVerification.REFUSED


# ---------------------------------------------------------------------------
# _describe_destroy_result + build_vastai_cleanup_policy (v4 factory)
# ---------------------------------------------------------------------------


def _describe_destroy_result(result: DestroyResult) -> str:
    """Build diagnostic text from v3 DestroyResult structured fields.

    Single shared helper used by both ``VastaiRunner.destroy_instance``
    and ``build_vastai_cleanup_policy._destroy``. Includes verdict +
    refusal so the fallback "unrecognised result" log exposes the
    actual typed outcome. Handles ``None`` fields gracefully (v3 uses
    optional diagnostics). Always produces non-empty output.
    """
    return (
        f"verdict={result.verdict!r}, refusal={result.refusal!r}, "
        f"attempts={result.attempts}, "
        f"last_status_code={result.last_status_code}, "
        f"verify_error={result.verify_error!r}, "
        f"stop_error={result.stop_error!r}"
    )


def _refusal_from_destroy(
    candidate: InstanceCandidate,
    result: DestroyResult,
    refusal: CleanupRefusal,
    error_prefix: str,
) -> CleanupResult:
    """Translate a v3 DestroyRefusal into a v4 CleanupResult.

    Includes the structured destroy diagnostic in the error
    message so the operator can see WHY the v3 adapter refused
    (the v4 cleanup_policy doesn't suppress it).
    """
    return CleanupResult(
        refusal=refusal,
        error=(f"{error_prefix} {candidate.instance_id!r}; {_describe_destroy_result(result)}"),
    )


def _verdict_from_destroy(
    candidate: InstanceCandidate,
    result: DestroyResult,
    verdict: CleanupVerdict,
) -> CleanupResult:
    """Translate a v3 DestroyVerdict into a v4 CleanupResult."""
    return CleanupResult(
        verdict=verdict,
        error=_describe_destroy_result(result),
    )


def _cli_fallback(
    candidate: InstanceCandidate,
    ownership: OwnershipPolicy,
) -> CleanupResult:
    """CLI fallback path for ABSENT credentials.

    The v3 adapter returns NO_CREDENTIALS when the API key is
    not configured. We then verify ownership via the CLI auth
    context (which uses the file-based key) and dispatch on
    the verifier's tagged result:

    - ``OWNED``    → run CLI destroy; report ``CLI_ATTEMPTED``
                     (destruction unconfirmed — CLI exit is not
                     a REST confirmation).
    - ``DISABLED`` → ownership checking is off; proceed straight
                     to CLI destroy (same ``CLI_ATTEMPTED``
                     translation as ``OWNED``).
    - ``ABSENT``   → the verifier proved the instance is already
                     gone from a fully well-formed response.
                     Short-circuit to ``ALREADY_GONE`` without
                     invoking CLI destroy (a 'not found' CLI
                     error would otherwise fabricate a spurious
                     ``UNKNOWN``).
    - ``REFUSED``  → the instance exists but is unowned, or the
                     response is too malformed to prove absence.
                     Refuse with ``OWNERSHIP`` and a diagnostic.

    The function never raises: any unexpected exception in
    CLI destroy becomes an ``UNKNOWN`` outcome with a
    non-empty error string.
    """
    try:
        verification = verify_instance_ownership(candidate.instance_id, ownership=ownership)
    except Exception as exc:
        return CleanupResult(
            verdict=CleanupVerdict.UNKNOWN,
            error=(f"CLI ownership verification raised {type(exc).__name__}: {exc}"),
        )
    if verification == OwnershipVerification.REFUSED:
        return CleanupResult(
            refusal=CleanupRefusal.OWNERSHIP,
            error=(
                f"CLI ownership check refused {candidate.instance_id!r} "
                "(instance unowned or response malformed)"
            ),
        )
    if verification == OwnershipVerification.ABSENT:
        return CleanupResult(verdict=CleanupVerdict.ALREADY_GONE)
    if verification not in (
        OwnershipVerification.OWNED,
        OwnershipVerification.DISABLED,
    ):
        # Defensive fail-closed: any unexpected verifier result
        # (including the bool->OwnershipVerification conversion
        # failing to migrate cleanly) refuses destruction.
        return CleanupResult(
            refusal=CleanupRefusal.OWNERSHIP,
            error=(
                "CLI ownership verifier returned unexpected result "
                f"{verification!r}; refusing to destroy "
                f"{candidate.instance_id!r}"
            ),
        )
    # OWNED or DISABLED — proceed to CLI destroy.
    try:
        vastai_cmd(["destroy", "instance", candidate.instance_id], timeout=15)
        return CleanupResult(
            verdict=CleanupVerdict.CLI_ATTEMPTED,
            error="CLI fallback ran; destruction not confirmed via REST",
        )
    except Exception as exc:
        return CleanupResult(
            verdict=CleanupVerdict.UNKNOWN,
            error=f"CLI destroy raised {type(exc).__name__}: {exc}",
        )


def _check_eligibility(candidate: InstanceCandidate) -> CleanupResult | None:
    """Eligibility check (negative allowlist: skip terminal states).

    Returns a ``CleanupResult(INELIGIBLE_STATE)`` if the candidate
    is not eligible, otherwise ``None`` to proceed.
    """
    if not candidate.state:
        return CleanupResult(
            refusal=CleanupRefusal.INELIGIBLE_STATE,
            error="empty state",
        )
    if candidate.state in VASTAI_TERMINAL_STATES:
        return CleanupResult(
            refusal=CleanupRefusal.INELIGIBLE_STATE,
            error=f"state {candidate.state!r} is terminal (already destroyed)",
        )
    return None


def _destroy_one(
    candidate: InstanceCandidate,
    ownership: OwnershipPolicy,
    credentials: CredentialResolution,
) -> CleanupResult:
    """Run the v3 adapter on one candidate; translate to v4 CleanupResult.

    Steps:
    1. Eligibility check (skip terminal states like ``destroyed``).
    2. CREDENTIALS_DISABLED: refuse without provider calls.
    3. Delegate to v3 adapter with the canonical ownership + credentials.
    4. Translate the v3 ``DestroyResult`` into a v4 ``CleanupResult``:
       - NO_CREDENTIALS → run the CLI fallback (the v4 factory owns
         this dispatch).
       - OWNERSHIP / CREDENTIALS_DISABLED → translate the refusal.
       - DESTROYED → verdict=DESTROYED (empty error).
       - LEAKED → verdict=LEAKED with structured diagnostic.
       - UNKNOWN → verdict=UNKNOWN with structured diagnostic.
    """
    ineligible = _check_eligibility(candidate)
    if ineligible is not None:
        return ineligible

    if credentials.state == CredentialState.EXPLICITLY_DISABLED:
        return CleanupResult(
            refusal=CleanupRefusal.CREDENTIALS_DISABLED,
            error="VASTAI_API_KEY explicitly empty",
        )

    result = destroy_vastai_instance(
        candidate.instance_id, ownership=ownership, credentials=credentials
    )

    if result.refusal == DestroyRefusal.NO_CREDENTIALS:
        return _cli_fallback(candidate, ownership)
    if result.refusal == DestroyRefusal.OWNERSHIP:
        return _refusal_from_destroy(
            candidate,
            result,
            CleanupRefusal.OWNERSHIP,
            "v3 adapter refused: ownership check rejected",
        )
    if result.refusal == DestroyRefusal.CREDENTIALS_DISABLED:
        return _refusal_from_destroy(
            candidate,
            result,
            CleanupRefusal.CREDENTIALS_DISABLED,
            "v3 adapter refused: VASTAI_API_KEY explicitly empty;",
        )
    if result.verdict == DestroyVerdict.DESTROYED:
        return CleanupResult(verdict=CleanupVerdict.DESTROYED)
    if result.verdict == DestroyVerdict.LEAKED:
        return _verdict_from_destroy(candidate, result, CleanupVerdict.LEAKED)
    return _verdict_from_destroy(candidate, result, CleanupVerdict.UNKNOWN)


def build_vastai_cleanup_policy(
    *,
    ownership: OwnershipPolicy,
    credentials: CredentialResolution,
) -> ProviderCleanupPolicy:
    """Build a Vast.ai cleanup policy from the canonical objects.

    The two arguments are the same ``OwnershipPolicy`` and
    ``CredentialResolution`` instances that the runner was
    constructed with (the batch command extracts them from the
    ``VastaiProviderConfig``; the cleanup command constructs them
    directly from CLI args).

    The wired ``list_instances_fn`` uses the canonical credential
    snapshot: ``AVAILABLE`` enumerates through REST with the exact
    API key, ``ABSENT`` uses the ambient CLI context, and
    ``EXPLICITLY_DISABLED`` returns ``[]`` without any provider call.

    The wired ``destroy_fn`` runs:
        1. Eligibility check (skip terminal states like ``destroyed``).
        2. ``destroy_vastai_instance`` with the canonical ownership + credentials.
        3. If the adapter returns ``NO_CREDENTIALS``, run the CLI
           fallback path: CLI ownership verification → CLI destroy →
           ``CLI_ATTEMPTED`` (destruction unconfirmed) or ``UNKNOWN``.
        4. Translate all three v3 verdicts and all three v3 refusals
           into the v4 ``CleanupResult`` shape.
    """

    def _list_instances() -> list[InstanceCandidate]:
        return list_vastai_instances(credentials=credentials)

    def _destroy(candidate: InstanceCandidate) -> CleanupResult:
        return _destroy_one(candidate, ownership, credentials)

    return ProviderCleanupPolicy(
        provider=Provider.VASTAI,
        list_instances_fn=_list_instances,
        destroy_fn=_destroy,
    )


# ---------------------------------------------------------------------------
# VastaiProviderConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VastaiProviderConfig:
    """Canonical Vast.ai configuration shared by runner factory + cleanup policy.

    The runner factory (``VastaiRunner.from_config``) reads
    ``ownership``, ``credentials``, ``label_prefix``, ``docker_image``, and
    ``setup_commands`` from this. The cleanup-policy factory
    (``build_vastai_cleanup_policy``) reads only ``ownership`` and
    ``credentials`` — the deployment-image invariant does not apply
    to listing/cleanup-only commands.

    Invariants:
        - ``docker_image`` is non-empty and pre-stripped.
        - ``docker_image`` is in ``ownership.owned_images`` unless
          ``ownership.owned_images is None`` (ownership check disabled).
        - ``credentials`` is a v3 ``CredentialResolution`` (frozen).
        - ``label_prefix`` is either ``None`` (non-batch runner) or
          a non-empty, pre-stripped string; batch composition always
          supplies a unique scope.
    """

    docker_image: str = DEFAULT_IMAGE
    ownership: OwnershipPolicy = field(default_factory=OwnershipPolicy)
    credentials: CredentialResolution = field(
        default_factory=lambda: CredentialResolution(state=CredentialState.ABSENT)
    )
    label_prefix: str | None = None
    min_gpu_vram_mib: int = MIN_GPU_VRAM_MIB
    setup_commands: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate label_prefix, docker_image, and docker_image ownership invariant."""
        if self.label_prefix is not None and (
            not isinstance(self.label_prefix, str)  # pyright: ignore[reportUnnecessaryIsInstance]
            or not self.label_prefix
            or self.label_prefix != self.label_prefix.strip()
        ):
            raise ValueError(
                "VastaiProviderConfig.label_prefix must be None or a non-empty, pre-stripped string"
            )
        if not self.docker_image or self.docker_image != self.docker_image.strip():
            raise ValueError(
                "VastaiProviderConfig.docker_image must be a non-empty, pre-stripped reference"
            )
        if self.ownership.owned_images is not None and not self.ownership.matches(
            self.docker_image
        ):
            raise ValueError(
                f"VastaiProviderConfig invariant violated: docker_image="
                f"{self.docker_image!r} is not owned by "
                f"ownership.owned_images="
                f"{set(self.ownership.owned_images)!r}"
            )

    @classmethod
    def from_env(
        cls,
        *,
        docker_image: str | None = None,
        owned_images: AbstractSet[str] | None = None,
        label_prefix: str | None = None,
    ) -> VastaiProviderConfig:
        """Build from environment / config files.

        Reads the v3 ``read_vastai_api_key()`` into ``credentials``.
        The CLI uses ``dataclasses.replace`` to overlay the
        project-specific values on top of this base.

        Note: explicit empty ``docker_image`` is rejected by
        ``__post_init__``; only ``None`` triggers the default.
        """
        resolved_image = DEFAULT_IMAGE if docker_image is None else docker_image
        return cls(
            docker_image=resolved_image,
            ownership=OwnershipPolicy(owned_images=owned_images),
            credentials=read_vastai_api_key(),
            label_prefix=label_prefix,
        )


class VastaiRunner(CloudRunner):
    """Vast.ai marketplace runner with hardened deployment.

    Args:
        config: Deployment configuration.
        ownership: Ownership policy. The runner and the cleanup
            adapter both call ``ownership.matches(image_ref)``.
        credentials: Pre-resolved credential state. When None,
            the runner passes ``None`` to the adapter, which
            calls ``read_vastai_api_key()`` (the v3 back-compat path).
        label_prefix: Immutable batch label scope. Batch composition
            always supplies a unique, validated scope; non-batch
            callers may leave it None.
        docker_image: Docker image to use for new instances.
        min_gpu_vram_mib: Minimum GPU VRAM required (default 20 GB).
        setup_commands: Optional pre-instance setup commands.

    Note: ``allowed_images`` is a deprecated back-compat alias
    that builds an ``OwnershipPolicy`` from the given set.
    Simultaneous ``ownership=`` and ``allowed_images=`` raises
    ``ValueError`` — silently ignoring one destruction-safety
    configuration is unsafe.
    """

    def __init__(
        self,
        config: DeploymentConfig | None = None,
        *,
        ownership: OwnershipPolicy | None = None,
        credentials: CredentialResolution | None = None,
        label_prefix: str | None = None,
        allowed_images: frozenset[str] | None = None,  # DEPRECATED
        docker_image: str = DEFAULT_IMAGE,
        min_gpu_vram_mib: int = MIN_GPU_VRAM_MIB,
        setup_commands: list[str] | None = None,
    ) -> None:
        """Initialize Vast.ai runner with deployment config and safety guards."""
        if label_prefix is not None and (
            not isinstance(label_prefix, str)  # pyright: ignore[reportUnnecessaryIsInstance]
            or not label_prefix
            or label_prefix != label_prefix.strip()
        ):
            raise ValueError(
                "VastaiRunner.label_prefix must be None or a non-empty, pre-stripped string"
            )
        if ownership is not None and allowed_images is not None:
            raise ValueError(
                "VastaiRunner: supply either ownership= or allowed_images= "
                "(deprecated), not both — silently ignoring one destruction-"
                "safety configuration is unsafe."
            )
        if ownership is None and allowed_images is not None:
            warnings.warn(
                "VastaiRunner(allowed_images=...) is deprecated; "
                "build an OwnershipPolicy and pass ownership= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            ownership = OwnershipPolicy(owned_images=frozenset(allowed_images))
        super().__init__(config)
        self.ownership = ownership
        self.credentials = credentials
        self.label_prefix = label_prefix
        # Back-compat read access for existing callers.
        self._allowed_images_for_backcompat = (
            frozenset(ownership.owned_images)
            if ownership and ownership.owned_images is not None
            else None
        )
        self.docker_image = docker_image
        self.min_gpu_vram_mib = min_gpu_vram_mib
        self._setup_commands = setup_commands or []

    @property
    def allowed_images(self) -> frozenset[str] | None:
        """Back-compat read access. New code should use ``self.ownership``."""
        return self._allowed_images_for_backcompat

    @classmethod
    def from_config(cls, canonical: VastaiProviderConfig) -> VastaiRunner:
        """Build a VastaiRunner from the canonical config.

        Preserves the canonical ``OwnershipPolicy`` and
        ``CredentialResolution`` instances unchanged — the runner
        and the cleanup policy will call ``ownership.matches()`` and
        use ``credentials`` from the same instances.
        """
        return cls(
            ownership=canonical.ownership,
            credentials=canonical.credentials,
            label_prefix=canonical.label_prefix,
            docker_image=canonical.docker_image,
            min_gpu_vram_mib=canonical.min_gpu_vram_mib,
            setup_commands=list(canonical.setup_commands),
        )

    def _next_instance_label(self) -> str:
        """Create a unique label inside this runner's immutable batch scope."""
        prefix = self.label_prefix or "gpu-runner"
        return f"{prefix}-{uuid4().hex[:12]}"

    def search_offers(self, **kwargs: object) -> list[dict[str, object]]:
        """Search Vast.ai marketplace for matching GPU offers."""
        docker_img = str(kwargs.get("docker_image", self.docker_image))
        gpu_name = GPU_NAME_MAP.get(self.config.gpu_model, self.config.gpu_model)
        cuda_ver = _get_image_cuda_version(docker_img)
        logger.info("Filtering Vast.ai offers for CUDA >= %s (from image)", cuda_ver)
        query = (
            f'gpu_name="{gpu_name}" '
            f"num_gpus=1 "
            f"rentable=true "
            f"cuda_max_good>={cuda_ver} "
            f"dph<={self.config.max_cost_per_hour} "
            f"inet_down>={self.config.min_network_mbps} "
            f"reliability>={self.config.min_reliability}"
        )

        try:
            output = vastai_cmd(
                ["search", "offers", query, "--order", "dph", "--limit", "20", "--raw"],
                timeout=30,
            )
            offers: list[dict[str, object]] = json.loads(output)
            logger.info("Found %d Vast.ai offers for %s", len(offers), gpu_name)
            return offers
        except (RuntimeError, json.JSONDecodeError) as exc:
            logger.error("Failed to search Vast.ai offers: %s", exc)
            return []

    def create_instance(self, offer: Mapping[str, object]) -> CloudInstance:
        """Create a Vast.ai instance from an offer."""
        offer_id = str(offer.get("id", ""))
        label = self._next_instance_label()

        try:
            output = vastai_cmd(
                [
                    "create",
                    "instance",
                    offer_id,
                    "--image",
                    self.docker_image,
                    "--disk",
                    str(self.config.min_disk_gb),
                    "--label",
                    label,
                    "--raw",
                ],
                timeout=30,
            )

            data = json.loads(output)
            instance_id = str(data.get("new_contract", data.get("id", offer_id)))

            return CloudInstance(
                provider=Provider.VASTAI,
                instance_id=instance_id,
                gpu_model=str(offer.get("gpu_name", self.config.gpu_model)),
                cost_per_hour=float(str(offer.get("dph_total", 0.0))),
                status=InstanceStatus.CREATING,
                label=label,
            )
        except (RuntimeError, json.JSONDecodeError, KeyError) as exc:
            msg = f"Failed to create Vast.ai instance: {exc}"
            raise RuntimeError(msg) from exc

    def wait_for_boot(self, instance: CloudInstance) -> bool:
        """Wait for Vast.ai instance to reach 'running' status."""
        deadline = time.time() + self.config.boot_timeout_seconds
        instance.status = InstanceStatus.BOOTING

        while time.time() < deadline:
            try:
                output = vastai_cmd(
                    ["show", "instance", instance.instance_id, "--raw"],
                    timeout=15,
                )
                data = json.loads(output)
                status = data.get("actual_status", "")

                if status == "running":
                    ssh_host = data.get("ssh_host", "")
                    ssh_port = int(data.get("ssh_port", 22))
                    if ssh_host:
                        instance.ssh_host = ssh_host
                        instance.ssh_port = ssh_port
                        instance.status = InstanceStatus.RUNNING
                        logger.info(
                            "Instance %s is running (SSH: %s:%d)",
                            instance.instance_id,
                            ssh_host,
                            ssh_port,
                        )
                        return True

            except (RuntimeError, json.JSONDecodeError):
                pass

            time.sleep(5)

        logger.warning(
            "Instance %s stuck in boot after %ds",
            instance.instance_id,
            self.config.boot_timeout_seconds,
        )
        # Caller (_try_one_offer) now owns the cleanup path: it calls
        # capture_deploy_failure_diagnostics BEFORE destroy_instance so
        # subclasses can pull ``vastai logs`` / ssh diagnostics while
        # the instance still exists. Previously we destroyed here
        # inline, which erased the container before diagnostics could
        # run and made boot-timeout failures unobservable.
        instance.status = InstanceStatus.FAILED
        return False

    def verify_gpu(self, instance: CloudInstance) -> bool:
        """Verify GPU is accessible and has sufficient VRAM."""
        deadline = time.time() + self.config.gpu_verify_timeout

        while time.time() < deadline:
            rc, output = ssh_cmd(
                instance,
                "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits",
            )
            if rc == 0:
                try:
                    parts = output.strip().split("\n")[0].split(",")
                    mem_used = int(parts[0].strip())
                    mem_total = int(parts[1].strip())
                    if mem_total >= self.min_gpu_vram_mib:
                        logger.info(
                            "GPU verified: %d/%d MiB (used/total)",
                            mem_used,
                            mem_total,
                        )
                        return True
                    logger.warning(
                        "GPU VRAM too low: %d MiB < %d MiB required",
                        mem_total,
                        self.min_gpu_vram_mib,
                    )
                    return False
                except (ValueError, IndexError):
                    pass
            time.sleep(3)

        logger.warning("GPU verification failed for instance %s", instance.instance_id)
        return False

    def deploy_files(
        self,
        instance: CloudInstance,
        files: dict[str, Path],
    ) -> bool:
        """Upload files via SCP."""
        ssh_cmd(instance, f"mkdir -p {self.config.workspace_dir}")

        for remote_name, local_path in files.items():
            if not local_path.exists():
                logger.warning("Local file not found: %s", local_path)
                continue

            remote_path = f"{self.config.workspace_dir}/{remote_name}"
            if not scp_upload(instance, local_path, remote_path):
                return False

        return True

    def setup_environment(self, instance: CloudInstance) -> bool:
        """Run environment setup commands on the instance.

        If ``setup_commands`` was provided at construction, runs those.
        Otherwise, if ``conda_env_spec`` is set in the config, installs
        micromamba + creates a conda environment. If neither is set,
        skips setup entirely (assumes Docker image is ready).

        Override this method for fully custom setup logic.
        """
        commands = self._setup_commands
        if not commands and not self.config.conda_env_spec:
            logger.info("No setup commands or conda_env_spec — skipping environment setup")
            return True

        if not commands:
            # Default: micromamba + conda env
            commands = [
                "apt-get update -qq && apt-get install -y -qq bzip2 ca-certificates",
                "curl -kL -o /tmp/mm.tar.bz2 "
                '"https://micro.mamba.pm/api/micromamba/linux-64/latest"',
                "mkdir -p /opt/micromamba"
                " && tar -xjf /tmp/mm.tar.bz2 -C /opt/micromamba --strip-components=1",
                "/opt/micromamba/bin/micromamba create -y -n env"
                f" -c conda-forge {self.config.conda_env_spec}",
            ]

        for cmd in commands:
            rc, output = ssh_cmd(instance, cmd, timeout=600)
            if rc != 0:
                logger.error("Setup command failed: %s -> %s", cmd[:50], output[:200])
                return False
            logger.debug("Setup OK: %s", cmd[:50])

        logger.info("Environment setup complete on %s", instance.instance_id)
        return True

    def launch_worker(self, instance: CloudInstance) -> bool:
        """Launch the worker script on the instance."""
        ws = self.config.workspace_dir
        worker_script = self.config.worker_script

        # Check for duplicate workers
        rc, output = ssh_cmd(instance, f"pgrep -f {worker_script}")
        if rc == 0 and output.strip():
            logger.warning("Worker already running on %s — skipping launch", instance.instance_id)
            return True

        launch_cmd = f"cd {ws} && nohup bash {worker_script} > {ws}/worker.log 2>&1 &"

        rc, _ = ssh_cmd(instance, launch_cmd, timeout=30)
        if rc != 0:
            logger.error("Worker launch failed on %s", instance.instance_id)
            return False

        time.sleep(5)
        rc, output = ssh_cmd(instance, f"pgrep -f {worker_script}")
        if rc != 0:
            logger.error("Worker process not found after launch on %s", instance.instance_id)
            return False

        logger.info("Worker launched on %s", instance.instance_id)
        return True

    def check_progress(self, instance: CloudInstance) -> dict[str, object]:
        """Check worker progress via DONE file and PID liveness."""
        ws = self.config.workspace_dir

        rc, _ = ssh_cmd(instance, f"test -f {ws}/DONE")
        if rc == 0:
            return {"running": False, "complete": True}

        # Check if worker PID is alive (detects silent preemption)
        rc_pid, pid_str = ssh_cmd(instance, f"cat {ws}/worker.pid 2>/dev/null", timeout=5)
        if rc_pid == 0 and pid_str.strip().isdigit():
            rc_alive, _ = ssh_cmd(instance, f"kill -0 {pid_str.strip()} 2>/dev/null", timeout=5)
            if rc_alive != 0:
                logger.warning(
                    "Worker PID %s is dead on %s but no DONE file — silent crash",
                    pid_str.strip(),
                    instance.instance_id,
                )
                return {
                    "running": False,
                    "complete": False,
                    "worker_dead": True,
                    "log_tail": f"Worker PID {pid_str.strip()} dead, no DONE file",
                }

        rc, output = ssh_cmd(instance, f"tail -3 {ws}/worker.log", timeout=10)
        return {
            "running": True,
            "complete": False,
            "log_tail": output,
        }

    def list_remote_files(self, instance: CloudInstance) -> list[str]:
        """List all files in workspace."""
        ws = self.config.workspace_dir
        rc, output = ssh_cmd(instance, f"ls -1 {ws}/", timeout=10)
        if rc != 0:
            return []
        return [f.strip() for f in output.splitlines() if f.strip()]

    def download_file(
        self,
        instance: CloudInstance,
        remote_name: str,
        local_path: Path,
    ) -> bool:
        """Download a single file via SCP."""
        remote_path = f"{self.config.workspace_dir}/{remote_name}"
        return scp_download(instance, remote_path, local_path)

    def capture_deploy_failure_diagnostics(
        self,
        instance: CloudInstance,
        error: str,
        attempt: int,
    ) -> None:
        """Pull ``vastai logs`` + SSH dmesg/nvidia-smi before destroy.

        Vast.ai does not retain container logs after ``destroy_instance``
        (``vastai logs <id>`` returns 404 on the underlying docker
        container). This is our one chance to capture why a deploy gate
        failed. Saves to ``batch_diagnostics/deploy__{unit_or_id}_{ts}.log``
        under the current working directory — mirrors the layout used by
        ``BatchOrchestrator.capture_preempt_diagnostics``.

        Always swallows exceptions; a diagnostic capture must NEVER block
        the destroy that follows.
        """
        try:
            diag_dir = Path.cwd() / "batch_diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            iid = instance.instance_id or "unknown"
            out_path = diag_dir / f"deploy__{iid}_{timestamp}.log"

            sections: list[str] = [
                f"# deploy-failure diagnostics for instance {iid}",
                f"# attempt: {attempt}",
                f"# error: {error}",
                f"# ssh: {instance.ssh_user}@{instance.ssh_host}:{instance.ssh_port}",
                f"# captured_at: {timestamp}",
            ]

            # vastai-level container logs (fetched from Vast's log storage,
            # which holds content for some seconds after container stop).
            try:
                vlogs = vastai_cmd(["logs", iid], timeout=30)
                sections.extend(["", "## vastai logs ##", vlogs])
            except Exception as exc:
                sections.extend(["", "## vastai logs FAILED ##", str(exc)])

            # SSH-level diagnostics: workspace worker.log if it exists,
            # plus dmesg tail + nvidia-smi for kernel/driver state.
            ws = self.config.workspace_dir
            rc, output = ssh_cmd(
                instance,
                (
                    f"cat {ws}/worker.log 2>/dev/null; "
                    f"echo '---DMESG---'; dmesg -T 2>/dev/null | tail -50; "
                    f"echo '---NVIDIA-SMI---'; nvidia-smi 2>&1 | head -30; "
                    f"echo '---DF---'; df -h {ws} 2>/dev/null || df -h"
                ),
                timeout=20,
            )
            sections.extend(
                [
                    "",
                    f"## ssh diagnostics (rc={rc}) ##",
                    output or "(empty)",
                ]
            )

            out_path.write_text("\n".join(sections) + "\n")
            logger.info(
                "Deploy-failure diagnostics captured (%d sections) → %s",
                len(sections),
                out_path,
            )
        except Exception as exc:
            logger.warning(
                "capture_deploy_failure_diagnostics swallowed exception: %s",
                exc,
            )

    def destroy_instance(self, instance: CloudInstance) -> bool:
        """Destroy a Vast.ai instance — delegates to the v3 adapter + logs typed results.

        Per the v4 doc:
            The v2 implementation did inline ownership pre-check +
            CLI destroy + belt-and-suspenders REST destroy + always
            returned True. The v3 design moves all of that into
            ``destroy_vastai_instance`` (the adapter); this method is
            a single adapter call with no inline destroy logic.

            Returns True iff the adapter reports ``verdict=DESTROYED``.
            For every other outcome, the typed ``DestroyResult`` is
            logged before returning False so operators can diagnose
            why the destroy failed (LEAKED, UNKNOWN, or any refusal).
        """
        result = destroy_vastai_instance(
            instance.instance_id,
            ownership=self.ownership,
            credentials=self.credentials,
        )

        if result.verdict == DestroyVerdict.DESTROYED:
            instance.status = InstanceStatus.DESTROYED
            logger.info("Destroyed instance %s (verified)", instance.instance_id)
            return True

        # Non-DESTROYED: log the typed result before collapsing to False.
        if result.verdict == DestroyVerdict.LEAKED:
            logger.error(
                "Destroy %s LEAKED — manual review required: %s",
                instance.instance_id,
                _describe_destroy_result(result),
            )
        elif result.verdict == DestroyVerdict.UNKNOWN:
            logger.warning(
                "Destroy %s outcome=UNKNOWN: %s",
                instance.instance_id,
                _describe_destroy_result(result),
            )
        elif result.refusal == DestroyRefusal.OWNERSHIP:
            logger.error(
                "Destroy %s refused (ownership check rejected): %s",
                instance.instance_id,
                _describe_destroy_result(result),
            )
        elif result.refusal == DestroyRefusal.NO_CREDENTIALS:
            logger.error(
                "Destroy %s refused (no credentials): %s",
                instance.instance_id,
                _describe_destroy_result(result),
            )
        elif result.refusal == DestroyRefusal.CREDENTIALS_DISABLED:
            logger.error(
                "Destroy %s refused (credentials disabled): %s",
                instance.instance_id,
                _describe_destroy_result(result),
            )
        else:
            logger.error(
                "Destroy %s returned unrecognised result: %s",
                instance.instance_id,
                _describe_destroy_result(result),
            )
        return False


# The v2 module-level helpers (_read_vastai_api_key, _rest_stop,
# _rest_delete_with_retries, _rest_verify_and_redestroy) are DELETED.
# The v3 destroy adapter (``providers/destroy_adapters/vastai.py``)
# owns the equivalent behaviour. This is the v3 doc's "What changes
# vs v2" deletion: providers/vastai.py:_rest_stop,
# _rest_delete_with_retries, _rest_verify_and_redestroy are
# absorbed into the Vast.ai adapter. Folding the v3 doc's step 8
# deletion into this runner refactor commit so the runner doesn't
# carry dead-code helpers.
