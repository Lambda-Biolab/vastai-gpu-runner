"""Provider-agnostic cleanup policy + DTOs (v4).

The core module imports nothing from ``providers/`` and stores no
provider-specific ownership policy. Provider-owned factories
capture their policy data inside the ``list_instances_fn`` and
``destroy_fn`` callbacks. See
``docs/architecture-v4-cleanup-policy.md`` (migration step 2).
"""

from __future__ import annotations

import ipaddress
import logging
import re
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from enum import Enum

from vastai_gpu_runner.types import Provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _repository helper — tag-insensitive image reference normalisation
# ---------------------------------------------------------------------------


_DIGEST_RE = re.compile(r"[a-z0-9]+(?:[+._-][a-z0-9]+)*:[A-Za-z0-9=_-]+")
_REGISTERED_DIGEST_LENGTHS = {
    "sha256": 64,
    "sha512": 128,
    "blake3": 64,
}


def _valid_digest(digest: str) -> bool:
    match = _DIGEST_RE.fullmatch(digest)
    if match is None:
        return False
    algorithm, encoded = digest.split(":", 1)
    required_length = _REGISTERED_DIGEST_LENGTHS.get(algorithm)
    if required_length is None:
        return True
    return (
        len(encoded) == required_length
        and encoded == encoded.lower()
        and re.fullmatch(r"[0-9a-f]+", encoded) is not None
    )


_TAG_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
_PATH_COMPONENT_RE = re.compile(r"[a-z0-9]+(?:(?:[._]|__|[-]+)[a-z0-9]+)*")
_REGISTRY_LABEL_RE = re.compile(r"[A-Za-z0-9](?:(?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)?")


# ---------------------------------------------------------------------------
# Enums: CleanupVerdict, CleanupRefusal
# ---------------------------------------------------------------------------


class CleanupVerdict(Enum):
    """Outcome verdicts returned by ``policy.destroy``.

    ``DESTROYED`` is confirmed destruction. ``ALREADY_GONE`` is a
    successful cleanup end-state proved by the ownership verifier;
    no destruction was attempted. The remaining outcomes are
    observable unresolved states that the orchestrator logs distinctly.

    ``CLI_ATTEMPTED`` is a v4 verdict produced by the factory's
    CLI fallback path — it is NOT a v3 protocol verdict.
    ``ALREADY_GONE`` is a v4 verdict produced by the CLI
    fallback path when the ownership verifier proves absence
    from a fully well-formed response — the fallback short-
    circuits and does NOT invoke CLI destruction (which would
    otherwise invent a spurious ``UNKNOWN`` from a 'not found'
    CLI exit code). The orchestrator does NOT count
    ``ALREADY_GONE`` toward ``killed`` (no destroy happened);
    it logs at ``INFO`` because the desired end-state was
    already achieved by the absence.
    """

    DESTROYED = "destroyed"
    CLI_ATTEMPTED = "cli_attempted"  # CLI fallback ran; destruction not confirmed
    ALREADY_GONE = "already_gone"  # CLI verifier proved absence; no destroy
    LEAKED = "leaked"  # Protocol ran but instance was resurrected
    UNKNOWN = "unknown"  # Protocol did not report a clear outcome


class CleanupRefusal(Enum):
    """Pre-protocol refusal reasons returned by ``policy.destroy``.

    These are policy-level decisions — the destroy protocol is
    never entered. The orchestrator logs the refusal and skips
    the candidate.
    """

    OWNERSHIP = "ownership"  # image/template not owned by this policy
    NO_CREDENTIALS = "no_credentials"  # no API key configured; CLI fallback attempted
    CREDENTIALS_DISABLED = "credentials_disabled"  # VASTAI_API_KEY="" — CLI fallback forbidden
    INELIGIBLE_STATE = "ineligible_state"  # candidate.state is terminal (destroyed) or malformed
    PROVIDER_MISMATCH = "provider_mismatch"  # candidate.provider != policy.provider


# Successful cleanup end-states have empty ``error``. Unresolved verdicts
# and refusals always have non-empty ``error``. The frozenset captures
# this exactly so the ``__post_init__`` invariant check is O(1).
_SUCCESSFUL_END_STATES: frozenset[CleanupVerdict] = frozenset(
    {
        CleanupVerdict.DESTROYED,
        CleanupVerdict.ALREADY_GONE,
    }
)


def _valid_registry_port(raw_port: str) -> bool:
    return raw_port.isdigit() and 1 <= int(raw_port) <= 65_535


def _valid_ipv6_registry(registry: str) -> bool:
    """Validate an IPv6 registry of the form ``[addr]`` or ``[addr]:port``."""
    closing = registry.find("]")
    if closing < 0:
        return False
    try:
        ipaddress.IPv6Address(registry[1:closing])
    except ValueError:
        return False
    suffix = registry[closing + 1 :]
    return not suffix or (suffix.startswith(":") and _valid_registry_port(suffix[1:]))


def _valid_dns_registry(registry: str) -> bool:
    """Validate a DNS registry host with optional port (or ``localhost``)."""
    if ":" in registry:
        if registry.count(":") > 1:
            return False
        host, port = registry.rsplit(":", 1)
        if not _valid_registry_port(port):
            return False
    else:
        host = registry
    if host == "localhost":
        return True
    if not host or len(host) > 253:
        return False
    return all(_REGISTRY_LABEL_RE.fullmatch(label) is not None for label in host.split("."))


def _valid_registry(registry: str) -> bool:
    """Validate a DNS/IPv4/IPv6 registry host and optional port."""
    if registry.startswith("["):
        return _valid_ipv6_registry(registry)
    return _valid_dns_registry(registry)


def _valid_repository(repository: str) -> bool:
    """Validate registry and lower-case repository path components."""
    parts = repository.split("/")
    if any(not part for part in parts):
        return False
    first = parts[0]
    has_registry = len(parts) > 1 and (
        first == "localhost" or first.startswith("[") or "." in first or ":" in first
    )
    path_parts = parts
    if has_registry:
        if not _valid_registry(first):
            return False
        path_parts = parts[1:]
    return bool(path_parts) and all(
        _PATH_COMPONENT_RE.fullmatch(part) is not None for part in path_parts
    )


def _strip_digest(ref: str) -> str | None:
    """Strip a valid digest suffix from ``ref``; return ``None`` on bad input."""
    if "@" not in ref:
        return ref
    without_digest, digest = ref.split("@", 1)
    if not without_digest or not _valid_digest(digest):
        return None
    return without_digest


def _strip_tag(without_digest: str) -> str | None:
    """Strip a valid tag suffix from ``without_digest``; return ``None`` on bad tag."""
    last_slash = without_digest.rfind("/")
    last_colon = without_digest.rfind(":")
    if last_colon <= last_slash:
        return without_digest
    tag = without_digest[last_colon + 1 :]
    if _TAG_RE.fullmatch(tag) is None:
        return None
    return without_digest[:last_colon]


def _repository(ref: str) -> str:
    """Return the tag-insensitive repository from a valid image reference.

    Validates the complete Docker/OCI name before stripping its tag or
    digest. Returns ``""`` for malformed references, including invalid
    registry hosts/ports, repository components, tags, or digests.
    """
    ref = ref.strip()
    if not ref or any(char.isspace() for char in ref) or ref.count("@") > 1:
        return ""

    without_digest = _strip_digest(ref)
    if without_digest is None:
        return ""

    repository = _strip_tag(without_digest)
    if repository is None:
        return ""

    if not _valid_repository(repository):
        return ""
    return repository


# ---------------------------------------------------------------------------
# OwnershipPolicy
# ---------------------------------------------------------------------------


class OwnershipVerification(Enum):
    """Result of ``verify_instance_ownership`` (the CLI ownership check).

    Tagged union — replaces v2's conflated ``bool`` (which conflated
    "owned and present" with "absent and therefore safe"). Used by the
    CLI fallback path in ``build_vastai_cleanup_policy._cli_fallback``
    to dispatch:

    - ``OWNED``    → instance exists, image matches the policy → CLI
                     destroy may proceed (returns ``CLI_ATTEMPTED``).
    - ``ABSENT``   → instance is gone from the API; the verifier has
                     proved absence from a well-formed response. The
                     fallback can translate this directly into the
                     configured "already gone" verdict (caller's
                     choice — not the verifier's).
    - ``REFUSED``  → instance exists but image does not match, or the
                     API response was malformed in a way that prevents
                     proving absence (non-dict record, missing / null /
                     empty / whitespace-only ``id``, non-string/null
                     ``image_uuid``, non-list response,
                     unexpected exception). The verifier refuses the
                     destroy to the caller.
    - ``DISABLED`` → ownership checking is disabled (no policy).
                     Bypasses the API call entirely.
    """

    OWNED = "owned"
    ABSENT = "absent"
    REFUSED = "refused"
    DISABLED = "disabled"


def normalize_instance_id(raw_id: object) -> str | None:
    """Canonicalize a Vast.ai instance ID for comparison.

    Shared normalizer used by ``list_vastai_instances`` and
    ``verify_instance_ownership`` so the two agree on what counts as
    a "valid ID" and how to compare.

    Returns:
        - ``None`` for inputs that are not valid IDs (``None``,
          ``bool``, anything that is not ``str`` or ``int``, empty
          string, blank whitespace-only string).
        - The stripped string for valid inputs.

    Rationale:

    - Vast.ai sometimes returns numeric IDs (JSON ``int``). We
      stringify them. Booleans are explicitly rejected: ``True``
      stringifies to ``"True"`` which is not the intended ID and
      would silently shadow a real one.
    - Whitespace-only IDs are rejected (after ``strip()``, empty).
      Padded IDs (``" 123 "``) are accepted and canonicalised to
      ``"123"``.
    - The same rule is applied uniformly by both enumeration and
      verification; padded input on the caller side and padded value
      on the API side both reduce to the canonical form.
    """
    if isinstance(raw_id, bool) or raw_id is None:
        return None
    if not isinstance(raw_id, (str, int)):
        return None
    canonical = str(raw_id).strip()
    if not canonical:
        return None
    return canonical


@dataclass(frozen=True)
class OwnershipPolicy:
    """Shared ownership semantics — runner and cleanup adapter both call this.

    The v3 design requires image matching to be exact reference OR
    tag-insensitive repository equality. This dataclass encodes that
    contract in one place. The runner's `destroy_instance` ownership
    guard and the cleanup policy's `destroy` ownership refusal both
    consume the same `matches()` method.

    Args:
        owned_images: Set of Docker image references owned by this
            project. ``None`` explicitly disables the ownership
            check (every image is considered owned). An empty set
            is distinct from ``None`` — it means the project owns
            no images and every ownership check fails (fail-closed).
        _normalised: Internal cache of normalised repositories.
            Precomputed in ``__post_init__`` so ``matches()`` is O(1).
            Excluded from equality, hashing, and repr.
    """

    owned_images: AbstractSet[str] | None = None
    _normalised: frozenset[str] | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self) -> None:
        """Normalise owned_images into the _normalised cache and freeze owned_images."""
        if self.owned_images is None:
            object.__setattr__(self, "_normalised", None)
            return
        frozen = frozenset(self.owned_images)
        object.__setattr__(self, "owned_images", frozen)
        object.__setattr__(
            self,
            "_normalised",
            frozenset(repo for o in frozen if (repo := _repository(o))),
        )

    def matches(self, image_ref: str) -> bool:
        """Return True if ``image_ref`` is owned by this policy.

        Returns True unconditionally when ``owned_images is None``
        (ownership check disabled). Otherwise returns True iff
        the tag-insensitive repository of ``image_ref`` matches
        the tag-insensitive repository of any entry in
        ``owned_images``.

        The local ``normalised`` variable narrows the
        ``_normalised`` type so Pyright strict mode can infer the
        cross-field invariant (when ``owned_images is not None``,
        ``_normalised`` is a frozenset, not None).
        """
        normalised = self._normalised
        if self.owned_images is None:
            return True
        if normalised is None:
            return False
        if not image_ref:
            return False
        repo = _repository(image_ref)
        if not repo:
            return False
        return repo in normalised


# ---------------------------------------------------------------------------
# Dataclasses: InstanceCandidate, CleanupResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstanceCandidate:
    """Read-only snapshot of one instance returned by ``list_*_instances``.

    Frozen so the policy can hold it without aliasing surprises.
    Fields are the union of what every provider can feasibly
    expose; providers that do not have a field (e.g. RunPod has
    no ``image_uuid``) leave it empty.

    Invariant: ``instance_id`` must be non-empty and pre-stripped
    (an explicit JSON ``null`` is the caller's responsibility to
    filter before constructing the candidate).
    """

    provider: Provider
    instance_id: str
    label: str
    state: str
    image_uuid: str = ""
    ownership_key: str = ""
    gpu_model: str = ""
    cost_per_hour: float = 0.0
    started_at: float = 0.0

    def __post_init__(self) -> None:
        """Validate provider, instance_id, scalar string fields, and numeric fields."""
        # Defensive runtime checks against malformed construction. The
        # dataclass annotations narrow the types for pyright, but
        # ``InstanceCandidate`` is a public DTO that callers may pass
        # bad data through — these checks fail-closed.
        if not isinstance(self.provider, Provider):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("InstanceCandidate.provider must be a Provider")
        if (
            not isinstance(self.instance_id, str)  # pyright: ignore[reportUnnecessaryIsInstance]
            or not self.instance_id
            or self.instance_id != self.instance_id.strip()
        ):
            raise ValueError("InstanceCandidate.instance_id must be non-empty and pre-stripped")
        for field_name in (
            "label",
            "state",
            "image_uuid",
            "ownership_key",
            "gpu_model",
        ):
            if not isinstance(getattr(self, field_name), str):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ValueError(f"InstanceCandidate.{field_name} must be a string")
        for field_name in ("cost_per_hour", "started_at"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))  # pyright: ignore[reportUnnecessaryIsInstance]
                or value != value  # NaN
                or value in (float("inf"), float("-inf"))
                or value < 0
            ):
                raise ValueError(
                    f"InstanceCandidate.{field_name} must be a non-negative finite real number"
                )


@dataclass(frozen=True)
class CleanupResult:
    """Outcome of ``policy.destroy``.

    Exactly one of ``verdict`` (CleanupVerdict) or ``refusal``
    (CleanupRefusal) is set — never both, never neither. ``error``
    is empty for the two successful cleanup end-states
    (``DESTROYED`` and ``ALREADY_GONE``) and non-empty for every
    unresolved verdict or refusal.

    The cleanup policy wraps the v3 ``DestroyResult`` (which has
    ``verdict: DestroyVerdict`` + ``refusal: DestroyRefusal``)
    in this tighter shape for the orchestrator.
    """

    verdict: CleanupVerdict | None = None
    refusal: CleanupRefusal | None = None
    error: str = ""

    def _check_field_types(self) -> None:
        """Defensive runtime type checks for verdict/refusal/error."""
        # Defensive runtime type checks — the dataclass annotation narrows
        # the types for pyright, but external constructors may pass
        # malformed data through (the orchestrator catches every error).
        if self.verdict is not None and not isinstance(self.verdict, CleanupVerdict):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("CleanupResult.verdict must be a CleanupVerdict or None")
        if self.refusal is not None and not isinstance(self.refusal, CleanupRefusal):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("CleanupResult.refusal must be a CleanupRefusal or None")
        if not isinstance(self.error, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("CleanupResult.error must be a string")

    def _check_verdict_refusal_exclusive(self) -> None:
        """Exactly one of verdict or refusal must be set."""
        if (self.verdict is None) == (self.refusal is None):
            raise ValueError("CleanupResult: exactly one of verdict or refusal must be set")

    def _check_per_state_error_invariant(self) -> None:
        """Successful end-states have empty error; everything else has non-empty."""
        if self.verdict in _SUCCESSFUL_END_STATES:
            if self.error:
                raise ValueError("CleanupResult successful end-states must have empty error")
            return
        if not self.error:
            # Pyright can't narrow from the prior `(self.verdict is None)
            # != (self.refusal is None)` exclusivity check, but the
            # invariant is established above.
            ident = (
                self.verdict.value  # pyright: ignore[reportOptionalMemberAccess]
                if self.verdict is not None
                else self.refusal.value  # pyright: ignore[reportOptionalMemberAccess]
            )
            raise ValueError(f"CleanupResult: {ident} must have non-empty error")

    def __post_init__(self) -> None:
        """Validate verdict/refusal exclusivity, type-correctness, and per-state error invariant."""
        self._check_field_types()
        self._check_verdict_refusal_exclusive()
        self._check_per_state_error_invariant()


# ---------------------------------------------------------------------------
# ProviderCleanupPolicy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ProviderCleanupPolicy:
    """Per-provider cleanup contract.

    The core module is provider-agnostic: it imports nothing from
    ``providers/`` and stores no provider-specific ownership policy.
    Provider-owned factories capture their policy data inside the
    ``list_instances_fn`` and ``destroy_fn`` callbacks.
    """

    provider: Provider
    list_instances_fn: Callable[[], list[InstanceCandidate]] = field(repr=False)
    destroy_fn: Callable[[InstanceCandidate], CleanupResult] = field(repr=False)

    def list_instances(self) -> list[InstanceCandidate]:
        """Read-only enumeration of provider instances; never raises."""
        try:
            result = self.list_instances_fn()
        except Exception as exc:
            logger.error("Cleanup policy: list_instances raised: %s", exc)
            return []
        if not isinstance(result, list):  # pyright: ignore[reportUnnecessaryIsInstance]
            logger.error(
                "Cleanup policy: list_instances returned invalid type %s",
                type(result).__name__,
            )
            return []
        if any(not isinstance(candidate, InstanceCandidate) for candidate in result):  # pyright: ignore[reportUnnecessaryIsInstance]
            logger.error("Cleanup policy: list_instances returned a non-InstanceCandidate element")
            return []
        return result

    def destroy(self, candidate: InstanceCandidate) -> CleanupResult:
        """Run the per-provider destroy on one candidate; never raises.

        The protected boundary validates the candidate and its provider,
        delegates to ``destroy_fn``, and validates the callback result.
        Invalid inputs or callback failures become typed outcomes rather
        than escaping into the orchestrator.
        """
        candidate_id = "<invalid>"
        try:
            if not isinstance(candidate, InstanceCandidate):  # pyright: ignore[reportUnnecessaryIsInstance]
                return CleanupResult(
                    verdict=CleanupVerdict.UNKNOWN,
                    error=(
                        f"cleanup policy received invalid candidate type {type(candidate).__name__}"
                    ),
                )
            candidate_id = candidate.instance_id
            if candidate.provider != self.provider:
                return CleanupResult(
                    refusal=CleanupRefusal.PROVIDER_MISMATCH,
                    error=(
                        f"candidate {candidate.instance_id!r} belongs to provider "
                        f"{candidate.provider!r}; cleanup policy expects {self.provider!r}"
                    ),
                )
            result = self.destroy_fn(candidate)
            if not isinstance(result, CleanupResult):  # pyright: ignore[reportUnnecessaryIsInstance]
                logger.error(
                    "Cleanup: destroy_fn for %s returned invalid result "
                    "type %s — substituting UNKNOWN",
                    candidate_id,
                    type(result).__name__,
                )
                return CleanupResult(
                    verdict=CleanupVerdict.UNKNOWN,
                    error=(f"destroy_fn returned invalid result type {type(result).__name__}"),
                )
            return result
        except Exception as exc:
            logger.error(
                "Cleanup: destroy_fn boundary failed for %s: %s",
                candidate_id,
                exc,
            )
            return CleanupResult(
                verdict=CleanupVerdict.UNKNOWN,
                error=f"{type(exc).__name__}: {exc}",
            )
