"""Cloudflare R2 bucket lifecycle administration.

Lifecycle configuration is bucket-level Cloudflare/R2 storage that
controls automatic deletion of objects by prefix after a retention
period. This module is the *administration* plane for that
configuration — it does NOT add, change, or touch any object data.

Key safety invariants:

- ``R2Sink`` (the data plane) has no bucket-administration side effects.
- Lifecycle rules are explicitly applied through the operator CLI; they
  are never applied implicitly during package install, ``R2Sink``
  construction, batch startup, or worker execution.
- Admin credentials are explicitly supplied by the caller; worker
  credentials are never reused for bucket-policy mutation.
- The module owns exactly ONE managed rule, identified by a deterministic
  rule ID derived from the bucket and canonical prefix. Unrelated
  rules are preserved verbatim across ``apply`` and ``remove``.
- ``apply`` and ``remove`` are fail-closed: a stale source fingerprint,
  an incompatible managed-rule collision, or a post-write read-after-write
  mismatch all abort with a typed exception. No automatic retries.

Excluded by design:

- Storage-class transitions.
- Bucket-wide expiration (root prefix).
- Lifecycle expiration dates (only age-in-days).
- Arbitrary lifecycle JSON pass-through.

Usage::

    from vastai_gpu_runner.storage.r2_lifecycle import (
        R2AdminCredentials,
        R2ExpirationPolicy,
        R2LifecycleManager,
    )

    creds = R2AdminCredentials.from_file("~/.cloud-credentials.r2-admin")
    mgr = R2LifecycleManager(creds)
    plan = mgr.plan_apply(
        R2ExpirationPolicy(bucket="my-bucket", prefix="project/batches", expire_after_days=30),
    )
    if not plan.no_op:
        mgr.apply(plan)  # read-after-write verified
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class R2LifecycleError(Exception):
    """Base class for all R2 lifecycle administration errors."""


class R2LifecycleValidationError(R2LifecycleError, ValueError):
    """Caller-supplied arguments are invalid (prefix, retention, bucket)."""


class R2LifecycleCredentialsError(R2LifecycleError):
    """Required admin credentials are missing or malformed."""


class R2LifecycleAccessDeniedError(R2LifecycleError):
    """The supplied credentials lack permission for the requested operation."""


class R2LifecycleCollisionError(R2LifecycleError):
    """The managed rule ID already exists with an incompatible shape."""


class R2LifecycleStalePlanError(R2LifecycleError):
    """The bucket lifecycle changed between plan and execute."""


class R2LifecycleVerificationError(R2LifecycleError):
    """The post-write read-after-write state did not match the plan."""


class R2LifecycleRuleLimitError(R2LifecycleError):
    """Adding the managed rule would exceed the provider's rule limit."""


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class R2AdminCredentials:
    """Explicit admin credentials for lifecycle mutation.

    Attributes:
        endpoint: R2/S3 endpoint URL.
        access_key_id: R2 access key id.
        secret_access_key: R2 secret access key (never logged).
    """

    endpoint: str
    access_key_id: str
    secret_access_key: str

    def __repr__(self) -> str:
        """Redact the secret access key from the repr."""
        return (
            f"R2AdminCredentials(endpoint={self.endpoint!r}, "
            f"access_key_id={self.access_key_id!r}, "
            f"secret_access_key=<redacted>)"
        )

    @classmethod
    def from_file(
        cls,
        credentials_file: str | Path,
    ) -> R2AdminCredentials:
        """Load admin credentials from a shell-export file.

        Lines must be of the form ``export R2_ADMIN_<key>="value"``.
        Worker-style ``R2_*`` keys are explicitly rejected to enforce
        credential separation — passing the worker's
        ``~/.cloud-credentials`` file would expose object-write
        credentials to bucket-policy mutations, which violates
        least privilege.

        Args:
            credentials_file: Path to the credentials file.

        Raises:
            R2LifecycleCredentialsError: Required variables are missing.
        """
        creds_path = Path(credentials_file).expanduser()
        if not creds_path.exists():
            raise R2LifecycleCredentialsError(
                f"Credentials file not found: {creds_path}",
            )

        env: dict[str, str] = {}
        for line in creds_path.read_text().splitlines():
            line = line.strip()
            if not line.startswith("export "):
                continue
            parts = line[len("export ") :].split("=", 1)
            if len(parts) != 2:
                continue
            env[parts[0]] = parts[1].strip('"').strip("'")

        # Admin credentials are EXPLICITLY separated from worker credentials.
        # Only ``R2_ADMIN_*`` keys are accepted — passing the worker's
        # ``~/.cloud-credentials`` file would expose object-write credentials to
        # bucket-policy mutations, which violates least privilege.
        endpoint = env.get("R2_ADMIN_ENDPOINT", "")
        key_id = env.get("R2_ADMIN_ACCESS_KEY_ID", "")
        secret = env.get("R2_ADMIN_SECRET_ACCESS_KEY", "")

        missing = [
            name
            for name, val in (
                ("R2_ADMIN_ENDPOINT", endpoint),
                ("R2_ADMIN_ACCESS_KEY_ID", key_id),
                ("R2_ADMIN_SECRET_ACCESS_KEY", secret),
            )
            if not val
        ]
        if missing:
            raise R2LifecycleCredentialsError(
                "Missing admin credentials: " + ", ".join(missing),
            )
        return cls(
            endpoint=endpoint,
            access_key_id=key_id,
            secret_access_key=secret,
        )


@dataclass(frozen=True)
class R2ExpirationPolicy:
    """A user-requested expiration policy for one bucket prefix.

    Attributes:
        bucket: R2 bucket name (non-empty).
        prefix: Object key prefix (canonical: no leading ``/``, exactly
            one trailing ``/``, no duplicate separators).
        expire_after_days: Retention period in whole days. Must be an
            ``int >= 1``. No maximum.
    """

    bucket: str
    prefix: str
    expire_after_days: int

    def __post_init__(self) -> None:
        """Validate the policy fields. Raises ``R2LifecycleValidationError``."""
        # pyright flags isinstance() on declared types as unnecessary, but
        # frozen dataclasses can be re-instantiated with wrong-typed
        # values via dataclasses.replace(); keep the runtime guards.
        if not isinstance(self.bucket, str) or not self.bucket.strip():  # pyright: ignore[reportUnnecessaryIsInstance]
            raise R2LifecycleValidationError("bucket must be a non-empty string")
        if not isinstance(self.expire_after_days, int) or isinstance(self.expire_after_days, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise R2LifecycleValidationError(
                "expire_after_days must be an integer (not bool, float, or str)",
            )
        if self.expire_after_days < 1:
            raise R2LifecycleValidationError(
                f"expire_after_days must be >= 1, got {self.expire_after_days}",
            )

    @property
    def canonical_prefix(self) -> str:
        """Return the prefix in canonical form (trailing ``/``, no leading ``/``)."""
        return canonicalise_prefix(self.prefix)


def canonicalise_prefix(raw: str) -> str:
    """Canonicalise a user-supplied prefix.

    Rules:
        - Reject ``None``, empty, and ``/`` (bucket-wide rule).
        - Trim surrounding whitespace.
        - Remove a leading ``/``.
        - Collapse duplicate ``/`` separators.
        - Enforce exactly one trailing ``/``.

    Args:
        raw: Caller-supplied prefix.

    Returns:
        Canonical prefix ending in exactly one ``/``.

    Raises:
        R2LifecycleValidationError: Prefix is empty, root, or otherwise invalid.
    """
    # Public API; callers may pass wrong-typed values by accident.
    if not isinstance(raw, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise R2LifecycleValidationError("prefix must be a string")
    stripped = raw.strip()
    if not stripped:
        raise R2LifecycleValidationError("prefix must not be empty")
    if stripped == "/":
        raise R2LifecycleValidationError(
            "prefix '/' is rejected — bucket-wide rules are not supported",
        )
    # Collapse duplicate separators.
    while "//" in stripped:
        stripped = stripped.replace("//", "/")
    # Strip leading slash (we own the bucket; "/" means root).
    if stripped.startswith("/"):
        stripped = stripped[1:]
    if not stripped:
        raise R2LifecycleValidationError(
            "prefix must contain a non-root segment after canonicalisation",
        )
    # Enforce exactly one trailing separator.
    stripped = stripped.rstrip("/")
    return stripped + "/"


def managed_rule_id(bucket: str, canonical_prefix: str) -> str:
    """Deterministic rule ID derived from bucket + prefix.

    The ID is stable across runs, so the same input always identifies
    the same managed rule. Other administrators' rules are unaffected.

    Args:
        bucket: Bucket name.
        canonical_prefix: Canonical prefix.

    Returns:
        Rule ID of the form ``vastai-gpu-runner-expire-<12-hex>``.
    """
    h = hashlib.sha256(f"{bucket}\x00{canonical_prefix}".encode()).hexdigest()
    return f"vastai-gpu-runner-expire-{h[:12]}"


def _build_expiration_rule(
    rule_id: str,
    canonical_prefix: str,
    expire_after_days: int,
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    """Build an S3 lifecycle expiration rule dict.

    Args:
        rule_id: Rule identifier (must be deterministic).
        canonical_prefix: Canonical prefix (ending in ``/``).
        expire_after_days: Retention in days.
        enabled: Whether the rule is enabled.

    Returns:
        Rule dict suitable for ``put_bucket_lifecycle_configuration``.
    """
    return {
        "ID": rule_id,
        "Status": "Enabled" if enabled else "Disabled",
        "Filter": {"Prefix": canonical_prefix},
        "Expiration": {"Days": expire_after_days},
    }


# ---------------------------------------------------------------------------
# Plan / result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LifecyclePlan:
    """A computed plan describing a future lifecycle mutation.

    Attributes:
        operation: ``"apply"`` or ``"remove"``.
        bucket: Bucket name.
        canonical_prefix: Canonical prefix.
        managed_rule_id: Deterministic ID for the managed rule.
        expire_after_days: Requested retention (apply only).
        before_rules: Tuple of rule dicts as they currently exist.
        after_rules: Tuple of rule dicts after the mutation.
        no_op: True iff the mutation would not change anything.
        source_fingerprint: Stable hash of the source configuration
            (used to detect external edits between plan and execute).
        warnings: Tuple of human-readable warnings (e.g. overlapping
            foreign rules).
    """

    operation: str
    bucket: str
    canonical_prefix: str
    managed_rule_id: str
    expire_after_days: int | None
    before_rules: tuple[dict[str, Any], ...]
    after_rules: tuple[dict[str, Any], ...]
    no_op: bool
    source_fingerprint: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LifecycleResult:
    """The outcome of an executed lifecycle mutation.

    Attributes:
        operation: ``"apply"`` or ``"remove"``.
        bucket: Bucket name.
        canonical_prefix: Canonical prefix.
        managed_rule_id: Deterministic ID for the managed rule.
        verified: True iff read-after-write matched the plan.
        no_op: True iff no PUT was issued (already in desired state).
        rules_count: Number of rules after the mutation.
    """

    operation: str
    bucket: str
    canonical_prefix: str
    managed_rule_id: str
    verified: bool
    no_op: bool
    rules_count: int


# ---------------------------------------------------------------------------
# S3 client factory (separated for easy mocking in tests)
# ---------------------------------------------------------------------------


def build_admin_client(creds: R2AdminCredentials):  # type: ignore[no-untyped-def]
    """Construct a boto3 S3 client configured for R2 administration.

    Args:
        creds: Admin credentials.

    Returns:
        boto3 S3 client.
    """
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=creds.endpoint,
        aws_access_key_id=creds.access_key_id,
        aws_secret_access_key=creds.secret_access_key,
        region_name="auto",
    )


# ---------------------------------------------------------------------------
# Manager — orchestrates inspection / planning / mutation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ReadResult:
    """Internal — read-back state used for verification."""

    rules: tuple[dict[str, Any], ...]
    fingerprint: str


class R2LifecycleManager:
    """Inspect, plan, apply, and remove one managed lifecycle rule.

    The manager owns exactly one rule per (bucket, prefix) pair, identified
    by a deterministic ID. Other rules on the bucket are preserved verbatim
    and surfaced as warnings when they overlap with the managed prefix.

    Args:
        credentials: Explicit admin credentials.
        client: Optional pre-built boto3 S3 client (used by tests).
    """

    def __init__(
        self,
        credentials: R2AdminCredentials,
        *,
        client: Any = None,
    ) -> None:
        """Initialize the lifecycle manager."""
        self.credentials = credentials
        self._client = client if client is not None else build_admin_client(credentials)

    def __repr__(self) -> str:
        """Redact the secret access key from the repr."""
        return f"R2LifecycleManager(credentials={self.credentials!r})"

    # -- Inspection -------------------------------------------------------

    def read_lifecycle(self, bucket: str) -> _ReadResult:
        """Read the current lifecycle configuration from R2.

        Args:
            bucket: Bucket name.

        Returns:
            ``_ReadResult`` containing the current rules and a stable
            fingerprint of the configuration. A bucket with no lifecycle
            configuration returns an empty rule set, not an error.

        Raises:
            R2LifecycleAccessDeniedError: Caller lacks ``s3:GetLifecycleConfiguration``.
            R2LifecycleError: Any other provider error.
        """
        try:
            response = self._client.get_bucket_lifecycle_configuration(Bucket=bucket)
        except Exception as exc:
            self._raise_for_access(exc, op="get_bucket_lifecycle_configuration")
            if _is_no_such_lifecycle_error(exc):
                rules: list[dict[str, Any]] = []
                return _ReadResult(
                    rules=(),
                    fingerprint=fingerprint_rules(rules),
                )
            raise R2LifecycleError(str(exc)) from exc

        rules = _normalize_rules(response.get("Rules", []))
        return _ReadResult(
            rules=tuple(rules),
            fingerprint=fingerprint_rules(rules),
        )

    def inspect_managed_rule(
        self,
        bucket: str,
        prefix: str,
    ) -> dict[str, Any] | None:
        """Return the current managed rule for the given prefix, or None.

        Args:
            bucket: Bucket name.
            prefix: Caller-supplied prefix (will be canonicalised).

        Returns:
            The managed rule dict if it exists, else None.
        """
        canonical_prefix = canonicalise_prefix(prefix)
        rule_id = managed_rule_id(bucket, canonical_prefix)
        read = self.read_lifecycle(bucket)
        for rule in read.rules:
            if rule.get("ID") == rule_id:
                return rule
        return None

    # -- Planning ---------------------------------------------------------

    def plan_apply(self, policy: R2ExpirationPolicy) -> LifecyclePlan:
        """Compute a plan to apply the given expiration policy.

        Args:
            policy: Caller-supplied policy.

        Returns:
            A ``LifecyclePlan`` describing the would-be mutation.
        """
        canonical_prefix = policy.canonical_prefix
        rule_id = managed_rule_id(policy.bucket, canonical_prefix)
        read = self.read_lifecycle(policy.bucket)
        warnings = self._detect_warnings(read.rules, canonical_prefix, rule_id)
        after = _build_after_rules_apply(
            existing=read.rules,
            rule_id=rule_id,
            canonical_prefix=canonical_prefix,
            expire_after_days=policy.expire_after_days,
        )
        no_op = _rules_equal(read.rules, after)
        return LifecyclePlan(
            operation="apply",
            bucket=policy.bucket,
            canonical_prefix=canonical_prefix,
            managed_rule_id=rule_id,
            expire_after_days=policy.expire_after_days,
            before_rules=read.rules,
            after_rules=after,
            no_op=no_op,
            source_fingerprint=read.fingerprint,
            warnings=warnings,
        )

    def plan_remove(self, bucket: str, prefix: str) -> LifecyclePlan:
        """Compute a plan to remove the managed rule for the given prefix.

        Args:
            bucket: Bucket name.
            prefix: Caller-supplied prefix (will be canonicalised).

        Returns:
            A ``LifecyclePlan`` describing the would-be removal.
        """
        canonical_prefix = canonicalise_prefix(prefix)
        rule_id = managed_rule_id(bucket, canonical_prefix)
        read = self.read_lifecycle(bucket)
        after = _build_after_rules_remove(read.rules, rule_id)
        no_op = _rules_equal(read.rules, after)
        return LifecyclePlan(
            operation="remove",
            bucket=bucket,
            canonical_prefix=canonical_prefix,
            managed_rule_id=rule_id,
            expire_after_days=None,
            before_rules=read.rules,
            after_rules=after,
            no_op=no_op,
            source_fingerprint=read.fingerprint,
            warnings=(),
        )

    # -- Mutation ---------------------------------------------------------

    def apply(self, plan: LifecyclePlan) -> LifecycleResult:
        """Apply the given plan.

        Performs:
            1. Re-reads the bucket lifecycle.
            2. Aborts with ``R2LifecycleStalePlanError`` if the source
               fingerprint changed since the plan was built (optimistic:
               an external writer can still race the subsequent PUT).
            3. Skips the PUT if the plan is a no-op.
            4. PUTs the new configuration.
            5. Re-reads and verifies the post-write state matches the
               plan's full expected ruleset (managed + preserved foreign).

        Args:
            plan: A plan produced by ``plan_apply``.

        Returns:
            ``LifecycleResult`` describing the outcome.

        Raises:
            R2LifecycleStalePlanError: Bucket changed between plan and apply.
            R2LifecycleCollisionError: An incompatible managed rule exists.
            R2LifecycleRuleLimitError: Provider rejected the rule count.
            R2LifecycleAccessDeniedError: Caller lacks PUT permission.
            R2LifecycleVerificationError: Read-after-write mismatch.
        """
        self._assert_plan_fresh(plan)
        _assert_no_incompatible_managed(
            plan.before_rules,
            plan.managed_rule_id,
            expected_prefix=plan.canonical_prefix,
        )

        if plan.no_op:
            logger.info(
                "R2 lifecycle already in desired state for %s/%s — no-op",
                plan.bucket,
                plan.canonical_prefix,
            )
            return LifecycleResult(
                operation=plan.operation,
                bucket=plan.bucket,
                canonical_prefix=plan.canonical_prefix,
                managed_rule_id=plan.managed_rule_id,
                verified=True,
                no_op=True,
                rules_count=len(plan.after_rules),
            )

        self._put_lifecycle(plan.bucket, list(plan.after_rules))
        verify = self.read_lifecycle(plan.bucket)
        if not _ruleset_matches_plan(verify.rules, plan):
            raise R2LifecycleVerificationError(
                f"Read-after-write mismatch for {plan.bucket}/{plan.canonical_prefix}",
            )
        return LifecycleResult(
            operation=plan.operation,
            bucket=plan.bucket,
            canonical_prefix=plan.canonical_prefix,
            managed_rule_id=plan.managed_rule_id,
            verified=True,
            no_op=False,
            rules_count=len(verify.rules),
        )

    def remove(self, plan: LifecyclePlan) -> LifecycleResult:
        """Remove the managed rule described by the plan.

        If the post-removal rule set is empty, this method calls
        ``delete_bucket_lifecycle`` instead of PUTting an empty rule
        collection — Cloudflare / S3 require at least one rule, and
        DELETE is the documented way to clear the configuration.

        Args:
            plan: A plan produced by ``plan_remove``.

        Returns:
            ``LifecycleResult`` describing the outcome.
        """
        self._assert_plan_fresh(plan)
        _assert_no_incompatible_managed(
            plan.before_rules,
            plan.managed_rule_id,
            expected_prefix=plan.canonical_prefix,
        )
        if plan.no_op:
            logger.info(
                "R2 lifecycle already without managed rule for %s/%s — no-op",
                plan.bucket,
                plan.canonical_prefix,
            )
            return LifecycleResult(
                operation=plan.operation,
                bucket=plan.bucket,
                canonical_prefix=plan.canonical_prefix,
                managed_rule_id=plan.managed_rule_id,
                verified=True,
                no_op=True,
                rules_count=len(plan.after_rules),
            )
        if plan.after_rules:
            self._put_lifecycle(plan.bucket, list(plan.after_rules))
        else:
            self._delete_lifecycle(plan.bucket)
        verify = self.read_lifecycle(plan.bucket)
        if not _ruleset_matches_plan(verify.rules, plan):
            raise R2LifecycleVerificationError(
                f"Read-after-write mismatch for {plan.bucket}/{plan.canonical_prefix}",
            )
        return LifecycleResult(
            operation=plan.operation,
            bucket=plan.bucket,
            canonical_prefix=plan.canonical_prefix,
            managed_rule_id=plan.managed_rule_id,
            verified=True,
            no_op=False,
            rules_count=len(verify.rules),
        )

    # -- Internals --------------------------------------------------------

    def _assert_plan_fresh(self, plan: LifecyclePlan) -> None:
        """Abort with ``R2LifecycleStalePlanError`` if the bucket changed.

        This is OPTIMISTIC stale-plan detection: a fingerprint mismatch
        detected here means the bucket changed between plan and execute,
        but an external writer can still race the subsequent PUT. The
        check narrows the window for lost updates; it does not provide
        a strict compare-and-swap guarantee.
        """
        read = self.read_lifecycle(plan.bucket)
        if read.fingerprint != plan.source_fingerprint:
            raise R2LifecycleStalePlanError(
                f"Bucket {plan.bucket} lifecycle changed since plan was built; "
                "re-plan and re-apply",
            )

    def _put_lifecycle(self, bucket: str, rules: list[dict[str, Any]]) -> None:
        """PUT the lifecycle configuration; translate provider errors.

        The boto3 / S3 / Cloudflare R2 API requires the rules to be
        nested under ``LifecycleConfiguration``. Top-level ``Rules=``
        raises ``ParamValidationError`` before a request is sent.
        """
        try:
            self._client.put_bucket_lifecycle_configuration(
                Bucket=bucket,
                LifecycleConfiguration={"Rules": rules},
            )
        except Exception as exc:
            self._raise_for_access(exc, op="put_bucket_lifecycle_configuration")
            if _is_rule_limit_error(exc):
                raise R2LifecycleRuleLimitError(
                    f"Provider rejected rule count for {bucket}",
                ) from exc
            raise R2LifecycleError(str(exc)) from exc

    def _delete_lifecycle(self, bucket: str) -> None:
        """DELETE the bucket lifecycle configuration; translate provider errors.

        Used when removing the last remaining rule, since R2 / S3
        require at least one rule per configuration and reject an
        empty ``LifecycleConfiguration``.
        """
        try:
            self._client.delete_bucket_lifecycle(Bucket=bucket)
        except Exception as exc:
            self._raise_for_access(exc, op="delete_bucket_lifecycle")
            raise R2LifecycleError(str(exc)) from exc

    def _raise_for_access(self, exc: BaseException, *, op: str) -> None:
        """Re-raise ``ClientError`` ``AccessDenied`` as a typed exception."""
        code = (
            getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if hasattr(exc, "response")
            else ""
        )
        if code in {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}:
            raise R2LifecycleAccessDeniedError(
                f"Access denied for {op}: {code}",
            ) from exc

    def _detect_warnings(
        self,
        existing_rules: tuple[dict[str, Any], ...],
        canonical_prefix: str,
        rule_id: str,
    ) -> tuple[str, ...]:
        """Surface overlapping foreign rules as warnings."""
        prefix_no_slash = canonical_prefix.rstrip("/")
        return tuple(
            msg
            for msg in (
                _format_foreign_warning(rule, rule_id, prefix_no_slash) for rule in existing_rules
            )
            if msg is not None
        )


def _format_foreign_warning(
    rule: dict[str, Any],
    managed_rule_id: str,
    prefix_no_slash: str,
) -> str | None:
    """Return a warning string for a foreign rule that overlaps the prefix."""
    rid = rule.get("ID", "")
    if rid == managed_rule_id or rid.startswith("vastai-gpu-runner-"):
        return None
    rule_prefix = _rule_filter_prefix(rule)
    if not rule_prefix:
        return None
    if not _prefixes_overlap(rule_prefix, prefix_no_slash):
        return None
    return (
        f"Overlapping foreign rule {rid!r} (prefix={rule_prefix!r}) "
        f"may expire objects earlier than the managed rule"
    )


def _prefixes_overlap(prefix_a: str, prefix_b: str) -> bool:
    """Return True iff two prefixes share a key."""
    return prefix_a == prefix_b or prefix_b.startswith(prefix_a) or prefix_a.startswith(prefix_b)


# ---------------------------------------------------------------------------
# Pure helpers — extracted to keep the manager's complexity under the ceiling
# ---------------------------------------------------------------------------


def fingerprint_rules(rules: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
    """Stable fingerprint of a rule collection.

    The fingerprint is order-sensitive: two rule sets that differ only
    in ordering produce different fingerprints. This is intentional —
    the order in which rules appear in the bucket affects evaluation
    semantics on R2/S3.

    Args:
        rules: Iterable of rule dicts.

    Returns:
        16-character hex digest.
    """
    payload = json.dumps(list(rules), sort_keys=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalize_rules(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise rule dicts while preserving unknown fields from foreign rules.

    The lifecycle configuration contains a fixed schema for our managed
    rules. Foreign rules may contain fields we don't model (storage-class
    transitions, abort-incomplete-multipart, etc.). We pass those through
    unchanged so an ``apply`` or ``remove`` does not delete them.
    """
    return [dict(rule) for rule in raw]


def _build_after_rules_apply(
    *,
    existing: tuple[dict[str, Any], ...],
    rule_id: str,
    canonical_prefix: str,
    expire_after_days: int,
) -> tuple[dict[str, Any], ...]:
    """Return the rule set with the managed rule applied."""
    new_rule = _build_expiration_rule(
        rule_id=rule_id,
        canonical_prefix=canonical_prefix,
        expire_after_days=expire_after_days,
    )
    kept = [r for r in existing if r.get("ID") != rule_id]
    kept.append(new_rule)
    return tuple(kept)


def _build_after_rules_remove(
    existing: tuple[dict[str, Any], ...],
    rule_id: str,
) -> tuple[dict[str, Any], ...]:
    """Return the rule set with the managed rule removed."""
    return tuple(r for r in existing if r.get("ID") != rule_id)


def _rules_equal(
    a: tuple[dict[str, Any], ...],
    b: tuple[dict[str, Any], ...],
) -> bool:
    """Return True iff two rule tuples are deep-equal."""
    if len(a) != len(b):
        return False
    return all(
        json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)
        for x, y in zip(a, b, strict=True)
    )


def _assert_no_incompatible_managed(
    rules: tuple[dict[str, Any], ...],
    rule_id: str,
    expected_prefix: str | None = None,
) -> None:
    """Abort if a rule with the managed ID exists but disagrees on prefix/action.

    Args:
        rules: The current rule collection.
        rule_id: The deterministic managed-rule ID.
        expected_prefix: When provided, also reject a same-ID rule whose
            Filter.Prefix disagrees with the expected canonical prefix.
            This guards against an external operator who created the
            rule under our naming convention for a different prefix.
    """
    for rule in rules:
        if rule.get("ID") != rule_id:
            continue
        _check_managed_rule_shape(rule, rule_id)
        if expected_prefix is not None:
            rule_prefix = _rule_filter_prefix(rule)
            if rule_prefix != expected_prefix:
                raise R2LifecycleCollisionError(
                    f"Managed rule {rule_id} is bound to prefix {rule_prefix!r}, "
                    f"not the expected {expected_prefix!r}",
                )


def _check_managed_rule_shape(rule: dict[str, Any], rule_id: str) -> None:
    """Validate that a managed-rule-shaped dict has Expiration.Days."""
    expiration = rule.get("Expiration", {})
    if not isinstance(expiration, dict) or "Days" not in expiration:
        raise R2LifecycleCollisionError(
            f"Managed rule {rule_id} has incompatible shape (no Expiration.Days)",
        )


def _is_rule_limit_error(exc: BaseException) -> bool:
    """Heuristic: detect ``TooManyRules`` / R2 rule-count rejections."""
    code = (
        getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if hasattr(exc, "response")
        else ""
    )
    if not code:
        return False
    code_lower = str(code).lower()
    return "toomanyrules" in code_lower or ("rule" in code_lower and "limit" in code_lower)


def _is_no_such_lifecycle_error(exc: BaseException) -> bool:
    """Heuristic: detect ``NoSuchLifecycleConfiguration`` from a fresh bucket."""
    code = (
        getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if hasattr(exc, "response")
        else ""
    )
    if not code:
        return False
    return str(code).strip().lower() == "nosuchlifecycleconfiguration"


def _ruleset_matches_plan(
    rules: tuple[dict[str, Any], ...],
    plan: LifecyclePlan,
) -> bool:
    """Return True iff *rules* equal the plan's expected full post-state.

    Compares the complete normalised rule collection — both the
    managed rule's shape AND the preservation of every unrelated
    rule on the bucket — against ``plan.after_rules``.
    """
    return _rules_equal(rules, plan.after_rules)


def _rule_filter_prefix(rule: dict[str, Any]) -> str:
    """Return the Filter.Prefix field of an S3 lifecycle rule dict."""
    rule_filter = rule.get("Filter", {})
    if not isinstance(rule_filter, dict):
        return ""
    return rule_filter.get("Prefix", "")


# Re-export the bucket-level rule for external type-checking helpers.
__all__ = [
    "LifecyclePlan",
    "LifecycleResult",
    "R2AdminCredentials",
    "R2ExpirationPolicy",
    "R2LifecycleAccessDeniedError",
    "R2LifecycleCollisionError",
    "R2LifecycleCredentialsError",
    "R2LifecycleError",
    "R2LifecycleManager",
    "R2LifecycleRuleLimitError",
    "R2LifecycleStalePlanError",
    "R2LifecycleValidationError",
    "R2LifecycleVerificationError",
    "build_admin_client",
    "canonicalise_prefix",
    "fingerprint_rules",
    "managed_rule_id",
]
