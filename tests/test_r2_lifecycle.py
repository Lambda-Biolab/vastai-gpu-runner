# pyright: reportPrivateUsage=warning, reportMissingParameterType=warning
"""Tests for ``vastai_gpu_runner.storage.r2_lifecycle``.

All tests run against a mock boto3 S3 client — no live R2 calls. The
manager accepts an injected client so the planning / mutation logic
can be exercised without network access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vastai_gpu_runner.storage.r2_lifecycle import (
    LifecyclePlan,
    LifecycleResult,
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
    _assert_no_incompatible_managed,
    _build_after_rules_apply,
    _build_after_rules_remove,
    _is_rule_limit_error,
    _normalize_rules,
    _rules_equal,
    build_admin_client,
    canonicalise_prefix,
    fingerprint_rules,
    managed_rule_id,
)

# ---------------------------------------------------------------------------
# Mock S3 client
# ---------------------------------------------------------------------------


class _RuleLimitError(Exception):
    """Stand-in for ``botocore.exceptions.ClientError`` on rule-limit."""


class _AccessDeniedError(Exception):
    """Stand-in for an access-denied ClientError."""


def _client_error(code: str, message: str = "") -> Exception:
    """Build a minimal ClientError-shaped exception for tests."""

    class _ClientError(Exception):
        pass

    err = _ClientError(message or code)
    err.response = {"Error": {"Code": code, "Message": message or code}}  # type: ignore[attr-defined]
    return err


class FakeS3:
    """Minimal S3 client that records put/get/delete calls and tracks state."""

    def __init__(self, initial_rules: list[dict[str, Any]] | None = None) -> None:
        self.rules: list[dict[str, Any]] = list(initial_rules or [])
        self.put_calls: list[list[dict[str, Any]]] = []
        self.delete_calls = 0
        self.get_calls = 0
        self.fail_get: Exception | None = None
        self.fail_put: Exception | None = None
        self.fail_delete: Exception | None = None

    def get_bucket_lifecycle_configuration(self, *, Bucket: str) -> dict[str, Any]:  # noqa: N803
        self.get_calls += 1
        if self.fail_get is not None:
            raise self.fail_get
        return {"Rules": list(self.rules)}

    def put_bucket_lifecycle_configuration(
        self,
        *,
        Bucket: str,  # noqa: N803
        LifecycleConfiguration: dict[str, Any],  # noqa: N803
    ) -> dict[str, Any]:
        if self.fail_put is not None:
            raise self.fail_put
        rules = LifecycleConfiguration.get("Rules", [])
        self.put_calls.append(list(rules))
        self.rules = list(rules)
        return {}

    def delete_bucket_lifecycle(self, *, Bucket: str) -> dict[str, Any]:  # noqa: N803
        if self.fail_delete is not None:
            raise self.fail_delete
        self.delete_calls += 1
        self.rules = []
        return {}


# ---------------------------------------------------------------------------
# canonicalise_prefix / managed_rule_id
# ---------------------------------------------------------------------------


class TestCanonicalisePrefix:
    def test_trims_whitespace_and_adds_trailing_slash(self) -> None:
        assert canonicalise_prefix("  project/batches  ") == "project/batches/"

    def test_strips_leading_slash(self) -> None:
        assert canonicalise_prefix("/project/batches/") == "project/batches/"

    def test_collapses_duplicate_separators(self) -> None:
        assert canonicalise_prefix("a//b///c/") == "a/b/c/"

    def test_rejects_empty(self) -> None:
        with pytest.raises(R2LifecycleValidationError):
            canonicalise_prefix("")

    def test_rejects_root(self) -> None:
        with pytest.raises(R2LifecycleValidationError, match="bucket-wide"):
            canonicalise_prefix("/")

    def test_rejects_non_string(self) -> None:
        with pytest.raises(R2LifecycleValidationError):
            canonicalise_prefix(123)  # type: ignore[arg-type]


class TestManagedRuleId:
    def test_is_deterministic(self) -> None:
        a = managed_rule_id("bkt", "project/")
        b = managed_rule_id("bkt", "project/")
        assert a == b

    def test_differs_for_different_bucket(self) -> None:
        a = managed_rule_id("bkt1", "project/")
        b = managed_rule_id("bkt2", "project/")
        assert a != b

    def test_differs_for_different_prefix(self) -> None:
        a = managed_rule_id("bkt", "project/")
        b = managed_rule_id("bkt", "other/")
        assert a != b

    def test_has_expected_prefix(self) -> None:
        rid = managed_rule_id("bkt", "project/")
        assert rid.startswith("vastai-gpu-runner-expire-")
        assert len(rid) == len("vastai-gpu-runner-expire-") + 12


# ---------------------------------------------------------------------------
# R2ExpirationPolicy validation
# ---------------------------------------------------------------------------


class TestR2ExpirationPolicy:
    def test_rejects_empty_bucket(self) -> None:
        with pytest.raises(R2LifecycleValidationError):
            R2ExpirationPolicy(bucket="", prefix="p/", expire_after_days=30)

    def test_rejects_zero_retention(self) -> None:
        with pytest.raises(R2LifecycleValidationError, match=">= 1"):
            R2ExpirationPolicy(bucket="b", prefix="p/", expire_after_days=0)

    def test_rejects_negative_retention(self) -> None:
        with pytest.raises(R2LifecycleValidationError):
            R2ExpirationPolicy(bucket="b", prefix="p/", expire_after_days=-1)

    def test_rejects_float_retention(self) -> None:
        with pytest.raises(R2LifecycleValidationError, match="integer"):
            R2ExpirationPolicy(bucket="b", prefix="p/", expire_after_days=1.5)  # type: ignore[arg-type]

    def test_rejects_string_retention(self) -> None:
        with pytest.raises(R2LifecycleValidationError):
            R2ExpirationPolicy(bucket="b", prefix="p/", expire_after_days="30")  # type: ignore[arg-type]

    def test_rejects_bool_retention(self) -> None:
        with pytest.raises(R2LifecycleValidationError):
            R2ExpirationPolicy(bucket="b", prefix="p/", expire_after_days=True)  # type: ignore[arg-type]

    def test_canonicalises_prefix_on_init(self) -> None:
        p = R2ExpirationPolicy(bucket="b", prefix="  project//x/", expire_after_days=5)
        assert p.canonical_prefix == "project/x/"


# ---------------------------------------------------------------------------
# R2AdminCredentials.from_file
# ---------------------------------------------------------------------------


class TestR2AdminCredentialsFromFile:
    def test_loads_admin_keys(self, tmp_path: Path) -> None:
        cred_file = tmp_path / "creds"
        cred_file.write_text(
            'export R2_ADMIN_ENDPOINT="https://r2.example"\n'
            'export R2_ADMIN_ACCESS_KEY_ID="akey"\n'
            'export R2_ADMIN_SECRET_ACCESS_KEY="test-secret-fixture"\n'
        )
        creds = R2AdminCredentials.from_file(cred_file)
        assert creds.endpoint == "https://r2.example"
        assert creds.access_key_id == "akey"
        assert creds.secret_access_key == "test-secret-fixture"  # noqa: S105

    def test_rejects_worker_only_keys(self, tmp_path: Path) -> None:
        """Worker-style ``R2_*`` keys are NOT accepted as admin credentials."""
        cred_file = tmp_path / "creds"
        cred_file.write_text(
            'export R2_ENDPOINT="https://r2.example"\n'
            'export R2_ACCESS_KEY_ID="akey"\n'
            'export R2_SECRET_ACCESS_KEY="test-secret-fixture"\n'
        )
        with pytest.raises(R2LifecycleCredentialsError, match="R2_ADMIN_ENDPOINT"):
            R2AdminCredentials.from_file(cred_file)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(R2LifecycleCredentialsError, match="not found"):
            R2AdminCredentials.from_file("/no/such/path")

    def test_missing_keys_raises(self, tmp_path: Path) -> None:
        cred_file = tmp_path / "creds"
        cred_file.write_text('export R2_ADMIN_ENDPOINT="https://r2.example"\n')
        with pytest.raises(R2LifecycleCredentialsError, match="R2_ADMIN_ACCESS_KEY_ID"):
            R2AdminCredentials.from_file(cred_file)

    def test_secret_is_not_in_repr(self) -> None:
        creds = R2AdminCredentials(
            endpoint="https://r2.example",
            access_key_id="akey",
            secret_access_key="topsecret",  # noqa: S106
        )
        text = repr(creds)
        assert "topsecret" not in text


# ---------------------------------------------------------------------------
# Fingerprint + rule helpers
# ---------------------------------------------------------------------------


class TestFingerprintRules:
    def test_stable_for_same_input(self) -> None:
        rules = [{"ID": "x", "Status": "Enabled", "Filter": {"Prefix": "p/"}}]
        assert fingerprint_rules(rules) == fingerprint_rules(list(rules))

    def test_differs_for_different_input(self) -> None:
        a = [{"ID": "x", "Status": "Enabled"}]
        b = [{"ID": "x", "Status": "Disabled"}]
        assert fingerprint_rules(a) != fingerprint_rules(b)


class TestNormalizeRules:
    def test_copies_each_rule(self) -> None:
        rules = [{"ID": "x", "Status": "Enabled"}]
        out = _normalize_rules(rules)
        out[0]["Status"] = "Disabled"
        assert rules[0]["Status"] == "Enabled"


class TestBuildAfterRulesApply:
    def test_appends_when_no_existing_managed_rule(self) -> None:
        out = _build_after_rules_apply(
            existing=({"ID": "f", "Status": "Enabled"},),
            rule_id="m",
            canonical_prefix="p/",
            expire_after_days=30,
        )
        ids = {r["ID"] for r in out}
        assert ids == {"f", "m"}

    def test_replaces_existing_managed_rule(self) -> None:
        existing = (
            {"ID": "m", "Status": "Enabled", "Filter": {"Prefix": "p/"}, "Expiration": {"Days": 1}},
            {"ID": "f", "Status": "Enabled"},
        )
        out = _build_after_rules_apply(
            existing=existing,
            rule_id="m",
            canonical_prefix="p/",
            expire_after_days=99,
        )
        m_rules = [r for r in out if r["ID"] == "m"]
        assert len(m_rules) == 1
        assert m_rules[0]["Expiration"]["Days"] == 99


class TestBuildAfterRulesRemove:
    def test_removes_only_managed(self) -> None:
        existing = (
            {"ID": "m", "Status": "Enabled"},
            {"ID": "f", "Status": "Enabled"},
        )
        out = _build_after_rules_remove(existing, "m")
        ids = [r["ID"] for r in out]
        assert ids == ["f"]


class TestRulesEqual:
    def test_equal_for_same_content(self) -> None:
        a = ({"ID": "x", "Status": "Enabled"},)
        b = ({"ID": "x", "Status": "Enabled"},)
        assert _rules_equal(a, b)

    def test_unequal_for_different_length(self) -> None:
        assert not _rules_equal(({"ID": "x"},), ())

    def test_unequal_for_different_content(self) -> None:
        assert not _rules_equal(
            ({"ID": "x", "Status": "Enabled"},),
            ({"ID": "x", "Status": "Disabled"},),
        )


# ---------------------------------------------------------------------------
# Manager — inspection / planning
# ---------------------------------------------------------------------------


def _manager(initial: list[dict[str, Any]] | None = None) -> tuple[R2LifecycleManager, FakeS3]:
    client = FakeS3(initial or [])
    creds = R2AdminCredentials(
        endpoint="https://r2.example",
        access_key_id="akey",
        secret_access_key="test-secret-fixture",  # noqa: S106
    )
    return R2LifecycleManager(creds, client=client), client


class TestReadLifecycle:
    def test_empty_when_no_lifecycle(self) -> None:
        mgr, _ = _manager()
        result = mgr.read_lifecycle("bkt")
        assert result.rules == ()

    def test_returns_existing_rules(self) -> None:
        existing = [
            {"ID": "f", "Status": "Enabled", "Filter": {"Prefix": "f/"}, "Expiration": {"Days": 1}},
        ]
        mgr, _ = _manager(existing)
        result = mgr.read_lifecycle("bkt")
        assert result.rules == tuple(existing)

    def test_access_denied_translated(self) -> None:
        mgr, client = _manager()
        client.fail_get = _client_error("AccessDenied")
        with pytest.raises(R2LifecycleAccessDeniedError):
            mgr.read_lifecycle("bkt")

    def test_no_such_lifecycle_treated_as_empty(self) -> None:
        """A fresh bucket returns ``NoSuchLifecycleConfiguration``; treat as empty."""
        mgr, client = _manager()
        client.fail_get = _client_error("NoSuchLifecycleConfiguration")
        result = mgr.read_lifecycle("bkt")
        assert result.rules == ()
        # Fingerprint must be stable and equal to fingerprint_rules([]).
        assert result.fingerprint == fingerprint_rules([])

    def test_apply_on_fresh_bucket_succeeds(self) -> None:
        """First apply against a fresh bucket must succeed (404 -> empty)."""
        mgr, client = _manager()
        # Only fail the FIRST read; subsequent reads must see the post-PUT
        # state so the verification can succeed.
        call_state = {"failed": True}

        def maybe_fail_get(*, Bucket):  # noqa: N803
            if call_state["failed"]:
                call_state["failed"] = False
                raise _client_error("NoSuchLifecycleConfiguration")
            return {"Rules": list(client.rules)}

        client.get_bucket_lifecycle_configuration = maybe_fail_get  # type: ignore[method-assign]
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))
        result = mgr.apply(plan)
        assert result.verified is True
        assert result.no_op is False


class TestPlanApply:
    def test_no_op_when_existing_managed_rule_matches(self) -> None:
        mgr_id = managed_rule_id("bkt", "p/")
        existing = [
            {
                "ID": mgr_id,
                "Status": "Enabled",
                "Filter": {"Prefix": "p/"},
                "Expiration": {"Days": 30},
            }
        ]
        mgr, _ = _manager(existing)
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))
        assert plan.no_op is True
        assert plan.before_rules == plan.after_rules

    def test_creates_rule_when_absent(self) -> None:
        mgr, _ = _manager()
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))
        assert plan.no_op is False
        assert any(r["ID"] == plan.managed_rule_id for r in plan.after_rules)
        assert all(r["ID"] != plan.managed_rule_id for r in plan.before_rules)

    def test_preserves_foreign_rules(self) -> None:
        foreign = {
            "ID": "foreign",
            "Status": "Enabled",
            "Filter": {"Prefix": "f/"},
            "Expiration": {"Days": 5},
            "Transition": {"Days": 1, "StorageClass": "GLACIER"},  # foreign field
        }
        mgr, _ = _manager([foreign])
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))
        kept = [r for r in plan.after_rules if r["ID"] == "foreign"]
        assert len(kept) == 1
        # Foreign fields preserved verbatim.
        assert kept[0].get("Transition") == foreign["Transition"]

    def test_warns_on_overlapping_foreign_rule(self) -> None:
        foreign = {
            "ID": "foreign",
            "Status": "Enabled",
            "Filter": {"Prefix": "p/"},
            "Expiration": {"Days": 1},
        }
        mgr, _ = _manager([foreign])
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))
        assert any("Overlapping foreign rule" in w for w in plan.warnings)

    def test_does_not_warn_on_unrelated_foreign_rule(self) -> None:
        foreign = {
            "ID": "foreign",
            "Status": "Enabled",
            "Filter": {"Prefix": "other/"},
            "Expiration": {"Days": 5},
        }
        mgr, _ = _manager([foreign])
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))
        assert plan.warnings == ()

    def test_incompatible_managed_collision_raises(self) -> None:
        mgr_id = managed_rule_id("bkt", "p/")
        existing = [
            {
                "ID": mgr_id,
                "Status": "Enabled",
                "Filter": {"Prefix": "p/"},
                "Expiration": {"Days": 30},
            }
        ]
        _mgr, _ = _manager(existing)
        # Simulate an incompatible managed-rule shape by post-mutating
        # _assert_no_incompatible_managed's view: rule with no Days.
        with pytest.raises(R2LifecycleCollisionError):
            _assert_no_incompatible_managed(
                ({"ID": mgr_id, "Status": "Enabled", "Expiration": {}},),
                mgr_id,
            )


class TestPlanRemove:
    def test_no_op_when_absent(self) -> None:
        mgr, _ = _manager()
        plan = mgr.plan_remove("bkt", "p/")
        assert plan.no_op is True

    def test_removes_when_present(self) -> None:
        mgr_id = managed_rule_id("bkt", "p/")
        existing = [
            {"ID": mgr_id, "Status": "Enabled", "Filter": {"Prefix": "p/"}},
            {"ID": "f", "Status": "Enabled"},
        ]
        mgr, _ = _manager(existing)
        plan = mgr.plan_remove("bkt", "p/")
        assert plan.no_op is False
        assert all(r["ID"] != mgr_id for r in plan.after_rules)
        assert any(r["ID"] == "f" for r in plan.after_rules)


# ---------------------------------------------------------------------------
# Manager — mutation + read-after-write
# ---------------------------------------------------------------------------


class TestApply:
    def test_no_op_does_not_call_put(self) -> None:
        mgr_id = managed_rule_id("bkt", "p/")
        existing = [
            {
                "ID": mgr_id,
                "Status": "Enabled",
                "Filter": {"Prefix": "p/"},
                "Expiration": {"Days": 30},
            }
        ]
        mgr, client = _manager(existing)
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))
        result = mgr.apply(plan)
        assert result.no_op is True
        assert result.verified is True
        assert client.put_calls == []

    def test_writes_and_verifies(self) -> None:
        mgr, client = _manager()
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))
        result = mgr.apply(plan)
        assert result.no_op is False
        assert result.verified is True
        assert len(client.put_calls) == 1
        # Post-write rules must contain the managed rule with right shape.
        assert client.rules[0]["ID"] == plan.managed_rule_id
        assert client.rules[0]["Expiration"]["Days"] == 30

    def test_stale_plan_raises(self) -> None:
        mgr, client = _manager()
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))
        # Mutate external state to simulate concurrent admin edit.
        client.rules.append({"ID": "other", "Status": "Enabled"})
        with pytest.raises(R2LifecycleStalePlanError):
            mgr.apply(plan)

    def test_read_after_write_mismatch_raises(self) -> None:
        mgr, client = _manager()
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))

        # Intercept put to corrupt the resulting state before the
        # read-after-write verification. Mutate a *fresh* dict so we
        # do not also mutate ``plan.after_rules`` (which holds the
        # same dict references by construction).
        original_put = client.put_bucket_lifecycle_configuration

        def corrupt_put(
            *,
            Bucket: str,  # noqa: N803
            LifecycleConfiguration: dict[str, Any],  # noqa: N803
        ) -> dict[str, Any]:
            original_put(
                Bucket=Bucket,
                LifecycleConfiguration=LifecycleConfiguration,
            )
            # Replace the stored rules with a fresh list of fresh dicts
            # so the corruption does not bleed into plan.after_rules.
            client.rules = [
                {**r, "Expiration": {"Days": 999}} if r.get("ID") == plan.managed_rule_id else r
                for r in client.rules
            ]
            return {}

        client.put_bucket_lifecycle_configuration = corrupt_put  # type: ignore[method-assign]
        with pytest.raises(R2LifecycleVerificationError):
            mgr.apply(plan)

    def test_rule_limit_translated(self) -> None:
        mgr, client = _manager()
        client.fail_put = _client_error("TooManyRules", "limit reached")
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))
        with pytest.raises(R2LifecycleRuleLimitError):
            mgr.apply(plan)

    def test_access_denied_on_put_translated(self) -> None:
        mgr, client = _manager()
        client.fail_put = _client_error("AccessDenied")
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))
        with pytest.raises(R2LifecycleAccessDeniedError):
            mgr.apply(plan)

    def test_same_id_different_prefix_on_apply_raises_collision(self) -> None:
        """A managed-rule ID bound to a different prefix must reject apply."""
        mgr_id = managed_rule_id("bkt", "p/")
        # The bucket already has our rule bound to a different prefix.
        existing = [
            {
                "ID": mgr_id,
                "Status": "Enabled",
                "Filter": {"Prefix": "other/"},
                "Expiration": {"Days": 30},
            }
        ]
        mgr, _ = _manager(existing)
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))
        with pytest.raises(R2LifecycleCollisionError, match="bound to prefix"):
            mgr.apply(plan)

    def test_same_id_missing_filter_on_apply_raises_collision(self) -> None:
        """A same-ID rule with no Filter field is not ours — reject."""
        mgr_id = managed_rule_id("bkt", "p/")
        existing = [
            {
                "ID": mgr_id,
                "Status": "Enabled",
                # No Filter at all — rule is not bound to our prefix.
                "Expiration": {"Days": 30},
            }
        ]
        mgr, _ = _manager(existing)
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))
        with pytest.raises(R2LifecycleCollisionError, match="bound to prefix"):
            mgr.apply(plan)

    def test_same_id_empty_prefix_on_apply_raises_collision(self) -> None:
        """A same-ID rule with ``Filter: {"Prefix": ""}`` is not ours — reject."""
        mgr_id = managed_rule_id("bkt", "p/")
        existing = [
            {
                "ID": mgr_id,
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "Expiration": {"Days": 30},
            }
        ]
        mgr, _ = _manager(existing)
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))
        with pytest.raises(R2LifecycleCollisionError, match="bound to prefix"):
            mgr.apply(plan)

    def test_same_id_missing_prefix_key_on_apply_raises_collision(self) -> None:
        """A same-ID rule with ``Filter`` but no ``Prefix`` key — reject."""
        mgr_id = managed_rule_id("bkt", "p/")
        existing = [
            {
                "ID": mgr_id,
                "Status": "Enabled",
                "Filter": {},
                "Expiration": {"Days": 30},
            }
        ]
        mgr, _ = _manager(existing)
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))
        with pytest.raises(R2LifecycleCollisionError, match="bound to prefix"):
            mgr.apply(plan)

    def test_verifies_full_ruleset_after_put(self) -> None:
        """apply must reject if a foreign rule is dropped by the provider."""
        mgr, client = _manager(
            [
                {"ID": "f", "Status": "Enabled", "Filter": {"Prefix": "f/"}},
            ]
        )
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))

        original_put = client.put_bucket_lifecycle_configuration

        def drop_foreign(*, Bucket, LifecycleConfiguration):  # noqa: N803
            original_put(
                Bucket=Bucket,
                LifecycleConfiguration=LifecycleConfiguration,
            )
            # Provider silently drops the foreign rule.
            client.rules = [r for r in client.rules if r.get("ID") != "f"]
            return {}

        client.put_bucket_lifecycle_configuration = drop_foreign  # type: ignore[method-assign]
        with pytest.raises(R2LifecycleVerificationError):
            mgr.apply(plan)


class TestRemove:
    def test_no_op_does_not_call_put(self) -> None:
        mgr, client = _manager()
        plan = mgr.plan_remove("bkt", "p/")
        result = mgr.remove(plan)
        assert result.no_op is True
        assert client.put_calls == []

    def test_removes_and_verifies(self) -> None:
        mgr_id = managed_rule_id("bkt", "p/")
        existing = [
            {
                "ID": mgr_id,
                "Status": "Enabled",
                "Filter": {"Prefix": "p/"},
                "Expiration": {"Days": 30},
            },
            {"ID": "f", "Status": "Enabled"},
        ]
        mgr, client = _manager(existing)
        plan = mgr.plan_remove("bkt", "p/")
        result = mgr.remove(plan)
        assert result.no_op is False
        assert result.verified is True
        # Managed rule must be gone, foreign preserved.
        assert all(r["ID"] != mgr_id for r in client.rules)
        assert any(r["ID"] == "f" for r in client.rules)

    def test_same_id_different_prefix_on_remove_raises_collision(self) -> None:
        """remove must reject if the same-ID rule is bound to a different prefix."""
        mgr_id = managed_rule_id("bkt", "p/")
        existing = [
            {
                "ID": mgr_id,
                "Status": "Enabled",
                "Filter": {"Prefix": "other/"},
                "Expiration": {"Days": 30},
            }
        ]
        mgr, _ = _manager(existing)
        plan = mgr.plan_remove("bkt", "p/")
        with pytest.raises(R2LifecycleCollisionError, match="bound to prefix"):
            mgr.remove(plan)

    def test_same_id_missing_filter_on_remove_raises_collision(self) -> None:
        """remove must reject if a same-ID rule has no Filter field."""
        mgr_id = managed_rule_id("bkt", "p/")
        existing = [
            {
                "ID": mgr_id,
                "Status": "Enabled",
                "Expiration": {"Days": 30},
            }
        ]
        mgr, _ = _manager(existing)
        plan = mgr.plan_remove("bkt", "p/")
        with pytest.raises(R2LifecycleCollisionError, match="bound to prefix"):
            mgr.remove(plan)

    def test_remove_only_rule_calls_delete_not_put(self) -> None:
        """When the managed rule is the only one, remove calls delete_bucket_lifecycle."""
        mgr_id = managed_rule_id("bkt", "p/")
        existing = [
            {
                "ID": mgr_id,
                "Status": "Enabled",
                "Filter": {"Prefix": "p/"},
                "Expiration": {"Days": 30},
            }
        ]
        mgr, client = _manager(existing)
        plan = mgr.plan_remove("bkt", "p/")
        result = mgr.remove(plan)
        assert result.verified is True
        # PUT must NOT be called when the only rule is the managed one.
        assert client.put_calls == []
        # DELETE must be called exactly once.
        assert client.delete_calls == 1
        # Bucket lifecycle configuration is empty.
        assert client.rules == []

    def test_remove_keeps_put_when_foreign_rules_remain(self) -> None:
        """When foreign rules remain, remove calls PUT (preserves them)."""
        mgr_id = managed_rule_id("bkt", "p/")
        existing = [
            {
                "ID": mgr_id,
                "Status": "Enabled",
                "Filter": {"Prefix": "p/"},
                "Expiration": {"Days": 30},
            },
            {"ID": "f", "Status": "Enabled"},
        ]
        mgr, client = _manager(existing)
        plan = mgr.plan_remove("bkt", "p/")
        result = mgr.remove(plan)
        assert result.verified is True
        assert len(client.put_calls) == 1
        assert client.delete_calls == 0
        # Managed rule removed, foreign preserved.
        assert all(r["ID"] != mgr_id for r in client.rules)
        assert any(r["ID"] == "f" for r in client.rules)

    def test_remove_verifies_full_ruleset(self) -> None:
        """remove must reject if a foreign rule is dropped by the provider."""
        mgr_id = managed_rule_id("bkt", "p/")
        existing = [
            {
                "ID": mgr_id,
                "Status": "Enabled",
                "Filter": {"Prefix": "p/"},
                "Expiration": {"Days": 30},
            },
            {"ID": "f", "Status": "Enabled"},
        ]
        mgr, client = _manager(existing)
        plan = mgr.plan_remove("bkt", "p/")

        original_put = client.put_bucket_lifecycle_configuration

        def drop_foreign(*, Bucket, LifecycleConfiguration):  # noqa: N803
            original_put(
                Bucket=Bucket,
                LifecycleConfiguration=LifecycleConfiguration,
            )
            client.rules = [r for r in client.rules if r.get("ID") != "f"]
            return {}

        client.put_bucket_lifecycle_configuration = drop_foreign  # type: ignore[method-assign]
        with pytest.raises(R2LifecycleVerificationError):
            mgr.remove(plan)


class TestInspectManagedRule:
    def test_returns_none_when_absent(self) -> None:
        mgr, _ = _manager()
        assert mgr.inspect_managed_rule("bkt", "p/") is None

    def test_returns_rule_when_present(self) -> None:
        mgr_id = managed_rule_id("bkt", "p/")
        rule = {
            "ID": mgr_id,
            "Status": "Enabled",
            "Filter": {"Prefix": "p/"},
            "Expiration": {"Days": 30},
        }
        mgr, _ = _manager([rule])
        assert mgr.inspect_managed_rule("bkt", "p/") == rule


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


class TestIsRuleLimitError:
    def test_too_many_rules(self) -> None:
        assert _is_rule_limit_error(_client_error("TooManyRules")) is True

    def test_unrelated_error(self) -> None:
        assert _is_rule_limit_error(_client_error("AccessDenied")) is False


class TestBuildAdminClient:
    def test_returns_boto3_client(self) -> None:
        # Smoke test — actual boto3 client creation without network.
        creds = R2AdminCredentials(
            endpoint="https://r2.example",
            access_key_id="akey",
            secret_access_key="test-secret-fixture",  # noqa: S106
        )
        client = build_admin_client(creds)
        # boto3 clients have a meta attribute pointing to the service-2 model.
        assert client is not None


class TestBoto3Contract:
    """Verify the manager actually speaks the boto3 contract.

    These tests use ``botocore.stub.Stubber`` against a real boto3
    S3 client, so any incorrect parameter shape (e.g. ``Rules=`` at
    the top level instead of ``LifecycleConfiguration={"Rules": ...}``)
    raises ``ParamValidationError`` BEFORE the stubber even runs.
    """

    def _stubbed_client(self):
        """Build a real boto3 client wired to a ``botocore.stub.Stubber``."""
        import boto3
        from botocore.stub import Stubber

        client = boto3.client(
            "s3",
            endpoint_url="https://r2.example",
            aws_access_key_id="akey",
            aws_secret_access_key="test-secret-fixture",  # noqa: S106
            region_name="auto",
        )
        return client, Stubber(client)

    def test_apply_calls_put_with_lifecycle_configuration_wrapper(self) -> None:
        """The manager must nest rules under ``LifecycleConfiguration``.

        This is a CONTRACT test against a real boto3 client with
        ``botocore.stub.Stubber`` activated — any incorrect parameter
        shape raises ``ParamValidationError`` before the stub is
        consulted, catching the bug where ``Rules=`` is passed at the
        top level instead of under ``LifecycleConfiguration=``.
        """
        client, stubber = self._stubbed_client()
        mgr_id = managed_rule_id("bkt", "p/")
        mgr_rule = {
            "ID": mgr_id,
            "Status": "Enabled",
            "Filter": {"Prefix": "p/"},
            "Expiration": {"Days": 30},
        }
        # Manager order: GET (plan), GET (fresh), PUT (mutation), GET (verify).
        # Stubber cannot return different values for the same method on
        # repeated calls, so we wrap the method to dispatch on call count.
        call_count = {"gets": 0}

        def get_with_state_dependent_response(**_kwargs):
            call_count["gets"] += 1
            if call_count["gets"] == 3:
                # Verify: return the post-write state.
                return {"Rules": [mgr_rule]}
            # Plan + freshness: bucket is empty.
            return {"Rules": []}

        client.get_bucket_lifecycle_configuration = get_with_state_dependent_response  # type: ignore[method-assign]
        stubber.add_response(
            "put_bucket_lifecycle_configuration",
            {},
            {
                "Bucket": "bkt",
                "LifecycleConfiguration": {"Rules": [mgr_rule]},
            },
        )
        stubber.activate()
        creds = R2AdminCredentials(
            endpoint="https://r2.example",
            access_key_id="akey",
            secret_access_key="test-secret-fixture",  # noqa: S106
        )
        mgr = R2LifecycleManager(creds, client=client)
        plan = mgr.plan_apply(R2ExpirationPolicy("bkt", "p/", 30))
        result = mgr.apply(plan)
        stubber.assert_no_pending_responses()
        assert result.verified is True

    def test_remove_only_rule_calls_delete_bucket_lifecycle(self) -> None:
        """When removing the only rule, the manager calls DELETE not PUT.

        This is a CONTRACT test against a real boto3 client with
        ``botocore.stub.Stubber`` activated — verifies the request
        shape matches the boto3 S3 API.
        """
        client, stubber = self._stubbed_client()
        mgr_id = managed_rule_id("bkt", "p/")
        mgr_rule = {
            "ID": mgr_id,
            "Status": "Enabled",
            "Filter": {"Prefix": "p/"},
            "Expiration": {"Days": 30},
        }
        # Manager order: GET (plan), GET (fresh), DELETE (mutation),
        # GET (verify — bucket returns NoSuchLifecycleConfiguration,
        # which the manager treats as empty rules).
        # Stubber doesn't accept ClientError stubs, so we wrap the
        # stubbed GET to convert the success-response into the right
        # error AFTER the DELETE has been issued.
        call_count = {"gets": 0}

        def get_with_post_delete_error(**_kwargs):
            call_count["gets"] += 1
            if call_count["gets"] <= 2:
                # First two GETs (plan + fresh): return the rule.
                return {"Rules": [mgr_rule]}
            # Third GET (verify): bucket is now empty.
            import botocore.exceptions as _boto_exc

            err = _boto_exc.ClientError(
                {"Error": {"Code": "NoSuchLifecycleConfiguration"}},
                "GetBucketLifecycleConfiguration",
            )
            raise err

        client.get_bucket_lifecycle_configuration = get_with_post_delete_error  # type: ignore[method-assign]
        stubber.add_response(
            "delete_bucket_lifecycle",
            {},
            {"Bucket": "bkt"},
        )
        stubber.activate()
        creds = R2AdminCredentials(
            endpoint="https://r2.example",
            access_key_id="akey",
            secret_access_key="test-secret-fixture",  # noqa: S106
        )
        mgr = R2LifecycleManager(creds, client=client)
        plan = mgr.plan_remove("bkt", "p/")
        result = mgr.remove(plan)
        stubber.assert_no_pending_responses()
        assert result.verified is True


class TestSecretRedaction:
    """The credentials object and CLI must not leak secrets into repr/str."""

    def test_credentials_repr_omits_secret(self) -> None:
        creds = R2AdminCredentials(
            endpoint="https://r2.example",
            access_key_id="akey",
            secret_access_key="topsecret_value",  # noqa: S106
        )
        text = repr(creds)
        assert "topsecret_value" not in text

    def test_manager_repr_omits_secret(self) -> None:
        creds = R2AdminCredentials(
            endpoint="https://r2.example",
            access_key_id="akey",
            secret_access_key="topsecret_value",  # noqa: S106
        )
        mgr = R2LifecycleManager(creds, client=FakeS3())
        text = repr(mgr)
        assert "topsecret_value" not in text


# ---------------------------------------------------------------------------
# Fixture smoke tests for typed result objects
# ---------------------------------------------------------------------------


def test_lifecycle_result_construction() -> None:
    r = LifecycleResult(
        operation="apply",
        bucket="b",
        canonical_prefix="p/",
        managed_rule_id="x",
        verified=True,
        no_op=False,
        rules_count=2,
    )
    assert r.rules_count == 2


def test_lifecycle_plan_construction() -> None:
    p = LifecyclePlan(
        operation="apply",
        bucket="b",
        canonical_prefix="p/",
        managed_rule_id="x",
        expire_after_days=30,
        before_rules=(),
        after_rules=(),
        no_op=True,
        source_fingerprint="abc",
        warnings=("warn",),
    )
    assert p.warnings == ("warn",)


def test_r2_lifecycle_error_is_base() -> None:
    err = R2LifecycleError("x")
    assert isinstance(err, Exception)


def test_credentials_error_is_subclass() -> None:
    err = R2LifecycleCredentialsError("x")
    assert isinstance(err, R2LifecycleError)
