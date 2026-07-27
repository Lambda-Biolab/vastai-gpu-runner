"""Tests for cleanup_policy — provider-agnostic DTOs + generic policy.

Per docs/architecture-v4-cleanup-policy.md migration step 2.
"""

from __future__ import annotations

import inspect

import pytest

from vastai_gpu_runner.cleanup_policy import (
    CleanupRefusal,
    CleanupResult,
    CleanupVerdict,
    InstanceCandidate,
    OwnershipPolicy,
    OwnershipVerification,
    ProviderCleanupPolicy,
    _repository,
    normalize_instance_id,
)
from vastai_gpu_runner.types import Provider

# ---------------------------------------------------------------------------
# _repository — Docker/OCI grammar
# ---------------------------------------------------------------------------


class TestRepositoryValid:
    def test_simple_repository(self) -> None:
        assert _repository("nginx") == "nginx"
        assert _repository("library/nginx") == "library/nginx"

    def test_repository_with_tag(self) -> None:
        assert _repository("nginx:1.27") == "nginx"
        assert _repository("library/nginx:latest") == "library/nginx"

    def test_repository_with_digest_sha256(self) -> None:
        digest = "sha256:" + "0" * 64
        assert _repository(f"nginx@{digest}") == "nginx"

    def test_repository_with_digest_sha512(self) -> None:
        digest = "sha512:" + "0" * 128
        assert _repository(f"nginx@{digest}") == "nginx"

    def test_repository_with_blake3(self) -> None:
        digest = "blake3:" + "0" * 64
        assert _repository(f"nginx@{digest}") == "nginx"

    def test_repository_with_registry_port(self) -> None:
        assert _repository("registry.local:5000/myorg/app:1.0") == "registry.local:5000/myorg/app"

    def test_repository_with_dns_registry(self) -> None:
        assert _repository("docker.io/library/nginx:latest") == "docker.io/library/nginx"

    def test_repository_with_ipv6_registry(self) -> None:
        assert _repository("[::1]:5000/myorg/app:1.0") == "[::1]:5000/myorg/app"

    def test_repository_localhost(self) -> None:
        assert _repository("localhost/myorg/app:1.0") == "localhost/myorg/app"

    def test_repository_with_double_underscore(self) -> None:
        assert _repository("myorg/my__app:1.0") == "myorg/my__app"

    def test_repository_with_path_dots_underscores_hyphens(self) -> None:
        assert _repository("my.org/app_v1.2-beta:1.0") == "my.org/app_v1.2-beta"

    def test_underscore_in_path_component_is_allowed(self) -> None:
        # Docker permits `_` and `__` as separators within a component.
        assert _repository("myorg/my_app:1.0") == "myorg/my_app"


class TestRepositoryInvalid:
    @pytest.mark.parametrize(
        "ref",
        [
            "",  # empty
            " ",  # whitespace
            "bad!tag",  # no tag, invalid component
            "nginx:bad!tag",  # invalid tag chars
            "nginx:" + ("a" * 129),  # tag over 128 chars
            "nginx@" + "0" * 64,  # no algorithm
            "nginx@sha256:" + "a" * 63,  # sha256 too short
            "nginx@sha256:" + "Z" * 64,  # sha256 not lowercase hex
            "nginx@SHA256:" + "0" * 64,  # algorithm uppercase
            "nginx@Sha256:" + "0" * 64,  # algorithm mixed case
            "nginx@sha256:" + "G" * 64,  # non-hex
            "registry:abc/myorg/app",  # bad port
            "registry:99999/myorg/app",  # port out of range
            "registry..invalid/myorg/app",  # double-dot label
            "-leading-hyphen/myorg/app",  # leading hyphen in label
            "myORG/app",  # upper-case path component
            "nginx@" + ("a" * 200),  # invalid generic digest
            "myorg/app@digest@sha256:abc",  # multiple @
        ],
    )
    def test_returns_empty(self, ref: str) -> None:
        assert _repository(ref) == ""


# ---------------------------------------------------------------------------
# OwnershipPolicy
# ---------------------------------------------------------------------------


class TestOwnershipPolicyMatches:
    def test_reflexive_exact_match(self) -> None:
        policy = OwnershipPolicy(owned_images=frozenset({"myorg/app:1.0"}))
        assert policy.matches("myorg/app:1.0") is True

    def test_tag_insensitive_match(self) -> None:
        policy = OwnershipPolicy(owned_images=frozenset({"myorg/app:1.0"}))
        assert policy.matches("myorg/app:latest") is True
        assert policy.matches("myorg/app:2.0-beta") is True

    def test_sha256_by_repository(self) -> None:
        digest = "sha256:" + "0" * 64
        policy = OwnershipPolicy(owned_images=frozenset({"myorg/app:1.0"}))
        assert policy.matches(f"myorg/app@{digest}") is True

    def test_registry_port_aware(self) -> None:
        policy = OwnershipPolicy(owned_images=frozenset({"myorg/app:1.0"}))
        assert policy.matches("registry.local:5000/myorg/app:1.0") is False

        policy2 = OwnershipPolicy(owned_images=frozenset({"registry.local:5000/myorg/app:1.0"}))
        assert policy2.matches("myorg/app:1.0") is False

    def test_empty_set_fails_closed(self) -> None:
        policy = OwnershipPolicy(owned_images=frozenset())
        assert policy.matches("myorg/app:1.0") is False

    def test_none_disables_check(self) -> None:
        policy = OwnershipPolicy(owned_images=None)
        assert policy.matches("myorg/app:1.0") is True
        assert policy.matches("") is True

    def test_malformed_reference_fails_closed(self) -> None:
        policy = OwnershipPolicy(owned_images=frozenset({"myorg/app:1.0"}))
        assert policy.matches("bad!tag") is False
        assert policy.matches("") is False

    def test_owned_images_is_frozen(self) -> None:
        policy = OwnershipPolicy(owned_images={"a"})
        assert isinstance(policy.owned_images, frozenset)

    def test_matches_uses_precomputed_cache(self) -> None:
        policy = OwnershipPolicy(owned_images=frozenset({"myorg/app:1.0"}))
        assert policy._normalised == frozenset({"myorg/app"})


class TestOwnershipPolicyNormalisedCache:
    def test_normalised_field_declared(self) -> None:
        fields = {f.name for f in OwnershipPolicy.__dataclass_fields__.values()}
        assert "_normalised" in fields

    def test_precomputed_in_post_init(self) -> None:
        policy = OwnershipPolicy(owned_images=frozenset({"myorg/app:1.0"}))
        assert policy._normalised == frozenset({"myorg/app"})

    def test_normalised_skips_malformed_entries(self) -> None:
        policy = OwnershipPolicy(owned_images=frozenset({"myorg/app:1.0", "bad!tag"}))
        assert policy._normalised == frozenset({"myorg/app"})

    def test_match_is_o1(self) -> None:
        # Sanity: a single match() call should not re-iterate owned_images
        # beyond the precomputed cache lookup.
        policy = OwnershipPolicy(owned_images=frozenset(f"myorg/app{i}:1.0" for i in range(1000)))
        assert policy.matches("myorg/app500:latest") is True


# ---------------------------------------------------------------------------
# InstanceCandidate invariants
# ---------------------------------------------------------------------------


class TestInstanceCandidateValid:
    def test_minimal_required_fields(self) -> None:
        c = InstanceCandidate(
            provider=Provider.VASTAI,
            instance_id="123",
            label="prod",
            state="running",
        )
        assert c.instance_id == "123"
        assert c.provider == Provider.VASTAI


class TestInstanceCandidateInvalid:
    def test_invalid_provider(self) -> None:
        with pytest.raises(ValueError, match="provider must be a Provider"):
            InstanceCandidate(
                provider="vastai",  # type: ignore[arg-type]
                instance_id="123",
                label="prod",
                state="running",
            )

    def test_empty_instance_id(self) -> None:
        with pytest.raises(ValueError, match="instance_id must be non-empty"):
            InstanceCandidate(
                provider=Provider.VASTAI,
                instance_id="",
                label="prod",
                state="running",
            )

    def test_padded_instance_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="instance_id must be non-empty"):
            InstanceCandidate(
                provider=Provider.VASTAI,
                instance_id=" 123 ",
                label="prod",
                state="running",
            )

    def test_non_string_label(self) -> None:
        with pytest.raises(ValueError, match="label must be a string"):
            InstanceCandidate(
                provider=Provider.VASTAI,
                instance_id="123",
                label=42,  # type: ignore[arg-type]
                state="running",
            )

    @pytest.mark.parametrize(
        "cost",
        [-1.0, float("nan"), float("inf"), float("-inf")],
    )
    def test_invalid_cost_per_hour(self, cost: float) -> None:
        with pytest.raises(ValueError, match="cost_per_hour"):
            InstanceCandidate(
                provider=Provider.VASTAI,
                instance_id="123",
                label="prod",
                state="running",
                cost_per_hour=cost,
            )

    def test_boolean_cost_per_hour_rejected(self) -> None:
        with pytest.raises(ValueError, match="cost_per_hour"):
            InstanceCandidate(
                provider=Provider.VASTAI,
                instance_id="123",
                label="prod",
                state="running",
                cost_per_hour=True,  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# CleanupResult invariants
# ---------------------------------------------------------------------------


class TestCleanupResultValid:
    def test_destroyed_no_error(self) -> None:
        r = CleanupResult(verdict=CleanupVerdict.DESTROYED)
        assert r.verdict == CleanupVerdict.DESTROYED
        assert r.error == ""

    def test_already_gone_no_error(self) -> None:
        r = CleanupResult(verdict=CleanupVerdict.ALREADY_GONE)
        assert r.verdict == CleanupVerdict.ALREADY_GONE
        assert r.error == ""

    def test_leaked_requires_error(self) -> None:
        r = CleanupResult(verdict=CleanupVerdict.LEAKED, error="resurrected")
        assert r.verdict == CleanupVerdict.LEAKED
        assert r.error == "resurrected"

    def test_ownership_refusal(self) -> None:
        r = CleanupResult(refusal=CleanupRefusal.OWNERSHIP, error="image mismatch")
        assert r.refusal == CleanupRefusal.OWNERSHIP
        assert r.error == "image mismatch"


class TestCleanupResultInvalid:
    def test_neither_verdict_nor_refusal(self) -> None:
        with pytest.raises(ValueError, match="exactly one of verdict or refusal"):
            CleanupResult()

    def test_both_verdict_and_refusal(self) -> None:
        with pytest.raises(ValueError, match="exactly one of verdict or refusal"):
            CleanupResult(
                verdict=CleanupVerdict.DESTROYED,
                refusal=CleanupRefusal.OWNERSHIP,
            )

    def test_destroyed_with_error_rejected(self) -> None:
        with pytest.raises(ValueError, match="successful end-states must have empty error"):
            CleanupResult(verdict=CleanupVerdict.DESTROYED, error="oops")

    def test_already_gone_with_error_rejected(self) -> None:
        with pytest.raises(ValueError, match="successful end-states must have empty error"):
            CleanupResult(verdict=CleanupVerdict.ALREADY_GONE, error="oops")

    def test_leaked_without_error_rejected(self) -> None:
        with pytest.raises(ValueError, match="must have non-empty error"):
            CleanupResult(verdict=CleanupVerdict.LEAKED)

    def test_ownership_refusal_without_error_rejected(self) -> None:
        with pytest.raises(ValueError, match="must have non-empty error"):
            CleanupResult(refusal=CleanupRefusal.OWNERSHIP)

    def test_string_verdict_rejected(self) -> None:
        with pytest.raises(ValueError, match="verdict must be a CleanupVerdict"):
            CleanupResult(verdict="destroyed", error="x")  # type: ignore[arg-type]

    def test_none_error_rejected(self) -> None:
        with pytest.raises(ValueError, match="error must be a string"):
            CleanupResult(
                verdict=CleanupVerdict.LEAKED,
                error=None,  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# ProviderCleanupPolicy
# ---------------------------------------------------------------------------


def _candidate(
    instance_id: str = "123",
    *,
    provider: Provider = Provider.VASTAI,
    state: str = "running",
) -> InstanceCandidate:
    return InstanceCandidate(
        provider=provider,
        instance_id=instance_id,
        label="prod",
        state=state,
    )


class TestProviderCleanupPolicyConstruction:
    def test_minimum_required_fields(self) -> None:
        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=lambda: [],
            destroy_fn=lambda c: CleanupResult(verdict=CleanupVerdict.DESTROYED),
        )
        assert policy.provider == Provider.VASTAI

    def test_kw_only(self) -> None:
        # Positional construction should fail because of kw_only=True.
        with pytest.raises(TypeError):
            ProviderCleanupPolicy(  # type: ignore[misc]
                Provider.VASTAI,
                lambda: [],
                lambda c: CleanupResult(verdict=CleanupVerdict.DESTROYED),
            )


class TestProviderCleanupPolicyListInstances:
    def test_returns_valid_candidates(self) -> None:
        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=lambda: [_candidate()],
            destroy_fn=lambda c: CleanupResult(verdict=CleanupVerdict.DESTROYED),
        )
        result = policy.list_instances()
        assert len(result) == 1
        assert result[0].instance_id == "123"

    def test_callback_exception_returns_empty(self, caplog) -> None:
        def raises() -> list[InstanceCandidate]:
            raise RuntimeError("boom")

        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=raises,
            destroy_fn=lambda c: CleanupResult(verdict=CleanupVerdict.DESTROYED),
        )
        with caplog.at_level("ERROR"):
            assert policy.list_instances() == []
        assert "list_instances raised" in caplog.text

    def test_non_list_result_returns_empty(self, caplog) -> None:
        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=lambda: "not a list",  # type: ignore[return-value]
            destroy_fn=lambda c: CleanupResult(verdict=CleanupVerdict.DESTROYED),
        )
        with caplog.at_level("ERROR"):
            assert policy.list_instances() == []
        assert "returned invalid type" in caplog.text

    def test_non_candidate_element_returns_empty(self, caplog) -> None:
        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=lambda: [_candidate(), "bad"],  # type: ignore[list-item]
            destroy_fn=lambda c: CleanupResult(verdict=CleanupVerdict.DESTROYED),
        )
        with caplog.at_level("ERROR"):
            assert policy.list_instances() == []
        assert "non-InstanceCandidate element" in caplog.text


class TestProviderCleanupPolicyDestroy:
    def test_provider_mismatch_returns_proper_diagnostic(self) -> None:
        destroy_called = []

        def destroy_fn(c: InstanceCandidate) -> CleanupResult:
            destroy_called.append(c)
            return CleanupResult(verdict=CleanupVerdict.DESTROYED)

        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=lambda: [],
            destroy_fn=destroy_fn,
        )
        bad = _candidate(provider=Provider.RUNPOD)
        result = policy.destroy(bad)
        assert result.refusal == CleanupRefusal.PROVIDER_MISMATCH
        assert "RUNPOD" in result.error
        assert "VASTAI" in result.error
        assert destroy_called == []

    def test_invalid_candidate_returns_unknown(self) -> None:
        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=lambda: [],
            destroy_fn=lambda c: CleanupResult(verdict=CleanupVerdict.DESTROYED),
        )
        result = policy.destroy("not a candidate")  # type: ignore[arg-type]
        assert result.verdict == CleanupVerdict.UNKNOWN
        assert "invalid candidate type" in result.error

    def test_destroy_fn_returns_none_substitutes_unknown(self, caplog) -> None:
        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=lambda: [],
            destroy_fn=lambda c: None,  # type: ignore[return-value]
        )
        with caplog.at_level("ERROR"):
            result = policy.destroy(_candidate())
        assert result.verdict == CleanupVerdict.UNKNOWN
        assert "NoneType" in result.error

    def test_destroy_fn_returns_non_cleanup_result(self, caplog) -> None:
        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=lambda: [],
            destroy_fn=lambda c: "not a cleanup result",  # type: ignore[return-value]
        )
        with caplog.at_level("ERROR"):
            result = policy.destroy(_candidate())
        assert result.verdict == CleanupVerdict.UNKNOWN

    def test_destroy_fn_raises_substitutes_unknown(self, caplog) -> None:
        def raises(c: InstanceCandidate) -> CleanupResult:
            raise RuntimeError("kaboom")

        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=lambda: [],
            destroy_fn=raises,
        )
        with caplog.at_level("ERROR"):
            result = policy.destroy(_candidate())
        assert result.verdict == CleanupVerdict.UNKNOWN
        assert "RuntimeError" in result.error
        assert "kaboom" in result.error

    def test_destroy_fn_returns_valid_result(self) -> None:
        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=lambda: [],
            destroy_fn=lambda c: CleanupResult(verdict=CleanupVerdict.LEAKED, error="resurrected"),
        )
        result = policy.destroy(_candidate())
        assert result.verdict == CleanupVerdict.LEAKED
        assert result.error == "resurrected"


# ---------------------------------------------------------------------------
# normalize_instance_id — shared between enum and verify
# ---------------------------------------------------------------------------


class TestNormalizeInstanceId:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, None),
            (True, None),
            (False, None),
            ("", None),
            ("   ", None),
            (" 123 ", "123"),
            (123, "123"),
            ("abc", "abc"),
            ("  abc  ", "abc"),
        ],
    )
    def test_returns_canonical_or_none(self, raw: object, expected: str | None) -> None:
        assert normalize_instance_id(raw) == expected

    def test_non_str_non_int_returns_none(self) -> None:
        assert normalize_instance_id([1, 2]) is None
        assert normalize_instance_id({"id": "1"}) is None
        assert normalize_instance_id(1.5) is None


# ---------------------------------------------------------------------------
# OwnershipVerification — tagged enum contract
# ---------------------------------------------------------------------------


class TestOwnershipVerification:
    def test_has_four_members(self) -> None:
        assert {m.value for m in OwnershipVerification} == {
            "owned",
            "absent",
            "refused",
            "disabled",
        }


# ---------------------------------------------------------------------------
# cleanup_policy imports no provider modules (orchestrator invariant)
# ---------------------------------------------------------------------------


class TestCleanupPolicyModuleImports:
    def test_no_provider_imports(self) -> None:
        from vastai_gpu_runner import cleanup_policy

        source = inspect.getsource(cleanup_policy)
        # Forbidden: any import that would couple the core module to a
        # provider module. The v4 doc's invariant: imports nothing
        # from ``providers/``.
        assert "from vastai_gpu_runner.providers" not in source
        assert "import vastai_gpu_runner.providers" not in source
