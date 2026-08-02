# pyright: reportPrivateUsage=warning, reportMissingParameterType=warning
"""Tests for providers.vastai.build_vastai_cleanup_policy — v4 factory.

Per docs/architecture-v4-cleanup-policy.md migration step 3d.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

from vastai_gpu_runner.cleanup_policy import (
    CleanupRefusal,
    CleanupVerdict,
    InstanceCandidate,
    OwnershipPolicy,
    OwnershipVerification,
    ProviderCleanupPolicy,
)
from vastai_gpu_runner.providers.destroy import (
    DestroyRefusal,
    DestroyResult,
    DestroyVerdict,
)
from vastai_gpu_runner.providers.destroy_adapters.vastai import (
    CredentialResolution,
    CredentialState,
)
from vastai_gpu_runner.providers.vastai import (
    VastaiRunner,
    _describe_destroy_result,
    build_vastai_cleanup_policy,
)
from vastai_gpu_runner.types import CloudInstance, Provider


def _candidate(
    instance_id: str = "123",
    *,
    image_uuid: str = "myorg/app:1.0",
    state: str = "running",
    label: str = "prod",
) -> InstanceCandidate:
    return InstanceCandidate(
        provider=Provider.VASTAI,
        instance_id=instance_id,
        label=label,
        state=state,
        image_uuid=image_uuid,
        ownership_key=image_uuid,
    )


def _ownership() -> OwnershipPolicy:
    return OwnershipPolicy(owned_images=frozenset({"myorg/app:1.0"}))


# ---------------------------------------------------------------------------
# _describe_destroy_result
# ---------------------------------------------------------------------------


class TestDescribeDestroyResult:
    def test_includes_all_structured_fields(self) -> None:
        result = DestroyResult(
            verdict=DestroyVerdict.LEAKED,
            attempts=2,
            stop_error=None,
            last_status_code=200,
            verify_error="",
        )
        text = _describe_destroy_result(result)
        assert "verdict=" in text
        assert "refusal=" in text
        assert "attempts=" in text
        assert "last_status_code=" in text
        assert "verify_error=" in text
        assert "stop_error=" in text

    def test_always_non_empty(self) -> None:
        # Even a minimum DestroyResult produces non-empty diagnostic.
        text = _describe_destroy_result(DestroyResult(verdict=DestroyVerdict.UNKNOWN))
        assert text


# ---------------------------------------------------------------------------
# build_vastai_cleanup_policy — wired callbacks
# ---------------------------------------------------------------------------


class TestBuildVastaiCleanupPolicy:
    def test_returns_provider_cleanup_policy_with_provider_vastai(self) -> None:
        policy = build_vastai_cleanup_policy(
            ownership=_ownership(),
            credentials=CredentialResolution(state=CredentialState.ABSENT),
        )
        assert isinstance(policy, ProviderCleanupPolicy)
        assert policy.provider == Provider.VASTAI

    def test_list_instances_delegates_to_list_vastai_instances(self) -> None:
        """EXPLICITLY_DISABLED: list returns [] without REST or CLI."""
        policy = build_vastai_cleanup_policy(
            ownership=_ownership(),
            credentials=CredentialResolution(state=CredentialState.EXPLICITLY_DISABLED),
        )
        with (
            patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_rest") as rest,
            patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_cli") as cli,
        ):
            result = policy.list_instances()
        assert result == []
        rest.assert_not_called()
        cli.assert_not_called()

    def test_list_instances_available_uses_rest_not_cli(self) -> None:
        policy = build_vastai_cleanup_policy(
            ownership=_ownership(),
            credentials=CredentialResolution(state=CredentialState.AVAILABLE, key="canonical-key"),
        )
        # REST returns raw dicts; list_vastai_instances constructs
        # InstanceCandidate objects from them.
        raw_record: dict[str, object] = {
            "id": "123",
            "image_uuid": "myorg/app:1.0",
            "label": "prod",
            "actual_status": "running",
        }
        with (
            patch.object(
                __import__(
                    "vastai_gpu_runner.providers.vastai",
                    fromlist=["_list_vastai_instances_rest"],
                ),
                "_list_vastai_instances_rest",
                return_value=[raw_record],
            ) as rest,
            patch.object(
                __import__(
                    "vastai_gpu_runner.providers.vastai",
                    fromlist=["_list_vastai_instances_cli"],
                ),
                "_list_vastai_instances_cli",
            ) as cli,
        ):
            result = policy.list_instances()
        assert len(result) == 1
        assert result[0].instance_id == "123"
        rest.assert_called_once_with("canonical-key")
        cli.assert_not_called()


# ---------------------------------------------------------------------------
# destroy_fn outcomes
# ---------------------------------------------------------------------------


class TestDestroyDispatchEligibility:
    def _policy(self) -> ProviderCleanupPolicy:
        return build_vastai_cleanup_policy(
            ownership=_ownership(),
            credentials=CredentialResolution(state=CredentialState.ABSENT),
        )

    def test_empty_state_returns_ineligible(self) -> None:
        policy = self._policy()
        result = policy.destroy(_candidate(state=""))
        assert result.refusal == CleanupRefusal.INELIGIBLE_STATE
        assert "empty state" in result.error

    def test_terminal_state_returns_ineligible(self) -> None:
        policy = self._policy()
        result = policy.destroy(_candidate(state="destroyed"))
        assert result.refusal == CleanupRefusal.INELIGIBLE_STATE
        assert "terminal" in result.error

    def test_credentials_disabled_returns_refusal(self) -> None:
        policy = build_vastai_cleanup_policy(
            ownership=_ownership(),
            credentials=CredentialResolution(state=CredentialState.EXPLICITLY_DISABLED),
        )
        result = policy.destroy(_candidate())
        assert result.refusal == CleanupRefusal.CREDENTIALS_DISABLED


class TestDestroyAdapterVerdictTranslation:
    def _policy(self, credentials: CredentialResolution | None = None) -> ProviderCleanupPolicy:
        return build_vastai_cleanup_policy(
            ownership=_ownership(),
            credentials=credentials
            or CredentialResolution(state=CredentialState.AVAILABLE, key="k"),
        )

    def test_destroyed_translates_to_verdict_destroyed_no_error(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
            return_value=DestroyResult(verdict=DestroyVerdict.DESTROYED, attempts=1),
        ):
            result = self._policy().destroy(_candidate())
        assert result.verdict == CleanupVerdict.DESTROYED
        assert result.error == ""

    def test_leaked_translates_with_diagnostic(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
            return_value=DestroyResult(
                verdict=DestroyVerdict.LEAKED,
                attempts=2,
                last_status_code=200,
            ),
        ):
            result = self._policy().destroy(_candidate())
        assert result.verdict == CleanupVerdict.LEAKED
        # Diagnostic is the structured v3 description (which includes
        # the verdict repr). The orchestrator dispatches by the
        # typed CleanupVerdict, not by string parsing.
        assert "attempts=" in result.error

    def test_unknown_translates_with_diagnostic(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
            return_value=DestroyResult(verdict=DestroyVerdict.UNKNOWN, attempts=0),
        ):
            result = self._policy().destroy(_candidate())
        assert result.verdict == CleanupVerdict.UNKNOWN
        assert result.error


class TestDestroyAdapterRefusalTranslation:
    def _policy(self) -> ProviderCleanupPolicy:
        return build_vastai_cleanup_policy(
            ownership=_ownership(),
            credentials=CredentialResolution(state=CredentialState.AVAILABLE, key="k"),
        )

    def test_ownership_refusal_translates(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
            return_value=DestroyResult(refusal=DestroyRefusal.OWNERSHIP),
        ):
            result = self._policy().destroy(_candidate())
        assert result.refusal == CleanupRefusal.OWNERSHIP
        assert "ownership" in result.error.lower()

    def test_credentials_disabled_refusal_translates(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
            return_value=DestroyResult(refusal=DestroyRefusal.CREDENTIALS_DISABLED),
        ):
            result = self._policy().destroy(_candidate())
        assert result.refusal == CleanupRefusal.CREDENTIALS_DISABLED
        # CONCERN 2: diagnostic includes the structured v3 description
        assert "VASTAI_API_KEY" in result.error


class TestCliFallbackDispatch:
    """``NO_CREDENTIALS`` from the v3 adapter triggers the CLI fallback path."""

    def _absent_policy(self) -> ProviderCleanupPolicy:
        return build_vastai_cleanup_policy(
            ownership=_ownership(),
            credentials=CredentialResolution(state=CredentialState.ABSENT),
        )

    def test_no_credentials_triggers_cli_fallback(self) -> None:
        with (
            patch(
                "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
                return_value=DestroyResult(refusal=DestroyRefusal.NO_CREDENTIALS),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.verify_instance_ownership",
                return_value=OwnershipVerification.OWNED,
            ) as verify,
            patch("vastai_gpu_runner.providers.vastai.vastai_cmd") as cmd,
        ):
            result = self._absent_policy().destroy(_candidate())
        assert result.verdict == CleanupVerdict.CLI_ATTEMPTED
        verify.assert_called_once()
        cmd.assert_called_once_with(["destroy", "instance", "123"], timeout=15)

    def test_cli_verifier_disabled_runs_cli_destroy(self) -> None:
        with (
            patch(
                "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
                return_value=DestroyResult(refusal=DestroyRefusal.NO_CREDENTIALS),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.verify_instance_ownership",
                return_value=OwnershipVerification.DISABLED,
            ),
            patch("vastai_gpu_runner.providers.vastai.vastai_cmd"),
        ):
            result = self._absent_policy().destroy(_candidate())
        assert result.verdict == CleanupVerdict.CLI_ATTEMPTED

    def test_cli_verifier_absent_short_circuits_to_already_gone(self) -> None:
        """ABSENT from CLI verifier → ALREADY_GONE without invoking CLI destroy."""
        with (
            patch(
                "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
                return_value=DestroyResult(refusal=DestroyRefusal.NO_CREDENTIALS),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.verify_instance_ownership",
                return_value=OwnershipVerification.ABSENT,
            ),
            patch("vastai_gpu_runner.providers.vastai.vastai_cmd") as cmd,
        ):
            result = self._absent_policy().destroy(_candidate())
        assert result.verdict == CleanupVerdict.ALREADY_GONE
        assert result.error == ""
        # CLI destroy must NOT be invoked (would otherwise invent UNKNOWN
        # from a 'not found' exit code).
        cmd.assert_not_called()

    def test_cli_verifier_refused_returns_ownership_refusal(self) -> None:
        with (
            patch(
                "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
                return_value=DestroyResult(refusal=DestroyRefusal.NO_CREDENTIALS),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.verify_instance_ownership",
                return_value=OwnershipVerification.REFUSED,
            ),
        ):
            result = self._absent_policy().destroy(_candidate())
        assert result.refusal == CleanupRefusal.OWNERSHIP

    def test_cli_verifier_unexpected_result_fails_closed(self) -> None:
        """Defensive: any non-tagged result fails closed as OWNERSHIP."""
        with (
            patch(
                "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
                return_value=DestroyResult(refusal=DestroyRefusal.NO_CREDENTIALS),
            ),
            # Pretend a stale boolean slipped through.
            patch(
                "vastai_gpu_runner.providers.vastai.verify_instance_ownership",
                return_value=True,  # type: ignore[return-value]
            ),
            patch("vastai_gpu_runner.providers.vastai.vastai_cmd") as cmd,
        ):
            result = self._absent_policy().destroy(_candidate())
        assert result.refusal == CleanupRefusal.OWNERSHIP
        cmd.assert_not_called()

    def test_cli_verify_raises_returns_unknown(self) -> None:
        with (
            patch(
                "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
                return_value=DestroyResult(refusal=DestroyRefusal.NO_CREDENTIALS),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.verify_instance_ownership",
                side_effect=RuntimeError("kaboom"),
            ),
        ):
            result = self._absent_policy().destroy(_candidate())
        assert result.verdict == CleanupVerdict.UNKNOWN
        assert "kaboom" in result.error

    def test_cli_destroy_raises_returns_unknown(self) -> None:
        with (
            patch(
                "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
                return_value=DestroyResult(refusal=DestroyRefusal.NO_CREDENTIALS),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.verify_instance_ownership",
                return_value=OwnershipVerification.OWNED,
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.vastai_cmd",
                side_effect=RuntimeError("CLI broken"),
            ),
        ):
            result = self._absent_policy().destroy(_candidate())
        assert result.verdict == CleanupVerdict.UNKNOWN
        assert "CLI broken" in result.error


# ---------------------------------------------------------------------------
# VastaiRunner.destroy_instance typed logging
# ---------------------------------------------------------------------------


class TestVastaiRunnerDestroyInstance:
    def _runner(self) -> VastaiRunner:
        from vastai_gpu_runner.providers.vastai import VastaiRunner

        return VastaiRunner(
            ownership=_ownership(),
            credentials=CredentialResolution(state=CredentialState.AVAILABLE, key="k"),
        )

    def _instance(self) -> CloudInstance:
        from vastai_gpu_runner.types import CloudInstance, InstanceStatus, Provider

        return CloudInstance(
            provider=Provider.VASTAI,
            instance_id="123",
            gpu_model="RTX 4090",
            cost_per_hour=0.5,
            status=InstanceStatus.RUNNING,
        )

    def test_destroyed_returns_true(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
            return_value=DestroyResult(verdict=DestroyVerdict.DESTROYED, attempts=1),
        ):
            assert self._runner().destroy_instance(self._instance()) is True

    def test_leaked_logs_error_and_returns_false(self, caplog) -> None:
        with (
            patch(
                "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
                return_value=DestroyResult(
                    verdict=DestroyVerdict.LEAKED,
                    attempts=2,
                    last_status_code=200,
                ),
            ),
            caplog.at_level(logging.ERROR, logger="vastai_gpu_runner.providers.vastai"),
        ):
            assert self._runner().destroy_instance(self._instance()) is False
        assert any("LEAKED" in r.message for r in caplog.records)

    def test_unknown_logs_warning_and_returns_false(self, caplog) -> None:
        with (
            patch(
                "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
                return_value=DestroyResult(verdict=DestroyVerdict.UNKNOWN, attempts=0),
            ),
            caplog.at_level(logging.WARNING, logger="vastai_gpu_runner.providers.vastai"),
        ):
            assert self._runner().destroy_instance(self._instance()) is False
        assert any("UNKNOWN" in r.message for r in caplog.records)

    def test_ownership_refusal_logs_error_and_returns_false(self, caplog) -> None:
        with (
            patch(
                "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
                return_value=DestroyResult(refusal=DestroyRefusal.OWNERSHIP),
            ),
            caplog.at_level(logging.ERROR, logger="vastai_gpu_runner.providers.vastai"),
        ):
            assert self._runner().destroy_instance(self._instance()) is False
        assert any("ownership" in r.message.lower() for r in caplog.records)

    def test_credentials_disabled_logs_error_and_returns_false(self, caplog) -> None:
        with (
            patch(
                "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
                return_value=DestroyResult(refusal=DestroyRefusal.CREDENTIALS_DISABLED),
            ),
            caplog.at_level(logging.ERROR, logger="vastai_gpu_runner.providers.vastai"),
        ):
            assert self._runner().destroy_instance(self._instance()) is False
        assert any("credentials disabled" in r.message for r in caplog.records)

    def test_no_credentials_returns_false(self) -> None:
        """The runner does NOT auto-invoke the CLI fallback (v4 factory owns it)."""
        with patch(
            "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
            return_value=DestroyResult(refusal=DestroyRefusal.NO_CREDENTIALS),
        ):
            assert self._runner().destroy_instance(self._instance()) is False
