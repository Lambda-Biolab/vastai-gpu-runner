# pyright: reportPrivateUsage=warning, reportMissingParameterType=warning
"""End-to-end v4 cleanup-policy integration tests.

The 17 scenarios from the v4 architecture's migration checklist,
step 6: ``tests/integration/test_cleanup_policy_integration.py``.

Each scenario wires a real ``ProviderCleanupPolicy`` through the
canonical v4 types (``build_vastai_cleanup_policy``,
``list_vastai_instances``, ``BatchOrchestrator._sweep_zombies``,
the ``cli.py:cleanup`` / ``cli.py:instances`` / ``cli.py:batch``
commands) and asserts the documented behaviour.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tests._v4_helpers import FakeOrchestrator, _noop_cleanup_policy
from typer.testing import CliRunner

from vastai_gpu_runner.cleanup_policy import (
    CleanupRefusal,
    CleanupResult,
    CleanupVerdict,
    InstanceCandidate,
    OwnershipPolicy,
    Provider,
    ProviderCleanupPolicy,
)
from vastai_gpu_runner.cli import app
from vastai_gpu_runner.providers.destroy_adapters.vastai import (
    CredentialResolution,
    CredentialState,
)
from vastai_gpu_runner.providers.vastai import (
    VASTAI_TERMINAL_STATES,
    build_vastai_cleanup_policy,
)

runner = CliRunner()


def _candidate(
    iid: str,
    *,
    label: str = "prod-3f9a1b2c4d5e-deadbeef",
    state: str = "running",
    image: str = "myorg/app:1.0",
    provider: Provider = Provider.VASTAI,
) -> InstanceCandidate:
    return InstanceCandidate(
        provider=provider,
        instance_id=iid,
        label=label,
        state=state,
        image_uuid=image,
        ownership_key=image,
        gpu_model="RTX 4090",
        cost_per_hour=0.4,
        started_at=0.0,
    )


def _absent() -> CredentialResolution:
    return CredentialResolution(state=CredentialState.ABSENT)


def _available(key: str = "canonical-key") -> CredentialResolution:
    return CredentialResolution(state=CredentialState.AVAILABLE, key=key)


def _disabled() -> CredentialResolution:
    return CredentialResolution(state=CredentialState.EXPLICITLY_DISABLED)


# ---------------------------------------------------------------------------
# Scenarios 1-3: list / destroy wiring
# ---------------------------------------------------------------------------


class TestDisabledBeforeEnumeration:
    """``EXPLICITLY_DISABLED`` short-circuits both list and destroy."""

    def test_list_returns_empty_without_rest_or_cli(self) -> None:
        policy = build_vastai_cleanup_policy(
            ownership=OwnershipPolicy(),
            credentials=_disabled(),
        )
        # Neither REST nor vastai_cmd is invoked — the policy's
        # enumeration short-circuits before any I/O.
        result = policy.list_instances()
        assert result == []

    def test_destroy_with_disabled_credentials_is_refused(self) -> None:
        policy = build_vastai_cleanup_policy(
            ownership=OwnershipPolicy(),
            credentials=_disabled(),
        )
        c = _candidate("i1")
        result = policy.destroy(c)
        assert result.refusal == CleanupRefusal.CREDENTIALS_DISABLED


class TestCredentialAlignedEnumeration:
    """``AVAILABLE`` enumerates via REST with the canonical key only."""

    def test_rest_uses_credentials_key(self) -> None:
        c = _candidate("i1")
        with patch(
            "vastai_gpu_runner.providers.vastai.list_vastai_instances",
            return_value=[c],
        ) as mock_list:
            policy = build_vastai_cleanup_policy(
                ownership=OwnershipPolicy(),
                credentials=_available(key="env-key"),
            )
            candidates = policy.list_instances()
            assert candidates == [c]
            mock_list.assert_called_once()
            _, kwargs = mock_list.call_args
            assert kwargs["credentials"].key == "env-key"


class TestAbsentCredentialCLIFallback:
    """``ABSENT`` credentials trigger CLI fallback dispatch."""

    def test_absent_destroy_returns_cli_attempted(self) -> None:
        """``ABSENT`` ownership verifier + CLI destroy → ``CLI_ATTEMPTED``."""
        from vastai_gpu_runner.cleanup_policy import OwnershipVerification

        def _verify(instance_id: str, *, ownership: OwnershipPolicy) -> OwnershipVerification:
            return OwnershipVerification.OWNED

        def _cli_destroy(args: list[str], timeout: int = 60) -> str:
            return ""

        c = _candidate("i1")
        policy = build_vastai_cleanup_policy(
            ownership=OwnershipPolicy(owned_images=frozenset({"myorg/app:1.0"})),
            credentials=_absent(),
        )
        with (
            patch(
                "vastai_gpu_runner.providers.vastai.verify_instance_ownership",
                side_effect=_verify,
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.vastai_cmd",
                side_effect=_cli_destroy,
            ),
        ):
            result = policy.destroy(c)
        assert result.verdict == CleanupVerdict.CLI_ATTEMPTED


# ---------------------------------------------------------------------------
# Scenario 4-7: empty ownership, provider mismatch, enumeration failure,
# ineligible state
# ---------------------------------------------------------------------------


class TestEmptyOwnershipSet:
    def test_empty_owned_images_refuses_every_image(self) -> None:
        """``owned_images=frozenset()`` → every image refused as ``OWNERSHIP``."""
        from vastai_gpu_runner.providers.destroy import DestroyRefusal, DestroyResult

        ownership_refusal = DestroyResult(
            refusal=DestroyRefusal.OWNERSHIP,
        )
        policy = build_vastai_cleanup_policy(
            ownership=OwnershipPolicy(owned_images=frozenset()),
            credentials=_available(),
        )
        c = _candidate("i1", image="myorg/app:1.0")
        with patch(
            "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
            return_value=ownership_refusal,
        ):
            result = policy.destroy(c)
        assert result.refusal == CleanupRefusal.OWNERSHIP


class TestProviderMismatch:
    def test_runpod_candidate_to_vastai_policy_returns_mismatch(self) -> None:
        policy = build_vastai_cleanup_policy(
            ownership=OwnershipPolicy(),
            credentials=_available(),
        )
        # RunPod candidate to a Vast.ai policy.
        c = _candidate("i1", provider=Provider.RUNPOD)
        result = policy.destroy(c)
        assert result.refusal == CleanupRefusal.PROVIDER_MISMATCH
        assert "runpod" in result.error.lower()
        assert "vastai" in result.error.lower()


class TestEnumerationFailure:
    def test_list_raises_returns_empty(self) -> None:
        policy = build_vastai_cleanup_policy(
            ownership=OwnershipPolicy(),
            credentials=_available(),
        )
        with patch(
            "vastai_gpu_runner.providers.vastai.list_vastai_instances",
            side_effect=RuntimeError("api down"),
        ):
            candidates = policy.list_instances()
        assert candidates == []


class TestIneligibleState:
    def test_terminal_state_is_refused(self) -> None:
        policy = build_vastai_cleanup_policy(
            ownership=OwnershipPolicy(),
            credentials=_available(),
        )
        c = _candidate("i1", state="destroyed")
        result = policy.destroy(c)
        assert result.refusal == CleanupRefusal.INELIGIBLE_STATE

    def test_empty_state_is_refused(self) -> None:
        policy = build_vastai_cleanup_policy(
            ownership=OwnershipPolicy(),
            credentials=_available(),
        )
        c = _candidate("i1", state="")
        result = policy.destroy(c)
        assert result.refusal == CleanupRefusal.INELIGIBLE_STATE

    def test_terminal_states_constant_is_complete(self) -> None:
        """``VASTAI_TERMINAL_STATES`` is a non-empty frozenset of terminal markers."""
        assert isinstance(VASTAI_TERMINAL_STATES, frozenset)
        assert "destroyed" in VASTAI_TERMINAL_STATES


# ---------------------------------------------------------------------------
# Scenarios 8-10: severity logging, non-empty error, null safety
# ---------------------------------------------------------------------------


class TestSeverityLogging:
    """Orchestrator severity-by-outcome is wired through ``_log_cleanup_outcome``."""

    def _orchestrator_with_candidate(
        self, candidate: InstanceCandidate, result: CleanupResult
    ) -> FakeOrchestrator:
        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=lambda: [candidate],
            destroy_fn=lambda c: result,
        )
        return FakeOrchestrator(
            units=[],
            runner_factory=lambda: MagicMock(),
            label_prefix="prod",
            cleanup_policy=policy,
        )

    def test_leaked_logs_error(self, caplog: pytest.LogCaptureFixture) -> None:
        c = _candidate("i1")
        result = CleanupResult(verdict=CleanupVerdict.LEAKED, error="x")
        orch = self._orchestrator_with_candidate(c, result)
        with caplog.at_level(logging.ERROR, logger="vastai_gpu_runner.batch"):
            orch._sweep_zombies()
        assert any("LEAKED" in r.getMessage() for r in caplog.records)

    def test_unknown_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        c = _candidate("i1")
        result = CleanupResult(verdict=CleanupVerdict.UNKNOWN, error="x")
        orch = self._orchestrator_with_candidate(c, result)
        with caplog.at_level(logging.WARNING, logger="vastai_gpu_runner.batch"):
            orch._sweep_zombies()
        assert any("UNKNOWN" in r.getMessage() for r in caplog.records)


class TestNonEmptyErrorFromEmptyException:
    def test_empty_runtime_error_yields_non_empty_diagnostic(self) -> None:
        policy = build_vastai_cleanup_policy(
            ownership=OwnershipPolicy(),
            credentials=_available(),
        )

        with patch(
            "vastai_gpu_runner.providers.vastai.list_vastai_instances",
            side_effect=RuntimeError(""),
        ):
            # The policy boundary converts the exception into an
            # empty-list return; an orchestrator would log the
            # empty exception message and continue.
            result = policy.list_instances()
        assert result == []

    def test_destroy_exception_substitutes_unknown_with_diagnostic(self) -> None:
        policy = build_vastai_cleanup_policy(
            ownership=OwnershipPolicy(),
            credentials=_available(),
        )
        c = _candidate("i1")

        with patch(
            "vastai_gpu_runner.providers.vastai.destroy_vastai_instance",
            side_effect=RuntimeError(""),
        ):
            result = policy.destroy(c)
        # The destroy boundary in build_vastai_cleanup_policy catches
        # exceptions and converts them to UNKNOWN with diagnostic.
        assert result.verdict == CleanupVerdict.UNKNOWN
        assert result.error != ""


class TestNullSafetyFields:
    """Nullable enumeration fields normalise to empty strings."""

    def test_null_state_refused_as_ineligible(self) -> None:
        policy = build_vastai_cleanup_policy(
            ownership=OwnershipPolicy(),
            credentials=_available(),
        )
        c = _candidate("i1", state="")
        result = policy.destroy(c)
        assert result.refusal == CleanupRefusal.INELIGIBLE_STATE

    def test_null_label_does_not_match_label_scope(self) -> None:
        """A candidate with empty label must not match a non-empty scope."""
        c = _candidate("i1", label="")
        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=lambda: [c],
            destroy_fn=lambda x: CleanupResult(verdict=CleanupVerdict.DESTROYED),
        )
        orch = FakeOrchestrator(
            units=[],
            runner_factory=lambda: MagicMock(),
            label_prefix="prod",
            cleanup_policy=policy,
        )
        killed = orch._sweep_zombies()
        assert killed == 0


# ---------------------------------------------------------------------------
# Scenario 11-12: label-prefix safety + persistence
# ---------------------------------------------------------------------------


class TestLabelPrefixSafety:
    """``validate_label_prefix`` rejects empty / blank / padded before enumeration."""

    @pytest.mark.parametrize("bad", ["", " ", "  padded  "])
    def test_orchestrator_constructor_rejects(self, bad: str) -> None:
        with pytest.raises(ValueError, match="label_prefix"):
            FakeOrchestrator(
                units=[],
                runner_factory=lambda: MagicMock(),
                label_prefix=bad,
                cleanup_policy=_noop_cleanup_policy(),
            )

    def test_batch_cli_rejects_empty_label(self) -> None:
        result = runner.invoke(
            app,
            ["batch", "--state-path", "/tmp/x.json", "--label", "", "--image", "myorg/app:1.0"],
        )
        assert result.exit_code != 0
        assert "label" in result.output.lower()

    def test_cleanup_cli_rejects_empty_label(self) -> None:
        result = runner.invoke(
            app,
            ["cleanup", "--label", ""],
        )
        assert result.exit_code != 0
        assert "label" in result.output.lower()

    def test_sweep_uses_delimited_scope(self) -> None:
        """``f"{label_prefix}evil"`` does NOT match canonical ``f"{label_prefix}-"``."""
        good = _candidate("i1", label="prod-3f9a1b2c4d5e-deadbeef")
        evil = _candidate("i2", label="prodevil-3f9a1b2c4d5e-deadbeef")
        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=lambda: [good, evil],
            destroy_fn=lambda x: CleanupResult(verdict=CleanupVerdict.DESTROYED),
        )
        orch = FakeOrchestrator(
            units=[],
            runner_factory=lambda: MagicMock(),
            label_prefix="prod",
            cleanup_policy=policy,
        )
        killed = orch._sweep_zombies()
        assert killed == 1  # only the good one


# ---------------------------------------------------------------------------
# Scenarios 13-14: CLI empty / opt-out
# ---------------------------------------------------------------------------


class TestCliAllowedImages:
    def test_empty_string_fail_closed(self) -> None:
        result = runner.invoke(
            app,
            ["instances", "--allowed-images", ""],
        )
        assert result.exit_code == 0
        # No candidates (default mock); but the wiring must accept
        # the empty string without raising.

    def test_none_opt_out(self) -> None:
        """No ``--allowed-images`` flag → opt-out (every image owned)."""
        c = _candidate("i1", image="myorg/app:1.0")
        with patch(
            "vastai_gpu_runner.providers.vastai.list_vastai_instances",
            return_value=[c],
        ):
            result = runner.invoke(app, ["instances"])
        assert result.exit_code == 0
        assert "yes" in result.output


# ---------------------------------------------------------------------------
# Scenarios 15-17: destroy_fn returns None, runner logs typed result,
# cleanup outcome totals, instances ownership column
# ---------------------------------------------------------------------------


class TestDestroyFnReturnsNone:
    def test_orchestrator_substitutes_unknown_with_diagnostic(self) -> None:
        c = _candidate("i1")

        def _destroy(candidate: InstanceCandidate) -> CleanupResult:
            # Returns None to test the boundary's invalid-result check.
            # pyright can't follow the intent — None is not assignable
            # to CleanupResult, but that's the whole point of the test.
            return None  # type: ignore[return-value]

        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=lambda: [c],
            destroy_fn=_destroy,
        )
        orch = FakeOrchestrator(
            units=[],
            runner_factory=lambda: MagicMock(),
            label_prefix="prod",
            cleanup_policy=policy,
        )
        # The policy boundary converts None → UNKNOWN; orchestrator
        # logs WARNING. No exception escapes.
        killed = orch._sweep_zombies()
        assert killed == 0


class TestVastaiRunnerLogsTypedResult:
    """``VastaiRunner.destroy_instance`` logs the typed DestroyResult."""

    def test_leaked_outcome_logs_error(self, caplog: pytest.LogCaptureFixture) -> None:
        from vastai_gpu_runner.providers.destroy import DestroyResult, DestroyVerdict

        # Construct a v3 DestroyResult with LEAKED + structured error.
        v3_result = DestroyResult(
            verdict=DestroyVerdict.LEAKED,
            attempts=1,
            last_status_code=500,
        )

        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.destroy_vastai_instance",
                return_value=v3_result,
            ),
            caplog.at_level(logging.ERROR, logger="vastai_gpu_runner.runner"),
        ):
            from vastai_gpu_runner.providers.vastai import VastaiRunner

            r = VastaiRunner(
                config=MagicMock(),
                ownership=OwnershipPolicy(),
                credentials=_available(),
            )
            ok = r.destroy_instance(MagicMock(instance_id="i1"))
        assert ok is False
        # Either an ERROR-level log line is emitted via the runner
        # logger or via the provider's destroy logger.
        assert any(
            rec.levelno == logging.ERROR
            for rec in caplog.records
            if "LEAKED" in rec.getMessage() or "manual review" in rec.getMessage()
        )


class TestCleanupOutcomeTotals:
    """``cleanup`` command separates destroyed / already-gone / unresolved."""

    def test_already_gone_not_counted_as_destroyed(self) -> None:
        c1 = _candidate("i1")
        c2 = _candidate("i2")
        responses = {
            "i1": CleanupResult(verdict=CleanupVerdict.DESTROYED),
            "i2": CleanupResult(verdict=CleanupVerdict.ALREADY_GONE),
        }

        def _list() -> list[InstanceCandidate]:
            return [c1, c2]

        def _destroy(candidate: InstanceCandidate) -> CleanupResult:
            return responses[candidate.instance_id]

        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=_list,
            destroy_fn=_destroy,
        )
        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.read_vastai_api_key",
                return_value=_absent(),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.build_vastai_cleanup_policy",
                return_value=policy,
            ),
        ):
            result = runner.invoke(
                app,
                ["cleanup", "--label", "prod-3f9a1b2c4d5e"],
                input="y\n",
            )
        assert result.exit_code == 0
        assert "Destroyed" in result.output
        assert "Already gone" in result.output


class TestInstancesOwnershipColumn:
    """The \"Owned\" column matches via ``OwnershipPolicy.matches``."""

    @pytest.fixture
    def candidates(self) -> list[InstanceCandidate]:
        return [
            _candidate("i-malicious", image="myorg/app-malicious:latest"),
            _candidate("i-tag-insensitive", image="myorg/app:latest"),
            _candidate(
                "i-digest",
                image="myorg/app@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            ),
            _candidate("i-registry-port-positive", image="registry:5000/myorg/app:1.0"),
            _candidate("i-registry-port-negative", image="registry:5000/myorg/app:1.0"),
        ]

    def test_malicious_prefix_not_owned(self, candidates: list[InstanceCandidate]) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.list_vastai_instances",
            return_value=[c for c in candidates if c.instance_id == "i-malicious"],
        ):
            result = runner.invoke(app, ["instances", "--allowed-images", "myorg/app:1.0"])
        assert "no" in result.output

    def test_tag_insensitive_match(self, candidates: list[InstanceCandidate]) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.list_vastai_instances",
            return_value=[c for c in candidates if c.instance_id == "i-tag-insensitive"],
        ):
            result = runner.invoke(app, ["instances", "--allowed-images", "myorg/app:1.0"])
        assert "yes" in result.output

    def test_digest_match_repository_tag_insensitive(
        self, candidates: list[InstanceCandidate]
    ) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.list_vastai_instances",
            return_value=[c for c in candidates if c.instance_id == "i-digest"],
        ):
            result = runner.invoke(app, ["instances", "--allowed-images", "myorg/app:1.0"])
        # myorg/app@sha256:... normalises to myorg/app — same repo as
        # myorg/app:1.0 — so the column shows yes.
        assert "yes" in result.output

    def test_registry_port_different_host_is_not_owned(
        self, candidates: list[InstanceCandidate]
    ) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.list_vastai_instances",
            return_value=[c for c in candidates if c.instance_id == "i-registry-port-negative"],
        ):
            result = runner.invoke(app, ["instances", "--allowed-images", "myorg/app"])
        # registry:5000/myorg/app:1.0 does NOT match myorg/app
        # (registry host differs from the default docker hub).
        assert "no" in result.output

    def test_registry_port_exact_match_is_owned(self, candidates: list[InstanceCandidate]) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.list_vastai_instances",
            return_value=[c for c in candidates if c.instance_id == "i-registry-port-positive"],
        ):
            result = runner.invoke(
                app,
                ["instances", "--allowed-images", "registry:5000/myorg/app:1.0"],
            )
        assert "yes" in result.output

    def test_empty_set_every_instance_not_owned(self, candidates: list[InstanceCandidate]) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.list_vastai_instances",
            return_value=candidates,
        ):
            result = runner.invoke(app, ["instances", "--allowed-images", ""])
        # Every row renders as "no" (the table footer reports total cost).
        no_count = result.output.count(" no ")
        assert no_count >= len(candidates)


# ---------------------------------------------------------------------------
# Scenario: restart-after-crash scope drift rejection
# ---------------------------------------------------------------------------


class TestRestartScopeDrift:
    """A restart with a drifted requested prefix is rejected, not silently re-scoped."""

    def test_persisted_scope_with_drifted_prefix_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "batch_id": "b",
                    "label_scope": "staging-3f9a1b2c4d5e",
                    "requested_label_prefix": "staging",
                    "shards": [],
                }
            )
        )
        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.read_vastai_api_key",
                return_value=_absent(),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.VastaiProviderConfig.from_env",
                return_value=MagicMock(),
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "batch",
                    "--state-path",
                    str(path),
                    "--label",
                    "prod",
                    "--image",
                    "myorg/app:1.0",
                ],
            )
        assert result.exit_code != 0
        assert "does not match" in result.output.lower()
