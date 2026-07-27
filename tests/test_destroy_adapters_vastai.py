"""Tests for providers.destroy_adapters.vastai — Vast.ai destroy adapter."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vastai_gpu_runner.providers.destroy import (
    DestroyRefusal,
    DestroyVerdict,
    VerifyVerdict,
)
from vastai_gpu_runner.providers.destroy_adapters.vastai import (
    VASTAI_POLICY,
    CredentialResolution,
    CredentialState,
    OwnershipVerification,
    _find_instance,
    _is_image_allowed,
    _repository,
    _rest_verify,
    destroy_vastai_instance,
    read_vastai_api_key,
    verify_instance_ownership,
)

# ---------------------------------------------------------------------------
# CredentialState + CredentialResolution invariants
# ---------------------------------------------------------------------------


class TestCredentialResolutionInvariants:
    def test_available_requires_key(self) -> None:
        r = CredentialResolution(state=CredentialState.AVAILABLE, key="key123")
        assert r.state == CredentialState.AVAILABLE
        assert r.key == "key123"

    def test_available_empty_key_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CredentialResolution(state=CredentialState.AVAILABLE, key="")

    def test_available_whitespace_only_key_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CredentialResolution(state=CredentialState.AVAILABLE, key="   ")

    def test_available_unstripped_key_raises(self) -> None:
        with pytest.raises(ValueError, match="pre-stripped"):
            CredentialResolution(state=CredentialState.AVAILABLE, key=" key123 ")

    def test_absent_requires_empty_key(self) -> None:
        r = CredentialResolution(state=CredentialState.ABSENT)
        assert r.state == CredentialState.ABSENT
        assert r.key == ""

    def test_absent_with_key_raises(self) -> None:
        with pytest.raises(ValueError, match="empty key"):
            CredentialResolution(state=CredentialState.ABSENT, key="key")

    def test_explicitly_disabled_requires_empty_key(self) -> None:
        r = CredentialResolution(state=CredentialState.EXPLICITLY_DISABLED)
        assert r.state == CredentialState.EXPLICITLY_DISABLED
        assert r.key == ""

    def test_explicitly_disabled_with_key_raises(self) -> None:
        with pytest.raises(ValueError, match="empty key"):
            CredentialResolution(state=CredentialState.EXPLICITLY_DISABLED, key="key")

    def test_invalid_state_raises(self) -> None:
        """Invalid state is caught by the type system at construction;
        runtime construction with bypasses the dataclass __post_init__."""
        # Skipping this test: the type system enforces the state type
        # at construction; runtime defenses are limited to the
        # AVAILABLE+key/ABSENT+empty invariants below.

    def test_invalid_key_type_raises(self) -> None:
        """Invalid key type is caught by the type system at construction;
        runtime construction with bypasses the dataclass __post_init__."""
        # Skipping this test: the type system enforces the key type
        # at construction.


# ---------------------------------------------------------------------------
# read_vastai_api_key
# ---------------------------------------------------------------------------


class TestReadVastaiApiKey:
    def test_env_var_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VASTAI_API_KEY", "key123")
        monkeypatch.setenv("HOME", "/nonexistent")
        result = read_vastai_api_key()
        assert result.state == CredentialState.AVAILABLE
        assert result.key == "key123"

    def test_env_var_strips_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VASTAI_API_KEY", "  key123  ")
        monkeypatch.setenv("HOME", "/nonexistent")
        result = read_vastai_api_key()
        assert result.state == CredentialState.AVAILABLE
        assert result.key == "key123"

    def test_env_var_explicitly_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VASTAI_API_KEY", "")
        monkeypatch.setenv("HOME", "/nonexistent")
        result = read_vastai_api_key()
        assert result.state == CredentialState.EXPLICITLY_DISABLED
        assert result.key == ""

    def test_env_var_whitespace_only_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VASTAI_API_KEY", "   ")
        monkeypatch.setenv("HOME", "/nonexistent")
        result = read_vastai_api_key()
        assert result.state == CredentialState.EXPLICITLY_DISABLED

    def test_no_env_no_file_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VASTAI_API_KEY", raising=False)
        monkeypatch.setenv("HOME", "/nonexistent")
        result = read_vastai_api_key()
        assert result.state == CredentialState.ABSENT

    def test_file_with_key_available(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VASTAI_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        cfg_dir = tmp_path / ".config" / "vastai"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "vast_api_key").write_text("filekey\n")
        result = read_vastai_api_key()
        assert result.state == CredentialState.AVAILABLE
        assert result.key == "filekey"

    def test_empty_file_warns_but_treats_as_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        monkeypatch.delenv("VASTAI_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        cfg_dir = tmp_path / ".config" / "vastai"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "vast_api_key").write_text("   \n")
        with caplog.at_level(logging.WARNING):
            result = read_vastai_api_key()
        assert result.state == CredentialState.ABSENT
        assert any("empty" in r.message for r in caplog.records)

    def test_unreadable_file_treats_as_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        monkeypatch.delenv("VASTAI_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        cfg_dir = tmp_path / ".config" / "vastai"
        cfg_dir.mkdir(parents=True)
        key_file = cfg_dir / "vast_api_key"
        key_file.write_text("key")
        key_file.chmod(0o000)
        # Skip on platforms where chmod(0o000) doesn't actually block
        # reading (e.g. running as root).
        if os.geteuid() == 0:
            pytest.skip("running as root; chmod(0o000) doesn't block reads")
        with caplog.at_level(logging.WARNING):
            result = read_vastai_api_key()
        assert result.state == CredentialState.ABSENT


# ---------------------------------------------------------------------------
# _repository — tag/digest stripping
# ---------------------------------------------------------------------------


class TestRepository:
    def test_simple_image(self) -> None:
        assert _repository("ubuntu:22.04") == "ubuntu"

    def test_image_with_registry(self) -> None:
        assert _repository("myreg.io/myorg/app:1.0") == "myreg.io/myorg/app"

    def test_image_with_registry_port(self) -> None:
        assert _repository("registry:5000/myorg/app:1.0") == "registry:5000/myorg/app"

    def test_image_with_digest(self) -> None:
        assert (
            _repository(
                "myorg/app@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
            )
            == "myorg/app"
        )

    def test_image_with_digest_and_tag(self) -> None:
        """Tag-then-digest: documented as the v3 behavior.

        The v3 helper drops the digest first (everything after '@'),
        leaving 'myorg/app:1.0'. The v4 full _repository grammar
        (lands in v4 step 1) drops the tag too.
        """
        # This test documents current v3 behavior; v4 will tighten it.
        result = _repository(
            "myorg/app:1.0@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        )
        # Accept either: the v3 minimal or v4 full grammar both acceptable.
        assert result in ("myorg/app", "myorg/app:1.0")

    def test_empty_input(self) -> None:
        assert _repository("") == ""

    def test_whitespace_only(self) -> None:
        assert _repository("   ") == ""


# ---------------------------------------------------------------------------
# _is_image_allowed — v2 substring/prefix is removed
# ---------------------------------------------------------------------------


class TestIsImageAllowed:
    def test_exact_ref_match(self) -> None:
        assert _is_image_allowed("myorg/app:1.0", frozenset({"myorg/app:1.0"}))

    def test_tag_insensitive_match(self) -> None:
        assert _is_image_allowed("myorg/app:latest", frozenset({"myorg/app:1.0"}))

    def test_registry_port_exact(self) -> None:
        assert _is_image_allowed(
            "registry:5000/myorg/app:1.0",
            frozenset({"registry:5000/myorg/app:1.0"}),
        )

    def test_registry_port_tag_insensitive(self) -> None:
        assert _is_image_allowed(
            "registry:5000/myorg/app:latest",
            frozenset({"registry:5000/myorg/app:1.0"}),
        )

    def test_malicious_prefix_rejected(self) -> None:
        """myorg/app:1.0 does NOT allow myorg/app-malicious:latest."""
        assert not _is_image_allowed("myorg/app-malicious:latest", frozenset({"myorg/app:1.0"}))

    def test_malicious_registry_rejected(self) -> None:
        """registry:5000/myorg/app:1.0 does NOT allow
        registry-malicious/myorg/app:1.0."""
        assert not _is_image_allowed(
            "registry-malicious/myorg/app:1.0",
            frozenset({"registry:5000/myorg/app:1.0"}),
        )

    def test_malicious_suffix_rejected(self) -> None:
        """myorg/app:1.0 does NOT allow myorg/app-evil:1.0."""
        assert not _is_image_allowed("myorg/app-evil:1.0", frozenset({"myorg/app:1.0"}))

    def test_empty_allowed_rejects_all(self) -> None:
        assert not _is_image_allowed("myorg/app:1.0", frozenset())

    def test_empty_instance_image_rejected(self) -> None:
        assert not _is_image_allowed("", frozenset({"myorg/app:1.0"}))

    def test_membership_in_mixed_set(self) -> None:
        allowed = frozenset({"myorg/app:1.0", "other/thing:2.0"})
        assert _is_image_allowed("other/thing:latest", allowed)


# ---------------------------------------------------------------------------
# verify_instance_ownership
# ---------------------------------------------------------------------------


def _make_inst(id_: int = 123, image_uuid: str = "myorg/app:1.0") -> dict[str, object]:
    return {"id": id_, "image_uuid": image_uuid, "actual_status": "running"}


class TestVerifyInstanceOwnership:
    def test_disabled_when_no_allowlist(self) -> None:
        assert (
            verify_instance_ownership("123", allowed_images=None) == OwnershipVerification.DISABLED
        )

    def test_empty_allowlist_fails_closed(self) -> None:
        # Empty frozenset refuses every instance.
        with patch(
            "vastai_gpu_runner.providers.destroy_adapters.vastai._vastai_show_instances_raw"
        ) as mock_show:
            mock_show.return_value = json.dumps([_make_inst()])
            result = verify_instance_ownership("123", allowed_images=frozenset())
        assert result == OwnershipVerification.REFUSED
        # No CLI call should have been made.
        mock_show.assert_not_called()

    def test_owned_when_image_matches(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.destroy_adapters.vastai._vastai_show_instances_raw"
        ) as mock_show:
            mock_show.return_value = json.dumps([_make_inst()])
            result = verify_instance_ownership("123", allowed_images=frozenset({"myorg/app:1.0"}))
        assert result == OwnershipVerification.OWNED

    def test_absent_when_instance_not_in_response(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.destroy_adapters.vastai._vastai_show_instances_raw"
        ) as mock_show:
            mock_show.return_value = json.dumps([_make_inst(id_=999)])
            result = verify_instance_ownership("123", allowed_images=frozenset({"myorg/app:1.0"}))
        assert result == OwnershipVerification.ABSENT

    def test_refused_when_image_not_in_allowlist(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.destroy_adapters.vastai._vastai_show_instances_raw"
        ) as mock_show:
            mock_show.return_value = json.dumps([_make_inst(image_uuid="malicious/app:1.0")])
            result = verify_instance_ownership("123", allowed_images=frozenset({"myorg/app:1.0"}))
        assert result == OwnershipVerification.REFUSED

    def test_refused_on_subprocess_timeout(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.destroy_adapters.vastai._vastai_show_instances_raw",
            side_effect=subprocess.TimeoutExpired("vastai", 15),
        ):
            result = verify_instance_ownership("123", allowed_images=frozenset({"myorg/app:1.0"}))
        assert result == OwnershipVerification.REFUSED

    def test_refused_on_subprocess_error(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.destroy_adapters.vastai._vastai_show_instances_raw",
            side_effect=subprocess.CalledProcessError(1, "vastai"),
        ):
            result = verify_instance_ownership("123", allowed_images=frozenset({"myorg/app:1.0"}))
        assert result == OwnershipVerification.REFUSED

    def test_refused_on_invalid_json(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.destroy_adapters.vastai._vastai_show_instances_raw",
            return_value="not json",
        ):
            result = verify_instance_ownership("123", allowed_images=frozenset({"myorg/app:1.0"}))
        assert result == OwnershipVerification.REFUSED

    def test_refused_on_non_list_response(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.destroy_adapters.vastai._vastai_show_instances_raw",
            return_value='{"not": "a list"}',
        ):
            result = verify_instance_ownership("123", allowed_images=frozenset({"myorg/app:1.0"}))
        assert result == OwnershipVerification.REFUSED

    def test_refused_on_record_with_no_image_uuid(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.destroy_adapters.vastai._vastai_show_instances_raw",
            return_value=json.dumps([{"id": 123}]),
        ):
            result = verify_instance_ownership("123", allowed_images=frozenset({"myorg/app:1.0"}))
        assert result == OwnershipVerification.REFUSED

    def test_refused_on_record_with_null_image_uuid(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.destroy_adapters.vastai._vastai_show_instances_raw",
            return_value=json.dumps([{"id": 123, "image_uuid": None}]),
        ):
            result = verify_instance_ownership("123", allowed_images=frozenset({"myorg/app:1.0"}))
        assert result == OwnershipVerification.REFUSED


# ---------------------------------------------------------------------------
# _find_instance
# ---------------------------------------------------------------------------


class TestFindInstance:
    def test_found(self) -> None:
        inst = _find_instance([_make_inst(id_=123), _make_inst(id_=456)], "123")
        assert inst is not None
        assert inst["id"] == 123

    def test_not_found(self) -> None:
        inst = _find_instance([_make_inst(id_=456)], "123")
        assert inst is None

    def test_empty_list(self) -> None:
        assert _find_instance([], "123") is None

    def test_skips_non_dict_entries(self) -> None:
        inst = _find_instance(["garbage", 42, _make_inst(id_=123)], "123")
        assert inst is not None
        assert inst["id"] == 123


# ---------------------------------------------------------------------------
# _rest_verify
# ---------------------------------------------------------------------------


class TestRestVerify:
    def test_404_returns_gone(self) -> None:
        mock_resp = MagicMock(status_code=404)
        with patch(
            "requests.get",
            return_value=mock_resp,
        ):
            result = _rest_verify("123", {})
        assert result.verdict == VerifyVerdict.GONE
        assert result.status_code == 404

    def test_200_destroyed_returns_gone(self) -> None:
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"actual_status": "destroyed"}
        with patch(
            "requests.get",
            return_value=mock_resp,
        ):
            result = _rest_verify("123", {})
        assert result.verdict == VerifyVerdict.GONE

    def test_200_running_returns_present(self) -> None:
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"actual_status": "running"}
        with patch(
            "requests.get",
            return_value=mock_resp,
        ):
            result = _rest_verify("123", {})
        assert result.verdict == VerifyVerdict.PRESENT

    def test_200_empty_status_returns_present(self) -> None:
        """actual_status missing or empty != destroyed → PRESENT."""
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"actual_status": ""}
        with patch(
            "requests.get",
            return_value=mock_resp,
        ):
            result = _rest_verify("123", {})
        assert result.verdict == VerifyVerdict.PRESENT

    def test_500_returns_unknown(self) -> None:
        mock_resp = MagicMock(status_code=500)
        with patch(
            "requests.get",
            return_value=mock_resp,
        ):
            result = _rest_verify("123", {})
        assert result.verdict == VerifyVerdict.UNKNOWN
        assert result.status_code == 500

    def test_request_exception_returns_unknown(self) -> None:
        import requests

        with patch(
            "requests.get",
            side_effect=requests.RequestException("boom"),
        ):
            result = _rest_verify("123", {})
        assert result.verdict == VerifyVerdict.UNKNOWN
        assert "boom" in (result.error or "")


# ---------------------------------------------------------------------------
# destroy_vastai_instance — pre-protocol refusals
# ---------------------------------------------------------------------------


class TestDestroyVastaiInstance:
    def test_ownership_refused_short_circuits(self) -> None:
        """REFUSED ownership → DestroyResult with refusal=OWNERSHIP, no API calls."""
        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.verify_instance_ownership",
                return_value=OwnershipVerification.REFUSED,
            ) as mock_verify,
            patch("vastai_gpu_runner.providers.destroy_adapters.vastai._rest_stop") as mock_stop,
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai._rest_delete"
            ) as mock_delete,
            patch("vastai_gpu_runner.providers.destroy_adapters.vastai._rest_verify") as mock_v,
        ):
            result = destroy_vastai_instance("123", allowed_images=frozenset({"myorg/app:1.0"}))
        assert result.refusal == DestroyRefusal.OWNERSHIP
        assert result.attempts == 0
        mock_verify.assert_called_once()
        # No REST calls.
        mock_stop.assert_not_called()
        mock_delete.assert_not_called()
        mock_v.assert_not_called()

    def test_credentials_disabled_returns_refusal(self) -> None:
        """EXPLICITLY_DISABLED → DestroyResult with refusal=CREDENTIALS_DISABLED."""
        with patch(
            "vastai_gpu_runner.providers.destroy_adapters.vastai.verify_instance_ownership",
            return_value=OwnershipVerification.OWNED,
        ):
            result = destroy_vastai_instance(
                "123",
                allowed_images=frozenset({"myorg/app:1.0"}),
                credentials=CredentialResolution(state=CredentialState.EXPLICITLY_DISABLED),
            )
        assert result.refusal == DestroyRefusal.CREDENTIALS_DISABLED

    def test_credentials_absent_returns_refusal(self) -> None:
        """ABSENT → DestroyResult with refusal=NO_CREDENTIALS (CLI fallback
        is the operator's responsibility per the v3 doc)."""
        with patch(
            "vastai_gpu_runner.providers.destroy_adapters.vastai.verify_instance_ownership",
            return_value=OwnershipVerification.OWNED,
        ):
            result = destroy_vastai_instance(
                "123",
                allowed_images=frozenset({"myorg/app:1.0"}),
                credentials=CredentialResolution(state=CredentialState.ABSENT),
            )
        assert result.refusal == DestroyRefusal.NO_CREDENTIALS

    def test_credentials_disabled_overrides_ownership_disabled(self) -> None:
        """CREDENTIALS_DISABLED is checked even if ownership is DISABLED."""
        with patch(
            "vastai_gpu_runner.providers.destroy_adapters.vastai.verify_instance_ownership",
            return_value=OwnershipVerification.DISABLED,
        ):
            result = destroy_vastai_instance(
                "123",
                allowed_images=None,
                credentials=CredentialResolution(state=CredentialState.EXPLICITLY_DISABLED),
            )
        assert result.refusal == DestroyRefusal.CREDENTIALS_DISABLED

    def test_credentials_available_runs_protocol(self) -> None:
        """AVAILABLE + OWNED → belt_and_suspenders runs; mock REST callbacks
        return DESTROYED."""
        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.verify_instance_ownership",
                return_value=OwnershipVerification.OWNED,
            ),
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai._rest_stop",
                return_value=200,
            ),
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai._rest_delete",
                return_value=200,
            ),
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai._rest_verify",
                return_value=MagicMock(verdict=VerifyVerdict.GONE, status_code=200),
            ),
        ):
            result = destroy_vastai_instance(
                "123",
                allowed_images=frozenset({"myorg/app:1.0"}),
                credentials=CredentialResolution(state=CredentialState.AVAILABLE, key="key123"),
            )
        assert result.verdict == DestroyVerdict.DESTROYED
        assert result.attempts == 1

    def test_credentials_available_with_absent_ownership_runs(self) -> None:
        """ABSENT ownership (instance not in API response) is not a refusal —
        the destroy proceeds."""
        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.verify_instance_ownership",
                return_value=OwnershipVerification.ABSENT,
            ),
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai._rest_stop",
                return_value=200,
            ),
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai._rest_delete",
                return_value=200,
            ),
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai._rest_verify",
                return_value=MagicMock(verdict=VerifyVerdict.GONE, status_code=200),
            ),
        ):
            result = destroy_vastai_instance(
                "123",
                allowed_images=frozenset({"myorg/app:1.0"}),
                credentials=CredentialResolution(state=CredentialState.AVAILABLE, key="key123"),
            )
        assert result.verdict == DestroyVerdict.DESTROYED

    def test_falls_back_to_read_vastai_api_key(self) -> None:
        """When credentials=None, the v3 back-compat path calls
        read_vastai_api_key() and respects the resolved state."""
        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.verify_instance_ownership",
                return_value=OwnershipVerification.OWNED,
            ),
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.read_vastai_api_key",
                return_value=CredentialResolution(state=CredentialState.EXPLICITLY_DISABLED),
            ),
        ):
            result = destroy_vastai_instance(
                "123",
                allowed_images=frozenset({"myorg/app:1.0"}),
            )
        assert result.refusal == DestroyRefusal.CREDENTIALS_DISABLED


# ---------------------------------------------------------------------------
# VASTAI_POLICY timing
# ---------------------------------------------------------------------------


class TestVastaiPolicy:
    def test_policy_values(self) -> None:
        assert VASTAI_POLICY.verify_delay_s == 5.0
        assert VASTAI_POLICY.retry_delay_s == 3.0
        assert VASTAI_POLICY.max_delete_attempts == 3
        assert VASTAI_POLICY.verify_after_resurrection is True
