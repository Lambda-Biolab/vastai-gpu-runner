# pyright: reportPrivateUsage=warning, reportMissingParameterType=warning, reportUnusedFunction=false, reportUnusedClass=false
"""Tests for providers.vastai.verify_instance_ownership — CLI-side ownership check.

Per docs/architecture-v4-cleanup-policy.md migration step 3c.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from vastai_gpu_runner.cleanup_policy import OwnershipPolicy, OwnershipVerification
from vastai_gpu_runner.providers.vastai import verify_instance_ownership


def _inst(
    id_: object = 123,
    image_uuid: object = "myorg/app:1.0",
) -> dict[str, object]:
    return {"id": id_, "image_uuid": image_uuid, "actual_status": "running"}


def _cli(records: list[object]) -> str:
    return json.dumps(records)


class TestVerifyOwnershipDisabled:
    def test_disabled_when_owned_images_is_none(self) -> None:
        # DISABLED short-circuits the API call.
        with patch("vastai_gpu_runner.providers.vastai.vastai_cmd") as mock:
            result = verify_instance_ownership("123", ownership=OwnershipPolicy(owned_images=None))
        assert result == OwnershipVerification.DISABLED
        mock.assert_not_called()


class TestVerifyOwnershipCanonicalization:
    @pytest.mark.parametrize(
        "raw_id",
        ["123", 123, " 123 ", " 123"],
    )
    def test_canonical_target_matches_all_id_forms(self, raw_id: object) -> None:
        """Padded, numeric, and string IDs reduce to the same canonical form."""
        with patch("vastai_gpu_runner.providers.vastai.vastai_cmd") as mock:
            mock.return_value = _cli([_inst(123)])
            result = verify_instance_ownership(
                raw_id,  # type: ignore[arg-type]
                ownership=OwnershipPolicy(owned_images=frozenset({"myorg/app:1.0"})),
            )
        assert result == OwnershipVerification.OWNED

    @pytest.mark.parametrize(
        "bad_id",
        [None, True, False, "", "   ", [1, 2], {"id": "1"}],
    )
    def test_refused_on_invalid_target_id(self, bad_id: object) -> None:
        with patch("vastai_gpu_runner.providers.vastai.vastai_cmd") as mock:
            mock.return_value = _cli([_inst(123)])
            result = verify_instance_ownership(
                bad_id,  # type: ignore[arg-type]
                ownership=OwnershipPolicy(owned_images=frozenset({"myorg/app:1.0"})),
            )
        assert result == OwnershipVerification.REFUSED


class TestVerifyOwnershipOutcomes:
    def _policy(self) -> OwnershipPolicy:
        return OwnershipPolicy(owned_images=frozenset({"myorg/app:1.0"}))

    def test_owned_when_image_matches(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai.vastai_cmd") as mock:
            mock.return_value = _cli([_inst()])
            result = verify_instance_ownership("123", ownership=self._policy())
        assert result == OwnershipVerification.OWNED

    def test_refused_on_image_mismatch(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai.vastai_cmd") as mock:
            mock.return_value = _cli([_inst(image_uuid="malicious/app:1.0")])
            result = verify_instance_ownership("123", ownership=self._policy())
        assert result == OwnershipVerification.REFUSED

    def test_absent_when_target_not_in_response(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai.vastai_cmd") as mock:
            mock.return_value = _cli([_inst(id_=999)])
            result = verify_instance_ownership("123", ownership=self._policy())
        assert result == OwnershipVerification.ABSENT

    def test_refused_on_empty_allowlist_with_matching_instance(self) -> None:
        # Empty owned_images = "destroy nothing owned" — every check fails.
        # The CLI IS called to distinguish OWNED-but-mismatched from ABSENT.
        with patch("vastai_gpu_runner.providers.vastai.vastai_cmd") as mock:
            mock.return_value = _cli([_inst()])
            result = verify_instance_ownership(
                "123", ownership=OwnershipPolicy(owned_images=frozenset())
            )
        assert result == OwnershipVerification.REFUSED

    def test_absent_on_empty_allowlist_with_no_target_record(self) -> None:
        # Empty owned_images + target not in response → ABSENT
        # (verifier proved absence from a well-formed response).
        with patch("vastai_gpu_runner.providers.vastai.vastai_cmd") as mock:
            mock.return_value = _cli([_inst(id_=999)])
            result = verify_instance_ownership(
                "123", ownership=OwnershipPolicy(owned_images=frozenset())
            )
        assert result == OwnershipVerification.ABSENT


class TestVerifyOwnershipResponseShape:
    def _policy(self) -> OwnershipPolicy:
        return OwnershipPolicy(owned_images=frozenset({"myorg/app:1.0"}))

    def test_refused_on_cli_runtime_error(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.vastai_cmd",
            side_effect=RuntimeError("CLI broken"),
        ):
            result = verify_instance_ownership("123", ownership=self._policy())
        assert result == OwnershipVerification.REFUSED

    def test_refused_on_invalid_json(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.vastai_cmd",
            return_value="not json",
        ):
            result = verify_instance_ownership("123", ownership=self._policy())
        assert result == OwnershipVerification.REFUSED

    def test_refused_on_non_list_response(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.vastai_cmd",
            return_value='{"not": "a list"}',
        ):
            result = verify_instance_ownership("123", ownership=self._policy())
        assert result == OwnershipVerification.REFUSED

    def test_refused_on_non_dict_record(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.vastai_cmd",
            return_value=_cli(["garbage", _inst(123)]),
        ):
            result = verify_instance_ownership("123", ownership=self._policy())
        assert result == OwnershipVerification.REFUSED

    def test_refused_on_missing_id(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.vastai_cmd",
            return_value=_cli([{"image_uuid": "myorg/app:1.0"}]),
        ):
            result = verify_instance_ownership("123", ownership=self._policy())
        assert result == OwnershipVerification.REFUSED

    def test_refused_on_null_id(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.vastai_cmd",
            return_value=_cli([{"id": None, "image_uuid": "myorg/app:1.0"}]),
        ):
            result = verify_instance_ownership("123", ownership=self._policy())
        assert result == OwnershipVerification.REFUSED

    @pytest.mark.parametrize(
        "bad_uuid",
        [None, 123, 1.5, True, ["x"], {"y": 1}],
    )
    def test_refused_on_non_string_image_uuid(self, bad_uuid: object) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.vastai_cmd",
            return_value=_cli([_inst(image_uuid=bad_uuid)]),
        ):
            result = verify_instance_ownership("123", ownership=self._policy())
        assert result == OwnershipVerification.REFUSED

    def test_refused_on_duplicate_canonical_ids(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.vastai_cmd",
            return_value=_cli([_inst(id_="123"), _inst(id_="123")]),
        ):
            result = verify_instance_ownership("123", ownership=self._policy())
        assert result == OwnershipVerification.REFUSED

    def test_refused_on_duplicate_across_padding_forms(self) -> None:
        # "123" and " 123 " are the same canonical ID; duplicate → REFUSED.
        with patch(
            "vastai_gpu_runner.providers.vastai.vastai_cmd",
            return_value=_cli([_inst(id_="123"), _inst(id_=" 123 ")]),
        ):
            result = verify_instance_ownership("123", ownership=self._policy())
        assert result == OwnershipVerification.REFUSED

    def test_refused_on_malformed_record_after_matching(self) -> None:
        # Even when the matching record is found, a malformed entry
        # elsewhere in the response fails the entire check.
        with patch(
            "vastai_gpu_runner.providers.vastai.vastai_cmd",
            return_value=_cli([_inst(123), "garbage"]),
        ):
            result = verify_instance_ownership("123", ownership=self._policy())
        assert result == OwnershipVerification.REFUSED

    def test_refused_on_malformed_record_before_matching(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.vastai_cmd",
            return_value=_cli(["garbage", _inst(123)]),
        ):
            result = verify_instance_ownership("123", ownership=self._policy())
        assert result == OwnershipVerification.REFUSED


class TestVerifyOwnershipOuterBoundary:
    """The outer ``except Exception`` prevents any leak from the function."""

    def test_unexpected_exception_becomes_refused(self) -> None:
        policy = OwnershipPolicy(owned_images=frozenset({"myorg/app:1.0"}))
        # Replace one of the inner helpers with a raiser.
        with (
            patch(
                "vastai_gpu_runner.providers.vastai._eval_ownership_match",
                side_effect=RuntimeError("kaboom"),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.vastai_cmd",
                return_value=_cli([_inst()]),
            ),
        ):
            result = verify_instance_ownership("123", ownership=policy)
        assert result == OwnershipVerification.REFUSED
