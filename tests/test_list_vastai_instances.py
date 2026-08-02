# pyright: reportPrivateUsage=warning, reportMissingParameterType=warning
"""Tests for providers.vastai.list_vastai_instances — credential-aware enumeration.

Per docs/architecture-v4-cleanup-policy.md migration step 3
(``list_vastai_instances`` + REST/CLI dispatch).
"""

# ruff: noqa: S105, S106  (test scaffolding uses literal token fixtures)

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from vastai_gpu_runner.cleanup_policy import InstanceCandidate
from vastai_gpu_runner.providers.destroy_adapters.vastai import (
    CredentialResolution,
    CredentialState,
)
from vastai_gpu_runner.providers.vastai import (
    VASTAI_INSTANCES_PAGE_SIZE,
    VASTAI_INSTANCES_URL,
    _float_or_zero,
    _list_vastai_instances_rest,
    _string_or_empty,
    list_vastai_instances,
)
from vastai_gpu_runner.types import Provider

# ---------------------------------------------------------------------------
# _string_or_empty + _float_or_zero helpers
# ---------------------------------------------------------------------------


class TestStringOrEmpty:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("abc", "abc"),
            ("", ""),
            (None, ""),
            (123, ""),
            (1.5, ""),
            (True, ""),
            (False, ""),
            ([], ""),
            ({}, ""),
        ],
    )
    def test_normalises(self, value: object, expected: str) -> None:
        assert _string_or_empty(value) == expected


class TestFloatOrZero:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, 0.0),
            (1.5, 1.5),
            (123, 123.0),
            (None, 0.0),
            ("", 0.0),
            ("abc", 0.0),
            ([], 0.0),
            (True, 0.0),  # booleans rejected
            (False, 0.0),
        ],
    )
    def test_normalises(self, value: object, expected: float) -> None:
        assert _float_or_zero(value) == expected


# ---------------------------------------------------------------------------
# _list_vastai_instances_rest — pagination
# ---------------------------------------------------------------------------


def _response(
    *,
    instances: list[Any],
    next_token: str | None,
    status_code: int = 200,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"instances": instances, "next_token": next_token})
    return resp


class TestListVastaiInstancesRest:
    def test_single_page_no_next_token(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai.requests.get") as get_mock:
            get_mock.return_value = _response(
                instances=[{"id": "1"}, {"id": "2"}],
                next_token=None,
            )
            result = _list_vastai_instances_rest("canonical-key")
        assert result == [{"id": "1"}, {"id": "2"}]
        get_mock.assert_called_once()
        # Verify the Bearer key is in the Authorization header
        _, kwargs = get_mock.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer canonical-key"

    def test_pagination_collects_all_pages(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai.requests.get") as get_mock:
            get_mock.side_effect = [
                _response(
                    instances=[{"id": "1"}, {"id": "2"}],
                    next_token="tok-1",
                ),
                _response(
                    instances=[{"id": "3"}],
                    next_token=None,
                ),
            ]
            result = _list_vastai_instances_rest("canonical-key")
        assert result == [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        assert get_mock.call_count == 2
        # Both pages used the exact Bearer key
        for call in get_mock.call_args_list:
            assert call.kwargs["headers"]["Authorization"] == "Bearer canonical-key"

    def test_sends_pagination_after_token(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai.requests.get") as get_mock:
            get_mock.side_effect = [
                _response(instances=[{"id": "1"}], next_token="tok-1"),
                _response(instances=[{"id": "2"}], next_token=None),
            ]
            _list_vastai_instances_rest("canonical-key")
        # First call: no after_token
        first_params = get_mock.call_args_list[0].kwargs["params"]
        assert "after_token" not in first_params
        assert first_params["limit"] == VASTAI_INSTANCES_PAGE_SIZE
        # Second call: has after_token
        second_params = get_mock.call_args_list[1].kwargs["params"]
        assert second_params["after_token"] == "tok-1"

    def test_non_object_payload_raises(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai.requests.get") as get_mock:
            get_mock.return_value.json.return_value = ["not", "an", "object"]
            with pytest.raises(ValueError, match="must be an object"):
                _list_vastai_instances_rest("canonical-key")

    def test_non_list_instances_raises(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai.requests.get") as get_mock:
            get_mock.return_value.json.return_value = {
                "instances": "not-a-list",
                "next_token": None,
            }
            with pytest.raises(ValueError, match="no list-valued"):
                _list_vastai_instances_rest("canonical-key")

    def test_empty_next_token_raises(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai.requests.get") as get_mock:
            get_mock.return_value.json.return_value = {
                "instances": [{"id": "1"}],
                "next_token": "",
            }
            with pytest.raises(ValueError, match="invalid pagination token"):
                _list_vastai_instances_rest("canonical-key")

    def test_repeated_next_token_raises(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai.requests.get") as get_mock:
            get_mock.side_effect = [
                _response(instances=[{"id": "1"}], next_token="tok-1"),
                _response(instances=[{"id": "2"}], next_token="tok-1"),
            ]
            with pytest.raises(ValueError, match="invalid pagination token"):
                _list_vastai_instances_rest("canonical-key")

    def test_page_limit_exceeded_raises(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai.requests.get") as get_mock:
            # Returns a unique next_token each time so the
            # "repeated token" check doesn't fire first.
            counter = {"n": 0}

            def unique_token_resp(*_args: object, **_kwargs: object) -> MagicMock:
                counter["n"] += 1
                return _response(
                    instances=[{"id": str(counter["n"])}],
                    next_token=f"tok-{counter['n']}",
                )

            get_mock.side_effect = unique_token_resp
            with pytest.raises(ValueError, match="pagination exceeded"):
                _list_vastai_instances_rest("canonical-key")

    def test_http_error_propagates(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai.requests.get") as get_mock:
            get_mock.return_value.raise_for_status.side_effect = requests.HTTPError(
                "500 Server Error"
            )
            with pytest.raises(requests.HTTPError):
                _list_vastai_instances_rest("canonical-key")


# ---------------------------------------------------------------------------
# list_vastai_instances — credential dispatch
# ---------------------------------------------------------------------------


def _inst(
    instance_id: object = "123",
    *,
    image_uuid: object = "img-uuid",
    label: object = "prod",
    actual_status: object = "running",
    gpu_name: object = "RTX 4090",
    dph_total: object = 0.5,
    start_date: object = 1.0,
) -> dict[str, object]:
    return {
        "id": instance_id,
        "image_uuid": image_uuid,
        "label": label,
        "actual_status": actual_status,
        "gpu_name": gpu_name,
        "dph_total": dph_total,
        "start_date": start_date,
    }


class TestListVastaiInstancesDispatch:
    def test_explicitly_disabled_returns_empty_without_any_call(self) -> None:
        creds = CredentialResolution(state=CredentialState.EXPLICITLY_DISABLED)
        with (
            patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_rest") as rest_mock,
            patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_cli") as cli_mock,
        ):
            result = list_vastai_instances(credentials=creds)
        assert result == []
        rest_mock.assert_not_called()
        cli_mock.assert_not_called()

    def test_available_uses_rest_with_exact_key(self) -> None:
        creds = CredentialResolution(state=CredentialState.AVAILABLE, key="canonical-key")
        with (
            patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_rest") as rest_mock,
            patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_cli") as cli_mock,
            patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_cli"),
        ):
            rest_mock.return_value = [_inst("123")]
            result = list_vastai_instances(credentials=creds)
        assert len(result) == 1
        rest_mock.assert_called_once_with("canonical-key")
        cli_mock.assert_not_called()

    def test_absent_uses_cli_not_rest(self) -> None:
        creds = CredentialResolution(state=CredentialState.ABSENT)
        with (
            patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_rest") as rest_mock,
            patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_cli") as cli_mock,
        ):
            cli_mock.return_value = [_inst("123")]
            result = list_vastai_instances(credentials=creds)
        assert len(result) == 1
        cli_mock.assert_called_once()
        rest_mock.assert_not_called()


class TestListVastaiInstancesRecordValidation:
    def _creds(self) -> CredentialResolution:
        return CredentialResolution(state=CredentialState.ABSENT)

    def test_skips_non_object_records(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_cli") as cli_mock:
            cli_mock.return_value = [_inst("123"), "bad", ["also", "bad"], 42]
            result = list_vastai_instances(credentials=self._creds())
        assert len(result) == 1
        assert result[0].instance_id == "123"

    @pytest.mark.parametrize(
        "bad_id",
        [None, True, False, "", "   ", [1, 2], {"id": 1}],
    )
    def test_skips_invalid_ids(self, bad_id: object) -> None:
        with patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_cli") as cli_mock:
            cli_mock.return_value = [
                _inst(bad_id),
                _inst("123"),  # one valid record
            ]
            result = list_vastai_instances(credentials=self._creds())
        assert len(result) == 1
        assert result[0].instance_id == "123"

    def test_canonicalises_padded_ids(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_cli") as cli_mock:
            cli_mock.return_value = [_inst(" 123 ")]
            result = list_vastai_instances(credentials=self._creds())
        assert len(result) == 1
        assert result[0].instance_id == "123"

    def test_canonicalises_integer_ids(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_cli") as cli_mock:
            cli_mock.return_value = [_inst(456)]
            result = list_vastai_instances(credentials=self._creds())
        assert len(result) == 1
        assert result[0].instance_id == "456"

    @pytest.mark.parametrize(
        ("source_field", "candidate_attr"),
        [("image_uuid", "image_uuid"), ("label", "label"), ("actual_status", "state")],
    )
    def test_null_string_fields_normalise_to_empty(
        self, source_field: str, candidate_attr: str
    ) -> None:
        kwargs = {"image_uuid": "img", "label": "prod", "actual_status": "running"}
        kwargs[source_field] = None  # type: ignore[assignment]
        with patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_cli") as cli_mock:
            cli_mock.return_value = [_inst(**kwargs)]
            result = list_vastai_instances(credentials=self._creds())
        assert len(result) == 1
        assert getattr(result[0], candidate_attr) == ""

    def test_duplicate_canonical_id_within_page_returns_empty(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_cli") as cli_mock:
            cli_mock.return_value = [_inst("123"), _inst("123")]
            result = list_vastai_instances(credentials=self._creds())
        assert result == []

    def test_padded_and_unpadded_id_are_same_canonical(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_cli") as cli_mock:
            cli_mock.return_value = [_inst("123"), _inst(" 123 ")]
            result = list_vastai_instances(credentials=self._creds())
        assert result == []

    def test_returns_empty_on_enumeration_exception(self) -> None:
        creds = CredentialResolution(state=CredentialState.ABSENT)
        with patch(
            "vastai_gpu_runner.providers.vastai._list_vastai_instances_cli",
            side_effect=RuntimeError("CLI broken"),
        ):
            result = list_vastai_instances(credentials=creds)
        assert result == []

    def test_rest_error_returns_empty(self) -> None:
        creds = CredentialResolution(state=CredentialState.AVAILABLE, key="key")
        with patch(
            "vastai_gpu_runner.providers.vastai._list_vastai_instances_rest",
            side_effect=RuntimeError("REST broken"),
        ):
            result = list_vastai_instances(credentials=creds)
        assert result == []

    def test_populates_all_fields(self) -> None:
        with patch("vastai_gpu_runner.providers.vastai._list_vastai_instances_cli") as cli_mock:
            cli_mock.return_value = [
                _inst(
                    "123",
                    image_uuid="img-uuid",
                    label="prod",
                    actual_status="running",
                    gpu_name="RTX 4090",
                    dph_total=0.45,
                    start_date=1.5,
                )
            ]
            result = list_vastai_instances(
                credentials=CredentialResolution(state=CredentialState.ABSENT)
            )
        assert len(result) == 1
        c: InstanceCandidate = result[0]
        assert c.provider == Provider.VASTAI
        assert c.instance_id == "123"
        assert c.image_uuid == "img-uuid"
        assert c.ownership_key == "img-uuid"
        assert c.gpu_model == "RTX 4090"
        assert c.cost_per_hour == 0.45
        assert c.label == "prod"
        assert c.state == "running"
        assert c.started_at == 1.5


class TestRestUrlConstant:
    def test_url_is_official(self) -> None:
        assert VASTAI_INSTANCES_URL == "https://console.vast.ai/api/v1/instances"
