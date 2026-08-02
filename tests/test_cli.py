# pyright: reportPrivateUsage=warning, reportMissingParameterType=warning, reportUnusedFunction=false, reportUnusedClass=false
"""Tests for the v4 CLI composition roots.

Pins the new ``--label`` validation, the empty / blank / padded
rejection, and the empty ``--allowed-images`` fail-closed semantic.
Uses ``typer.testing.CliRunner`` to exercise each command end-to-end
with mocked ``vastai_cmd`` and ``read_vastai_api_key`` boundaries.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from typer.testing import CliRunner

from vastai_gpu_runner.cleanup_policy import InstanceCandidate
from vastai_gpu_runner.cli import app
from vastai_gpu_runner.providers.destroy_adapters.vastai import (
    CredentialResolution,
    CredentialState,
)

runner = CliRunner()


def _absent_creds() -> CredentialResolution:
    return CredentialResolution(state=CredentialState.ABSENT)


def _available_creds(key: str = "canonical-key") -> CredentialResolution:
    return CredentialResolution(state=CredentialState.AVAILABLE, key=key)


# ---------------------------------------------------------------------------
# `instances` command — uses v4 OwnershipPolicy.matches()
# ---------------------------------------------------------------------------


class TestInstancesCommand:
    @staticmethod
    def _v4_candidate(iid: str, image: str, label: str = "scope-x") -> object:
        from vastai_gpu_runner.cleanup_policy import (
            InstanceCandidate,
            Provider,
        )

        return InstanceCandidate(
            provider=Provider.VASTAI,
            instance_id=iid,
            label=label,
            state="running",
            image_uuid=image,
            ownership_key=image,
            gpu_model="RTX 4090",
            cost_per_hour=0.4,
            started_at=0.0,
        )

    def test_no_allowed_images_renders_all_as_not_owned(self) -> None:
        """``None`` (opt-out) → every instance shown as owned."""
        c = self._v4_candidate("i1", "docker.io/myorg/app:1.0")
        with patch(
            "vastai_gpu_runner.providers.vastai.list_vastai_instances",
            return_value=[c],
        ):
            result = runner.invoke(app, ["instances"])
        assert result.exit_code == 0
        # No --allowed-images means opt-out: every image is owned.
        assert "yes" in result.output

    def test_provided_images_match(self) -> None:
        """``--allowed-images`` matches tag-insensitively by repository."""
        c = self._v4_candidate("i1", "myorg/app:1.0")
        with patch(
            "vastai_gpu_runner.providers.vastai.list_vastai_instances",
            return_value=[c],
        ):
            result = runner.invoke(
                app,
                ["instances", "--allowed-images", "myorg/app:latest"],
            )
        assert result.exit_code == 0
        assert "yes" in result.output

    def test_empty_allowed_images_renders_all_as_not_owned(self) -> None:
        """Empty string → fail-closed (every instance shown as not owned)."""
        c = self._v4_candidate("i1", "docker.io/myorg/app:1.0")
        with patch(
            "vastai_gpu_runner.providers.vastai.list_vastai_instances",
            return_value=[c],
        ):
            result = runner.invoke(
                app,
                ["instances", "--allowed-images", ""],
            )
        assert result.exit_code == 0
        assert "no" in result.output

    def test_malicious_prefix_does_not_match(self) -> None:
        """``myorg/app-malicious:latest`` is not owned by ``myorg/app:1.0``."""
        c = self._v4_candidate("i1", "myorg/app-malicious:latest")
        with patch(
            "vastai_gpu_runner.providers.vastai.list_vastai_instances",
            return_value=[c],
        ):
            result = runner.invoke(
                app,
                ["instances", "--allowed-images", "myorg/app:1.0"],
            )
        assert result.exit_code == 0
        assert "no" in result.output

    def test_tag_insensitive_match(self) -> None:
        """Different tags on the same repo match (repository-based)."""
        c = self._v4_candidate("i1", "myorg/app:latest")
        with patch(
            "vastai_gpu_runner.providers.vastai.list_vastai_instances",
            return_value=[c],
        ):
            result = runner.invoke(
                app,
                ["instances", "--allowed-images", "myorg/app:1.0"],
            )
        assert result.exit_code == 0
        assert "yes" in result.output

    def test_no_candidates(self) -> None:
        with patch(
            "vastai_gpu_runner.providers.vastai.list_vastai_instances",
            return_value=[],
        ):
            result = runner.invoke(app, ["instances"])
        assert result.exit_code == 0
        assert "No active instances" in result.output


# ---------------------------------------------------------------------------
# `cleanup` command — fail-closed empty, validates scope, refused outcomes
# ---------------------------------------------------------------------------


class TestCleanupCommand:
    @staticmethod
    def _v4_candidate(iid: str, label: str, image: str = "myorg/app:1.0") -> object:
        from vastai_gpu_runner.cleanup_policy import (
            InstanceCandidate,
            Provider,
        )

        return InstanceCandidate(
            provider=Provider.VASTAI,
            instance_id=iid,
            label=label,
            state="running",
            image_uuid=image,
            ownership_key=image,
            gpu_model="RTX 4090",
            cost_per_hour=0.4,
            started_at=0.0,
        )

    @staticmethod
    def _patched_policy(candidates: list[InstanceCandidate]) -> object:
        """Patch ``build_vastai_cleanup_policy`` so the candidates list is fed in."""
        from vastai_gpu_runner.cleanup_policy import (
            CleanupResult,
            CleanupVerdict,
            Provider,
            ProviderCleanupPolicy,
        )

        def _list() -> list[InstanceCandidate]:
            return list(candidates)

        def _destroy(candidate: InstanceCandidate) -> CleanupResult:
            return CleanupResult(verdict=CleanupVerdict.DESTROYED)

        return ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=_list,
            destroy_fn=_destroy,
        )

    def test_rejects_empty_label(self) -> None:
        result = runner.invoke(app, ["cleanup", "--label", ""])
        assert result.exit_code != 0
        assert "label" in result.output.lower()

    def test_rejects_blank_label(self) -> None:
        result = runner.invoke(app, ["cleanup", "--label", " "])
        assert result.exit_code != 0
        assert "label" in result.output.lower()

    def test_rejects_bare_prefix_without_hex_suffix(self) -> None:
        result = runner.invoke(app, ["cleanup", "--label", "prod"])
        assert result.exit_code != 0
        assert "label" in result.output.lower()

    def test_accepts_canonical_scope(self) -> None:
        c = self._v4_candidate("i1", "prod-3f9a1b2c4d5e-deadbeef")
        policy = self._patched_policy(cast("list[InstanceCandidate]", [c]))
        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.read_vastai_api_key",
                return_value=_absent_creds(),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.build_vastai_cleanup_policy",
                return_value=policy,
            ),
        ):
            result = runner.invoke(
                app,
                ["cleanup", "--label", "prod-3f9a1b2c4d5e", "--dry-run"],
            )
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_dry_run_does_not_destroy(self) -> None:
        c = self._v4_candidate("i1", "prod-3f9a1b2c4d5e-deadbeef")
        policy = self._patched_policy(cast("list[InstanceCandidate]", [c]))
        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.read_vastai_api_key",
                return_value=_absent_creds(),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.build_vastai_cleanup_policy",
                return_value=policy,
            ),
        ):
            result = runner.invoke(
                app,
                ["cleanup", "--label", "prod-3f9a1b2c4d5e", "--dry-run"],
            )
        assert "Dry run" in result.output

    def test_empty_allowed_images_refuses_everything(self) -> None:
        """Empty ``--allowed-images `` → fail-closed (every instance refused)."""
        c1 = self._v4_candidate("i1", "prod-3f9a1b2c4d5e-deadbeef")
        c2 = self._v4_candidate("i2", "prod-3f9a1b2c4d5e-cafebabe")
        from vastai_gpu_runner.cleanup_policy import (
            CleanupRefusal,
            CleanupResult,
            Provider,
            ProviderCleanupPolicy,
        )

        def _list() -> list[InstanceCandidate]:
            return cast("list[InstanceCandidate]", [c1, c2])

        def _destroy(candidate: InstanceCandidate) -> CleanupResult:
            # OwnershipPolicy with empty set refuses every image.
            return CleanupResult(
                refusal=CleanupRefusal.OWNERSHIP,
                error="not in empty owned_images set",
            )

        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=_list,
            destroy_fn=_destroy,
        )
        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.read_vastai_api_key",
                return_value=_absent_creds(),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.build_vastai_cleanup_policy",
                return_value=policy,
            ),
        ):
            # typer.confirm needs interactive input; pre-feed "y"
            result = runner.invoke(
                app,
                [
                    "cleanup",
                    "--label",
                    "prod-3f9a1b2c4d5e",
                    "--allowed-images",
                    "",
                ],
                input="y\n",
            )
        assert result.exit_code == 0
        assert "ownership" in result.output.lower()

    def test_no_candidates_prints_no_match(self) -> None:
        policy = self._patched_policy([])
        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.read_vastai_api_key",
                return_value=_absent_creds(),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.build_vastai_cleanup_policy",
                return_value=policy,
            ),
        ):
            result = runner.invoke(
                app,
                ["cleanup", "--label", "prod-3f9a1b2c4d5e"],
            )
        assert "No instances matching" in result.output

    def test_delimiter_safety(self) -> None:
        """``prod-3f9a1b2c4d5eevil-...`` does NOT match canonical scope ``prod-3f9a1b2c4d5e``."""
        evil = self._v4_candidate("i1", "prod-3f9a1b2c4d5eevil-abcdef012345")
        policy = self._patched_policy(cast("list[InstanceCandidate]", [evil]))
        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.read_vastai_api_key",
                return_value=_absent_creds(),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.build_vastai_cleanup_policy",
                return_value=policy,
            ),
        ):
            result = runner.invoke(
                app,
                ["cleanup", "--label", "prod-3f9a1b2c4d5e"],
            )
        assert "No instances matching" in result.output

    def test_adjacent_scopes_match_broadly(self) -> None:
        """``--allow-adjacent-scopes`` enables broad prefix matching (DANGEROUS)."""
        evil = self._v4_candidate("i1", "prod-3f9a1b2c4d5eevil-abcdef012345")
        policy = self._patched_policy(cast("list[InstanceCandidate]", [evil]))
        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.read_vastai_api_key",
                return_value=_absent_creds(),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.build_vastai_cleanup_policy",
                return_value=policy,
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "cleanup",
                    "--label",
                    "prod-3f9a1b2c4d5e",
                    "--allow-adjacent-scopes",
                    "--dry-run",
                ],
            )
        assert "Found 1 instance" in result.output

    def test_already_gone_reported_separately(self) -> None:
        """``ALREADY_GONE`` is reported separately from destroyed."""
        from vastai_gpu_runner.cleanup_policy import (
            CleanupResult,
            CleanupVerdict,
            Provider,
            ProviderCleanupPolicy,
        )

        c1 = self._v4_candidate("i1", "prod-3f9a1b2c4d5e-deadbeef")
        c2 = self._v4_candidate("i2", "prod-3f9a1b2c4d5e-cafebabe")
        responses = {
            "i1": CleanupResult(verdict=CleanupVerdict.DESTROYED),
            "i2": CleanupResult(verdict=CleanupVerdict.ALREADY_GONE),
        }

        def _list() -> list[InstanceCandidate]:
            return cast("list[InstanceCandidate]", [c1, c2])

        def _destroy(candidate: InstanceCandidate) -> CleanupResult:
            return responses[candidate.instance_id]  # type: ignore[attr-defined]

        policy = ProviderCleanupPolicy(
            provider=Provider.VASTAI,
            list_instances_fn=_list,
            destroy_fn=_destroy,
        )
        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.read_vastai_api_key",
                return_value=_absent_creds(),
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


# ---------------------------------------------------------------------------
# `batch` command — v4 state migration + canonical config
# ---------------------------------------------------------------------------


class TestBatchCommand:
    @staticmethod
    def _v4_state_path(tmp_path: Path, contents: dict[str, object]) -> Path:
        path = tmp_path / "batch_state.json"
        path.write_text(json.dumps(contents))
        return path

    def test_rejects_empty_label(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "batch",
                "--label",
                "",
                "--image",
                "myorg/app:1.0",
                "--state-path",
                str(tmp_path / "x.json"),
            ],
        )
        assert result.exit_code != 0
        assert "label" in result.output.lower()

    def test_rejects_blank_label(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "batch",
                "--label",
                " ",
                "--image",
                "myorg/app:1.0",
                "--state-path",
                str(tmp_path / "x.json"),
            ],
        )
        assert result.exit_code != 0
        assert "label" in result.output.lower()

    def test_rejects_padded_label(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "batch",
                "--label",
                " padded ",
                "--image",
                "myorg/app:1.0",
                "--state-path",
                str(tmp_path / "x.json"),
            ],
        )
        assert result.exit_code != 0
        assert "label" in result.output.lower()

    def test_new_identity_writes_scope(self, tmp_path: Path) -> None:
        path = tmp_path / "fresh.json"
        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.read_vastai_api_key",
                return_value=_absent_creds(),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.VastaiProviderConfig.from_env",
                return_value=_empty_vastai_config(),
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "batch",
                    "--label",
                    "prod",
                    "--image",
                    "myorg/app:1.0",
                    "--state-path",
                    str(path),
                ],
            )
        assert result.exit_code == 0
        # The state file is written with a fresh scope.
        state = json.loads(path.read_text())
        assert state["label_scope"].startswith("prod-")
        assert state["requested_label_prefix"] == "prod"
        assert state["schema_version"] == 1

    def test_reuses_persisted_scope(self, tmp_path: Path) -> None:
        path = self._v4_state_path(
            tmp_path,
            {
                "schema_version": 1,
                "batch_id": "b",
                "label_scope": "prod-3f9a1b2c4d5e",
                "requested_label_prefix": "prod",
                "shards": [],
            },
        )
        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.read_vastai_api_key",
                return_value=_absent_creds(),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.VastaiProviderConfig.from_env",
                return_value=_empty_vastai_config(),
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "batch",
                    "--label",
                    "prod",
                    "--image",
                    "myorg/app:1.0",
                    "--state-path",
                    str(path),
                ],
            )
        assert result.exit_code == 0
        state = json.loads(path.read_text())
        assert state["label_scope"] == "prod-3f9a1b2c4d5e"

    def test_migrates_pre_v4_state(self, tmp_path: Path) -> None:
        path = self._v4_state_path(
            tmp_path,
            {
                "label": "prod-3f9a1b2c4d5e",
                "batch_id": "legacy",
                "shards": [{"shard_id": 0, "status": "deployed"}],
            },
        )
        with (
            patch(
                "vastai_gpu_runner.providers.destroy_adapters.vastai.read_vastai_api_key",
                return_value=_absent_creds(),
            ),
            patch(
                "vastai_gpu_runner.providers.vastai.VastaiProviderConfig.from_env",
                return_value=_empty_vastai_config(),
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "batch",
                    "--label",
                    "prod",
                    "--image",
                    "myorg/app:1.0",
                    "--state-path",
                    str(path),
                ],
            )
        assert result.exit_code == 0
        state = json.loads(path.read_text())
        assert state["requested_label_prefix"] == "prod"
        assert state["label_scope"] == "prod-3f9a1b2c4d5e"
        assert state["schema_version"] == 1

    def test_rejects_persisted_prefix_drift(self, tmp_path: Path) -> None:
        path = self._v4_state_path(
            tmp_path,
            {
                "schema_version": 1,
                "batch_id": "b",
                "label_scope": "staging-3f9a1b2c4d5e",
                "requested_label_prefix": "staging",
                "shards": [],
            },
        )
        result = runner.invoke(
            app,
            [
                "batch",
                "--label",
                "prod",
                "--image",
                "myorg/app:1.0",
                "--state-path",
                str(path),
            ],
        )
        assert result.exit_code != 0
        assert "does not match" in result.output.lower()


def _empty_vastai_config() -> object:
    """Build a VastaiProviderConfig with ``from_env`` mocked."""

    from vastai_gpu_runner.cleanup_policy import OwnershipPolicy
    from vastai_gpu_runner.providers.destroy_adapters.vastai import (
        CredentialResolution,
        CredentialState,
    )
    from vastai_gpu_runner.providers.vastai import VastaiProviderConfig

    base = VastaiProviderConfig(
        docker_image="placeholder",  # placeholder; will be replaced by CLI
        ownership=OwnershipPolicy(),
        credentials=CredentialResolution(state=CredentialState.ABSENT),
        label_prefix=None,
    )
    # Caller will replace docker_image + ownership via dataclasses.replace.
    return base


# ---------------------------------------------------------------------------
# r2-lifecycle sub-commands
# ---------------------------------------------------------------------------


class _FakeS3Lifecycle:
    """Minimal S3 client that records lifecycle calls."""

    def __init__(
        self,
        rules: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        # boto3's response shape is dict[str, object] under the hood,
        # but tests pass narrower dict types (e.g. dict[str, str]) for
        # convenience. Accept any mapping and coerce on storage.
        self.rules: list[dict[str, object]] = [cast("dict[str, object]", r) for r in (rules or [])]
        self.put_calls: list[list[dict[str, object]]] = []
        self.delete_calls = 0
        self.fail_get: Exception | None = None
        self.fail_put: Exception | None = None
        self.fail_delete: Exception | None = None

    def get_bucket_lifecycle_configuration(self, *, Bucket: str) -> dict[str, object]:  # noqa: N803
        if self.fail_get is not None:
            raise self.fail_get
        return {"Rules": list(self.rules)}

    def put_bucket_lifecycle_configuration(
        self,
        *,
        Bucket: str,  # noqa: N803
        LifecycleConfiguration: dict[str, object],  # noqa: N803
    ) -> dict[str, object]:
        if self.fail_put is not None:
            raise self.fail_put
        rules = LifecycleConfiguration.get("Rules", [])
        # pyright's list invariance means we can't cast a list[Any]
        # to list[dict[str, object]] directly. The runtime value is
        # boto3-shaped (list[dict[str, object]]) and tests pass that
        # shape, so the cast is safe.
        self.put_calls.append(rules)  # type: ignore[arg-type]
        self.rules = rules  # type: ignore[assignment]
        return {}

    def delete_bucket_lifecycle(self, *, Bucket: str) -> dict[str, object]:  # noqa: N803
        if self.fail_delete is not None:
            raise self.fail_delete
        self.delete_calls += 1
        self.rules = []
        return {}


def _client_error(code: str) -> Exception:
    class _FakeClientError(Exception):
        pass

    err = _FakeClientError(code)
    err.response = {"Error": {"Code": code}}  # type: ignore[attr-defined]
    return err


def _creds_file(tmp_path: Path) -> Path:
    """Write an admin credentials file with redacted-test secrets."""
    p = tmp_path / "creds"
    p.write_text(
        'export R2_ADMIN_ENDPOINT="https://r2.example"\n'
        'export R2_ADMIN_ACCESS_KEY_ID="akey"\n'
        'export R2_ADMIN_SECRET_ACCESS_KEY="test-secret-fixture"\n',
    )
    return p


def _patch_lifecycle_client(monkeypatch, client: _FakeS3Lifecycle) -> None:
    """Replace ``build_admin_client`` so the CLI uses our fake client."""

    def _builder(_creds: object) -> _FakeS3Lifecycle:
        return client

    monkeypatch.setattr(
        "vastai_gpu_runner.storage.r2_lifecycle.build_admin_client",
        _builder,
    )


class TestR2LifecycleShow:
    def test_show_prints_not_configured(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        client = _FakeS3Lifecycle()
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "show",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "not configured" in result.output
        assert "project/" in result.output

    def test_show_prints_configured_when_rule_present(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        rule_id = "vastai-gpu-runner-expire-4d59ad9c6433"  # deterministic for bkt/project/
        client = _FakeS3Lifecycle(
            rules=[
                {
                    "ID": rule_id,
                    "Status": "Enabled",
                    "Filter": {"Prefix": "project/"},
                    "Expiration": {"Days": 30},
                }
            ],
        )
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "show",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
            ],
        )
        assert result.exit_code == 0
        assert "configured" in result.output
        assert "30" in result.output

    def test_show_rejects_empty_prefix(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        client = _FakeS3Lifecycle()
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "show",
                "--bucket",
                "bkt",
                "--prefix",
                "/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
            ],
        )
        assert result.exit_code != 0
        assert "validation" in result.output or "bucket-wide" in result.output

    def test_show_access_denied_returns_4(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        client = _FakeS3Lifecycle()
        client.fail_get = _client_error("AccessDenied")
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "show",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
            ],
        )
        assert result.exit_code == 4
        assert "access denied" in result.output

    def test_show_no_secret_in_output(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        client = _FakeS3Lifecycle()
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "show",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
                "-v",
            ],
        )
        # test-secret-fixture is the secret in the test creds file.
        assert "test-secret-fixture" not in result.output


class TestR2LifecycleApply:
    def test_apply_dry_run_does_not_put(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        client = _FakeS3Lifecycle()
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "apply",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
                "--expire-after-days",
                "30",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert client.put_calls == []
        assert "dry-run" in result.output

    def test_apply_no_op_does_not_put(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        rule_id = "vastai-gpu-runner-expire-4d59ad9c6433"
        existing = [
            {
                "ID": rule_id,
                "Status": "Enabled",
                "Filter": {"Prefix": "project/"},
                "Expiration": {"Days": 30},
            }
        ]
        client = _FakeS3Lifecycle(rules=existing)
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "apply",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
                "--expire-after-days",
                "30",
                "--yes",
            ],
        )
        assert result.exit_code == 0
        assert "no-op" in result.output
        assert client.put_calls == []

    def test_apply_yes_executes_put(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        client = _FakeS3Lifecycle()
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "apply",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
                "--expire-after-days",
                "30",
                "--yes",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "applied" in result.output
        assert len(client.put_calls) == 1

    def test_apply_refuses_without_yes_when_stdin_not_tty(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        # typer.confirm uses click's getchar; the CliRunner doesn't simulate
        # a TTY by default, so we expect the refusal path.
        client = _FakeS3Lifecycle()
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "apply",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
                "--expire-after-days",
                "30",
            ],
        )
        # Either refused with exit 9, or typer exited cleanly because
        # confirm() raised Abort. Both mean no PUT happened.
        assert client.put_calls == []
        assert result.exit_code != 0 or "refusing" in result.output

    def test_apply_verification_failure_returns_8(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        client = _FakeS3Lifecycle()

        original_put = client.put_bucket_lifecycle_configuration

        def corrupt_put(
            *,
            Bucket: str,  # noqa: N803
            LifecycleConfiguration: dict[str, object],  # noqa: N803
        ) -> dict[str, object]:
            original_put(
                Bucket=Bucket,
                LifecycleConfiguration=LifecycleConfiguration,
            )
            # Corrupt the resulting state so read-after-write fails.
            # Use a fresh list of fresh dicts to avoid mutating the
            # rule references held by the manager's plan.
            client.rules = [{**r, "Expiration": {"Days": 999}} for r in client.rules]
            return {}

        client.put_bucket_lifecycle_configuration = corrupt_put  # type: ignore[method-assign]
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "apply",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
                "--expire-after-days",
                "30",
                "--yes",
            ],
        )
        assert result.exit_code == 8, result.output
        assert "verification" in result.output

    def test_apply_rule_limit_returns_7(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        client = _FakeS3Lifecycle()
        client.fail_put = _client_error("TooManyRules")
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "apply",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
                "--expire-after-days",
                "30",
                "--yes",
            ],
        )
        assert result.exit_code == 7

    def test_apply_missing_retention_rejected(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        client = _FakeS3Lifecycle()
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        # Typer Option(min=1) → "Invalid value" if user passes 0 or omits.
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "apply",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
                "--expire-after-days",
                "0",
                "--yes",
            ],
        )
        assert result.exit_code != 0
        assert client.put_calls == []

    def test_apply_no_secret_in_output(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        client = _FakeS3Lifecycle()
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "apply",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
                "--expire-after-days",
                "30",
                "--yes",
                "-v",
            ],
        )
        assert "skey" not in result.output


class TestR2LifecycleRemove:
    def test_remove_dry_run_does_not_put(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        rule_id = "vastai-gpu-runner-expire-4d59ad9c6433"
        existing = [
            {
                "ID": rule_id,
                "Status": "Enabled",
                "Filter": {"Prefix": "project/"},
                "Expiration": {"Days": 30},
            }
        ]
        client = _FakeS3Lifecycle(rules=existing)
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "remove",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert client.put_calls == []
        assert "dry-run" in result.output

    def test_remove_yes_executes(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        rule_id = "vastai-gpu-runner-expire-4d59ad9c6433"
        existing = [
            {
                "ID": rule_id,
                "Status": "Enabled",
                "Filter": {"Prefix": "project/"},
                "Expiration": {"Days": 30},
            }
        ]
        client = _FakeS3Lifecycle(rules=existing)
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "remove",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
                "--yes",
            ],
        )
        assert result.exit_code == 0
        assert "removed" in result.output
        # This is the only rule, so remove calls delete_bucket_lifecycle,
        # NOT put_bucket_lifecycle_configuration.
        assert len(client.put_calls) == 0
        assert client.delete_calls == 1
        assert all(r["ID"] != rule_id for r in client.rules)

    def test_remove_no_op_when_absent(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        client = _FakeS3Lifecycle()
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "remove",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
                "--yes",
            ],
        )
        assert result.exit_code == 0
        assert "no-op" in result.output
        assert client.put_calls == []

    def test_remove_with_foreign_uses_put_not_delete(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        """When foreign rules remain, remove uses PUT (not DELETE)."""
        rule_id = "vastai-gpu-runner-expire-4d59ad9c6433"
        existing = [
            {
                "ID": rule_id,
                "Status": "Enabled",
                "Filter": {"Prefix": "project/"},
                "Expiration": {"Days": 30},
            },
            {"ID": "f", "Status": "Enabled"},
        ]
        client = _FakeS3Lifecycle(rules=existing)
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "remove",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
                "--yes",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "removed" in result.output
        assert len(client.put_calls) == 1
        assert client.delete_calls == 0
        assert all(r["ID"] != rule_id for r in client.rules)

    def test_remove_collision_returns_5(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        rule_id = "vastai-gpu-runner-expire-4d59ad9c6433"
        existing = [
            {
                "ID": rule_id,
                "Status": "Enabled",
                "Filter": {"Prefix": "project/"},
                # No Days field — triggers collision check during apply,
                # not remove. To exercise the remove path we instead make
                # remove fail by stubbing the verification.
                "Expiration": {"Days": 30},
            }
        ]
        client = _FakeS3Lifecycle(rules=existing)
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "remove",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(_creds_file(tmp_path)),
                "--yes",
            ],
        )
        assert result.exit_code == 0  # No collision on the remove path.


class TestR2LifecycleCLIIntegration:
    def test_help_prints(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["r2-lifecycle", "--help"])
        assert result.exit_code == 0
        assert "show" in result.output
        assert "apply" in result.output
        assert "remove" in result.output

    def test_missing_credentials_file_exits_3(
        self,
        tmp_path: Path,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "show",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(tmp_path / "missing"),
            ],
        )
        # Missing credentials file is now a typed ``CredentialsError``
        # which maps to exit code 3.
        assert result.exit_code == 3
        assert "credentials" in result.output.lower()

    def test_missing_creds_keys_exits_3(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        creds = tmp_path / "creds"
        creds.write_text('export R2_ENDPOINT="https://r2.example"\n')
        client = _FakeS3Lifecycle()
        _patch_lifecycle_client(monkeypatch, client)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "r2-lifecycle",
                "show",
                "--bucket",
                "bkt",
                "--prefix",
                "project/",
                "--credentials-file",
                str(creds),
            ],
        )
        assert result.exit_code == 3
