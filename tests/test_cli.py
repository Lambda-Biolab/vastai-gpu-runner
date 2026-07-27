"""Tests for the v4 CLI composition roots.

Pins the new ``--label`` validation, the empty / blank / padded
rejection, and the empty ``--allowed-images`` fail-closed semantic.
Uses ``typer.testing.CliRunner`` to exercise each command end-to-end
with mocked ``vastai_cmd`` and ``read_vastai_api_key`` boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

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
    def _patched_policy(candidates: list[object]) -> object:
        """Patch ``build_vastai_cleanup_policy`` so the candidates list is fed in."""
        from vastai_gpu_runner.cleanup_policy import (
            CleanupResult,
            CleanupVerdict,
            Provider,
            ProviderCleanupPolicy,
        )

        def _list() -> list[object]:
            return list(candidates)

        def _destroy(candidate: object) -> CleanupResult:
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
        policy = self._patched_policy([c])
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
        policy = self._patched_policy([c])
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

        def _list() -> list[object]:
            return [c1, c2]

        def _destroy(candidate: object) -> CleanupResult:
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
        policy = self._patched_policy([evil])
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
        policy = self._patched_policy([evil])
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

        def _list() -> list[object]:
            return [c1, c2]

        def _destroy(candidate: object) -> CleanupResult:
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
