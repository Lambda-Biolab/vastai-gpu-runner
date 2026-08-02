# pyright: reportPrivateUsage=warning, reportMissingParameterType=warning, reportUnusedFunction=false, reportUnusedClass=false
"""Tests for state module — batch state persistence and properties."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vastai_gpu_runner.state import (
    CURRENT_SCHEMA_VERSION,
    BatchState,
    JobBatchState,
    JobState,
    ShardState,
    StateMigrationError,
    _migrate_pre_v4,
    _validate_label_scope_shape,
    load_batch_state,
    resolve_label_scope,
    validate_label_prefix,
    validate_label_scope_for_cleanup,
)


class TestShardState:
    def test_defaults(self) -> None:
        shard = ShardState(shard_id=0)
        assert shard.status == "pending"
        assert shard.items_expected == 0
        assert shard.retry_count == 0

    def test_item_ids_list(self) -> None:
        shard = ShardState(shard_id=1, item_ids=["a", "b", "c"])
        assert len(shard.item_ids) == 3


class TestBatchState:
    def test_empty_batch(self) -> None:
        batch = BatchState(batch_id="test-001")
        assert batch.active_shards == []
        assert batch.failed_shards == []
        assert batch.pending_shards == []

    def test_shard_filtering(self) -> None:
        shards = [
            ShardState(shard_id=0, status="pending"),
            ShardState(shard_id=1, status="deployed"),
            ShardState(shard_id=2, status="running"),
            ShardState(shard_id=3, status="downloaded"),
            ShardState(shard_id=4, status="failed"),
        ]
        batch = BatchState(batch_id="test", shards=shards)
        assert len(batch.pending_shards) == 1
        assert len(batch.active_shards) == 2
        assert len(batch.downloaded_shards) == 1
        assert len(batch.failed_shards) == 1

    def test_save_and_load(self, tmp_path: Path) -> None:
        shards = [
            ShardState(shard_id=0, status="pending", item_ids=["x", "y"]),
            ShardState(shard_id=1, status="downloaded"),
        ]
        original = BatchState(batch_id="round-trip", num_gpus=2, shards=shards)
        state_path = tmp_path / "batch_state.json"
        original.save(state_path)

        loaded = BatchState.load(state_path)
        assert loaded.batch_id == "round-trip"
        assert loaded.num_gpus == 2
        assert len(loaded.shards) == 2
        assert loaded.shards[0].item_ids == ["x", "y"]
        assert loaded.shards[1].status == "downloaded"
        assert loaded.updated_at > 0

    def test_atomic_save(self, tmp_path: Path) -> None:
        """Save uses tmp+rename for atomicity."""
        state_path = tmp_path / "batch_state.json"
        batch = BatchState(batch_id="atomic")
        batch.save(state_path)
        assert state_path.exists()
        assert not state_path.with_suffix(".tmp").exists()

    def test_metadata_round_trip(self, tmp_path: Path) -> None:
        batch = BatchState(batch_id="meta", metadata={"target": "VicK"})
        state_path = tmp_path / "state.json"
        batch.save(state_path)
        loaded = BatchState.load(state_path)
        assert loaded.metadata["target"] == "VicK"


class TestJobState:
    def test_defaults(self) -> None:
        job = JobState(job_name="VicK_001")
        assert job.status == "pending"
        assert job.cost_usd == 0.0

    def test_cost_calculation(self) -> None:
        job = JobState(
            job_name="test",
            submit_time="2026-04-01T10:00:00+00:00",
            complete_time="2026-04-01T12:00:00+00:00",
            cost_per_hour=0.30,
        )
        assert abs(job.cost_usd - 0.60) < 0.01


class TestJobBatchState:
    def test_job_filtering(self) -> None:
        jobs = [
            JobState(job_name="a", status="pending"),
            JobState(job_name="b", status="running"),
            JobState(job_name="c", status="completed"),
        ]
        batch = JobBatchState(batch_id="md-001", jobs=jobs)
        assert len(batch.pending_jobs) == 1
        assert len(batch.active_jobs) == 1
        assert len(batch.completed_jobs) == 1

    def test_save_and_load(self, tmp_path: Path) -> None:
        jobs = [JobState(job_name="j1", status="running", cost_per_hour=0.25)]
        original = JobBatchState(batch_id="md-rt", jobs=jobs)
        state_path = tmp_path / "md_state.json"
        original.save(state_path)

        loaded = JobBatchState.load(state_path)
        assert loaded.batch_id == "md-rt"
        assert len(loaded.jobs) == 1
        assert loaded.jobs[0].cost_per_hour == 0.25

    def test_total_cost(self) -> None:
        jobs = [
            JobState(
                job_name="a",
                submit_time="2026-04-01T10:00:00+00:00",
                complete_time="2026-04-01T11:00:00+00:00",
                cost_per_hour=0.30,
            ),
            JobState(
                job_name="b",
                submit_time="2026-04-01T10:00:00+00:00",
                complete_time="2026-04-01T12:00:00+00:00",
                cost_per_hour=0.30,
            ),
        ]
        batch = JobBatchState(batch_id="cost", jobs=jobs)
        assert abs(batch.total_cost - 0.90) < 0.01


class TestBatchStateArchive:
    """archive_if_all_terminal + load_or_none — shared hygiene for resuming batches."""

    def test_archive_all_terminal(self, tmp_path: Path) -> None:
        path = tmp_path / "batch_state.json"
        BatchState(
            batch_id="terminal",
            shards=[
                ShardState(shard_id=0, status="downloaded"),
                ShardState(shard_id=1, status="failed"),
            ],
        ).save(path)
        BatchState.archive_if_all_terminal(path)
        assert not path.exists()
        archives = list(tmp_path.glob("batch_state_*.json"))
        assert len(archives) == 1

    def test_no_archive_when_active_shard(self, tmp_path: Path) -> None:
        path = tmp_path / "batch_state.json"
        BatchState(
            batch_id="active",
            shards=[
                ShardState(shard_id=0, status="downloaded"),
                ShardState(shard_id=1, status="running"),
            ],
        ).save(path)
        BatchState.archive_if_all_terminal(path)
        assert path.exists()
        assert list(tmp_path.glob("batch_state_*.json")) == []

    def test_no_archive_when_file_missing(self, tmp_path: Path) -> None:
        BatchState.archive_if_all_terminal(tmp_path / "absent.json")
        # no error, no files created

    def test_no_archive_when_corrupt(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.json"
        path.write_text("{ not json")
        BatchState.archive_if_all_terminal(path)
        assert path.exists()  # corrupt file preserved for load_or_none to warn

    def test_load_or_none_present(self, tmp_path: Path) -> None:
        path = tmp_path / "batch_state.json"
        BatchState(batch_id="ok", shards=[ShardState(shard_id=0)]).save(path)
        loaded = BatchState.load_or_none(path)
        assert loaded is not None
        assert loaded.batch_id == "ok"

    def test_load_or_none_absent(self, tmp_path: Path) -> None:
        assert BatchState.load_or_none(tmp_path / "absent.json") is None

    def test_load_or_none_corrupt(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.json"
        path.write_text("not json")
        assert BatchState.load_or_none(path) is None

    def test_subclass_custom_terminal_statuses(self, tmp_path: Path) -> None:
        """Subclass narrows TERMINAL_STATUSES — archive only fires for its set."""

        class NarrowBatch(BatchState):
            TERMINAL_STATUSES = frozenset({"failed"})

        path = tmp_path / "narrow_state.json"
        NarrowBatch(
            batch_id="narrow",
            shards=[ShardState(shard_id=0, status="downloaded")],
        ).save(path)
        NarrowBatch.archive_if_all_terminal(path)
        assert path.exists()  # "downloaded" not in subclass's terminal set


class TestJobBatchStateArchive:
    """Parallel coverage for the job-based variant."""

    def test_archive_all_terminal(self, tmp_path: Path) -> None:
        path = tmp_path / "md_batch_state.json"
        JobBatchState(
            batch_id="terminal",
            jobs=[
                JobState(job_name="a", status="completed"),
                JobState(job_name="b", status="downloaded"),
            ],
        ).save(path)
        JobBatchState.archive_if_all_terminal(path)
        assert not path.exists()
        assert len(list(tmp_path.glob("md_batch_state_*.json"))) == 1

    def test_no_archive_when_pending_job(self, tmp_path: Path) -> None:
        path = tmp_path / "md_batch_state.json"
        JobBatchState(
            batch_id="active",
            jobs=[JobState(job_name="a", status="pending")],
        ).save(path)
        JobBatchState.archive_if_all_terminal(path)
        assert path.exists()

    def test_load_or_none_absent(self, tmp_path: Path) -> None:
        assert JobBatchState.load_or_none(tmp_path / "absent.json") is None

    def test_load_or_none_corrupt(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.json"
        path.write_text("]")
        assert JobBatchState.load_or_none(path) is None


# ---------------------------------------------------------------------------
# v4 schema + label-scope helpers
# ---------------------------------------------------------------------------


class TestValidateLabelPrefix:
    """``validate_label_prefix`` is the v4 boundary for safe prefixes."""

    def test_accepts_non_empty_pre_stripped(self) -> None:
        assert validate_label_prefix("prod") == "prod"

    @pytest.mark.parametrize("bad", ["", " ", "  padded  "])
    def test_rejects_empty_blank_padded(self, bad: str) -> None:
        with pytest.raises(ValueError, match="label_prefix"):
            validate_label_prefix(bad)

    def test_rejects_non_string(self) -> None:
        with pytest.raises(ValueError, match="label_prefix"):
            validate_label_prefix(None)  # type: ignore[arg-type]


class TestValidateLabelScopeForCleanup:
    """Manual cleanup requires a full canonical scope (12-hex suffix)."""

    def test_accepts_full_canonical_scope(self) -> None:
        assert validate_label_scope_for_cleanup("prod-3f9a1b2c4d5e") == "prod-3f9a1b2c4d5e"

    @pytest.mark.parametrize(
        "bad",
        [
            "prod",  # bare prefix, no suffix
            "prod-3F9A1B2C4D5E",  # uppercase hex
            "prod-3f9a1b2c4d5",  # 11 hex chars
            "prod-3f9a1b2c4d5eX",  # 12 hex + non-hex
            "",  # empty
            " ",
        ],
    )
    def test_rejects_malformed_scope(self, bad: str) -> None:
        with pytest.raises(ValueError):
            validate_label_scope_for_cleanup(bad)


class TestValidateLabelScopeShape:
    """`_validate_label_scope_shape` enforces scope prefix + suffix grammar."""

    def test_accepts_well_formed_scope(self) -> None:
        assert _validate_label_scope_shape("prod-3f9a1b2c4d5e", "prod") == "prod-3f9a1b2c4d5e"

    def test_rejects_mismatched_prefix(self) -> None:
        with pytest.raises(StateMigrationError, match="does not start"):
            _validate_label_scope_shape("staging-3f9a1b2c4d5e", "prod")

    def test_rejects_uppercase_hex(self) -> None:
        with pytest.raises(StateMigrationError, match="malformed suffix"):
            _validate_label_scope_shape("prod-3F9A1B2C4D5E", "prod")

    def test_rejects_short_suffix(self) -> None:
        with pytest.raises(StateMigrationError, match="malformed suffix"):
            _validate_label_scope_shape("prod-3f9a1b2c4d5", "prod")


class TestMigratePreV4:
    """Schema-0 migration recovers identity from legacy ``label`` field."""

    def test_strips_legacy_hex_suffix(self) -> None:
        data = {
            "label": "prod-3f9a1b2c4d5e",
            "shards": [],
        }
        migrated, err = _migrate_pre_v4(data, state_cls=BatchState)
        assert err is None
        assert migrated["requested_label_prefix"] == "prod"
        assert migrated["label_scope"] == "prod-3f9a1b2c4d5e"
        assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION
        assert "label" not in migrated

    def test_rejects_nonterminal_scope_less_state(self) -> None:
        data = {
            "shards": [{"shard_id": 0, "status": "running"}],
        }
        migrated, err = _migrate_pre_v4(data, state_cls=BatchState)
        assert migrated is data
        assert err is not None
        assert "recoverable label scope" in err

    def test_rejects_non_list_shards(self) -> None:
        """A dict-valued ``shards`` cannot produce a misleading truthy ``raw_units``."""
        data = {
            "label": "prod-3f9a1b2c4d5e",
            "shards": {"key": "value"},  # not a list
        }
        migrated, err = _migrate_pre_v4(data, state_cls=BatchState)
        assert migrated is data
        assert err is not None
        assert "shards must be a list" in err

    def test_rejects_non_list_jobs(self) -> None:
        data = {
            "label": "prod-3f9a1b2c4d5e",
            "jobs": {"k": "v"},
        }
        migrated, err = _migrate_pre_v4(data, state_cls=JobBatchState)
        assert migrated is data
        assert err is not None
        assert "jobs must be a list" in err

    def test_terminal_scope_less_returns_empty_prefix(self) -> None:
        """Verified-terminal scope-less legacy is migrated, not rejected."""
        data = {
            "shards": [{"shard_id": 0, "status": "downloaded"}],
        }
        migrated, err = _migrate_pre_v4(data, state_cls=BatchState)
        assert err is None
        assert migrated["requested_label_prefix"] == ""
        assert migrated["label_scope"] == ""

    def test_legacy_label_with_nonterminal_unit_returns_scope(self) -> None:
        data = {
            "label": "prod-3f9a1b2c4d5e",
            "shards": [{"shard_id": 0, "status": "deployed"}],
        }
        migrated, err = _migrate_pre_v4(data, state_cls=BatchState)
        assert err is None
        assert migrated["requested_label_prefix"] == "prod"
        assert migrated["label_scope"] == "prod-3f9a1b2c4d5e"

    def test_conflicting_label_and_label_scope_rejected(self) -> None:
        data = {
            "label": "prod-3f9a1b2c4d5e",
            "label_scope": "staging-3f9a1b2c4d5e",
        }
        migrated, err = _migrate_pre_v4(data, state_cls=BatchState)
        assert migrated is data
        assert err is not None
        assert "disagree" in err


class TestLoadBatchState:
    """``load_batch_state`` is the v4 loader; raises ``StateMigrationError``."""

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        assert load_batch_state(tmp_path / "absent.json", state_cls=BatchState) is None

    def test_loads_v4_state(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "batch_id": "b",
                    "label_scope": "prod-3f9a1b2c4d5e",
                    "requested_label_prefix": "prod",
                    "shards": [{"shard_id": 0, "status": "pending"}],
                }
            )
        )
        loaded = load_batch_state(path, state_cls=BatchState)
        assert loaded is not None
        assert loaded.batch_id == "b"
        assert loaded.label_scope == "prod-3f9a1b2c4d5e"
        assert len(loaded.shards) == 1
        assert isinstance(loaded.shards[0], ShardState)

    def test_migrates_pre_v4_state(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.json"
        path.write_text(
            json.dumps(
                {
                    "label": "prod-3f9a1b2c4d5e",
                    "batch_id": "legacy",
                    "shards": [{"shard_id": 0, "status": "deployed"}],
                }
            )
        )
        loaded = load_batch_state(path, state_cls=BatchState)
        assert loaded is not None
        assert loaded.requested_label_prefix == "prod"
        assert loaded.label_scope == "prod-3f9a1b2c4d5e"
        assert loaded.schema_version == CURRENT_SCHEMA_VERSION
        # second load returns the same migrated state without re-archive
        again = load_batch_state(path, state_cls=BatchState)
        assert again is not None
        assert again.batch_id == "legacy"

    def test_archives_terminal_scope_less_legacy(self, tmp_path: Path) -> None:
        path = tmp_path / "terminal.json"
        path.write_text(
            json.dumps(
                {
                    "label": "",
                    "shards": [{"shard_id": 0, "status": "downloaded"}],
                }
            )
        )
        loaded = load_batch_state(path, state_cls=BatchState)
        assert loaded is None
        assert not path.exists()
        archives = list(tmp_path.glob("terminal_archived_*.json"))
        assert len(archives) == 1

    def test_rejects_nonterminal_scope_less_legacy(self, tmp_path: Path) -> None:
        path = tmp_path / "nonterminal.json"
        path.write_text(
            json.dumps(
                {
                    "label": "",
                    "shards": [{"shard_id": 0, "status": "running"}],
                }
            )
        )
        with pytest.raises(StateMigrationError, match="recoverable label scope"):
            load_batch_state(path, state_cls=BatchState)

    def test_rejects_non_integer_schema_version(self, tmp_path: Path) -> None:
        path = tmp_path / "bool_version.json"
        path.write_text(json.dumps({"schema_version": True, "shards": []}))
        with pytest.raises(StateMigrationError, match="schema_version must be an integer"):
            load_batch_state(path, state_cls=BatchState)

    def test_rejects_unsupported_schema_version(self, tmp_path: Path) -> None:
        path = tmp_path / "future.json"
        path.write_text(json.dumps({"schema_version": 99, "shards": []}))
        with pytest.raises(StateMigrationError, match="unsupported schema_version"):
            load_batch_state(path, state_cls=BatchState)

    def test_rejects_non_dict_root(self, tmp_path: Path) -> None:
        path = tmp_path / "list_root.json"
        path.write_text("[]")
        with pytest.raises(StateMigrationError, match="root must be an object"):
            load_batch_state(path, state_cls=BatchState)

    def test_rejects_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json")
        with pytest.raises(StateMigrationError, match="JSON is invalid"):
            load_batch_state(path, state_cls=BatchState)

    def test_loads_v4_job_state(self, tmp_path: Path) -> None:
        path = tmp_path / "jobs.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "batch_id": "md",
                    "label_scope": "md-3f9a1b2c4d5e",
                    "requested_label_prefix": "md",
                    "jobs": [{"job_name": "j1", "status": "running"}],
                }
            )
        )
        loaded = load_batch_state(path, state_cls=JobBatchState)
        assert loaded is not None
        assert len(loaded.jobs) == 1
        assert isinstance(loaded.jobs[0], JobState)

    def test_second_load_does_not_re_archive(self, tmp_path: Path) -> None:
        """Loading the same terminal-scope-less fixture twice archives only once."""
        path = tmp_path / "terminal.json"
        path.write_text(
            json.dumps(
                {
                    "shards": [{"shard_id": 0, "status": "downloaded"}],
                }
            )
        )
        # First load archives the file.
        first = load_batch_state(path, state_cls=BatchState)
        assert first is None
        # The path is now gone; a second load returns None on the missing file.
        second = load_batch_state(path, state_cls=BatchState)
        assert second is None


class TestResolveLabelScope:
    """``resolve_label_scope`` reuses persisted identity or creates a new one."""

    def test_creates_scope_for_new_identity(self) -> None:
        scope = resolve_label_scope("prod", None, None)
        assert scope.startswith("prod-")
        assert len(scope.split("-")[-1]) == 12

    def test_reuses_persisted_scope(self) -> None:
        scope = resolve_label_scope(
            "prod", persisted_scope="prod-3f9a1b2c4d5e", persisted_requested_prefix="prod"
        )
        assert scope == "prod-3f9a1b2c4d5e"

    def test_rejects_partial_persisted_identity(self) -> None:
        with pytest.raises(StateMigrationError, match="partial"):
            resolve_label_scope(
                "prod", persisted_scope="prod-3f9a1b2c4d5e", persisted_requested_prefix=None
            )

    def test_rejects_drifted_prefix(self) -> None:
        with pytest.raises(StateMigrationError, match="does not match"):
            resolve_label_scope(
                "prod", persisted_scope="prod-3f9a1b2c4d5e", persisted_requested_prefix="prod-old"
            )

    def test_rejects_overlapping_prefix_drift(self) -> None:
        """``prod-us`` does not match ``prod`` — explicit substring drift."""
        with pytest.raises(StateMigrationError, match="does not match"):
            resolve_label_scope(
                "prod", persisted_scope="prod-us-3f9a1b2c4d5e", persisted_requested_prefix="prod-us"
            )

    def test_rejects_invalid_requested_prefix(self) -> None:
        with pytest.raises(ValueError):
            resolve_label_scope("", None, None)

    def test_rejects_malformed_persisted_scope(self) -> None:
        with pytest.raises(StateMigrationError, match="malformed suffix"):
            resolve_label_scope(
                "prod", persisted_scope="prod-bad", persisted_requested_prefix="prod"
            )


class TestBatchStateV4Fields:
    """``BatchState`` and ``JobBatchState`` carry the v4 identity fields."""

    def test_batch_state_default_label_scope_is_empty(self) -> None:
        b = BatchState()
        assert b.label_scope == ""
        assert b.requested_label_prefix == ""
        assert b.schema_version == CURRENT_SCHEMA_VERSION

    def test_job_batch_state_default_label_scope_is_empty(self) -> None:
        b = JobBatchState()
        assert b.label_scope == ""
        assert b.requested_label_prefix == ""
        assert b.schema_version == CURRENT_SCHEMA_VERSION
