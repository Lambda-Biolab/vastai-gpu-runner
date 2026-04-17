"""Tests for state module — batch state persistence and properties."""

from __future__ import annotations

from pathlib import Path

from vastai_gpu_runner.state import (
    BatchState,
    JobBatchState,
    JobState,
    ShardState,
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
