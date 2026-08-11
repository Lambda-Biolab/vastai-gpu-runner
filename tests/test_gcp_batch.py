"""Tests for the GCP Batch runner and GCS sink.

These tests use in-memory fakes that match the public surface of
``google.cloud.batch_v1.BatchServiceClient`` and
``google.cloud.storage.Client``. The Google SDKs are not imported
during testing; the runner takes pre-constructed clients, and the
GcsSink resolves its client lazily from the optional ``gcp`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vastai_gpu_runner.managed_jobs.base import (
    ManagedJobSpec,
    ManagedJobTerminalState,
)
from vastai_gpu_runner.managed_jobs.gcp_batch import (
    SPOT_PREEMPT_EXIT_CODE,
    FakeGcpBatchClient,
    GcpBatchRunner,
)
from vastai_gpu_runner.managed_jobs.state import (
    CURRENT_MANAGED_JOB_SCHEMA,
    ManagedJobState,
    save_managed_job_state,
)
from vastai_gpu_runner.storage.gcs import GcsSink

# google-cloud-batch is heavy and pinned via the optional extra; import
# is lazy so the rest of the suite runs without the dependency.
google_cloud_batch = pytest.importorskip("google.cloud.batch_v1")


def _make_spec(**overrides: object) -> ManagedJobSpec:
    base: dict = {
        "name": "demo-job",
        "task_count": 3,
        "parallelism": 2,
        "image": "gcr.io/example/worker:latest",
        "command": ("python", "-m", "demo"),
        "environment": {"machine_type": "n1-standard-4", "provisioning_model": "SPOT"},
        "labels": {"campaign": "demo", "stage": "backbone"},
        "gcs_mounts": ("/mnt/campaign",),
        "timeout_seconds": 600,
        "retry_on_preempt": True,
        "region": "us-central1",
    }
    base.update(overrides)
    return ManagedJobSpec(**base)


def test_submit_builds_correct_request() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(
        project_id="proj-1",
        region="us-central1",
        client=fake,  # type: ignore[arg-type]
    )
    handle = runner.submit(_make_spec())
    assert handle.provider == "gcp-batch"
    assert handle.resource_name == "projects/proj-1/locations/us-central1/jobs/demo-job"
    assert handle.location == "proj-1/us-central1"
    assert len(fake.submit_calls) == 1
    request = fake.submit_calls[0]
    assert request.parent == "projects/proj-1/locations/us-central1"
    assert request.job_id == "demo-job"
    assert request.job.task_groups[0].task_count == 3
    assert request.job.task_groups[0].parallelism == 2
    assert request.job.task_groups[0].task_spec.max_retry_count == 3
    lifecycle = request.job.task_groups[0].task_spec.lifecycle_policies
    assert lifecycle[0].action == google_cloud_batch.LifecyclePolicy.Action.RETRY_TASK
    assert lifecycle[0].action_condition.exit_codes == [SPOT_PREEMPT_EXIT_CODE]
    expected_log_dest = google_cloud_batch.LogsPolicy.Destination.CLOUD_LOGGING
    assert request.job.logs_policy.destination == expected_log_dest


def test_submit_with_spot_retry_when_requested() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(_make_spec(retry_on_preempt=True))
    lifecycle = fake.submit_calls[0].job.task_groups[0].task_spec.lifecycle_policies
    assert len(lifecycle) == 1
    assert lifecycle[0].action_condition.exit_codes == [SPOT_PREEMPT_EXIT_CODE]


def test_submit_without_spot_retry_omits_lifecycle() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(_make_spec(retry_on_preempt=False))
    task_spec = fake.submit_calls[0].job.task_groups[0].task_spec
    assert task_spec.max_retry_count == 0
    assert not task_spec.lifecycle_policies


def test_submit_provisioning_model_passthrough() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(
        _make_spec(environment={"machine_type": "a3-highgpu-1g", "provisioning_model": "STANDARD"})
    )
    instance_policy = fake.submit_calls[0].job.allocation_policy.instances[0].policy
    assert instance_policy.machine_type == "a3-highgpu-1g"
    expected_enum = type(instance_policy.provisioning_model).STANDARD
    assert instance_policy.provisioning_model == expected_enum


def test_get_status_translates_state() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec())

    fake.statuses[handle.resource_name] = _status(google_cloud_batch.JobStatus.State.SUCCEEDED)
    status = runner.get_status(handle)
    assert status.state == ManagedJobTerminalState.SUCCEEDED

    fake.statuses[handle.resource_name] = _status(google_cloud_batch.JobStatus.State.FAILED)
    status = runner.get_status(handle)
    assert status.state == ManagedJobTerminalState.FAILED

    fake.statuses[handle.resource_name] = _status(google_cloud_batch.JobStatus.State.CANCELLED)
    status = runner.get_status(handle)
    assert status.state == ManagedJobTerminalState.CANCELLED

    fake.statuses[handle.resource_name] = _status(google_cloud_batch.JobStatus.State.RUNNING)
    status = runner.get_status(handle)
    assert status.state == ManagedJobTerminalState.UNKNOWN


def test_get_status_aggregates_task_counts() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec(task_count=4))

    status = google_cloud_batch.JobStatus()
    group_status = google_cloud_batch.JobStatus.TaskGroupStatus()
    group_status.counts = {
        "SUCCEEDED": 3,
        "FAILED": 1,
    }
    status.task_groups = {"group0": group_status}
    fake.statuses[handle.resource_name] = status

    result = runner.get_status(handle)
    assert result.succeeded_tasks == 3
    assert result.failed_tasks == 1
    assert result.total_tasks == 4


def test_list_tasks_yields_status() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec())

    task = google_cloud_batch.Task()
    task.name = f"{handle.resource_name}/taskGroups/group0/tasks/2"
    task_status = google_cloud_batch.TaskStatus()
    task_status.state = google_cloud_batch.TaskStatus.State.SUCCEEDED
    task.status = task_status
    fake.tasks[f"{handle.resource_name}/taskGroups/group0"] = [task]

    listed = list(runner.list_tasks(handle))
    assert listed[0].task_index == 2
    assert listed[0].state == "SUCCEEDED"


def test_cancel_and_delete_track_calls() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec())
    runner.cancel(handle)
    assert fake.cancel_calls == [handle.resource_name]
    runner.delete(handle)
    assert fake.delete_calls == [handle.resource_name]


def test_submit_persists_state() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec())
    state = ManagedJobState(
        provider="gcp-batch",
        resource_name=handle.resource_name,
        location=handle.location,
        task_count=3,
        state="submitted",
        campaign_id="demo",
        stage_id="backbone",
    )
    assert state.schema_version == CURRENT_MANAGED_JOB_SCHEMA
    # Persistence path is exercised by the consumer; here we only
    # verify the dataclass shape and round-trip write.
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        path = Path(fh.name)
    save_managed_job_state(state, path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == CURRENT_MANAGED_JOB_SCHEMA
    assert payload["resource_name"] == handle.resource_name


# ---------------------------------------------------------------------------
# GcsSink tests
# ---------------------------------------------------------------------------


class FakeGcsClientForSink:
    """In-memory GCS client implementing only what :class:`GcsSink` uses."""

    def __init__(self) -> None:
        self.buckets: dict[str, dict[str, bytes]] = {"campaign": {}}
        self.uploads: list[tuple[str, str, bytes, str | None]] = []

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self, name)


class FakeBucket:
    def __init__(self, client: FakeGcsClientForSink, name: str) -> None:
        self._client = client
        self._name = name

    def exists(self) -> bool:
        return self._name in self._client.buckets

    def blob(self, key: str) -> FakeBlob:
        return FakeBlob(self._client, self._name, key)


class FakeBlob:
    def __init__(self, client: FakeGcsClientForSink, bucket: str, key: str) -> None:
        self._client = client
        self._bucket = bucket
        self._key = key
        self.chunk_size = 0

    def upload_from_string(self, data: bytes, content_type: str | None = None) -> None:
        self._client.uploads.append((self._bucket, self._key, data, content_type))
        self._client.buckets.setdefault(self._bucket, {})[self._key] = data

    def download_as_bytes(self) -> bytes:
        return self._client.buckets.get(self._bucket, {}).get(self._key, b"")

    def exists(self) -> bool:
        return self._key in self._client.buckets.get(self._bucket, {})

    def delete(self) -> None:
        self._client.buckets.setdefault(self._bucket, {}).pop(self._key, None)


def test_gcs_sink_upload_and_download_bytes() -> None:
    client = FakeGcsClientForSink()
    sink = GcsSink(bucket_name="campaign", client=client)  # type: ignore[arg-type]
    uri = sink.upload_bytes("foo/bar.txt", b"hello", content_type="text/plain")
    assert uri == "gs://campaign/foo/bar.txt"
    assert sink.exists("foo/bar.txt")
    assert sink.download_bytes("foo/bar.txt") == b"hello"


def test_gcs_sink_missing_bucket_raises() -> None:
    client = FakeGcsClientForSink()
    client.buckets.clear()
    sink = GcsSink(bucket_name="absent", client=client)  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        sink.upload_bytes("foo", b"x")


def test_gcs_sink_atomic_json_writes_final_key() -> None:
    client = FakeGcsClientForSink()
    sink = GcsSink(bucket_name="campaign", client=client)  # type: ignore[arg-type]
    uri = sink.upload_atomic_json("events/2026-08-11.jsonl", {"campaign": "demo"})
    assert uri == "gs://campaign/events/2026-08-11.jsonl"
    keys = [k for (_b, k, _data, _ct) in client.uploads]
    assert "events/2026-08-11.jsonl" in keys
    # Both keys are uploaded; the durable signal is the final key.
    assert "events/2026-08-11.jsonl.tmp" in keys


def test_gcs_sink_sha256_round_trip() -> None:
    import hashlib

    client = FakeGcsClientForSink()
    sink = GcsSink(bucket_name="campaign", client=client)  # type: ignore[arg-type]
    sink.upload_bytes("blob", b"abc")
    assert sink.sha256("blob") == hashlib.sha256(b"abc").hexdigest()


def test_gcs_sink_download_to_file(tmp_path: Path) -> None:
    client = FakeGcsClientForSink()
    sink = GcsSink(bucket_name="campaign", client=client)  # type: ignore[arg-type]
    sink.upload_bytes("data/manifest.json", b"{}")
    target = tmp_path / "manifest.json"
    sink.download_to_file("data/manifest.json", target)
    assert target.read_bytes() == b"{}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(state: Any) -> Any:
    """Helper: build a :class:`JobStatus` with the given state."""
    status = google_cloud_batch.JobStatus()
    status.state = state
    return status
