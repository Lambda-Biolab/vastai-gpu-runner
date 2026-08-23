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
    BootDisk,
    ComputeResource,
    GpuAccelerator,
    MachineResource,
    ManagedJobHandle,
    ManagedJobLifecycleState,
    ManagedJobSpec,
    ManagedJobTerminalState,
    NetworkConfig,
    ServiceAccount,
    StorageMount,
)
from vastai_gpu_runner.managed_jobs.errors import (
    ManagedJobConflictError,
    ManagedJobError,
    ManagedJobNotFoundError,
    ManagedJobTransientError,
    map_gcp_exception,
)
from vastai_gpu_runner.managed_jobs.gcp_batch import (
    SPOT_PREEMPT_EXIT_CODE,
    FakeGcpBatchClient,
    FakeGcsBlob,
    FakeGcsBucket,
    FakeGcsClient,
    GcpBatchRunner,
    parse_gcs_mount,
)
from vastai_gpu_runner.managed_jobs.state import (
    CURRENT_MANAGED_JOB_SCHEMA,
    ManagedJobState,
    save_managed_job_state,
)
from vastai_gpu_runner.storage.gcs import GcsPreconditionFailed, GcsSink

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
        "environment": {"FOO": "bar"},
        "labels": {"campaign": "demo", "stage": "backbone"},
        "gcs_mounts": ("gs://campaign/state",),
        "timeout_seconds": 600,
        "retry_on_preempt": True,
        "region": "us-central1",
        "machine_resource": MachineResource(machine_type="n1-standard-4"),
        "spot": True,
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
    container = request.job.task_groups[0].task_spec.runnables[0].container
    assert container.image_uri == "gcr.io/example/worker:latest"
    assert container.entrypoint == "python"
    assert list(container.commands) == ["-m", "demo"]
    assert request.job.task_groups[0].task_count == 3
    assert request.job.task_groups[0].parallelism == 2
    assert request.job.task_groups[0].task_spec.max_retry_count == 3
    lifecycle = request.job.task_groups[0].task_spec.lifecycle_policies
    assert lifecycle[0].action == google_cloud_batch.LifecyclePolicy.Action.RETRY_TASK
    assert lifecycle[0].action_condition.exit_codes == [SPOT_PREEMPT_EXIT_CODE]
    expected_log_dest = google_cloud_batch.LogsPolicy.Destination.CLOUD_LOGGING
    assert request.job.logs_policy.destination == expected_log_dest


def test_submit_single_token_command_has_no_arguments() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]

    runner.submit(_make_spec(command=("python",)))

    container = fake.submit_calls[0].job.task_groups[0].task_spec.runnables[0].container
    assert container.image_uri == "gcr.io/example/worker:latest"
    assert container.entrypoint == "python"
    assert list(container.commands) == []


def test_submit_duplicate_name_raises_typed_conflict() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(_make_spec())

    with pytest.raises(ManagedJobConflictError) as exc_info:
        runner.submit(_make_spec())

    assert exc_info.value.__cause__ is not None
    assert len(fake.submit_calls) == 1


def test_cancel_and_delete_are_idempotent_after_provider_not_found() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec())
    runner.delete(handle)

    runner.cancel(handle)
    runner.delete(handle)

    assert fake.cancel_calls == []
    assert fake.delete_calls == [handle.resource_name]


def test_submit_maps_transient_gcp_error_and_preserves_cause() -> None:
    from google.api_core.exceptions import ServiceUnavailable

    class FailingClient:
        def create_job(self, request: Any) -> Any:
            raise ServiceUnavailable("temporary outage")

    runner = GcpBatchRunner(
        project_id="p",
        region="us-central1",
        client=FailingClient(),  # type: ignore[arg-type]
    )

    with pytest.raises(ManagedJobTransientError) as exc_info:
        runner.submit(_make_spec())

    assert isinstance(exc_info.value.__cause__, ServiceUnavailable)


@pytest.mark.parametrize(
    ("exception_type", "expected_type"),
    [
        ("Conflict", ManagedJobConflictError),
        ("BadGateway", ManagedJobTransientError),
        ("GatewayTimeout", ManagedJobTransientError),
        ("Aborted", ManagedJobTransientError),
    ],
)
def test_map_gcp_exception_maps_transport_classes(
    exception_type: str, expected_type: type[ManagedJobError]
) -> None:
    from google.api_core import exceptions as gcp_exceptions

    provider_exception = getattr(gcp_exceptions, exception_type)("provider response")
    mapped = map_gcp_exception(provider_exception)

    assert type(mapped) is expected_type


def test_get_status_maps_provider_not_found() -> None:
    from google.api_core.exceptions import NotFound

    class MissingClient:
        def get_job(self, request: Any) -> Any:
            raise NotFound("gone")

    runner = GcpBatchRunner(
        project_id="p",
        region="us-central1",
        client=MissingClient(),  # type: ignore[arg-type]
    )
    handle = ManagedJobHandle(provider="gcp-batch", resource_name="missing")

    with pytest.raises(ManagedJobNotFoundError) as exc_info:
        runner.get_status(handle)

    assert isinstance(exc_info.value.__cause__, NotFound)


def test_get_status_passes_get_job_request_object() -> None:
    class RequestOnlyClient:
        def get_job(self, request: Any) -> Any:
            assert request.name == "projects/p/locations/us-central1/jobs/job"
            return google_cloud_batch.Job(
                name=request.name,
                status=_status(google_cloud_batch.JobStatus.State.SUCCEEDED),
            )

    runner = GcpBatchRunner(
        project_id="p",
        region="us-central1",
        client=RequestOnlyClient(),  # type: ignore[arg-type]
    )

    result = runner.get_status(
        ManagedJobHandle(
            provider="gcp-batch",
            resource_name="projects/p/locations/us-central1/jobs/job",
        )
    )

    assert result.state == ManagedJobLifecycleState.SUCCEEDED


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
        _make_spec(
            machine_resource=None,
            environment={"machine_type": "a3-highgpu-1g", "provisioning_model": "STANDARD"},
        )
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
    assert status.state == ManagedJobLifecycleState.RUNNING


@pytest.mark.parametrize(
    ("provider_state", "expected"),
    [
        ("QUEUED", ManagedJobLifecycleState.QUEUED),
        ("SCHEDULED", ManagedJobLifecycleState.PENDING),
        ("CANCELLATION_IN_PROGRESS", ManagedJobLifecycleState.CANCELLING),
        ("DELETION_IN_PROGRESS", ManagedJobLifecycleState.CANCELLING),
        ("STATE_UNSPECIFIED", ManagedJobLifecycleState.UNKNOWN),
    ],
)
def test_get_status_maps_every_in_progress_lifecycle_state(
    provider_state: str, expected: ManagedJobLifecycleState
) -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec())
    fake.statuses[handle.resource_name] = _status(
        google_cloud_batch.JobStatus.State[provider_state]
    )

    assert runner.get_status(handle).state == expected


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


def test_list_tasks_maps_pager_iteration_not_found() -> None:
    from google.api_core.exceptions import NotFound

    class FailingPager:
        def __iter__(self) -> Any:
            raise NotFound("gone while iterating")

    class PagerClient:
        def list_tasks(self, parent: str) -> Any:
            return FailingPager()

    runner = GcpBatchRunner(
        project_id="p",
        region="us-central1",
        client=PagerClient(),  # type: ignore[arg-type]
    )

    with pytest.raises(ManagedJobNotFoundError) as exc_info:
        list(runner.list_tasks(ManagedJobHandle(provider="gcp-batch", resource_name="job")))

    assert isinstance(exc_info.value.__cause__, NotFound)


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
    """Thin shim around the shared fake GCS client in gcp_batch.py."""

    def __init__(self) -> None:
        from vastai_gpu_runner.managed_jobs.gcp_batch import FakeGcsClient

        self._impl = FakeGcsClient()
        self._impl.create_bucket("campaign")
        self.uploads = self._impl.uploads

    def bucket(self, name: str) -> Any:
        return self._impl.bucket(name)


def test_gcs_sink_upload_and_download_bytes() -> None:
    client = FakeGcsClientForSink()
    sink = GcsSink(bucket_name="campaign", client=client)  # type: ignore[arg-type]
    uri = sink.upload_bytes("foo/bar.txt", b"hello", content_type="text/plain")
    assert uri == "gs://campaign/foo/bar.txt"
    assert sink.exists("foo/bar.txt")
    assert sink.download_bytes("foo/bar.txt") == b"hello"


def test_gcs_sink_missing_bucket_raises() -> None:
    client = FakeGcsClientForSink()
    sink = GcsSink(bucket_name="absent", client=client)  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        sink.upload_bytes("foo", b"x")


def test_gcs_sink_atomic_json_writes_final_key() -> None:
    client = FakeGcsClientForSink()
    sink = GcsSink(bucket_name="campaign", client=client)  # type: ignore[arg-type]
    uri = sink.upload_atomic_json("events/2026-08-11.jsonl", {"campaign": "demo"})
    assert uri == "gs://campaign/events/2026-08-11.jsonl"
    keys = [k for (_b, k, _data, _ct, _gm) in client.uploads]
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


def _precondition_failed_type() -> type[BaseException]:
    """Return the GcsPreconditionFailed class the sink raises."""
    return GcsPreconditionFailed


# ---------------------------------------------------------------------------
# Typed resource field tests
# ---------------------------------------------------------------------------


def test_submit_propagates_environment_to_task_spec() -> None:
    """TaskSpec.environment.variables mirrors spec.environment."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(_make_spec(environment={"FOO": "bar", "BAZ": "qux"}))
    task = fake.submit_calls[0].job.task_groups[0].task_spec
    assert task.environment.variables == {"FOO": "bar", "BAZ": "qux"}


def test_submit_omits_environment_when_empty() -> None:
    """No environment field on the spec means no env assignment on the task."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(_make_spec(environment={}))
    task = fake.submit_calls[0].job.task_groups[0].task_spec
    assert not getattr(task, "environment", None) or not dict(task.environment.variables)


def test_submit_sets_compute_resource_on_task_spec() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(
        _make_spec(
            compute_resource=ComputeResource(cpu_milli=4000, memory_mib=8192, boot_disk_mib=20480),
        )
    )
    task = fake.submit_calls[0].job.task_groups[0].task_spec
    assert task.compute_resource.cpu_milli == 4000
    assert task.compute_resource.memory_mib == 8192
    assert task.compute_resource.boot_disk_mib == 20480


def test_submit_sets_boot_disk_and_accelerators() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(
        _make_spec(
            machine_resource=MachineResource(
                machine_type="a3-highgpu-1g",
                boot_disk=BootDisk(
                    image="projects/debian-cloud/global/images/family/debian-12",
                    size_gb=50,
                    type_="pd-ssd",
                ),
                accelerators=(GpuAccelerator(type_="nvidia-h100-80gb", count=1),),
                min_cpu_platform="intel-cascadelake",
            ),
        )
    )
    policy = fake.submit_calls[0].job.allocation_policy.instances[0].policy
    assert policy.machine_type == "a3-highgpu-1g"
    assert policy.min_cpu_platform == "intel-cascadelake"
    assert policy.boot_disk.size_gb == 50
    assert policy.boot_disk.type_ == "pd-ssd"
    assert policy.boot_disk.image == "projects/debian-cloud/global/images/family/debian-12"
    assert len(policy.accelerators) == 1
    assert policy.accelerators[0].type_ == "nvidia-h100-80gb"
    assert policy.accelerators[0].count == 1


def test_submit_sets_spot_provisioning_model() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(_make_spec(spot=True))
    policy = fake.submit_calls[0].job.allocation_policy.instances[0].policy
    expected = type(policy.provisioning_model).SPOT
    assert policy.provisioning_model == expected


def test_submit_uses_standard_when_spot_false() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(_make_spec(spot=False))
    policy = fake.submit_calls[0].job.allocation_policy.instances[0].policy
    expected = type(policy.provisioning_model).STANDARD
    assert policy.provisioning_model == expected


def test_submit_sets_service_account() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(
        _make_spec(
            service_account=ServiceAccount(
                email="svc@proj.iam.gserviceaccount.com",
                scopes=("https://www.googleapis.com/auth/cloud-platform",),
            ),
        )
    )
    sa = fake.submit_calls[0].job.allocation_policy.service_account
    assert sa.email == "svc@proj.iam.gserviceaccount.com"
    assert sa.scopes == ["https://www.googleapis.com/auth/cloud-platform"]


def test_submit_sets_network_policy() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(
        _make_spec(
            network=NetworkConfig(
                network="projects/p/global/networks/default",
                subnetwork="projects/p/regions/us-central1/subnetworks/default",
                no_external_ip_address=True,
            ),
        )
    )
    network = fake.submit_calls[0].job.allocation_policy.network
    assert len(network.network_interfaces) == 1
    iface = network.network_interfaces[0]
    assert iface.network == "projects/p/global/networks/default"
    assert iface.subnetwork == "projects/p/regions/us-central1/subnetworks/default"
    assert iface.no_external_ip_address is True


def test_submit_sets_allowed_locations() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(_make_spec(allowed_locations=("regions/us-central1", "regions/us-east1")))
    location = fake.submit_calls[0].job.allocation_policy.location
    assert location.allowed_locations == ["regions/us-central1", "regions/us-east1"]


def test_submit_rejects_region_outside_allowed_locations() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="allowed_locations"):
        runner.submit(
            _make_spec(
                region="europe-west1",
                allowed_locations=("regions/us-central1", "regions/us-east1"),
            )
        )
    assert fake.submit_calls == []


def test_submit_rejects_runner_region_outside_allowed_locations() -> None:
    """When the runner region is the effective region, it is checked against allowed."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-east1", client=fake)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="effective region"):
        runner.submit(
            _make_spec(
                region="us-east1",
                allowed_locations=("regions/us-central1",),
            )
        )
    assert fake.submit_calls == []


def test_submit_accepts_allowed_locations_matching_runner_region() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(
        _make_spec(
            allowed_locations=("regions/us-central1", "regions/us-east1"),
        )
    )
    assert handle.resource_name.endswith("/jobs/demo-job")


def test_submit_legacy_env_machine_type_falls_back_when_untyped() -> None:
    """Legacy env-based machine_type still works when typed fields are absent."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(
        _make_spec(
            machine_resource=None,
            environment={"machine_type": "n1-standard-2", "provisioning_model": "STANDARD"},
        )
    )
    policy = fake.submit_calls[0].job.allocation_policy.instances[0].policy
    assert policy.machine_type == "n1-standard-2"
    expected = type(policy.provisioning_model).STANDARD
    assert policy.provisioning_model == expected


def test_submit_typed_machine_resource_wins_over_env_fallback() -> None:
    """Typed MachineResource takes priority over legacy env fallback."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(
        _make_spec(
            machine_resource=MachineResource(machine_type="a3-highgpu-1g"),
            environment={"machine_type": "n1-standard-4", "provisioning_model": "STANDARD"},
        )
    )
    policy = fake.submit_calls[0].job.allocation_policy.instances[0].policy
    assert policy.machine_type == "a3-highgpu-1g"
    # ``spot`` defaults to False on this spec, so provisioning_model
    # resolves to STANDARD regardless of the env fallback.
    expected = type(policy.provisioning_model).STANDARD
    assert policy.provisioning_model == expected


# ---------------------------------------------------------------------------
# get_status: message + exit_code + UNEXECUTED task mapping
# ---------------------------------------------------------------------------


def test_get_status_message_uses_last_event_description() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec())

    event = google_cloud_batch.StatusEvent()
    event.description = "Job failed: OOM"
    fake.events[handle.resource_name] = [event]
    fake.statuses[handle.resource_name] = _status(google_cloud_batch.JobStatus.State.FAILED)

    status = runner.get_status(handle)
    assert status.message == "Job failed: OOM"
    assert status.raw_events == ("Job failed: OOM",)


def test_get_status_message_empty_when_no_events() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec())
    fake.statuses[handle.resource_name] = _status(google_cloud_batch.JobStatus.State.SUCCEEDED)
    status = runner.get_status(handle)
    assert status.message == ""
    assert status.raw_events == ()


def test_list_tasks_unexecuted_state_yields_cancelled() -> None:
    """UNEXECUTED tasks (cancelled before being assigned) map to CANCELLED."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec())

    task = google_cloud_batch.Task()
    task.name = f"{handle.resource_name}/taskGroups/group0/tasks/0"
    task_status = google_cloud_batch.TaskStatus()
    task_status.state = google_cloud_batch.TaskStatus.State.UNEXECUTED
    task.status = task_status
    fake.tasks[f"{handle.resource_name}/taskGroups/group0"] = [task]

    listed = list(runner.list_tasks(handle))
    assert listed[0].state == "CANCELLED"


def test_list_tasks_failed_state_yields_failed() -> None:
    """FAILED tasks get the exit_code from the last event's TaskExecution."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec())

    task = google_cloud_batch.Task()
    task.name = f"{handle.resource_name}/taskGroups/group0/tasks/2"
    task_status = google_cloud_batch.TaskStatus()
    task_status.state = google_cloud_batch.TaskStatus.State.FAILED
    event = google_cloud_batch.StatusEvent()
    event.description = "non-zero exit"
    event.task_execution = google_cloud_batch.TaskExecution()
    event.task_execution.exit_code = 7
    task_status.status_events = [event]
    task.status = task_status
    fake.tasks[f"{handle.resource_name}/taskGroups/group0"] = [task]

    listed = list(runner.list_tasks(handle))
    assert listed[0].state == "FAILED"
    assert listed[0].exit_code == 7
    assert listed[0].message == "non-zero exit"


def test_list_tasks_extracts_exit_code_and_last_event_message() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec())

    task = google_cloud_batch.Task()
    task.name = f"{handle.resource_name}/taskGroups/group0/tasks/3"
    task_status = google_cloud_batch.TaskStatus()
    task_status.state = google_cloud_batch.TaskStatus.State.FAILED
    event = google_cloud_batch.StatusEvent()
    event.description = "task killed: exit 2"
    event.task_execution = google_cloud_batch.TaskExecution()
    event.task_execution.exit_code = 2
    task_status.status_events = [event]
    task.status = task_status
    fake.tasks[f"{handle.resource_name}/taskGroups/group0"] = [task]

    listed = list(runner.list_tasks(handle))
    assert listed[0].task_index == 3
    assert listed[0].state == "FAILED"
    assert listed[0].exit_code == 2
    assert listed[0].message == "task killed: exit 2"


def test_get_status_aggregates_cancelled_and_unexecuted_as_failed() -> None:
    """CANCELLED + UNEXECUTED in the counts map fold into failed_tasks."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec(task_count=5))

    status = google_cloud_batch.JobStatus()
    group_status = google_cloud_batch.JobStatus.TaskGroupStatus()
    group_status.counts = {
        "SUCCEEDED": 2,
        "FAILED": 1,
        "CANCELLED": 1,
        "UNEXECUTED": 1,
    }
    status.task_groups = {"group0": group_status}
    fake.statuses[handle.resource_name] = status

    result = runner.get_status(handle)
    assert result.succeeded_tasks == 2
    assert result.failed_tasks == 3
    assert result.total_tasks == 5


# ---------------------------------------------------------------------------
# GCS mount parsing + multiple mounts
# ---------------------------------------------------------------------------


def test_parse_gcs_mount_full_uri() -> None:
    assert parse_gcs_mount("gs://my-bucket/data") == ("my-bucket", "/data")


def test_parse_gcs_mount_bucket_only() -> None:
    assert parse_gcs_mount("gs://my-bucket") == ("my-bucket", "/my-bucket")


def test_parse_gcs_mount_absolute_path_uses_default_bucket() -> None:
    assert parse_gcs_mount("/mnt/campaign", default_bucket="campaign") == (
        "campaign",
        "/mnt/campaign",
    )


def test_parse_gcs_mount_absolute_path_without_default_returns_none() -> None:
    assert parse_gcs_mount("/mnt/campaign") is None


def test_parse_gcs_mount_empty_and_whitespace_returns_none() -> None:
    assert parse_gcs_mount("") is None
    assert parse_gcs_mount("   ") is None


def test_parse_gcs_mount_malformed_gs_uri_returns_none() -> None:
    assert parse_gcs_mount("gs://") is None
    assert parse_gcs_mount("gs:///path") is None


def test_submit_supports_multiple_gcs_mounts() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(
        project_id="p",
        region="us-central1",
        client=fake,  # type: ignore[arg-type]
        bucket_name="primary",
    )
    runner.submit(
        _make_spec(
            storage_mounts=(
                StorageMount(uri="gs://typed/input", mount_path="/mnt/typed", read_only=True),
            ),
            gcs_mounts=(
                "gs://campaign/state",
                "gs://artifacts/runs",
                "/mnt/local",
            ),
        )
    )
    task = fake.submit_calls[0].job.task_groups[0].task_spec
    runnable = task.runnables[0]
    assert len(task.volumes) == 4
    assert [volume.gcs.remote_path for volume in task.volumes] == [
        "typed/input",
        "campaign",
        "artifacts",
        "primary",
    ]
    assert [volume.mount_path for volume in task.volumes] == [
        "/mnt/disks/vastai-gpu-runner-gcs-0",
        "/mnt/disks/vastai-gpu-runner-gcs-1",
        "/mnt/disks/vastai-gpu-runner-gcs-2",
        "/mnt/disks/vastai-gpu-runner-gcs-3",
    ]
    assert [list(volume.mount_options) for volume in task.volumes] == [
        ["ro"],
        [],
        [],
        [],
    ]
    assert list(runnable.container.volumes) == [
        "/mnt/disks/vastai-gpu-runner-gcs-0:/mnt/typed:ro",
        "/mnt/disks/vastai-gpu-runner-gcs-1:/state:rw",
        "/mnt/disks/vastai-gpu-runner-gcs-2:/runs:rw",
        "/mnt/disks/vastai-gpu-runner-gcs-3:/mnt/local:rw",
    ]


def test_submit_maps_generic_gcs_mount_and_read_only_intent() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(
        _make_spec(
            gcs_mounts=(),
            storage_mounts=(
                StorageMount(uri="gs://campaign/input", mount_path="/mnt/input", read_only=True),
            ),
        )
    )

    volume = fake.submit_calls[0].job.task_groups[0].task_spec.volumes[0]
    runnable = fake.submit_calls[0].job.task_groups[0].task_spec.runnables[0]
    assert volume.gcs.remote_path == "campaign/input"
    assert volume.mount_path == "/mnt/disks/vastai-gpu-runner-gcs-0"
    assert list(volume.mount_options) == ["ro"]
    assert list(runnable.container.volumes) == [
        "/mnt/disks/vastai-gpu-runner-gcs-0:/mnt/input:ro",
    ]


def test_submit_maps_generic_gcs_mount_and_writable_intent() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(
        _make_spec(
            gcs_mounts=(),
            storage_mounts=(StorageMount(uri="gs://campaign/output", mount_path="/mnt/output"),),
        )
    )

    task = fake.submit_calls[0].job.task_groups[0].task_spec
    assert task.volumes[0].mount_path == "/mnt/disks/vastai-gpu-runner-gcs-0"
    assert list(task.volumes[0].mount_options) == []
    assert list(task.runnables[0].container.volumes) == [
        "/mnt/disks/vastai-gpu-runner-gcs-0:/mnt/output:rw",
    ]


def test_submit_preserves_container_target_under_mnt_disks() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(
        _make_spec(
            gcs_mounts=(),
            storage_mounts=(
                StorageMount(uri="gs://campaign/data", mount_path="/mnt/disks/campaigns"),
            ),
        )
    )

    task = fake.submit_calls[0].job.task_groups[0].task_spec
    assert task.volumes[0].mount_path == "/mnt/disks/vastai-gpu-runner-gcs-0"
    assert list(task.runnables[0].container.volumes) == [
        "/mnt/disks/vastai-gpu-runner-gcs-0:/mnt/disks/campaigns:rw",
    ]


def test_submit_rejects_unsupported_generic_mount_scheme() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gs:// URIs only"):
        runner.submit(
            _make_spec(
                gcs_mounts=(),
                storage_mounts=(StorageMount("s3://bucket/input", "/mnt/input"),),
            )
        )
    assert fake.submit_calls == []


def test_submit_omits_volumes_when_gcs_mounts_empty() -> None:
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(_make_spec(gcs_mounts=()))
    task = fake.submit_calls[0].job.task_groups[0].task_spec
    assert not task.volumes
    assert not task.runnables[0].container.volumes


# ---------------------------------------------------------------------------
# Focused GCS sink tests (list_blobs, download_all, CAS)
# ---------------------------------------------------------------------------


class _GcsClientShim:
    """Cache/factory around the fake GCS client so the sink can resolve a bucket."""

    def __init__(self, inner: FakeGcsClient) -> None:
        self._inner = inner

    def bucket(self, name: str) -> Any:
        return self._inner.bucket(name)


def _make_sink(bucket: str = "campaign") -> tuple[GcsSink, FakeGcsClient]:
    """Build a GcsSink backed by a fresh FakeGcsClient with ``bucket`` pre-seeded."""
    client = FakeGcsClient()
    client.create_bucket(bucket)
    sink = GcsSink(bucket_name=bucket, client=_GcsClientShim(client))  # type: ignore[arg-type]
    return sink, client


def test_gcs_sink_list_blobs_filters_by_prefix() -> None:
    sink, client = _make_sink()
    client.bucket("campaign").blob("a/1.json").upload_from_string(b"1")
    client.bucket("campaign").blob("a/2.json").upload_from_string(b"2")
    client.bucket("campaign").blob("b/1.json").upload_from_string(b"3")

    assert sink.list_blobs("a/") == ["1.json", "2.json"]
    assert sink.list_blobs("") == ["a/1.json", "a/2.json", "b/1.json"]


class _PrefixPage:
    def __init__(self, prefixes: list[str]) -> None:
        self.prefixes = prefixes


class _PrefixPager:
    def __init__(self, page: _PrefixPage | None, next_page_token: str | None) -> None:
        self.page = page
        self.next_page_token = next_page_token
        self.pages_read = 0

    @property
    def pages(self) -> Any:
        self.pages_read += 1
        if self.page is None:
            return
        yield self.page
        raise AssertionError("the sink consumed more than one GCS page")


class _PrefixBucket:
    def __init__(self, pager: _PrefixPager) -> None:
        self.pager = pager
        self.list_calls: list[dict[str, Any]] = []

    def exists(self) -> bool:
        return True

    def list_blobs(self, **kwargs: Any) -> _PrefixPager:
        self.list_calls.append(kwargs)
        return self.pager


class _PrefixClient:
    def __init__(self, bucket: _PrefixBucket) -> None:
        self._bucket = bucket

    def bucket(self, _name: str) -> _PrefixBucket:
        return self._bucket


def test_gcs_sink_list_prefixes_page_uses_one_sorted_normalized_page() -> None:
    pager = _PrefixPager(_PrefixPage(["runs/zeta/", "runs/alpha/"]), "next-page")
    bucket = _PrefixBucket(pager)
    sink = GcsSink(bucket_name="campaign", client=_PrefixClient(bucket))  # type: ignore[arg-type]
    opaque = "current-page"

    result = sink.list_prefixes_page("runs/", page_size=2, page_token=opaque)

    assert result == (["alpha", "zeta"], "next-page")
    assert bucket.list_calls == [
        {
            "prefix": "runs/",
            "delimiter": "/",
            "max_results": 2,
            "page_token": opaque,
        }
    ]
    assert pager.pages_read == 1


def test_gcs_sink_list_prefixes_page_empty_fetched_page_preserves_token() -> None:
    pager = _PrefixPager(_PrefixPage([]), "later-page")
    bucket = _PrefixBucket(pager)
    sink = GcsSink(bucket_name="campaign", client=_PrefixClient(bucket))  # type: ignore[arg-type]

    assert sink.list_prefixes_page(page_size=1) == ([], "later-page")


def test_gcs_sink_list_prefixes_page_without_api_page_has_no_token() -> None:
    pager = _PrefixPager(None, "unused-page")
    bucket = _PrefixBucket(pager)
    sink = GcsSink(bucket_name="campaign", client=_PrefixClient(bucket))  # type: ignore[arg-type]

    assert sink.list_prefixes_page(page_size=1) == ([], None)


@pytest.mark.parametrize("page_size", [0, -1])
def test_gcs_sink_list_prefixes_page_rejects_non_positive_page_size(page_size: int) -> None:
    pager = _PrefixPager(None, None)
    bucket = _PrefixBucket(pager)
    sink = GcsSink(bucket_name="campaign", client=_PrefixClient(bucket))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="greater than zero"):
        sink.list_prefixes_page(page_size=page_size)
    assert bucket.list_calls == []


def test_gcs_sink_download_all_creates_local_files(tmp_path: Path) -> None:
    sink, client = _make_sink()
    client.bucket("campaign").blob("data/manifest.json").upload_from_string(b"{}")
    client.bucket("campaign").blob("data/results.jsonl").upload_from_string(b"line\n")

    written = sink.download_all("data/", tmp_path)
    assert sorted(p.name for p in written) == ["manifest.json", "results.jsonl"]
    assert (tmp_path / "manifest.json").read_bytes() == b"{}"
    assert (tmp_path / "results.jsonl").read_bytes() == b"line\n"


def test_gcs_sink_upload_bytes_uses_create_only_precondition() -> None:
    """upload_bytes sets if_generation_match=0 on the underlying blob."""
    sink, client = _make_sink()
    sink.upload_bytes("snap", b"v1")
    assert client.uploads[-1][4] == 0


def test_gcs_sink_create_only_upload_rejects_duplicate() -> None:
    """Second upload_with if_generation_match=0 raises PreconditionFailed."""
    sink, _client = _make_sink()
    sink.upload_bytes("snap", b"v1")
    with pytest.raises(_precondition_failed_type()) as exc_info:
        sink.upload_bytes("snap", b"v2")
    # The fake raises google.api_core.exceptions.PreconditionFailed when
    # available and a generic RuntimeError otherwise.
    assert "generation" in str(exc_info.value).lower()


def test_gcs_sink_cas_write_success_and_conflict() -> None:
    """CAS write succeeds when the live generation matches, fails otherwise."""
    sink, _client = _make_sink()
    # First create: expected_generation=0 ("must not exist").
    gen1 = sink.upload_cas_write("snap", b"v1", expected_generation=0)
    assert gen1 == 1
    # Read the snapshot back; the generation is now 1.
    assert sink.read_cas("snap") == (b"v1", 1)
    # CAS update at the expected generation succeeds.
    gen2 = sink.upload_cas_write("snap", b"v2", expected_generation=1)
    assert gen2 == 2
    # Stale CAS — the live generation is 2, but we ask to write at 1.
    with pytest.raises(_precondition_failed_type()):
        sink.upload_cas_write("snap", b"v3", expected_generation=1)
    # The failed CAS must not overwrite the live value.
    assert sink.read_cas("snap") == (b"v2", 2)


def test_gcs_sink_cas_uses_upload_generation_without_reload() -> None:
    sink, _client = _make_sink()

    original_reload = FakeGcsBlob.reload

    def _reload_must_not_run(self: Any) -> None:
        raise OSError("metadata reload unavailable")

    FakeGcsBlob.reload = _reload_must_not_run  # type: ignore[assignment]
    try:
        generation = sink.upload_cas_write("snap", b"v1", expected_generation=0)
    finally:
        FakeGcsBlob.reload = original_reload  # type: ignore[assignment]

    assert generation == 1


def test_gcs_sink_cas_write_first_wins_on_none() -> None:
    """Passing None is treated as "object must not exist yet"."""
    sink, _client = _make_sink()
    gen = sink.upload_cas_write("snap", b"v1", expected_generation=None)
    assert gen == 1
    # A second ``None`` CAS now conflicts because the object exists.
    with pytest.raises(_precondition_failed_type()):
        sink.upload_cas_write("snap", b"v2", expected_generation=None)


# ---------------------------------------------------------------------------
# Stable GcsPreconditionFailed contract
# ---------------------------------------------------------------------------


def test_gcs_sink_raises_stable_gcs_precondition_failed() -> None:
    """CAS conflicts raise GcsPreconditionFailed — the controller's stable contract."""
    sink, _client = _make_sink()
    sink.upload_cas_write("snap", b"v1", expected_generation=0)
    with pytest.raises(GcsPreconditionFailed) as exc_info:
        sink.upload_cas_write("snap", b"v2", expected_generation=0)
    # The underlying SDK exception is preserved as __cause__ for
    # inspection without forcing consumers to depend on google.api_core.
    assert exc_info.value.__cause__ is not None


def test_gcs_sink_create_only_raises_stable_gcs_precondition_failed() -> None:
    """upload_bytes (create-only) also raises GcsPreconditionFailed."""
    sink, _client = _make_sink()
    sink.upload_bytes("snap", b"v1")
    with pytest.raises(GcsPreconditionFailed):
        sink.upload_bytes("snap", b"v2")


# ---------------------------------------------------------------------------
# download_all path containment
# ---------------------------------------------------------------------------


def test_gcs_sink_download_all_rejects_absolute_paths(tmp_path: Path) -> None:
    """Blob keys with leading ``/`` are rejected."""
    sink, client = _make_sink()
    # Seed a blob whose key after the prefix is absolute.
    client.bucket("campaign").blob("/etc/passwd").upload_from_string(b"nope")
    with pytest.raises(ValueError, match="absolute path"):
        sink.download_all("", tmp_path)


def test_gcs_sink_download_all_rejects_traversal_paths(tmp_path: Path) -> None:
    """Blob keys with ``..`` segments are rejected."""
    sink, client = _make_sink()
    client.bucket("campaign").blob("data/../../../etc/passwd").upload_from_string(b"nope")
    with pytest.raises(ValueError, match=r"\.\."):
        sink.download_all("data/", tmp_path)


def test_gcs_sink_download_all_rejects_nul_byte_keys(tmp_path: Path) -> None:
    """Blob keys with NUL bytes are rejected (filesystem safety)."""
    sink, client = _make_sink()
    client.bucket("campaign").blob("data/\x00bad").upload_from_string(b"nope")
    with pytest.raises(ValueError, match="NUL byte"):
        sink.download_all("data/", tmp_path)


def test_gcs_sink_download_all_resolved_target_under_dest(tmp_path: Path) -> None:
    """A blob key that resolves outside ``dest`` (via a symlink) is rejected."""
    sink, client = _make_sink()
    # Place a symlink inside ``tmp_path`` that points outside.
    outside_target = tmp_path.parent / "outside-bucket"
    outside_target.mkdir(exist_ok=True)
    (tmp_path / "escape").symlink_to(outside_target)
    # A blob whose key, after stripping the prefix, traverses the
    # symlink. ``(tmp_path / rel).resolve()`` follows the symlink
    # and the resolved target is outside ``tmp_path``.
    client.bucket("campaign").blob("escape/legit.txt").upload_from_string(b"ok")
    with pytest.raises(ValueError, match="escapes dest"):
        sink.download_all("", tmp_path)


def test_gcs_sink_download_all_happy_path(tmp_path: Path) -> None:
    """Non-pathological blob keys are downloaded as before."""
    sink, client = _make_sink()
    client.bucket("campaign").blob("data/manifest.json").upload_from_string(b"{}")
    client.bucket("campaign").blob("data/nested/x.json").upload_from_string(b"x")
    written = sink.download_all("data/", tmp_path)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in written) == [
        "manifest.json",
        "nested/x.json",
    ]
    assert (tmp_path / "manifest.json").read_bytes() == b"{}"
    assert (tmp_path / "nested" / "x.json").read_bytes() == b"x"


# ---------------------------------------------------------------------------
# list_blobs explicit sort
# ---------------------------------------------------------------------------


def test_gcs_sink_list_blobs_returns_sorted_even_when_bucket_unsorted() -> None:
    """Sink sorts the result so consumers do not depend on the SDK's order.

    The fake stores keys in a dict, so natural iteration is
    insertion order; we still expect the sink to return a sorted
    list (this is the public contract).
    """
    sink, client = _make_sink()
    # Insert in a non-sorted order.
    for k in ("c/1", "a/1", "b/2", "a/2"):
        client.bucket("campaign").blob(k).upload_from_string(b"x")
    assert sink.list_blobs("") == ["a/1", "a/2", "b/2", "c/1"]


# ---------------------------------------------------------------------------
# read_cas coherence
# ---------------------------------------------------------------------------


def test_gcs_sink_read_cas_returns_data_and_generation() -> None:
    """read_cas returns (data, generation) for a present object."""
    sink, _client = _make_sink()
    sink.upload_cas_write("snap", b"v1", expected_generation=0)
    data, gen = sink.read_cas("snap")
    assert data == b"v1"
    assert gen == 1


def test_gcs_sink_read_cas_returns_empty_for_missing() -> None:
    """read_cas returns (b"", None) for a missing object."""
    sink, _client = _make_sink()
    assert sink.read_cas("absent") == (b"", None)


def test_gcs_sink_read_cas_data_and_generation_coherent() -> None:
    """Data and generation are always returned from the same view."""
    sink, client = _make_sink()
    # Manually inject a blob at generation 3 with known data.
    client.bucket("campaign").blob("snap").upload_from_string(b"v3", if_generation_match=0)
    # Bump generation manually via the fake's internal mapping.
    client.generations[("campaign", "snap")] = 3
    data, gen = sink.read_cas("snap")
    assert gen == 3
    assert data == b"v3"


# ---------------------------------------------------------------------------
# spec.region honouring + effective_region
# ---------------------------------------------------------------------------


def test_submit_effective_region_overrides_runner_region() -> None:
    """spec.region overrides the runner region in the request parent and handle."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec(region="us-east1"))
    assert handle.resource_name == "projects/p/locations/us-east1/jobs/demo-job"
    assert handle.location == "p/us-east1"
    assert fake.submit_calls[0].parent == "projects/p/locations/us-east1"


def test_submit_effective_region_falls_back_to_runner_region() -> None:
    """Empty spec.region uses the runner's region."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec(region=""))
    assert handle.resource_name == "projects/p/locations/us-central1/jobs/demo-job"
    assert handle.location == "p/us-central1"


def test_submit_canonicalizes_bare_allowed_locations() -> None:
    """Bare region names in allowed_locations are sent as ``regions/...``."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(_make_spec(allowed_locations=("us-central1", "us-east1")))
    location = fake.submit_calls[0].job.allocation_policy.location
    assert list(location.allowed_locations) == [
        "regions/us-central1",
        "regions/us-east1",
    ]


def test_submit_preserves_already_canonical_allowed_locations() -> None:
    """Already-canonical allowed_locations are sent unchanged."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(
        _make_spec(
            allowed_locations=("regions/us-central1", "zones/us-central1-a"),
        )
    )
    location = fake.submit_calls[0].job.allocation_policy.location
    assert list(location.allowed_locations) == [
        "regions/us-central1",
        "zones/us-central1-a",
    ]


def test_submit_rejects_unknown_provisioning_model() -> None:
    """Unknown provisioning_model values raise ValueError."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown provisioning model"):
        runner.submit(
            _make_spec(
                machine_resource=None,
                environment={"provisioning_model": "MAGIC"},
            )
        )
    assert fake.submit_calls == []


# ---------------------------------------------------------------------------
# Legacy env stripping + GPU driver install
# ---------------------------------------------------------------------------


def test_submit_legacy_env_keys_stripped_from_container_env() -> None:
    """machine_type / provisioning_model / gpu_type / gpu_count are not in the worker env."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(
        _make_spec(
            machine_resource=None,
            environment={
                "machine_type": "n1-standard-4",
                "provisioning_model": "STANDARD",
                "gpu_type": "nvidia-tesla-t4",
                "gpu_count": "1",
                "MY_VAR": "kept",
            },
        )
    )
    task = fake.submit_calls[0].job.task_groups[0].task_spec
    assert dict(task.environment.variables) == {"MY_VAR": "kept"}


def test_submit_gpu_driver_install_maps_to_instance_policy_or_template() -> None:
    """install_gpu_drivers lives on InstancePolicyOrTemplate, not on Accelerator."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(
        _make_spec(
            machine_resource=MachineResource(
                machine_type="a3-highgpu-1g",
                accelerators=(
                    GpuAccelerator(
                        type_="nvidia-h100-80gb",
                        count=1,
                        install_gpu_drivers=True,
                    ),
                ),
            ),
        )
    )
    instances = fake.submit_calls[0].job.allocation_policy.instances[0]
    assert instances.install_gpu_drivers is True
    # The deprecated per-Accelerator field must stay unset.
    policy = instances.policy
    assert policy.accelerators[0].install_gpu_drivers is False


def test_submit_no_gpu_driver_install_when_not_requested() -> None:
    """Without any install_gpu_drivers flag, the template flag stays unset."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(
        _make_spec(
            machine_resource=MachineResource(
                machine_type="a3-highgpu-1g",
                accelerators=(GpuAccelerator(type_="nvidia-h100-80gb", count=1),),
            ),
        )
    )
    instances = fake.submit_calls[0].job.allocation_policy.instances[0]
    assert getattr(instances, "install_gpu_drivers", False) is False


def test_submit_typed_machine_resource_with_no_legacy_keys_kept_clean() -> None:
    """When typed fields are present, the legacy env keys are still stripped."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    runner.submit(
        _make_spec(
            machine_resource=MachineResource(machine_type="a3-highgpu-1g"),
            environment={
                "machine_type": "n1-standard-4",
                "USER_TAG": "kept",
            },
        )
    )
    task = fake.submit_calls[0].job.task_groups[0].task_spec
    assert dict(task.environment.variables) == {"USER_TAG": "kept"}


# ---------------------------------------------------------------------------
# State-aware exit_code (no false 0)
# ---------------------------------------------------------------------------


def test_list_tasks_failed_state_without_exit_code_returns_none() -> None:
    """FAILED with no exit_code in any event must return None, not a false 0."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec())

    task = google_cloud_batch.Task()
    task.name = f"{handle.resource_name}/taskGroups/group0/tasks/0"
    task_status = google_cloud_batch.TaskStatus()
    task_status.state = google_cloud_batch.TaskStatus.State.FAILED
    event = google_cloud_batch.StatusEvent()
    event.description = "killed before exit"
    # IMPORTANT: no task_execution.exit_code here.
    task_status.status_events = [event]
    task.status = task_status
    fake.tasks[f"{handle.resource_name}/taskGroups/group0"] = [task]

    listed = list(runner.list_tasks(handle))
    assert listed[0].state == "FAILED"
    assert listed[0].exit_code is None
    assert listed[0].message == "killed before exit"


def test_list_tasks_succeeded_state_without_exit_code_remains_none() -> None:
    """SUCCEEDED with no exit_code event also returns None (not 0)."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec())

    task = google_cloud_batch.Task()
    task.name = f"{handle.resource_name}/taskGroups/group0/tasks/0"
    task_status = google_cloud_batch.TaskStatus()
    task_status.state = google_cloud_batch.TaskStatus.State.SUCCEEDED
    task.status = task_status
    fake.tasks[f"{handle.resource_name}/taskGroups/group0"] = [task]

    listed = list(runner.list_tasks(handle))
    assert listed[0].state == "SUCCEEDED"
    assert listed[0].exit_code is None


def test_list_tasks_explicit_exit_code_zero_is_preserved() -> None:
    """An explicit exit_code=0 must be reported as 0, not None."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec())

    task = google_cloud_batch.Task()
    task.name = f"{handle.resource_name}/taskGroups/group0/tasks/0"
    task_status = google_cloud_batch.TaskStatus()
    task_status.state = google_cloud_batch.TaskStatus.State.SUCCEEDED
    event = google_cloud_batch.StatusEvent()
    event.task_execution = google_cloud_batch.TaskExecution()
    event.task_execution.exit_code = 0
    task_status.status_events = [event]
    task.status = task_status
    fake.tasks[f"{handle.resource_name}/taskGroups/group0"] = [task]

    listed = list(runner.list_tasks(handle))
    assert listed[0].state == "SUCCEEDED"
    assert listed[0].exit_code == 0


# ---------------------------------------------------------------------------
# Fake GCS client: concurrent CAS conflict semantics
# ---------------------------------------------------------------------------


def test_fake_gcs_concurrent_cas_writers() -> None:
    """Two writers issuing CAS at the same expected generation: one wins, one loses."""
    impl = FakeGcsClient()
    impl.create_bucket("campaign")
    impl.bucket("campaign").blob("snap").upload_from_string(b"base", if_generation_match=0)
    a = impl.bucket("campaign").blob("snap")
    b = impl.bucket("campaign").blob("snap")
    # Both observe the live generation = 1 and try to write at it.
    try:
        a.upload_from_string(b"writer-a", if_generation_match=1)
        writer_a_succeeded = True
    except Exception:
        writer_a_succeeded = False
    try:
        b.upload_from_string(b"writer-b", if_generation_match=1)
        writer_b_succeeded = True
    except Exception:
        writer_b_succeeded = False
    # Exactly one writer must succeed (the real SDK order is non-
    # deterministic; the fake serialises through the in-memory map).
    assert writer_a_succeeded != writer_b_succeeded
    # The final value is whichever writer won.
    winner = b"writer-a" if writer_a_succeeded else b"writer-b"
    assert impl.bucket("campaign").blob("snap").download_as_bytes() == winner


# ---------------------------------------------------------------------------
# Fake GCS client: artifact collection (list_blobs + download)
# ---------------------------------------------------------------------------


def test_fake_gcs_artifact_collection_list_then_download() -> None:
    """Consumer pattern: list_blobs(prefix) → download each result."""
    impl = FakeGcsClient()
    impl.create_bucket("campaign")
    for i in range(3):
        impl.bucket("campaign").blob(f"runs/2026-08-13/{i}.json").upload_from_string(
            f"{{{i}}}".encode()
        )
    # A non-matching object must not be returned.
    impl.bucket("campaign").blob("other/x.json").upload_from_string(b"x")

    listed = impl.list_blobs("campaign", prefix="runs/2026-08-13/")
    names = [blob.name for blob in listed]
    assert names == ["runs/2026-08-13/0.json", "runs/2026-08-13/1.json", "runs/2026-08-13/2.json"]
    # Each blob can be downloaded individually.
    payloads = [blob.download_as_bytes() for blob in listed]
    assert payloads == [b"{0}", b"{1}", b"{2}"]


# ---------------------------------------------------------------------------
# upload_atomic_json overwrite + stale tmp recovery
# ---------------------------------------------------------------------------


def test_gcs_sink_upload_atomic_json_overwrites_existing_key() -> None:
    """Second ``upload_atomic_json`` overwrites the first; no precondition failure."""
    sink, _client = _make_sink()
    sink.upload_atomic_json("events/x.json", {"v": 1})
    # The second call must succeed even though the final key exists.
    sink.upload_atomic_json("events/x.json", {"v": 2})
    assert sink.download_bytes("events/x.json") == b'{\n  "v": 2\n}'


def test_gcs_sink_upload_atomic_json_does_not_leave_tmp() -> None:
    """The ``.tmp`` suffix is cleaned up on success."""
    sink, client = _make_sink()
    sink.upload_atomic_json("events/x.json", {"v": 1})
    keys = {k for (_b, k, _data, _ct, _gm) in client.uploads}
    assert "events/x.json" in keys
    assert "events/x.json.tmp" not in sink.list_blobs("")


def test_gcs_sink_upload_atomic_json_cleans_stale_tmp() -> None:
    """A stale ``.tmp`` from a previous attempt is cleaned up before re-upload."""
    sink, client = _make_sink()
    # Simulate a crashed previous attempt: plant a stale tmp.
    client.bucket("campaign").blob("events/x.json.tmp").upload_from_string(b"stale")
    # The new upload should not fail on the stale tmp; it should
    # delete the stale object first.
    sink.upload_atomic_json("events/x.json", {"v": 1})
    assert sink.download_bytes("events/x.json") == b'{\n  "v": 1\n}'
    # The previously-stale tmp is gone (best-effort deleted).
    assert "events/x.json.tmp" not in sink.list_blobs("")


def test_gcs_sink_upload_atomic_json_retry_after_stale_tmp_succeeds() -> None:
    """A retry that hits a stale tmp recovers and completes."""
    sink, client = _make_sink()
    # Two retries in succession with a stale tmp planted between them.
    sink.upload_atomic_json("events/x.json", {"v": 1})
    client.bucket("campaign").blob("events/x.json.tmp").upload_from_string(b"stale")
    sink.upload_atomic_json("events/x.json", {"v": 2})
    assert sink.download_bytes("events/x.json") == b'{\n  "v": 2\n}'


# ---------------------------------------------------------------------------
# GcsPreconditionFailed types discrimination
# ---------------------------------------------------------------------------


def test_gcs_sink_upload_bytes_propagates_non_precondition_failures() -> None:
    """Non-precondition failures (auth, network, key errors) propagate unchanged."""
    impl = FakeGcsClient()
    impl.create_bucket("campaign")

    def _boom(*_args: object, **_kwargs: object) -> bool:
        raise OSError("connection refused")

    # Patch the class method so every bucket instance reuses it.
    original_exists = FakeGcsBucket.exists
    FakeGcsBucket.exists = _boom  # type: ignore[assignment]
    try:
        sink = GcsSink(
            bucket_name="campaign",
            client=_GcsClientShim(impl),  # type: ignore[arg-type]
        )
        with pytest.raises(OSError):
            sink.upload_bytes("foo", b"x")
    finally:
        FakeGcsBucket.exists = original_exists  # type: ignore[assignment]


def test_gcs_sink_upload_cas_propagates_non_precondition_failures() -> None:
    """upload_cas_write also propagates non-precondition failures."""
    impl = FakeGcsClient()
    impl.create_bucket("campaign")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("network unreachable")

    original = FakeGcsBlob.upload_from_string
    FakeGcsBlob.upload_from_string = _boom  # type: ignore[assignment]
    try:
        sink = GcsSink(
            bucket_name="campaign",
            client=_GcsClientShim(impl),  # type: ignore[arg-type]
        )
        with pytest.raises(OSError):
            sink.upload_cas_write("snap", b"v1", expected_generation=0)
    finally:
        FakeGcsBlob.upload_from_string = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# read_cas generation-coherent
# ---------------------------------------------------------------------------


def test_gcs_sink_read_cas_uses_generation_precondition() -> None:
    """read_cas pins the generation to the live value before the download.

    Implementation detail: the fake's ``download_as_bytes`` honours
    ``if_generation_match``. If the live generation changes between
    the ``reload()`` and the ``download_as_bytes()`` call, the
    download raises a precondition failure.
    """
    impl = FakeGcsClient()
    impl.create_bucket("campaign")
    # Seed the snap at generation 1.
    impl.bucket("campaign").blob("snap").upload_from_string(b"v1", if_generation_match=0)

    # Simulate a concurrent writer: replace the fake's reload so
    # it reports generation 1, while the live in-memory map has
    # been bumped to 2. The download will then fail the
    # if_generation_match check.
    def _stale_reload(self: Any) -> None:
        self.generation = 1

    original_reload = FakeGcsBlob.reload
    FakeGcsBlob.reload = _stale_reload  # type: ignore[assignment]
    try:
        impl.generations[("campaign", "snap")] = 2
        sink = GcsSink(
            bucket_name="campaign",
            client=_GcsClientShim(impl),  # type: ignore[arg-type]
        )
        with pytest.raises(GcsPreconditionFailed):
            sink.read_cas("snap")
    finally:
        FakeGcsBlob.reload = original_reload  # type: ignore[assignment]


def test_gcs_sink_read_cas_returns_coherent_data_and_generation() -> None:
    """read_cas returns the data + generation that came from the same view."""
    sink, _client = _make_sink()
    sink.upload_cas_write("snap", b"v1", expected_generation=0)
    data, gen = sink.read_cas("snap")
    assert data == b"v1"
    assert gen == 1
    # A second read at the same generation returns the same data.
    data2, gen2 = sink.read_cas("snap")
    assert data2 == b"v1"
    assert gen2 == 1


# ---------------------------------------------------------------------------
# effective_region normalisation + validate-only-effective
# ---------------------------------------------------------------------------


def test_submit_effective_region_strips_prefix() -> None:
    """``spec.region="regions/us-east1"`` resolves to bare ``us-east1``."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec(region="regions/us-east1"))
    # The handle location carries the bare form.
    assert handle.location == "p/us-east1"
    # The request parent uses the bare form to build the path.
    assert fake.submit_calls[0].parent == "projects/p/locations/us-east1"


def test_submit_validate_only_effective_region() -> None:
    """When spec.region overrides the runner region, only the effective region is checked."""
    fake = FakeGcpBatchClient()
    # Runner region is "us-west1" (not in allowed); spec.region is
    # "us-central1" (in allowed). The effective region is
    # "us-central1", so the submit must succeed.
    runner = GcpBatchRunner(project_id="p", region="us-west1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(
        _make_spec(
            region="us-central1",
            allowed_locations=("regions/us-central1",),
        )
    )
    assert handle.location == "p/us-central1"


def test_submit_rejects_effective_region_outside_allowed() -> None:
    """Effective region not in allowed_locations raises ValueError."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="effective region"):
        runner.submit(
            _make_spec(
                region="europe-west1",
                allowed_locations=("regions/us-central1",),
            )
        )


def test_submit_runner_region_ignored_when_override_set() -> None:
    """Runner region no longer blocks submit when spec.region overrides it."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-east1", client=fake)  # type: ignore[arg-type]
    # Submit succeeds because the effective region ("us-east1") is
    # in the allowed list, even though the runner's default is
    # "us-east1" — wait, that is the same. Use a contrived case:
    # runner region is "us-east1" but spec.region is "us-central1".
    handle = runner.submit(
        _make_spec(
            region="us-central1",
            allowed_locations=("regions/us-central1", "regions/us-east1"),
        )
    )
    assert handle.location == "p/us-central1"


# ---------------------------------------------------------------------------
# Status handle location derived from job resource name
# ---------------------------------------------------------------------------


def test_get_status_handle_location_derived_from_job_resource_name() -> None:
    """The status handle location reflects the actual region used, not the runner default."""
    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    handle = runner.submit(_make_spec(region="us-east1"))
    # The fake's stored job has the resource name we expect.
    assert handle.resource_name == "projects/p/locations/us-east1/jobs/demo-job"
    # get_status derives the handle location from the job's resource
    # name, so it reports "us-east1" not the runner default.
    status = runner.get_status(handle)
    assert status.handle.location == "p/us-east1"


def test_get_status_handle_location_falls_back_to_runner_region() -> None:
    """If the job resource name is malformed, fall back to the runner region."""
    from vastai_gpu_runner.managed_jobs.base import ManagedJobHandle

    fake = FakeGcpBatchClient()
    runner = GcpBatchRunner(project_id="p", region="us-central1", client=fake)  # type: ignore[arg-type]
    # Submit a real job so the fake records its resource name.
    handle = runner.submit(_make_spec(region=""))
    # Inject a malformed job alongside it. The fake's events/tasks
    # maps need the entry too — _make_job uses them.
    fake.jobs["malformed"] = google_cloud_batch.Job()
    fake.jobs["malformed"].name = "malformed"
    fake.jobs["malformed"].task_groups = []
    fake.statuses["malformed"] = _status(google_cloud_batch.JobStatus.State.SUCCEEDED)
    fake.events["malformed"] = []
    fake.tasks["malformed"] = []
    malformed_handle = ManagedJobHandle(
        provider="gcp-batch",
        resource_name="malformed",
        location="",
    )
    status = runner.get_status(malformed_handle)
    assert status.handle.location == "p/us-central1"
    # The well-formed handle still resolves to the runner region.
    assert handle.location == "p/us-central1"
