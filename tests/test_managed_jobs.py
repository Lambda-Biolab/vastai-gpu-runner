"""Tests for the provider-neutral managed-job interface and state loader."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from vastai_gpu_runner.managed_jobs import (
    CURRENT_MANAGED_JOB_SCHEMA,
    BootDisk,
    ComputeResource,
    GpuAccelerator,
    MachineResource,
    ManagedJobHandle,
    ManagedJobSpec,
    ManagedJobState,
    ManagedJobStateError,
    ManagedJobStatus,
    ManagedJobTerminalState,
    ManagedTaskStatus,
    NetworkConfig,
    ServiceAccount,
    load_managed_job_state,
    load_or_none,
    save_managed_job_state,
)


class FakeRunner:
    """In-memory provider used to exercise the protocol shape."""

    def __init__(self) -> None:
        self.submitted: list[ManagedJobSpec] = []
        self._states: dict[str, ManagedJobStatus] = {}

    @property
    def provider_name(self) -> str:
        return "fake-batch"

    def submit(self, spec: ManagedJobSpec) -> ManagedJobHandle:
        self.submitted.append(spec)
        handle = ManagedJobHandle(
            provider="fake-batch",
            resource_name=f"projects/test/jobs/{spec.name}",
        )
        self._states[handle.resource_name] = ManagedJobStatus(
            handle=handle,
            state=ManagedJobTerminalState.UNKNOWN,
            total_tasks=spec.task_count,
        )
        return handle

    def get_status(self, handle: ManagedJobHandle) -> ManagedJobStatus:
        return self._states[handle.resource_name]

    def list_tasks(self, handle: ManagedJobHandle) -> Iterable[ManagedTaskStatus]:
        return iter([ManagedTaskStatus(task_index=0, state="RUNNING")])

    def cancel(self, handle: ManagedJobHandle) -> None:
        self._states[handle.resource_name] = ManagedJobStatus(
            handle=handle,
            state=ManagedJobTerminalState.CANCELLED,
            total_tasks=self._states[handle.resource_name].total_tasks,
        )

    def delete(self, handle: ManagedJobHandle) -> None:
        self._states.pop(handle.resource_name, None)


def test_fake_runner_implements_protocol() -> None:
    runner = FakeRunner()
    # Protocol classes are runtime-checkable; verify the fake qualifies.
    from vastai_gpu_runner.managed_jobs.base import ManagedJobRunner

    assert isinstance(runner, ManagedJobRunner)


def test_spec_round_trip_dict() -> None:
    spec = ManagedJobSpec(
        name="hello",
        task_count=3,
        parallelism=2,
        image="gcr.io/x/y",
        command=("python", "-m", "hello"),
        environment={"FOO": "bar"},
        labels={"campaign": "demo"},
        gcs_mounts=("gs://example/x",),
        timeout_seconds=3600,
        retry_on_preempt=True,
        region="us-central1",
    )
    payload = spec.to_dict()
    assert payload["name"] == "hello"
    assert payload["command"] == ["python", "-m", "hello"]
    assert payload["environment"] == {"FOO": "bar"}
    assert payload["gcs_mounts"] == ["gs://example/x"]


def test_fake_runner_submit_and_poll() -> None:
    runner = FakeRunner()
    spec = ManagedJobSpec(name="a", task_count=2)
    handle = runner.submit(spec)
    assert handle.provider == "fake-batch"
    assert handle.resource_name.endswith("/a")
    status = runner.get_status(handle)
    assert status.state == ManagedJobTerminalState.UNKNOWN
    assert status.total_tasks == 2
    tasks = list(runner.list_tasks(handle))
    assert tasks == [ManagedTaskStatus(task_index=0, state="RUNNING")]


def test_fake_runner_cancel_updates_state() -> None:
    runner = FakeRunner()
    handle = runner.submit(ManagedJobSpec(name="b"))
    runner.cancel(handle)
    assert runner.get_status(handle).state == ManagedJobTerminalState.CANCELLED


def test_state_round_trip(tmp_path: Path) -> None:
    state = ManagedJobState(
        provider="fake-batch",
        resource_name="projects/x/jobs/y",
        location="us-central1",
        task_count=5,
        state="RUNNING",
        succeeded_tasks=3,
        failed_tasks=0,
        attempt=0,
        campaign_id="demo",
        stage_id="backbone",
    )
    target = tmp_path / "state.json"
    save_managed_job_state(state, target)
    loaded = load_managed_job_state(target)
    assert loaded.provider == "fake-batch"
    assert loaded.resource_name == "projects/x/jobs/y"
    assert loaded.task_count == 5
    assert loaded.state == "RUNNING"


def test_state_atomic_write_leaves_no_tmp(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "state.json"
    save_managed_job_state(ManagedJobState(resource_name="r"), target)
    assert target.is_file()
    assert list(target.parent.glob("*.tmp")) == []


def test_unknown_schema_rejected(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.json"
    bogus.write_text(json.dumps({"schema_version": 99, "provider": "x"}))
    with pytest.raises(ManagedJobStateError):
        load_managed_job_state(bogus)


def test_missing_required_keys_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": CURRENT_MANAGED_JOB_SCHEMA}))
    with pytest.raises(ManagedJobStateError, match="missing"):
        load_managed_job_state(bad)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ManagedJobStateError):
        load_managed_job_state(tmp_path / "absent.json")


def test_invalid_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ManagedJobStateError):
        load_managed_job_state(bad)


def test_v0_payload_migrates_to_v1(tmp_path: Path) -> None:
    payload = {
        "schema_version": 0,
        "provider": "fake-batch",
        "resource_name": "r",
        "state": "SUCCEEDED",
    }
    target = tmp_path / "v0.json"
    target.write_text(json.dumps(payload))
    loaded = load_managed_job_state(target)
    assert loaded.schema_version == CURRENT_MANAGED_JOB_SCHEMA
    assert loaded.task_count == 1
    assert loaded.succeeded_tasks == 0
    assert loaded.attempt == 0


def test_load_or_none_swallows_errors(tmp_path: Path) -> None:
    assert load_or_none(tmp_path / "absent.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("nope")
    assert load_or_none(bad) is None


def test_status_to_dict_round_trip() -> None:
    handle = ManagedJobHandle(provider="gcp-batch", resource_name="r", location="us-central1")
    status = ManagedJobStatus(
        handle=handle,
        state=ManagedJobTerminalState.SUCCEEDED,
        succeeded_tasks=8,
        failed_tasks=0,
        total_tasks=8,
        message="ok",
        raw_events=("submitted", "running", "succeeded"),
    )
    payload = status.to_dict()
    assert payload == {
        "provider": "gcp-batch",
        "resource_name": "r",
        "location": "us-central1",
        "state": "succeeded",
        "succeeded_tasks": 8,
        "failed_tasks": 0,
        "total_tasks": 8,
        "message": "ok",
        "raw_events": ("submitted", "running", "succeeded"),
    }


# ---------------------------------------------------------------------------
# Typed resource field tests
# ---------------------------------------------------------------------------


def test_spec_carries_typed_resource_fields() -> None:
    """MachineResource, ComputeResource, SA, network, allowed_locations, spot."""
    spec = ManagedJobSpec(
        name="job",
        machine_resource=MachineResource(
            machine_type="a3-highgpu-1g",
            boot_disk=BootDisk(
                image="projects/debian-cloud/global/images/family/debian-12",
                size_gb=50,
            ),
            accelerators=(GpuAccelerator(type_="nvidia-h100-80gb", count=1),),
            min_cpu_platform="intel-cascadelake",
        ),
        compute_resource=ComputeResource(cpu_milli=4000, memory_mib=8192, boot_disk_mib=20480),
        service_account=ServiceAccount(
            email="svc@proj.iam.gserviceaccount.com",
            scopes=("https://www.googleapis.com/auth/cloud-platform",),
        ),
        network=NetworkConfig(
            network="projects/proj/global/networks/default",
            subnetwork="projects/proj/regions/us-central1/subnetworks/default",
            no_external_ip_address=True,
        ),
        allowed_locations=("regions/us-central1",),
        spot=True,
    )
    assert spec.machine_resource is not None
    assert spec.machine_resource.machine_type == "a3-highgpu-1g"
    assert spec.machine_resource.boot_disk is not None
    assert spec.machine_resource.boot_disk.size_gb == 50
    assert spec.machine_resource.accelerators[0].count == 1
    assert spec.compute_resource is not None
    assert spec.compute_resource.cpu_milli == 4000
    assert spec.service_account is not None
    assert spec.service_account.email == "svc@proj.iam.gserviceaccount.com"
    assert spec.network is not None
    assert spec.network.no_external_ip_address is True
    assert spec.allowed_locations == ("regions/us-central1",)
    assert spec.spot is True


def test_spec_to_dict_serializes_typed_fields() -> None:
    """``to_dict`` produces a JSON-safe shape for the typed fields.

    No ``from_dict`` exists (and is intentionally not added — see
    ``ManagedJobSpec`` for the de-scope). The dict is a one-way
    serialization for persistence, not a transport contract.
    """
    spec = ManagedJobSpec(
        name="job",
        machine_resource=MachineResource(
            machine_type="n1-standard-4",
            boot_disk=BootDisk(size_gb=20, type_="pd-ssd"),
            accelerators=(GpuAccelerator(type_="nvidia-tesla-t4", count=2),),
        ),
        compute_resource=ComputeResource(cpu_milli=2000, memory_mib=4096),
        service_account=ServiceAccount(email="svc@x.iam.gserviceaccount.com"),
        network=NetworkConfig(subnetwork="projects/p/regions/r/subnetworks/s"),
        allowed_locations=("regions/us-central1", "regions/us-east1"),
        spot=True,
    )
    payload: dict[str, object] = spec.to_dict()
    assert payload["machine_resource"]["machine_type"] == "n1-standard-4"  # type: ignore[index]
    assert payload["machine_resource"]["boot_disk"]["size_gb"] == 20  # type: ignore[index]
    assert payload["machine_resource"]["boot_disk"]["type"] == "pd-ssd"  # type: ignore[index]
    assert payload["machine_resource"]["accelerators"][0]["count"] == 2  # type: ignore[index]
    assert payload["compute_resource"] == {  # type: ignore[comparison-overlap]
        "cpu_milli": 2000,
        "memory_mib": 4096,
        "boot_disk_mib": 0,
    }
    assert payload["service_account"]["email"] == "svc@x.iam.gserviceaccount.com"  # type: ignore[index]
    assert payload["network"]["subnetwork"] == "projects/p/regions/r/subnetworks/s"  # type: ignore[index]
    assert payload["allowed_locations"] == ["regions/us-central1", "regions/us-east1"]
    assert payload["spot"] is True


def test_spec_to_dict_omits_unset_typed_fields() -> None:
    """When typed fields are None, the dict payload also reports None."""
    spec = ManagedJobSpec(name="job")
    payload = spec.to_dict()
    assert payload["machine_resource"] is None
    assert payload["compute_resource"] is None
    assert payload["service_account"] is None
    assert payload["network"] is None
    assert payload["allowed_locations"] == []
    assert payload["spot"] is False


def test_spec_to_dict_for_task_status_includes_exit_code_and_message() -> None:
    """Per-task dict includes exit_code and message fields."""
    task = ManagedTaskStatus(task_index=2, state="FAILED", exit_code=2, message="OOM")
    payload = task.to_dict()
    assert payload == {
        "task_index": 2,
        "state": "FAILED",
        "exit_code": 2,
        "message": "OOM",
    }


def test_spec_to_dict_with_no_typed_fields_preserves_legacy_shape() -> None:
    """Existing fields (without typed resource fields) still serialize cleanly."""
    spec = ManagedJobSpec(
        name="hello",
        task_count=3,
        parallelism=2,
        image="gcr.io/x/y",
        command=("python", "-m", "hello"),
        environment={"FOO": "bar"},
        labels={"campaign": "demo"},
        gcs_mounts=("gs://example/x",),
        timeout_seconds=3600,
        retry_on_preempt=True,
        region="us-central1",
    )
    payload = spec.to_dict()
    assert payload["name"] == "hello"
    assert payload["command"] == ["python", "-m", "hello"]
    assert payload["environment"] == {"FOO": "bar"}
    assert payload["gcs_mounts"] == ["gs://example/x"]
    assert payload["machine_resource"] is None
    assert payload["compute_resource"] is None
    assert payload["service_account"] is None
    assert payload["network"] is None
    assert payload["allowed_locations"] == []
    assert payload["spot"] is False
