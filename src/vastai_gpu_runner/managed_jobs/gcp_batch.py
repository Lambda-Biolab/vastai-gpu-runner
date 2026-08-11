"""GCP Batch implementation of :class:`ManagedJobRunner`.

Maps the provider-neutral :class:`ManagedJobSpec` onto
:class:`google.cloud.batch_v1.Job` and translates job/task state back
onto :class:`ManagedJobStatus` / :class:`ManagedTaskStatus`. The
Google SDKs are imported lazily so installations without the optional
``gcp`` extra do not pay the dependency cost.

Every GCP API call goes through a single seam (``_client`` /
``_storage_client``) that tests override with a fake. The class never
makes a network call on import.

Public surface
--------------

* :class:`GcpBatchRunner` — production runner.
* :class:`FakeGcpBatchClient` / :class:`FakeGcsClient` — in-memory
  fakes used by tests.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from vastai_gpu_runner.managed_jobs.base import (
    ManagedJobHandle,
    ManagedJobRunner,
    ManagedJobSpec,
    ManagedJobStatus,
    ManagedJobTerminalState,
    ManagedTaskStatus,
)

if TYPE_CHECKING:
    from google.cloud import batch_v1, storage

logger = logging.getLogger(__name__)

# Exit code 50001 = Batch-documented Spot VM preemption.
SPOT_PREEMPT_EXIT_CODE = 50001

# Map GCP Batch State enum strings onto our normalized terminal states.
_STATE_MAP: dict[str, ManagedJobTerminalState] = {
    "STATE_UNSPECIFIED": ManagedJobTerminalState.UNKNOWN,
    "QUEUED": ManagedJobTerminalState.UNKNOWN,
    "SCHEDULED": ManagedJobTerminalState.UNKNOWN,
    "RUNNING": ManagedJobTerminalState.UNKNOWN,
    "SUCCEEDED": ManagedJobTerminalState.SUCCEEDED,
    "FAILED": ManagedJobTerminalState.FAILED,
    "CANCELLED": ManagedJobTerminalState.CANCELLED,
    "CANCELLATION_IN_PROGRESS": ManagedJobTerminalState.CANCELLED,
    "DELETION_IN_PROGRESS": ManagedJobTerminalState.CANCELLED,
}

# Map TaskGroup state strings.
_TASK_STATE_MAP: dict[str, str] = {
    "PENDING": "PENDING",
    "ASSIGNED": "ASSIGNED",
    "RUNNING": "RUNNING",
    "SUCCEEDED": "SUCCEEDED",
    "FAILED": "FAILED",
    "CANCELLLED": "CANCELLED",
}


class GcpBatchRunner(ManagedJobRunner):
    """Concrete :class:`ManagedJobRunner` backed by GCP Batch."""

    @property
    def provider_name(self) -> str:
        """Short provider identifier (e.g. ``"gcp-batch"``)."""
        return "gcp-batch"

    def __init__(
        self,
        project_id: str,
        region: str,
        client: batch_v1.BatchServiceClient | None = None,
        storage_client: storage.Client | None = None,
        bucket_name: str = "",
    ) -> None:
        """Construct a runner; clients are constructed lazily if omitted.

        Passing ``client`` and ``storage_client`` is the seam for fake
        implementations in tests and for stubbed CI runs.
        """
        self._project_id = project_id
        self._region = region
        self._bucket_name = bucket_name
        self._client = client
        self._storage_client = storage_client

    @property
    def project_id(self) -> str:
        """Return the configured GCP project ID."""
        return self._project_id

    @property
    def region(self) -> str:
        """Return the configured GCP region."""
        return self._region

    def submit(self, spec: ManagedJobSpec) -> ManagedJobHandle:
        """Submit ``spec`` to GCP Batch and return the resource handle.

        Translates :class:`ManagedJobSpec` onto a
        :class:`batch_v1.CreateJobRequest`. The runner attaches a
        Spot-preemption retry policy when ``spec.retry_on_preempt``
        is true and uses CLOUD_LOGGING as the log destination.
        """
        client = self._require_client()
        job = self._build_job(spec)
        request = self._create_request(spec)
        request.job = job
        try:
            created = client.create_job(request=request)
        except Exception:  # pragma: no cover - depends on real GCP response
            logger.exception("GCP Batch submit failed for %s", spec.name)
            raise
        resource_name = getattr(created, "name", "") or f"{request.parent}/jobs/{spec.name}"
        location = f"{self._project_id}/{self._region}"
        return ManagedJobHandle(
            provider=self.provider_name,
            resource_name=resource_name,
            location=location,
        )

    def get_status(self, handle: ManagedJobHandle) -> ManagedJobStatus:
        """Return the latest status snapshot for ``handle``.

        Counts are aggregated from per-task-group state maps; events
        are flattened from ``JobStatus.status_events``.
        """
        client = self._require_client()
        view = self._build_view_request(handle)
        try:
            job = client.get_job(view=view, name=handle.resource_name)
        except Exception:  # pragma: no cover - depends on real GCP response
            logger.exception("GCP Batch get_job failed for %s", handle.resource_name)
            raise
        return self._translate_status(job)

    def list_tasks(self, handle: ManagedJobHandle) -> Iterable[ManagedTaskStatus]:
        """Yield per-task status, when the provider supports it."""
        client = self._require_client()
        try:
            tasks = client.list_tasks(parent=self._task_parent(handle))
        except Exception:  # pragma: no cover - depends on real GCP response
            logger.exception("GCP Batch list_tasks failed for %s", handle.resource_name)
            raise
        for task in tasks:
            raw_state = getattr(getattr(task, "status", None), "state", None)
            if raw_state is None:
                key = "UNKNOWN"
            elif hasattr(raw_state, "name"):
                key = raw_state.name
            else:
                key = str(raw_state)
            yield ManagedTaskStatus(
                task_index=self._extract_task_index(task),
                state=_TASK_STATE_MAP.get(key, "UNKNOWN"),
            )

    def cancel(self, handle: ManagedJobHandle) -> None:
        """Request cancellation of the job. Idempotent."""
        client = self._require_client()
        try:
            client.cancel_job(name=handle.resource_name)
        except Exception:  # pragma: no cover - depends on real GCP response
            logger.exception("GCP Batch cancel_job failed for %s", handle.resource_name)
            raise

    def delete(self, handle: ManagedJobHandle) -> None:
        """Permanently remove the job from GCP Batch. Idempotent."""
        client = self._require_client()
        try:
            client.delete_job(name=handle.resource_name)
        except Exception:  # pragma: no cover - depends on real GCP response
            logger.exception("GCP Batch delete_job failed for %s", handle.resource_name)
            raise

    def _require_client(self) -> batch_v1.BatchServiceClient:
        if self._client is not None:
            return self._client
        from google.cloud import batch_v1

        self._client = batch_v1.BatchServiceClient()
        return self._client

    def _build_job(self, spec: ManagedJobSpec) -> batch_v1.Job:
        from google.cloud import batch_v1

        runnable = batch_v1.Runnable()
        runnable.container = batch_v1.Runnable.Container()
        runnable.container.image_uri = spec.image
        if spec.command:
            runnable.container.entrypoint = spec.command[0]
            runnable.container.commands = list(spec.command)

        task = batch_v1.TaskSpec()
        task.runnables = [runnable]
        task.max_retry_count = 3 if spec.retry_on_preempt else 0

        if spec.retry_on_preempt:
            policy = batch_v1.LifecyclePolicy()
            policy.action = batch_v1.LifecyclePolicy.Action.RETRY_TASK
            policy.action_condition = batch_v1.LifecyclePolicy.ActionCondition()
            policy.action_condition.exit_codes = [SPOT_PREEMPT_EXIT_CODE]
            task.lifecycle_policies = [policy]

        if spec.timeout_seconds is not None:
            task.max_run_duration = f"{spec.timeout_seconds}s"

        group = batch_v1.TaskGroup()
        group.task_count = spec.task_count
        group.parallelism = spec.parallelism
        group.task_spec = task

        instance_policy = batch_v1.AllocationPolicy.InstancePolicy()
        instance_policy.machine_type = spec.environment.get("machine_type", "e2-standard-4")
        instance_policy.provisioning_model = self._resolve_provisioning_model(
            spec.environment.get("provisioning_model", "STANDARD")
        )

        instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
        instances.policy = instance_policy

        allocation_policy = batch_v1.AllocationPolicy()
        allocation_policy.instances = [instances]

        if self._bucket_name and spec.gcs_mounts:
            volume = batch_v1.Volume()
            volume.gcs = batch_v1.GCS()
            volume.gcs.remote_path = self._bucket_name
            # Mount is a nested class on Volume in this client version;
            # reference via the parent to keep pyright happy.
            mount_cls = type(volume).Mount
            mount = mount_cls()
            mount.mount_path = spec.gcs_mounts[0]
            volume.mount = mount
            task.volumes = [volume]

        job = batch_v1.Job()
        job.task_groups = [group]
        job.allocation_policy = allocation_policy
        job.labels = dict(spec.labels)
        job.logs_policy = batch_v1.LogsPolicy()
        job.logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING

        return job

    def _resolve_provisioning_model(
        self, name: str
    ) -> batch_v1.AllocationPolicy.InstancePolicy.ProvisioningModel:
        from google.cloud import batch_v1

        enum_cls = type(batch_v1.AllocationPolicy.InstancePolicy().provisioning_model)
        mapping = {
            "STANDARD": enum_cls.STANDARD,
            "SPOT": enum_cls.SPOT,
            "PREEMPTIBLE": enum_cls.PREEMPTIBLE,
            "FLEX_START": enum_cls.FLEX_START,
            "RESERVATION_BOUND": enum_cls.RESERVATION_BOUND,
        }
        return mapping[name]

    def _create_request(self, spec: ManagedJobSpec) -> batch_v1.CreateJobRequest:
        from google.cloud import batch_v1

        request = batch_v1.CreateJobRequest()
        request.job_id = spec.name
        request.parent = f"projects/{self._project_id}/locations/{self._region}"
        return request

    def _build_view_request(self, handle: ManagedJobHandle) -> batch_v1.GetJobRequest:
        from google.cloud import batch_v1

        view = batch_v1.GetJobRequest()
        view.name = handle.resource_name
        return view

    def _task_parent(self, handle: ManagedJobHandle) -> str:
        return f"{handle.resource_name}/taskGroups/group0"

    def _extract_task_index(self, task: Any) -> int:
        """Extract the task index from a GCP Task.

        GCP stores the task index in the trailing segment of the task
        resource name (``.../tasks/TASK_INDEX``). Falls back to 0
        when the name is missing or malformed so callers always see a
        usable index without raising.
        """
        name = getattr(task, "name", "") or ""
        if not name:
            return 0
        try:
            return int(name.rsplit("/", 1)[-1])
        except (TypeError, ValueError):
            return 0

    def _translate_status(self, job: Any) -> ManagedJobStatus:
        status = getattr(job, "status", None)
        state = self._translate_state(status)
        succeeded, failed = self._aggregate_task_counts(job, status)
        total = sum(
            int(getattr(group, "task_count", 0) or 0)
            for group in getattr(job, "task_groups", []) or []
        )
        return ManagedJobStatus(
            handle=ManagedJobHandle(
                provider=self.provider_name,
                resource_name=getattr(job, "name", "") or "",
                location=self._region,
            ),
            state=state,
            succeeded_tasks=succeeded,
            failed_tasks=failed,
            total_tasks=total,
            message="",
            raw_events=tuple(self._collect_events(status)),
        )

    def _translate_state(self, status: Any) -> ManagedJobTerminalState:
        if status is None:
            return ManagedJobTerminalState.UNKNOWN
        raw_state = getattr(status, "state", None)
        if raw_state is None:
            return ManagedJobTerminalState.UNKNOWN
        if hasattr(raw_state, "name"):
            key = raw_state.name
        else:
            key = str(raw_state)
        return _STATE_MAP.get(key, ManagedJobTerminalState.UNKNOWN)

    def _aggregate_task_counts(self, job: Any, status: Any) -> tuple[int, int]:
        _ = job
        succeeded = 0
        failed = 0
        status_task_groups = getattr(status, "task_groups", None) or {}
        for _name, group_status in self._iter_group_statuses(status_task_groups):
            for raw_state, count in self._iter_counts(group_status):
                state_name = self._state_name(raw_state)
                if state_name == "SUCCEEDED":
                    succeeded += int(count or 0)
                elif state_name in {"FAILED", "CANCELLED"}:
                    failed += int(count or 0)
        return succeeded, failed

    @staticmethod
    def _iter_group_statuses(mapping: Any) -> Iterable[Any]:
        if hasattr(mapping, "items"):
            yield from mapping.items() or []
        else:
            return

    @staticmethod
    def _iter_counts(group_status: Any) -> Iterable[tuple[Any, int]]:
        counts = getattr(group_status, "counts", None) or {}
        if hasattr(counts, "items"):
            yield from counts.items() or []

    @staticmethod
    def _state_name(raw_state: Any) -> str:
        if hasattr(raw_state, "name"):
            return raw_state.name
        if hasattr(raw_state, "value"):
            return str(raw_state.value)
        return str(raw_state)

    @staticmethod
    def _collect_events(status: Any) -> list[str]:
        events: list[str] = []
        for event in getattr(status, "status_events", []) or []:
            description = getattr(event, "description", "")
            if description:
                events.append(description)
        return events


# ---------------------------------------------------------------------------
# In-memory fakes used by the test suite.
# ---------------------------------------------------------------------------


class FakeGcpBatchClient:
    """In-memory stand-in for :class:`batch_v1.BatchServiceClient`.

    Records submitted jobs by ``resource_name`` and serves them to
    :meth:`get_job` / :meth:`list_tasks`. Tests can preset
    :attr:`statuses` to drive status transitions.
    """

    def __init__(self) -> None:
        """Initialise an empty fake: no jobs, all state maps blank."""
        self.jobs: dict[str, Any] = {}
        self.statuses: dict[str, Any] = {}
        self.events: dict[str, list[Any]] = {}
        self.tasks: dict[str, list[Any]] = {}
        self.submit_calls: list[Any] = []
        self.cancel_calls: list[str] = []
        self.delete_calls: list[str] = []

    def create_job(self, request: Any) -> Any:
        """Record the request and return a synthetic :class:`batch_v1.Job`."""
        self.submit_calls.append(request)
        resource_name = f"{request.parent}/jobs/{request.job_id}"
        self.jobs[resource_name] = request.job
        self.statuses.setdefault(resource_name, _make_job_status("RUNNING"))
        self.events.setdefault(resource_name, [])
        self.tasks.setdefault(resource_name, [])
        return _make_job(
            name=resource_name,
            task_groups=request.job.task_groups,
            status=self.statuses[resource_name],
            status_events=self.events[resource_name],
        )

    def get_job(self, view: Any, name: str) -> Any:
        """Return the most recent job record, augmented with current status."""
        job = self.jobs[name]
        return _make_job(
            name=name,
            task_groups=job.task_groups,
            status=self.statuses[name],
            status_events=self.events[name],
        )

    def list_tasks(self, parent: str) -> Iterable[Any]:
        """Yield preset tasks for the given job parent path."""
        return iter(self.tasks.get(parent, []))

    def cancel_job(self, name: str) -> None:
        """Record the cancellation call and mark the job as cancelling."""
        self.cancel_calls.append(name)
        self.statuses[name] = _make_job_status("CANCELLATION_IN_PROGRESS")

    def delete_job(self, name: str) -> None:
        """Record the deletion call and drop the cached job state."""
        self.delete_calls.append(name)
        self.jobs.pop(name, None)
        self.statuses.pop(name, None)


def _make_job_status(state: str) -> Any:
    from google.cloud import batch_v1

    status = batch_v1.JobStatus()
    status.state = batch_v1.JobStatus.State[state]
    return status


def _make_job(*, name: str, task_groups: list[Any], status: Any, status_events: list[Any]) -> Any:
    from google.cloud import batch_v1

    job = batch_v1.Job()
    job.name = name
    job.task_groups = task_groups
    job.status = status
    return job


class FakeGcsClient:
    """In-memory stand-in for :class:`storage.Client`.

    Implements the surface used by :class:`GcsSink`: ``bucket``,
    ``get_bucket``, and per-bucket ``blob`` round-tripping. Tests can
    inspect :attr:`uploads` / :attr:`downloads` to assert on call
    sequences.
    """

    def __init__(self) -> None:
        """Initialise an empty in-memory GCS namespace."""
        self.buckets: dict[str, dict[str, bytes]] = {}
        self.uploads: list[tuple[str, str, bytes, str | None]] = []
        self.downloads: list[tuple[str, str]] = []

    def bucket(self, name: str) -> FakeGcsBucket:
        """Return a fake bucket handle; raise if the bucket is unknown."""
        if name not in self.buckets:
            raise LookupError(f"bucket not found: {name}")
        return FakeGcsBucket(self, name)

    def get_bucket(self, name: str) -> FakeGcsBucket:
        """Alias for :meth:`bucket`."""
        return self.bucket(name)

    def create_bucket(self, name: str) -> None:
        """Pre-seed a bucket so ``bucket()`` resolves."""
        self.buckets.setdefault(name, {})

    def exists(self, name: str) -> bool:
        """Return True if a bucket of this name is pre-seeded."""
        return name in self.buckets


class FakeGcsBucket:
    """Stand-in for :class:`google.cloud.storage.bucket.Bucket`."""

    def __init__(self, client: FakeGcsClient, name: str) -> None:
        """Bind this bucket to its fake client and name."""
        self._client = client
        self._name = name

    def blob(self, key: str) -> FakeGcsBlob:
        """Return a fake blob handle for ``key``."""
        return FakeGcsBlob(self._client, self._name, key)

    def exists(self, key: str = "") -> bool:
        """Return True if ``key`` exists in this bucket.

        With no ``key`` argument, the real :class:`google.cloud.storage.bucket.Bucket`
        reports whether the bucket itself exists. The GcsSink only ever
        asks about keys, so the default-argument form keeps the fake
        compatible with both the sink and any future bucket-level
        existence checks.
        """
        if not key:
            return self._name in self._client.buckets
        return key in self._client.buckets[self._name]


class FakeGcsBlob:
    """Stand-in for :class:`google.cloud.storage.blob.Blob`."""

    def __init__(self, client: FakeGcsClient, bucket: str, key: str) -> None:
        """Bind this blob to its fake client, bucket, and key."""
        self._client = client
        self._bucket = bucket
        self._key = key

    def upload_from_string(self, data: bytes, content_type: str | None = None) -> None:
        """Store ``data`` in the in-memory bucket."""
        self._client.uploads.append((self._bucket, self._key, data, content_type))
        self._client.buckets.setdefault(self._bucket, {})[self._key] = data

    def download_as_bytes(self) -> bytes:
        """Return the previously stored bytes, or empty bytes."""
        self._client.downloads.append((self._bucket, self._key))
        return self._client.buckets.get(self._bucket, {}).get(self._key, b"")

    def exists(self) -> bool:
        """Return True if this blob is in the in-memory bucket."""
        return self._key in self._client.buckets.get(self._bucket, {})

    def delete(self) -> None:
        """Remove the blob from the in-memory bucket."""
        self._client.buckets.setdefault(self._bucket, {}).pop(self._key, None)
