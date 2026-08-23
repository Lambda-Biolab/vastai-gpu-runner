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
    BootDisk,
    ComputeResource,
    GpuAccelerator,
    ManagedJobHandle,
    ManagedJobLifecycleState,
    ManagedJobRunner,
    ManagedJobSpec,
    ManagedJobStatus,
    ManagedTaskStatus,
    NetworkConfig,
    ServiceAccount,
    StorageMount,
)
from vastai_gpu_runner.managed_jobs.errors import (
    ManagedJobNotFoundError,
    map_gcp_exception,
)

if TYPE_CHECKING:
    from google.cloud import batch_v1, storage

logger = logging.getLogger(__name__)

# Exit code 50001 = Batch-documented Spot VM preemption.
SPOT_PREEMPT_EXIT_CODE = 50001

# Environment keys that the runner interprets as job-shape overrides
# (machine_type, provisioning_model, gpu_type, gpu_count). These are
# stripped from the container env that the SDK sends to the worker
# so they don't pollute the actual runtime environment. New code
# should use the typed ``MachineResource`` / ``spot`` fields instead;
# this constant exists to keep the legacy env shim working without
# polluting the container env.
_LEGACY_RESOURCE_KEYS = frozenset({"machine_type", "provisioning_model", "gpu_type", "gpu_count"})

# Map GCP Batch State enum strings onto the normalized lifecycle states.
_STATE_MAP: dict[str, ManagedJobLifecycleState] = {
    "STATE_UNSPECIFIED": ManagedJobLifecycleState.UNKNOWN,
    "QUEUED": ManagedJobLifecycleState.QUEUED,
    "SCHEDULED": ManagedJobLifecycleState.PENDING,
    "RUNNING": ManagedJobLifecycleState.RUNNING,
    "SUCCEEDED": ManagedJobLifecycleState.SUCCEEDED,
    "FAILED": ManagedJobLifecycleState.FAILED,
    "CANCELLED": ManagedJobLifecycleState.CANCELLED,
    "CANCELLATION_IN_PROGRESS": ManagedJobLifecycleState.CANCELLING,
    "DELETION_IN_PROGRESS": ManagedJobLifecycleState.CANCELLING,
}

# Map TaskGroup state strings. UNEXECUTED (a task that never started
# because the job was cancelled before it was assigned) folds into
# CANCELLED so the consumer sees a consistent "cancelled" signal.
_TASK_STATE_MAP: dict[str, str] = {
    "PENDING": "PENDING",
    "ASSIGNED": "ASSIGNED",
    "RUNNING": "RUNNING",
    "SUCCEEDED": "SUCCEEDED",
    "FAILED": "FAILED",
    "CANCELLED": "CANCELLED",
    "UNEXECUTED": "CANCELLED",
}

# Names of the GCP Batch provisioning model enum members that this
# runner knows how to map. Anything outside this set raises
# :class:`ValueError` from :meth:`_provisioning_model_from_name`.
_KNOWN_PROVISIONING_MODELS = frozenset(
    {"STANDARD", "SPOT", "PREEMPTIBLE", "FLEX_START", "RESERVATION_BOUND"}
)

# Location-prefix tokens that GCP Batch's ``LocationPolicy.allowed_locations``
# accepts. Bare names like ``us-central1`` are normalised to the
# ``regions/...`` form before being sent.
_LOCATION_PREFIXES = ("regions/", "zones/")


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
        """Return the configured GCP region (the runner's default)."""
        return self._region

    def effective_region(self, spec: ManagedJobSpec) -> str:
        """Return the region actually used for ``spec`` in bare form.

        ``spec.region`` overrides the runner's default when set. The
        returned value is the bare region name (no ``regions/`` /
        ``zones/`` prefix) — prefixed inputs are stripped so callers
        always see a clean bare name regardless of how the spec
        spelled it.
        """
        raw = spec.region.strip() if spec.region else self._region
        return self._strip_location_prefix(raw)

    def submit(self, spec: ManagedJobSpec) -> ManagedJobHandle:
        """Submit ``spec`` to GCP Batch and return the resource handle.

        Translates :class:`ManagedJobSpec` onto a
        :class:`batch_v1.CreateJobRequest`. The runner attaches a
        Spot-preemption retry policy when ``spec.retry_on_preempt``
        is true and uses CLOUD_LOGGING as the log destination.

        ``spec.region`` overrides the runner's default region; the
        override is honoured consistently in the request parent and
        the returned handle. When ``spec.allowed_locations`` is set,
        the *effective* region (the one used in the request parent)
        must appear in the allow-list.
        """
        self._validate_region(spec)
        client = self._require_client()
        job = self._build_job(spec)
        request = self._create_request(spec)
        request.job = job
        try:
            created = client.create_job(request=request)
        except Exception as exc:  # pragma: no cover - depends on real GCP response
            logger.exception("GCP Batch submit failed for %s", spec.name)
            raise map_gcp_exception(exc, operation="GCP Batch submit") from exc
        resource_name = getattr(created, "name", "") or f"{request.parent}/jobs/{spec.name}"
        location = self._format_handle_location(spec)
        return ManagedJobHandle(
            provider=self.provider_name,
            resource_name=resource_name,
            location=location,
        )

    def get_status(self, handle: ManagedJobHandle) -> ManagedJobStatus:
        """Return the latest status snapshot for ``handle``.

        Counts are aggregated from per-task-group state maps; the
        human-readable ``message`` is the most recent status-event
        description, useful for surfacing "why did this fail" without
        a separate log query.
        """
        client = self._require_client()
        view = self._build_view_request(handle)
        try:
            job = client.get_job(request=view)
        except Exception as exc:  # pragma: no cover - depends on real GCP response
            logger.exception("GCP Batch get_job failed for %s", handle.resource_name)
            raise map_gcp_exception(exc, operation="GCP Batch get_status") from exc
        return self._translate_status(job)

    def list_tasks(self, handle: ManagedJobHandle) -> Iterable[ManagedTaskStatus]:
        """Yield per-task status, when the provider supports it.

        Each entry carries the task index (parsed from the resource
        name), the normalised state, and — when the underlying task
        has run — the exit code and the most recent status-event
        description. The exit code is reported only when the SDK
        actually provided one; downstream code distinguishes
        "exited 0" from "no exit code recorded" via the ``None``
        sentinel.
        """
        client = self._require_client()
        try:
            tasks = client.list_tasks(parent=self._task_parent(handle))
            for task in tasks:
                yield self._translate_task(task)
        except Exception as exc:  # pragma: no cover - depends on real GCP response
            logger.exception("GCP Batch list_tasks failed for %s", handle.resource_name)
            raise map_gcp_exception(exc, operation="GCP Batch list_tasks") from exc

    def cancel(self, handle: ManagedJobHandle) -> None:
        """Request cancellation of the job. Idempotent."""
        client = self._require_client()
        try:
            client.cancel_job(name=handle.resource_name)
        except Exception as exc:  # pragma: no cover - depends on real GCP response
            logger.exception("GCP Batch cancel_job failed for %s", handle.resource_name)
            error = map_gcp_exception(exc, operation="GCP Batch cancel")
            if isinstance(error, ManagedJobNotFoundError):
                return
            raise error from exc

    def delete(self, handle: ManagedJobHandle) -> None:
        """Permanently remove the job from GCP Batch. Idempotent."""
        client = self._require_client()
        try:
            client.delete_job(name=handle.resource_name)
        except Exception as exc:  # pragma: no cover - depends on real GCP response
            logger.exception("GCP Batch delete_job failed for %s", handle.resource_name)
            error = map_gcp_exception(exc, operation="GCP Batch delete")
            if isinstance(error, ManagedJobNotFoundError):
                return
            raise error from exc

    def _require_client(self) -> batch_v1.BatchServiceClient:
        if self._client is not None:
            return self._client
        from google.cloud import batch_v1

        self._client = batch_v1.BatchServiceClient()
        return self._client

    def _validate_region(self, spec: ManagedJobSpec) -> None:
        """Reject ``effective_region(spec)`` outside the allowed locations.

        Only the effective region (the one that ends up in the
        request parent) is validated. The runner region alone is
        not checked because the runner is just a default — when
        ``spec.region`` overrides it, the runner region has no
        influence on the request.

        Bare names (``"us-central1"``) and the prefixed form
        (``"regions/us-central1"``) are normalised so they compare
        equal to the canonical allow-list entries.
        """
        allowed = tuple(spec.allowed_locations)
        if not allowed:
            return
        normalised_allowed = {self._canonicalise_location(a) for a in allowed}
        effective = self._canonicalise_location(self.effective_region(spec))
        if effective not in normalised_allowed:
            raise ValueError(
                f"effective region {self.effective_region(spec)!r} "
                f"is not in allowed_locations {list(allowed)}"
            )

    @staticmethod
    def _canonicalise_location(value: str) -> str:
        """Normalise "us-central1" → "regions/us-central1" for comparison.

        Passes through any value that already carries a ``regions/``
        or ``zones/`` prefix.
        """
        if not value:
            return value
        if any(value.startswith(prefix) for prefix in _LOCATION_PREFIXES):
            return value
        return f"regions/{value}"

    @staticmethod
    def _strip_location_prefix(value: str) -> str:
        """Strip the ``regions/`` or ``zones/`` prefix from ``value``.

        Used to expose the effective region in bare form regardless
        of how the spec spelled it.
        """
        for prefix in _LOCATION_PREFIXES:
            if value.startswith(prefix):
                return value[len(prefix) :]
        return value

    @staticmethod
    def _canonicalise_allowed_locations(
        locations: Iterable[str],
    ) -> list[str]:
        """Canonicalise bare names to ``regions/...`` before sending to the SDK.

        The SDK accepts ``regions/X`` and ``zones/Y`` in
        ``LocationPolicy.allowed_locations``; bare ``X`` is rejected.
        We canonicalise so the runner's validation (which uses the
        same comparison) and the SDK request agree on the form.
        """
        return [GcpBatchRunner._canonicalise_location(loc) for loc in locations if loc]

    def _format_handle_location(self, spec: ManagedJobSpec) -> str:
        """Return the canonical ``project/region`` form for the handle.

        ``handle.location`` mirrors the request parent so consumers
        can correlate the handle with the request that produced it.
        The region is always the bare form.
        """
        return f"{self._project_id}/{self.effective_region(spec)}"

    def _build_job(self, spec: ManagedJobSpec) -> batch_v1.Job:
        task = self._build_task_spec(spec)
        group = self._build_task_group(spec, task)
        allocation_policy = self._build_allocation_policy(spec)
        return self._build_job_envelope(spec, group, allocation_policy)

    def _build_task_spec(self, spec: ManagedJobSpec) -> Any:
        """Populate the ``TaskSpec`` (runnable, env, lifecycle, compute, volumes)."""
        from google.cloud import batch_v1

        runnable = batch_v1.Runnable()
        runnable.container = batch_v1.Runnable.Container()
        runnable.container.image_uri = spec.image
        if spec.command:
            runnable.container.entrypoint = spec.command[0]
            runnable.container.commands = list(spec.command[1:])

        volumes, container_volumes = self._build_volumes(spec)
        if container_volumes:
            runnable.container.volumes = container_volumes

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

        container_env = self._filter_container_env(spec.environment)
        if container_env:
            task.environment = batch_v1.Environment()
            task.environment.variables = dict(container_env)

        if spec.compute_resource is not None:
            task.compute_resource = self._build_compute_resource(spec.compute_resource)

        # Set the volumes BEFORE handing the task to the TaskGroup:
        # proto-plus copies the underlying protobuf on assignment, so
        # post-group mutations on the local task are dropped.
        if volumes:
            task.volumes = volumes
        return task

    def _build_task_group(self, spec: ManagedJobSpec, task: Any) -> Any:
        from google.cloud import batch_v1

        group = batch_v1.TaskGroup()
        group.task_count = spec.task_count
        group.parallelism = spec.parallelism
        group.task_spec = task
        return group

    def _build_allocation_policy(self, spec: ManagedJobSpec) -> Any:
        from google.cloud import batch_v1

        instance_policy, install_gpu_drivers = self._build_instance_policy(spec)
        instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
        instances.policy = instance_policy
        # GPU driver installation lives on the InstancePolicyOrTemplate,
        # not on the deprecated ``Accelerator.install_gpu_drivers``
        # field. Setting it here covers the whole instance template.
        if install_gpu_drivers:
            instances.install_gpu_drivers = True

        allocation_policy = batch_v1.AllocationPolicy()
        allocation_policy.instances = [instances]

        if spec.service_account is not None:
            allocation_policy.service_account = self._build_service_account(spec.service_account)
        if spec.network is not None:
            allocation_policy.network = self._build_network_policy(spec.network)
        if spec.allowed_locations:
            allocation_policy.location = batch_v1.AllocationPolicy.LocationPolicy()
            allocation_policy.location.allowed_locations = self._canonicalise_allowed_locations(
                spec.allowed_locations
            )
        return allocation_policy

    def _build_job_envelope(self, spec: ManagedJobSpec, group: Any, allocation_policy: Any) -> Any:
        from google.cloud import batch_v1

        job = batch_v1.Job()
        job.task_groups = [group]
        job.allocation_policy = allocation_policy
        job.labels = dict(spec.labels)
        job.logs_policy = batch_v1.LogsPolicy()
        job.logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING
        return job

    @staticmethod
    def _filter_container_env(env: Any) -> dict[str, str]:
        """Strip legacy resource/provisioning keys from the container env.

        The runner reads ``machine_type`` / ``provisioning_model`` /
        ``gpu_type`` / ``gpu_count`` from ``spec.environment`` as a
        backwards-compat shim. Those keys are control-plane inputs,
        not runtime env vars, so they must not be exposed to the
        worker container. Everything else is forwarded verbatim.
        """
        if not env:
            return {}
        return {k: v for k, v in env.items() if k not in _LEGACY_RESOURCE_KEYS}

    def _build_instance_policy(self, spec: ManagedJobSpec) -> tuple[Any, bool]:
        """Populate the GCP ``InstancePolicy`` from the typed spec fields.

        Returns ``(policy, install_gpu_drivers)``: the second element
        is the flag that should be hoisted onto the
        ``InstancePolicyOrTemplate`` (which is the *current* GCP
        location for the install flag — the legacy
        ``Accelerator.install_gpu_drivers`` field is deprecated).

        Legacy fallback: ``spec.environment["machine_type"]`` and
        ``spec.environment["provisioning_model"]`` still feed
        through, but the typed ``machine_resource`` / ``spot``
        fields are preferred.
        """
        from google.cloud import batch_v1

        policy = batch_v1.AllocationPolicy.InstancePolicy()
        machine_type = self._resolve_machine_type(spec) or "e2-standard-4"
        policy.machine_type = machine_type
        policy.provisioning_model = self._resolve_provisioning_model(spec)
        install_gpu_drivers = self._apply_typed_machine_extras(policy, spec)
        install_gpu_drivers |= self._apply_legacy_accelerator_fallback(policy, spec)
        return policy, install_gpu_drivers

    def _apply_typed_machine_extras(self, policy: Any, spec: ManagedJobSpec) -> bool:
        """Copy the typed ``MachineResource`` extras onto the policy.

        Returns ``True`` if any accelerator asked for GPU driver
        installation — the caller will hoist that flag onto the
        ``InstancePolicyOrTemplate``.
        """
        resource = spec.machine_resource
        if resource is None:
            return False
        if resource.min_cpu_platform:
            policy.min_cpu_platform = resource.min_cpu_platform
        if resource.boot_disk is not None:
            policy.boot_disk = self._build_boot_disk(resource.boot_disk)
        install_gpu_drivers = False
        if resource.accelerators:
            policy.accelerators = [self._build_accelerator(a) for a in resource.accelerators]
            install_gpu_drivers = any(a.install_gpu_drivers for a in resource.accelerators)
        return install_gpu_drivers

    def _apply_legacy_accelerator_fallback(self, policy: Any, spec: ManagedJobSpec) -> bool:
        """Honour legacy ``environment["gpu_type"]`` / ``gpu_count`` keys.

        Only applies when the typed ``MachineResource`` is absent.
        Returns ``True`` if any accelerator asked for GPU driver
        installation.
        """
        if spec.machine_resource is not None:
            return False
        legacy_type = spec.environment.get("gpu_type")
        legacy_count = spec.environment.get("gpu_count")
        if not (legacy_type and legacy_count and legacy_count.isdigit()):
            return False
        policy.accelerators = [
            self._build_accelerator(GpuAccelerator(type_=legacy_type, count=int(legacy_count)))
        ]
        # No legacy flag for driver install — the typed API is the
        # only way to request it.
        return False

    def _resolve_machine_type(self, spec: ManagedJobSpec) -> str:
        """Pick the machine type from the typed or legacy env field."""
        if spec.machine_resource is not None and spec.machine_resource.machine_type:
            return spec.machine_resource.machine_type
        return spec.environment.get("machine_type", "")

    def _resolve_provisioning_model(self, spec: ManagedJobSpec) -> Any:
        """Translate ``spot`` (or legacy env) onto the Batch provisioning enum.

        Priority order: explicit ``environment["provisioning_model"]``
        (legacy override) wins, then ``spec.spot`` (typed), then
        ``STANDARD``. Unknown values raise :class:`ValueError`.
        """
        from google.cloud import batch_v1

        enum_cls = type(batch_v1.AllocationPolicy.InstancePolicy().provisioning_model)
        legacy = spec.environment.get("provisioning_model")
        if legacy:
            return self._provisioning_model_from_name(legacy, enum_cls)
        if spec.spot:
            return enum_cls.SPOT
        return enum_cls.STANDARD

    @staticmethod
    def _provisioning_model_from_name(name: str, enum_cls: Any) -> Any:
        """Map a provisioning-model name to the GCP Batch enum.

        Unknown names raise :class:`ValueError` so a typo in the
        spec is surfaced immediately rather than silently coerced.
        """
        if name not in _KNOWN_PROVISIONING_MODELS:
            raise ValueError(
                f"unknown provisioning model {name!r}; "
                f"expected one of {sorted(_KNOWN_PROVISIONING_MODELS)}"
            )
        mapping = {
            "STANDARD": enum_cls.STANDARD,
            "SPOT": enum_cls.SPOT,
            "PREEMPTIBLE": enum_cls.PREEMPTIBLE,
            "FLEX_START": enum_cls.FLEX_START,
            "RESERVATION_BOUND": enum_cls.RESERVATION_BOUND,
        }
        return mapping[name]

    def _build_compute_resource(self, resource: ComputeResource) -> Any:
        from google.cloud import batch_v1

        cr = batch_v1.ComputeResource()
        if resource.cpu_milli:
            cr.cpu_milli = resource.cpu_milli
        if resource.memory_mib:
            cr.memory_mib = resource.memory_mib
        if resource.boot_disk_mib:
            cr.boot_disk_mib = resource.boot_disk_mib
        return cr

    def _build_boot_disk(self, disk: BootDisk) -> Any:
        from google.cloud import batch_v1

        out = batch_v1.AllocationPolicy.Disk()
        if disk.image:
            out.image = disk.image
        if disk.size_gb:
            out.size_gb = disk.size_gb
        if disk.type_:
            out.type_ = disk.type_
        return out

    def _build_accelerator(self, accelerator: GpuAccelerator) -> Any:
        """Build a ``batch_v1.AllocationPolicy.Accelerator``.

        The deprecated ``install_gpu_drivers`` field is intentionally
        not set here; the runner hoists that flag onto the
        ``InstancePolicyOrTemplate`` instead (see
        :meth:`_build_instance_policy`).
        """
        from google.cloud import batch_v1

        out = batch_v1.AllocationPolicy.Accelerator()
        if accelerator.type_:
            out.type_ = accelerator.type_
        if accelerator.count:
            out.count = accelerator.count
        if accelerator.driver_version:
            out.driver_version = accelerator.driver_version
        return out

    def _build_service_account(self, sa: ServiceAccount) -> Any:
        from google.cloud import batch_v1

        out = batch_v1.ServiceAccount()
        if sa.email:
            out.email = sa.email
        if sa.scopes:
            out.scopes = list(sa.scopes)
        return out

    def _build_network_policy(self, network: NetworkConfig) -> Any:
        from google.cloud import batch_v1

        policy = batch_v1.AllocationPolicy.NetworkPolicy()
        interface = batch_v1.AllocationPolicy.NetworkInterface()
        if network.network:
            interface.network = network.network
        if network.subnetwork:
            interface.subnetwork = network.subnetwork
        if network.no_external_ip_address:
            interface.no_external_ip_address = network.no_external_ip_address
        policy.network_interfaces = [interface]
        return policy

    def _build_volumes(self, spec: ManagedJobSpec) -> tuple[list[Any], list[str]]:
        """Translate storage mounts into GCP volumes and container bindings."""
        from google.cloud import batch_v1

        mounts_to_build = self._storage_mounts(spec)
        if not mounts_to_build:
            return [], []
        mounts: list[Any] = []
        container_volumes: list[str] = []
        for index, mount in enumerate(mounts_to_build):
            remote_path = self._gcs_remote_path(mount.uri)
            host_path = f"/mnt/disks/vastai-gpu-runner-gcs-{index}"
            volume = batch_v1.Volume()
            volume.gcs = batch_v1.GCS()
            volume.gcs.remote_path = remote_path
            volume.mount_path = host_path
            if mount.read_only:
                volume.mount_options = ["-o", "ro"]
            mounts.append(volume)
            mode = "ro" if mount.read_only else "rw"
            container_volumes.append(f"{host_path}:{mount.mount_path}:{mode}")
        return mounts, container_volumes

    def _storage_mounts(self, spec: ManagedJobSpec) -> list[StorageMount]:
        """Combine typed mounts with the deprecated GCS compatibility field."""
        mounts = list(spec.storage_mounts)
        for entry in spec.gcs_mounts:
            stripped = entry.strip()
            if not stripped:
                continue
            parsed = parse_gcs_mount(entry, default_bucket=self._bucket_name)
            if parsed is not None:
                bucket, mount_path = parsed
                mounts.append(StorageMount(uri=f"gs://{bucket}", mount_path=mount_path))
            elif not stripped.startswith("/") or self._bucket_name:
                raise ValueError(f"invalid or unsupported legacy GCS mount: {entry!r}")
        return mounts

    @staticmethod
    def _gcs_remote_path(uri: str) -> str:
        """Return the GCS remote path or fail clearly for other schemes."""
        stripped = uri.strip()
        if not stripped.startswith("gs://"):
            raise ValueError(f"GCP Batch storage mounts support gs:// URIs only; got {uri!r}")
        remote_path = stripped[len("gs://") :].strip("/")
        if not remote_path:
            raise ValueError(f"invalid GCS storage mount URI: {uri!r}")
        return remote_path

    def _create_request(self, spec: ManagedJobSpec) -> batch_v1.CreateJobRequest:
        from google.cloud import batch_v1

        request = batch_v1.CreateJobRequest()
        request.job_id = spec.name
        # The request parent uses the effective region — ``spec.region``
        # when it overrides the runner default. The runner region is
        # still validated against ``allowed_locations`` in
        # ``_validate_region``, but the resources are created in the
        # effective region.
        request.parent = f"projects/{self._project_id}/locations/{self.effective_region(spec)}"
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

    def _translate_task(self, task: Any) -> ManagedTaskStatus:
        """Translate a single ``batch_v1.Task`` into a :class:`ManagedTaskStatus`."""
        status = getattr(task, "status", None)
        raw_state = getattr(status, "state", None)
        if raw_state is None:
            key = "UNKNOWN"
        elif hasattr(raw_state, "name"):
            key = raw_state.name
        else:
            key = str(raw_state)
        exit_code, message = self._extract_task_exit(status, key)
        return ManagedTaskStatus(
            task_index=self._extract_task_index(task),
            state=_TASK_STATE_MAP.get(key, "UNKNOWN"),
            exit_code=exit_code,
            message=message,
        )

    @staticmethod
    def _extract_task_exit(status: Any, state_key: str) -> tuple[int | None, str]:
        """Pull ``exit_code`` + last event description from a ``TaskStatus``.

        The exit code is reported only when the SDK actually provided
        one. Reporting ``0`` for a FAILED task whose events do not
        carry an exit code would falsely imply a clean exit, so the
        sentinel is ``None`` whenever the SDK did not surface a
        concrete number — and state-aware: ``exit_code=0`` paired
        with ``state=FAILED`` is treated as the protobuf default
        (unset) rather than a real exit. A SUCCEEDED task with the
        default ``0`` is preserved as the legitimate success code.
        """
        if status is None:
            return None, ""
        exit_code = GcpBatchRunner._last_event_exit_code(status, state_key)
        message = GcpBatchRunner._last_event_description(status)
        return exit_code, message

    @staticmethod
    def _last_event_description(status: Any) -> str:
        events = getattr(status, "status_events", None) or []
        for event in reversed(list(events)):
            description = getattr(event, "description", "") or ""
            if description:
                return description
        return ""

    @staticmethod
    def _last_event_exit_code(status: Any, state_key: str) -> int | None:
        events = getattr(status, "status_events", None) or []
        for event in reversed(list(events)):
            task_execution = getattr(event, "task_execution", None)
            if task_execution is None:
                continue
            code = getattr(task_execution, "exit_code", None)
            if code is None:
                continue
            # State-aware: ``exit_code=0`` on a FAILED task is the
            # protobuf default (no exit code was actually recorded);
            # surface it as ``None`` so consumers do not mistake a
            # default for a real "exited 0". SUCCEEDED tasks keep 0.
            if int(code) == 0 and state_key == "FAILED":
                continue
            return int(code)
        return None

    def _translate_status(self, job: Any) -> ManagedJobStatus:
        status = getattr(job, "status", None)
        state = self._translate_state(status)
        succeeded, failed = self._aggregate_task_counts(job, status)
        total = sum(
            int(getattr(group, "task_count", 0) or 0)
            for group in getattr(job, "task_groups", []) or []
        )
        events = tuple(self._collect_events(status))
        message = events[-1] if events else ""
        return ManagedJobStatus(
            handle=ManagedJobHandle(
                provider=self.provider_name,
                resource_name=getattr(job, "name", "") or "",
                location=self._location_from_job(job),
            ),
            state=state,
            succeeded_tasks=succeeded,
            failed_tasks=failed,
            total_tasks=total,
            message=message,
            raw_events=events,
        )

    def _location_from_job(self, job: Any) -> str:
        """Return the ``project/region`` form derived from the job resource name.

        Resource names look like
        ``projects/{project}/locations/{region}/jobs/{name}``. The
        region in the resource name is the canonical one — the
        submission endpoint may have been a different region (if
        ``spec.region`` overrode the runner), so we read it from
        the resource name rather than using ``self._region``.
        """
        name = getattr(job, "name", "") or ""
        parts = name.split("/")
        if len(parts) >= 5 and parts[0] == "projects" and parts[2] == "locations":
            project = parts[1]
            region = self._strip_location_prefix(parts[3])
            return f"{project}/{region}"
        return f"{self._project_id}/{self._region}"

    def _translate_state(self, status: Any) -> ManagedJobLifecycleState:
        if status is None:
            return ManagedJobLifecycleState.UNKNOWN
        raw_state = getattr(status, "state", None)
        if raw_state is None:
            return ManagedJobLifecycleState.UNKNOWN
        if hasattr(raw_state, "name"):
            key = raw_state.name
        else:
            key = str(raw_state)
        return _STATE_MAP.get(key, ManagedJobLifecycleState.UNKNOWN)

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
                elif state_name in {"FAILED", "CANCELLED", "UNEXECUTED"}:
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


def parse_gcs_mount(entry: str, *, default_bucket: str = "") -> tuple[str, str] | None:
    """Parse a managed-job spec gcs_mount entry into ``(bucket, mount_path)``.

    Recognizes ``gs://bucket/path`` (mounts at ``/path``),
    ``gs://bucket`` (mounts at ``/bucket``), and bare ``/absolute/path``
    (uses ``default_bucket`` as the remote bucket). Returns ``None`` for
    empty / whitespace-only entries so callers can no-op them in a
    tuple of multiple mounts.
    """
    stripped = entry.strip()
    if not stripped:
        return None
    if stripped.startswith("gs://"):
        return _parse_gs_uri(stripped)
    if stripped.startswith("/"):
        return _parse_absolute_path(stripped, default_bucket)
    return None


def _parse_gs_uri(stripped: str) -> tuple[str, str] | None:
    rest = stripped[len("gs://") :]
    if not rest:
        return None
    bucket, _, path = rest.partition("/")
    bucket = bucket.strip()
    if not bucket:
        return None
    path = path.strip("/")
    mount_path = f"/{path}" if path else f"/{bucket}"
    return bucket, mount_path


def _parse_absolute_path(stripped: str, default_bucket: str) -> tuple[str, str] | None:
    if not default_bucket:
        return None
    return default_bucket, stripped


def _already_exists(message: str) -> Exception:
    """Create the provider conflict used by the in-memory client."""
    try:
        from google.api_core.exceptions import AlreadyExists

        return AlreadyExists(message)
    except ImportError:  # pragma: no cover - optional GCP dependency
        return RuntimeError(message)


def _not_found(message: str) -> Exception:
    """Create the provider not-found error used by the in-memory client."""
    try:
        from google.api_core.exceptions import NotFound

        return NotFound(message)
    except ImportError:  # pragma: no cover - optional GCP dependency
        return LookupError(message)


# ---------------------------------------------------------------------------
# In-memory fakes used by the test suite.
# ---------------------------------------------------------------------------


class FakeGcpBatchClient:
    """In-memory stand-in for :class:`batch_v1.BatchServiceClient`.

    Records submitted jobs by ``resource_name`` and serves them to
    :meth:`get_job` / :meth:`list_tasks`. Tests can preset
    :attr:`statuses` to drive status transitions, :attr:`events` to
    drive status-event messages, and :attr:`tasks` to drive per-task
    snapshots.
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
        resource_name = f"{request.parent}/jobs/{request.job_id}"
        if resource_name in self.jobs:
            raise _already_exists(f"job already exists: {resource_name}")
        self.submit_calls.append(request)
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

    def get_job(self, request: Any) -> Any:
        """Return the most recent job record, augmented with current status."""
        name = getattr(request, "name", None)
        if name is None and isinstance(request, dict):
            name = request.get("name")
        if not name:
            raise ValueError("get_job request must include name")
        if name not in self.jobs:
            raise _not_found(f"job not found: {name}")
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
        if name not in self.jobs:
            raise _not_found(f"job not found: {name}")
        self.cancel_calls.append(name)
        self.statuses[name] = _make_job_status("CANCELLATION_IN_PROGRESS")

    def delete_job(self, name: str) -> None:
        """Record the deletion call and drop the cached job state."""
        if name not in self.jobs:
            raise _not_found(f"job not found: {name}")
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

    # proto-plus copies the underlying protobuf on assignment, so we
    # populate the status's events BEFORE handing it to the job.
    if status is not None and status_events:
        status.status_events = list(status_events)
    job = batch_v1.Job()
    job.name = name
    job.task_groups = task_groups
    job.status = status
    return job


class FakeGcsClient:
    """In-memory stand-in for :class:`storage.Client`.

    Implements the surface used by :class:`GcsSink`: ``bucket``,
    ``get_bucket``, ``list_blobs``, and per-bucket ``blob``
    round-tripping. Tests can inspect :attr:`uploads` /
    :attr:`downloads` / :attr:`copy_calls` / :attr:`delete_calls` to
    assert on call sequences and preconditions.
    """

    def __init__(self) -> None:
        """Initialise an empty in-memory GCS namespace."""
        self.buckets: dict[str, dict[str, bytes]] = {}
        self.generations: dict[tuple[str, str], int] = {}
        self.uploads: list[tuple[str, str, bytes, str | None, int | None]] = []
        self.downloads: list[tuple[str, str, int | None]] = []
        self.copy_calls: list[tuple[str, str, str, int | None, int | None]] = []
        self.delete_calls: list[tuple[str, str, int | None]] = []
        self.list_calls: list[tuple[str, str | None]] = []
        self.errors: dict[tuple[str, str], Exception] = {}

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

    def list_blobs(self, bucket_name: str, prefix: str | None = None) -> list[FakeGcsBlob]:
        """Return the blobs under ``prefix`` (sorted)."""
        self.list_calls.append((bucket_name, prefix))
        bucket = self.buckets.get(bucket_name, {})
        return [
            FakeGcsBlob(self, bucket_name, key)
            for key in sorted(bucket)
            if prefix is None or key.startswith(prefix)
        ]

    def copy_blob(
        self,
        blob: Any,
        destination_bucket: str,
        new_name: str,
        *,
        if_generation_match: int | None = None,
        if_source_generation_match: int | None = None,
    ) -> FakeGcsBlob:
        """Pretend to copy ``blob`` to ``destination_bucket/new_name``.

        Honours ``if_generation_match`` (destination) and
        ``if_source_generation_match`` (source) by raising the same
        ``PreconditionFailed`` error the real SDK would raise.
        """
        bucket_name = getattr(blob, "_bucket", "")
        key = getattr(blob, "_key", "")
        dest = f"{destination_bucket}/{new_name}"
        self.copy_calls.append(
            (bucket_name, key, dest, if_generation_match, if_source_generation_match)
        )
        # Source precondition: read the live object.
        src_gen = self.generations.get((bucket_name, key))
        if if_source_generation_match is not None and src_gen != if_source_generation_match:
            raise _precondition_failed("source generation mismatch")
        data = self.buckets.get(bucket_name, {}).get(key, b"")
        # Destination precondition: the destination must be at the
        # generation we expect (0 = does not exist).
        dest_gen = self.generations.get((destination_bucket, new_name))
        if if_generation_match is not None and dest_gen != if_generation_match:
            raise _precondition_failed("destination generation mismatch")
        return self._put(destination_bucket, new_name, data)

    def _put(self, bucket_name: str, key: str, data: bytes) -> FakeGcsBlob:
        bucket = self.buckets.setdefault(bucket_name, {})
        bucket[key] = data
        gen = self.generations.get((bucket_name, key), 0) + 1
        self.generations[(bucket_name, key)] = gen
        return FakeGcsBlob(self, bucket_name, key, generation=gen)

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

    def list_blobs(self, prefix: str | None = None) -> list[FakeGcsBlob]:
        """List blobs under ``prefix`` in this bucket."""
        return self._client.list_blobs(self._name, prefix)

    def copy_blob(
        self,
        blob: Any,
        new_name: str,
        *,
        if_generation_match: int | None = None,
        if_source_generation_match: int | None = None,
    ) -> FakeGcsBlob:
        """Convenience pass-through to ``client.copy_blob``."""
        return self._client.copy_blob(
            blob,
            self._name,
            new_name,
            if_generation_match=if_generation_match,
            if_source_generation_match=if_source_generation_match,
        )

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
    """Stand-in for :class:`google.cloud.storage.blob.Blob`.

    Tracks generation numbers so preconditioned uploads/deletes raise
    the same :class:`google.api_core.exceptions.PreconditionFailed`
    the real SDK would raise.
    """

    def __init__(
        self,
        client: FakeGcsClient,
        bucket: str,
        key: str,
        generation: int | None = None,
    ) -> None:
        """Bind this blob to its fake client, bucket, and key."""
        self._client = client
        self._bucket = bucket
        self._key = key
        self.name = key
        self.generation = generation

    def upload_from_string(
        self,
        data: bytes,
        content_type: str | None = None,
        if_generation_match: int | None = None,
        if_generation_not_match: int | None = None,
    ) -> None:
        """Store ``data`` in the in-memory bucket; honour preconditions."""
        if (self._bucket, self._key) in self._client.errors:
            raise self._client.errors[(self._bucket, self._key)]
        current = self._client.generations.get((self._bucket, self._key))
        self._enforce_precondition(
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            current=current,
        )
        self._client.uploads.append(
            (self._bucket, self._key, data, content_type, if_generation_match)
        )
        self._client.buckets.setdefault(self._bucket, {})[self._key] = data
        next_gen = (current or 0) + 1
        self._client.generations[(self._bucket, self._key)] = next_gen
        self.generation = next_gen

    def download_as_bytes(
        self,
        if_generation_match: int | None = None,
        if_generation_not_match: int | None = None,
    ) -> bytes:
        """Return the previously stored bytes, or empty bytes."""
        current = self._client.generations.get((self._bucket, self._key))
        self._enforce_precondition(
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            current=current,
        )
        self._client.downloads.append((self._bucket, self._key, if_generation_match))
        return self._client.buckets.get(self._bucket, {}).get(self._key, b"")

    def exists(self) -> bool:
        """Return True if this blob is in the in-memory bucket."""
        return self._key in self._client.buckets.get(self._bucket, {})

    def delete(
        self,
        if_generation_match: int | None = None,
        if_generation_not_match: int | None = None,
    ) -> None:
        """Remove the blob from the in-memory bucket; honour preconditions."""
        current = self._client.generations.get((self._bucket, self._key))
        self._enforce_precondition(
            if_generation_match=if_generation_match,
            if_generation_not_match=if_generation_not_match,
            current=current,
        )
        self._client.delete_calls.append((self._bucket, self._key, if_generation_match))
        self._client.buckets.setdefault(self._bucket, {}).pop(self._key, None)
        self._client.generations.pop((self._bucket, self._key), None)

    def download_to_filename(
        self,
        filename: str,
        **kwargs: Any,
    ) -> None:
        """Stream the blob to a local file path."""
        from pathlib import Path

        data = self.download_as_bytes(**kwargs)
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def reload(self) -> None:
        """Refresh :attr:`generation` from the live in-memory map."""
        # When the object does not exist, the real SDK returns ``None``
        # for ``generation`` after reload. The fake mirrors that.
        self.generation = self._client.generations.get((self._bucket, self._key))

    def _enforce_precondition(
        self,
        *,
        if_generation_match: int | None,
        if_generation_not_match: int | None,
        current: int | None,
    ) -> None:
        # The GCS API treats a non-existent object as generation 0 for
        # the purposes of if_generation_match. Normalise so the create-
        # only path (if_generation_match=0) succeeds the first time.
        normalised_current = current if current is not None else 0
        if if_generation_match is not None and normalised_current != if_generation_match:
            raise _precondition_failed(
                f"if_generation_match={if_generation_match} but current={current}"
            )
        if if_generation_not_match is not None and normalised_current == if_generation_not_match:
            raise _precondition_failed(
                f"if_generation_not_match={if_generation_not_match} but current={current}"
            )


def _precondition_failed(message: str) -> Exception:
    """Build the same exception class the real SDK raises on generation mismatch.

    Imported lazily so the test runner does not require
    ``google.api_core`` at import time.
    """
    try:
        from google.api_core.exceptions import PreconditionFailed

        return PreconditionFailed(message)
    except Exception:  # pragma: no cover - depends on api_core
        return RuntimeError(message)
