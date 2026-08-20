# API reference

This reference describes the public surface in the current `0.6.0` package.
The recommended next release is `0.7.0`; it is not published.

## Public exports

`vastai_gpu_runner` exports:

```text
BatchOrchestrator, BatchUnit, CloudInstance, CloudRunner, ComputeMode,
DeploymentConfig, DeploymentResult, FailureVerdict, GcpBatchRunner,
InstanceStatus, ManagedJobAlreadyExistsError, ManagedJobConflictError,
ManagedJobError, ManagedJobHandle, ManagedJobLifecycleState,
ManagedJobNotFoundError, ManagedJobPermanentError, ManagedJobRunner,
ManagedJobSpec, ManagedJobStatus, ManagedJobTerminalState,
ManagedJobTransientError, ManagedTaskStatus, Provider, StorageMount
```

The managed-job package additionally exports `BootDisk`, `ComputeResource`,
`GpuAccelerator`, `MachineResource`, `NetworkConfig`, `ServiceAccount`,
`CURRENT_MANAGED_JOB_SCHEMA`, `ManagedJobState`, `ManagedJobStateError`,
`load_managed_job_state`, `load_or_none`, and `save_managed_job_state`.
Import these from `vastai_gpu_runner.managed_jobs`.

`vastai_gpu_runner.storage` exports `GcsPreconditionFailed`, `GcsSink`,
`streaming_upload`, and `upload_json_atomic`. `R2Sink` remains available from
`vastai_gpu_runner.storage.r2`; it is not re-exported by the package root.

## Core types (`vastai_gpu_runner.types`)

| Class | Contract |
|---|---|
| `Provider` | `VASTAI`, `RUNPOD`, or `LOCAL`. `RUNPOD` is an enum value only; no RunPod runner is included. |
| `InstanceStatus` | `CREATING`, `BOOTING`, `RUNNING`, `FAILED`, or `DESTROYED`. |
| `DeploymentConfig` | Direct-VM deployment settings: GPU, cost, timeouts, workspace, worker script, and checkpoint flags. |
| `CloudInstance` | Direct-VM instance metadata, including provider, ID, GPU, cost, status, label, and SSH fields. |
| `DeploymentResult` | Deployment success, optional instance, error text, and output-file names. |

## Runner (`vastai_gpu_runner.runner`)

`CloudRunner` is the direct VM lifecycle base class. Its constructor is
`CloudRunner(config: DeploymentConfig | None = None)`. Subclasses implement
the provider operations; `run_full_cycle` runs through worker launch and
returns a `DeploymentResult`. It does not poll, download, or destroy after
launch.

| Method | Signature | Return |
|---|---|---|
| `search_offers` | `search_offers(**kwargs: object)` | `list[dict[str, object]]` |
| `create_instance` | `create_instance(offer: Mapping[str, object])` | `CloudInstance` |
| `wait_for_boot` | `wait_for_boot(instance)` | `bool` |
| `verify_gpu` | `verify_gpu(instance)` | `bool` |
| `deploy_files` | `deploy_files(instance, files: dict[str, Path])` | `bool` |
| `setup_environment` | `setup_environment(instance)` | `bool` |
| `launch_worker` | `launch_worker(instance)` | `bool` |
| `check_progress` | `check_progress(instance)` | `dict[str, object]` |
| `list_remote_files` | `list_remote_files(instance)` | `list[str]` |
| `download_file` | `download_file(instance, remote_name, local_path)` | `bool` |
| `destroy_instance` | `destroy_instance(instance)` | `bool` |
| `download_all_results` | `download_all_results(instance, local_dir, *, remote_subdir="", critical_files=None)` | `list[str]` |
| `run_full_cycle` | `run_full_cycle(files, local_output_dir, *, max_retries=3, offers=None, used_machine_ids=None, machine_lock=None)` | `DeploymentResult` |

The `run_full_cycle` `local_output_dir` parameter is reserved by the current
implementation. Callers own polling, collection, and destruction.

## Vast.ai provider (`vastai_gpu_runner.providers.vastai`)

`VastaiRunner` implements `CloudRunner`:

```python
VastaiRunner(
    config: DeploymentConfig | None = None,
    *,
    ownership: OwnershipPolicy | None = None,
    credentials: CredentialResolution | None = None,
    label_prefix: str | None = None,
    allowed_images: frozenset[str] | None = None,  # deprecated
    docker_image: str = DEFAULT_IMAGE,
    min_gpu_vram_mib: int = MIN_GPU_VRAM_MIB,
    setup_commands: list[str] | None = None,
)
```

Use `ownership=OwnershipPolicy(...)` for new code. `allowed_images=` is a
deprecated compatibility alias; passing both names raises `ValueError`.
`VastaiRunner.from_config(canonical)` accepts a `VastaiProviderConfig`.

The provider module also exposes `list_vastai_instances(*, credentials)`,
`verify_instance_ownership(instance_id, *, ownership)`, and
`build_vastai_cleanup_policy(*, ownership, credentials)`. Ownership
verification returns the tagged `OwnershipVerification` result, not a
boolean. `VASTAI_TERMINAL_STATES` contains the provider's terminal states.

## Local provider (`vastai_gpu_runner.providers.local`)

`LocalRunner(config: DeploymentConfig | None = None)` runs one worker as a
local subprocess. It needs no cloud credentials, SSH, Docker, or network
provider. `build_local_cleanup_policy()` returns the local cleanup policy.
The `run` CLI command is the synchronous poll/collect/destroy wrapper around
this lifecycle.

## Batch orchestrator (`vastai_gpu_runner.batch`)

`BatchOrchestrator` is an abstract, generic coordinator above `CloudRunner`.
Its constructor is keyword-only:

```python
BatchOrchestrator(
    *,
    runner_factory,
    label_prefix,
    cleanup_policy,
    workspace_dir="/workspace",
    r2_sink=None,
    r2_batch_id="",
    budget_usd=0.0,
    max_retries=2,
    max_parallel_deploys=16,
    max_parallel_collects=1,
    poll_interval_seconds=30,
    zombie_sweep_every_n_cycles=5,
    poll_timeout_seconds=0.0,
)
```

`cleanup_policy` is required. Consumer subclasses implement unit iteration,
state persistence, payload construction, collection, failure classification,
and state-mutation callbacks. Zombie cleanup uses the policy's
`list_instances()` and `destroy(candidate)` methods.

## Managed jobs (`vastai_gpu_runner.managed_jobs`)

`ManagedJobRunner` is a `@runtime_checkable` protocol for declarative batch
providers. It is separate from `CloudRunner`: the managed provider owns the
VM lifecycle, while `CloudRunner` manages a direct VM/SSH lifecycle.

Required protocol methods:

```python
@property
def provider_name(self) -> str: ...

def submit(self, spec: ManagedJobSpec) -> ManagedJobHandle: ...
def get_status(self, handle: ManagedJobHandle) -> ManagedJobStatus: ...
def list_tasks(self, handle: ManagedJobHandle) -> Iterable[ManagedTaskStatus]: ...
def cancel(self, handle: ManagedJobHandle) -> None: ...
def delete(self, handle: ManagedJobHandle) -> None: ...
```

### Data-transfer objects

| Type | Fields |
|---|---|
| `ManagedJobSpec` | `name`, `task_count`, `parallelism`, `image`, `command`, `environment`, `labels`, `gcs_mounts` (deprecated), `storage_mounts`, `timeout_seconds`, `retry_on_preempt`, `region`, `machine_resource`, `compute_resource`, `service_account`, `network`, `allowed_locations`, `spot` |
| `StorageMount` | `uri`, `mount_path`, `read_only=False` |
| `ManagedJobHandle` | `provider`, `resource_name`, `location` |
| `ManagedJobStatus` | `handle`, `state`, `succeeded_tasks`, `failed_tasks`, `total_tasks`, `message`, `raw_events` |
| `ManagedTaskStatus` | `task_index`, `state`, `exit_code`, `message` |
| `BootDisk` | `image`, `size_gb`, `type_` |
| `GpuAccelerator` | `type_`, `count`, `driver_version`, `install_gpu_drivers` |
| `MachineResource` | `machine_type`, `boot_disk`, `accelerators`, `min_cpu_platform` |
| `ComputeResource` | `cpu_milli`, `memory_mib`, `boot_disk_mib` |
| `ServiceAccount` | `email`, `scopes` |
| `NetworkConfig` | `network`, `subnetwork`, `no_external_ip_address` |

`ManagedJobLifecycleState` is the canonical enum:
`QUEUED`, `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLING`,
`CANCELLED`, and `UNKNOWN`. Only `SUCCEEDED`, `FAILED`, `CANCELLED`, and
`UNKNOWN` are terminal. `ManagedJobTerminalState` is a compatibility alias to
the same enum; it does not collapse in-progress states into `UNKNOWN`.

`ManagedJobState` uses schema 2 and neutral `correlation_metadata`. Loading
schema 0 or 1 migrates it to schema 2 while preserving legacy `campaign_id`
and `stage_id` fields. Unknown schemas, missing files, malformed JSON, and
missing required state keys raise `ManagedJobStateError` through
`load_managed_job_state`; `load_or_none` is the compatibility loader that
returns `None` for those failures. `save_managed_job_state` writes a temporary
file and replaces the destination.

### Errors and operation semantics

`ManagedJobError` is the base class. `ManagedJobTransientError` is normally
retryable; `ManagedJobPermanentError` is not. `ManagedJobConflictError`
(`ManagedJobAlreadyExistsError` is its alias) reports a duplicate provider
resource, including a duplicate `ManagedJobSpec.name`. `ManagedJobNotFoundError`
reports a missing resource. GCP provider exceptions are mapped to these
types, with the original SDK exception preserved as the cause.

`submit` does not treat a duplicate name as implicit success. `cancel` and
`delete` are behaviorally idempotent for provider not-found responses.

### GCP Batch implementation

`GcpBatchRunner` implements `ManagedJobRunner`:

```python
GcpBatchRunner(
    project_id: str,
    region: str,
    client=None,
    storage_client=None,
    bucket_name: str = "",
)
```

The `client` and `storage_client` arguments are optional seams for fakes and
stubbed runs. `bucket_name` is the current constructor name; `project` and
`location` are not constructor parameters. The runner uses `spec.region` when
present, validates `allowed_locations`, and returns handles in
`project/region` form. GCP Batch currently supports `gs://` `StorageMount`
URIs and rejects other schemes. `gcs_mounts` remains for compatibility.

`list_tasks` exposes task index, normalized task state, optional exit code, and
the latest task-event description. Job status exposes aggregate task counts,
the latest event message, and raw event descriptions.

## GCS storage (`vastai_gpu_runner.storage.gcs`)

`GcsSink(bucket_name, client=None, chunk_size=40 * 1024 * 1024)` provides
upload, download, listing, CAS, and SHA-256 operations. A missing bucket
raises `KeyError`.

- `upload_bytes`, `upload_file`, and `upload_text` are create-only writes and
  use `if_generation_match=0`.
- `upload_cas_write` compares the expected object generation. `None` means
  create-only. Generation mismatches raise `GcsPreconditionFailed`; other SDK
  errors propagate unchanged.
- `read_cas` returns `(data, generation)` and returns `(b"", None)` when the
  object is missing.
- `upload_atomic_json` and `upload_json_atomic` are compatibility names for a
  staged temporary write followed by final-key overwrite. They do not perform
  an atomic rename.
- `download_all(prefix, dest)` returns written paths and rejects unsafe object
  names. Completion is the expected GCS object set; `GcsSink` neither creates
  nor requires a `DONE` marker.

## CLI

The installed command is `vastai-gpu-runner`. Current commands are:

```text
check
instances
estimate
cleanup
batch
run
r2-lifecycle
```

`batch` composes the direct Vast.ai batch state/configuration and prints a JSON
summary; it does not submit a `ManagedJobSpec` to GCP Batch. Managed jobs are
submitted through the Python `ManagedJobRunner` API.

`r2-lifecycle` provides `show`, `apply`, and `remove` for one managed
Cloudflare R2 expiration rule per prefix. It requires a credentials file with
`R2_ADMIN_*` keys and uses exit codes 0 through 9 for the documented success,
validation, credential, access, collision, stale-plan, rule-limit,
verification, and confirmation outcomes.
