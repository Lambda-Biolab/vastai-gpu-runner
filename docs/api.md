# API reference

## Types (`vastai_gpu_runner.types`)

| Class | Description |
|-------|-------------|
| `Provider` | Enum: `VASTAI`, `RUNPOD` |
| `InstanceStatus` | Enum: `CREATING`, `BOOTING`, `RUNNING`, `FAILED`, `DESTROYED` |
| `DeploymentConfig` | GPU model, cost limits, timeouts, workspace, reliability thresholds |
| `CloudInstance` | Instance metadata: ID, SSH host/port, GPU model, cost, status |
| `DeploymentResult` | Success flag, instance reference, error message, output files |

### `DeploymentConfig` fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu_model` | `str` | `"RTX_4090"` | GPU model identifier |
| `max_cost_per_hour` | `float` | `0.45` | Maximum $/hr for offers |
| `boot_timeout_seconds` | `int` | `300` | Max wait for instance boot |
| `gpu_verify_timeout` | `int` | `120` | Max wait for GPU verification |
| `min_disk_gb` | `int` | `40` | Minimum disk space |
| `min_network_mbps` | `int` | `800` | Minimum download bandwidth |
| `min_reliability` | `float` | `0.995` | Minimum host reliability score |
| `worker_script` | `str` | `"worker.sh"` | Worker script filename |
| `workspace_dir` | `str` | `"/workspace"` | Remote workspace path |
| `conda_env_spec` | `str` | `""` | Conda packages to install (empty = skip setup) |

## Runner (`vastai_gpu_runner.runner`)

| Method | Returns | Description |
|--------|---------|-------------|
| `search_offers(**kwargs)` | `list[dict]` | Search marketplace for GPU offers |
| `create_instance(offer)` | `CloudInstance` | Create instance from an offer |
| `wait_for_boot(instance)` | `bool` | Wait for running status |
| `verify_gpu(instance)` | `bool` | Verify GPU via nvidia-smi |
| `deploy_files(instance, files)` | `bool` | Upload files via SCP |
| `setup_environment(instance)` | `bool` | Install deps (micromamba/conda) |
| `launch_worker(instance)` | `bool` | Start worker process |
| `check_progress(instance)` | `dict` | Check DONE file / PID liveness |
| `list_remote_files(instance)` | `list[str]` | List workspace files |
| `download_file(instance, name, path)` | `bool` | Download single file via SCP |
| `destroy_instance(instance)` | `bool` | Tear down instance |
| `run_full_cycle(files, output_dir, ...)` | `DeploymentResult` | Full lifecycle with retry |
| `download_all_results(instance, dir, ...)` | `list[str]` | Bulk rsync download |

### `run_full_cycle` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `files` | `dict[str, Path]` | required | Remote name -> local path mapping |
| `local_output_dir` | `Path` | required | Where to download results |
| `max_retries` | `int` | `3` | Maximum deployment attempts |
| `offers` | `list[dict] \| None` | `None` | Pre-fetched offers (avoids re-query) |
| `used_machine_ids` | `set[str] \| None` | `None` | Machine IDs claimed by other threads |
| `machine_lock` | `Lock \| None` | `None` | Lock protecting `used_machine_ids` |

## Vast.ai Provider (`vastai_gpu_runner.providers.vastai`)

| Class/Function | Description |
|----------------|-------------|
| `VastaiRunner(config, allowed_images, docker_image, min_gpu_vram_mib)` | Hardened Vast.ai runner |
| `vastai_cmd(args, timeout)` | Run `vastai` CLI command, returns stdout |
| `verify_instance_ownership(instance_id, allowed_images)` | Check instance belongs to project |
| `GPU_NAME_MAP` | Dict mapping model IDs to Vast.ai GPU names |
| `DEFAULT_IMAGE` | Default Docker image (`nvidia/cuda:12.4.0-devel-ubuntu22.04`) |

## SSH (`vastai_gpu_runner.ssh`)

| Function | Returns | Description |
|----------|---------|-------------|
| `ssh_cmd(instance, command, timeout=30)` | `(int, str)` | Execute command, returns (rc, stdout) |
| `scp_upload(instance, local_path, remote_path, timeout=300)` | `bool` | Upload file |
| `scp_download(instance, remote_path, local_path, timeout=600)` | `bool` | Download file (checks non-empty) |

All SSH functions use `StrictHostKeyChecking=no` (ephemeral IPs) and `stdin=DEVNULL` (prevents stdin stealing).

## State (`vastai_gpu_runner.state`)

### Shard-based (`BatchState`)

| Class | Description |
|-------|-------------|
| `ShardState` | Per-shard: `shard_id`, `instance_id`, `status`, `item_ids`, `cost_per_hour`, `retry_count` |
| `BatchState` | Collection with `save(path)`, `load(path)`, `active_shards`, `failed_shards`, `pending_shards`, `downloaded_shards` |

Status flow: `pending` -> `deployed` -> `running` -> `downloaded` -> `destroyed` | `failed`

### Job-based (`JobBatchState`)

| Class | Description |
|-------|-------------|
| `JobState` | Per-job: `job_name`, `status`, `instance_id`, `cost_per_hour`, `cost_usd` (computed) |
| `JobBatchState` | Collection with `save(path)`, `load(path)`, `pending_jobs`, `active_jobs`, `completed_jobs`, `total_cost` |

Status flow: `pending` -> `deploying` -> `running` -> `completed` -> `downloaded` | `failed`

## Worker (`vastai_gpu_runner.worker`)

### `BaseWorker`

| Method | Override? | Description |
|--------|-----------|-------------|
| `main()` | No | Template method — runs the full lifecycle |
| `write_pid()` | No | Write `worker.pid` |
| `check_gpu(min_memory_mib, max_temp_c)` | No | GPU health via nvidia-smi |
| `preflight_gates()` | Optional | Return list of `() -> bool` callables. Default: `[_check_r2]` |
| `run_workload()` | **Required** | Your GPU code. Return 0 for success. |
| `upload_results()` | Optional | Default: call `r2_upload.py --done` |
| `self_destruct()` | No | Vast.ai REST DELETE (reads env vars) |

### Health checks (`vastai_gpu_runner.worker.health`)

| Function | Returns | Description |
|----------|---------|-------------|
| `check_gpu(min_memory_mib=0, max_temp_c=90)` | `bool` | nvidia-smi temp + ECC + optional VRAM check |
| `check_r2_connectivity(workspace)` | `bool` | Run `r2_upload.py --check` if script exists |

## Orchestrator (`vastai_gpu_runner.orchestrator`)

| Function | Returns | Description |
|----------|---------|-------------|
| `sweep_zombie_instances(live_runners, label_prefix, r2_sink, r2_batch_id)` | `int` | Destroy orphaned instances, returns count |
| `ensure_detached(log_path, pid_path, detach_env_var)` | `None` | Fork + setsid for SSH disconnect survival |
| `check_budget(spent, ceiling)` | `bool` | True if within budget, warns at 80% |
| `poll_instance_progress(instance, workspace)` | `dict` | 3-layer check: DONE file -> PID liveness -> log tail |
| `load_vastai_api_key()` | `str` | Read from `~/.config/vastai/vast_api_key` or `~/.vast_api_key` |

## Estimator (`vastai_gpu_runner.estimator`)

### Core (`vastai_gpu_runner.estimator.core`)

| Name | Type | Description |
|------|------|-------------|
| `GPU_SPEED_FACTOR` | `dict` | `RTX_3090=0.77`, `RTX_4090=1.0`, `RTX_5090=1.43` |
| `GPU_TYPES` | `list` | `["RTX_3090", "RTX_4090", "RTX_5090"]` |
| `FALLBACK_PRICES` | `dict` | Static $/hr: 3090=$0.15, 4090=$0.32, 5090=$0.60 |
| `PriceSummary` | dataclass | GPU pricing snapshot: min/max/median $/hr, count |
| `ScalingRow` | dataclass | One table row: GPUs, wall time, cost range, notes |
| `EstimateResult` | dataclass | Full result with `to_dict()` and `to_rich_table()` |
| `build_scaling_table(work_hours, gpu_counts, pricing, ...)` | `list[ScalingRow]` | Compute scaling table |
| `fallback_pricing(gpu_types)` | `dict[str, PriceSummary]` | Static prices (offline) |
| `cheapest_gpu_type(pricing)` | `str` | Lowest median price GPU |
| `format_time(hours)` | `str` | `2.25` -> `"2h 15m"` |
| `record_timing(path, workload, **metrics)` | `None` | Append to JSONL benchmarks |
| `load_calibration(path, workload)` | `list[dict]` | Load benchmark records |

### Pricing (`vastai_gpu_runner.estimator.pricing`)

| Function | Returns | Description |
|----------|---------|-------------|
| `query_vastai_pricing(gpu_types, max_cost_per_hour, ...)` | `dict[str, PriceSummary]` | Live marketplace query with fallback |
