---
title: Agent Learning Documentation
description: Non-obvious patterns that prevent repeated mistakes across sprints
---

## Template

- **Context**: When/where this applies
- **Problem**: What issue this solves
- **Solution**: Implementation approach
- **Example**: Working code
- **References**: Related files

## Learned Patterns

### R2 lifecycle admin: separate credentials, fail-closed verification

- **Context**: Administering Cloudflare R2 bucket lifecycle rules
  (storage-class / expiry) without leaking worker credentials and
  without silently overwriting unrelated rules.
- **Problem**: Worker credentials should never carry bucket-policy
  write authority; the bucket may already host unrelated lifecycle
  rules owned by other operators; an external edit between plan and
  apply must abort rather than silently overwrite.
- **Solution**: `R2AdminCredentials` reads ONLY `R2_ADMIN_*` env
  vars — worker-style `R2_*` keys are explicitly rejected to
  enforce least-privilege separation between object-write
  credentials and bucket-policy admin credentials. The CLI's
  documentation must reflect this; documenting a fallback would
  let the worker credentials file be passed off as admin
  credentials. `R2LifecycleManager` owns exactly one rule
  identified by a deterministic ID derived from `(bucket,
  canonical_prefix)`. Plans carry a `source_fingerprint`;
  `apply` re-reads and compares before PUTting. Every actual
  write is followed by a read-after-write verification against
  the *full* rule collection (not just the managed rule).
- **Example**: `src/vastai_gpu_runner/storage/r2_lifecycle.py`
- **References**: PR #39 (planned); see `docs/architecture-r2-collection-handshake.md`
  for the long-term bounded-teardown protocol.

### Worker upload: timeout must bound teardown, not workload

- **Context**: The worker's final R2 upload (DONE marker + exit code)
  runs after the workload succeeds. If R2 is rate-limited or briefly
  unavailable, the worker should not delay instance teardown for the
  full upload timeout.
- **Problem**: A 300-second upload timeout meant a stalled uploader
  cost up to five minutes of leaked instance time per worker, per
  rate-limit incident. Worse, the prior implementation logged "R2
  upload complete" for ANY return code, making transport failures
  invisible.
- **Solution**: Bound the upload at 90 seconds. Catch
  `subprocess.TimeoutExpired` separately, log a warning that
  explicitly notes teardown continues, and inspect `returncode` to
  log success only on `rc == 0`. The workload exit code and
  unconditional `self_destruct()` are preserved unchanged.
- **Example**: `src/vastai_gpu_runner/worker/base.py::upload_results`
- **References**: PR #39 (planned).

### Generated uploader: fail closed on partial completion

- **Context**: Auto-generated `r2_upload.py --done` scripts run on
  workers to publish DONE markers. A non-zero exit code was treated
  as success by the prior logic, but the orchestrator treats the
  DONE marker as authoritative for accepting completion.
- **Problem**: If a worker exitcode upload fails, the orchestrator
  must NOT see a DONE marker — that would let it accept an
  incomplete result set as committed. The prior script unconditionally
  wrote DONE after attempting the final uploads.
- **Solution**: Track every required upload explicitly; publish DONE
  only when all required uploads succeed. `worker.exitcode` upload
  must precede DONE publication. Exit non-zero on any required
  failure so the worker's `upload_results()` sees a non-zero
  returncode.
- **Example**: `src/vastai_gpu_runner/storage/r2.py::generate_upload_script`
- **References**: PR #39 (planned).

### Fake boto3 in subprocess tests: sys.path.strip + insert

- **Context**: Behavioural tests for generated R2 upload scripts need
  to execute the script under a fake `boto3` module that records
  calls without hitting AWS.
- **Problem**: Setting only `PYTHONPATH` does not override an
  installed `boto3` in site-packages; the real client is loaded
  before the fake takes effect.
- **Solution**: Prepend a script-level prologue that strips
  site-packages from `sys.path` and inserts the fake directory at
  index 0, *before* the generated `import boto3` statement. Use
  `sys.exit(rc)` patterns and atexit-registered persistence for
  reliable call-log capture.
- **Example**: `tests/test_r2_upload_scripts.py::_run_script`
- **References**: PR #39 (planned).

### Fail-closed uploader: positive completion proof, not just negative evidence

- **Context**: The worker shard DONE marker is authoritative for the
  orchestrator. A generated upload script that publishes DONE despite
  a failed required upload lets the orchestrator accept an incomplete
  result set as committed.
- **Problem**: A failure-only sentinel is not enough:
  - Sentinel write itself can fail (disk full, permission denied).
  - A subsequent successful `--done` invocation sees no failure
    evidence and publishes DONE.
  - A no-arg `upload_all()` upload of `outputs/<flat_file>` (no
    prediction subdirectory) has no per-prediction completion marker
    to track.
  - A failed `upload_all()` followed by a successful retry cannot
    clear the prior `<no-arg>` failure entry.
- **Solution**: Positive completion proof via atomically-written
  markers.
  - Each `--prediction` writes `<workspace>/prediction_completed/<name>`.
  - Each successful `upload_all()` writes the same per-prediction
    markers AND `<workspace>/shard_completed` atomically. It also
    clears any stale `<no-arg>` failure entry.
  - `--done` requires EITHER the shard_completed marker OR
    per-prediction completion markers for every prediction directory
    under `outputs/`. Flat files at the top of `outputs/` require
    `shard_completed` (no per-prediction marker applies).
  - Sentinel-write failures fall through to a global
    `SENTINEL_WRITE_FAILED` entry; even if THAT also fails, the
    unreadable-file state itself counts as evidence of unresolved
    failures.
- **Example**: `src/vastai_gpu_runner/storage/r2.py`
  (`_has_unresolved_upload_failures`, `_record_shard_complete`,
  `_record_prediction_success`, `_clear_shard_complete_marker`).
- **References**: PR #39.

### boto3 lifecycle contract: LifecycleConfiguration wrapper required

- **Context**: Calling `put_bucket_lifecycle_configuration(Rules=[...])`
  on a real boto3 client raises `ParamValidationError` before any
  request leaves the SDK. The correct request shape is
  `LifecycleConfiguration={"Rules": [...]}`.
- **Problem**: A test double that accepts the wrong shape masks the
  bug at the unit-test layer. The production client fails on the
  first non-no-op apply or remove.
- **Solution**: Use a real boto3 client wired to `botocore.stub.Stubber`
  for at least one contract test per mutation path. `Stubber.add_response`
  validates the parameter names against the service model — any
  incorrect shape raises before the stub is consulted. The fake
  `FakeS3` used by the rest of the suite mirrors the real
  `LifecycleConfiguration=` signature so the two test populations
  stay consistent.
- **Example**: `tests/test_r2_lifecycle.py::TestBoto3Contract`.
- **References**: PR #39.

### Bucket removal: DELETE when no rules remain

- **Context**: Cloudflare R2 / S3 reject `put_bucket_lifecycle_configuration`
  with an empty `Rules` array — a lifecycle configuration must
  contain at least one rule.
- **Problem**: `remove()` that always PUTs the post-state fails when
  the managed rule was the only rule on the bucket.
- **Solution**: Branch on the post-removal state. If any rules
  remain, `put_bucket_lifecycle_configuration`. If none remain,
  `delete_bucket_lifecycle`. Both paths still re-read after the
  mutation for verification.
- **Example**: `src/vastai_gpu_runner/storage/r2_lifecycle.py::remove`
  + `_delete_lifecycle`.
- **References**: PR #39.

### `run_full_cycle` deploys through launch only; callers poll, collect, destroy

- **Context**: Any synchronous path that needs a worker's *result* —
  the `run` CLI command and programmatic callers of `LocalRunner`
  (or any `CloudRunner` backend).
- **Problem**: The inherited `CloudRunner.run_full_cycle` name
  suggests the whole lifecycle, but it stops after `launch_worker`:
  it never polls for completion, downloads, or destroys. Code that
  treats its return as "done" leaks the instance/process and misses
  results.
- **Solution**: Treat `run_full_cycle` as deploy-through-launch. The
  caller must poll `check_progress` for the `DONE` marker (or worker
  death), collect via `download_all_results` / `download_file`, and
  destroy in a `finally` block. `cli_run.py` is the reference
  implementation of this sequence.
- **Example**: `src/vastai_gpu_runner/cli_run.py`;
  `tests/integration/test_local_runner_integration.py`.
- **References**: `docs/api.md` (Runner section), `docs/extending.md`
  (programmatic local-run example).
