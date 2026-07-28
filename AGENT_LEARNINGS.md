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
