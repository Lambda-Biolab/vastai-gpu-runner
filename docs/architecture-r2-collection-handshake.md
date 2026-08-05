# Long-term R2 collection handshake (proposed — not implemented)

> **Status:** future architecture. **Not implemented in the current change.**
> This document records the target state machine and protocol that will
> replace the best-effort rsync fallback once `R2_FINAL_UPLOAD_TIMEOUT_SECONDS`
> is no longer adequate. See `docs/guide.md` for the operator-facing
> lifecycle policy and `docs/api.md` for the current API surface.

## Motivation

The current worker teardown path is:

1. `BaseWorker.upload_results()` runs the generated uploader with a
   fixed 90-second timeout (`R2_FINAL_UPLOAD_TIMEOUT_SECONDS`).
2. On `TimeoutExpired` or non-zero return code, the worker logs a
   warning and continues.
3. `self_destruct()` runs unconditionally from `main()`'s `finally`.
4. The orchestrator later attempts rsync-based collection if the
   instance is still alive or `R2Sink.is_shard_done` reports missing.

This works for transient R2 outages but has two known failure modes:

- **Pre-poll deployment race.** The orchestrator completes the entire
  parallel deployment phase before starting the polling phase. A
  worker that finishes its workload and uploads DONE during the
  deployment of other units can sit idle until the polling phase
  catches up, while the orchestrator's polling loop only learns about
  completion after `poll_interval_seconds` have elapsed.
- **No acknowledgement handshake.** The worker does not know whether
  the orchestrator has received its results, so it cannot tell the
  difference between "DONE marker is up but orchestrator has not
  collected yet" and "DONE marker is up and orchestrator is
  permanently stuck". Unconditional self-destruct avoids the worst
  case (a leaked instance) but can destroy work-in-progress on the
  remote filesystem before rsync has finished transferring it.

The target architecture replaces this with an explicit, bounded
handshake that closes both failure modes.

## Target state machine

Each shard / job moves through the following states during its
teardown sequence:

```text
WORKLOAD_RUNNING ──► WORKLOAD_COMPLETE ──► TRANSFER_PENDING
                                              │
                                              ├─► R2_COMMITTED ──► COLLECTED ──► DESTROYED
                                              │
                                              └─► RSYNC_REQUIRED ──► COLLECTED ──► DESTROYED
```text

| State | Owner | Description |
|-------|-------|-------------|
| `WORKLOAD_RUNNING` | worker | Workload subprocess is active. |
| `WORKLOAD_COMPLETE` | worker | Workload subprocess exited; local results written to `/workspace`. |
| `TRANSFER_PENDING` | worker + orchestrator | Worker is uploading to R2; orchestrator may also start rsync. |
| `R2_COMMITTED` | orchestrator | Orchestrator has read-after-write verified the full artifact manifest from R2. |
| `RSYNC_REQUIRED` | orchestrator | R2 verification failed for one or more artifacts; orchestrator falls back to rsync from the still-alive instance. |
| `COLLECTED` | orchestrator | All required artifacts present locally; the instance is no longer needed. |
| `DESTROYED` | orchestrator | Instance has been destroyed. Orchestrator-owned normal destruction. |

Transitions are committed by the orchestrator, not the worker, so
the worker never has to second-guess whether the orchestrator has
collected.

## Protocol components

### 1. Orchestrator-owned normal destruction

The orchestrator never trusts a worker's self-report of "done".
Self-destruct becomes a safety net only — the orchestrator waits for
`COLLECTED` (R2 or rsync) and then calls
`ProviderCleanupPolicy.destroy(instance)`. The worker keeps an
unconditional `self_destruct()` only as a last-resort backstop for
the case where the orchestrator itself dies mid-batch.

### 2. Worker-side dead-man lease / watchdog

Each worker holds a short-lived lease on R2 (e.g. a TTL-key written
to a known prefix). If the worker crashes or the network partitions,
the lease expires and the orchestrator treats the unit as
unrecoverable. This replaces the current "trust the worker to call
self_destruct" assumption.

### 3. Concurrent polling while deployments continue

The orchestrator's deployment phase stops blocking the polling
phase. As soon as any unit reaches `WORKLOAD_COMPLETE`, its
`_check_unit` starts running on the polling loop without waiting for
the rest of the batch to deploy. This eliminates the pre-poll
deployment race.

### 4. Manifest with expected artifacts, sizes, and checksums

When the worker enters `WORKLOAD_COMPLETE`, it uploads an immutable
manifest object to R2 alongside the artifacts:

```json
{
  "shard_id": 0,
  "batch_id": "b1",
  "exit_code": 0,
  "artifacts": [
    {"name": "outputs/pred_a.txt", "size_bytes": 12345, "sha256": "..."},
    {"name": "outputs/pred_b.txt", "size_bytes": 67890, "sha256": "..."}
  ]
}
```text

The orchestrator reads this manifest, then verifies each artifact
against the listed size and checksum. Manifest mismatch ⇒
`RSYNC_REQUIRED`. Empty or missing manifest ⇒ `RSYNC_REQUIRED`.

### 5. Immutable `COMMITTED` marker written last

After all artifacts are uploaded and verified, the worker writes a
final, immutable `COMMITTED` object. The orchestrator treats this
object as authoritative; the current `DONE` marker becomes a
best-effort hint for backward compatibility. The `COMMITTED` object
is only written when the manifest verifies against the actual
uploaded objects.

### 6. R2 verification before accepting completion

`_check_unit` does a full read-after-write verification: list the
shard prefix, compare against the manifest, verify checksums. This
replaces the current "one HEAD request on `DONE`" cheap poll.

### 7. Rsync fallback before lease expiry

If R2 verification fails, the orchestrator switches to `RSYNC_REQUIRED`
and begins rsync against the still-alive instance. The worker lease
gives the orchestrator a bounded window — long enough for a full
rsync of the worst-case artifact size, short enough that a leaked
instance cannot bill for long. If the lease expires before rsync
completes, the orchestrator transitions to `DESTROYED` with the
partial result marked failed.

### 8. Cleanup policy as tertiary orphan remediation

The bucket lifecycle policy (see
`docs/guide.md#r2-bucket-lifecycle-administration`) acts as the final
backstop: artifacts left behind by an orchestrator crash expire
after the configured retention. The lifecycle rule does *not* replace
the handshake — it just bounds the cost of a complete orchestrator
failure.

## Why not implement this now

This change is scoped to the current, well-defined gap:

- Bounded the final R2 upload to 90 seconds instead of 300.
- Made the generated uploader fail closed so the orchestrator
  cannot accept an incomplete result set as committed.
- Added a user-configurable bucket lifecycle policy as the tertiary
  backstop described in step 8.

Implementing the full handshake protocol requires changes that
ripple across `BatchOrchestrator`, `BaseWorker`, `R2Sink`, and the
generated uploader scripts, plus new state-machine documentation and
new integration tests. Each of those slices can land independently,
but they should not block the smaller hardening landed in this
change.

## Open questions for a future review

- **Lease TTL granularity.** Too long ⇒ leaked instances bill for
  longer; too short ⇒ rsync fallback window shrinks. Defaults
  should probably be derived from the largest expected artifact size
  and the worker's upload bandwidth.
- **Manifest format versioning.** Should the manifest carry a schema
  version so future changes to artifact layout don't break
  compatibility with already-running orchestrators?
- **Worker trust model.** If the worker is the only writer of the
  manifest and the orchestrator is the only verifier, an attacker
  with R2-write authority on the worker could forge a manifest. The
  current architecture already assumes worker trust; this protocol
  does not change that. A follow-up could add worker-attested
  manifests (e.g. signed by the orchestrator's instance-creation
  token).
- **Rsync partial collection.** What is the right behaviour when
  rsync partially succeeds and the worker self-destructs mid-stream?
  Today: the orchestrator gives up and the unit is marked failed. A
  future improvement could retry rsync against a new ephemeral
  worker or fall back to the lifecycle rule's expiration window.
