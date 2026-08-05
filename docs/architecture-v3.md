# Architecture v3 (target)

This doc describes the **target architecture** after the unit-lifecycle decision tree and belt-and-suspenders destroy refactor land. For the current-state architecture (today's code) see
[`architecture.md`](architecture.md). For the next-step target see
[`architecture-v2.md`](architecture-v2.md).

## What changes vs v2

In one paragraph: the unit-lifecycle decision tree (R2 → SSH → worker_dead re-check → action) moves out of `BatchOrchestrator` into a new `unit_lifecycle` deep module exposing one function `decide_next_action(unit, runner, instance, is_done_in_r2) -> UnitAction`, where `UnitAction` is a tagged union of three frozen dataclasses (`Continue | Complete | Preempt`). The four-step belt-and-suspenders destroy protocol moves out of `VastaiRunner._rest_destroy` and the standalone helpers in `providers/vastai.py` into a new `providers/destroy` deep module exposing `belt_and_suspenders(stop_fn, delete_fn, verify_fn, *, policy: DestroyPolicy) -> DestroyResult`. The destroy protocol returns a typed `DestroyResult` with `verdict` (protocol outcome) or `refusal` (pre-protocol outcome), never both. The destroy module owns the loop shape; provider-specific timing policy lives in a `DestroyPolicy` dataclass supplied by the adapter. Two latent bugs are corrected: the orchestrator's `_destroy_via_rest` silently treats HTTP errors as "instance gone," and the current `VastaiRunner.destroy_instance` ignores the destroy result and unconditionally returns `True`.

Diff vs v2:

- **+** `src/vastai_gpu_runner/unit_lifecycle.py` — `decide_next_action`, `Action` enum, `Continue | Complete | Preempt` plan types, `PreemptCause` enum, `ProgressSnapshot` dataclass
- **+** `src/vastai_gpu_runner/providers/destroy.py` — `VerifyVerdict` + `DestroyVerdict` + `DestroyRefusal` enums, `VerifyResult` + `DestroyResult` + `DestroyPolicy` dataclasses, `belt_and_suspenders()` function
- **+** `src/vastai_gpu_runner/providers/destroy_adapters/vastai.py` — Vast.ai REST callbacks (`stop_fn`, `delete_fn`, `verify_fn`), `destroy_vastai_instance`, `read_vastai_api_key` (env-first precedence, fail-closed), `VASTAI_POLICY`
- **~** `BatchOrchestrator._check_unit` becomes a thin wrapper (one deprecation cycle only — the API reference documents it as inherited lifecycle surface) that delegates to `decide_next_action` and a shared dispatch helper. `_classify_live_unit` is **deleted**, not wrapped.
- **~** `BatchOrchestrator._poll_cycle_once` and `_check_unit` share a single dispatch helper so their logic cannot diverge
- **~** `BatchOrchestrator._sweep_zombies` routes through `destroy_vastai_instance` (with CLI fallback when no API key) and increments `killed` on confirmed destroy only
- **~** `VastaiRunner._rest_destroy` delegates to `destroy_vastai_instance` with the `allowed_images` ownership guard preserved
- **—** `orchestrator.py:_destroy_via_rest` (simplified copy, latent bug) **deleted**
- **—** `orchestrator.py:poll_instance_progress` (dead public API, no callers anywhere) **deleted**
- **—** `orchestrator.py:ensure_detached` (dead public API, no callers anywhere) **deleted**
- **—** `orchestrator.py:load_vastai_api_key` (byte-for-byte duplicate of `_read_vastai_api_key`) **deleted**; credential loading moves to `providers/destroy_adapters/vastai.py` with env-first precedence and fail-closed semantics
- **—** `providers/vastai.py:_rest_stop`, `_rest_delete_with_retries`, `_rest_verify_and_redestroy` **deleted**; absorbed into the Vast.ai adapter
- **—** `BatchOrchestrator._classify_live_unit` **deleted** (not wrapped); its tests move to `test_unit_lifecycle.py`
- **—** No changes to `runner.py` ABC signature, `storage/r2.py`, `worker/base.py`, `estimator/`

## Module taxonomy

The v3 doc adds two new deep modules around the existing ABCs. The `CloudRunner` ABC is unchanged; the new modules sit at the same layer as the existing provider modules.

### Existing ABCs (unchanged)

`CloudRunner` (Lane A provider lifecycle) and `BatchOrchestrator` (multi-unit orchestration) keep their public surface. The refactor narrows the implementation behind those ABCs.

### New: `unit_lifecycle` — owns the per-cycle decision tree

Owns: the decision rules (R2-first → SSH → worker_dead re-check → action mapping), the `Action` enum, the three plan dataclasses, the `PreemptCause` enum, the `ProgressSnapshot` normaliser.

Does **not** own: side effects (destroy, collect, capture_preempt_diagnostics, on_unit_*), lock acquisition, parallel fan-out, consumer hooks. Those stay on the orchestrator.

### New: `providers/destroy` — owns the belt-and-suspenders destroy protocol

Owns: the loop shape (stop → DELETE×retry → verify → re-destroy, with second verification after resurrection cleanup). The generic loop takes a `DestroyPolicy` for timing/retry constants so the loop itself is policy-agnostic. Returns a typed `DestroyResult`, never a boolean.

Does **not** own: REST URLs, API key paths, image-ownership guard, timing constants. The constants live in the policy (supplied by the adapter); the URLs and credentials live in the adapter.

### New: `providers/destroy_adapters/vastai.py` — Vast.ai REST callbacks

Owns: the three Vast.ai REST endpoints (`PUT state=stopped`, `DELETE`, `GET for verify`), the `read_vastai_api_key` call with env-first precedence and fail-closed semantics, the `allowed_images` ownership guard with `is not None` semantics, the `DestroyPolicy` constants (the Vast.ai-discovered 5s verify delay, 3s retry sleep, 3 max delete attempts), the `destroy_vastai_instance` function that wraps the protocol with pre-protocol refusals.

The RunPod adapter lands separately when `RunPodRunner` ships (roadmap item 2). It will register its own `stop_fn` / `delete_fn` / `verify_fn` callbacks and its own `DestroyPolicy` (different retry timing may be appropriate for RunPod's API).

## Layered design (v3)

```
┌─────────────────────────────────────────────────┐
│  CLI (cli.py)                                   │  User-facing commands
├─────────────────────────────────────────────────┤
│  BatchOrchestrator (batch.py)                   │  Phase loop + side-effect dispatchers
│    └── calls unit_lifecycle.decide_next_action  │  (per-unit decision is delegated)
├─────────────────────────────────────────────────┤
│  unit_lifecycle (unit_lifecycle.py)             │  NEW: decision tree, no side effects
│    └── Action enum + Continue/Complete/Preempt  │
├─────────────────────────────────────────────────┤
│  Orchestrator utils (orchestrator.py)           │  Zombie sweep routes through destroy
├─────────────────────────────────────────────────┤
│  providers/destroy (providers/destroy.py)       │  NEW: belt-and-suspenders protocol
├─────────────────────────────────────────────────┤
│  providers/destroy_adapters/vastai.py           │  NEW: Vast.ai REST callbacks + policy
├─────────────────────────────────────────────────┤
│  CloudRunner (runner.py) ── Lane A ABC          │  Provider-agnostic lifecycle
├──────────────┬──────────────┬───────────────────┤
│ VastaiRunner │ RunPodRunner │ LocalRunner       │  Lane A implementations
├──────────────┴──────────────┴───────────────────┤
│  SSH (ssh.py)      — used by Vast.ai, RunPod    │
│  subprocess        — used by Local              │
├─────────────────────────────────────────────────┤
│  Workers (worker/base.py)                       │  GPU-side execution
├─────────────────────────────────────────────────┤
│  Storage (storage/r2.py)                        │  Result persistence
├─────────────────────────────────────────────────┤
│  State (state.py)                               │  Crash recovery
└─────────────────────────────────────────────────┘
```

## `unit_lifecycle` shape

Single public entry point, single tagged union, three frozen dataclasses, plus a structured preempt cause and a snapshot normaliser.

```python
# src/vastai_gpu_runner/unit_lifecycle.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, ClassVar, Protocol, TypeVar

UnitT = TypeVar("UnitT")
logger = logging.getLogger(__name__)


class Action(StrEnum):
    CONTINUE = "continue"
    COMPLETE = "complete"
    PREEMPT = "preempt"


class PreemptCause(StrEnum):
    """Why the decision module believes the worker died.

    The enum is a stable contract: ``classify_failure`` and other
    consumer-side hooks may eventually branch on the cause. Display
    text belongs in ``Preempt.detail`` (free-form, for humans only).
    """
    WORKER_DIED = "worker_died"  # PID alive check failed, no DONE file
    # Reserved for future use:
    # BOOT_TIMEOUT_RETRY_EXHAUSTED = "boot_timeout_retry_exhausted"
    # SSH_LOST = "ssh_lost"
    # GPU_OOM = "gpu_oom"


@dataclass(frozen=True)
class ProgressSnapshot:
    """Normalised view of the runner's ``check_progress`` response.

    Missing or contradictory fields default to the conservative
    read (both flags clear on contradiction). Missing or
    non-dict envelope fails closed (returns a snapshot with
    all defaults).
    """
    complete: bool = False
    worker_dead: bool = False
    log_tail: str = ""


@dataclass(frozen=True)
class Continue:
    action: ClassVar[Action] = Action.CONTINUE


@dataclass(frozen=True)
class Complete:
    action: ClassVar[Action] = Action.COMPLETE


@dataclass(frozen=True)
class Preempt:
    action: ClassVar[Action] = Action.PREEMPT
    cause: PreemptCause
    detail: str | None = None


UnitAction = Continue | Complete | Preempt


class CloudRunnerLike(Protocol):
    def check_progress(self, instance: object) -> dict[str, object]: ...


def decide_next_action(
    unit: UnitT,
    runner: CloudRunnerLike,
    instance: object,
    is_done_in_r2: Callable[[UnitT], bool],
) -> UnitAction:
    """Classify a live unit and return the next action.

    Performs a localised observation sequence: R2 first, then SSH,
    then R2 again on worker_dead. The two R2 reads are in immediate
    succession inside this function — no orchestrator code can
    interleave between them.

    Exception policy: R2 raises → log + continue to SSH;
    ``check_progress`` raises → Continue; worker_dead re-check raises
    → Continue (don't destroy on unknown final-upload status).
    """
    try:
        if is_done_in_r2(unit):
            return Complete()
    except Exception as exc:
        logger.warning("decide_next_action: initial R2 check raised %s", exc)
    try:
        progress = ProgressSnapshot.from_raw(runner.check_progress(instance))
    except Exception as exc:
        logger.warning("decide_next_action: check_progress raised %s", exc)
        return Continue()
    if progress.complete:
        return Complete()
    if progress.worker_dead:
        try:
            if is_done_in_r2(unit):
                return Complete()
        except Exception as exc:
            logger.warning("decide_next_action: worker_dead R2 re-check raised %s", exc)
            return Continue()
        return Preempt(cause=PreemptCause.WORKER_DIED, detail=progress.log_tail)
    return Continue()
```

See the full implementation at `src/vastai_gpu_runner/unit_lifecycle.py` in the implementation PR (not in this design doc; the design doc is the contract, not the code).

## `providers/destroy` shape

```python
# src/vastai_gpu_runner/providers/destroy.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


class VerifyVerdict(StrEnum):
    GONE = "gone"       # 404 OR 200 + instances.actual_status == "destroyed"
    PRESENT = "present" # 200 + instances.actual_status != "destroyed" (incl. empty, missing, non-string)
    UNKNOWN = "unknown" # other HTTP status, parse failure, network error


class DestroyVerdict(StrEnum):
    DESTROYED = "destroyed"
    LEAKED = "leaked"
    UNKNOWN = "unknown"


class DestroyRefusal(StrEnum):
    OWNERSHIP = "ownership"
    NO_CREDENTIALS = "no_credentials"
    CREDENTIALS_DISABLED = "credentials_disabled"


@dataclass(frozen=True)
class DestroyResult:
    verdict: DestroyVerdict | None = None
    refusal: DestroyRefusal | None = None
    attempts: int = 0
    stop_error: str | None = None
    last_status_code: int | None = None
    verify_error: str | None = None

    def __post_init__(self) -> None:
        if (self.verdict is None) == (self.refusal is None):
            raise ValueError(...)
        if self.refusal is not None:
            # Protocol never ran
            if self.attempts != 0: raise ValueError(...)
            if self.stop_error is not None: raise ValueError(...)
            if self.last_status_code is not None: raise ValueError(...)
            if self.verify_error is not None: raise ValueError(...)
        else:
            # Protocol ran; attempts >= 1
            if self.attempts < 1: raise ValueError(...)


@dataclass(frozen=True)
class DestroyPolicy:
    verify_delay_s: float
    retry_delay_s: float
    max_delete_attempts: int
    verify_after_resurrection: bool = True

    def __post_init__(self) -> None:
        if self.verify_delay_s < 0: raise ValueError(...)
        if self.retry_delay_s < 0: raise ValueError(...)
        if self.max_delete_attempts < 1: raise ValueError(...)
```

`belt_and_suspenders` accepts callbacks and a `DestroyPolicy`, returns a `DestroyResult`. The Vast.ai adapter registers REST callbacks and the Vast.ai-discovered policy.

## Vast.ai adapter shape

The adapter wraps the protocol with pre-protocol refusals:
- `OWNERSHIP` — image allowlist rejected the instance (fail-closed)
- `NO_CREDENTIALS` — no API key configured (CLI fallback permitted)
- `CREDENTIALS_DISABLED` — `VASTAI_API_KEY=""` (CLI fallback forbidden)

The CLI fallback path uses the existing CLI-based `verify_instance_ownership` (separate auth context) before invoking the CLI destroy. Empty allowlist rejects every image; the safety-critical contract.

`read_vastai_api_key()` returns a `CredentialResolution` with three states: `AVAILABLE`, `ABSENT`, `EXPLICITLY_DISABLED`. Strict invariants: `AVAILABLE` requires non-empty pre-stripped key; non-`AVAILABLE` requires empty key.

Image matching uses exact reference OR tag-insensitive repository equality (via `_repository(ref)`). No substring/prefix/registry-port-as-prefix matches. `myorg/app:1.0` does **not** allow `myorg/app-malicious:latest`; `registry:5000/myorg/app:1.0` does **not** allow `registry-malicious/myorg/app:1.0`.

## Zombie sweep integration

`orchestrator.py:sweep_zombie_instances` short-circuits on `EXPLICITLY_DISABLED` (returns 0 before the CLI enumeration step). For other cases it routes through `destroy_vastai_instance`, branches on `verdict` vs `refusal`, and enforces the allowlist on the CLI fallback path. `cli_attempted` is tracked separately from `killed` (killed only on confirmed `DESTROYED`).

## Migration checklist

12 steps; see the [HTML report](https://github.com/Lambda-Biolab/vastai-gpu-runner/pull/18) for the full list. Implementation lands in a follow-up PR.

## Known limitation

`allowed_images` on `BatchOrchestrator` is the immediate fix; the longer-term shape is `ProviderCleanupPolicy`. Tracked in [#19](https://github.com/Lambda-Biolab/vastai-gpu-runner/issues/19).

## Review process

This design went through five ChatGPT-with-GitHub-plugin review passes. The full design doc with the migration checklist, all critical test cases, and the resolved review comments lives in PR #18. The full v3 doc is 1,884 lines and includes the complete code-shape sketches, the per-pass review fixes, the test catalogue, and the glossary.

PR #18 was approved by the reviewer on commit `d24123d`. The doc was then merged into `main` via the standard path (squash, all commits GPG-signed).
