# Architecture v3 (target)

This doc describes the **target architecture** after the unit-lifecycle decision tree and belt-and-suspenders destroy refactor land. For the current-state architecture (today's code) see
[`architecture.md`](architecture.md). For the next-step target see
[`architecture-v2.md`](architecture-v2.md). For the architectural review that
motivated this doc see the [HTML report][review].

[review]: file:///tmp/architecture-review.Jxytgw/vastai-gpu-runner-review.html

## What changes vs v2

In one paragraph: the unit-lifecycle decision tree (R2 → SSH → worker_dead re-check → action) moves out of `BatchOrchestrator` into a new `unit_lifecycle` deep module exposing one function `decide_next_action(unit, runner, instance, is_done_in_r2) -> UnitAction`, where `UnitAction` is a tagged union of three frozen dataclasses (`Continue | Complete | Preempt`). The four-step belt-and-suspenders destroy protocol (stop → DELETE×3 → verify → re-destroy) moves out of `VastaiRunner._rest_destroy` and the standalone helpers in `providers/vastai.py` into a new `providers/destroy` deep module exposing `belt_and_suspenders.destroy(stop_fn, delete_fn, verify_fn, ...)`. Two latent bugs disappear with the move: the orchestrator's zombie-sweep `_destroy_via_rest` (orchestrator.py:173) is a simplified copy that cannot handle resurrected instances — the case the protocol was designed for; and the `BatchOrchestrator._check_unit` dispatch is mixed with state mutations, lock acquisition, and side effects across three methods. Both are corrected by the new module boundaries.

Diff vs v2:

- **+** `src/vastai_gpu_runner/unit_lifecycle.py` — `decide_next_action`, `Action` enum, `Continue | Complete | Preempt` plan types
- **+** `src/vastai_gpu_runner/providers/destroy.py` — `belt_and_suspenders.destroy()` + protocol constants (5s verify delay, 3s retry sleep)
- **+** `src/vastai_gpu_runner/providers/destroy_adapters/vastai.py` — Vast.ai REST callbacks (`stop_fn`, `delete_fn`, `verify_fn`)
- **~** `BatchOrchestrator._check_unit` and `_classify_live_unit` become thin wrappers (v15-style) that delegate to `unit_lifecycle.decide_next_action` and the new dispatch
- **~** `BatchOrchestrator._sweep_zombies` routes through `providers/destroy.belt_and_suspenders` instead of constructing REST URLs inline
- **~** `VastaiRunner._rest_destroy` delegates to the new belt-and-suspenders module with the Vast.ai adapter
- **—** `orchestrator.py:_destroy_via_rest` (simplified copy, latent bug) **deleted**
- **—** `orchestrator.py:poll_instance_progress` (dead public API, no callers anywhere) **deleted**
- **—** `orchestrator.py:ensure_detached` (dead public API, no callers anywhere) **deleted**
- **—** `orchestrator.py:load_vastai_api_key` (byte-for-byte duplicate of `_read_vastai_api_key`) **deleted**; `_read_vastai_api_key` promoted to public `read_vastai_api_key()` in `providers/vastai.py`
- **—** `providers/vastai.py:_rest_stop`, `_rest_delete_with_retries`, `_rest_verify_and_redestroy` **deleted**; absorbed into the Vast.ai adapter
- **—** No changes to `runner.py` ABC signature
- **—** No changes to `state.py`, `storage/r2.py`, `worker/base.py`, `estimator/`

## Module taxonomy

The v3 doc adds two new deep modules around the existing ABCs. The `CloudRunner` ABC is unchanged; the new modules sit at the same layer as the existing provider modules.

### Existing ABCs (unchanged)

`CloudRunner` (Lane A provider lifecycle) and `BatchOrchestrator` (multi-unit orchestration) keep their public surface. The refactor narrows the implementation behind those ABCs.

### New: `unit_lifecycle` — owns the per-cycle decision tree

Owns: the decision rules (R2-first → SSH → worker_dead re-check → action mapping), the `Action` enum, the three plan dataclasses.

Does **not** own: side effects (destroy, collect, capture_preempt_diagnostics, on_unit_*), lock acquisition, parallel fan-out, consumer hooks. Those stay on the orchestrator.

### New: `providers/destroy` — owns the belt-and-suspenders destroy protocol

Owns: the 4-step loop (stop → DELETE×retry → verify → re-destroy), the timing constants (5s verify delay, 3s retry sleep), the retry-counter policy.

Does **not** own: REST URLs, API key paths, image-ownership guard. Those live in the per-provider adapter.

### New: `providers/destroy_adapters/vastai.py` — Vast.ai REST callbacks

Owns: the three Vast.ai REST endpoints (`PUT state=stopped`, `DELETE`, `GET for verify`), the `_read_vastai_api_key` call, the `allowed_images` ownership guard.

The RunPod adapter lands separately when `RunPodRunner` ships (roadmap item 2).

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
│  Orchestrator utils (orchestrator.py)           │  Zombie sweep routes through destroy module
│    └── sweep_zombies (delegates to destroy)     │
├─────────────────────────────────────────────────┤
│  providers/destroy (providers/destroy.py)       │  NEW: belt-and-suspenders protocol
│    └── belt_and_suspenders.destroy()            │
├─────────────────────────────────────────────────┤
│  providers/destroy_adapters/vastai.py           │  NEW: Vast.ai REST callbacks
│  providers/destroy_adapters/runpod.py (future)  │  [roadmap item 2]
├─────────────────────────────────────────────────┤
│  CloudRunner (runner.py) ── Lane A ABC          │  Provider-agnostic lifecycle
├──────────────┬──────────────┬───────────────────┤
│ VastaiRunner │ RunPodRunner │ LocalRunner       │  Lane A implementations
├──────────────┴──────────────┴───────────────────┤
│  SSH (ssh.py)      — used by Vast.ai, RunPod    │
│  subprocess        — used by Local              │
├─────────────────────────────────────────────────┤
│  Workers (worker/base.py)                       │  GPU-side execution
│    └── imports inference/ (Lane C, optional)    │
├─────────────────────────────────────────────────┤
│  Storage (storage/r2.py)                        │  Result persistence
├─────────────────────────────────────────────────┤
│  State (state.py)                               │  Crash recovery
└─────────────────────────────────────────────────┘
```

## `unit_lifecycle` shape

Single public entry point, single tagged union, three frozen dataclasses.

```python
# src/vastai_gpu_runner/unit_lifecycle.py
from enum import StrEnum
from dataclasses import dataclass
from typing import Callable, ClassVar, Protocol, TypeVar

UnitT = TypeVar("UnitT")


class Action(StrEnum):
    CONTINUE = "continue"
    COMPLETE = "complete"
    PREEMPT = "preempt"


@dataclass(frozen=True)
class Continue:
    """The unit is still running. No action this cycle."""
    action: ClassVar[Action] = Action.CONTINUE


@dataclass(frozen=True)
class Complete:
    """The unit has finished (R2 confirms DONE OR SSH reports complete).
    Orchestrator should finalise: collect + destroy."""
    action: ClassVar[Action] = Action.COMPLETE


@dataclass(frozen=True)
class Preempt:
    """The worker died silently without uploading. Orchestrator should
    capture diagnostics, destroy the instance, and invoke the consumer's
    classify_failure hook to decide retry-vs-fatal."""
    action: ClassVar[Action] = Action.PREEMPT
    reason: str  # currently always "worker died silently"; reserved for future


UnitAction = Continue | Complete | Preempt


class CloudRunnerLike(Protocol):
    """Structural type for the runner dependency. The real CloudRunner
    satisfies this; tests pass a tiny fake."""
    def check_progress(self, instance: object) -> dict[str, object]: ...


def decide_next_action(
    unit: UnitT,
    runner: CloudRunnerLike,
    instance: object,
    is_done_in_r2: Callable[[UnitT], bool],
) -> UnitAction:
    """Classify a live unit and return the next action.

    Reads R2 first (R2-first poll protocol), then SSH, then re-reads R2
    on worker_dead (the between-check upload case). The two reads are
    in immediate succession inside this function — no orchestrator code
    can interleave between them. This preserves the v15 single-read
    invariant: the SSH verdict and the R2 verdict cannot disagree about
    a checkpoint that landed between the two reads.

    Exceptions from ``runner.check_progress`` are logged and treated as
    "still running" (transient SSH flakiness — the next poll cycle will
    retry).
    """
    ...
```

Why per-unit, per-cycle (not per-run). v15's `decide()` is called once before the lifecycle starts and the plan is terminal. The vastai-gpu-runner decision is repeated every poll cycle (default 30s) for every live unit, so the plan types are *transient actions* ("what now?"), not *terminal plans* ("what's the verdict?"). Per-unit keeps the function's interface small; the orchestrator iterates and fans out the parallel finalise.

Why the `is_done_in_r2` callable. The current `_classify_live_unit` is a method on `BatchOrchestrator` and calls `self.unit_is_done_in_r2(unit)` — the consumer's hook. To make `decide_next_action` a module-level function (not a method), we pass the hook as a callable. The orchestrator does `self.unit_is_done_in_r2` in the call site. This decouples the decision from the orchestrator without forcing the consumer to implement a method on the new module.

Why the protocol type for `runner`. Injecting the full `CloudRunner` class would couple the decision module to the runner.py hierarchy. A structural protocol with one method (`check_progress`) lets tests pass a tiny stub without dragging in the rest of the ABC.

## `providers/destroy` shape

Single public entry point, three callbacks per provider, protocol constants at module level.

```python
# src/vastai_gpu_runner/providers/destroy.py
from typing import Callable, Protocol
import time

VERIFY_DELAY_S = 5.0      # time between destroy and verify
RETRY_DELAY_S = 3.0       # time between DELETE retries
MAX_DELETE_ATTEMPTS = 3   # DELETE retries before giving up


class StopFn(Protocol):
    """Force-stop the instance (kills stuck Docker pulls)."""
    def __call__(self) -> None: ...


class DeleteFn(Protocol):
    """DELETE the instance. Returns True on 2xx/4xx (already gone)."""
    def __call__(self) -> bool: ...


class VerifyFn(Protocol):
    """Verify the instance is gone. Returns True if resurrected."""
    def __call__(self) -> bool: ...


def belt_and_suspenders(
    *,
    stop_fn: StopFn,
    delete_fn: DeleteFn,
    verify_fn: VerifyFn,
) -> bool:
    """Four-step belt-and-suspenders destroy.

    1. ``stop_fn`` — kills stuck Docker pulls.
    2. ``delete_fn`` retried up to ``MAX_DELETE_ATTEMPTS`` times with
       ``RETRY_DELAY_S`` between attempts.
    3. After ``VERIFY_DELAY_S``, ``verify_fn`` checks if the instance
       resurrected.
    4. If resurrected, run stop + delete again with one retry.

    Returns True on confirmed destroy, False on resurrection (callers
    should treat as leaked instance and surface to user).
    """
    stop_fn()
    for attempt in range(MAX_DELETE_ATTEMPTS):
        if delete_fn():
            break
        if attempt < MAX_DELETE_ATTEMPTS - 1:
            time.sleep(RETRY_DELAY_S)
    time.sleep(VERIFY_DELAY_S)
    if verify_fn():
        # Resurrected — try one more time, no retries.
        stop_fn()
        time.sleep(RETRY_DELAY_S)
        delete_fn()
        return False
    return True
```

Why callback-based, not subclassed. v14's `checkpoint.py` is a single concrete module (no adapters) because the checkpoint format is provider-agnostic. The destroy protocol is provider-agnostic in its *loop* but provider-specific in its *endpoints*. By accepting `stop_fn`, `delete_fn`, `verify_fn` callbacks, the protocol stays portable across Vast.ai, RunPod, and any future adapter without forcing a class hierarchy.

Why module-level constants. The 5s verify delay and 3s retry sleep are production-discovered values (the Vast.ai API occasionally resurrects instances after a single DELETE). They are the same for every provider. Concrete values at module level are easier to audit and override in tests than parameters to the function.

## Vast.ai adapter shape

```python
# src/vastai_gpu_runner/providers/destroy_adapters/vastai.py
from vastai_gpu_runner.providers.vastai import (
    read_vastai_api_key,
    verify_instance_ownership,
)
from vastai_gpu_runner.providers.destroy import belt_and_suspenders
import requests

BASE_URL = "https://console.vast.ai/api/v0/instances"


def vastai_stop(instance_id: str, hdrs: dict[str, str]) -> None:
    requests.put(
        f"{BASE_URL}/{instance_id}/",
        headers={**hdrs, "Content-Type": "application/json"},
        json={"state": "stopped"},
        timeout=10,
    )


def vastai_delete(instance_id: str, hdrs: dict[str, str]) -> bool:
    resp = requests.delete(f"{BASE_URL}/{instance_id}/", headers=hdrs, timeout=15)
    return resp.status_code in (200, 204, 404)


def vastai_verify(instance_id: str, hdrs: dict[str, str]) -> bool:
    """Returns True if the instance is resurrected (still alive)."""
    verify = requests.get(f"{BASE_URL}/{instance_id}/", headers=hdrs, timeout=10)
    if verify.status_code != 200:
        return False
    status = verify.json().get("actual_status", "")
    return status not in ("", "destroyed")


def destroy_vastai_instance(
    instance_id: str,
    *,
    allowed_images: set[str] | None = None,
) -> bool:
    """Belt-and-suspenders destroy with the optional ownership guard.

    The ownership guard is applied first: if the instance is running an
    image not in ``allowed_images``, refuse and return False without
    any API call. This protects against cross-project accidents on
    shared Vast.ai accounts.
    """
    if allowed_images and not verify_instance_ownership(instance_id, allowed_images):
        return False
    api_key = read_vastai_api_key()
    if not api_key:
        return False
    hdrs = {"Authorization": f"Bearer {api_key}"}
    return belt_and_suspenders(
        stop_fn=lambda: vastai_stop(instance_id, hdrs),
        delete_fn=lambda: vastai_delete(instance_id, hdrs),
        verify_fn=lambda: vastai_verify(instance_id, hdrs),
    )
```

The existing `VastaiRunner._rest_destroy` collapses into `destroy_vastai_instance(instance_id, allowed_images=self.allowed_images)`. The `allowed_images` ownership guard is preserved — it lives at the adapter layer, not in the protocol module, because the guard is Vast.ai-specific (the other providers do not currently have an image-allowlist feature).

## Thin-wrapper plan

The v15 design keeps the old multi-call surface (`load_checkpoint` + `is_run_complete` + `load_terminal_payload`) as thin wrappers that delegate to `inspect_checkpoint`. v3 follows the same discipline.

```python
# In BatchOrchestrator, after the refactor:

def _classify_live_unit(self, runner, instance, unit) -> str:
    """Deprecated. Use ``unit_lifecycle.decide_next_action`` directly.

    Returns the legacy string verdict for back-compat with direct callers
    and unit tests that mock the dispatcher at the string level.
    """
    from vastai_gpu_runner.unit_lifecycle import (
        decide_next_action,
        Action,
    )
    action = decide_next_action(unit, runner, instance, self.unit_is_done_in_r2)
    return _action_to_legacy_verdict(action)


def _check_unit(self, runner, instance, unit) -> Literal["completed", "running", "preempted", "failed"]:
    """Deprecated. Now a thin wrapper that calls ``decide_next_action``
    and dispatches to the side-effect handlers.

    Kept for back-compat with the docstring guarantee: "Kept for
    backwards-compat with direct callers (unit tests, consumers that
    prefer synchronous single-unit polling)."
    """
    unit_key = self.unit_key(unit)
    from vastai_gpu_runner.unit_lifecycle import (
        decide_next_action,
        Continue, Complete, Preempt,
    )
    action = decide_next_action(unit, runner, instance, self.unit_is_done_in_r2)
    match action:
        case Continue():
            return "running"
        case Complete():
            return self._finalise_completed(runner, instance, unit, unit_key)
        case Preempt():
            with contextlib.suppress(Exception):
                self.capture_preempt_diagnostics(runner, instance, unit)
            with contextlib.suppress(Exception):
                runner.destroy_instance(instance)
            with self._state_lock:
                self._handle_instance_loss(unit, unit_key, action.reason)
            return "preempted"
```

The match statement is the v15-faithful shape: the runner is a thin dispatcher that matches on the plan type. The deprecation warnings are emitted via `warnings.warn(..., DeprecationWarning, stacklevel=2)` so static analysers and unit tests can flag them.

## ABC changes required

**None.** `CloudRunner` keeps its public interface. The new `unit_lifecycle` module does not subclass the runner; it takes a `runner` parameter via protocol. The new `providers/destroy` module takes callbacks; `VastaiRunner` still calls `destroy_vastai_instance` from its own `destroy_instance` method.

The RunPod adapter does not exist yet. When `RunPodRunner` ships (roadmap item 2), it gets a sibling `providers/destroy_adapters/runpod.py` with the same callback shape. No ABC change.

## Glossary of new terms

These terms are introduced by the v3 refactor. The project had no `CONTEXT.md` before this doc; this section is the canonical glossary.

| Term | Meaning |
|---|---|
| `Action` | StrEnum; one of `CONTINUE`, `COMPLETE`, `PREEMPT`. The action the orchestrator should take based on the decision. |
| `UnitAction` | The tagged union `Continue | Complete | Preempt`. The return type of `decide_next_action`. |
| `Continue` | Plan dataclass; the unit is still running, no action this cycle. |
| `Complete` | Plan dataclass; the unit has finished (R2 or SSH confirms). Orchestrator should finalise. |
| `Preempt` | Plan dataclass; the worker died silently. Orchestrator should capture diagnostics, destroy, and invoke consumer's `classify_failure`. Carries a `reason` field (currently always `"worker died silently"`). |
| `decide_next_action` | The single public function in `unit_lifecycle.py`. Reads R2 then SSH then R2 again (on worker_dead), returns the action. |
| `belt_and_suspenders` | The single public function in `providers/destroy.py`. Four-step destroy loop with stop → DELETE×3 → verify → re-destroy. |
| `StopFn` / `DeleteFn` / `VerifyFn` | The callback protocols for the destroy loop. Each provider registers its own implementations. |
| `destroy_vastai_instance` | The Vast.ai adapter for the belt-and-suspenders protocol. Lives in `providers/destroy_adapters/vastai.py`. |

## Open questions

These are flagged for resolution as the implementation lands. They are **not** blocking the design.

### Context module location

The project's existing ABCs (`CloudRunner`, `BatchOrchestrator`) have no separate `CONTEXT.md`. The new terms above live in this design doc. If the project later wants a single glossary file, this section is the seed.

### Credentials unification

[architecture-v2.md](architecture-v2.md) L156 flags "Credentials unification" as an open question. The v3 refactor simplifies the situation (`load_vastai_api_key` deleted, `_read_vastai_api_key` promoted to `read_vastai_api_key`) but does not solve the multi-provider case. Defer until RunPod lands and we have a second key-loading case.

### `DestroyFn` return value

The current draft has `delete_fn` returning `bool` (success). An alternative is to have `delete_fn` raise on transient failure and return on success, with the retry loop wrapping the call. The bool-returning shape is simpler to test but conflates "200 OK" with "404 already gone" (both are good). The raising shape is more explicit but adds ceremony. Resolved during implementation.

### `preempt` reason field

The `Preempt` dataclass carries a `reason: str` field. Currently it is always `"worker died silently"`. If the orchestrator later distinguishes "GPU OOM crash" from "SSH lost" from "boot timeout retry exhausted", the reason field is the place to record it. For now it is reserved for that future use.

### `unit_lifecycle` test surface

The decision tree is tested with a fake runner (`check_progress` → dict) and a fake `is_done_in_r2` callable. No real SSH, no real R2. The test file is `tests/test_unit_lifecycle.py`. Specific edge cases (exception from `check_progress`, worker_dead + R2 re-check race, empty R2-done state) are enumerated in the implementation PR, not here.

## Migration checklist

For the implementer (one PR, one PR-number):

1. Delete `orchestrator.py:poll_instance_progress` and `orchestrator.py:ensure_detached`. Update `CHANGELOG.md`, `docs/api.md`, `docs/architecture.md` to reflect the deletion.
2. Delete `orchestrator.py:load_vastai_api_key`. Promote `providers/vastai.py:_read_vastai_api_key` to `read_vastai_api_key()`. Update `VastaiRunner._rest_destroy` to use the public name.
3. New file `src/vastai_gpu_runner/providers/destroy.py` (the protocol module).
4. New file `src/vastai_gpu_runner/providers/destroy_adapters/vastai.py` (the Vast.ai adapter).
5. Update `VastaiRunner._rest_destroy` to delegate to `destroy_vastai_instance`. Delete the standalone helpers `_rest_stop`, `_rest_delete_with_retries`, `_rest_verify_and_redestroy` from `providers/vastai.py`.
6. Update `orchestrator.py:_sweep_zombies` to route through `destroy_vastai_instance` (or the new destroy module). Delete `orchestrator.py:_destroy_via_rest`.
7. New file `src/vastai_gpu_runner/unit_lifecycle.py` (the decision module).
8. Update `BatchOrchestrator._poll_cycle_once` and `_check_unit` to use `decide_next_action` + match. Convert `_classify_live_unit` and `_check_unit` to thin wrappers.
9. New tests `tests/test_unit_lifecycle.py` and `tests/test_destroy.py`. Update `tests/test_batch.py` and `tests/test_batch_orchestrator.py` to patch the new locations.
10. Update `CHANGELOG.md` with the v3 changes.

## Review process

This design is published as a PR for review before implementation begins. Reviewers (code owner + ChatGPT-with-GitHub-plugin) should focus on:

- **Seam placement.** Is the `unit_lifecycle` ↔ `BatchOrchestrator` seam at the right boundary? Is the `providers/destroy` ↔ `providers/destroy_adapters/vastai.py` split clean?
- **Tagged-union shape.** Is the 3-type `UnitAction` (Continue | Complete | Preempt) sufficient? Should `Fail` be a distinct type, or does it fold into `Preempt`?
- **Thin-wrapper migration.** Are the deprecation warnings on `_classify_live_unit` and `_check_unit` sufficient for back-compat, or should the wrappers be removed entirely?
- **Latent-bug fix.** Is the `_destroy_via_rest` deletion the right way to fix the resurrected-instance bug, or does the design need a more explicit test for the "verify returned True → re-destroy" path?
- **Biolab-runners precedent.** The biolab-runners v15 PR ([#88](https://github.com/Lambda-Biolab/biolab-runners/pull/88)) and v14 PR ([#82](https://github.com/Lambda-Biolab/biolab-runners/pull/82)) are the precedent. Does this design match v15's discipline (single coherent read, frozen dataclass plans, ClassVar action enum)?

Discussion thread: the PR comments.
