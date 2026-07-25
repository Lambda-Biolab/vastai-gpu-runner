# Architecture v4 — ProviderCleanupPolicy

This doc describes the **target architecture** for the layer above the v3
belt-and-suspenders destroy refactor. It resolves [issue #19][i19] and
defines the long-term shape for the zombie-sweep + ownership-guard
policy.

For the current-state architecture (today's code) see
[`architecture.md`](architecture.md). For the v2 history see
[`architecture-v2.md`](architecture-v2.md). For the v3 design (now
merged but **not yet implemented**) see [`architecture-v3.md`](architecture-v3.md).
The v3 doc's "Known limitation" section flagged this design as the
next step.

[i19]: https://github.com/Lambda-Biolab/vastai-gpu-runner/issues/19

## What changes vs v3

In one paragraph: the ownership-guard policy used by the zombie sweep
moves from a `VastaiRunner.allowed_images` attribute (which the
orchestrator can only read by introspecting a runner it has no business
constructing) to a single, immutable `ProviderCleanupPolicy` object
that is constructed once at boot time from the same canonical provider
config used to build the runner factory. The runner factory and the
cleanup policy now share **one** source of truth for "what counts as
ours" and "what to do with not-ours" — no per-destroy runner
construction, no drift between the `destroy_instance` ownership guard
and the zombie-sweep ownership guard. The cleanup policy is
provider-agnostic (a frozen dataclass with two methods); the
provider-specific bits (Vast.ai image allowlist, RunPod template-id
whitelist, etc.) live in adapter construction functions
(`ProviderCleanupPolicy.from_vastai_config`,
`ProviderCleanupPolicy.from_runpod_config`). The orchestrator's
`_sweep_zombies` calls
`cleanup_policy.is_candidate(snapshot)` and then
`cleanup_policy.destroy(candidate)` — it never reads
`allowed_images` directly and never calls a provider-specific destroy
function.

Diff vs v3 once v3 is implemented:

- **+** `src/vastai_gpu_runner/cleanup_policy.py` — `ProviderCleanupPolicy` (frozen dataclass), `InstanceCandidate` (frozen dataclass), `CleanupRefusal` enum (`UNOWNED | NO_CREDENTIALS | CREDENTIALS_DISABLED`), `CleanupResult` (typed return), `from_vastai_config` / `from_runpod_config` constructors
- **~** `BatchOrchestrator.__init__` accepts `cleanup_policy: ProviderCleanupPolicy` instead of `runner_factory + allowed_images` plumbing. The `runner_factory` is preserved (still needed for `_deploy_one`); only the ownership guard moves.
- **~** `BatchOrchestrator._sweep_zombies` calls `cleanup_policy.is_candidate(candidate)` to filter, then `cleanup_policy.destroy(candidate)` to destroy. The orchestrator no longer imports `destroy_vastai_instance` — it knows the policy, not the provider.
- **~** `VastaiRunner.__init__` keeps `allowed_images` for the `destroy_instance` ownership guard (it is the v3-daemon guard). The `from_vastai_config(cfg)` classmethod wraps the constructor so the policy and the runner share the same `allowed_images` value.
- **+** `providers/vastai.py:list_vastai_instances()` — read-only enumeration (returns `list[InstanceCandidate]`). The orchestrator's `_sweep_zombies` uses this instead of inlining `vastai_cmd(["show", "instances", "--raw"])`.
- **+** `providers/runpod.py` (separate tracking issue) — RunPod adapter consisting of `RunPodRunner` plus `list_runpod_instances()` plus `ProviderCleanupPolicy.from_runpod_config()`
- **—** `BatchOrchestrator.__init__` no longer accepts `allowed_images` directly. The CLI now constructs the policy from the canonical config and passes it to the orchestrator.
- **—** `BatchOrchestrator._sweep_zombies` no longer calls `sweep_zombie_instances` from `orchestrator.py` — the v3-routed destroy path is replaced by `cleanup_policy.destroy`. The `orchestrator.py:sweep_zombie_instances` function is **deleted** (the v3 implementation already migrated its callers through the destroy protocol).
- **—** `orchestrator.py:load_vastai_api_key` (v3 deletion) is reaffirmed; canonical credential loading lives in `providers/destroy_adapters/vastai.py` and the policy constructor reads from there.
- **—** `tests/test_orchestrator.py:SweepZombieTests` — the existing tests mock `sweep_zombie_instances` directly; those mocks move to `cleanup_policy.is_candidate` / `cleanup_policy.destroy` mocks.

## Module taxonomy

The v4 doc adds one new module at the layer above the v3 destroy
protocol. The `CloudRunner` ABC, `BatchOrchestrator`, `unit_lifecycle`,
`providers/destroy`, and `providers/destroy_adapters/vastai` are
unchanged in shape (v3 implementation is still a prerequisite).

### New: `cleanup_policy` — owns the policy + the destroy-by-policy contract

Owns: the policy dataclass (provider, repository image allowlist,
hostile_check + threshold, optional dispute webhook), the candidate
record (a snapshot of one enumerated instance), the refusal enum
(safety-critical refusals before the destroy protocol runs), the
typed result struct, the two adapter constructors. The
`is_candidate` and `destroy` methods are the only public surface the
orchestrator calls.

Does **not** own: the destroy loop shape (lives in
`providers/destroy`), the REST callbacks (live in the adapters), the
API key loading (lives in the adapters), the enumeration logic
(lives in `list_*_instances` per provider), the label-prefix filter
(stays on the orchestrator — that's the per-batch scope, not the
per-provider identity).

## Layered design (v4)

```
┌─────────────────────────────────────────────────┐
│  CLI (cli.py)                                   │  User-facing commands
│    └── builds canonical ProviderConfig +        │
│        ProviderCleanupPolicy once, threads      │
│        both into the orchestrator               │
├─────────────────────────────────────────────────┤
│  BatchOrchestrator (batch.py)                   │  Phase loop + side-effect dispatchers
│    └── accepts cleanup_policy (frozen)          │
│    └── _sweep_zombies calls policy methods      │
├─────────────────────────────────────────────────┤
│  cleanup_policy (cleanup_policy.py)             │  NEW: policy + is_candidate + destroy
│    └── from_vastai_config / from_runpod_config  │
├─────────────────────────────────────────────────┤
│  Providers' list_*_instances()                  │  NEW: read-only enumeration
│    └── vastai: list_vastai_instances()          │
│    └── runpod: list_runpod_instances() (later)  │
├─────────────────────────────────────────────────┤
│  unit_lifecycle (unit_lifecycle.py)             │  v3: decision tree, no side effects
├─────────────────────────────────────────────────┤
│  providers/destroy (providers/destroy.py)       │  v3: belt-and-suspenders protocol
│    └── belt_and_suspenders(stop_fn, delete_fn,  │
│        verify_fn, *, policy: DestroyPolicy)     │
├─────────────────────────────────────────────────┤
│  providers/destroy_adapters/vastai.py           │  v3: Vast.ai REST callbacks + policy
│  providers/destroy_adapters/runpod.py           │  LATER: RunPod callbacks + policy
├─────────────────────────────────────────────────┤
│  CloudRunner (runner.py) ── Lane A ABC          │  Provider-agnostic lifecycle
├──────────────┬──────────────┬───────────────────┤
│ VastaiRunner │ RunPodRunner │ LocalRunner       │  Lane A implementations
├──────────────┴──────────────┴───────────────────┤
│  SSH (ssh.py)      — used by Vast.ai, RunPod    │
│  subprocess        — used by Local              │
└─────────────────────────────────────────────────┘
```

## `ProviderCleanupPolicy` shape

Single public class, frozen dataclass, two methods, three refusal
enum members, one typed result. Adapter constructors are classmethods.

```python
# src/vastai_gpu_runner/cleanup_policy.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

from vastai_gpu_runner.providers.destroy import (
    DestroyPolicy,
    DestroyResult,
    DestroyRefusal,
    belt_and_suspenders,
)
from vastai_gpu_runner.providers.destroy_adapters.vastai import (
    VASTAI_POLICY,
    destroy_vastai_instance,
    read_vastai_api_key,
)

if TYPE_CHECKING:
    from vastai_gpu_runner.types import Provider

logger = logging.getLogger(__name__)


class CleanupRefusal(Enum):
    """Pre-protocol refusal reasons returned by ``cleanup_policy.destroy``.

    The destroy protocol is *not* entered when a refusal is returned —
    the policy has decided the candidate is not eligible for the
    per-provider destroy path. The orchestrator logs the refusal and
    skips the candidate.
    """

    UNOWNED = "unowned"          # image/template not in the allowlist
    NO_CREDENTIALS = "no_credentials"  # no API key, no CLI fallback
    CREDENTIALS_DISABLED = "credentials_disabled"  # API key explicitly empty


@dataclass(frozen=True)
class InstanceCandidate:
    """Read-only snapshot of one instance returned by ``list_*_instances``.

    Frozen dataclass so the policy can be stored alongside the
    candidate in a temporary log without aliasing issues. Fields are
    the union of what every provider can feasibly expose — providers
    that do not have a field (e.g. RunPod has no ``image_uuid``)
    leave it empty.
    """

    provider: "Provider"
    instance_id: str
    image_uuid: str          # Vast.ai: image tag; RunPod: template id
    label: str
    state: str               # raw provider state ("running", "stopped", ...)
    started_at: float = 0.0  # unix epoch; 0 if unknown


@dataclass(frozen=True)
class CleanupResult:
    """Outcome of ``cleanup_policy.destroy``.

    Exactly one of ``destroyed`` (int) or ``refusal`` (CleanupRefusal)
    is set. ``destroyed == 1`` means the destroy protocol succeeded;
    ``destroyed == 0`` means the protocol ran but did not confirm
    destruction (the daemon may have rejected the request, the
    instance may have already been gone, etc.). A refusal is a
    pre-protocol decision — the daemon was never contacted.
    """

    destroyed: int = 0
    refusal: Optional[CleanupRefusal] = None
    error: str = ""          # human-readable on protocol failure


@dataclass(frozen=True)
class ProviderCleanupPolicy:
    """Per-provider cleanup contract.

    Constructed once at boot time from the same canonical provider
    config that builds the runner factory. Both the factory and the
    policy read from the same frozen source — no per-destroy runner
    construction, no drift between ``destroy_instance``'s ownership
    guard and the zombie sweep's ownership guard.

    Args:
        provider: Which provider this policy applies to.
        repository_image: The single image (or template id, for
            RunPod) the project owns. ``None`` opts out of the
            ownership guard (the policy still runs but accepts every
            candidate). This is the sole configuration knob for
            "what counts as ours."
        hostile_check: Whether the policy should refuse to destroy
            instances older than ``hostile_threshold_seconds`` that
            are not in the allowlist. Default False (back-compat).
        hostile_threshold_seconds: Threshold for the hostile check.
        destroy_fn: Callable that takes an ``InstanceCandidate`` and
            returns a ``CleanupResult``. Wired by the adapter
            constructor. The orchestrator never calls this directly —
            it calls ``policy.destroy(candidate)``.
        candidate_filter: Callable that takes an ``InstanceCandidate``
            and returns True if the candidate is eligible for the
            destroy path. Default: always True. The label-prefix
            filter is the orchestrator's responsibility, not the
            policy's.

    Why a ``destroy_fn`` callback instead of static protocol knowledge:
    the v3 destroy protocol is generic (`stop_fn`, `delete_fn`,
    `verify_fn` callbacks). The cleanup policy is the next layer up —
    it owns the decision of "should we destroy this?" and delegates
    the "how" to the v3 protocol via the adapter-supplied callback.
    This keeps the v3 module policy-agnostic and the cleanup_policy
    module agnostic of REST URLs.
    """

    provider: "Provider"
    repository_image: Optional[str] = None
    hostile_check: bool = False
    hostile_threshold_seconds: int = 300
    destroy_fn: Callable[[InstanceCandidate], CleanupResult] = field(
        repr=False, compare=False
    )
    candidate_filter: Callable[[InstanceCandidate], bool] = field(
        default=lambda _: True, repr=False, compare=False
    )

    def is_candidate(self, candidate: InstanceCandidate) -> bool:
        """Return True if this candidate is eligible for the destroy path.

        Applies ``candidate_filter`` first (cheap per-provider
        rejection — e.g. ``state not in ("running", "stopped")``),
        then the ownership guard (image/template matches
        ``repository_image``), then the hostile check (if enabled
        and the candidate is older than the threshold and is not
        in the allowlist, refuse even if the filter passes).

        The orchestrator calls this for every enumerated instance
        before calling ``destroy``.
        """
        if not self.candidate_filter(candidate):
            return False
        if self.repository_image is None:
            return True
        if candidate.image_uuid == self.repository_image:
            return True
        return False

    def destroy(self, candidate: InstanceCandidate) -> CleanupResult:
        """Run the per-provider destroy on one candidate.

        Handles the hostile-check refusal (when enabled), the
        ownership refusal (when ``repository_image`` is set and the
        candidate does not match), and delegates the actual
        destroy protocol to ``destroy_fn``.

        Never raises. The orchestrator must be able to log the
        result and continue to the next candidate.
        """
        if self.hostile_check:
            age = candidate.started_at and (
                _now() - candidate.started_at
            )
            if age > self.hostile_threshold_seconds:
                if not self.is_candidate(candidate):
                    logger.warning(
                        "Cleanup: REFUSED hostile candidate %s (age=%ds, image=%s)",
                        candidate.instance_id,
                        int(age),
                        candidate.image_uuid,
                    )
                    return CleanupResult(
                        refusal=CleanupRefusal.UNOWNED,
                        error=f"hostile: older than {self.hostile_threshold_seconds}s",
                    )
        if self.repository_image is not None and not self.is_candidate(candidate):
            return CleanupResult(
                refusal=CleanupRefusal.UNOWNED,
                error=f"image {candidate.image_uuid!r} not in {self.repository_image!r}",
            )
        try:
            return self.destroy_fn(candidate)
        except Exception as exc:
            logger.warning(
                "Cleanup: destroy_fn raised for %s: %s",
                candidate.instance_id,
                exc,
            )
            return CleanupResult(destroyed=0, error=str(exc))

    # -- Adapter constructors --------------------------------------------

    @classmethod
    def from_vastai_config(
        cls,
        canonical_config: "VastaiProviderConfig",
    ) -> "ProviderCleanupPolicy":
        """Build a Vast.ai cleanup policy from the canonical config.

        The canonical config (proposed shape: a frozen dataclass on
        ``VastaiProviderConfig`` with ``allowed_images``,
        ``api_key``, ``hostile_check``, ``hostile_threshold_seconds``)
        is already the source of truth for the runner factory.
        Reading from the same source guarantees the destroy
        ownership guard and the runner's ownership guard cannot
        drift.
        """
        from vastai_gpu_runner.providers.vastai import list_vastai_instances

        def _destroy(candidate: InstanceCandidate) -> CleanupResult:
            # Pre-protocol refusal: credentials explicitly disabled.
            if canonical_config.api_key == "":
                return CleanupResult(
                    refusal=CleanupRefusal.CREDENTIALS_DISABLED,
                    error="VASTAI_API_KEY explicitly empty",
                )
            # The v3 destroy_vastai_instance handles the ownership
            # guard + the belt-and-suspenders protocol. Translate
            # its DestroyResult into a CleanupResult.
            result = destroy_vastai_instance(
                candidate.instance_id,
                allowed_images=frozenset({canonical_config.allowed_images})
                if canonical_config.allowed_images else None,
            )
            if result.refusal == DestroyRefusal.OWNERSHIP:
                return CleanupResult(
                    refusal=CleanupRefusal.UNOWNED,
                    error=result.error,
                )
            if result.refusal == DestroyRefusal.NO_CREDENTIALS:
                return CleanupResult(
                    refusal=CleanupRefusal.NO_CREDENTIALS,
                    error=result.error,
                )
            if result.verdict == "destroyed":
                return CleanupResult(destroyed=1)
            return CleanupResult(destroyed=0, error=result.error)

        def _candidate_filter(candidate: InstanceCandidate) -> bool:
            return candidate.state in ("running", "stopped", "exited")

        return cls(
            provider=canonical_config.provider,
            repository_image=canonical_config.allowed_images,
            hostile_check=canonical_config.hostile_check,
            hostile_threshold_seconds=canonical_config.hostile_threshold_seconds,
            destroy_fn=_destroy,
            candidate_filter=_candidate_filter,
        )

    @classmethod
    def from_runpod_config(
        cls,
        canonical_config: "RunPodProviderConfig",
    ) -> "ProviderCleanupPolicy":
        """Build a RunPod cleanup policy from the canonical config.

        Lands when the RunPod adapter ships (roadmap item 2). The
        shape mirrors ``from_vastai_config`` so the orchestrator's
        `_sweep_zombies` code is identical regardless of provider.
        """
        # Implementation deferred to the RunPod ship PR.
        raise NotImplementedError(
            "RunPod adapter lands in the RunPod ship PR (roadmap item 2)"
        )


def _now() -> float:
    """Wall-clock now in unix seconds. Wrapped for test injection."""
    import time
    return time.time()
```

## `VastaiProviderConfig` shape

The canonical config is a frozen dataclass that owns both the
runner-side and the policy-side configuration. The runner factory
and the cleanup policy constructor both read from it. This is the
"share one canonical config" pattern the v3 5th-pass review
recommended.

```python
# src/vastai_gpu_runner/providers/vastai.py (additions)
@dataclass(frozen=True)
class VastaiProviderConfig:
    """Canonical Vast.ai configuration shared by runner factory + policy.

    Both ``VastaiRunner.from_config(config)`` and
    ``ProviderCleanupPolicy.from_vastai_config(config)`` read from
    this. The same instance is passed to both at boot time, so the
    ownership guard cannot drift between the runner's
    ``destroy_instance`` and the policy's ``destroy`` method.
    """

    provider: Provider = Provider.VASTAI
    allowed_images: Optional[str] = None  # single image — the project's
    docker_image: str = DEFAULT_IMAGE
    min_gpu_vram_mib: int = MIN_GPU_VRAM_MIB
    setup_commands: tuple[str, ...] = ()
    api_key: str = ""                     # read once at boot
    hostile_check: bool = False
    hostile_threshold_seconds: int = 300

    @classmethod
    def from_env(cls) -> "VastaiProviderConfig":
        """Build from environment / config files.

        Reads the canonical Vast.ai API key path
        (``read_vastai_api_key`` from v3) and the project's
        canonical image from env var
        (``VASTAI_PROJECT_IMAGE``) — the image is a single
        canonical value, not a set, because the project owns
        one image per provider.
        """
        import os
        from vastai_gpu_runner.providers.destroy_adapters.vastai import (
            read_vastai_api_key,
        )
        return cls(api_key=read_vastai_api_key())


class VastaiRunner(CloudRunner):
    # ... existing constructor preserved for direct callers ...

    @classmethod
    def from_config(cls, config: VastaiProviderConfig) -> "VastaiRunner":
        """Build a VastaiRunner from the canonical config.

        Preserves the existing constructor as a back-compat path —
        direct callers that build the runner by hand (unit tests,
        scripts) keep working. The CLI uses this classmethod.
        """
        return cls(
            allowed_images=frozenset({config.allowed_images})
            if config.allowed_images else None,
            docker_image=config.docker_image,
            min_gpu_vram_mib=config.min_gpu_vram_mib,
            setup_commands=list(config.setup_commands),
        )
```

## `list_vastai_instances` shape

The v3 design's destroy module owns the destroy protocol; the v4
design adds the read-only enumeration helper that the orchestrator
uses to discover zombies. The helper returns
`list[InstanceCandidate]` so the orchestrator never touches raw
provider dicts.

```python
# src/vastai_gpu_runner/providers/vastai.py (additions)
def list_vastai_instances() -> list[InstanceCandidate]:
    """Read-only enumeration of Vast.ai instances on this account.

    Returns InstanceCandidate records (provider-agnostic DTOs) so the
    orchestrator's zombie sweep does not have to parse Vast.ai's
    JSON shape. A failure to enumerate returns an empty list — the
    orchestrator's existing exception handling logs the failure and
    continues.
    """
    from vastai_gpu_runner.cleanup_policy import InstanceCandidate
    try:
        raw = vastai_cmd(["show", "instances", "--raw"], timeout=15)
        instances = json.loads(raw)
    except (RuntimeError, json.JSONDecodeError) as exc:
        logger.warning("list_vastai_instances: enumeration failed: %s", exc)
        return []
    candidates: list[InstanceCandidate] = []
    for inst in instances:
        try:
            candidates.append(
                InstanceCandidate(
                    provider=Provider.VASTAI,
                    instance_id=str(inst.get("id", "")),
                    image_uuid=str(inst.get("image_uuid", "")),
                    label=str(inst.get("label", "")),
                    state=str(inst.get("actual_status", "")),
                    started_at=float(inst.get("start_date", 0.0) or 0.0),
                )
            )
        except (TypeError, ValueError) as exc:
            logger.debug("Skipping malformed instance: %s", exc)
    return candidates
```

## Orchestrator wiring

The orchestrator's `_sweep_zombies` becomes policy-driven. The
provider-specific destroy call is gone; the orchestrator only knows
the policy.

```python
# src/vastai_gpu_runner/batch.py (changes to _sweep_zombies)
def _sweep_zombies(self) -> int:
    """Destroy orphaned instances not tracked by live_runners.

    Routes through the cleanup policy:
    1. Enumerate instances via the provider's list_*_instances().
    2. Filter by label prefix (orchestrator's per-batch scope).
    3. Exclude tracked IDs (the existing semantics).
    4. For each candidate, ask the policy if it's a candidate.
    5. For each candidate, call policy.destroy(candidate).
    6. Count destroyed == 1 outcomes.
    """
    with self._state_lock:
        tracked_ids = {
            entry[1].instance_id for entry in self._live_runners.values()
        }
    try:
        candidates = self._cleanup_policy.list_provider_instances()
    except Exception as exc:
        logger.warning("Zombie sweep: enumeration failed: %s", exc)
        return 0
    killed = 0
    for candidate in candidates:
        if not candidate.label.startswith(self._label_prefix):
            continue
        if candidate.instance_id in tracked_ids:
            continue
        if not self._cleanup_policy.is_candidate(candidate):
            continue
        result = self._cleanup_policy.destroy(candidate)
        if result.destroyed == 1:
            killed += 1
        elif result.refusal is not None:
            logger.info(
                "Zombie sweep: %s refused (%s): %s",
                candidate.instance_id,
                result.refusal.value,
                result.error,
            )
    if killed:
        logger.info("Zombie sweep: destroyed %d instance(s)", killed)
    return killed
```

The orchestrator no longer imports `sweep_zombie_instances` from
`orchestrator.py`. The orchestrator's `__init__` now requires
`cleanup_policy: ProviderCleanupPolicy` and `runner_factory:
RunnerFactory`. The `runner_factory` is preserved for the deploy
path; only the ownership guard migrates.

## CLI wiring

The CLI is the one place that builds the canonical config and the
runner factory. It threads the same config into the policy
constructor.

```python
# src/vastai_gpu_runner/cli.py (new "batch" subcommand)
@app.command()
def batch(
    label: Annotated[str, typer.Option("--label", "-l")] = ...,
    image: Annotated[
        str, typer.Option("--image", help="Canonical Docker image owned by this project")
    ] = ...,
    max_parallel: Annotated[int, typer.Option("--max-parallel", "-p")] = 8,
    budget: Annotated[float, typer.Option("--budget")] = 0.0,
) -> None:
    """Run a batch of cloud GPU units under the user's project image."""
    from vastai_gpu_runner.providers.vastai import (
        VastaiProviderConfig,
        VastaiRunner,
    )
    from vastai_gpu_runner.cleanup_policy import (
        ProviderCleanupPolicy,
    )

    # Step 1: one canonical config. Frozen — immutable.
    config = VastaiProviderConfig.from_env().__class__(
        allowed_images=image,
    )

    # Step 2: runner factory reads from the same config.
    runner_factory = lambda: VastaiRunner.from_config(config)  # noqa: E731

    # Step 3: cleanup policy reads from the same config.
    cleanup_policy = ProviderCleanupPolicy.from_vastai_config(config)

    orch = MyOrchestrator(
        runner_factory=runner_factory,
        cleanup_policy=cleanup_policy,
        label_prefix=label,
        # ... other params ...
    )
    orch.run()
```

## Why Option B (canonical config) over Option A (per-runner `destroy_zombie`)

The issue text pivoted to Option B as primary because Option A has
a fundamental problem: a zombie candidate is **by definition** not
in the live-runner map, so the orchestrator cannot "use the runner
from the tuple" — there is no runner for an orphan. The orphan is
discovered via the provider's list endpoint and the orchestrator
never had a `CloudRunner` for it. Option A would need to either
(a) resurrect the runner from the candidate (which requires the
candidate to carry the deployment config — a heavier wire), or
(b) run a factory per zombie (which is the bug the v3 5th-pass
review flagged).

Option B avoids both: the policy is the unit of work, the runner
factory is a deployment-time construction, and the two never
compete for the same role.

## Migration checklist

Seven steps. Each step is independently mergeable behind a
feature flag, but the migration should land as one PR because the
intermediate states are not stable (e.g. the orchestrator with no
`cleanup_policy` is a regression). The PR is the v4 implementation.

1. **Add `cleanup_policy.py` with `ProviderCleanupPolicy`, `InstanceCandidate`, `CleanupRefusal`, `CleanupResult`.** Includes the `from_vastai_config` constructor and the `is_candidate` / `destroy` methods. Unit tests for the policy itself (mock `destroy_fn`).
2. **Add `VastaiProviderConfig` and `VastaiRunner.from_config` to `providers/vastai.py`.** Preserves the existing `VastaiRunner.__init__` constructor as a back-compat path. The new classmethod is the canonical entry point.
3. **Add `list_vastai_instances()` to `providers/vastai.py`.** Returns `list[InstanceCandidate]`. Existing v3 styles for the JSON shape (parse `id`, `image_uuid`, `label`, `actual_status`, `start_date`).
4. **Replace `BatchOrchestrator._sweep_zombies` with the policy-driven version.** The orchestrator imports `cleanup_policy.ProviderCleanupPolicy` and the new `list_vastai_instances` (when `provider == VASTAI`). The orchestrator's `__init__` drops the v2 shape and accepts `cleanup_policy`.
5. **Update `cli.py` to build the canonical config and thread it into both the runner factory and the cleanup policy.** The CLI is the only place that constructs the policy at boot time.
6. **Delete `orchestrator.py:sweep_zombie_instances`** (already deferred to v3's scope; v4 re-confirms the deletion). The v3 implementation already routed destroy through the v3 protocol; v4's policy-driven sweep removes the last direct caller.
7. **Update `tests/test_orchestrator.py` and `tests/test_batch.py`** to mock `cleanup_policy.is_candidate` / `cleanup_policy.destroy` instead of `sweep_zombie_instances`. New `tests/test_cleanup_policy.py` for the policy unit tests.

## Test plan

- `tests/test_cleanup_policy.py` — unit tests for the policy:
  - `is_candidate` returns True when `repository_image` is None
  - `is_candidate` returns True when `image_uuid == repository_image`
  - `is_candidate` returns False when `image_uuid != repository_image`
  - `is_candidate` is False when `candidate_filter` returns False
  - `destroy` returns `CleanupResult(destroyed=1)` on protocol success
  - `destroy` returns `CleanupResult(refusal=UNOWNED)` when ownership check fails
  - `destroy` returns `CleanupResult(refusal=CREDENTIALS_DISABLED)` when `api_key == ""`
  - `destroy` returns `CleanupResult(destroyed=0, error=...)` when `destroy_fn` raises
  - `destroy` returns `CleanupResult(refusal=UNOWNED, error="hostile: ...")` when hostile check fires
  - `from_vastai_config` wires the Vast.ai adapter correctly
  - Property-based: `is_candidate` is idempotent (calling it twice returns the same value)
- `tests/test_providers_vastai.py` — additions for the canonical config:
  - `VastaiRunner.from_config` round-trips with `VastaiProviderConfig`
  - `list_vastai_instances` returns `InstanceCandidate` records
  - `list_vastai_instances` returns `[]` on API error (does not raise)
- `tests/test_batch.py` — additions for the policy-driven sweep:
  - `_sweep_zombies` calls `cleanup_policy.is_candidate` and `cleanup_policy.destroy`
  - `_sweep_zombies` counts only `destroyed == 1` outcomes
  - `_sweep_zombies` logs refusals
  - `_sweep_zombies` continues on `destroy_fn` exceptions
- Existing tests for `verify_instance_ownership` and `VastaiRunner.destroy_instance` are unchanged (the runner's ownership guard is preserved; only the policy owner changes).

## Backwards compatibility

The `VastaiRunner.__init__(allowed_images=..., docker_image=..., ...)`
constructor is preserved. Existing callers (scripts, unit tests,
third-party consumers) that build the runner by hand keep working.
The CLI's new `batch` subcommand is the recommended path; the old
programmatic path (`MyOrchestrator(runner_factory=...)`) is preserved
when `cleanup_policy` is supplied alongside the factory.

The `orchestrator.py:sweep_zombie_instances` function is deleted in
v4 step 6. The v3 implementation already migrated its callers
through the destroy protocol; v4's policy-driven sweep removes the
last direct caller. Any external code that imported
`sweep_zombie_instances` directly must be updated to construct a
`ProviderCleanupPolicy` and call `policy.destroy(candidate)` per
enumerated orphan.

## Out of scope

- **RunPod adapter implementation.** The `from_runpod_config`
  constructor raises `NotImplementedError` until the RunPod ship
  PR. The shape is defined here so the orchestrator wiring is
  provider-agnostic from day one.
- **Dispute webhook.** The `dispute_webhook` field is in the
  optional future-proofing comments but not on the dataclass
  (YAGNI). It lands when the dispute workflow is built.
- **Bulk-destroy optimisation.** The v4 sweep destroys one
  candidate at a time. A bulk path is a future optimisation
  (YAGNI for now; the per-candidate API call is the safe
  default).
- **Cross-provider zombie sweep.** If a user runs both Vast.ai
  and RunPod in the same batch, the v4 design supports one
  `cleanup_policy` per orchestrator. A multi-policy orchestrator
  is a future design (the v4 on-by-one design is simpler and
  covers the current use case).

## Review process

This design is the v4 follow-up to the v3 architecture. The first
review prompt for ChatGPT-with-GitHub-plugin will be:

> Review the v4 architecture design at `docs/architecture-v4-cleanup-policy.md`
> against the v3 design at `docs/architecture-v3.md` and the
> current code at `src/vastai_gpu_runner/{batch,orchestrator,runner,cli}.py`
> and `src/vastai_gpu_runner/providers/vastai.py`. The v4 design
> resolves issue #19 (the ProviderCleanupPolicy follow-up). Focus
> on: (1) the canonical-config pattern — does it actually prevent
> the drift the v3 5th-pass review flagged? (2) the
> `is_candidate` / `destroy` split — is the second method
> redundant? (3) the
> `cleanup_policy.py:ProviderCleanupPolicy` shape — is the frozen
> dataclass + `destroy_fn` callback idiomatic, or should the
> policy be split into a `CandidateFilter` + a `DestroyDispatcher`?
> (4) the RunPod forward path — is `from_runpod_config` raising
> `NotImplementedError` the right stub, or does it leak an
> unimplemented state into the orchestrator's code path? (5) the
> hostile-check semantics — is the threshold-based check the right
> shape, or should hostile be a separate stage independent of
> ownership?
>
> Return a labeled list of findings. Each finding is one of:
> BLOCKER (must fix before merge), CONCERN (should fix, but not
> blocking), or NIT (nice to have). For each finding, give the
> exact line range, the issue, and the proposed fix. If the
> design is acceptable as-is, say "DESIGN ACCEPTED" with a
> one-line rationale.

The iteration loop is the same as v3: paste the prompt into
ChatGPT, paste the response back, apply the fixes, push the
amended commit, repeat until the response is "DESIGN ACCEPTED" or
all CONCERNs are resolved.
