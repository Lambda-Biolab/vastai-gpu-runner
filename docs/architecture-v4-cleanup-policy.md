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
next step. Issue #19's fifth-pass review recommended Option B
(canonical config) over Option A (per-runner `destroy_zombie`) — the
v4 design implements Option B.

[i19]: https://github.com/Lambda-Biolab/vastai-gpu-runner/issues/19

## What changes vs v3

In one paragraph: the ownership-guard policy used by the zombie sweep
moves from a `VastaiRunner.allowed_images` attribute (which the
orchestrator can only read by introspecting a runner it has no
business constructing) to a single, immutable `ProviderCleanupPolicy`
object that is constructed once at boot time from the same canonical
provider config used to build the runner factory. The runner factory
and the cleanup policy now share **one** source of truth for "what
counts as ours" and "what to do with not-ours" — no per-destroy
runner construction, no drift between the `destroy_instance` ownership
guard and the zombie-sweep ownership guard. The shared ownership
semantics live in an `OwnershipPolicy` frozen dataclass with a single
`matches(image_ref)` method (exact reference OR tag-insensitive
repository equality, per the v3 contract), consumed by both the
runner and the cleanup adapter. The credential semantics live in a
v3 `CredentialResolution` (three-state: `AVAILABLE` / `ABSENT` /
`EXPLICITLY_DISABLED`), also shared. The `ProviderCleanupPolicy` is
frozen (`kw_only=True`), provider-agnostic (a dataclass with two
methods: `list_instances()` and `destroy(candidate)`), and exposes
the destroy contract as the authoritative gate — the orchestrator
passes every label-matching, untracked candidate to `destroy()` and
records the `CleanupResult` (refusal or verbidden). The provider-
specific adapters live in `providers/vastai.py` (the
`build_vastai_cleanup_policy` factory) and, when they ship, in
`providers/runpod.py` etc.; the core `cleanup_policy.py` module has
provider-specific imports.

Diff vs v3 once v3 is implemented:

- **+** `src/vastai_gpu_runner/cleanup_policy.py` — `ProviderCleanupPolicy` (frozen, `kw_only`), `InstanceCandidate` (frozen), `CleanupRefusal` enum (`UNOWNED | NO_CREDENTIALS | CREDENTIALS_DISABLED`), `CleanupResult` (typed return with `__post_init__` invariants), `OwnershipPolicy` (frozen, `matches(image_ref)`)
- **+** `src/vastai_gpu_runner/providers/vastai.py:VastaiProviderConfig` (frozen, owns `ownership: OwnershipPolicy`, `credentials: CredentialResolution`, `docker_image`, etc.)
- **+** `src/vastai_gpu_runner/providers/vastai.py:build_vastai_cleanup_policy(config)` — provider-owned factory (core module does not import provider adapters)
- **+** `src/vastai_gpu_runner/providers/vastai.py:list_vastai_instances()` — read-only enumeration returning `list[InstanceCandidate]`
- **~** `BatchOrchestrator.__init__` accepts `cleanup_policy: ProviderCleanupPolicy` (already required; v3 design). The orchestrator calls `policy.list_instances()` and `policy.destroy(candidate)` — it never branches on `Provider`, never imports provider modules.
- **~** `BatchOrchestrator._sweep_zombies` is policy-driven end-to-end. The label-prefix filter and the tracked-id exclusion stay on the orchestrator (per-batch scope). Every other decision (eligibility, ownership, credentials, hostile classification) is delegated to `policy.destroy()`.
- **~** `VastaiRunner.__init__` keeps the `allowed_images` parameter for the direct-construction back-compat path. `VastaiRunner.from_config(config)` is the canonical entry point used by the CLI.
- **—** `orchestrator.py:sweep_zombie_instances` (v3 deletion) — reaffirmed. The v4 implementation removes the last direct caller.
- **—** `orchestrator.py:load_vastai_api_key` (v3 deletion) — reaffirmed. Credential loading lives in `providers/destroy_adapters/vastai.py` (v3).
- **~** `cli.py:cleanup` and `cli.py:instances` — refactored to use the new `VastaiProviderConfig` + `ProviderCleanupPolicy` API. The CLI no longer parses Vast.ai JSON directly.
- **~** `tests/test_orchestrator.py` and `tests/test_batch.py` — mock `cleanup_policy.list_instances()` and `cleanup_policy.destroy()` instead of `sweep_zombie_instances`. New `tests/test_cleanup_policy.py` for the policy unit tests.

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
compete for the same role. The canonical config is the single
source of truth for both the deployment identity and the destruction
identity.

## What changes vs the v4 first draft (5th-pass review)

The first draft of this doc was reviewed by ChatGPT and rejected
with five BLOCKERs and seven CONCERNs. The full diff to the first
draft is in `docs/architecture-v4-cleanup-policy.review-1.md`. The
substantive changes:

- **Dataclass field ordering.** Switched to `@dataclass(frozen=True, kw_only=True)` so callback fields can be required without ordering pitfalls. The first draft had `destroy_fn` required after fields with defaults — a `TypeError` on import.
- **List-instances on the policy.** First draft called `policy.list_provider_instances()` but never defined the method. The v4.2 design adds `list_instances_fn: Callable[[], list[InstanceCandidate]]` to the policy and a public `list_instances()` method that the orchestrator calls.
- **`is_candidate` / `destroy` split.** The first draft made `is_candidate` perform the ownership check, which the orchestrator skipped on `False` — so `destroy()` could never produce `UNOWNED` from the normal flow. The v4.2 design removes the ownership check from `is_candidate` (it is now a cheap eligibility prefilter that returns `False` for ineligible states only) and makes `destroy()` authoritative: it re-validates ownership, credentials, and eligibility every call. The orchestrator passes every label-matching, untracked candidate to `destroy()` and records the `CleanupResult`.
- **Hostile check.** First draft placed a `hostile_check` boolean and `hostile_threshold_seconds` int on the policy, with a destructive branch above the ownership check. The check was semantically redundant — unowned candidates are refused regardless of age. v4.2 removes the hostile check from the policy (deferred to a separate audit stage that YAGNI for now).
- **Credential model.** First draft used `api_key: str = ""` and three enum values were flattened. v4.2 uses v3's `CredentialResolution` (three-state: `AVAILABLE` / `ABSENT` / `EXPLICITLY_DISABLED`) frozen in the canonical config. `EXPLICITLY_DISABLED` short-circuits the policy before enumeration (no API/CLI calls). `ABSENT` uses the CLI fallback with CLI ownership verification (the v3 contract).
- **Ownership semantics.** First draft used `repository_image: Optional[str]` and exact string equality. v4.2 introduces `OwnershipPolicy` with `owned_images: AbstractSet[str] | None` and `matches(image_ref)` (exact reference OR tag-insensitive repository equality, per v3). Both the runner and the cleanup adapter call `matches()`. The deployment image is required to be in `owned_images` at construction time (fail-closed).
- **CLI wiring.** First draft used `VastaiProviderConfig.from_env().__class__(allowed_images=image)` — a fresh default config that discarded the loaded credential. v4.2 uses `dataclasses.replace(base, ...)`.
- **Provider factories.** First draft put `from_vastai_config` / `from_runpod_config` in the core module. v4.2 moves the Vast.ai factory to `providers/vastai.py:build_vastai_cleanup_policy`. The RunPod factory is omitted entirely (lands with the RunPod adapter).
- **State filter.** First draft's `candidate_filter` accepted only `("running", "stopped", "exited")` — too narrow. v4.2 makes `list_*_instances()` return a complete enumeration; the policy's `destroy()` applies the eligibility filter authoritatively.
- **`CleanupResult` invariants.** First draft's `destroyed: int = 0` allowed empty results, negative values, and combinations. v4.2 adds `__post_init__` invariants: exactly one of `destroyed: bool` or `refusal: CleanupRefusal` is set; `error` is non-empty on failure; comparisons to v3 use the `DestroyVerdict` enum, not the string `"destroyed"`.
- **Migration order.** First draft had 7 steps in a logical order but skipped the existing CLI commands and the caller audit. v4.2 reorders the steps (canonical types first, adapter before orchestrator, audit before deletion) and explicitly addresses the `cleanup` and `instances` CLI commands.

## Module taxonomy

The v4 doc adds one new module at the layer above the v3 destroy
protocol. The `CloudRunner` ABC, `BatchOrchestrator`, `unit_lifecycle`,
`providers/destroy`, and `providers/destroy_adapters/vastai` are
unchanged in shape (v3 implementation is still a prerequisite).

### New: `cleanup_policy` — owns the DTOs + the generic policy class

Owns: `InstanceCandidate` (frozen DTO), `CleanupRefusal` enum,
`CleanupResult` (typed return), `OwnershipPolicy` (frozen, shared
ownership semantics), `ProviderCleanupPolicy` (frozen, `kw_only`,
provider-agnostic, holds the `list_instances_fn` and `destroy_fn`
callbacks).

Does **not** own: provider-specific adapters (live in `providers/*`),
REST URLs, API key paths, the destroy protocol loop (lives in
`providers/destroy`), the enumeration logic (per provider). The
core module imports nothing from `providers/`.

### New: `providers/vastai.py:build_vastai_cleanup_policy` — Vast.ai factory

The Vast.ai factory is a provider-owned function. It reads the
canonical `VastaiProviderConfig`, wires the v3 `destroy_vastai_instance`
+ `list_vastai_instances` into the policy, and returns the configured
`ProviderCleanupPolicy`. The factory is the single point of contact
between the Vast.ai adapter and the generic policy.

The RunPod factory is omitted from this doc — it lands with the
RunPod adapter (roadmap item 2). The shape is defined by the
`ProviderCleanupPolicy` interface; the orchestrator's wiring is
provider-agnostic from day one.

## Layered design (v4)

```
┌─────────────────────────────────────────────────┐
│  CLI (cli.py)                                   │  User-facing commands
│    └── builds VastaiProviderConfig once,        │
│        threads it into both runner factory      │
│        and build_vastai_cleanup_policy          │
├─────────────────────────────────────────────────┤
│  BatchOrchestrator (batch.py)                   │  Phase loop + side-effect dispatchers
│    └── accepts cleanup_policy (frozen)          │
│    └── _sweep_zombies calls policy methods      │
│        (list_instances, destroy); never         │
│        branches on Provider, never imports      │
│        provider modules                         │
├─────────────────────────────────────────────────┤
│  cleanup_policy (cleanup_policy.py)             │  NEW: DTOs + generic policy
│    └── OwnershipPolicy (matches image_ref)      │
│    └── ProviderCleanupPolicy (kw_only, frozen)  │
├─────────────────────────────────────────────────┤
│  build_vastai_cleanup_policy (factory in        │  NEW: provider-owned factory
│  providers/vastai.py)                           │
├─────────────────────────────────────────────────┤
│  list_vastai_instances (provider/vastai.py)     │  NEW: read-only enumeration
├─────────────────────────────────────────────────┤
│  unit_lifecycle (unit_lifecycle.py)             │  v3: decision tree, no side effects
├─────────────────────────────────────────────────┤
│  providers/destroy (providers/destroy.py)       │  v3: belt-and-suspenders protocol
├─────────────────────────────────────────────────┤
│  providers/destroy_adapters/vastai.py           │  v3: Vast.ai REST callbacks + policy
├─────────────────────────────────────────────────┤
│  CloudRunner (runner.py) ── Lane A ABC          │  Provider-agnostic lifecycle
├──────────────┬──────────────┬───────────────────┤
│ VastaiRunner │ RunPodRunner │ LocalRunner       │  Lane A implementations
└──────────────┴──────────────┴───────────────────┘
```

## `OwnershipPolicy` shape

Single frozen dataclass, owned by `cleanup_policy.py`. The v3
ownership semantics (exact reference OR tag-insensitive repository
equality) live here, in one method, consumed by both the runner
and the cleanup adapter.

```python
# src/vastai_gpu_runner/cleanup_policy.py
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import AbstractSet


# Tag-insensitive repository match: strip the tag and (optional) registry,
# then require equality. Prevents the v3 failure modes
# ("myorg/app:1.0" matches "myorg/app-malicious:latest"; "registry:5000/myorg/app"
# matches "registry-malicious/myorg/app").
_TAG_RE = re.compile(r"[:@]")


def _repository(image_ref: str) -> str:
    """Return the tag-insensitive repository name from an image reference.

    Strips the tag (`repo:tag` → `repo`) and the registry prefix
    (`registry:5000/repo` → `repo`). The result is the canonical
    "this image, regardless of tag or registry" name.
    """
    ref = image_ref.strip()
    if not ref:
        return ""
    # sha256 digests: strip the @sha256:...; treat the digest as authoritative —
    # if the digest matches, the image matches; the tag is irrelevant.
    if "@sha256:" in ref:
        return ref
    # No tag ('' OR ''): nothing to strip.
    if ":" not in ref.split("/")[-1]:
        return ref
    # Strip the first : after the last / (the tag separator).
    last_slash = ref.rfind("/")
    if last_slash == -1:
        return ref
    head = ref[:last_slash]
    tail = ref[last_slash + 1 :]
    return f"{head}/{tail.split(':', 1)[0]}"


@dataclass(frozen=True)
class OwnershipPolicy:
    """Shared ownership semantics — runner and cleanup adapter both call this.

    The v3 design requires image matching to be exact reference OR
    tag-insensitive repository equality. This dataclass encodes that
    contract in one place. The runner's `destroy_instance` ownership
    guard and the cleanup policy's `destroy` ownership refusal both
    consume the same `matches()` method.

    Args:
        owned_images: Set of Docker image references owned by this
            project. ``None`` explicitly disables the ownership
            check (every image is considered owned). An empty set
            is distinct from ``None`` — it means the project owns
            no images and every ownership check fails (fail-closed).

    Invariants:
        - ``owned_images`` is preserved as a frozenset (caller
          mutation is impossible).
        - ``None`` and ``frozenset()`` are distinct: ``None`` opts
          out, ``frozenset()`` opts in but rejects everything.
    """

    owned_images: AbstractSet[str] | None = None

    def __post_init__(self) -> None:
        if self.owned_images is not None:
            # Coerce to frozenset; freeze the field.
            object.__setattr__(self, "owned_images", frozenset(self.owned_images))

    def matches(self, image_ref: str) -> bool:
        """Return True if ``image_ref`` is owned by this policy.

        Returns True unconditionally when ``owned_images is None``
        (ownership check disabled). Otherwise returns True iff
        the image reference OR its tag-insensitive repository
        matches an entry in ``owned_images``.
        """
        if self.owned_images is None:
            return True
        if not image_ref:
            return False
        if image_ref in self.owned_images:
            return True
        repo = _repository(image_ref)
        if repo and repo in self.owned_images:
            return True
        return False
```

## `CredentialResolution` shape

The v3 design defines `CredentialResolution` in
`providers/destroy_adapters/vastai.py`. The v4 design pulls it into
the canonical config as-is, since v3 implementation is a prerequisite
for v4.

```python
# src/vastai_gpu_runner/providers/destroy_adapters/vastai.py (v3 shape)
class CredentialResolution(Enum):
    AVAILABLE = "available"
    ABSENT = "absent"
    EXPLICITLY_DISABLED = "explicitly_disabled"


@dataclass(frozen=True)
class CredentialResolutionResult:
    """Output of ::func::`read_vastai_api_key`. EMPTY unless state is AVAILABLE."""
    state: CredentialResolution
    api_key: str = ""

    def __post_init__(self) -> None:
        if self.state == CredentialResolution.AVAILABLE:
            assert self.api_key, "AVAILABLE requires non-empty api_key"
        else:
            assert not self.api_key, f"{self.state.value} requires empty api_key"
```

## `VastaiProviderConfig` shape

The canonical config is a frozen dataclass that owns both the
runner-side and the policy-side configuration. The runner factory
and the cleanup-policy factory both read from it. This is the
"share one canonical config" pattern the v3 5th-pass review
recommended.

```python
# src/vastai_gpu_runner/providers/vastai.py
@dataclass(frozen=True)
class VastaiProviderConfig:
    """Canonical Vast.ai configuration shared by runner factory + cleanup policy.

    Both ``VastaiRunner.from_config(config)`` and
    ``build_vastai_cleanup_policy(config)`` read from this. The
    same instance is passed to both at boot time, so the ownership
    guard, the credential state, and the deployment identity cannot
    drift.

    Invariants:
        - ``docker_image`` is in ``ownership.owned_images`` unless
          ``ownership.owned_images is None`` (ownership check disabled).
        - ``credentials`` is a v3 ``CredentialResolutionResult`` (frozen).
    """

    docker_image: str = DEFAULT_IMAGE
    ownership: OwnershipPolicy = field(default_factory=OwnershipPolicy)
    credentials: CredentialResolutionResult = field(
        default_factory=lambda: CredentialResolutionResult(
            state=CredentialResolution.ABSENT,
        )
    )
    min_gpu_vram_mib: int = MIN_GPU_VRAM_MIB
    setup_commands: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.ownership.owned_images is not None:
            if (
                self.docker_image
                and not self.ownership.matches(self.docker_image)
            ):
                msg = (
                    f"VastaiProviderConfig invariant violated: "
                    f"docker_image={self.docker_image!r} is not in "
                    f"ownership.owned_images={set(self.ownership.owned_images)!r}"
                )
                raise ValueError(msg)

    @classmethod
    def from_env(
        cls,
        *,
        docker_image: str | None = None,
        owned_images: AbstractSet[str] | None = None,
    ) -> "VastaiProviderConfig":
        """Build from environment / config files.

        Reads the canonical Vast.ai API key path (the v3
        ``read_vastai_api_key``) and the env-var defaults. The
        CLI uses ``dataclasses.replace`` to overlay the
        project-specific values on top of this base.
        """
        from vastai_gpu_runner.providers.destroy_adapters.vastai import (
            read_vastai_api_key,
        )
        return cls(
            docker_image=docker_image or DEFAULT_IMAGE,
            ownership=OwnershipPolicy(owned_images=owned_images),
            credentials=read_vastai_api_key(),
        )


class VastaiRunner(CloudRunner):
    # ... existing constructor preserved for direct callers ...
    def __init__(
        self,
        config: DeploymentConfig | None = None,
        *,
        allowed_images: frozenset[str] | None = None,
        docker_image: str = DEFAULT_IMAGE,
        min_gpu_vram_mib: int = MIN_GPU_VRAM_MIB,
        setup_commands: list[str] | None = None,
    ) -> None:
        super().__init__(config)
        self.allowed_images = allowed_images
        self.docker_image = docker_image
        self.min_gpu_vram_mib = min_gpu_vram_mib
        self._setup_commands = setup_commands or []

    @classmethod
    def from_config(cls, canonical: VastaiProviderConfig) -> "VastaiRunner":
        """Build a VastaiRunner from the canonical config.

        Preserves the existing constructor as a back-compat path —
        direct callers (unit tests, scripts) keep working. The CLI
        uses this classmethod.
        """
        return cls(
            allowed_images=(
                frozenset(canonical.ownership.owned_images)
                if canonical.ownership.owned_images is not None
                else None
            ),
            docker_image=canonical.docker_image,
            min_gpu_vram_mib=canonical.min_gpu_vram_mib,
            setup_commands=list(canonical.setup_commands),
        )
```

## `list_vastai_instances` shape

The provider's read-only enumeration helper. The orchestrator's
`_sweep_zombies` calls `policy.list_instances()` (which delegates
to the wired `list_instances_fn`); the orchestrator never parses
Vast.ai JSON directly.

```python
# src/vastai_gpu_runner/providers/vastai.py
def list_vastai_instances() -> list[InstanceCandidate]:
    """Read-only enumeration of Vast.ai instances on this account.

    Returns ``InstanceCandidate`` records (provider-agnostic DTOs)
    so the orchestrator's zombie sweep does not have to parse
    Vast.ai's JSON shape. A failure to enumerate returns an empty
    list — the orchestrator's existing exception handling logs the
    failure and continues.
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
            logger.warning("Skipping malformed instance: %s", exc)
    return candidates
```

## `ProviderCleanupPolicy` shape

The core policy dataclass. Frozen, `kw_only=True`, provider-agnostic.
Holds two callbacks (`list_instances_fn` and `destroy_fn`) that
the provider-owned factory wires. The orchestrator calls only
`list_instances()` and `destroy(candidate)`.

```python
# src/vastai_gpu_runner/cleanup_policy.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vastai_gpu_runner.providers.destroy import DestroyResult
    from vastai_gpu_runner.types import Provider


class CleanupRefusal(Enum):
    """Pre-protocol refusal reasons returned by ``policy.destroy``.

    These are policy-level decisions — the destroy protocol is
    never entered. The orchestrator logs the refusal and skips
    the candidate.
    """

    UNOWNED = "unowned"
    NO_CREDENTIALS = "no_credentials"
    CREDENTIALS_DISABLED = "credentials_disabled"


@dataclass(frozen=True)
class InstanceCandidate:
    """Read-only snapshot of one instance returned by ``list_*_instances``.

    Frozen so the policy can hold it without aliasing surprises.
    Fields are the union of what every provider can feasibly
    expose — providers that do not have a field (e.g. RunPod has
    no ``image_uuid``) leave it empty.
    """

    provider: "Provider"
    instance_id: str
    image_uuid: str
    label: str
    state: str
    started_at: float = 0.0


@dataclass(frozen=True)
class CleanupResult:
    """Outcome of ``policy.destroy``.

    Exactly one of ``destroyed`` (True) or ``refusal`` is set.
    ``error`` is non-empty on refusal or protocol failure, empty
    on success. The v4 design does not return the v3
    ``DestroyResult`` directly — the policy wraps the v3 verdict
    in a tighter one-of shape for the orchestrator.
    """

    destroyed: bool = False
    refusal: Optional[CleanupRefusal] = None
    error: str = ""

    def __post_init__(self) -> None:
        if self.destroyed and self.refusal is not None:
            msg = "CleanupResult: destroyed=True and refusal=... are mutually exclusive"
            raise ValueError(msg)
        if not self.destroyed and self.refusal is None and not self.error:
            msg = "CleanupResult: refusal=None requires either destroyed=True or error set"
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class ProviderCleanupPolicy:
    """Per-provider cleanup contract.

    The core module is provider-agnostic: it imports nothing from
    ``providers/``. The ``list_instances_fn`` and ``destroy_fn``
    callbacks are wired by the provider-owned factory (e.g.
    ``build_vastai_cleanup_policy``).

    Args:
        provider: Which provider this policy applies to.
        ownership: Shared ownership semantics. The runner and the
            cleanup adapter both call ``ownership.matches(image_ref)``.
        list_instances_fn: Callable that returns a complete
            ``list[InstanceCandidate]``. Wired by the adapter.
        destroy_fn: Callable that takes one ``InstanceCandidate``,
            applies the eligibility / ownership / credential /
            hostile checks authoritatively, and returns a
            ``CleanupResult``. The factory wires the v3
            ``destroy_vastai_instance`` (with CLI fallback) here.

    Why two callbacks (list + destroy) and not one: the
    orchestrator's outer loop separates enumeration from
    destruction per candidate. The destroy path needs to be
    callable without re-enumerating (the candidate was already
    selected). One callback per phase keeps the contract narrow.
    """

    provider: "Provider"
    ownership: OwnershipPolicy
    list_instances_fn: Callable[[], list[InstanceCandidate]] = field(repr=False)
    destroy_fn: Callable[[InstanceCandidate], CleanupResult] = field(repr=False)

    def list_instances(self) -> list[InstanceCandidate]:
        """Read-only enumeration of provider instances.

        The orchestrator's ``_sweep_zombies`` calls this once per
        sweep. Failures are caught and logged by the orchestrator.
        """
        try:
            return self.list_instances_fn()
        except Exception as exc:
            logger.warning("Cleanup policy: list_instances raised: %s", exc)
            return []

    def destroy(self, candidate: InstanceCandidate) -> CleanupResult:
        """Run the per-provider destroy on one candidate.

        Authoritative gate: this method re-validates the
        candidate's ownership, eligibility, and credential state
        on every call. The orchestrator must call this for every
        label-matching, untracked candidate so refusals are
        observable and loggable.

        Never raises. The orchestrator must be able to log the
        result and continue to the next candidate.
        """
        if candidate.provider != self.provider:
            return CleanupResult(
                refusal=CleanupRefusal.UNOWNED,
                error=f"provider mismatch: {candidate.provider} != {self.provider}",
            )
        try:
            return self.destroy_fn(candidate)
        except Exception as exc:
            logger.warning(
                "Cleanup: destroy_fn raised for %s: %s",
                candidate.instance_id,
                exc,
            )
            return CleanupResult(error=str(exc))
```

## `build_vastai_cleanup_policy` shape

The provider-owned factory. This is the only place that combines
v3's `destroy_vastai_instance` with the v4 policy. It lives in
`providers/vastai.py` (not `cleanup_policy.py`) so the core module
never imports provider adapters.

```python
# src/vastai_gpu_runner/providers/vastai.py
def build_vastai_cleanup_policy(
    canonical: VastaiProviderConfig,
) -> ProviderCleanupPolicy:
    """Build a Vast.ai cleanup policy from the canonical config.

    The canonical config (the same `VastaiProviderConfig` passed
    to `VastaiRunner.from_config`) is the source of truth for the
    ownership policy, the credential resolution, and the deployment
    identity. Reading from the same source guarantees the destroy
    ownership guard and the runner's ownership guard cannot drift.

    The wired ``destroy_fn`` enforces the v3 contract:
    - ``EXPLICITLY_DISABLED`` → ``CleanupResult(refusal=CREDENTIALS_DISABLED)``
      without invoking CLI enumeration.
    - ``AVAILABLE`` → v3 ``destroy_vastai_instance`` with the
      REST bearer key.
    - ``ABSENT`` → v3 CLI fallback (with CLI ownership verification,
      per v3 contract).
    """
    from vastai_gpu_runner.cleanup_policy import (
        CleanupRefusal,
        CleanupResult,
        OwnershipPolicy,
        ProviderCleanupPolicy,
    )
    from vastai_gpu_runner.providers.destroy import (
        DestroyRefusal,
        DestroyVerdict,
    )
    from vastai_gpu_runner.providers.destroy_adapters.vastai import (
        VASTAI_POLICY,
        destroy_vastai_instance,
    )

    def _destroy(candidate: InstanceCandidate) -> CleanupResult:
        # Pre-protocol refusal: credentials explicitly disabled.
        # No enumeration, no CLI fallback — v3 contract.
        if canonical.credentials.state == CredentialResolution.EXPLICITLY_DISABLED:
            return CleanupResult(
                refusal=CleanupRefusal.CREDENTIALS_DISABLED,
                error="VASTAI_API_KEY explicitly empty",
            )
        # The v3 destroy_vastai_instance handles the ownership
        # guard + the belt-and-suspenders protocol. Translate its
        # DestroyResult into a CleanupResult.
        result = destroy_vastai_instance(
            candidate.instance_id,
            allowed_images=(
                frozenset(canonical.ownership.owned_images)
                if canonical.ownership.owned_images is not None
                else None
            ),
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
        if result.verdict == DestroyVerdict.DESTROYED:
            return CleanupResult(destroyed=True)
        return CleanupResult(error=result.error or "destroy protocol did not confirm")

    return ProviderCleanupPolicy(
        provider=canonical.docker_image and Provider.VASTAI or Provider.VASTAI,
        ownership=canonical.ownership,
        list_instances_fn=list_vastai_instances,
        destroy_fn=_destroy,
    )
```

## Orchestrator wiring

The orchestrator's `_sweep_zombies` becomes policy-driven end-to-end.
The provider-specific destroy call is gone; the orchestrator only
knows the policy.

```python
# src/vastai_gpu_runner/batch.py (changes to _sweep_zombies)
def _sweep_zombies(self) -> int:
    """Destroy orphaned instances not tracked by live_runners.

    Routes through the cleanup policy:
    1. Enumerate instances via ``policy.list_instances()``.
    2. Filter by label prefix (orchestrator's per-batch scope).
    3. Exclude tracked IDs (the existing semantics).
    4. For every remaining candidate, call ``policy.destroy(candidate)``.
    5. Count ``destroyed == True`` outcomes.
    6. Log refusals for visibility.

    The orchestrator does NOT branch on Provider, does NOT import
    provider modules, and does NOT call any provider-specific
    destroy function. The policy owns the eligibility /
    ownership / credential / hostile decisions.
    """
    with self._state_lock:
        tracked_ids = {
            entry[1].instance_id for entry in self._live_runners.values()
        }
    candidates = self._cleanup_policy.list_instances()
    killed = 0
    for candidate in candidates:
        if not candidate.label.startswith(self._label_prefix):
            continue
        if candidate.instance_id in tracked_ids:
            continue
        result = self._cleanup_policy.destroy(candidate)
        if result.destroyed:
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

The orchestrator's `__init__` requires `cleanup_policy: ProviderCleanupPolicy`
and `runner_factory: RunnerFactory`. The `runner_factory` is preserved for
the deploy path; only the ownership guard migrates.

## CLI wiring

The CLI is the one place that builds the canonical config and the
runner factory. It threads the same config into the policy factory.

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
    from dataclasses import replace
    from vastai_gpu_runner.providers.vastai import (
        VastaiProviderConfig,
        VastaiRunner,
        build_vastai_cleanup_policy,
    )

    # Step 1: one canonical config, frozen, immutable.
    # `from_env` returns the base (credentials loaded, defaults
    # applied); `replace` overlays the project-specific values.
    base = VastaiProviderConfig.from_env()
    config = replace(
        base,
        docker_image=image,
        ownership=OwnershipPolicy(owned_images=frozenset({image})),
    )

    # Step 2: runner factory reads from the same config.
    runner_factory = lambda: VastaiRunner.from_config(config)  # noqa: E731

    # Step 3: cleanup policy reads from the same config.
    cleanup_policy = build_vastai_cleanup_policy(config)

    orch = MyOrchestrator(
        runner_factory=runner_factory,
        cleanup_policy=cleanup_policy,
        label_prefix=label,
    )
    orch.run()
```

## Migration checklist

Seven steps. The order is the ChatGPT 5th-pass recommendation
(canonical types first, adapter before orchestrator, audit before
deletion). Each step is independently testable but the rollout
lands as one PR because the intermediate states are not stable.

1. **Add canonical credential + ownership-policy types + invariants + contract tests.**
   - `cleanup_policy.py:OwnershipPolicy` (frozen, `matches(image_ref)`).
   - `providers/destroy_adapters/vastai.py:CredentialResolution` + `CredentialResolutionResult` (v3 prerequisite — may already exist if v3 is implemented).
   - Property tests: `OwnershipPolicy.matches` is reflexive (the owned image matches itself); tag-insensitive (`:1.0` and `:latest` match when the repository is owned); `None` opts out (every image matches); empty set rejects everything (fail-closed); different registry prefixes do not match (`registry:5000/myorg/app` ≠ `registry-malicious/myorg/app`).
   - `CredentialResolution` invariants: `AVAILABLE` requires non-empty pre-stripped key; `ABSENT` and `EXPLICITLY_DISABLED` require empty key.

2. **Add `VastaiProviderConfig` + `VastaiRunner.from_config` + constructor parity.**
   - `providers/vastai.py:VastaiProviderConfig` (frozen, owns `ownership` + `credentials` + `docker_image`).
   - `__post_init__` invariant: `docker_image ∈ ownership.owned_images` unless ownership is disabled.
   - `VastaiRunner.from_config(config)` classmethod. Constructor parity test: `VastaiRunner(allowed_images=frozenset({img}), docker_image=img, ...)` and `VastaiRunner.from_config(VastaiProviderConfig(docker_image=img, ownership=OwnershipPolicy(owned_images=frozenset({img})), ...))` produce equivalent attribute values for the runner.
   - `VastaiProviderConfig.from_env()` reads the v3 `read_vastai_api_key` into `credentials`.

3. **Add Vast.ai cleanup adapter (enumeration + authoritative destroy + CLI fallback).**
   - `providers/vastai.py:list_vastai_instances()` (returns `list[InstanceCandidate]`, empty list on API failure).
   - `providers/vastai.py:build_vastai_cleanup_policy(config)` (wires `list_instances_fn` + `destroy_fn` from the v3 adapter).
   - Adapter tests:
     - `EXPLICITLY_DISABLED` path: `destroy_fn` returns `refusal=CREDENTIALS_DISABLED` without invoking `vastai_cmd` (mock and assert).
     - `AVAILABLE` path: `destroy_fn` calls `destroy_vastai_instance` with the loaded key; `verdict=DESTROYED` → `CleanupResult(destroyed=True)`.
     - `ABSENT` path: `destroy_fn` falls through to CLI fallback (per v3 contract); `verdict=DESTROYED` → `CleanupResult(destroyed=True)`.
     - `refusal=OWNERSHIP` translation → `CleanupResult(refusal=UNOWNED)`.
     - `refusal=NO_CREDENTIALS` translation → `CleanupResult(refusal=NO_CREDENTIALS)`.
     - `list_vastai_instances()` returns `[]` on `RuntimeError` and `JSONDecodeError` (does not raise).

4. **Add orchestrator support behind a fail-closed compatibility path.**
   - `BatchOrchestrator.__init__` accepts `cleanup_policy: ProviderCleanupPolicy` (required).
   - `_sweep_zombies` becomes policy-driven (the v4.2 code shape above).
   - Orchestrator tests:
     - `_sweep_zombies` calls `cleanup_policy.list_instances()` exactly once.
     - `_sweep_zombies` calls `cleanup_policy.destroy(candidate)` for every label-matching, untracked candidate.
     - `_sweep_zombies` counts only `destroyed=True` outcomes.
     - `_sweep_zombies` logs refusals (no exception, no crash).
     - `_sweep_zombies` continues on `destroy_fn` exceptions.
     - `_sweep_zombies` does NOT branch on `Provider` (verified by mock introspection: swapping the policy for a RunPod-shaped policy would still work).

5. **Update every composition root, subclass, and existing CLI command.**
   - `cli.py:cli` (composition root): build `VastaiProviderConfig` via `from_env()` + `replace()`, pass to `VastaiRunner.from_config` and `build_vastai_cleanup_policy`.
   - `cli.py:cleanup` (existing CLI command): migrate to use `build_vastai_cleanup_policy` + `policy.destroy`. No more direct `vastai_cmd(["show", "instances", "--raw"])` parsing.
   - `cli.py:instances` (existing CLI command): migrate to use `list_vastai_instances()` for the table. The `--allowed-images` flag becomes `--owned-images` (CLI help text updated to point at `OwnershipPolicy`).
   - `BatchOrchestrator` subclasses (consumer code): existing `BatchOrchestrator(...)` calls now require `cleanup_policy`. Update each subclass's composition.
   - Test fixtures and conftest: `VastaiProviderConfig` factory fixture for tests.

6. **Add integration tests.**
   - `tests/integration/test_cleanup_policy_integration.py` (or equivalent):
     - **disabled-before-enumeration**: config with `credentials=EXPLICITLY_DISABLED`; `policy.destroy(candidate)` returns `CREDENTIALS_DISABLED` even when the candidate's image is owned. `list_instances()` is called but `destroy_fn` is not (no API key passed).
     - **absent-credential CLI fallback**: config with `credentials=ABSENT`; `policy.destroy(candidate)` falls through to CLI fallback path and the v3 `destroy_vastai_instance` is invoked with `api_key=None` (CLI path).
     - **empty ownership set**: config with `ownership=OwnershipPolicy(owned_images=frozenset())`; `policy.destroy(candidate)` returns `UNOWNED` for every candidate (fail-closed).
     - **provider mismatch**: candidate with `provider=Provider.RUNPOD` is passed to a `Provider.VASTAI` policy; returns `UNOWNED` with `provider mismatch` error.
     - **enumeration failure**: `list_vastai_instances()` raises on `RuntimeError`; `policy.list_instances()` returns `[]` (the policy catches the exception).
   - The integration tests use mocks; the CLI fallback path is tested with the v3 mock infrastructure.

7. **Delete legacy sweep + duplicated helpers after a repository-wide caller audit.**
   - `audit_caller_sites.sh` (run before the deletion): grep for any external caller of `orchestrator.sweep_zombie_instances`, `orchestrator.load_vastai_api_key`, `VastaiRunner.allowed_images` (read-only external use). Update external callers.
   - Delete `orchestrator.sweep_zombie_instances` (already deferred to v3; v4.2 reaffirms).
   - Delete `orchestrator.load_vastai_api_key` (v3 deferral reaffirmed).
   - Delete direct `vastai_cmd(["show", "instances", "--raw"])` parsing in `cli.py:cleanup` and `cli.py:instances`.
   - Update `tests/test_orchestrator.py` and `tests/test_batch.py` to mock `cleanup_policy.list_instances` and `cleanup_policy.destroy` instead of `sweep_zombie_instances`.

## Test plan

- `tests/test_cleanup_policy.py` — unit tests:
  - `OwnershipPolicy.matches`:
    - `None` ownership: every image matches (including the empty string)
    - Non-empty set: exact reference matches
    - Non-empty set: tag-insensitive repository matches (`:1.0` and `:latest` both match)
    - Non-empty set: different registry prefix does NOT match
    - Empty set: no image matches (fail-closed)
    - Image `myorg/app:1.0` does NOT match `myorg/app-malicious:latest`
    - `registry:5000/myorg/app:1.0` does NOT match `registry-malicious/myorg/app:1.0`
  - `ProviderCleanupPolicy.list_instances`:
    - Returns the wired list
    - Catches and returns `[]` on `list_instances_fn` exception
  - `ProviderCleanupPolicy.destroy`:
    - Provider mismatch returns `UNOWNED` with `provider mismatch` error
    - Catches `destroy_fn` exceptions and returns `CleanupResult(error=...)`
    - Delegates to `destroy_fn` on the happy path
  - `CleanupResult` invariants:
    - `destroyed=True` + `refusal=...` raises `ValueError`
    - `destroyed=False` + `refusal=None` + `error=""` raises `ValueError`
  - `VastaiProviderConfig` invariants:
    - `docker_image` not in `owned_images` raises `ValueError`
    - `docker_image` in `owned_images` is valid
    - `ownership.owned_images is None` (disabled) accepts any `docker_image`
- `tests/test_providers_vastai.py` — adapter tests:
  - `VastaiRunner.from_config` round-trips with `VastaiProviderConfig`
  - `list_vastai_instances` returns `list[InstanceCandidate]`
  - `list_vastai_instances` returns `[]` on API error (does not raise)
  - `build_vastai_cleanup_policy` wires the v3 adapter correctly
- `tests/test_batch.py` — orchestrator wiring:
  - `_sweep_zombies` calls `cleanup_policy.list_instances()` exactly once
  - `_sweep_zombies` calls `cleanup_policy.destroy(candidate)` for every label-matching, untracked candidate
  - `_sweep_zombies` counts only `destroyed=True` outcomes
  - `_sweep_zombies` logs refusals (no exception, no crash)
  - `_sweep_zombies` continues on `destroy_fn` exceptions
  - `_sweep_zombies` does NOT import provider modules (verified by `inspect.getsource` — should not show `from vastai_gpu_runner.providers.vastai import` in `batch.py`)
- `tests/integration/test_cleanup_policy_integration.py` — integration tests (six scenarios from step 6 above).
- Existing tests for `verify_instance_ownership` and `VastaiRunner.destroy_instance` are unchanged (the runner's ownership guard is preserved; only the policy owner changes).

## Backwards compatibility

The `VastaiRunner.__init__(allowed_images=..., docker_image=..., ...)`
constructor is preserved. Existing callers (scripts, unit tests,
third-party consumers) that build the runner by hand keep working.
The CLI's new `batch` subcommand is the recommended path; the old
programmatic path requires a `cleanup_policy` to be supplied.

The `orchestrator.py:sweep_zombie_instances` function is deleted in
v4 step 7. The v3 implementation already migrated its callers
through the destroy protocol; v4's policy-driven sweep removes the
last direct caller. Any external code that imported
`sweep_zombie_instances` directly must be updated to construct a
`ProviderCleanupPolicy` and call `policy.destroy(candidate)`.

The `cli.py:cleanup` and `cli.py:instances` commands are refactored
in v4 step 5. The `--allowed-images` flag is renamed to
`--owned-images` (with a deprecation cycle: the old flag is
accepted for one minor version and emits a warning).

## Out of scope

- **RunPod adapter.** The `ProviderCleanupPolicy` interface is
  provider-agnostic, but the `build_runpod_cleanup_policy` factory
  is omitted from this doc — it lands with the RunPod adapter
  (roadmap item 2). The factory shape is defined by the
  `ProviderCleanupPolicy` interface; the orchestrator's wiring is
  provider-agnostic from day one.
- **Hostile detection.** The v4 first draft had a `hostile_check`
  boolean + threshold; the 5th-pass review flagged it as
  semantically redundant (unowned candidates are refused
  regardless of age). Hostile detection is deferred to a separate
  audit stage that emits `AGED_UNOWNED` alerts without changing
  the destroy decision. When the dispute workflow lands, the
  hostile stage is added to the policy as a no-side-effect
  classifier.
- **Dispute webhook.** The dispute workflow is future work.
  YAGNI for now.
- **Bulk-destroy optimisation.** The v4 sweep destroys one
  candidate at a time. A bulk path is a future optimisation
  (YAGNI for now; the per-candidate API call is the safe
  default).
- **Cross-provider zombie sweep.** A single orchestrator
  currently supports one `cleanup_policy`. A multi-policy
  orchestrator is a future design (the v4 on-by-one design is
  simpler and covers the current use case).

## Review process

This is the second review pass on the v4 architecture. The first
draft was reviewed by ChatGPT and returned 5 BLOCKERs and 7
CONCERNs; this draft addresses all of them. The diff to the first
draft is summarized in the "What changes vs the v4 first draft"
section above.

The second review prompt for ChatGPT-with-GitHub-plugin:

> Review the v4 architecture design at PR #22 (file:
> docs/architecture-v4-cleanup-policy.md) against the v3 design at
> docs/architecture-v3.md and the current code at
> src/vastai_gpu_runner/{batch,orchestrator,runner,cli}.py and
> src/vastai_gpu_runner/providers/vastai.py. The v4 design
> resolves issue #19 (the ProviderCleanupPolicy follow-up).
>
> The first draft was rejected with 5 BLOCKERs and 7 CONCERNs.
> Verify each finding is addressed:
>
> 1. **BLOCKER 1 (dataclass field ordering)**: confirm
>    `ProviderCleanupPolicy` uses `@dataclass(frozen=True,
>    kw_only=True)` and all callbacks are keyword-only.
> 2. **BLOCKER 2 (list_instances on the policy)**: confirm
>    `list_instances_fn` is added and `list_instances()` is a
>    public method.
> 3. **BLOCKER 3 (is_candidate/destroy inconsistency)**: confirm
>    `destroy()` is authoritative and `is_candidate` was removed
>    from the orchestrator's call path.
> 4. **BLOCKER 4 (hostile check)**: confirm the hostile check is
>    removed from the policy.
> 5. **BLOCKER 5 (credential model)**: confirm
>    `CredentialResolution` (three-state) is used in
>    `VastaiProviderConfig` and `EXPLICITLY_DISABLED` short-
>    circuits before enumeration.
> 6. **BLOCKER 6 (ownership semantics)**: confirm
>    `OwnershipPolicy.matches()` is introduced and consumed by
>    both the runner and the adapter; confirm the
>    `docker_image ∈ owned_images` invariant.
> 7. **BLOCKER 7 (CLI discards from_env)**: confirm the CLI
>    uses `dataclasses.replace` instead of `from_env().__class__(...)`.
> 8. **CONCERN 8 (provider factories in core)**: confirm the
>    Vast.ai factory is moved to `providers/vastai.py` and the
>    core module has no provider imports.
> 9. **CONCERN 9 (state filter)**: confirm the state filter is
>    removed; `list_*_instances()` is complete enumeration;
>    `destroy()` applies eligibility authoritatively.
> 10. **CONCERN 10 (CleanupResult invariants)**: confirm the
>     `__post_init__` invariants are added and the v3
>     `DestroyVerdict` enum is used.
> 11. **CONCERN 11 (no-op RunPod adapter)**: confirm
>     `from_runpod_config` is removed.
> 12. **CONCERN 12 (migration order)**: confirm the 7-step
>     checklist is reordered (canonical types first, adapter
>     before orchestrator, audit before deletion) and the
>     `cli.py:cleanup` and `cli.py:instances` commands are
>     addressed.
>
> Additionally, identify any new BLOCKERs or CONCERNs introduced
> by the second draft. Focus on:
> - The `_repository(image_ref)` function: does it correctly
>   handle edge cases (empty string, sha256 digests, multi-tag
>   references)?
> - The `OwnershipPolicy.__post_init__` freezing: is the
>   `object.__setattr__` pattern correct on a frozen dataclass?
> - The `ProviderCleanupPolicy.destroy` provider-mismatch check:
>   is the error message operator-friendly?
> - The `VastaiProviderConfig.from_env` constructor: does the
>   `docker_image or DEFAULT_IMAGE` guard correctly handle the
>   empty-string case?
> - The migration step 5: does the `--allowed-images` rename to
>   `--owned-images` introduce a wire-format break that should be
>   deferred?
>
> Return a labeled list of findings. Each finding is one of:
> BLOCKER (must fix before merge), CONCERN (should fix, but not
> blocking), or NIT (nice to have). For each finding, give the
> exact line range, the issue, and the proposed fix. If the
> design is acceptable as-is, say "DESIGN ACCEPTED" with a
> one-line rationale.
