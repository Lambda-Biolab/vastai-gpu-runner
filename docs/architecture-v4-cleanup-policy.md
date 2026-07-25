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
counts as ours" and "what to do with not-ours" — no per-destroy runner
construction, no drift between the `destroy_instance` ownership guard
and the zombie-sweep ownership guard. The shared ownership semantics
live in an `OwnershipPolicy` frozen dataclass with a single
`matches(image_ref)` method (exact reference OR tag-insensitive
repository equality, per the v3 contract), consumed by both the
runner and the cleanup adapter via the v3 destroy adapter (which now
accepts `OwnershipPolicy` directly instead of a raw set). The
credential semantics live in the v3 `CredentialState` enum and
`CredentialResolution` dataclass (three-state: `AVAILABLE` / `ABSENT` /
`EXPLICITLY_DISABLED`), passed verbatim into the v3 adapter (no
internal re-resolution). The `ProviderCleanupPolicy` is frozen
(`kw_only=True`), provider-agnostic (a dataclass with two methods:
`list_instances()` and `destroy(candidate)`), and exposes the destroy
contract as the authoritative gate — the orchestrator passes every
label-matching, untracked candidate to `destroy()` and records the
`CleanupResult` (typed `CleanupVerdict` or `CleanupRefusal`). The
provider-specific adapters live in `providers/vastai.py` (the
`build_vastai_cleanup_policy` factory); the core `cleanup_policy.py`
module has **no** provider-specific imports. The Vast.ai factory
wraps `list_vastai_instances` so the v3 contract's
`EXPLICITLY_DISABLED` short-circuit is honored *before* enumeration.

Diff vs v3 once v3 is implemented:

- **+** `src/vastai_gpu_runner/cleanup_policy.py` — `ProviderCleanupPolicy` (frozen, `kw_only`), `InstanceCandidate` (frozen, with provider-neutral display fields), `CleanupVerdict` enum (`DESTROYED | CLI_ATTEMPTED | LEAKED | UNKNOWN`), `CleanupRefusal` enum (`OWNERSHIP | NO_CREDENTIALS | CREDENTIALS_DISABLED | INELIGIBLE_STATE | PROVIDER_MISMATCH`), `CleanupResult` (typed return with `__post_init__` invariants), `OwnershipPolicy` (frozen, `matches(image_ref)`)
- **+** `src/vastai_gpu_runner/providers/vastai.py:VastaiProviderConfig` (frozen, owns `ownership: OwnershipPolicy`, `credentials: CredentialResolution`, `docker_image`, etc., with `docker_image` non-empty + ownership invariant)
- **+** `src/vastai_gpu_runner/providers/vastai.py:build_vastai_cleanup_policy(config)` — provider-owned factory (core module has no provider imports)
- **+** `src/vastai_gpu_runner/providers/vastai.py:list_vastai_instances()` — read-only enumeration returning `list[InstanceCandidate]`
- **~** `providers/destroy_adapters/vastai.py:destroy_vastai_instance` accepts `ownership: OwnershipPolicy` directly (replaces `allowed_images: frozenset[str]`); accepts `credentials: CredentialResolution | None` (defaults to `read_vastai_api_key()` for back-compat direct callers). The v3 implementation must adopt this signature.
- **~** `BatchOrchestrator.__init__` accepts `cleanup_policy: ProviderCleanupPolicy` (already required; v3 design). The orchestrator calls `policy.list_instances()` and `policy.destroy(candidate)` — it never branches on `Provider`, never imports provider modules.
- **~** `BatchOrchestrator._sweep_zombies` is policy-driven end-to-end. The label-prefix filter and the tracked-id exclusion stay on the orchestrator (per-batch scope). Every other decision (eligibility, ownership, credentials) is delegated to `policy.destroy()`. The orchestrator logs every non-`DESTROYED` outcome (CLI fallbacks, LEAKED, UNKNOWN, refusals).
- **~** `VastaiRunner.__init__` keeps the `allowed_images` parameter for the direct-construction back-compat path (builds an `OwnershipPolicy` from it). `VastaiRunner.from_config(config)` is the canonical entry point used by the CLI and passes the `OwnershipPolicy` instance directly (no conversion to a set).
- **~** `cli.py:cleanup` and `cli.py:instances` — refactored to use the new `VastaiProviderConfig` + `ProviderCleanupPolicy` API. The CLI no longer parses Vast.ai JSON directly. The `--allowed-images` flag is preserved as canonical; `--owned-images` is added as a documented alias.
- **—** `orchestrator.py:sweep_zombie_instances` (v3 deletion) — reaffirmed. The v4 implementation removes the last direct caller.
- **—** `orchestrator.py:load_vastai_api_key` (v3 deletion) — reaffirmed. Credential loading lives in `providers/destroy_adapters/vastai.py` (v3).
- **—** `providers/vastai.py:_image_is_allowed` (raw set check) — replaced by `OwnershipPolicy.matches()`. The semantic drift between runner and adapter is impossible because both call the same method.
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

## What changes vs the v4 first + second drafts (5th + 6th pass reviews)

The first draft was rejected with 5 BLOCKERs and 7 CONCERNs. The
second draft was rejected with 7 BLOCKERs, 2 CONCERNs, and 3 NITs.
This third draft addresses every finding. The substantive changes:

### Applied from the 5th-pass review

- Dataclass field ordering switched to `@dataclass(frozen=True, kw_only=True)` (BLOCKER 1).
- `list_instances_fn` + `list_instances()` method on the policy (BLOCKER 2).
- `destroy()` is authoritative; `is_candidate` was removed from the orchestrator's call path (BLOCKER 3).
- Hostile check removed (BLOCKER 4).
- v3 `CredentialResolution` + `CredentialResolutionResult` used (now corrected — see 6th-pass BLOCKER 3 below).
- `OwnershipPolicy.matches()` introduced (now corrected — see 6th-pass BLOCKER 1 below).
- CLI uses `dataclasses.replace` (BLOCKER 7).
- Provider factories moved to `providers/vastai.py` (CONCERN 8).
- State filter removed; complete enumeration + authoritative eligibility (now corrected — see 6th-pass CONCERN 9 below).
- `CleanupResult` invariants added (now tightened — see 6th-pass CONCERN 8 below).
- `from_runpod_config` removed (CONCERN 11).
- Migration order revised (CONCERN 12).

### Applied from the 6th-pass review (this pass)

- **`_repository(image_ref)` fixed** to match v3 contract: strips digest, strips only the final tag separator (after the last `/`), preserves registry and port, normalises both candidate and owned entries. The previous implementation returned the digest reference unchanged and didn't normalise the owned set. (BLOCKER 1)
- **Runner and adapter consume `OwnershipPolicy.matches()` directly.** The runner's `__init__` keeps the `allowed_images` parameter as a deprecated back-compat alias (it builds an `OwnershipPolicy` from it); `from_config` passes the `OwnershipPolicy` instance unchanged. The v3 `destroy_vastai_instance` adapter is updated to accept `ownership: OwnershipPolicy` (replacing `allowed_images: frozenset[str]`); both the runner and the cleanup factory call `ownership.matches(image_uuid)`. (BLOCKER 2)
- **v3 type names used exactly.** The v3 design has `CredentialState` (enum) + `CredentialResolution` (dataclass with `state` and `key` fields, `ValueError` invariants). The v4 second draft misnamed these. (BLOCKER 3)
- **`EXPLICITLY_DISABLED` short-circuits before enumeration.** The Vast.ai factory wraps `list_vastai_instances` so that an explicit-empty credential skips the CLI enumeration step entirely (the v3 contract). The factory wires the wrapped function into `list_instances_fn`, not the raw helper. The integration test asserts that `vastai_cmd` is not called when credentials are disabled. (BLOCKER 4)
- **CLI fallback and DestroyResult translation corrected.** The `_destroy` callback now (a) refuses `EXPLICITLY_DISABLED` without provider calls, (b) uses the canonical API key for the REST path, (c) runs the v3 CLI fallback for `ABSENT`, (d) returns `CLI_ATTEMPTED` (not `DESTROYED`) after a successful CLI command, (e) translates all four v3 verdicts (`DESTROYED | CLI_ATTEMPTED | LEAKED | UNKNOWN`) and all three v3 refusals (`OWNERSHIP | NO_CREDENTIALS | CREDENTIALS_DISABLED`). `CleanupResult` uses `verdict: CleanupVerdict | None` and `refusal: CleanupRefusal | None` (mutually exclusive), not a boolean. The orchestrator logs every non-`DESTROYED` outcome. (BLOCKER 5)
- **`docker_image` non-empty enforced.** `VastaiProviderConfig.__post_init__` rejects empty or whitespace-only `docker_image`. `from_env()` uses `DEFAULT_IMAGE if docker_image is None else docker_image` — explicit empty string is a `ValueError`. (BLOCKER 6)
- **Undefined names removed.** `cleanup_policy.py` adds `import logging; logger = logging.getLogger(__name__)`. `cli.py` adds `from vastai_gpu_runner.cleanup_policy import OwnershipPolicy`. Unused `_TAG_RE`, `Iterable`, `DestroyResult` (TYPE_CHECKING), and `VASTAI_POLICY` imports are removed. (BLOCKER 7)
- **`CleanupResult` invariants tightened.** Now uses `verdict | refusal` (mutually exclusive) instead of `destroyed: bool`. `DESTROYED` requires empty `error`; every other outcome requires non-empty `error`. (CONCERN 8)
- **Authoritative eligibility implemented.** The factory's `_destroy` checks the candidate's state against an explicit `ELIGIBLE_STATES` set (defined as a module constant in `providers/vastai.py`) and returns `INELIGIBLE_STATE` on failure. The orchestrator passes every candidate to `destroy()`; the factory decides. (CONCERN 9)
- **`InstanceCandidate` enriched for CLI display.** Adds `gpu_model: str = ""`, `cost_per_hour: float = 0.0`, `ownership_key: str = ""` (provider-neutral) alongside the existing `image_uuid: str = ""` (Vast.ai-specific). The Vast.ai factory fills both `image_uuid` and `ownership_key` (same value). Future RunPod factory will fill `ownership_key` only. (CONCERN 10)
- **CLI flag renamed carefully.** `--allowed-images` is preserved as canonical; `--owned-images` is added as a documented alias. The rename is deferred to a separate CLI compatibility change. (CONCERN 11)
- **`provider=Provider.VASTAI` simplified** (NIT 12) and provider-mismatch diagnostic improved (NIT 13).
- **Stale prose cleaned up.** All references to `is_candidate`, "hostile classification", "verbidden", and "the core module has provider-specific imports" are corrected. (NIT 14)

## Module taxonomy

The v4 doc adds one new module at the layer above the v3 destroy
protocol. The `CloudRunner` ABC, `BatchOrchestrator`, `unit_lifecycle`,
`providers/destroy`, and `providers/destroy_adapters/vastai` are
unchanged in shape (v3 implementation is still a prerequisite, with
the v3 adapter signature amended per BLOCKER 2 above).

### New: `cleanup_policy` — owns the DTOs + the generic policy class

Owns: `InstanceCandidate` (frozen DTO), `CleanupVerdict` enum,
`CleanupRefusal` enum, `CleanupResult` (typed return), `OwnershipPolicy`
(frozen, shared ownership semantics), `ProviderCleanupPolicy`
(frozen, `kw_only`, provider-agnostic, holds the `list_instances_fn`
and `destroy_fn` callbacks).

Does **not** own: provider-specific adapters (live in `providers/*`),
REST URLs, API key paths, the destroy protocol loop (lives in
`providers/destroy`), the enumeration logic (per provider). The
core module imports nothing from `providers/`.

### New: `providers/vastai.py:build_vastai_cleanup_policy` — Vast.ai factory

The Vast.ai factory is a provider-owned function. It reads the
canonical `VastaiProviderConfig`, wires the v3 `destroy_vastai_instance`
+ a wrapped `list_vastai_instances` into the policy, and returns the
configured `ProviderCleanupPolicy`. The factory is the single point
of contact between the Vast.ai adapter and the generic policy.

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
│    └── destroy_vastai_instance(ownership=       │
│        OwnershipPolicy, credentials=            │
│        CredentialResolution | None)             │
├─────────────────────────────────────────────────┤
│  CloudRunner (runner.py) ── Lane A ABC          │  Provider-agnostic lifecycle
├──────────────┬──────────────┬───────────────────┤
│ VastaiRunner │ RunPodRunner │ LocalRunner       │  Lane A implementations
└──────────────┴──────────────┴───────────────────┘
```

## `OwnershipPolicy` shape

Single frozen dataclass, owned by `cleanup_policy.py`. The v3
ownership semantics (exact reference OR tag-insensitive repository
equality, preserving registry and port) live here, in one method,
consumed by both the runner and the cleanup adapter.

```python
# src/vastai_gpu_runner/cleanup_policy.py
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import AbstractSet, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vastai_gpu_runner.types import Provider

logger = logging.getLogger(__name__)


def _repository(ref: str) -> str:
    """Return the tag-insensitive repository name from an image reference.

    Strips the digest (`repo@sha256:...` → `repo`), strips only the
    final tag separator (after the last `/`), and preserves the
    registry and port. This prevents the v3 failure modes
    ("myorg/app:1.0" matching "myorg/app-malicious:latest";
    "registry:5000/myorg/app" matching "registry-malicious/myorg/app").

    Returns the empty string for malformed references (multiple
    `@`, multiple `:` in the final segment, whitespace-only,
    multiple-tag references).
    """
    ref = ref.strip()
    if not ref:
        return ""
    if ref.count("@") > 1:
        return ""
    without_digest = ref.split("@", 1)[0]
    if not without_digest:
        return ""
    final_segment = without_digest.rsplit("/", 1)[-1]
    if final_segment.count(":") > 1:
        return ""
    last_slash = without_digest.rfind("/")
    last_colon = without_digest.rfind(":")
    if last_colon > last_slash:
        return without_digest[:last_colon]
    return without_digest


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
        - The normalised owned repositories are precomputed and
          cached so ``matches()`` is O(1) per call.
    """

    owned_images: AbstractSet[str] | None = None

    def __post_init__(self) -> None:
        if self.owned_images is None:
            object.__setattr__(self, "_normalised", None)
            return
        frozen = frozenset(self.owned_images)
        object.__setattr__(self, "owned_images", frozen)
        object.__setattr__(
            self,
            "_normalised",
            frozenset(repo for o in frozen if (repo := _repository(o))),
        )

    def matches(self, image_ref: str) -> bool:
        """Return True if ``image_ref`` is owned by this policy.

        Returns True unconditionally when ``owned_images is None``
        (ownership check disabled). Otherwise returns True iff
        the tag-insensitive repository of ``image_ref`` matches
        the tag-insensitive repository of any entry in
        ``owned_images``.

        Examples (assuming ``owned_images={"myorg/app:1.0"}``):
            >>> matches("myorg/app:1.0")        # True (exact)
            >>> matches("myorg/app:latest")     # True (repo match)
            >>> matches("myorg/app-malicious")  # False (different repo)
            >>> matches("registry:5000/myorg/app:1.0")  # False (different registry)
        """
        if self.owned_images is None:
            return True
        if not image_ref:
            return False
        repo = _repository(image_ref)
        if not repo:
            return False
        return repo in self._normalised  # type: ignore[operator]
```

## `CredentialState` + `CredentialResolution` shape (v3 shapes, used unchanged)

The v3 design defines these types in `providers/destroy_adapters/vastai.py`.
The v4 design uses them unchanged — the names matter because the v3
adapter (when implemented) and the v4 factory (this design) must
agree.

```python
# src/vastai_gpu_runner/providers/destroy_adapters/vastai.py (v3 shape — verbatim)
class CredentialState(Enum):
    """Three-state credential resolution."""
    AVAILABLE = "available"
    ABSENT = "absent"
    EXPLICITLY_DISABLED = "explicitly_disabled"


@dataclass(frozen=True)
class CredentialResolution:
    """Output of read_vastai_api_key(). EMPTY unless state is AVAILABLE.

    Invariants (enforced in __post_init__):
        - AVAILABLE requires non-empty, pre-stripped key.
        - ABSENT and EXPLICITLY_DISABLED require empty key.

    Note: pre-stripped means the key has no leading/trailing
    whitespace. The v3 read_vastai_api_key() strips before checking.
    """
    state: CredentialState
    key: str = ""

    def __post_init__(self) -> None:
        if self.state == CredentialState.AVAILABLE:
            if not self.key or self.key != self.key.strip():
                raise ValueError(
                    "CredentialResolution.AVAILABLE requires non-empty, pre-stripped key"
                )
        else:
            if self.key:
                raise ValueError(
                    f"CredentialResolution.{self.state.value} requires empty key"
                )
```

## `VastaiProviderConfig` shape

The canonical config is a frozen dataclass that owns both the
runner-side and the policy-side configuration. The runner factory
and the cleanup-policy factory both read from it.

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
        - ``docker_image`` is non-empty and pre-stripped.
        - ``docker_image`` is in ``ownership.owned_images`` unless
          ``ownership.owned_images is None`` (ownership check disabled).
        - ``credentials`` is a v3 ``CredentialResolution`` (frozen).
    """

    docker_image: str = DEFAULT_IMAGE
    ownership: OwnershipPolicy = field(default_factory=OwnershipPolicy)
    credentials: CredentialResolution = field(
        default_factory=lambda: CredentialResolution(state=CredentialState.ABSENT)
    )
    min_gpu_vram_mib: int = MIN_GPU_VRAM_MIB
    setup_commands: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.docker_image or self.docker_image != self.docker_image.strip():
            raise ValueError(
                "VastaiProviderConfig.docker_image must be a non-empty, pre-stripped reference"
            )
        if self.ownership.owned_images is not None and not self.ownership.matches(
            self.docker_image
        ):
            raise ValueError(
                f"VastaiProviderConfig invariant violated: docker_image="
                f"{self.docker_image!r} is not owned by "
                f"ownership.owned_images="
                f"{set(self.ownership.owned_images)!r}"
            )

    @classmethod
    def from_env(
        cls,
        *,
        docker_image: str | None = None,
        owned_images: AbstractSet[str] | None = None,
    ) -> "VastaiProviderConfig":
        """Build from environment / config files.

        Reads the v3 ``read_vastai_api_key()`` into ``credentials``.
        The CLI uses ``dataclasses.replace`` to overlay the
        project-specific values on top of this base.

        Note: explicit empty ``docker_image`` is rejected by
        ``__post_init__``; only ``None`` triggers the default.
        """
        from vastai_gpu_runner.providers.destroy_adapters.vastai import (
            read_vastai_api_key,
        )
        resolved_image = DEFAULT_IMAGE if docker_image is None else docker_image
        return cls(
            docker_image=resolved_image,
            ownership=OwnershipPolicy(owned_images=owned_images),
            credentials=read_vastai_api_key(),
        )
```

## Vast.ai adapter signature change (v3 amendment)

The v3 design's `destroy_vastai_instance` accepts
`instance_id` and `allowed_images: frozenset[str]`. The v4 design
extends this signature: `allowed_images` is replaced by
`ownership: OwnershipPolicy`, and `credentials` is added as an
optional parameter (defaults to `read_vastai_api_key()` for
back-compat direct callers).

The v3 implementation must adopt this signature when it lands. The
canonical runner and cleanup-policy construction pass the
`OwnershipPolicy` instance unchanged (no conversion to a set).

```python
# src/vastai_gpu_runner/providers/destroy_adapters/vastai.py (v3 + v4 signature)
def destroy_vastai_instance(
    instance_id: str,
    *,
    ownership: OwnershipPolicy,
    credentials: Optional[CredentialResolution] = None,
) -> DestroyResult:
    """Stop + delete + verify a Vast.ai instance.

    Args:
        instance_id: Vast.ai instance ID.
        ownership: Shared ownership policy. The runner and the
            cleanup adapter both pass the same instance.
        credentials: Pre-resolved credential state. When None
            (the default), falls back to ``read_vastai_api_key()``
            — preserves the v3 back-compat path for direct callers.

    Returns:
        ``DestroyResult`` with either ``verdict`` or ``refusal``
        (never both). The cleanup policy translates this into a
        ``CleanupResult`` with ``CleanupVerdict`` or
        ``CleanupRefusal``.
    """
    resolution = credentials or read_vastai_api_key()
    # ... rest unchanged from v3 design, but using ownership.matches()
    # instead of the raw _image_is_allowed set check ...
```

The v2 `verify_instance_ownership` helper is updated to accept
`OwnershipPolicy` directly:

```python
# src/vastai_gpu_runner/providers/vastai.py (v4 amendment)
def verify_instance_ownership(
    instance_id: str,
    *,
    ownership: OwnershipPolicy,
) -> bool:
    """Check that a Vast.ai instance belongs to the caller before destruction.

    Uses ``ownership.matches(image_uuid)`` for the ownership check.
    """
    if ownership.owned_images is None:
        return True
    try:
        raw = vastai_cmd(["show", "instances", "--raw"], timeout=15)
        instances = json.loads(raw)
    except (RuntimeError, json.JSONDecodeError) as exc:
        logger.warning(
            "Cannot verify ownership of instance %s (API error: %s) — refusing to destroy",
            instance_id,
            exc,
        )
        return False
    inst = _find_instance(instances, instance_id)
    if inst is None:
        logger.info("Instance %s not found in account (already destroyed?)", instance_id)
        return True
    image = str(inst.get("image_uuid", ""))
    if ownership.matches(image):
        return True
    logger.error(
        "BLOCKED: instance %s belongs to another project (image=%s). Will NOT destroy.",
        instance_id,
        image,
    )
    return False
```

## `VastaiRunner` shape

The `__init__` keeps the `allowed_images` parameter as a deprecated
back-compat path (it builds an `OwnershipPolicy` from it). The
`from_config` classmethod is the canonical entry point.

```python
# src/vastai_gpu_runner/providers/vastai.py (VastaiRunner updates)
class VastaiRunner(CloudRunner):
    def __init__(
        self,
        config: DeploymentConfig | None = None,
        *,
        ownership: OwnershipPolicy | None = None,
        allowed_images: frozenset[str] | None = None,  # DEPRECATED: build OwnershipPolicy instead
        docker_image: str = DEFAULT_IMAGE,
        min_gpu_vram_mib: int = MIN_GPU_VRAM_MIB,
        setup_commands: list[str] | None = None,
    ) -> None:
        """Initialize Vast.ai runner with deployment config and safety guards."""
        if ownership is None and allowed_images is not None:
            # Back-compat: build OwnershipPolicy from the deprecated set.
            import warnings

            warnings.warn(
                "VastaiRunner(allowed_images=...) is deprecated; "
                "build an OwnershipPolicy and pass ownership= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            ownership = OwnershipPolicy(owned_images=frozenset(allowed_images))
        super().__init__(config)
        self.ownership = ownership
        # Preserve .allowed_images as a property for back-compat reads.
        self._allowed_images_for_backcompat = (
            frozenset(ownership.owned_images)
            if ownership and ownership.owned_images is not None
            else None
        )
        self.docker_image = docker_image
        self.min_gpu_vram_mib = min_gpu_vram_mib
        self._setup_commands = setup_commands or []

    @property
    def allowed_images(self) -> frozenset[str] | None:
        """Back-compat read access. New code should use ``self.ownership``."""
        return self._allowed_images_for_backcompat

    @classmethod
    def from_config(cls, canonical: VastaiProviderConfig) -> "VastaiRunner":
        """Build a VastaiRunner from the canonical config.

        Passes the canonical ``OwnershipPolicy`` instance unchanged
        — the runner and the cleanup policy will call the same
        ``matches()`` method on the same instance.
        """
        return cls(
            ownership=canonical.ownership,
            docker_image=canonical.docker_image,
            min_gpu_vram_mib=canonical.min_gpu_vram_mib,
            setup_commands=list(canonical.setup_commands),
        )

    def destroy_instance(self, instance: CloudInstance) -> bool:
        """Destroy a Vast.ai instance (with ownership safety guard)."""
        if not verify_instance_ownership(
            instance.instance_id,
            ownership=self.ownership or OwnershipPolicy(),
        ):
            logger.error(
                "REFUSED to destroy instance %s — ownership check failed.",
                instance.instance_id,
            )
            return False
        # ... rest unchanged from v2 ...
```

## `list_vastai_instances` shape

The provider's read-only enumeration helper. The Vast.ai factory
wraps this with the EXPLICITLY_DISABLED short-circuit; the core
helper is unchanged.

```python
# src/vastai_gpu_runner/providers/vastai.py
# Module-level constant: states that are safe for the cleanup sweep to
# act on. Anything not in this set returns CleanupRefusal.INELIGIBLE_STATE
# from the factory's _destroy callback.
VASTAI_ELIGIBLE_STATES: frozenset[str] = frozenset(
    {"running", "stopped", "exited", "loading", "created"}
)


def list_vastai_instances() -> list[InstanceCandidate]:
    """Read-only enumeration of Vast.ai instances on this account.

    Returns ``InstanceCandidate`` records so the orchestrator's
    zombie sweep does not have to parse Vast.ai's JSON shape. A
    failure to enumerate returns an empty list — the orchestrator's
    existing exception handling logs the failure and continues.

    NB: this helper does NOT enforce the EXPLICITLY_DISABLED
    short-circuit. The factory wraps this function with the
    short-circuit so the policy's list_instances() honors the v3
    contract.
    """
    candidates: list[InstanceCandidate] = []
    try:
        raw = vastai_cmd(["show", "instances", "--raw"], timeout=15)
        instances = json.loads(raw)
    except (RuntimeError, json.JSONDecodeError) as exc:
        logger.warning("list_vastai_instances: enumeration failed: %s", exc)
        return candidates
    for inst in instances:
        try:
            image_uuid = str(inst.get("image_uuid", ""))
            candidates.append(
                InstanceCandidate(
                    provider=Provider.VASTAI,
                    instance_id=str(inst.get("id", "")),
                    image_uuid=image_uuid,
                    ownership_key=image_uuid,  # Vast.ai's ownership key is its image
                    gpu_model=str(inst.get("gpu_name", "")),
                    cost_per_hour=float(inst.get("dph_total", 0.0) or 0.0),
                    label=str(inst.get("label", "")),
                    state=str(inst.get("actual_status", "")),
                    started_at=float(inst.get("start_date", 0.0) or 0.0),
                )
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping malformed instance: %s", exc)
    return candidates
```

## `InstanceCandidate`, `CleanupVerdict`, `CleanupRefusal`, `CleanupResult` shapes

```python
# src/vastai_gpu_runner/cleanup_policy.py (additions)


class CleanupVerdict(Enum):
    """Outcome verdicts returned by ``policy.destroy``.

    These are protocol outcomes — the destroy protocol ran to
    completion and reported a verdict. ``DESTROYED`` is the only
    success; the others are observable non-success states that
    the orchestrator logs distinctly.
    """

    DESTROYED = "destroyed"
    CLI_ATTEMPTED = "cli_attempted"  # CLI fallback ran, destruction not confirmed
    LEAKED = "leaked"                # Protocol ran but instance was resurrected
    UNKNOWN = "unknown"              # Protocol did not report a clear outcome


class CleanupRefusal(Enum):
    """Pre-protocol refusal reasons returned by ``policy.destroy``.

    These are policy-level decisions — the destroy protocol is
    never entered. The orchestrator logs the refusal and skips
    the candidate.
    """

    OWNERSHIP = "ownership"               # image/template not owned by this policy
    NO_CREDENTIALS = "no_credentials"     # no API key configured; CLI fallback attempted
    CREDENTIALS_DISABLED = "credentials_disabled"  # VASTAI_API_KEY="" — CLI fallback forbidden
    INELIGIBLE_STATE = "ineligible_state" # candidate.state not in provider's eligible_states
    PROVIDER_MISMATCH = "provider_mismatch"  # candidate.provider != policy.provider


@dataclass(frozen=True)
class InstanceCandidate:
    """Read-only snapshot of one instance returned by ``list_*_instances``.

    Frozen so the policy can hold it without aliasing surprises.
    Fields are the union of what every provider can feasibly
    expose; providers that do not have a field (e.g. RunPod has
    no ``image_uuid``) leave it empty.

    For the CLI's instance listing, ``gpu_model`` and
    ``cost_per_hour`` provide the table data without the
    orchestrator having to parse provider JSON.
    """

    provider: "Provider"
    instance_id: str
    label: str
    state: str
    image_uuid: str = ""           # Vast.ai: Docker image reference
    ownership_key: str = ""        # Generic: any provider's ownership token
    gpu_model: str = ""
    cost_per_hour: float = 0.0
    started_at: float = 0.0


@dataclass(frozen=True)
class CleanupResult:
    """Outcome of ``policy.destroy``.

    Exactly one of ``verdict`` (CleanupVerdict) or ``refusal``
    (CleanupRefusal) is set — never both, never neither. ``error``
    is non-empty on every non-DESTROYED outcome (it carries the
    structured diagnostic context); ``error`` is empty on
    DESTROYED (success has no error).

    The cleanup policy wraps the v3 ``DestroyResult`` (which has
    ``verdict: DestroyVerdict`` + ``refusal: DestroyRefusal``)
    in this tighter shape for the orchestrator.
    """

    verdict: Optional[CleanupVerdict] = None
    refusal: Optional[CleanupRefusal] = None
    error: str = ""

    def __post_init__(self) -> None:
        if (self.verdict is None) == (self.refusal is None):
            raise ValueError(
                "CleanupResult: exactly one of verdict or refusal must be set"
            )
        if self.verdict == CleanupVerdict.DESTROYED:
            if self.error:
                raise ValueError(
                    "CleanupResult.DESTROYED must have empty error"
                )
        elif not self.error:
            ident = (
                self.verdict.value
                if self.verdict is not None
                else self.refusal.value
            )
            raise ValueError(
                f"CleanupResult: {ident} must have non-empty error"
            )
```

## `ProviderCleanupPolicy` shape

```python
# src/vastai_gpu_runner/cleanup_policy.py (policy class)
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
            The factory may wrap the provider's list helper with
            the EXPLICITLY_DISABLED short-circuit.
        destroy_fn: Callable that takes one ``InstanceCandidate``,
            applies the eligibility / ownership / credential
            checks authoritatively, and returns a ``CleanupResult``.
            The factory wires the v3 ``destroy_vastai_instance``
            (with CLI fallback) here.

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
        The provider factory may wrap the underlying helper with
        the EXPLICITLY_DISABLED short-circuit.
        """
        try:
            return self.list_instances_fn()
        except Exception as exc:
            logger.warning("Cleanup policy: list_instances raised: %s", exc)
            return []

    def destroy(self, candidate: InstanceCandidate) -> CleanupResult:
        """Run the per-provider destroy on one candidate.

        Authoritative gate: this method re-validates the
        candidate's provider identity (PROVIDER_MISMATCH refusal)
        on every call, then delegates to ``destroy_fn`` which
        applies eligibility / ownership / credential checks.

        Never raises. The orchestrator must be able to log the
        result and continue to the next candidate.
        """
        if candidate.provider != self.provider:
            return CleanupResult(
                refusal=CleanupRefusal.PROVIDER_MISMATCH,
                error=(
                    f"candidate {candidate.instance_id!r} belongs to "
                    f"provider {candidate.provider.value!r}; cleanup "
                    f"policy expects {self.provider.value!r}"
                ),
            )
        try:
            return self.destroy_fn(candidate)
        except Exception as exc:
            logger.warning(
                "Cleanup: destroy_fn raised for %s: %s",
                candidate.instance_id,
                exc,
            )
            return CleanupResult(
                verdict=CleanupVerdict.UNKNOWN,
                error=str(exc),
            )
```

## `build_vastai_cleanup_policy` shape

The provider-owned factory. This is the only place that combines
v3's `destroy_vastai_instance` with the v4 policy. It lives in
`providers/vastai.py` so the core module has no provider imports.

```python
# src/vastai_gpu_runner/providers/vastai.py
def build_vastai_cleanup_policy(
    canonical: VastaiProviderConfig,
) -> ProviderCleanupPolicy:
    """Build a Vast.ai cleanup policy from the canonical config.

    The canonical config (the same ``VastaiProviderConfig`` passed
    to ``VastaiRunner.from_config``) is the source of truth for the
    ownership policy, the credential resolution, and the deployment
    identity. Reading from the same source guarantees the destroy
    ownership guard and the runner's ownership guard cannot drift.

    The wired ``list_instances_fn`` enforces the v3 contract:
    EXPLICITLY_DISABLED credentials return [] without invoking
    CLI enumeration (no API key, no CLI fallback — fail-closed).

    The wired ``destroy_fn`` translates the v3 DestroyResult into
    the v4 CleanupResult, preserving all four verdicts and all
    three refusals.
    """
    from vastai_gpu_runner.cleanup_policy import (
        CleanupRefusal,
        CleanupResult,
        CleanupVerdict,
        OwnershipPolicy,
        ProviderCleanupPolicy,
    )
    from vastai_gpu_runner.providers.destroy import (
        DestroyRefusal,
        DestroyVerdict,
    )
    from vastai_gpu_runner.providers.destroy_adapters.vastai import (
        destroy_vastai_instance,
    )

    def _list_instances() -> list[InstanceCandidate]:
        # v3 contract: EXPLICITLY_DISABLED short-circuits before
        # CLI enumeration. The CLI fallback path uses file
        # credentials which would silently bypass the explicit
        # disable — fail-closed.
        if canonical.credentials.state == CredentialState.EXPLICITLY_DISABLED:
            logger.warning(
                "Zombie sweep disabled: VASTAI_API_KEY is explicitly empty; "
                "provider enumeration was not attempted."
            )
            return []
        return list_vastai_instances()

    def _destroy(candidate: InstanceCandidate) -> CleanupResult:
        # Authoritative eligibility check (per v4 CONCERN 9).
        if candidate.state not in VASTAI_ELIGIBLE_STATES:
            return CleanupResult(
                refusal=CleanupRefusal.INELIGIBLE_STATE,
                error=(
                    f"state {candidate.state!r} not in "
                    f"VASTAI_ELIGIBLE_STATES={set(VASTAI_ELIGIBLE_STATES)!r}"
                ),
            )
        # CREDENTIALS_DISABLED: refuse without provider calls.
        if canonical.credentials.state == CredentialState.EXPLICITLY_DISABLED:
            return CleanupResult(
                refusal=CleanupRefusal.CREDENTIALS_DISABLED,
                error="VASTAI_API_KEY explicitly empty",
            )
        # Delegate to v3 adapter with the canonical ownership + credentials.
        result = destroy_vastai_instance(
            candidate.instance_id,
            ownership=canonical.ownership,
            credentials=canonical.credentials,
        )
        # Translate refusals first.
        if result.refusal is not None:
            mapping = {
                DestroyRefusal.OWNERSHIP: CleanupRefusal.OWNERSHIP,
                DestroyRefusal.NO_CREDENTIALS: CleanupRefusal.NO_CREDENTIALS,
                DestroyRefusal.CREDENTIALS_DISABLED: CleanupRefusal.CREDENTIALS_DISABLED,
            }
            return CleanupResult(
                refusal=mapping[result.refusal],
                error=result.error,
            )
        # Translate verdicts exhaustively.
        if result.verdict == DestroyVerdict.DESTROYED:
            return CleanupResult(verdict=CleanupVerdict.DESTROYED)
        if result.verdict == DestroyVerdict.CLI_ATTEMPTED:
            return CleanupResult(
                verdict=CleanupVerdict.CLI_ATTEMPTED,
                error=result.error,
            )
        if result.verdict == DestroyVerdict.LEAKED:
            return CleanupResult(
                verdict=CleanupVerdict.LEAKED,
                error=result.error,
            )
        return CleanupResult(
            verdict=CleanupVerdict.UNKNOWN,
            error=result.error or "v3 adapter returned no verdict",
        )

    return ProviderCleanupPolicy(
        provider=Provider.VASTAI,
        ownership=canonical.ownership,
        list_instances_fn=_list_instances,
        destroy_fn=_destroy,
    )
```

## Orchestrator wiring

The orchestrator's `_sweep_zombies` becomes policy-driven end-to-end.
The orchestrator logs every non-`DESTROYED` outcome.

```python
# src/vastai_gpu_runner/batch.py (changes to _sweep_zombies)
def _sweep_zombies(self) -> int:
    """Destroy orphaned instances not tracked by live_runners.

    Routes through the cleanup policy:
    1. Enumerate instances via ``policy.list_instances()`` (which
       may short-circuit on EXPLICITLY_DISABLED).
    2. Filter by label prefix (orchestrator's per-batch scope).
    3. Exclude tracked IDs (the existing semantics).
    4. For every remaining candidate, call ``policy.destroy(candidate)``.
    5. Count ``verdict == DESTROYED`` outcomes.
    6. Log every non-DESTROYED outcome distinctly.

    The orchestrator does NOT branch on Provider, does NOT import
    provider modules, and does NOT call any provider-specific
    destroy function. The policy owns the eligibility / ownership
    / credential decisions.
    """
    with self._state_lock:
        tracked_ids = {
            entry[1].instance_id for entry in self._live_runners.values()
        }
    candidates = self._cleanup_policy.list_instances()
    killed = 0
    cli_attempted = 0
    leaked = 0
    unknown = 0
    for candidate in candidates:
        if not candidate.label.startswith(self._label_prefix):
            continue
        if candidate.instance_id in tracked_ids:
            continue
        result = self._cleanup_policy.destroy(candidate)
        if result.verdict == CleanupVerdict.DESTROYED:
            killed += 1
            continue
        # Log every non-DESTROYED outcome distinctly.
        kind = (
            result.verdict.value
            if result.verdict is not None
            else result.refusal.value
        )
        logger.info(
            "Zombie sweep: %s outcome=%s: %s",
            candidate.instance_id,
            kind,
            result.error,
        )
        if result.verdict == CleanupVerdict.CLI_ATTEMPTED:
            cli_attempted += 1
        elif result.verdict == CleanupVerdict.LEAKED:
            leaked += 1
        elif result.verdict == CleanupVerdict.UNKNOWN:
            unknown += 1
    if killed:
        logger.info("Zombie sweep: destroyed %d instance(s)", killed)
    if cli_attempted:
        logger.info(
            "Zombie sweep: %d CLI fallback(s) attempted (destruction not confirmed)",
            cli_attempted,
        )
    if leaked:
        logger.warning("Zombie sweep: %d LEAKED instance(s) — manual review needed", leaked)
    if unknown:
        logger.warning("Zombie sweep: %d UNKNOWN outcome(s)", unknown)
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
    from vastai_gpu_runner.cleanup_policy import OwnershipPolicy
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

The existing `cli.py:cleanup` and `cli.py:instances` commands
migrate to use the new API in migration step 5:

```python
# src/vastai_gpu_runner/cli.py (cleanup command)
@app.command()
def cleanup(
    label_prefix: Annotated[str, typer.Option("--label", "-l")] = ...,
    owned_images: Annotated[
        str | None,
        typer.Option(
            "--owned-images",
            "--allowed-images",  # back-compat alias (kept canonical)
            help="Comma-separated Docker images owned by this project",
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Destroy orphaned Vast.ai instances matching a label prefix."""
    _setup_logging(verbose)
    from rich.console import Console
    from vastai_gpu_runner.cleanup_policy import (
        CleanupVerdict,
        OwnershipPolicy,
    )
    from vastai_gpu_runner.providers.vastai import (
        VastaiProviderConfig,
        build_vastai_cleanup_policy,
    )

    console = Console()
    ownership = (
        OwnershipPolicy(owned_images=frozenset(owned_images.split(",")))
        if owned_images
        else OwnershipPolicy()
    )
    config = VastaiProviderConfig.from_env(owned_images=ownership.owned_images)
    cleanup_policy = build_vastai_cleanup_policy(config)

    candidates = cleanup_policy.list_instances()
    matches = [c for c in candidates if c.label.startswith(label_prefix)]
    if not matches:
        console.print(f"No instances matching label prefix '{label_prefix}'.")
        return

    console.print(f"Found {len(matches)} instance(s) matching '{label_prefix}':")
    for c in matches:
        console.print(
            f"  {c.instance_id}: {c.gpu_model or '?'} "
            f"status={c.state or '?'} label={c.label}"
        )

    if dry_run:
        console.print("\n[yellow]Dry run — no instances destroyed.[/yellow]")
        return

    if not typer.confirm(f"\nDestroy {len(matches)} instance(s)?"):
        console.print("Aborted.")
        raise typer.Exit(0)

    destroyed = 0
    for c in matches:
        result = cleanup_policy.destroy(c)
        if result.verdict == CleanupVerdict.DESTROYED:
            console.print(f"  [green]Destroyed[/green] {c.instance_id}")
            destroyed += 1
        else:
            kind = (
                result.verdict.value
                if result.verdict is not None
                else result.refusal.value
            )
            console.print(f"  [red]{kind}[/red] {c.instance_id}: {result.error}")
    console.print(f"\nDestroyed {destroyed}/{len(matches)} instance(s).")
```

## Migration checklist

Seven steps. The order is the ChatGPT 5th + 6th pass recommendation
(canonical types first, adapter before orchestrator, audit before
deletion). Each step is independently testable but the rollout
lands as one PR because the intermediate states are not stable.

1. **Add canonical credential + ownership-policy types + invariants + contract tests.**
   - `cleanup_policy.py:OwnershipPolicy` (frozen, `matches(image_ref)` with the v3 `_repository` helper).
   - `providers/destroy_adapters/vastai.py:CredentialState` + `CredentialResolution` (v3 prerequisite — may already exist if v3 is implemented).
   - Property tests: `OwnershipPolicy.matches` is reflexive (the owned image matches itself); tag-insensitive (`:1.0` and `:latest` match when the repository is owned); `None` opts out (every image matches); empty set rejects everything (fail-closed); different registry prefixes do not match (`registry:5000/myorg/app` ≠ `registry-malicious/myorg/app`); sha256 digests match by repository; malformed references (multiple `@`, multiple `:`) return `False`.
   - `CredentialResolution` invariants: `AVAILABLE` requires non-empty pre-stripped key; `ABSENT` and `EXPLICITLY_DISABLED` require empty key.

2. **Add `VastaiProviderConfig` + `VastaiRunner.from_config` + constructor parity.**
   - `providers/vastai.py:VastaiProviderConfig` (frozen, owns `ownership` + `credentials` + `docker_image`).
   - `__post_init__` invariants: `docker_image` non-empty + pre-stripped; `docker_image ∈ ownership.owned_images` unless ownership is disabled.
   - `VastaiRunner.from_config(config)` classmethod. Constructor parity test: `VastaiRunner(allowed_images=frozenset({img}), docker_image=img, ...)` (deprecated path) and `VastaiRunner.from_config(VastaiProviderConfig(docker_image=img, ownership=OwnershipPolicy(owned_images=frozenset({img})), ...))` produce equivalent `ownership` attribute values.
   - `VastaiRunner.__init__` adds `ownership: OwnershipPolicy | None` parameter; `allowed_images` becomes a deprecated alias (emits `DeprecationWarning` when used).
   - `verify_instance_ownership(instance_id, *, ownership=OwnershipPolicy)` — replaces the raw `allowed_images` set check with `ownership.matches(image_uuid)`. The `_image_is_allowed` helper is **deleted**.
   - `VastaiProviderConfig.from_env()` reads the v3 `read_vastai_api_key()` into `credentials`.

3. **Add Vast.ai cleanup adapter (enumeration + authoritative destroy + CLI fallback).**
   - `providers/vastai.py:list_vastai_instances()` (returns `list[InstanceCandidate]`, empty list on API failure).
   - `providers/vastai.py:VASTAI_ELIGIBLE_STATES` module constant (frozenset of states the destroy path can act on).
   - `providers/vastai.py:build_vastai_cleanup_policy(config)` (wires `list_instances_fn` + `destroy_fn`).
   - The factory wraps `list_vastai_instances` with the EXPLICITLY_DISABLED short-circuit.
   - The factory's `_destroy` performs the eligibility check (`INELIGIBLE_STATE`) before delegating to `destroy_vastai_instance`.
   - Adapter tests:
     - **disabled-before-enumeration**: config with `credentials=EXPLICITLY_DISABLED`; `policy.list_instances()` returns `[]` without invoking `vastai_cmd` (mock and assert `vastai_cmd` was not called).
     - `AVAILABLE` path: `destroy_fn` calls `destroy_vastai_instance` with the loaded key; `verdict=DESTROYED` → `CleanupResult(verdict=DESTROYED)`.
     - `ABSENT` path: `destroy_fn` falls through to CLI fallback (per v3 contract); `verdict=CLI_ATTEMPTED` → `CleanupResult(verdict=CLI_ATTEMPTED, error=...)`.
     - `verdict=OWNERSHIP` translation → `CleanupResult(refusal=OWNERSHIP)`.
     - `verdict=NO_CREDENTIALS` translation → `CleanupResult(refusal=NO_CREDENTIALS)`.
     - `verdict=CREDENTIALS_DISABLED` translation → `CleanupResult(refusal=CREDENTIALS_DISABLED)`.
     - `verdict=LEAKED` translation → `CleanupResult(verdict=LEAKED, error=...)`.
     - `verdict=UNKNOWN` translation → `CleanupResult(verdict=UNKNOWN, error=...)`.
     - `INELIGIBLE_STATE` path: candidate.state not in `VASTAI_ELIGIBLE_STATES` → `CleanupResult(refusal=INELIGIBLE_STATE, error=...)` without invoking `destroy_vastai_instance`.
     - `list_vastai_instances()` returns `[]` on `RuntimeError` and `JSONDecodeError` (does not raise).

4. **Add orchestrator support behind a fail-closed compatibility path.**
   - `BatchOrchestrator.__init__` accepts `cleanup_policy: ProviderCleanupPolicy` (required).
   - `_sweep_zombies` becomes policy-driven (the v4.3 code shape above).
   - Orchestrator tests:
     - `_sweep_zombies` calls `cleanup_policy.list_instances()` exactly once.
     - `_sweep_zombies` calls `cleanup_policy.destroy(candidate)` for every label-matching, untracked candidate.
     - `_sweep_zombies` counts only `verdict=DESTROYED` outcomes.
     - `_sweep_zombies` logs `CLI_ATTEMPTED`, `LEAKED`, `UNKNOWN`, and every refusal distinctly.
     - `_sweep_zombies` continues on `destroy_fn` exceptions (they return `verdict=UNKNOWN`).
     - `_sweep_zombies` does NOT branch on `Provider` (verified by `inspect.getsource` — `batch.py` should not contain `from vastai_gpu_runner.providers.vastai import` outside the docstring).

5. **Update every composition root, subclass, and existing CLI command.**
   - `cli.py:batch` (composition root): build `VastaiProviderConfig` via `from_env()` + `replace()`, pass to `VastaiRunner.from_config` and `build_vastai_cleanup_policy`.
   - `cli.py:cleanup` (existing CLI command): migrate to use `build_vastai_cleanup_policy` + `policy.list_instances()` + `policy.destroy()`. No more direct `vastai_cmd(["show", "instances", "--raw"])` parsing.
   - `cli.py:instances` (existing CLI command): migrate to use `list_vastai_instances()` for the table. The `--allowed-images` flag is preserved as canonical; `--owned-images` is added as a documented alias (back-compat for any external scripts that already use `--owned-images`).
   - `BatchOrchestrator` subclasses (consumer code): existing `BatchOrchestrator(...)` calls now require `cleanup_policy`. Update each subclass's composition.
   - Test fixtures and conftest: `VastaiProviderConfig` factory fixture for tests.

6. **Add integration tests.**
   - `tests/integration/test_cleanup_policy_integration.py` (or equivalent):
     - **disabled-before-enumeration**: config with `credentials=EXPLICITLY_DISABLED`; `policy.list_instances()` returns `[]`; `vastai_cmd` was not called (verified by mock call count).
     - **absent-credential CLI fallback**: config with `credentials=ABSENT`; `policy.destroy(candidate)` falls through to CLI fallback path; the v3 `destroy_vastai_instance` is invoked with `credentials=None` (the CLI fallback path); `verdict=CLI_ATTEMPTED` translates to `CleanupResult(verdict=CLI_ATTEMPTED, error=...)`.
     - **empty ownership set**: config with `ownership=OwnershipPolicy(owned_images=frozenset())`; `policy.destroy(candidate)` returns `OWNERSHIP` for every candidate (fail-closed).
     - **provider mismatch**: candidate with `provider=Provider.RUNPOD` passed to a `Provider.VASTAI` policy; returns `PROVIDER_MISMATCH` with the operator-friendly diagnostic.
     - **enumeration failure**: `list_vastai_instances()` raises on `RuntimeError`; `policy.list_instances()` returns `[]` (the policy catches the exception).
     - **ineligible state**: candidate with `state="destroyed"` passed to the Vast.ai policy; returns `INELIGIBLE_STATE` without invoking `destroy_vastai_instance`.
     - **non-DESTROYED logging**: orchestrator logs `CLI_ATTEMPTED`, `LEAKED`, `UNKNOWN`, and every refusal distinctly (verified by `caplog`).

7. **Delete legacy sweep + duplicated helpers after a repository-wide caller audit.**
   - `audit_caller_sites.sh` (run before the deletion): grep for any external caller of `orchestrator.sweep_zombie_instances`, `orchestrator.load_vastai_api_key`, `VastaiRunner.allowed_images` (read-only external use). Update external callers.
   - Delete `orchestrator.sweep_zombie_instances` (already deferred to v3; v4.3 reaffirms).
   - Delete `orchestrator.load_vastai_api_key` (v3 deferral reaffirmed).
   - Delete `providers/vastai.py:_image_is_allowed` (replaced by `OwnershipPolicy.matches()`).
   - Delete direct `vastai_cmd(["show", "instances", "--raw"])` parsing in `cli.py:cleanup` and `cli.py:instances`.
   - Update `tests/test_orchestrator.py` and `tests/test_batch.py` to mock `cleanup_policy.list_instances` and `cleanup_policy.destroy` instead of `sweep_zombie_instances`.

## Test plan

- `tests/test_cleanup_policy.py` — unit tests:
  - `_repository`:
    - `myorg/app:1.0` → `myorg/app`
    - `myorg/app:latest` → `myorg/app`
    - `myorg/app@sha256:abc...` → `myorg/app` (digest stripped)
    - `ubuntu:22.04` → `ubuntu` (no slash)
    - `registry:5000/myorg/app:1.0` → `registry:5000/myorg/app` (registry preserved)
    - `registry-malicious/myorg/app:1.0` → `registry-malicious/myorg/app`
    - `myorg/app:1.0:malicious` → `""` (malformed multi-tag)
    - `myorg/app@sha1:abc@sha256:def` → `""` (malformed multi-digest)
    - `""` → `""` (empty)
    - `"   "` → `""` (whitespace only)
  - `OwnershipPolicy.matches`:
    - `None` ownership: every image matches (including the empty string and malformed references)
    - Non-empty set: exact reference matches
    - Non-empty set: tag-insensitive repository matches (`:1.0` and `:latest` both match)
    - Non-empty set: sha256 digest matches by repository
    - Non-empty set: different registry prefix does NOT match
    - Non-empty set: image `myorg/app:1.0` does NOT match `myorg/app-malicious:latest`
    - Non-empty set: `registry:5000/myorg/app:1.0` does NOT match `registry-malicious/myorg/app:1.0`
    - Empty set: no image matches (fail-closed), including for the empty string and malformed references
    - `_normalised` is precomputed and frozen: `matches` is O(1)
  - `ProviderCleanupPolicy.list_instances`:
    - Returns the wired list
    - Catches and returns `[]` on `list_instances_fn` exception
  - `ProviderCleanupPolicy.destroy`:
    - Provider mismatch returns `PROVIDER_MISMATCH` with the operator-friendly error
    - Catches `destroy_fn` exceptions and returns `CleanupResult(verdict=UNKNOWN, error=...)`
    - Delegates to `destroy_fn` on the happy path
  - `CleanupResult` invariants:
    - `verdict=DESTROYED` + non-empty `error` raises `ValueError`
    - `verdict=CLI_ATTEMPTED` + empty `error` raises `ValueError`
    - `refusal=...` + empty `error` raises `ValueError`
    - Both `verdict` and `refusal` set raises `ValueError`
    - Neither `verdict` nor `refusal` set raises `ValueError`
  - `VastaiProviderConfig` invariants:
    - Empty `docker_image` raises `ValueError`
    - Whitespace-only `docker_image` raises `ValueError`
    - `docker_image` not in `owned_images` raises `ValueError`
    - `docker_image` in `owned_images` is valid
    - `ownership.owned_images is None` (disabled) accepts any `docker_image`
    - `from_env(docker_image="")` raises `ValueError` (explicit empty is not None)
    - `from_env(docker_image=None)` returns the default image
- `tests/test_providers_vastai.py` — adapter tests:
  - `VastaiRunner.from_config` round-trips with `VastaiProviderConfig` (constructor parity)
  - `VastaiRunner(allowed_images=frozenset({img}))` (deprecated path) emits `DeprecationWarning` and builds an equivalent `OwnershipPolicy`
  - `verify_instance_ownership` uses `OwnershipPolicy.matches()`
  - `list_vastai_instances` returns `list[InstanceCandidate]` with `gpu_model` + `cost_per_hour` populated
  - `list_vastai_instances` returns `[]` on API error (does not raise)
  - `build_vastai_cleanup_policy` wires the v3 adapter correctly (all 4 verdicts, all 3 refusals, `INELIGIBLE_STATE`, `PROVIDER_MISMATCH`, EXPLICITLY_DISABLED short-circuit)
- `tests/test_batch.py` — orchestrator wiring:
  - `_sweep_zombies` calls `cleanup_policy.list_instances()` exactly once
  - `_sweep_zombies` calls `cleanup_policy.destroy(candidate)` for every label-matching, untracked candidate
  - `_sweep_zombies` counts only `verdict=DESTROYED` outcomes
  - `_sweep_zombies` logs `CLI_ATTEMPTED`, `LEAKED`, `UNKNOWN`, and every refusal distinctly (`caplog`)
  - `_sweep_zombies` continues on `destroy_fn` exceptions
  - `_sweep_zombies` does NOT import provider modules (verified by `inspect.getsource`)
- `tests/integration/test_cleanup_policy_integration.py` — integration tests (seven scenarios from step 6 above).
- Existing tests for `verify_instance_ownership` and `VastaiRunner.destroy_instance` are updated to use `OwnershipPolicy` (the v4 amendment).

## Backwards compatibility

The `VastaiRunner.__init__(allowed_images=..., docker_image=..., ...)`
constructor is preserved as a deprecated back-compat path. Existing
callers (scripts, unit tests, third-party consumers) that build the
runner by hand keep working but emit a `DeprecationWarning`. The
CLI's new `batch` subcommand uses `from_config`; the old programmatic
path requires a `cleanup_policy` to be supplied.

The `orchestrator.py:sweep_zombie_instances` function is deleted in
v4 step 7. The v3 implementation already migrated its callers
through the destroy protocol; v4's policy-driven sweep removes the
last direct caller. Any external code that imported
`sweep_zombie_instances` directly must be updated to construct a
`ProviderCleanupPolicy` and call `policy.destroy(candidate)`.

The `cli.py:cleanup` and `cli.py:instances` commands are refactored
in v4 step 5. The `--allowed-images` flag is preserved as canonical;
`--owned-images` is added as a documented alias. The rename is
deferred to a separate CLI compatibility change (no removal of
`--allowed-images` in this PR).

## Out of scope

- **RunPod adapter.** The `ProviderCleanupPolicy` interface is
  provider-agnostic, but the `build_runpod_cleanup_policy` factory
  is omitted from this doc — it lands with the RunPod adapter
  (roadmap item 2). The factory shape is defined by the
  `ProviderCleanupPolicy` interface; the orchestrator's wiring is
  provider-agnostic from day one.
- **Hostile detection.** The v4 first draft had a `hostile_check`
  boolean + threshold; the 5th-pass review flagged it as
  semantically redundant. The 6th-pass review reaffirmed the
  removal. Hostile detection is deferred to a separate audit
  stage that emits alerts without changing the destroy decision.
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

This is the third review pass on the v4 architecture. The first
draft was rejected with 5 BLOCKERs and 7 CONCERNs. The second draft
was rejected with 7 BLOCKERs, 2 CONCERNs, and 3 NITs. This third
draft addresses every finding.

The third review prompt for ChatGPT-with-GitHub-plugin:

> Review the v4 architecture design at PR #22 (file:
> docs/architecture-v4-cleanup-policy.md) against the v3 design at
> docs/architecture-v3.md and the current code at
> src/vastai_gpu_runner/{batch,orchestrator,runner,cli}.py and
> src/vastai_gpu_runner/providers/vastai.py. The v4 design
> resolves issue #19 (the ProviderCleanupPolicy follow-up).
>
> The first draft was rejected with 5 BLOCKERs and 7 CONCERNs.
> The second draft was rejected with 7 BLOCKERs, 2 CONCERNs, and
> 3 NITs. Verify each finding is addressed:
>
> 1. **BLOCKER 1 (_repository broken)**: confirm `_repository` now
>    strips digest, strips only the final tag separator (after
>    the last `/`), preserves registry and port, and rejects
>    malformed references.
> 2. **BLOCKER 2 (runner/adapter don't consume OwnershipPolicy)**:
>    confirm both call `OwnershipPolicy.matches()` and the v3
>    `destroy_vastai_instance` accepts `ownership: OwnershipPolicy`.
> 3. **BLOCKER 3 (v3 type names)**: confirm `CredentialState` (enum)
>    and `CredentialResolution` (dataclass with `state` and `key`)
>    are used.
> 4. **BLOCKER 4 (EXPLICITLY_DISABLED before enumeration)**: confirm
>    the factory wraps `list_vastai_instances` and the integration
>    test asserts `vastai_cmd` is not called when credentials are
>    disabled.
> 5. **BLOCKER 5 (CLI fallback + DestroyResult translation)**: confirm
>    the `_destroy` callback refuses EXPLICITLY_DISABLED without
>    provider calls, uses canonical API key for REST, runs CLI fallback
>    for ABSENT, returns `CLI_ATTEMPTED` after CLI command, translates
>    all four v3 verdicts and all three v3 refusals.
> 6. **BLOCKER 6 (docker_image invariant)**: confirm empty or
>    whitespace-only `docker_image` raises `ValueError`.
> 7. **BLOCKER 7 (undefined names)**: confirm `cleanup_policy.py`
>    imports `logging` and `cli.py` imports `OwnershipPolicy`.
> 8. **CONCERN 8 (CleanupResult invariants)**: confirm `verdict |
>    refusal` is mutually exclusive and non-DESTROYED outcomes have
>    non-empty `error`.
> 9. **CONCERN 9 (authoritative eligibility)**: confirm the factory's
>    `_destroy` checks `candidate.state not in VASTAI_ELIGIBLE_STATES`
>    and returns `INELIGIBLE_STATE`.
> 10. **CONCERN 10 (InstanceCandidate fields)**: confirm
>     `gpu_model`, `cost_per_hour`, `ownership_key` are added.
> 11. **CONCERN 11 (CLI flag rename)**: confirm `--allowed-images`
>     is preserved as canonical and `--owned-images` is an alias.
> 12. **NIT 12 (provider assignment)**: confirm `provider=Provider.VASTAI`
>     is hardcoded (not derived from canonical.docker_image).
> 13. **NIT 13 (provider-mismatch diagnostic)**: confirm the error
>     message includes the instance ID, actual provider, and
>     expected provider.
> 14. **NIT 14 (stale prose)**: confirm all references to
>     `is_candidate`, "hostile classification", "verbidden", and
>     "the core module has provider-specific imports" are removed.
>
> Additionally, identify any new BLOCKERs or CONCERNs introduced
> by the third draft. Focus on:
> - The `CleanupResult` verdict/refusal mapping in
>   `build_vastai_cleanup_policy._destroy`: is the translation
>   table complete and correct?
> - The `_destroy` catch-all exception path: does it return
>   `verdict=UNKNOWN` or `refusal=...` consistently?
> - The `ProviderCleanupPolicy.destroy` catch-all: same question.
> - The VastaiRunner `allowed_images` deprecation: is the
>   `DeprecationWarning` emitted with the correct stacklevel?
> - The `OwnershipPolicy._normalised` caching: is the
>   `object.__setattr__` pattern correct on a frozen dataclass
>   for an internal-cache field?
> - The `VastaiProviderConfig.__post_init__` ordering: are the
>   `docker_image` non-empty check and the ownership-membership
>   check in the right order?
> - The `build_vastai_cleanup_policy` EXPLICITLY_DISABLED short-
>   circuit: is the log message appropriate (warning vs. info)?
> - The `_destroy` eligibility check: is `VASTAI_ELIGIBLE_STATES`
>   complete (covers all stuck/loading/created states)?
> - The `_repository` helper: does it handle port-only registries
>   (`localhost:5000/myorg/app`) correctly?
>
> Return a labeled list of findings. Each finding is one of:
> BLOCKER (must fix before merge), CONCERN (should fix, but not
> blocking), or NIT (nice to have). For each finding, give the
> exact line range, the issue, and the proposed fix. If the
> design is acceptable as-is, say "DESIGN ACCEPTED" with a
> one-line rationale.