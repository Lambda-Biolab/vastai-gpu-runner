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
object that is constructed once at boot time from the canonical
`OwnershipPolicy` + `CredentialResolution` read at boot time. The
runner and the cleanup policy now share **one** ownership semantic
(`OwnershipPolicy.matches(image_ref)`) and **one** credential snapshot
(`CredentialResolution`), consumed verbatim by both — no per-destroy
runner construction, no semantic drift between the runner's destroy
guard and the zombie-sweep guard. The `ProviderCleanupPolicy` is
frozen (`kw_only=True`), provider-agnostic, and exposes
`list_instances()` + `destroy(candidate)` as the two methods the
orchestrator calls. The factory (`build_vastai_cleanup_policy`,
`ownership`, `credentials`) wires the v3 `destroy_vastai_instance`
adapter for the REST path and intercepts `NO_CREDENTIALS` to run the
CLI fallback (CLI ownership verification + CLI destroy →
`CLI_ATTEMPTED` verdict). The factory wraps `list_vastai_instances`
so `EXPLICITLY_DISABLED` credentials short-circuit before enumeration.
The orchestrator logs every non-`DESTROYED` outcome at a severity
matching its operational impact (`LEAKED` = `ERROR`, `UNKNOWN` /
`CLI_ATTEMPTED` / `CREDENTIALS_DISABLED` = `WARNING`, refusals =
`INFO`). The `VastaiRunner.destroy_instance` method delegates entirely
to the v3 adapter (no v2 regression).

Diff vs v3 once v3 is implemented:

- **+** `src/vastai_gpu_runner/cleanup_policy.py` — `ProviderCleanupPolicy` (frozen, `kw_only`), `InstanceCandidate` (frozen, non-empty `instance_id` invariant), `CleanupVerdict` enum (`DESTROYED | CLI_ATTEMPTED | LEAKED | UNKNOWN`), `CleanupRefusal` enum (`OWNERSHIP | NO_CREDENTIALS | CREDENTIALS_DISABLED | INELIGIBLE_STATE | PROVIDER_MISMATCH`), `CleanupResult` (typed return with `__post_init__` invariants), `OwnershipPolicy` (frozen, `matches(image_ref)`, declared `_normalised` cache field)
- **+** `src/vastai_gpu_runner/providers/vastai.py:VastaiProviderConfig` (frozen, owns `ownership: OwnershipPolicy`, `credentials: CredentialResolution`, `docker_image`, etc.)
- **+** `src/vastai_gpu_runner/providers/vastai.py:build_vastai_cleanup_policy(*, ownership, credentials)` — provider-owned factory that takes the two canonical objects directly (no config wrapper)
- **+** `src/vastai_gpu_runner/providers/vastai.py:list_vastai_instances()` — read-only enumeration returning `list[InstanceCandidate]`, validates `instance_id` non-empty
- **~** `providers/destroy_adapters/vastai.py:destroy_vastai_instance` accepts `ownership: OwnershipPolicy` directly (replaces `allowed_images: frozenset[str]`); accepts `credentials: CredentialResolution | None` (defaults to `read_vastai_api_key()` for back-compat direct callers). The v3 implementation must adopt this signature.
- **~** `providers/destroy_adapters/vastai.py:CredentialState` is `StrEnum` (matches v3 verbatim — the v4 third draft used plain `Enum`).
- **~** `VastaiRunner.__init__` accepts `ownership: OwnershipPolicy | None` and `credentials: CredentialResolution | None`; rejects simultaneous `ownership=` and deprecated `allowed_images=` with `ValueError`.
- **~** `VastaiRunner.from_config(config)` preserves both `canonical.ownership` and `canonical.credentials`.
- **~** `VastaiRunner.destroy_instance` delegates entirely to `destroy_vastai_instance(...)` — no v2-style inline ownership pre-check, no inline REST stop/delete/verify. Returns `bool` from the typed adapter result.
- **~** `BatchOrchestrator.__init__` accepts `cleanup_policy: ProviderCleanupPolicy` (required). The orchestrator calls `policy.list_instances()` and `policy.destroy(candidate)` — never branches on `Provider`, never imports provider modules.
- **~** `BatchOrchestrator._sweep_zombies` is policy-driven end-to-end. The label-prefix filter and tracked-id exclusion stay on the orchestrator. Every other decision is delegated to `policy.destroy()`. The orchestrator logs every non-`DESTROYED` outcome at severity matching operational impact.
- **~** `cli.py:cleanup` and `cli.py:instances` — refactored to use the new API. The `--allowed-images` flag is the canonical primary; `--owned-images` is an alias.
- **—** `orchestrator.py:sweep_zombie_instances` (v3 deletion) — reaffirmed. The v4 implementation removes the last direct caller.
- **—** `orchestrator.py:load_vastai_api_key` (v3 deletion) — reaffirmed.
- **—** `providers/vastai.py:_image_is_allowed` (raw set check) — replaced by `OwnershipPolicy.matches()`.
- **~** `tests/test_orchestrator.py` and `tests/test_batch.py` — mock `cleanup_policy.list_instances()` and `cleanup_policy.destroy()` instead of `sweep_zombie_instances`.

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
compete for the same role. The `OwnershipPolicy` and
`CredentialResolution` are the two canonical objects shared by the
runner factory and the cleanup-policy factory. The runner wraps them
in `VastaiProviderConfig` (because the runner also needs
`docker_image`, `setup_commands`, etc.); the cleanup-policy factory
takes them directly.

## What changes vs the v4 first + second + third drafts (5th + 6th + 7th pass reviews)

The first draft was rejected with 5 BLOCKERs and 7 CONCERNs. The
second draft was rejected with 7 BLOCKERs, 2 CONCERNs, and 3 NITs.
The third draft was rejected with 6 BLOCKERs, 4 CONCERNs, and 2 NITs.
This fourth draft addresses every finding.

### Applied from the 7th-pass review (this pass)

- **ABSENT-credential CLI fallback implemented** in the factory. The factory intercepts `DestroyRefusal.NO_CREDENTIALS` and runs the CLI fallback path: `verify_instance_ownership` (CLI auth context) → if not owned return `OWNERSHIP`, else `vastai_cmd(["destroy", "instance", ...])` → `CLI_ATTEMPTED` (destruction unconfirmed) on command success or `UNKNOWN` on command failure. The canonical ABSENT resolution is passed to the REST adapter (not `None`). (BLOCKER 1)
- **v3 `DestroyResult` translation corrected.** v3 `DestroyVerdict` has exactly `DESTROYED | LEAKED | UNKNOWN` (no `CLI_ATTEMPTED` — that's only in v4 `CleanupVerdict`). v3 `DestroyResult` has `attempts`, `stop_error`, `last_status_code`, `verify_error` — no generic `error` field. Diagnostic text is built from the structured fields via a `_describe(result)` helper. (BLOCKER 2)
- **`VastaiRunner.destroy_instance` delegates to v3 adapter.** No v2-style inline ownership pre-check, no inline REST stop/delete/verify. The runner adds `credentials` to its constructor, `from_config` preserves the canonical `CredentialResolution`, and `destroy_instance` is a single adapter call. Simultaneous `ownership=` and `allowed_images=` raises `ValueError`. (BLOCKER 3)
- **`never raises` catch path hardened.** The catch returns `error=f"{type(exc).__name__}: {exc}"` so `error` is always non-empty even for `RuntimeError()` with no message. The `CleanupResult.__post_init__` invariant is preserved. (BLOCKER 4)
- **`build_vastai_cleanup_policy` takes `ownership` + `credentials` directly.** No `VastaiProviderConfig` wrapper — the deployment-image invariant in `VastaiProviderConfig.__post_init__` does not apply to listing/cleanup-only commands. The batch command extracts `ownership` and `credentials` from `VastaiProviderConfig`; the cleanup command constructs them directly from CLI args. (BLOCKER 5)
- **`instance_id` non-empty validated at two boundaries.** `list_vastai_instances` skips records with empty `instance_id`; `InstanceCandidate.__post_init__` raises `ValueError` for empty or whitespace-only IDs. (BLOCKER 6)
- **`_normalised` declared as a dataclass field** with `field(init=False, repr=False, compare=False)`. The `object.__setattr__` pattern in `__post_init__` is retained. (CONCERN 7)
- **`CredentialState` uses `StrEnum`** matching v3 verbatim. (CONCERN 8)
- **`VASTAI_TERMINAL_STATES` (negative list)** replaces the previous positive allowlist. Only `destroyed` is skipped; new Vast.ai states are processed. (CONCERN 9)
- **Logging severity by outcome.** `LEAKED` = `ERROR`, `UNKNOWN` / `CLI_ATTEMPTED` / `CREDENTIALS_DISABLED` = `WARNING`, refusals = `INFO`. Each log line includes the instance ID. (CONCERN 10)
- **`--allowed-images` is the canonical CLI flag** (declared first in `typer.Option`); `--owned-images` is the alias. The Python parameter is `allowed_images`. (NIT 11)
- **Historical `is_candidate` reference removed** from the prose (the historical change summary no longer mentions it; the active flow is what matters). (NIT 12)

### Applied from the 6th-pass review (third pass)

- `_repository(image_ref)` strips digest, strips only the final tag separator (after the last `/`), preserves registry and port, rejects malformed references.
- Runner and adapter consume `OwnershipPolicy.matches()` directly.
- v3 type names used exactly (`CredentialState`, `CredentialResolution`).
- `EXPLICITLY_DISABLED` short-circuits before enumeration.
- CLI fallback + `DestroyResult` translation (superseded by 7th-pass BLOCKER 1 above).
- `docker_image` non-empty + pre-stripped invariant.
- Undefined names fixed (logger import, OwnershipPolicy imports).
- `CleanupResult` invariants tightened.
- Authoritative eligibility implemented (now superseded by 7th-pass CONCERN 9 negative-list).
- `InstanceCandidate` enriched with `gpu_model`, `cost_per_hour`, `ownership_key`.
- CLI flag kept as `--allowed-images` canonical.

### Applied from the 5th-pass review (second pass)

- `@dataclass(frozen=True, kw_only=True)` for `ProviderCleanupPolicy`.
- `list_instances_fn` + `list_instances()` method on the policy.
- `destroy()` is authoritative.
- Hostile check removed.
- v3 `CredentialResolution` types used.
- `OwnershipPolicy.matches()` introduced.
- CLI uses `dataclasses.replace`.
- Provider factories moved to `providers/vastai.py`.
- State filter removed (now superseded by 7th-pass CONCERN 9 negative-list).
- `CleanupResult` invariants added.
- `from_runpod_config` removed.
- Migration order revised.

### Applied from the 5th-pass review (first draft)

- The first draft had BLOCKERs 1-5 and CONCERNs 1-7; all addressed in subsequent passes.

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

The Vast.ai factory is a provider-owned function. It takes the two
canonical objects (`OwnershipPolicy`, `CredentialResolution`)
directly, wires the v3 `destroy_vastai_instance` + a wrapped
`list_vastai_instances` into the policy, and returns the configured
`ProviderCleanupPolicy`. The factory intercepts `NO_CREDENTIALS` for
the CLI fallback path.

The RunPod factory is omitted from this doc — it lands with the
RunPod adapter (roadmap item 2).

## Layered design (v4)

```
┌─────────────────────────────────────────────────┐
│  CLI (cli.py)                                   │  User-facing commands
│    └── builds OwnershipPolicy +                 │
│        CredentialResolution once, threads       │
│        both into VastaiRunner.from_config       │
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

## `cleanup_policy.py` — full module

All shapes that live in `cleanup_policy.py` are shown together so
imports and references resolve within one block.

```python
# src/vastai_gpu_runner/cleanup_policy.py
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import AbstractSet, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vastai_gpu_runner.types import Provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _repository helper — tag-insensitive image reference normalisation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# OwnershipPolicy
# ---------------------------------------------------------------------------


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
        _normalised: Internal cache of normalised repositories.
            Precomputed in ``__post_init__`` so ``matches()`` is O(1).
            Excluded from equality, hashing, and repr.
    """

    owned_images: AbstractSet[str] | None = None
    _normalised: frozenset[str] | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

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
        """
        if self.owned_images is None:
            return True
        if not image_ref:
            return False
        repo = _repository(image_ref)
        if not repo:
            return False
        return repo in self._normalised


# ---------------------------------------------------------------------------
# Enums: CleanupVerdict, CleanupRefusal
# ---------------------------------------------------------------------------


class CleanupVerdict(Enum):
    """Outcome verdicts returned by ``policy.destroy``.

    These are protocol outcomes — the destroy protocol ran to
    completion and reported a verdict. ``DESTROYED`` is the only
    success; the others are observable non-success states that
    the orchestrator logs distinctly.

    ``CLI_ATTEMPTED`` is a v4 verdict produced by the factory's
    CLI fallback path — it is NOT a v3 protocol verdict.
    """

    DESTROYED = "destroyed"
    CLI_ATTEMPTED = "cli_attempted"  # CLI fallback ran; destruction not confirmed
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
    INELIGIBLE_STATE = "ineligible_state" # candidate.state is terminal (destroyed) or malformed
    PROVIDER_MISMATCH = "provider_mismatch"  # candidate.provider != policy.provider


# ---------------------------------------------------------------------------
# Dataclasses: InstanceCandidate, CleanupResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstanceCandidate:
    """Read-only snapshot of one instance returned by ``list_*_instances``.

    Frozen so the policy can hold it without aliasing surprises.
    Fields are the union of what every provider can feasibly
    expose; providers that do not have a field (e.g. RunPod has
    no ``image_uuid``) leave it empty.

    Invariant: ``instance_id`` must be non-empty and pre-stripped.
    """

    provider: "Provider"
    instance_id: str
    label: str
    state: str
    image_uuid: str = ""
    ownership_key: str = ""
    gpu_model: str = ""
    cost_per_hour: float = 0.0
    started_at: float = 0.0

    def __post_init__(self) -> None:
        if (
            not self.instance_id
            or self.instance_id != self.instance_id.strip()
        ):
            raise ValueError(
                "InstanceCandidate.instance_id must be non-empty and pre-stripped"
            )


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


# ---------------------------------------------------------------------------
# ProviderCleanupPolicy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ProviderCleanupPolicy:
    """Per-provider cleanup contract.

    The core module is provider-agnostic: it imports nothing from
    ``providers/``. The ``list_instances_fn`` and ``destroy_fn``
    callbacks are wired by the provider-owned factory (e.g.
    ``build_vastai_cleanup_policy``).
    """

    provider: "Provider"
    ownership: OwnershipPolicy
    list_instances_fn: Callable[[], list[InstanceCandidate]] = field(repr=False)
    destroy_fn: Callable[[InstanceCandidate], CleanupResult] = field(repr=False)

    def list_instances(self) -> list[InstanceCandidate]:
        """Read-only enumeration of provider instances."""
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

        Never raises. The error string in the catch path is
        always non-empty (uses ``type(exc).__name__`` as a
        minimum) so the ``CleanupResult`` invariant is preserved.
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
                error=f"{type(exc).__name__}: {exc}",
            )
```

## `providers/destroy_adapters/vastai.py` — v3 + v4 additions

The v3 design's `CredentialState` + `CredentialResolution` types
plus the amended `destroy_vastai_instance` signature.

```python
# src/vastai_gpu_runner/providers/destroy_adapters/vastai.py
from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Optional

from vastai_gpu_runner.cleanup_policy import OwnershipPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Credential types (v3 shape — verbatim)
# ---------------------------------------------------------------------------


class CredentialState(StrEnum):
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


# ---------------------------------------------------------------------------
# read_vastai_api_key
# ---------------------------------------------------------------------------


def read_vastai_api_key() -> CredentialResolution:
    """Read the Vast.ai API key from the standard file paths.

    Returns a frozen CredentialResolution. The resolution distinguishes
    three states:
        - EXPLICITLY_DISABLED: the key file exists but is empty
          (the user has explicitly disabled credentials).
        - ABSENT: no key file is present.
        - AVAILABLE: a non-empty key was found.
    """
    for kp in (
        Path("~/.config/vastai/vast_api_key").expanduser(),
        Path("~/.vast_api_key").expanduser(),
    ):
        if kp.exists():
            raw = kp.read_text()
            stripped = raw.strip()
            if not stripped:
                return CredentialResolution(
                    state=CredentialState.EXPLICITLY_DISABLED
                )
            return CredentialResolution(
                state=CredentialState.AVAILABLE, key=stripped
            )
    return CredentialResolution(state=CredentialState.ABSENT)


# ---------------------------------------------------------------------------
# DestroyResult shape (v3)
# ---------------------------------------------------------------------------


class DestroyVerdict(StrEnum):
    """Outcome verdicts returned by the v3 belt-and-suspenders protocol.

    Three values. CLI_ATTEMPTED is NOT a v3 verdict — it is a v4
    CleanupVerdict produced by the factory's CLI fallback.
    """
    DESTROYED = "destroyed"
    LEAKED = "leaked"
    UNKNOWN = "unknown"


class DestroyRefusal(StrEnum):
    """Pre-protocol refusal reasons returned by destroy_vastai_instance."""
    OWNERSHIP = "ownership"
    NO_CREDENTIALS = "no_credentials"
    CREDENTIALS_DISABLED = "credentials_disabled"


@dataclass(frozen=True)
class DestroyResult:
    """Outcome of the v3 belt-and-suspenders destroy protocol.

    Exactly one of ``verdict`` (DestroyVerdict) or ``refusal``
    (DestroyRefusal) is set. The structured fields carry the
    diagnostic context for the verdict path; the refusal path
    uses stable message synthesis in the factory.

    The fields below are the v3 contract. No generic ``error``
    field exists — refusals carry structured refusal reasons,
    verdicts carry per-step status and error strings.
    """
    verdict: Optional[DestroyVerdict] = None
    refusal: Optional[DestroyRefusal] = None
    attempts: int = 0
    last_status_code: int = 0
    stop_error: str = ""
    verify_error: str = ""


# ---------------------------------------------------------------------------
# destroy_vastai_instance — amended signature
# ---------------------------------------------------------------------------


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
        (never both).

    Behaviour by credential state:
        - EXPLICITLY_DISABLED → refusal CREDENTIALS_DISABLED
        - ABSENT → refusal NO_CREDENTIALS (the v4 factory
          intercepts this and runs the CLI fallback)
        - AVAILABLE → REST path: ownership check via API, then
          belt-and-suspenders (stop → DELETE×retry → verify →
          re-destroy). Returns DESTROYED / LEAKED / UNKNOWN.

    CLI fallback is performed by the v4 factory, not by this
    adapter (the v3 contract keeps the adapter's credential
    handling simple and stateless).
    """
    resolution = credentials or read_vastai_api_key()
    if resolution.state == CredentialState.EXPLICITLY_DISABLED:
        return DestroyResult(refusal=DestroyRefusal.CREDENTIALS_DISABLED)
    if resolution.state == CredentialState.ABSENT:
        return DestroyResult(refusal=DestroyRefusal.NO_CREDENTIALS)
    # AVAILABLE: REST path.
    # ... ownership check via API, belt-and-suspenders, return
    # DestroyResult with verdict=DESTROYED/LEAKED/UNKNOWN ...
```

## `providers/vastai.py` — full module updates

The `VastaiProviderConfig`, `VastaiRunner`, `list_vastai_instances`,
and `build_vastai_cleanup_policy` shapes.

```python
# src/vastai_gpu_runner/providers/vastai.py
from __future__ import annotations

import json
import logging
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, Optional

from vastai_gpu_runner.cleanup_policy import (
    CleanupRefusal,
    CleanupResult,
    CleanupVerdict,
    InstanceCandidate,
    OwnershipPolicy,
    ProviderCleanupPolicy,
)
from vastai_gpu_runner.providers.destroy_adapters.vastai import (
    CredentialResolution,
    CredentialState,
    DestroyRefusal,
    DestroyResult,
    DestroyVerdict,
    destroy_vastai_instance,
    read_vastai_api_key,
    vastai_cmd,
    verify_instance_ownership,
)
from vastai_gpu_runner.runner import CloudRunner
from vastai_gpu_runner.ssh import scp_download, scp_upload, ssh_cmd
from vastai_gpu_runner.types import (
    CloudInstance,
    DeploymentConfig,
    InstanceStatus,
    Provider,
)

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "nvidia/cuda:12.4.0-devel-ubuntu22.04"
MIN_GPU_VRAM_MIB = 20_000

# States that are already-destroyed or otherwise terminal. Anything
# not in this set is processed by the cleanup policy. Using a negative
# allowlist (terminal-skip) is conservative: new Vast.ai states that
# are not yet terminal are processed, not silently skipped.
VASTAI_TERMINAL_STATES: frozenset[str] = frozenset({"destroyed"})


# ---------------------------------------------------------------------------
# VastaiProviderConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VastaiProviderConfig:
    """Canonical Vast.ai configuration shared by runner factory + cleanup policy.

    The runner factory (``VastaiRunner.from_config``) reads
    ``ownership``, ``credentials``, ``docker_image``, and
    ``setup_commands`` from this. The cleanup-policy factory
    (``build_vastai_cleanup_policy``) reads only ``ownership`` and
    ``credentials`` — the deployment-image invariant does not apply
    to listing/cleanup-only commands.

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
        resolved_image = DEFAULT_IMAGE if docker_image is None else docker_image
        return cls(
            docker_image=resolved_image,
            ownership=OwnershipPolicy(owned_images=owned_images),
            credentials=read_vastai_api_key(),
        )


# ---------------------------------------------------------------------------
# VastaiRunner — full shape (delegates destroy to v3 adapter)
# ---------------------------------------------------------------------------


class VastaiRunner(CloudRunner):
    """Vast.ai marketplace runner with hardened deployment.

    Args:
        config: Deployment configuration.
        ownership: Ownership policy. The runner and the cleanup
            adapter both call ``ownership.matches(image_ref)``.
        credentials: Pre-resolved credential state. When None,
            the runner uses ``read_vastai_api_key()`` (the v3
            back-compat path).
        docker_image: Docker image to use for new instances.
        min_gpu_vram_mib: Minimum GPU VRAM required (default 20 GB).
        setup_commands: Optional pre-instance setup commands.

    Note: ``allowed_images`` is a deprecated back-compat alias
    that builds an ``OwnershipPolicy`` from the given set. Simultaneous
    ``ownership=`` and ``allowed_images=`` raises ``ValueError`` —
    silently ignoring one destruction-safety configuration is unsafe.
    """

    def __init__(
        self,
        config: DeploymentConfig | None = None,
        *,
        ownership: OwnershipPolicy | None = None,
        credentials: CredentialResolution | None = None,
        allowed_images: frozenset[str] | None = None,  # DEPRECATED
        docker_image: str = DEFAULT_IMAGE,
        min_gpu_vram_mib: int = MIN_GPU_VRAM_MIB,
        setup_commands: list[str] | None = None,
    ) -> None:
        if ownership is not None and allowed_images is not None:
            raise ValueError(
                "VastaiRunner: supply either ownership= or allowed_images= "
                "(deprecated), not both — silently ignoring one destruction-"
                "safety configuration is unsafe."
            )
        if ownership is None and allowed_images is not None:
            warnings.warn(
                "VastaiRunner(allowed_images=...) is deprecated; "
                "build an OwnershipPolicy and pass ownership= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            ownership = OwnershipPolicy(owned_images=frozenset(allowed_images))
        super().__init__(config)
        self.ownership = ownership
        self.credentials = credentials
        # Back-compat read access for existing callers.
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

        Preserves the canonical ``OwnershipPolicy`` and
        ``CredentialResolution`` instances unchanged — the runner
        and the cleanup policy will call ``ownership.matches()`` and
        use ``credentials`` from the same instances.
        """
        return cls(
            ownership=canonical.ownership,
            credentials=canonical.credentials,
            docker_image=canonical.docker_image,
            min_gpu_vram_mib=canonical.min_gpu_vram_mib,
            setup_commands=list(canonical.setup_commands),
        )

    # ... wait_for_boot, verify_gpu, deploy_files, setup_environment,
    # launch_worker, check_progress, list_remote_files, download_file,
    # capture_deploy_failure_diagnostics: unchanged from v2 ...

    def destroy_instance(self, instance: CloudInstance) -> bool:
        """Destroy a Vast.ai instance — delegates entirely to the v3 adapter.

        The v2 implementation did inline ownership pre-check +
        CLI destroy + belt-and-suspenders REST destroy + always
        returned True. The v3 design moves all of that into
        ``destroy_vastai_instance`` (the adapter); this method is
        a single adapter call with no inline destroy logic.

        Returns True iff the adapter reports ``verdict=DESTROYED``.
        LEAKED, UNKNOWN, and all refusals return False.
        """
        result = destroy_vastai_instance(
            instance.instance_id,
            ownership=self.ownership or OwnershipPolicy(),
            credentials=self.credentials,
        )
        if result.verdict == DestroyVerdict.DESTROYED:
            instance.status = InstanceStatus.DESTROYED
            return True
        return False


# ---------------------------------------------------------------------------
# verify_instance_ownership — uses OwnershipPolicy.matches()
# ---------------------------------------------------------------------------


def verify_instance_ownership(
    instance_id: str,
    *,
    ownership: OwnershipPolicy,
) -> bool:
    """Check that a Vast.ai instance belongs to the caller before destruction.

    Uses ``ownership.matches(image_uuid)`` for the ownership check.
    This is the CLI-auth verification used by the v4 factory's
    CLI fallback path (separate auth context from the REST path).
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
    for inst in instances:
        if str(inst.get("id")) == str(instance_id):
            image = str(inst.get("image_uuid", ""))
            if ownership.matches(image):
                return True
            logger.error(
                "BLOCKED: instance %s belongs to another project (image=%s). Will NOT destroy.",
                instance_id,
                image,
            )
            return False
    logger.info("Instance %s not found in account (already destroyed?)", instance_id)
    return True


# ---------------------------------------------------------------------------
# list_vastai_instances — validates instance_id non-empty
# ---------------------------------------------------------------------------


def list_vastai_instances() -> list[InstanceCandidate]:
    """Read-only enumeration of Vast.ai instances on this account.

    Returns ``InstanceCandidate`` records. Records with missing
    or empty ``instance_id`` are skipped (logged). A failure to
    enumerate returns an empty list.

    NB: this helper does NOT enforce the EXPLICITLY_DISABLED
    short-circuit. The factory wraps this function with the
    short-circuit so the policy's ``list_instances()`` honors the
    v3 contract.
    """
    candidates: list[InstanceCandidate] = []
    try:
        raw = vastai_cmd(["show", "instances", "--raw"], timeout=15)
        instances = json.loads(raw)
    except (RuntimeError, json.JSONDecodeError) as exc:
        logger.warning("list_vastai_instances: enumeration failed: %s", exc)
        return candidates
    for inst in instances:
        instance_id = str(inst.get("id", "")).strip()
        if not instance_id:
            logger.warning(
                "Skipping instance with missing ID: %s",
                {k: inst.get(k) for k in ("label", "image_uuid", "actual_status")},
            )
            continue
        try:
            image_uuid = str(inst.get("image_uuid", ""))
            candidates.append(
                InstanceCandidate(
                    provider=Provider.VASTAI,
                    instance_id=instance_id,
                    image_uuid=image_uuid,
                    ownership_key=image_uuid,  # Vast.ai: image_uuid is the ownership key
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


# ---------------------------------------------------------------------------
# build_vastai_cleanup_policy — the v4 factory
# ---------------------------------------------------------------------------


def build_vastai_cleanup_policy(
    *,
    ownership: OwnershipPolicy,
    credentials: CredentialResolution,
) -> ProviderCleanupPolicy:
    """Build a Vast.ai cleanup policy from the canonical objects.

    The two arguments are the same ``OwnershipPolicy`` and
    ``CredentialResolution`` instances that the runner was
    constructed with (the batch command extracts them from the
    ``VastaiProviderConfig``; the cleanup command constructs them
    directly from CLI args).

    The wired ``list_instances_fn`` enforces the v3 contract:
    ``EXPLICITLY_DISABLED`` credentials return ``[]`` without
    invoking CLI enumeration (no API key, no CLI fallback — fail-closed).

    The wired ``destroy_fn`` runs:
        1. Eligibility check (skip terminal states like ``destroyed``).
        2. ``destroy_vastai_instance`` with the canonical ownership + credentials.
        3. If the adapter returns ``NO_CREDENTIALS``, run the CLI
           fallback path: CLI ownership verification → CLI destroy →
           ``CLI_ATTEMPTED`` (destruction unconfirmed) or ``UNKNOWN``.
        4. Translate all three v3 verdicts and all three v3 refusals
           into the v4 ``CleanupResult`` shape, with diagnostic text
           built from the structured ``DestroyResult`` fields.
    """
    provider = Provider.VASTAI

    def _list_instances() -> list[InstanceCandidate]:
        # v3 contract: EXPLICITLY_DISABLED short-circuits before
        # CLI enumeration. The CLI fallback path uses file
        # credentials which would silently bypass the explicit
        # disable — fail-closed.
        if credentials.state == CredentialState.EXPLICITLY_DISABLED:
            logger.warning(
                "Zombie sweep disabled: VASTAI_API_KEY is explicitly empty; "
                "provider enumeration was not attempted."
            )
            return []
        return list_vastai_instances()

    def _describe(result: DestroyResult) -> str:
        """Build diagnostic text from v3 DestroyResult structured fields."""
        return (
            f"attempts={result.attempts}, "
            f"last_status={result.last_status_code}, "
            f"verify_error={result.verify_error!r}, "
            f"stop_error={result.stop_error!r}"
        )

    def _cli_fallback(candidate: InstanceCandidate) -> CleanupResult:
        """CLI fallback path for ABSENT credentials.

        The v3 adapter returns NO_CREDENTIALS when the API key is
        not configured. We then verify ownership via the CLI auth
        context (which uses the file-based key), and if owned, run
        the CLI destroy. CLI destruction is unconfirmed — we report
        CLI_ATTEMPTED, not DESTROYED.
        """
        try:
            if not verify_instance_ownership(
                candidate.instance_id,
                ownership=ownership,
            ):
                return CleanupResult(
                    refusal=CleanupRefusal.OWNERSHIP,
                    error=(
                        "CLI ownership check rejected "
                        f"{candidate.instance_id!r}"
                    ),
                )
        except Exception as exc:
            return CleanupResult(
                verdict=CleanupVerdict.UNKNOWN,
                error=(
                    f"CLI ownership verification raised "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        try:
            vastai_cmd(["destroy", "instance", candidate.instance_id], timeout=15)
            return CleanupResult(
                verdict=CleanupVerdict.CLI_ATTEMPTED,
                error=(
                    "CLI fallback ran; destruction not confirmed via REST"
                ),
            )
        except Exception as exc:
            return CleanupResult(
                verdict=CleanupVerdict.UNKNOWN,
                error=f"CLI destroy raised {type(exc).__name__}: {exc}",
            )

    def _destroy(candidate: InstanceCandidate) -> CleanupResult:
        # 1. Eligibility check (negative allowlist: skip terminal states).
        if not candidate.state:
            return CleanupResult(
                refusal=CleanupRefusal.INELIGIBLE_STATE,
                error="empty state",
            )
        if candidate.state in VASTAI_TERMINAL_STATES:
            return CleanupResult(
                refusal=CleanupRefusal.INELIGIBLE_STATE,
                error=f"state {candidate.state!r} is terminal (already destroyed)",
            )
        # 2. CREDENTIALS_DISABLED: refuse without provider calls.
        # (Should already be short-circuited at enumeration, but
        # belt-and-suspenders in case a candidate was created by
        # another path.)
        if credentials.state == CredentialState.EXPLICITLY_DISABLED:
            return CleanupResult(
                refusal=CleanupRefusal.CREDENTIALS_DISABLED,
                error="VASTAI_API_KEY explicitly empty",
            )
        # 3. Delegate to v3 adapter with the canonical ownership + credentials.
        result = destroy_vastai_instance(
            candidate.instance_id,
            ownership=ownership,
            credentials=credentials,
        )
        # 4. Translate refusals first.
        if result.refusal == DestroyRefusal.NO_CREDENTIALS:
            return _cli_fallback(candidate)
        if result.refusal == DestroyRefusal.OWNERSHIP:
            return CleanupResult(
                refusal=CleanupRefusal.OWNERSHIP,
                error=(
                    f"v3 adapter refused: ownership check rejected "
                    f"{candidate.instance_id!r}"
                ),
            )
        if result.refusal == DestroyRefusal.CREDENTIALS_DISABLED:
            return CleanupResult(
                refusal=CleanupRefusal.CREDENTIALS_DISABLED,
                error=(
                    "v3 adapter refused: VASTAI_API_KEY explicitly empty"
                ),
            )
        # 5. Translate verdicts exhaustively.
        if result.verdict == DestroyVerdict.DESTROYED:
            return CleanupResult(verdict=CleanupVerdict.DESTROYED)
        if result.verdict == DestroyVerdict.LEAKED:
            return CleanupResult(
                verdict=CleanupVerdict.LEAKED,
                error=_describe(result),
            )
        return CleanupResult(
            verdict=CleanupVerdict.UNKNOWN,
            error=_describe(result),
        )

    return ProviderCleanupPolicy(
        provider=provider,
        ownership=ownership,
        list_instances_fn=_list_instances,
        destroy_fn=_destroy,
    )
```

## Orchestrator wiring

The orchestrator's `_sweep_zombies` is policy-driven end-to-end.
The orchestrator logs every non-`DESTROYED` outcome at severity
matching operational impact.

```python
# src/vastai_gpu_runner/batch.py (changes to _sweep_zombies)
# NB: `logger` is the module-level logger already defined at the
# top of batch.py (`logger = logging.getLogger(__name__)`). This
# block shows only the new/changed logic.

def _sweep_zombies(self) -> int:
    """Destroy orphaned instances not tracked by live_runners.

    Routes through the cleanup policy:
    1. Enumerate instances via ``policy.list_instances()`` (which
       may short-circuit on EXPLICITLY_DISABLED).
    2. Filter by label prefix (orchestrator's per-batch scope).
    3. Exclude tracked IDs (the existing semantics).
    4. For every remaining candidate, call ``policy.destroy(candidate)``.
    5. Count ``verdict == DESTROYED`` outcomes.
    6. Log every non-DESTROYED outcome with severity matching
       operational impact.
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
        if result.verdict == CleanupVerdict.DESTROYED:
            killed += 1
            continue
        # Severity-by-outcome logging.
        if result.verdict == CleanupVerdict.LEAKED:
            logger.error(
                "Zombie sweep: %s LEAKED — manual review required: %s",
                candidate.instance_id,
                result.error,
            )
        elif result.verdict == CleanupVerdict.UNKNOWN:
            logger.warning(
                "Zombie sweep: %s outcome=UNKNOWN: %s",
                candidate.instance_id,
                result.error,
            )
        elif result.verdict == CleanupVerdict.CLI_ATTEMPTED:
            logger.warning(
                "Zombie sweep: %s CLI fallback attempted (destruction not confirmed): %s",
                candidate.instance_id,
                result.error,
            )
        elif result.refusal == CleanupRefusal.CREDENTIALS_DISABLED:
            logger.warning(
                "Zombie sweep: %s refused (credentials disabled): %s",
                candidate.instance_id,
                result.error,
            )
        elif result.refusal in (
            CleanupRefusal.OWNERSHIP,
            CleanupRefusal.INELIGIBLE_STATE,
            CleanupRefusal.PROVIDER_MISMATCH,
        ):
            logger.info(
                "Zombie sweep: %s refused (%s): %s",
                candidate.instance_id,
                result.refusal.value,
                result.error,
            )
        elif result.refusal == CleanupRefusal.NO_CREDENTIALS:
            # Should not reach here — the factory's CLI fallback
            # intercepts NO_CREDENTIALS. Logged for visibility.
            logger.info(
                "Zombie sweep: %s refused (no credentials): %s",
                candidate.instance_id,
                result.error,
            )
    if killed:
        logger.info("Zombie sweep: destroyed %d instance(s)", killed)
    return killed
```

The orchestrator's `__init__` requires `cleanup_policy: ProviderCleanupPolicy`
and `runner_factory: RunnerFactory`.

## CLI wiring

The CLI is the one place that builds the canonical config (for the
runner) and the two canonical objects (for the cleanup policy). It
threads both into the v3 + v4 entry points.

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
    base = VastaiProviderConfig.from_env()
    config = replace(
        base,
        docker_image=image,
        ownership=OwnershipPolicy(owned_images=frozenset({image})),
    )

    # Step 2: runner factory reads from the same config.
    runner_factory = lambda: VastaiRunner.from_config(config)  # noqa: E731

    # Step 3: cleanup policy reads ownership + credentials from the
    # same config (no deployment-image wrapper needed).
    cleanup_policy = build_vastai_cleanup_policy(
        ownership=config.ownership,
        credentials=config.credentials,
    )

    orch = MyOrchestrator(
        runner_factory=runner_factory,
        cleanup_policy=cleanup_policy,
        label_prefix=label,
    )
    orch.run()
```

The existing `cli.py:cleanup` command migrates to use the new API
without constructing a `VastaiProviderConfig` (no deployment image):

```python
# src/vastai_gpu_runner/cli.py (cleanup command)
@app.command()
def cleanup(
    label_prefix: Annotated[str, typer.Option("--label", "-l")] = ...,
    allowed_images: Annotated[
        str | None,
        typer.Option(
            "--allowed-images",  # canonical
            "--owned-images",     # alias
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
    from vastai_gpu_runner.providers.destroy_adapters.vastai import (
        read_vastai_api_key,
    )
    from vastai_gpu_runner.providers.vastai import build_vastai_cleanup_policy

    console = Console()
    ownership = (
        OwnershipPolicy(owned_images=frozenset(allowed_images.split(",")))
        if allowed_images
        else OwnershipPolicy()
    )
    cleanup_policy = build_vastai_cleanup_policy(
        ownership=ownership,
        credentials=read_vastai_api_key(),
    )

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

Seven steps. Each step is independently testable but the rollout
lands as one PR because the intermediate states are not stable.

1. **Add canonical credential + ownership-policy types + invariants + contract tests.**
   - `cleanup_policy.py:OwnershipPolicy` (frozen, `matches(image_ref)` with the v3 `_repository` helper; declared `_normalised` cache field).
   - `providers/destroy_adapters/vastai.py:CredentialState` (StrEnum) + `CredentialResolution` (frozen dataclass with `state` and `key` fields).
   - Property tests: `OwnershipPolicy.matches` is reflexive, tag-insensitive, sha256-by-repo, registry-port-aware, fail-closed on empty sets and malformed references.
   - `CredentialResolution` invariants: AVAILABLE requires non-empty pre-stripped key; ABSENT and EXPLICITLY_DISABLED require empty key.

2. **Add v3 `DestroyResult` / `DestroyVerdict` / `DestroyRefusal` types + amend adapter signature.**
   - `providers/destroy_adapters/vastai.py:DestroyVerdict` (StrEnum: `DESTROYED | LEAKED | UNKNOWN`).
   - `providers/destroy_adapters/vastai.py:DestroyRefusal` (StrEnum: `OWNERSHIP | NO_CREDENTIALS | CREDENTIALS_DISABLED`).
   - `providers/destroy_adapters/vastai.py:DestroyResult` (frozen dataclass: `verdict | refusal`, `attempts`, `last_status_code`, `stop_error`, `verify_error` — no `error` field).
   - `destroy_vastai_instance(instance_id, *, ownership: OwnershipPolicy, credentials: CredentialResolution | None = None) -> DestroyResult`. Back-compat: when `credentials=None`, calls `read_vastai_api_key()`.
   - Contract tests: ownership match uses `OwnershipPolicy.matches()`; credential state drives refusal type; AVAILABLE runs the REST path; ABSENT returns `NO_CREDENTIALS`; EXPLICITLY_DISABLED returns `CREDENTIALS_DISABLED`.

3. **Add Vast.ai runner + cleanup adapter.**
   - `providers/vastai.py:VastaiProviderConfig` (frozen, with `__post_init__` invariants).
   - `providers/vastai.py:VastaiRunner.from_config(canonical)` — preserves `ownership` and `credentials`.
   - `VastaiRunner.__init__` adds `ownership: OwnershipPolicy | None` + `credentials: CredentialResolution | None` parameters; rejects simultaneous `ownership=` + `allowed_images=` with `ValueError`; `allowed_images` becomes a deprecated alias.
   - `VastaiRunner.destroy_instance` is a single `destroy_vastai_instance(...)` adapter call (no v2 regression).
   - `verify_instance_ownership(instance_id, *, ownership: OwnershipPolicy)` — replaces `_image_is_allowed`.
   - `list_vastai_instances()` returns `list[InstanceCandidate]`, validates `instance_id` non-empty.
   - `VASTAI_TERMINAL_STATES: frozenset[str]` module constant.
   - `build_vastai_cleanup_policy(*, ownership: OwnershipPolicy, credentials: CredentialResolution) -> ProviderCleanupPolicy` — wires list + destroy callbacks.
   - Adapter tests:
     - EXPLICITLY_DISABLED: `list_instances()` returns `[]` without invoking `vastai_cmd`.
     - ABSENT CLI fallback: v3 returns `NO_CREDENTIALS` → factory runs CLI ownership verification + CLI destroy → `CLI_ATTEMPTED` (not `DESTROYED`) on command success; `UNKNOWN` on command failure.
     - AVAILABLE: v3 returns `DESTROYED` → `verdict=DESTROYED`.
     - `verdict=LEAKED` / `verdict=UNKNOWN` translate correctly with `_describe(result)` diagnostic.
     - `refusal=OWNERSHIP` / `NO_CREDENTIALS` / `CREDENTIALS_DISABLED` translate correctly.
     - INELIGIBLE_STATE: candidate.state in `VASTAI_TERMINAL_STATES` → `refusal=INELIGIBLE_STATE`.
     - PROVIDER_MISMATCH: handled by the policy's `destroy()` method, not the factory.
     - `list_vastai_instances()` skips records with empty `instance_id`.

4. **Add orchestrator support behind a fail-closed compatibility path.**
   - `BatchOrchestrator.__init__` accepts `cleanup_policy: ProviderCleanupPolicy` (required).
   - `_sweep_zombies` is policy-driven; logs every non-`DESTROYED` outcome at severity matching operational impact.
   - Orchestrator tests:
     - Severity logging: `LEAKED` = `ERROR`, `UNKNOWN` / `CLI_ATTEMPTED` / `CREDENTIALS_DISABLED` = `WARNING`, refusals = `INFO` (verified with `caplog`).
     - `_sweep_zombies` continues on `destroy_fn` exceptions (they return `verdict=UNKNOWN` with `type(exc).__name__: exc`).
     - `_sweep_zombies` does NOT import provider modules (verified by `inspect.getsource`).

5. **Update every composition root, subclass, and existing CLI command.**
   - `cli.py:batch`: build `VastaiProviderConfig` via `from_env()` + `replace()`, pass to `VastaiRunner.from_config` and `build_vastai_cleanup_policy(ownership, credentials)`.
   - `cli.py:cleanup`: build `OwnershipPolicy` and `CredentialResolution` directly (no `VastaiProviderConfig`), pass to `build_vastai_cleanup_policy`.
   - `cli.py:instances`: use `list_vastai_instances()` for the table. `--allowed-images` canonical, `--owned-images` alias.
   - `BatchOrchestrator` subclasses: update composition to supply `cleanup_policy`.
   - Test fixtures: `VastaiProviderConfig` + `OwnershipPolicy` + `CredentialResolution` factory fixtures.

6. **Add integration tests.**
   - `tests/integration/test_cleanup_policy_integration.py`:
     - **disabled-before-enumeration**: `credentials=EXPLICITLY_DISABLED` → `policy.list_instances()` returns `[]`; `vastai_cmd` was not called (verified by mock call count).
     - **absent-credential CLI fallback**: `credentials=ABSENT` → `policy.destroy(candidate)` returns `verdict=CLI_ATTEMPTED` (NOT `DESTROYED`); the canonical `ABSENT` resolution was passed to the REST adapter (NOT `None`).
     - **empty ownership set**: `ownership=OwnershipPolicy(owned_images=frozenset())` → `verdict=...` → `OWNERSHIP` (fail-closed).
     - **provider mismatch**: candidate `Provider.RUNPOD` to a Vast.ai policy → `PROVIDER_MISMATCH` with operator-friendly diagnostic.
     - **enumeration failure**: `list_vastai_instances()` raises → `policy.list_instances()` returns `[]`.
     - **ineligible state**: candidate `state="destroyed"` → `INELIGIBLE_STATE` without invoking `destroy_vastai_instance`.
     - **severity logging**: orchestrator logs `LEAKED` at `ERROR`, `UNKNOWN` at `WARNING`, refusals at `INFO` (verified with `caplog`).
     - **non-empty error from empty exception**: `raise RuntimeError()` (no message) → orchestrator logs with non-empty error containing `RuntimeError: `.
     - **empty instance_id in enumeration**: malformed Vast.ai record skipped, not passed to destroy.

7. **Delete legacy sweep + duplicated helpers after a repository-wide caller audit.**
   - `audit_caller_sites.sh` (run before deletion): grep for external callers of `orchestrator.sweep_zombie_instances`, `orchestrator.load_vastai_api_key`, `VastaiRunner.allowed_images` (read-only external use), `providers.vastai._image_is_allowed`. Update external callers.
   - Delete `orchestrator.sweep_zombie_instances` (v3 deferral reaffirmed).
   - Delete `orchestrator.load_vastai_api_key` (v3 deferral reaffirmed).
   - Delete `providers/vastai.py:_image_is_allowed`.
   - Delete direct `vastai_cmd(["show", "instances", "--raw"])` parsing in `cli.py:cleanup` and `cli.py:instances`.
   - Update `tests/test_orchestrator.py` and `tests/test_batch.py` to mock `cleanup_policy.list_instances` and `cleanup_policy.destroy`.

## Test plan

- `tests/test_cleanup_policy.py`:
  - `_repository`: 13+ cases (digest, registry ports, malformed, empty, whitespace).
  - `OwnershipPolicy.matches`: reflexive, tag-insensitive, sha256-by-repo, registry-port-aware, fail-closed on empty sets, malformed reference rejection.
  - `OwnershipPolicy._normalised`: declared field; precomputed in `__post_init__`; `matches` is O(1) per call.
  - `ProviderCleanupPolicy.list_instances`: returns wired list; catches and returns `[]` on exception.
  - `ProviderCleanupPolicy.destroy`: provider mismatch returns `PROVIDER_MISMATCH`; catches `destroy_fn` exceptions with `f"{type(exc).__name__}: {exc}"` (non-empty even for `RuntimeError()` with no message).
  - `CleanupResult` invariants: 5 cases (verdict/refusal exclusivity, error non-empty on non-DESTROYED, error empty on DESTROYED).
  - `InstanceCandidate.__post_init__`: empty `instance_id` raises; whitespace-only `instance_id` raises.
- `tests/test_providers_vastai.py`:
  - `VastaiRunner.from_config` round-trips with `VastaiProviderConfig` (constructor parity).
  - `VastaiRunner(allowed_images=frozenset({img}))` (deprecated) emits `DeprecationWarning` + builds equivalent `OwnershipPolicy`.
  - Simultaneous `ownership=` and `allowed_images=` raises `ValueError`.
  - `verify_instance_ownership` uses `OwnershipPolicy.matches()`.
  - `list_vastai_instances` returns `list[InstanceCandidate]` with `gpu_model` + `cost_per_hour` populated; skips records with empty `instance_id`; returns `[]` on API error.
  - `build_vastai_cleanup_policy(*, ownership, credentials)`:
    - EXPLICITLY_DISABLED list: returns `[]` without `vastai_cmd`.
    - ABSENT destroy: factory intercepts `NO_CREDENTIALS` → CLI ownership verify → CLI destroy → `CLI_ATTEMPTED` (success) or `UNKNOWN` (failure).
    - AVAILABLE destroy: v3 DESTROYED → `verdict=DESTROYED`.
    - LEAKED / UNKNOWN translate with `_describe(result)` diagnostic.
    - OWNERSHIP / NO_CREDENTIALS / CREDENTIALS_DISABLED translate correctly.
    - INELIGIBLE_STATE for `state in VASTAI_TERMINAL_STATES` or empty state.
- `tests/test_batch.py`:
  - `_sweep_zombies` calls `cleanup_policy.list_instances()` exactly once.
  - `_sweep_zombies` calls `cleanup_policy.destroy(candidate)` for every label-matching, untracked candidate.
  - `_sweep_zombies` counts only `verdict=DESTROYED` outcomes.
  - `_sweep_zombies` logs `LEAKED` at `ERROR`, `UNKNOWN` / `CLI_ATTEMPTED` / `CREDENTIALS_DISABLED` at `WARNING`, refusals at `INFO` (`caplog`).
  - `_sweep_zombies` continues on `destroy_fn` exceptions.
  - `_sweep_zombies` does NOT import provider modules (`inspect.getsource`).
- `tests/integration/test_cleanup_policy_integration.py` — 9 scenarios from step 6 above.

## Backwards compatibility

The `VastaiRunner.__init__(allowed_images=..., docker_image=..., ...)`
constructor is preserved as a deprecated back-compat path (emits
`DeprecationWarning`). The CLI's new `batch` subcommand uses
`from_config`; the old programmatic path requires a `cleanup_policy`
to be supplied.

The `orchestrator.py:sweep_zombie_instances` function is deleted in
v4 step 7. External callers must construct a `ProviderCleanupPolicy`
and call `policy.destroy(candidate)`.

The `cli.py:cleanup` and `cli.py:instances` commands are refactored
in v4 step 5. The `--allowed-images` flag is preserved as canonical;
`--owned-images` is added as a documented alias. The rename is
deferred to a separate CLI compatibility change.

## Out of scope

- **RunPod adapter.** The factory ships when the RunPod adapter
  ships (roadmap item 2). The `ProviderCleanupPolicy` interface is
  provider-agnostic from day one.
- **Hostile detection.** Removed in the v4 second-pass review.
  Hostile detection is deferred to a separate audit stage that
  emits alerts without changing the destroy decision.
- **Dispute webhook.** Future work.
- **Bulk-destroy optimisation.** YAGNI for now.
- **Cross-provider zombie sweep.** A single orchestrator supports
  one `cleanup_policy`. Multi-policy is future design.

## Review process

This is the fourth review pass on the v4 architecture. The first
draft was rejected with 5 BLOCKERs + 7 CONCERNs. The second draft
was rejected with 7 BLOCKERs + 2 CONCERNs + 3 NITs. The third draft
was rejected with 6 BLOCKERs + 4 CONCERNs + 2 NITs. This fourth
draft addresses every finding.

The fourth review prompt for ChatGPT-with-GitHub-plugin:

> Review the v4 architecture design at PR #22 (file:
> docs/architecture-v4-cleanup-policy.md) against the v3 design at
> docs/architecture-v3.md and the current code at
> src/vastai_gpu_runner/{batch,orchestrator,runner,cli}.py and
> src/vastai_gpu_runner/providers/vastai.py. The v4 design
> resolves issue #19.
>
> The third draft was rejected with 6 BLOCKERs, 4 CONCERNs, and
> 2 NITs. Verify each finding is addressed:
>
> 1. **BLOCKER 1 (ABSENT CLI fallback)**: confirm the factory
>    intercepts `DestroyRefusal.NO_CREDENTIALS` and runs CLI
>    ownership verification + CLI destroy, returning
>    `CLI_ATTEMPTED` (not `DESTROYED`) on command success.
> 2. **BLOCKER 2 (v3 translation)**: confirm `CLI_ATTEMPTED` is
>    NOT read from `DestroyVerdict`; confirm `_describe(result)`
>    builds diagnostic text from `attempts`, `last_status_code`,
>    `stop_error`, `verify_error`.
> 3. **BLOCKER 3 (VastaiRunner delegation)**: confirm
>    `VastaiRunner.destroy_instance` is a single
>    `destroy_vastai_instance(...)` call; confirm `from_config`
>    preserves both `ownership` and `credentials`; confirm
>    simultaneous `ownership=` + `allowed_images=` raises
>    `ValueError`.
> 4. **BLOCKER 4 (non-empty catch error)**: confirm the catch
>    uses `f"{type(exc).__name__}: {exc}"` so the error is
>    always non-empty.
> 5. **BLOCKER 5 (build_vastai_cleanup_policy direct args)**:
>    confirm the factory takes `(*, ownership, credentials)`
>    directly (no `VastaiProviderConfig` wrapper); confirm the
>    cleanup CLI command does not construct a `VastaiProviderConfig`.
> 6. **BLOCKER 6 (instance_id validation)**: confirm
>    `list_vastai_instances` skips records with empty
>    `instance_id`; confirm `InstanceCandidate.__post_init__`
>    raises `ValueError` for empty or whitespace-only IDs.
> 7. **CONCERN 7 (_normalised declared field)**: confirm
>    `_normalised` is a `field(init=False, repr=False, compare=False)`.
> 8. **CONCERN 8 (StrEnum for CredentialState)**: confirm
>    `CredentialState(StrEnum)` (not plain `Enum`).
> 9. **CONCERN 9 (negative allowlist)**: confirm
>    `VASTAI_TERMINAL_STATES = frozenset({"destroyed"})` (skip
>    only terminal states); confirm new states are processed.
> 10. **CONCERN 10 (severity logging)**: confirm `LEAKED` =
>     `ERROR`, `UNKNOWN` / `CLI_ATTEMPTED` / `CREDENTIALS_DISABLED`
>     = `WARNING`, refusals = `INFO`.
> 11. **NIT 11 (canonical flag)**: confirm `--allowed-images` is
>     declared first in `typer.Option`; the Python parameter is
>     `allowed_images`.
> 12. **NIT 12 (is_candidate historical)**: confirm the historical
>     change summary no longer mentions `is_candidate`.
>
> Additionally, identify any new BLOCKERs or CONCERNs introduced
> by the fourth draft. Focus on:
> - The factory's `_cli_fallback` ownership check: does it use
>   `verify_instance_ownership` correctly? Does the failure path
>   produce a structured refusal with non-empty error?
> - The `_describe(result)` helper: does it handle all four
>   structured fields correctly? Are empty fields handled
>   gracefully?
> - The `InstanceCandidate.__post_init__` invariant: does it
>   correctly reject empty strings AND whitespace-only strings?
> - The `OwnershipPolicy._normalised` declared field: does the
>   `field(default=None)` correctly handle the case where
>   `__post_init__` doesn't override it (None ownership)?
> - The `build_vastai_cleanup_policy` signature change from
>   `(canonical)` to `(*, ownership, credentials)`: is the
>   batch command's extraction correct? Is the cleanup command's
>   direct construction correct?
> - The orchestrator's severity logging: does it correctly handle
>   the `verdict=None` and `refusal=None` case (defensive)?
> - The CLI's `--allowed-images` canonical declaration: does
>   `typer.Option("--allowed-images", "--owned-images", ...)`
>   accept both flags in the right order?
>
> Return a labeled list of findings. Each finding is one of:
> BLOCKER (must fix before merge), CONCERN (should fix, but not
> blocking), or NIT (nice to have). For each finding, give the
> exact line range, the issue, and the proposed fix. If the
> design is acceptable as-is, say "DESIGN ACCEPTED" with a
> one-line rationale.