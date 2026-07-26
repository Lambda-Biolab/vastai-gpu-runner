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

The v4 design **does not redefine any v3 types**. The v3 module
`providers/destroy.py` is a hard prerequisite: it owns
`DestroyVerdict`, `DestroyRefusal`, `DestroyResult`, `VerifyVerdict`,
`VerifyResult`, `DestroyPolicy`, the callback protocols, and
`belt_and_suspenders`. The v4 implementation either lands v3 first
or merges v3 as part of the same PR. The v4 doc references v3
verbatim and only amends `destroy_vastai_instance`'s signature.

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
matching its operational impact (`LEAKED` = `ERROR`, unexpected
`NO_CREDENTIALS` = `WARNING`, `UNKNOWN` / `CLI_ATTEMPTED` /
`CREDENTIALS_DISABLED` = `WARNING`, refusals = `INFO`). The
`VastaiRunner.destroy_instance` method delegates entirely to the v3
adapter (no v2 regression) and logs the typed `DestroyResult` for
non-`DESTROYED` outcomes. The `cli.py:instances` command's "Owned"
column uses `OwnershipPolicy.matches()` (the v2 substring/prefix
match is removed).

Diff vs v3 once v3 is implemented:

- **+** `src/vastai_gpu_runner/cleanup_policy.py` — `ProviderCleanupPolicy` (frozen, `kw_only`), `InstanceCandidate` (frozen, non-empty `instance_id` invariant), `CleanupVerdict` enum (`DESTROYED | CLI_ATTEMPTED | LEAKED | UNKNOWN`), `CleanupRefusal` enum (`OWNERSHIP | NO_CREDENTIALS | CREDENTIALS_DISABLED | INELIGIBLE_STATE | PROVIDER_MISMATCH`), `CleanupResult` (typed return with `__post_init__` invariants), `OwnershipPolicy` (frozen, `matches(image_ref)`, declared `_normalised` cache field, narrowed for strict type checking).
- **+** `src/vastai_gpu_runner/providers/vastai.py:VastaiProviderConfig` (frozen, owns `ownership: OwnershipPolicy`, `credentials: CredentialResolution`, `docker_image`, etc.)
- **+** `src/vastai_gpu_runner/providers/vastai.py:build_vastai_cleanup_policy(*, ownership, credentials)` — provider-owned factory that takes the two canonical objects directly (no config wrapper)
- **+** `src/vastai_gpu_runner/providers/vastai.py:list_vastai_instances()` — read-only enumeration returning `list[InstanceCandidate]`, validates `instance_id` is a non-`None` non-empty string
- **+** `src/vastai_gpu_runner/providers/vastai.py:_describe_destroy_result(result)` — single shared diagnostic helper, used by both the runner and the factory; includes `verdict` + `refusal` in the output
- **~** `providers/destroy_adapters/vastai.py:destroy_vastai_instance` accepts `ownership: OwnershipPolicy` directly (replaces `allowed_images: frozenset[str]`); accepts `credentials: CredentialResolution | None` (defaults to `read_vastai_api_key()` for back-compat direct callers). The v3 implementation must adopt this signature.
- **~** `providers/destroy_adapters/vastai.py:read_vastai_api_key()` — v3 env-first + fail-closed semantics (env var first, then file fallback with warning + `ABSENT` on blank/unreadable file)
- **~** `VastaiRunner.__init__` accepts `ownership: OwnershipPolicy | None` and `credentials: CredentialResolution | None`; rejects simultaneous `ownership=` and deprecated `allowed_images=` with `ValueError`.
- **~** `VastaiRunner.from_config(config)` preserves both `canonical.ownership` and `canonical.credentials`.
- **~** `VastaiRunner.destroy_instance` delegates entirely to `destroy_vastai_instance(...)` — no v2-style inline ownership pre-check, no inline REST stop/delete/verify. Returns `bool` from the typed adapter result, logging the typed `DestroyResult` for non-`DESTROYED` outcomes.
- **~** `BatchOrchestrator.__init__` accepts `cleanup_policy: ProviderCleanupPolicy` (required). The orchestrator calls `policy.list_instances()` and `policy.destroy(candidate)` — never branches on `Provider`, never imports provider modules.
- **~** `BatchOrchestrator._sweep_zombies` is policy-driven end-to-end. The label-prefix filter and tracked-id exclusion stay on the orchestrator. Every other decision is delegated to `policy.destroy()`. The orchestrator logs every non-`DESTROYED` outcome at severity matching operational impact.
- **~** `cli.py:cleanup` and `cli.py:instances` — refactored to use the new API. The `--allowed-images` flag is the canonical primary; `--owned-images` is an alias. Empty `--allowed-images ""` is **fail-closed** (empty set rejects every image), not opt-out. The `instances` command's "Owned" column uses `OwnershipPolicy.matches()` (the v2 unsafe substring/prefix match is removed).
- **—** `orchestrator.py:sweep_zombie_instances` (v3 deletion) — reaffirmed.
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

## What changes vs the v4 first + second + third + fourth + fifth + sixth + seventh + eighth drafts

The first draft was rejected with 5 BLOCKERs and 7 CONCERNs. The
second draft was rejected with 7 BLOCKERs, 2 CONCERNs, and 3 NITs.
The third draft was rejected with 6 BLOCKERs, 4 CONCERNs, and 2 NITs.
The fourth draft was rejected with 5 BLOCKERs and 3 CONCERNs. The
fifth draft was rejected with 2 BLOCKERs, 2 CONCERNs, and 1 NIT. The
sixth draft was rejected with 1 BLOCKER, 4 CONCERNs, and 2 NITs. The
seventh draft was rejected with 1 BLOCKER and 1 NIT. The eighth
draft was rejected with 3 BLOCKERs, 2 CONCERNs, and 1 NIT.
This ninth draft addresses every finding.

### Applied from the 12th-pass review (this pass)

- **Tagged `OwnershipVerification` result replaces conflated `bool`.**
  `verify_instance_ownership` now returns an `Enum` (`OWNED`,
  `ABSENT`, `REFUSED`, `DISABLED`) instead of a `bool` that
  conflated "owned and present" with "absent and therefore safe".
  The CLI fallback path dispatches on the four cases:
  `OWNED`/`DISABLED` → CLI destroy → `CLI_ATTEMPTED`; `ABSENT`
  → short-circuit to a new v4 verdict `ALREADY_GONE` without
  invoking CLI destroy; `REFUSED` → `OWNERSHIP` refusal. The
  `CleanupVerdict` enum gains a new `ALREADY_GONE` value, logged
  at `INFO` by the orchestrator (no kill count, no destroy
  happened, but the end-state is achieved). (BLOCKER 1)
- **Shared `_normalize_instance_id` normalizer.** Single helper
  used by both `list_vastai_instances` and `verify_instance_ownership`,
  rejecting `None`/`bool`/empty/blank and canonicalising the rest
  (stripped string for `int` or `str` IDs). Ensures padded input
  on the caller side and padded value on the API side reduce to
  the same canonical form. The verifier validates and normalises
  the **entire** response before locating the requested record,
  so a malformed record after a matching owned record is
  detected (no early-return short-circuit). (BLOCKER 2)
- **Outermost `except Exception` boundary.** The verifier's
  declared contract ("unexpected exceptions → REFUSED") is now
  actually enforced: a top-level `except Exception` boundary
  converts any escaped exception (subprocess errors downstream
  of the explicit `(RuntimeError, JSONDecodeError)` clauses,
  `ownership.matches` failures, etc.) to `REFUSED` with an
  `ERROR`-level log. All failure-path log levels are now
  `ERROR` (was `WARNING` for the original two), so the operator
  sees "destroy refused" rather than "ambiguous state". (BLOCKER 3)
- **`_describe_destroy_result` for `CREDENTIALS_DISABLED`.** The
  refusal path that previously discarded the shared diagnostic
  now includes `_describe_destroy_result(result)` in the error
  string, keeping the v4 commitment that the helper is used
  across both verdict and refusal paths. (CONCERN)
- **Test catalogue expanded.** The eight tests that produced
  the prior verification pass are replaced by a parameterized
  enumeration (`None`/`""`/`"   "` valid-ID cases), malformed-
  record-BEFORE / malformed-record-AFTER ordering tests,
  numeric and padded ID cases, and `caplog` severity tests
  for every failure path. Plus a new factory test asserting
  that an `ABSENT` verification result short-circuits to
  `ALREADY_GONE` WITHOUT invoking `vastai destroy instance`.
  The "eight tests" count is dropped in favor of describing
  the cases exhaustively. (CONCERN)
- **`is_candidate` references replaced with the actual mechanism.**
  The review-process section now refers to "orchestrator
  label/tracked-ID filtering followed by the factory `_destroy`
  eligibility gate" — both of which exist in the v4 design
  — instead of the non-existent `is_candidate` short-circuit.
  (NIT)

### Applied from the 11th-pass review (prior pass)

- **`verify_instance_ownership` fails closed on malformed records.**
  Non-dict list entries, missing `id` keys, `id=None`, and
  empty/blank `id` strings all return `False` (refuse the destroy)
  rather than being skipped. The function's return contract now
  states: `True` only when ownership is disabled OR a fully-well-
  formed response does not contain the requested instance;
  `False` for any API error, response-shape failure, or
  unparseable record that prevents proving absence. Eight new
  test cases in `tests/test_providers_vastai.py` cover the
  fail-closed paths plus the still-`True` "already destroyed"
  path. (BLOCKER 1)

### Applied from the 9th-pass review (prior pass)

- **Removed the v4 `providers/destroy.py` block.** The doc no longer claims to add or redefine `DestroyVerdict` / `DestroyRefusal` / `DestroyResult`. v3's `providers/destroy.py` is referenced as the authoritative source; the v4 migration checklist makes v3 implementation a hard prerequisite (either land v3 first or merge v3 as part of the same PR). (BLOCKER 2)
- **`DestroyResult` not redefined.** The v4 doc references v3's `DestroyResult` verbatim — `attempts: int = 0`, `stop_error: str | None = None`, `last_status_code: int | None = None`, `verify_error: str | None = None`, with v3's invariants (`attempts == 0` for refusal, `attempts >= 1` for verdict, no protocol context on refusals). The v4 doc's diagnostic helper reads fields by name (order-independent). (BLOCKER 1)
- **`_describe_destroy_result` consolidated and enriched.** Single shared helper in `providers/vastai.py`, used by both the runner and the factory. Output includes `verdict` and `refusal` (not just attempts + errors) so the fallback "unrecognised result" log exposes the actual typed outcome. (CONCERN 3)
- **`cli.py:instances` migration expanded.** The migration step now requires `OwnershipPolicy` construction (with comma trimming like `cleanup`), `ownership.matches(candidate.ownership_key)` for the "Owned" column (no v2 substring match), and tests covering malicious prefixes, registry ports, tags, digests, and empty sets. (CONCERN 4)
- **`OwnershipPolicy.matches` narrowed for strict type checking.** `self._normalised` is read into a local `normalised` variable; the function returns `False` if `normalised is None`, otherwise tests membership. Pyright strict mode can now infer the invariant. (CONCERN 5)
- **Unused imports removed.** `Iterable` and `StrEnum` are no longer imported in `cleanup_policy.py`; `Optional` is no longer imported in `providers/vastai.py`. (NIT 6)

### Applied from the 8th-pass review (fifth pass)

- Module ownership corrected (destroy types in `providers.destroy`, imports in adapter and runner).
- `read_vastai_api_key` env-first.
- Empty `--allowed-images` fail-closed.
- Null provider ID rejected.
- `destroy_fn` return type validated.
- `NO_CREDENTIALS` WARNING.
- `VastaiRunner.destroy_instance` typed logging.

### Applied from the 7th-pass review (fourth pass)

- ABSENT-credential CLI fallback.
- Correct v3 `DestroyResult` translation.
- `VastaiRunner.destroy_instance` delegation.
- Non-empty catch error.

### Applied from the 6th-pass review (third pass)

- `_repository` strips digest, strips only the final tag separator, preserves registry and port.
- Runner and adapter consume `OwnershipPolicy.matches()` directly.
- `EXPLICITLY_DISABLED` short-circuits before enumeration.
- `docker_image` non-empty invariant.
- `CleanupResult` invariants tightened.
- Authoritative eligibility (later replaced by negative terminal-states list).

### Applied from the 5th-pass review (second pass)

- `@dataclass(frozen=True, kw_only=True)` for `ProviderCleanupPolicy`.
- `list_instances_fn` + `list_instances()` method on the policy.
- `destroy()` is authoritative.
- Hostile check removed.
- CLI uses `dataclasses.replace`.
- Provider factories moved to `providers/vastai.py`.
- Migration order revised.

### Applied from the 5th-pass review (first draft)

- Initial structural fixes from the first review.

## Module taxonomy

The v4 doc adds one new module (`cleanup_policy.py`) and modifies
**four** existing modules (`providers/destroy_adapters/vastai.py`,
`providers/vastai.py`, `batch.py`, `cli.py`) and deletes legacy
helpers from `orchestrator.py`. The v3 modules
(`providers/destroy.py`, `unit_lifecycle.py`) and the `CloudRunner`
ABC are referenced unchanged. `BatchOrchestrator` is **modified**
(adds `cleanup_policy` parameter, replaces `_sweep_zombies`).

### Module ownership (clarified)

| Module | Owns | v4 changes |
|---|---|---|
| `providers/destroy.py` (v3 prerequisite) | `DestroyVerdict`, `DestroyRefusal`, `DestroyResult`, `VerifyVerdict`, `VerifyResult`, `DestroyPolicy`, callback protocols, `belt_and_suspenders` | None — referenced verbatim |
| `providers/destroy_adapters/vastai.py` (v3 + v4) | `CredentialState`, `CredentialResolution`, `read_vastai_api_key()` (env-first), `destroy_vastai_instance()` (amended signature) | Env-first credentials; amended adapter signature |
| `providers/vastai.py` (v4) | `VastaiProviderConfig`, `VastaiRunner`, `vastai_cmd`, `verify_instance_ownership`, `list_vastai_instances()`, `build_vastai_cleanup_policy()`, `_describe_destroy_result`, `VASTAI_TERMINAL_STATES` | Many (see diff) |
| `batch.py` (v4) | `BatchOrchestrator` (modified) | Adds required `cleanup_policy: ProviderCleanupPolicy` parameter; replaces `_sweep_zombies` (policy-driven); removes `sweep_zombie_instances` import |
| `cli.py` (v4) | CLI commands | Refactored `batch` / `cleanup` / `instances` commands; `cli.py:instances` "Owned" column uses `OwnershipPolicy.matches()` (v2 substring match removed) |
| `orchestrator.py` (v4 deletion) | — | Delete `sweep_zombie_instances`, `load_vastai_api_key` |
| `cleanup_policy.py` (v4) | `OwnershipPolicy`, `InstanceCandidate`, `CleanupVerdict`, `CleanupRefusal`, `CleanupResult`, `ProviderCleanupPolicy` | New module |

The core `cleanup_policy.py` module imports nothing from `providers/`.
The `providers/vastai.py` module imports from both `providers/destroy`
(types) and `providers/destroy_adapters/vastai` (the adapter).

### New: `cleanup_policy` — owns the DTOs + the generic policy class

Owns: `InstanceCandidate` (frozen DTO), `CleanupVerdict` enum,
`CleanupRefusal` enum, `CleanupResult` (typed return), `OwnershipPolicy`
(frozen, shared ownership semantics), `ProviderCleanupPolicy`
(frozen, `kw_only`, provider-agnostic, holds the `list_instances_fn`
and `destroy_fn` callbacks).

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
│  providers/destroy (providers/destroy.py)       │  v3 PREREQUITE: belt-and-suspenders
│    └── DestroyVerdict | DestroyRefusal |        │  protocol + canonical types
│        DestroyResult | belt_and_suspenders |   │
│        DestroyPolicy | callback protocols       │
├─────────────────────────────────────────────────┤
│  providers/destroy_adapters/vastai.py           │  v3 + v4: Vast.ai adapter
│    └── CredentialState | CredentialResolution | │
│        read_vastai_api_key (env-first) |         │
│        destroy_vastai_instance(ownership=,      │
│        credentials=)                            │
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
from dataclasses import dataclass, field
from enum import Enum
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


class OwnershipVerification(Enum):
    """Result of ``verify_instance_ownership`` (the CLI ownership check).

    Tagged union — replaces v2's conflated ``bool`` (which conflated
    "owned and present" with "absent and therefore safe"). Used by the
    CLI fallback path in ``build_vastai_cleanup_policy._cli_fallback``
    to dispatch:

    - ``OWNED``    → instance exists, image matches the policy → CLI
                     destroy may proceed (returns ``CLI_ATTEMPTED``).
    - ``ABSENT``   → instance is gone from the API; the verifier has
                     proved absence from a well-formed response. The
                     fallback can translate this directly into the
                     configured "already gone" verdict (caller's
                     choice — not the verifier's).
    - ``REFUSED``  → instance exists but image does not match, or the
                     API response was malformed in a way that prevents
                     proving absence (non-dict record, missing / null /
                     empty / whitespace-only ``id``, non-list response,
                     unexpected exception). The verifier refuses the
                     destroy to the caller.
    - ``DISABLED`` → ownership checking is disabled (no policy).
                     Bypasses the API call entirely.
    """

    OWNED = "owned"
    ABSENT = "absent"
    REFUSED = "refused"
    DISABLED = "disabled"


def _normalize_instance_id(raw_id: object) -> Optional[str]:
    """Canonicalize a Vast.ai instance ID for comparison.

    Shared normalizer used by ``list_vastai_instances`` and
    ``verify_instance_ownership`` so the two agree on what counts as
    a "valid ID" and how to compare.

    Returns:
        - ``None`` for inputs that are not valid IDs (``None``,
          ``bool``, anything that is not ``str`` or ``int``, empty
          string, blank whitespace-only string).
        - The stripped string for valid inputs.

    Rationale:

    - Vast.ai sometimes returns numeric IDs (JSON ``int``). We
      stringify them. Booleans are explicitly rejected: ``True``
      stringifies to ``"True"`` which is not the intended ID and
      would silently shadow a real one.
    - Whitespace-only IDs are rejected (after ``strip()``, empty).
      Padded IDs (``" 123 "``) are accepted and canonicalised to
      ``"123"``.
    - The same rule is applied uniformly by both enumeration and
      verification; padded input on the caller side and padded value
      on the API side both reduce to the canonical form.
    """
    if isinstance(raw_id, bool) or raw_id is None:
        return None
    if not isinstance(raw_id, (str, int)):
        return None
    canonical = str(raw_id).strip()
    if not canonical:
        return None
    return canonical


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

        The local ``normalised`` variable narrows the
        ``_normalised`` type so Pyright strict mode can infer the
        cross-field invariant (when ``owned_images is not None``,
        ``_normalised`` is a frozenset, not None).
        """
        normalised = self._normalised
        if self.owned_images is None:
            return True
        if normalised is None:
            return False
        if not image_ref:
            return False
        repo = _repository(image_ref)
        if not repo:
            return False
        return repo in normalised


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
    ``ALREADY_GONE`` is a v4 verdict produced by the CLI
    fallback path when the ownership verifier proves absence
    from a fully well-formed response — the fallback short-
    circuits and does NOT invoke CLI destruction (which would
    otherwise invent a spurious ``UNKNOWN`` from a 'not found'
    CLI exit code). The orchestrator does NOT count
    ``ALREADY_GONE`` toward ``killed`` (no destroy happened);
    it logs at ``INFO`` because the desired end-state was
    already achieved by the absence.
    """

    DESTROYED = "destroyed"
    CLI_ATTEMPTED = "cli_attempted"  # CLI fallback ran; destruction not confirmed
    ALREADY_GONE = "already_gone"    # CLI verifier proved absence; no destroy
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

    Invariant: ``instance_id`` must be non-empty and pre-stripped
    (an explicit JSON ``null`` is the caller's responsibility to
    filter before constructing the candidate).
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

        Never raises. Two safety guarantees:

        1. The catch-all uses ``f"{type(exc).__name__}: {exc}"``
           so the error string is always non-empty (preserves
           the ``CleanupResult`` invariant even for
           ``raise RuntimeError()`` with no message).
        2. The ``destroy_fn`` callback's return value is type-
           checked: if the callback returns ``None`` or any non-
           ``CleanupResult``, the policy substitutes a
           ``CleanupResult(verdict=UNKNOWN, error=...)`` so the
           orchestrator never sees an invalid result.
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
            result = self.destroy_fn(candidate)
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
        if not isinstance(result, CleanupResult):
            logger.warning(
                "Cleanup: destroy_fn for %s returned invalid result "
                "type %s — substituting UNKNOWN",
                candidate.instance_id,
                type(result).__name__,
            )
            return CleanupResult(
                verdict=CleanupVerdict.UNKNOWN,
                error=(
                    f"destroy_fn returned invalid result type "
                    f"{type(result).__name__}"
                ),
            )
        return result
```

## `providers/destroy_adapters/vastai.py` — v4 amendments (patch-style excerpt)

The v4 design adds the `CredentialState` + `CredentialResolution`
types plus the amended `destroy_vastai_instance` signature. The
adapter **imports** the destroy types from `providers/destroy.py`
(v3 prerequisite); it does not redefine them.

The block below is a **patch-style excerpt** showing only the
v4-amended content. The unchanged v3 imports (the existing REST
callbacks, `VASTAI_POLICY`, `belt_and_suspenders`, etc.) remain
in the file. Implementing this block literally does NOT replace
the v3 adapter — it only adds the new credential types, the
env-first resolver, and the amended `destroy_vastai_instance`
signature.

```python
# src/vastai_gpu_runner/providers/destroy_adapters/vastai.py
#
# --- v3 imports (unchanged) ---
#
# from vastai_gpu_runner.providers.destroy import (
#     VerifyVerdict, VerifyResult, DestroyPolicy,
#     belt_and_suspenders,
# )
# import requests  # for the v3 REST callbacks
# import logging  # VASTAI_POLICY timing support
#
# --- v4 additions ---

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from vastai_gpu_runner.cleanup_policy import OwnershipPolicy
from vastai_gpu_runner.providers.destroy import (
    DestroyRefusal,
    DestroyResult,
    DestroyVerdict,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Credential types (v3 shape)
# ---------------------------------------------------------------------------


class CredentialState(StrEnum):
    """Three-state credential resolution (v3 verbatim)."""
    AVAILABLE = "available"
    ABSENT = "absent"
    EXPLICITLY_DISABLED = "explicitly_disabled"


@dataclass(frozen=True)
class CredentialResolution:
    """Output of ``read_vastai_api_key``.

    Invariants (enforced in ``__post_init__``):
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
# read_vastai_api_key — v3 env-first, fail-closed
# ---------------------------------------------------------------------------


def read_vastai_api_key() -> CredentialResolution:
    """Read the Vast.ai API key — env-first, fail-closed.

    Resolution order:
        1. Inspect ``VASTAI_API_KEY`` env var.
        2. Present but empty/whitespace → ``EXPLICITLY_DISABLED``
           (the user has explicitly disabled credentials).
        3. Present and non-empty → ``AVAILABLE`` with stripped key.
        4. Then inspect credential files
           (``~/.config/vastai/vast_api_key``, ``~/.vast_api_key``).
        5. Blank file → warning, treat as ``ABSENT`` (CLI fallback
           is permitted).
        6. Unreadable file (``OSError``) → warning, treat as ``ABSENT``.
        7. No file present → ``ABSENT``.

    Invariant: ``EXPLICITLY_DISABLED`` is reserved for the case
    where the user has explicitly opted out via the env var.
    A blank or unreadable file does NOT mean disabled — that
    would silently block the controlled CLI fallback.
    """
    env_key = os.environ.get("VASTAI_API_KEY")
    if env_key is not None:
        stripped = env_key.strip()
        if not stripped:
            return CredentialResolution(state=CredentialState.EXPLICITLY_DISABLED)
        return CredentialResolution(state=CredentialState.AVAILABLE, key=stripped)

    for kp in (
        Path("~/.config/vastai/vast_api_key").expanduser(),
        Path("~/.vast_api_key").expanduser(),
    ):
        try:
            if kp.exists():
                raw = kp.read_text()
                stripped = raw.strip()
                if stripped:
                    return CredentialResolution(
                        state=CredentialState.AVAILABLE, key=stripped
                    )
                # Blank file: warn, continue as ABSENT.
                logger.warning(
                    "Credential file %s is empty; treating as ABSENT "
                    "(CLI fallback will be attempted)",
                    kp,
                )
        except OSError as exc:
            logger.warning(
                "Could not read credential file %s: %s; "
                "treating as ABSENT",
                kp,
                exc,
            )
            continue

    return CredentialResolution(state=CredentialState.ABSENT)


# ---------------------------------------------------------------------------
# destroy_vastai_instance — amended signature (v4)
# ---------------------------------------------------------------------------


def destroy_vastai_instance(
    instance_id: str,
    *,
    ownership: OwnershipPolicy,
    credentials: CredentialResolution | None = None,
) -> DestroyResult:
    """Stop + delete + verify a Vast.ai instance.

    Args:
        instance_id: Vast.ai instance ID.
        ownership: Shared ownership policy. The runner and the
            cleanup adapter both pass the same instance.
        credentials: Pre-resolved credential state. When ``None``
            (the default), falls back to ``read_vastai_api_key()``
            — preserves the v3 back-compat path for direct callers.

    Returns:
        ``DestroyResult`` with either ``verdict`` or ``refusal``
        (never both). CLI fallback is performed by the v4 factory,
        not by this adapter — the adapter's credential handling
        is simple and stateless.

    Behaviour by credential state:
        - EXPLICITLY_DISABLED → ``refusal=CREDENTIALS_DISABLED``
          (no provider calls).
        - ABSENT → ``refusal=NO_CREDENTIALS`` (the v4 factory
          intercepts this and runs the CLI fallback).
        - AVAILABLE → REST path: ownership check via API, then
          belt-and-suspenders (stop → DELETE×retry → verify →
          re-destroy). Returns ``verdict=DESTROYED`` on confirmed
          destruction, ``LEAKED`` on resurrection, ``UNKNOWN``
          on indeterminate outcome.
    """
    resolution = credentials or read_vastai_api_key()
    if resolution.state == CredentialState.EXPLICITLY_DISABLED:
        return DestroyResult(refusal=DestroyRefusal.CREDENTIALS_DISABLED)
    if resolution.state == CredentialState.ABSENT:
        return DestroyResult(refusal=DestroyRefusal.NO_CREDENTIALS)
    # AVAILABLE: REST path. The implementation calls
    # belt_and_suspenders(stop_fn, delete_fn, verify_fn, *, policy)
    # from providers/destroy.py, with the v3 policy. The runner
    # and the cleanup adapter both delegate to this entry point
    # via this function — they do NOT have inline REST logic.
```

## `providers/vastai.py` — full module updates

The `VastaiProviderConfig`, `VastaiRunner`, `list_vastai_instances`,
and `build_vastai_cleanup_policy` shapes. The destroy types are
imported from `providers/destroy.py`; the credential types and the
adapter are imported from `providers/destroy_adapters/vastai.py`.
The local `vastai_cmd`, `verify_instance_ownership`, and
`_describe_destroy_result` helpers are not imported from elsewhere.

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
from typing import AbstractSet

from vastai_gpu_runner.cleanup_policy import (
    CleanupRefusal,
    CleanupResult,
    CleanupVerdict,
    InstanceCandidate,
    OwnershipPolicy,
    ProviderCleanupPolicy,
)
from vastai_gpu_runner.providers.destroy import (
    DestroyRefusal,
    DestroyResult,
    DestroyVerdict,
)
from vastai_gpu_runner.providers.destroy_adapters.vastai import (
    CredentialResolution,
    CredentialState,
    destroy_vastai_instance,
    read_vastai_api_key,
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
# vastai_cmd + verify_instance_ownership — local provider helpers
# ---------------------------------------------------------------------------


def vastai_cmd(args: list[str], *, timeout: int = 30) -> str:
    """Run a vastai CLI command (local to providers/vastai.py).

    Raises ``RuntimeError`` on non-zero return, timeout, or missing CLI.
    """
    cmd = ["vastai", *args]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        if result.returncode != 0:
            msg = f"vastai {' '.join(args)} failed: {result.stderr.strip()}"
            raise RuntimeError(msg)
        return result.stdout.strip()
    except FileNotFoundError as exc:
        msg = "vastai CLI not installed. Install with: pip install vastai"
        raise RuntimeError(msg) from exc
    except subprocess.TimeoutExpired as exc:
        msg = f"vastai {' '.join(args)} timed out after {timeout}s"
        raise RuntimeError(msg) from exc


def verify_instance_ownership(
    instance_id: str,
    *,
    ownership: OwnershipPolicy,
) -> OwnershipVerification:
    """CLI-side ownership check for a single instance (v4 factory).

    This is the CLI-auth verification used by the v4 factory's
    CLI fallback path (separate auth context from the REST path).
    Returns a tagged ``OwnershipVerification`` so the caller can
    distinguish three operationally-different outcomes that v2
    conflated into ``bool``:

    - ``OWNED``    → instance exists and is owned; CLI destroy may
                     proceed.
    - ``ABSENT``   → instance is not in the API response (already
                     destroyed). The verifier has proved absence
                     from a fully well-formed response and the
                     caller can choose the desired verdict for
                     "already gone" without invoking destruction.
    - ``REFUSED``  → instance exists but is unowned, OR the API
                     response was so malformed that absence cannot
                     be proved, OR any other verifier-level failure.
                     Refusal means "do not destroy". Failure is
                     fail-closed.
    - ``DISABLED`` → ownership checking is disabled (no policy).
                     This is a short-circuit that bypasses the API
                     call entirely.

    Implementation notes (carefully ordered):

    1. **One short-circuit up front** (DISABLED): skip the API call
       entirely when ``owned_images is None``.
    2. **One canonical form for the requested ID**: ``_normalize_instance_id``.
       Padded, numeric, and string IDs reduce to the same canonical
       form before comparison.
    3. **One outermost ``except Exception`` boundary**: subprocess
       failures, JSON parsing, response-shape problems, or even
       downstream ``ownership.matches`` failures cannot escape the
       function. (They're translated to ``REFUSED``.)
    4. **Validate-and-normalize the entire response first**, then
       locate and evaluate the requested record. A single malformed
       entry anywhere in the list fails the whole check
       (conservatively — we cannot prove absence of one ID when
       another entry is unreadable). We do NOT short-circuit on
       the first match, because a malformed record *after* a match
       would otherwise be ignored.
    5. **Failures are logged at ERROR** level so the operator sees
       the destroy has been refused (not "ambiguous state").

    The function never returns ``True`` / ``False`` — the contract
    is encoded in the enum so callers must explicitly handle each
    case. v2's conflated bool cannot represent "instance is owned
    AND present" separately from "instance is already absent".
    """
    if ownership.owned_images is None:
        return OwnershipVerification.DISABLED

    canonical_target = _normalize_instance_id(instance_id)
    if canonical_target is None:
        logger.error(
            "REFUSING: requested instance_id %r is not a valid ID — refusing to destroy",
            instance_id,
        )
        return OwnershipVerification.REFUSED

    try:
        raw = vastai_cmd(["show", "instances", "--raw"], timeout=15)
        instances = json.loads(raw)
    except (RuntimeError, json.JSONDecodeError) as exc:
        logger.error(
            "REFUSING: cannot verify ownership of instance %s "
            "(API error: %s) — destroy refused. Resolve the API response and retry.",
            instance_id,
            exc,
        )
        return OwnershipVerification.REFUSED

    if not isinstance(instances, list):
        logger.error(
            "REFUSING: cannot verify ownership of instance %s — "
            "response is not a list (got %s). Destroy refused.",
            instance_id,
            type(instances).__name__,
        )
        return OwnershipVerification.REFUSED

    # Validate and normalize the entire response first. A single
    # malformed record anywhere in the list fails the check — we
    # cannot prove absence of the requested ID when another entry
    # has no usable ``id``. Refusing on a malformed record AFTER a
    # matching record is the conservative position.
    normalised_records: list[tuple[str, str]] = []  # (canonical_id, image_uuid)
    for raw_record in instances:
        if not isinstance(raw_record, dict):
            logger.error(
                "REFUSING: instance %s cannot be verified — "
                "response contains a non-object record: %r. "
                "Destroy refused.",
                instance_id,
                raw_record,
            )
            return OwnershipVerification.REFUSED
        canonical_id = _normalize_instance_id(raw_record.get("id"))
        if canonical_id is None:
            logger.error(
                "REFUSING: instance %s cannot be verified — "
                "record has missing/null/invalid id: %r. Destroy refused.",
                instance_id,
                raw_record,
            )
            return OwnershipVerification.REFUSED
        # Strip the image_uuid exactly the same way list_vastai_instances does,
        # so verification matches enumeration.
        image_uuid = str(raw_record.get("image_uuid", ""))
        normalised_records.append((canonical_id, image_uuid))

    # Locate the requested record in the validated, normalized list.
    for record_id, image_uuid in normalised_records:
        if record_id == canonical_target:
            if ownership.matches(image_uuid):
                return OwnershipVerification.OWNED
            logger.error(
                "BLOCKED: instance %s belongs to another project (image=%s). Will NOT destroy.",
                instance_id,
                image_uuid,
            )
            return OwnershipVerification.REFUSED

    logger.info(
        "Instance %s not found in account (already destroyed?)",
        instance_id,
    )
    return OwnershipVerification.ABSENT


# ---------------------------------------------------------------------------
# _describe_destroy_result — single shared diagnostic helper
# ---------------------------------------------------------------------------


def _describe_destroy_result(result: DestroyResult) -> str:
    """Build diagnostic text from v3 DestroyResult structured fields.

    Single shared helper used by both ``VastaiRunner.destroy_instance``
    and ``build_vastai_cleanup_policy._destroy``. Includes verdict +
    refusal so the fallback "unrecognised result" log exposes the
    actual typed outcome. Handles ``None`` fields gracefully (v3 uses
    optional diagnostics). Always produces non-empty output.
    """
    return (
        f"verdict={result.verdict!r}, refusal={result.refusal!r}, "
        f"attempts={result.attempts}, "
        f"last_status_code={result.last_status_code}, "
        f"verify_error={result.verify_error!r}, "
        f"stop_error={result.stop_error!r}"
    )


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
            the runner passes ``None`` to the adapter, which
            calls ``read_vastai_api_key()`` (the v3 back-compat path).
        docker_image: Docker image to use for new instances.
        min_gpu_vram_mib: Minimum GPU VRAM required (default 20 GB).
        setup_commands: Optional pre-instance setup commands.

    Note: ``allowed_images`` is a deprecated back-compat alias
    that builds an ``OwnershipPolicy`` from the given set.
    Simultaneous ``ownership=`` and ``allowed_images=`` raises
    ``ValueError`` — silently ignoring one destruction-safety
    configuration is unsafe.
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
        For every other outcome, the typed ``DestroyResult`` is
        logged before returning False so operators can diagnose
        why the destroy failed (LEAKED, UNKNOWN, or any refusal).
        """
        result = destroy_vastai_instance(
            instance.instance_id,
            ownership=self.ownership or OwnershipPolicy(),
            credentials=self.credentials,
        )
        if result.verdict == DestroyVerdict.DESTROYED:
            instance.status = InstanceStatus.DESTROYED
            return True
        # Non-DESTROYED: log the typed result before collapsing to False.
        if result.verdict == DestroyVerdict.LEAKED:
            logger.error(
                "Destroy %s LEAKED — manual review required: %s",
                instance.instance_id,
                _describe_destroy_result(result),
            )
        elif result.verdict == DestroyVerdict.UNKNOWN:
            logger.warning(
                "Destroy %s outcome=UNKNOWN: %s",
                instance.instance_id,
                _describe_destroy_result(result),
            )
        elif result.refusal == DestroyRefusal.OWNERSHIP:
            logger.error(
                "Destroy %s refused (ownership check rejected): %s",
                instance.instance_id,
                _describe_destroy_result(result),
            )
        elif result.refusal == DestroyRefusal.NO_CREDENTIALS:
            logger.error(
                "Destroy %s refused (no credentials): %s",
                instance.instance_id,
                _describe_destroy_result(result),
            )
        elif result.refusal == DestroyRefusal.CREDENTIALS_DISABLED:
            logger.error(
                "Destroy %s refused (credentials disabled): %s",
                instance.instance_id,
                _describe_destroy_result(result),
            )
        else:
            logger.error(
                "Destroy %s returned unrecognised result: %s",
                instance.instance_id,
                _describe_destroy_result(result),
            )
        return False


# ---------------------------------------------------------------------------
# list_vastai_instances — validates instance_id type + content
# ---------------------------------------------------------------------------


def list_vastai_instances() -> list[InstanceCandidate]:
    """Read-only enumeration of Vast.ai instances on this account.

    Returns ``InstanceCandidate`` records. Records with missing,
    null, or empty ``instance_id`` are skipped (logged). A failure
    to enumerate returns an empty list.

    Validation guards:
        - ``inst`` must be a dict (otherwise skip).
        - ``inst.get("id")`` must not be ``None`` (JSON null is not
          a valid Vast.ai instance ID; ``str(None)`` would produce
          the literal string ``"None"`` which passes falsy checks).
        - ``instance_id`` must be non-empty after strip.
        - Other malformed fields are caught by the per-record
          ``try`` block.

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
    if not isinstance(instances, list):
        logger.warning(
            "list_vastai_instances: expected list, got %s — skipping enumeration",
            type(instances).__name__,
        )
        return candidates
    for inst in instances:
        if not isinstance(inst, dict):
            logger.warning("Skipping non-object instance record: %r", inst)
            continue
        raw_id = inst.get("id")
        if raw_id is None:
            logger.warning(
                "Skipping instance with null ID: %r",
                {k: inst.get(k) for k in ("label", "image_uuid", "actual_status")},
            )
            continue
        instance_id = str(raw_id).strip()
        if not instance_id:
            logger.warning(
                "Skipping instance with empty ID: %r",
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
    invoking CLI enumeration.

    The wired ``destroy_fn`` runs:
        1. Eligibility check (skip terminal states like ``destroyed``).
        2. ``destroy_vastai_instance`` with the canonical ownership + credentials.
        3. If the adapter returns ``NO_CREDENTIALS``, run the CLI
           fallback path: CLI ownership verification → CLI destroy →
           ``CLI_ATTEMPTED`` (destruction unconfirmed) or ``UNKNOWN``.
        4. Translate all three v3 verdicts and all three v3 refusals
           into the v4 ``CleanupResult`` shape.
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

    def _cli_fallback(candidate: InstanceCandidate) -> CleanupResult:
        """CLI fallback path for ABSENT credentials.

        The v3 adapter returns NO_CREDENTIALS when the API key is
        not configured. We then verify ownership via the CLI auth
        context (which uses the file-based key) and dispatch on
        the verifier's tagged result:

        - ``OWNED``    → run CLI destroy; report ``CLI_ATTEMPTED``
                         (destruction unconfirmed — CLI exit is not
                         a REST confirmation).
        - ``DISABLED`` → ownership checking is off; proceed straight
                         to CLI destroy (same ``CLI_ATTEMPTED``
                         translation as ``OWNED``).
        - ``ABSENT``   → the verifier proved the instance is already
                         gone from a fully well-formed response.
                         Short-circuit to ``ALREADY_GONE`` without
                         invoking CLI destroy (a 'not found' CLI
                         error would otherwise fabricate a spurious
                         ``UNKNOWN``).
        - ``REFUSED``  → the instance exists but is unowned, or the
                         response is too malformed to prove absence.
                         Refuse with ``OWNERSHIP`` and a diagnostic.

        The function never raises: any unexpected exception in
        CLI destroy becomes an ``UNKNOWN`` outcome with a
        non-empty error string.
        """
        try:
            verification = verify_instance_ownership(
                candidate.instance_id,
                ownership=ownership,
            )
        except Exception as exc:
            return CleanupResult(
                verdict=CleanupVerdict.UNKNOWN,
                error=(
                    f"CLI ownership verification raised "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        if verification is OwnershipVerification.REFUSED:
            return CleanupResult(
                refusal=CleanupRefusal.OWNERSHIP,
                error=(
                    "CLI ownership check refused "
                    f"{candidate.instance_id!r} (instance unowned "
                    "or response malformed)"
                ),
            )
        if verification is OwnershipVerification.ABSENT:
            return CleanupResult(
                verdict=CleanupVerdict.ALREADY_GONE,
                error=(
                    "CLI ownership verifier proved absence in a "
                    "well-formed response; nothing to destroy."
                ),
            )
        # OWNED or DISABLED — proceed to CLI destroy.
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
                    f"{candidate.instance_id!r}; "
                    f"{_describe_destroy_result(result)}"
                ),
            )
        if result.refusal == DestroyRefusal.CREDENTIALS_DISABLED:
            return CleanupResult(
                refusal=CleanupRefusal.CREDENTIALS_DISABLED,
                error=(
                    f"v3 adapter refused: VASTAI_API_KEY explicitly empty; "
                    f"{_describe_destroy_result(result)}"
                ),
            )
        # 5. Translate verdicts exhaustively.
        if result.verdict == DestroyVerdict.DESTROYED:
            return CleanupResult(verdict=CleanupVerdict.DESTROYED)
        if result.verdict == DestroyVerdict.LEAKED:
            return CleanupResult(
                verdict=CleanupVerdict.LEAKED,
                error=_describe_destroy_result(result),
            )
        return CleanupResult(
            verdict=CleanupVerdict.UNKNOWN,
            error=_describe_destroy_result(result),
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
matching operational impact. Unexpected `NO_CREDENTIALS` is
`WARNING` (it means the factory's CLI fallback was bypassed and the
orphan wasn't cleaned up).

```python
# src/vastai_gpu_runner/batch.py (changes to _sweep_zombies)
# NB: `logger` is the module-level logger already defined at the
# top of batch.py (`logger = logging.getLogger(__name__)`).

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
        elif result.verdict == CleanupVerdict.ALREADY_GONE:
            # Verifier proved absence from a well-formed response;
            # no destroy was performed. INFO because the desired
            # end-state was already achieved — not an operational
            # failure.
            logger.info(
                "Zombie sweep: %s already gone (CLI verifier proved absence): %s",
                candidate.instance_id,
                result.error,
            )
        elif result.refusal == CleanupRefusal.CREDENTIALS_DISABLED:
            logger.warning(
                "Zombie sweep: %s refused (credentials disabled): %s",
                candidate.instance_id,
                result.error,
            )
        elif result.refusal == CleanupRefusal.NO_CREDENTIALS:
            # Should not reach here — the factory's CLI fallback
            # intercepts NO_CREDENTIALS. Logged at WARNING because
            # it indicates the orphan was not cleaned up.
            logger.warning(
                "Zombie sweep: %s unexpectedly returned NO_CREDENTIALS "
                "after fallback handling: %s",
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
    if killed:
        logger.info("Zombie sweep: destroyed %d instance(s)", killed)
    return killed
```

The orchestrator's `__init__` requires `cleanup_policy: ProviderCleanupPolicy`
and `runner_factory: RunnerFactory`.

## CLI wiring

The CLI is the one place that builds the canonical config (for the
runner) and the two canonical objects (for the cleanup policy). It
threads both into the v3 + v4 entry points. Empty
`--allowed-images` is **fail-closed** (empty set, reject every
image), not opt-out. The `cli.py:instances` command's "Owned"
column uses `OwnershipPolicy.matches()` (the v2 unsafe substring
match is removed).

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

    base = VastaiProviderConfig.from_env()
    config = replace(
        base,
        docker_image=image,
        ownership=OwnershipPolicy(owned_images=frozenset({image})),
    )

    runner_factory = lambda: VastaiRunner.from_config(config)  # noqa: E731

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

The existing `cli.py:cleanup` command migrates to use the new API.
Empty `--allowed-images` is **fail-closed**:

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
            help=(
                "Comma-separated Docker images owned by this project. "
                "Omit to opt out of the ownership check (DANGEROUS — "
                "every instance with the label prefix is destroyed). "
                "Pass an empty string to fail-closed (every instance "
                "is refused; use to test the wiring without risk)."
            ),
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

    # Distinguish None (flag omitted → opt-out) from "" (explicit empty →
    # fail-closed). Whitespace-only entries in comma-separated input are
    # stripped; comma-only input becomes an empty set.
    if allowed_images is None:
        ownership = OwnershipPolicy()
    else:
        images = frozenset(
            item.strip()
            for item in allowed_images.split(",")
            if item.strip()
        )
        ownership = OwnershipPolicy(owned_images=images)

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

The `cli.py:instances` command migrates to use `list_vastai_instances`
and the v4 `OwnershipPolicy.matches()` for the "Owned" column. The
v2 unsafe substring/prefix match is removed.

```python
# src/vastai_gpu_runner/cli.py (instances command)
@app.command()
def instances(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    allowed_images: Annotated[
        str | None,
        typer.Option(
            "--allowed-images",  # canonical
            "--owned-images",     # alias
            help=(
                "Comma-separated Docker images owned by this project. "
                "Used for the 'Owned' column. Empty string → fail-closed "
                "(every instance shown as not owned)."
            ),
        ),
    ] = None,
) -> None:
    """List active Vast.ai instances with status and ownership info."""
    _setup_logging(verbose)
    from rich.console import Console
    from rich.table import Table
    from vastai_gpu_runner.cleanup_policy import OwnershipPolicy
    from vastai_gpu_runner.providers.vastai import list_vastai_instances

    console = Console()
    candidates = list_vastai_instances()
    if not candidates:
        console.print("No active instances.")
        return

    # Comma trimming matches the cleanup command. None → opt-out.
    if allowed_images is None:
        ownership = OwnershipPolicy()
    else:
        images = frozenset(
            item.strip()
            for item in allowed_images.split(",")
            if item.strip()
        )
        ownership = OwnershipPolicy(owned_images=images)

    table = Table(title=f"{len(candidates)} Active Instance(s)")
    table.add_column("ID", style="cyan")
    table.add_column("GPU")
    table.add_column("Status")
    table.add_column("Label")
    table.add_column("$/hr", justify="right", style="green")
    table.add_column("Owned", justify="center")

    total_hourly = 0.0
    running = 0
    for c in candidates:
        owned = ownership.matches(c.ownership_key)
        total_hourly += c.cost_per_hour
        if c.state == "running":
            running += 1
        table.add_row(
            c.instance_id,
            c.gpu_model,
            c.state,
            c.label,
            f"${c.cost_per_hour:.3f}",
            "[green]yes[/green]" if owned else "[red]no[/red]",
        )

    console.print(table)
    console.print(f"\nRunning: {running}/{len(candidates)}, Total: ${total_hourly:.2f}/hr")
```

## Migration checklist

Seven steps. The v3 implementation is a **hard prerequisite**:
either land v3 first or merge v3 as part of the same PR. The v4
steps then build on v3's `providers/destroy.py` module
(`DestroyVerdict`, `DestroyRefusal`, `DestroyResult`, `VerifyVerdict`,
`VerifyResult`, `DestroyPolicy`, callback protocols,
`belt_and_suspenders`).

1. **Land v3 implementation.** Implement the `unit_lifecycle.py`,
   `providers/destroy.py` (with `belt_and_suspenders`, `DestroyPolicy`,
   `DestroyVerdict`, `DestroyRefusal`, `DestroyResult`, `VerifyVerdict`,
   `VerifyResult`, callback protocols), and `providers/destroy_adapters/vastai.py`
   (with the v3-shape credential types, `read_vastai_api_key()`,
   and the amended `destroy_vastai_instance()` signature). This must
   land first because v4 amends v3's adapter signature and references
   v3's `DestroyResult` / `DestroyVerdict` / `DestroyRefusal` types.

2. **Add canonical ownership-policy type + invariant tests.**
   - `cleanup_policy.py:OwnershipPolicy` (frozen, `matches(image_ref)` with `_repository`; declared `_normalised` cache field, narrowed for strict type checking).
   - Property tests: `OwnershipPolicy.matches` is reflexive, tag-insensitive, sha256-by-repo, registry-port-aware, fail-closed on empty sets, malformed reference rejection.
   - The `_normalised` is precomputed in `__post_init__`; `matches` is O(1) per call.

3. **Add Vast.ai runner + cleanup adapter.**
   - `providers/vastai.py:VastaiProviderConfig` (frozen, with `__post_init__` invariants).
   - `providers/vastai.py:VastaiRunner.from_config(canonical)` — preserves `ownership` and `credentials`.
   - `VastaiRunner.__init__` adds `ownership: OwnershipPolicy | None` + `credentials: CredentialResolution | None` parameters; rejects simultaneous `ownership=` + `allowed_images=` with `ValueError`; `allowed_images` becomes a deprecated alias.
   - `VastaiRunner.destroy_instance` is a single `destroy_vastai_instance(...)` adapter call; logs the typed `DestroyResult` for non-`DESTROYED` outcomes.
   - `verify_instance_ownership(instance_id, *, ownership: OwnershipPolicy)` — local to `providers/vastai.py`, replaces `_image_is_allowed`.
   - `list_vastai_instances()` returns `list[InstanceCandidate]`; validates `inst` is dict, `id` is not `None`, `instance_id` is non-empty after strip.
   - `VASTAI_TERMINAL_STATES: frozenset[str]` module constant.
   - `_describe_destroy_result(result)` — single shared diagnostic helper with `verdict` + `refusal` + structured fields.
   - `build_vastai_cleanup_policy(*, ownership: OwnershipPolicy, credentials: CredentialResolution) -> ProviderCleanupPolicy`.
   - Adapter tests:
     - EXPLICITLY_DISABLED: `list_instances()` returns `[]` without invoking `vastai_cmd`.
     - ABSENT CLI fallback: v3 returns `NO_CREDENTIALS` → factory runs CLI ownership + CLI destroy → `CLI_ATTEMPTED` (not `DESTROYED`) on command success; `UNKNOWN` on command failure.
     - AVAILABLE: v3 returns `DESTROYED` → `verdict=DESTROYED`.
     - `verdict=LEAKED` / `verdict=UNKNOWN` translate correctly with diagnostic.
     - `refusal=OWNERSHIP` / `NO_CREDENTIALS` / `CREDENTIALS_DISABLED` translate correctly.
     - INELIGIBLE_STATE for terminal or empty state.
     - `list_vastai_instances()` skips dict-non, id-None, id-empty records.
     - `ProviderCleanupPolicy.destroy()` returns `CleanupResult(verdict=UNKNOWN, error="...invalid result type...")` when `destroy_fn` returns `None` or non-`CleanupResult`.
     - `VastaiRunner.destroy_instance` logs the typed `DestroyResult` for non-`DESTROYED` outcomes.

4. **Add orchestrator support behind a fail-closed compatibility path.**
   - `BatchOrchestrator.__init__` accepts `cleanup_policy: ProviderCleanupPolicy` (required).
   - `_sweep_zombies` is policy-driven; logs every non-`DESTROYED` outcome at severity matching operational impact.
   - Orchestrator tests:
     - Severity logging: `LEAKED` = `ERROR`, `UNKNOWN` / `CLI_ATTEMPTED` / `CREDENTIALS_DISABLED` = `WARNING`, unexpected `NO_CREDENTIALS` = `WARNING`, refusals = `INFO` (verified with `caplog`).
     - `_sweep_zombies` continues on `destroy_fn` exceptions (they return `verdict=UNKNOWN` with `type(exc).__name__: exc`).
     - `_sweep_zombies` does NOT import provider modules (verified by `inspect.getsource`).

5. **Update every composition root, subclass, and existing CLI command.**
   - `cli.py:batch`: build `VastaiProviderConfig` via `from_env()` + `replace()`, pass to `VastaiRunner.from_config` and `build_vastai_cleanup_policy(ownership, credentials)`.
   - `cli.py:cleanup`: build `OwnershipPolicy` and `CredentialResolution` directly (no `VastaiProviderConfig`); distinguish `None` (opt-out) from `""` (fail-closed).
   - `cli.py:instances`: use `list_vastai_instances()` for the table; construct `OwnershipPolicy` (with comma trimming like `cleanup`); call `ownership.matches(candidate.ownership_key)` for the "Owned" column. The v2 unsafe substring/prefix match (`any(img.split(":")[0] in image for img in images_set)`) is **removed**.
   - `BatchOrchestrator` subclasses: update composition to supply `cleanup_policy`.
   - Test fixtures: `VastaiProviderConfig` + `OwnershipPolicy` + `CredentialResolution` factory fixtures.

6. **Add integration tests.**
   - `tests/integration/test_cleanup_policy_integration.py`:
     - **disabled-before-enumeration**: `credentials=EXPLICITLY_DISABLED` → `policy.list_instances()` returns `[]`; `vastai_cmd` was not called.
     - **absent-credential CLI fallback**: `credentials=ABSENT` → `policy.destroy(candidate)` returns `verdict=CLI_ATTEMPTED` (NOT `DESTROYED`); the canonical `ABSENT` resolution was passed to the REST adapter (NOT `None`).
     - **empty ownership set**: `ownership=OwnershipPolicy(owned_images=frozenset())` → `OWNERSHIP` (fail-closed).
     - **provider mismatch**: candidate `Provider.RUNPOD` to a Vast.ai policy → `PROVIDER_MISMATCH` with operator-friendly diagnostic.
     - **enumeration failure**: `list_vastai_instances()` raises → `policy.list_instances()` returns `[]`.
     - **ineligible state**: candidate `state="destroyed"` → `INELIGIBLE_STATE`.
     - **severity logging**: orchestrator logs `LEAKED` at `ERROR`, `UNKNOWN` at `WARNING`, unexpected `NO_CREDENTIALS` at `WARNING`, refusals at `INFO`.
     - **non-empty error from empty exception**: `raise RuntimeError()` → orchestrator logs with non-empty error.
     - **null instance_id in enumeration**: JSON `{id: null, ...}` skipped, not passed to destroy.
     - **CLI --allowed-images empty string**: fail-closed (empty set, refuses every candidate).
     - **CLI --allowed-images None**: opt-out (every image considered owned).
     - **destroy_fn returns None**: orchestrator receives `CleanupResult(verdict=UNKNOWN, error="...invalid result type NoneType")`.
     - **VastaiRunner logs typed result**: LEAKED outcome produces an `ERROR`-level log line with the structured diagnostic context.
     - **instances command ownership column**: malicious prefix `myorg/app-malicious:latest` shown as `no` when `--allowed-images myorg/app:1.0`; tag-insensitive `myorg/app:latest` shown as `yes`; digest `myorg/app@sha256:<digest>` shown as `yes` when `--allowed-images myorg/app:1.0` (tag-insensitive repository match); registry-port `registry:5000/myorg/app:1.0` shown as `no` when `--allowed-images myorg/app`; positive registry-port `registry:5000/myorg/app:1.0` shown as `yes` when `--allowed-images registry:5000/myorg/app:1.0`; empty set `--allowed-images ""` shows every instance as `no`.

7. **Delete legacy sweep + duplicated helpers after a repository-wide caller audit.**
   - `audit_caller_sites.sh` (run before deletion): grep for external callers of `orchestrator.sweep_zombie_instances`, `orchestrator.load_vastai_api_key`, `VastaiRunner.allowed_images` (read-only external use), `providers.vastai._image_is_allowed`, `cli.instances`'s substring match. Update external callers.
   - Delete `orchestrator.sweep_zombie_instances`.
   - Delete `orchestrator.load_vastai_api_key`.
   - Delete `providers/vastai.py:_image_is_allowed`.
   - Delete the v2 substring/prefix match in `cli.py:instances`.
   - Delete direct `vastai_cmd(["show", "instances", "--raw"])` parsing in `cli.py:cleanup` and `cli.py:instances`.
   - Update `tests/test_orchestrator.py` and `tests/test_batch.py` to mock `cleanup_policy.list_instances` and `cleanup_policy.destroy`.

## Test plan

- `tests/test_cleanup_policy.py`:
  - `_repository`: 13+ cases (digest, registry ports, malformed, empty, whitespace).
  - `OwnershipPolicy.matches`: reflexive, tag-insensitive, sha256-by-repo, registry-port-aware, fail-closed on empty sets, malformed reference rejection, narrow type-checked.
  - `OwnershipPolicy._normalised`: declared field; precomputed in `__post_init__`; `matches` is O(1) per call.
  - `ProviderCleanupPolicy.list_instances`: returns wired list; catches and returns `[]` on exception.
  - `ProviderCleanupPolicy.destroy`: provider mismatch returns `PROVIDER_MISMATCH`; catches `destroy_fn` exceptions with `f"{type(exc).__name__}: {exc}"` (non-empty even for `RuntimeError()` with no message); validates `destroy_fn` return is `CleanupResult` (returns `CleanupResult(verdict=UNKNOWN, ...)` on `None` or non-`CleanupResult`).
  - `CleanupResult` invariants: 5 cases (verdict/refusal exclusivity, error non-empty on non-DESTROYED, error empty on DESTROYED).
  - `InstanceCandidate.__post_init__`: empty `instance_id` raises; whitespace-only `instance_id` raises.
- `tests/test_providers_vastai.py`:
  - `read_vastai_api_key` env-first: `VASTAI_API_KEY=""` → `EXPLICITLY_DISABLED`; `VASTAI_API_KEY="key"` → `AVAILABLE`; no env + file → `AVAILABLE`; no env + blank file → `ABSENT` (with warning); `OSError` → `ABSENT` (with warning).
  - `VastaiRunner.from_config` round-trips with `VastaiProviderConfig`.
  - `VastaiRunner(allowed_images=frozenset({img}))` (deprecated) emits `DeprecationWarning` + builds equivalent `OwnershipPolicy`.
  - Simultaneous `ownership=` and `allowed_images=` raises `ValueError`.
  - `verify_instance_ownership` returns `OwnershipVerification`
    (tagged enum; **not** ``bool``). Test the four-way contract:
    - `DISABLED`: `OwnershipPolicy(owned_images=None)` returns
      `DISABLED` without an API call (`patch.object` on
      `vastai_cmd`).
    - `OWNED`: well-formed response containing the canonical
      requested ID with a matching `image_uuid` → `OWNED`.
    - `REFUSED` on ownership mismatch: well-formed response with
      the requested ID but a non-matching `image_uuid` →
      `REFUSED` + `ERROR`-level log.
    - `ABSENT`: well-formed response where the requested ID is
      not present → `ABSENT`. Crucially, the CLI fallback
      short-circuits to `ALREADY_GONE` (see factory tests).
    - `REFUSED` on malformed record AFTER a matching owned record:
      ensures the entire response is validated before the
      match is returned.
    - `REFUSED` on malformed record BEFORE a matching owned
      record: same fail-closed reasoning.
    - Invalid requested ID itself (`None`, `""`, `"   "`,
      non-str/non-int) → `REFUSED` + `ERROR`-level log.
  - `_normalize_instance_id` shared normalizer (parametric):
    - `None` → `None`
    - `True` / `False` (bool) → `None`
    - `""` → `None`
    - `"   "` (whitespace) → `None`
    - `" 123 "` (padded) → `"123"` (canonical stripped form)
    - `123` (int) → `"123"`
    - `"abc"` → `"abc"`
  - Invalid-API-response coverage with `caplog`:
    - `RuntimeError` from `vastai_cmd` → `REFUSED` + `ERROR`-level
    - `json.JSONDecodeError` → `REFUSED` + `ERROR`-level
    - non-list response (e.g. `{"foo": "bar"}`) → `REFUSED` +
      `ERROR`-level
    - non-dict record in a list → `REFUSED` + `ERROR`-level
    - record with `id=None` or `id=""` or `id="   "` (parameterized)
      → `REFUSED` + `ERROR`-level
    - Unexpected empty-message `RuntimeError("")` → `REFUSED` +
      `ERROR`-level (does not escape; covered by the outermost
      `except Exception` boundary).
  - `list_vastai_instances` returns `list[InstanceCandidate]` with `gpu_model` + `cost_per_hour` populated; skips records where `inst` is not a dict, `id` is `None`, `id` is empty after strip; returns `[]` on API error.
  - `build_vastai_cleanup_policy(*, ownership, credentials)`:
    - EXPLICITLY_DISABLED list: returns `[]` without `vastai_cmd`.
    - ABSENT destroy — CLI fallback dispatch:
      - verifier `OWNED` → CLI destroy (`vastai destroy instance`)
        → `CLI_ATTEMPTED`.
      - verifier `DISABLED` → CLI destroy → `CLI_ATTEMPTED`.
      - verifier `ABSENT` → short-circuit to `ALREADY_GONE`
        WITHOUT invoking `vastai destroy instance` (factory-level
        test; patch both `verify_instance_ownership` and
        `vastai_cmd` and assert the destroy invocation is
        absent).
      - verifier `REFUSED` → `OWNERSHIP` refusal.
      - CLI destroy exception → `UNKNOWN`.
    - AVAILABLE destroy: v3 DESTROYED → `verdict=DESTROYED`.
    - LEAKED / UNKNOWN translate with `_describe_destroy_result()` diagnostic.
    - OWNERSHIP / NO_CREDENTIALS / CREDENTIALS_DISABLED translate
      correctly; `CREDENTIALS_DISABLED` includes
      `_describe_destroy_result(result)` in the diagnostic
      (CONCERN 2).
    - INELIGIBLE_STATE for `state in VASTAI_TERMINAL_STATES` or
      empty state.
  - `VastaiRunner.destroy_instance` logs typed result: LEAKED → `ERROR`, UNKNOWN → `WARNING`, refusals → `ERROR`; returns `False`.
- `tests/test_batch.py`:
  - `_sweep_zombies` calls `cleanup_policy.list_instances()` exactly once.
  - `_sweep_zombies` calls `cleanup_policy.destroy(candidate)` for every label-matching, untracked candidate.
  - `_sweep_zombies` counts only `verdict=DESTROYED` outcomes (an
    `ALREADY_GONE` outcome is NOT counted as a kill — no destroy
    happened).
  - `_sweep_zombies` logs `LEAKED` at `ERROR`, `UNKNOWN` /
    `CLI_ATTEMPTED` / `CREDENTIALS_DISABLED` at `WARNING`,
    `ALREADY_GONE` at `INFO`, unexpected `NO_CREDENTIALS` at
    `WARNING`, refusals at `INFO` (`caplog`).
  - `_sweep_zombies` continues on `destroy_fn` exceptions.
  - `_sweep_zombies` does NOT import provider modules (`inspect.getsource`).
- `tests/integration/test_cleanup_policy_integration.py` — 14 scenarios from step 6 above.

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
`--owned-images` is added as a documented alias. Empty
`--allowed-images ""` is **fail-closed** (not opt-out) — this is a
behaviour change from the previous CLI default. The v2 substring
match in `cli.py:instances` is removed.

## Out of scope

- **RunPod adapter.** The factory ships when the RunPod adapter
  ships (roadmap item 2). The `ProviderCleanupPolicy` interface is
  provider-agnostic from day one.
- **Hostile detection.** Removed in the v4 second-pass review.
- **Dispute webhook.** Future work.
- **Bulk-destroy optimisation.** YAGNI for now.
- **Cross-provider zombie sweep.** A single orchestrator supports
  one `cleanup_policy`. Multi-policy is future design.

## Review process

This is the ninth review pass on the v4 architecture. Each prior
draft was rejected and addressed. This draft addresses every
finding from the 12th-pass review:

- BLOCKER 1: `verify_instance_ownership` conflated absent and
  owned; replaced with a tagged `OwnershipVerification` enum
  (`OWNED | ABSENT | REFUSED | DISABLED`).
- BLOCKER 2: ID-semantic mismatch between enumeration and
  verification; `None`/`bool`/empty/blank IDs now rejected via
  shared `_normalize_instance_id`. The verifier validates the
  entire response before locating the matching record.
- BLOCKER 3: Outermost `except Exception` boundary plus
  `ERROR`-level log on every failure path.
- CONCERN 1: Test catalogue expanded with parameterized
  invalid-ID cases, malformed-record-BEFORE/AFTER ordering
  tests, numeric/padded IDs, and `caplog` severity assertions
  for every failure path.
- CONCERN 2: `_describe_destroy_result(result)` now appears in
  the `CREDENTIALS_DISABLED` refusal diagnostic.
- NIT 3: `is_candidate` references replaced with the actual
  orchestrator label/tracked-ID filtering mechanism.

The 13th-pass review prompt for ChatGPT-with-GitHub-plugin:

> Review the v4 architecture design at PR #22 (file:
> docs/architecture-v4-cleanup-policy.md) against the v3 design at
> docs/architecture-v3.md and the current code at
> src/vastai_gpu_runner/{batch,orchestrator,runner,cli}.py and
> src/vastai_gpu_runner/providers/vastai.py. The v4 design
> resolves issue #19.
>
> The ninth draft (applied to 12th-pass findings) introduced:
>
> 1. **`OwnershipVerification` tagged enum** replacing the
>    conflated `bool`. Verify the four-way contract: `OWNED` for
>    present-and-matching, `ABSENT` for present-but-not-in-API,
>    `REFUSED` for unowned-or-malformed, `DISABLED` for ownership
>    checking off.
> 2. **`_normalize_instance_id` shared normalizer** between
>    `list_vastai_instances` and `verify_instance_ownership`,
>    rejecting invalid IDs (`None`/`bool`/empty/blank) and
>    canonicalising valid ones. The verifier validates the entire
>    response (no early-return on first match) before locating
>    the requested record.
> 3. **`except Exception` outermost boundary** on the verifier
>    with `ERROR`-level log for every failure path (was `WARNING`
>    on the original two paths).
> 4. **`ALREADY_GONE` v4 verdict** added to `CleanupVerdict`;
>    factory's CLI fallback short-circuits `ABSENT` to it
>    WITHOUT invoking CLI destroy.
> 5. **`_describe_destroy_result` for `CREDENTIALS_DISABLED`**.
> 6. **Test catalogue** exhaustively covers the new branches
>    (parameterized invalid-ID, malformed-record-BEFORE/AFTER,
>    numeric/padded IDs, caplog severity, factory
>    ABSENT-short-circuits-destroy).
>
> Additionally, identify any new BLOCKERs or CONCERNs. Focus on:
>
> - Does the new `ALREADY_GONE` verdict integrate cleanly with
>   the orchestrator's logging and kill-counting logic? (It
>   should NOT count toward `killed`, and it should log at
>   `INFO`.)
> - Does the tagged-result dispatch in `_cli_fallback` cover all
>   four cases without a default clause? (Adding a `match`
>   statement with an exhaustive `case` would prove it.)
> - Are there any places in the existing v2 code or v3 design
>   that pass `bool` (the old return type) to anything that
>   now expects `OwnershipVerification`? (The migration
>   checklist should call this out explicitly.)
> - Does the `OwnershipVerification` enum's `DISABLED` value
>   correctly distinguish from `OWNED`? (With no policy, every
>   instance is treated as owned — but we shouldn't pretend we
>   confirmed its presence. `DISABLED` is "no policy check was
>   run", `OWNED` is "the API confirmed presence".)
> - Could the new `ALREADY_GONE` verdict collide with a future
>   v3 verdict translation? (DESTROYED → DESTROYED; LEAKED →
>   LEAKED; UNKNOWN → UNKNOWN; new ALREADY_GONE has no v3
>   counterpart, so no collisions.)
> - Does the test plan still cover all the v3 prerequisite
>   scenarios from migration step 6 (14 integration scenarios)?
>
> Return a labeled list of findings. Each finding is one of:
> BLOCKER (must fix before merge), CONCERN (should fix, but not
> blocking), or NIT (nice to have). For each finding, give the
> exact line range, the issue, and the proposed fix. If the
> design is acceptable as-is, say "DESIGN ACCEPTED" with a
> one-line rationale.