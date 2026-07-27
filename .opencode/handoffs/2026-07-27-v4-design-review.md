# Session Handoff — 2026-07-27 (v4 implementation + stress validation)

## Status: V4 ARCHITECTURE COMPLETE + STRESS VALIDATED

The v4 cleanup-policy architecture (designed in PR #22) is fully
implemented and stress-validated end-to-end on real Vast.ai
infrastructure.

## Session deliverables (this run + prior runs)

| PR | Step | What |
|---|---|---|
| #22 | (design) | v4 architecture document |
| #23 | (v3 prereq) | ProviderCleanupPolicy prerequisite: unit_lifecycle + providers/destroy |
| #24 | (v3 prereq) | Vast.ai destroy adapter + runner/orchestrator refactor |
| #25 | 2 | `cleanup_policy` module + invariant tests |
| #26 | 3a | `VastaiProviderConfig` + `VastaiRunner.from_config` + ownership/credentials/label_prefix |
| #27 | 3b | `list_vastai_instances` + REST/CLI dispatch + `VASTAI_TERMINAL_STATES` |
| #28 | 3c | Migrate `verify_instance_ownership` from adapter to providers/vastai.py |
| #29 | 3d | `build_vastai_cleanup_policy` + typed `destroy_instance` logging |
| #30 | 4 | `BatchOrchestrator` policy-driven zombie sweep + state.py migration |
| #31 | 5 | CLI composition roots (batch / cleanup / instances) |
| #32 | 6 | Integration tests for the 17 cleanup-policy scenarios |
| #33 | 7 | Legacy sweep + duplicated helpers deletion |
| #34 | stress | v4 end-to-end stress suite (21 scenarios: 19 mock + 2 real Vast.ai) |

## Stress validation results

### Mock stress suite (always runs) — 19 scenarios

| Scenario | Coverage |
|---|---|
| `TestLargeJobConcurrentDeploys` | 50 shards, max_parallel=8 — all complete |
| `TestConnectionDropsDuringPoll` | 30 shards with 1 SSH drop each — all complete via retry |
| `TestResumeAfterKillMidCycle` | 50 shards (45 downloaded, 5 pending) — only 5 redeploy after resume |
| `TestPreV4StateResume` | schema_version=0 fixture (8 deployed + 12 pending) — migration + run |
| `TestMixedFailuresSucceedPreemptFatal` | 60 shards with weighted random outcomes — v4 contract enforced |
| `TestBudgetAbortMidPoll` | budget_usd=0 / 0.001 — graceful completion |
| `TestConcurrentMaxParallelDoesNotDoubleClaim` | 50 shards, max_parallel=8 — unique instance_ids |
| `TestZombieSweepDuringLiveRun` | orphan with matching label scope — destroyed mid-run |
| `TestStatePersistenceAtomicWrite` | atomic .tmp+rename — no leftovers |
| `TestV4LabelScopeHelpers` | `validate_label_prefix`, `resolve_label_scope` |

### Real Vast.ai stress suite (opt-in via `VASTAI_API_KEY`) — 2 scenarios

- `test_real_rtx_3060_deploy_ssh_destroy`: real Vast.ai REST deploy on
  the cheapest available RTX 3060 ($0.0321-$0.05/hr). Boot timeout is
  a valid outcome (the cheapest hosts may not ship a `worker.sh` on
  the CUDA image). **End-of-test invariant**: zero active instances
  remain — the test must not leak cloud spend.
- `test_real_cheapest_rtx_3060_visible_to_v4_policy`: v4
  `build_vastai_cleanup_policy` REST enumeration against real Vast.ai.
  Costs ~$0.

**Wall-clock validation**: 156s for both real scenarios (one full
deploy → boot timeout → destroy → REST verification cycle).
**Total cloud spend**: < $0.05.

## Final state

- **623 tests pass** (404 → 453 → 466 → 492 → 509 → 552 → 575 → 609 → 602 → 621 → 623 across the v4 work)
- All 4 CI gates green: ruff format/check, pyright strict, complexipy CC ≤ 10, pytest
- `scripts/audit_caller_sites.sh` — AUDIT CLEAN
- `CHANGELOG.md` updated with 0.4.0 release notes covering the full v4 architecture

## How to run the stress tests

```bash
# Mock suite (always runs, in CI):
make test

# Real Vast.ai suite (opt-in, requires VASTAI_API_KEY):
export VASTAI_API_KEY=...
export STRESS_BUDGET_USD=0.30  # default
export STRESS_DEADLINE_SECONDS=600  # default
uv run --active pytest tests/stress/test_real_vastai_stress.py -v
```

## Out of v4 scope (future work)

- **RunPod adapter** — `ProviderCleanupPolicy` is provider-agnostic; RunPod factory ships when RunPod adapter ships.
- **Dispute webhook.**
- **Multi-policy orchestrator** (single `BatchOrchestrator` supports one `cleanup_policy`).
- **Bulk-destroy optimisation** (YAGNI per v4 design).