"""Shared orchestration patterns for cloud GPU batch workloads.

The v4 architecture removed ``sweep_zombie_instances`` in favour of a
``ProviderCleanupPolicy`` (see ``batch.BatchOrchestrator._sweep_zombies``
and ``providers.vastai.build_vastai_cleanup_policy``). Only
``check_budget`` survives — the cost-ceiling check the
``BatchOrchestrator._deploy_budget_ok`` and
``BatchOrchestrator._poll_budget_ok`` methods consult.

DELETED in v4 step 7:

- ``sweep_zombie_instances`` and helpers
  (``_fetch_vastai_instances``, ``_sweep_zombies_for_instances``,
  ``_is_zombie``, ``_r2_says_done``, ``_destroy_zombie``,
  ``_log_sweep_outcome``). External callers must construct a
  ``ProviderCleanupPolicy`` and call ``policy.destroy(candidate)``.

DELETED earlier (v3 step 7):

- The v3 destroy adapter's ``read_vastai_api_key()`` (env-first,
  fail-closed) replaces the legacy helper.
- ``_destroy_via_rest()`` — absorbed into the v3 destroy adapter.
- ``ensure_detached()`` — dead public API, no callers in this repo.
- ``poll_instance_progress()`` — dead public API; the v3
  ``decide_next_action`` (unit_lifecycle.py) replaces the
  per-unit classification logic.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def check_budget(spent: float, ceiling: float) -> bool:
    """Check if cloud spend is within budget.

    Args:
        spent: Total spent so far in USD.
        ceiling: Budget ceiling in USD.

    Returns:
        True if within budget, False if over.
    """
    if spent >= ceiling:
        logger.error("BUDGET EXCEEDED: $%.2f >= $%.2f ceiling", spent, ceiling)
        return False
    if spent >= ceiling * 0.8:
        pct = spent / ceiling * 100
        logger.warning("BUDGET WARNING: $%.2f (%.0f%% of $%.2f)", spent, pct, ceiling)
    return True
