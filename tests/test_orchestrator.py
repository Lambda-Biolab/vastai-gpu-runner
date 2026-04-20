"""Tests for the zombie sweep classifier in vastai_gpu_runner.orchestrator."""

from __future__ import annotations

import time

from vastai_gpu_runner.orchestrator import (
    _ZOMBIE_GRACE_SECONDS,
    _is_zombie,
    _within_grace_period,
)


def _inst(
    iid: str,
    label: str,
    status: str,
    *,
    start_date: float | None = None,
) -> dict[str, object]:
    """Build a minimal Vast.ai-API-shaped instance dict for classifier tests."""
    d: dict[str, object] = {"id": iid, "label": label, "cur_state": status}
    if start_date is not None:
        d["start_date"] = start_date
    return d


LABEL_PREFIX = "oralamp-md-abc123"


class TestIsZombie:
    """Covers the ``_is_zombie`` decision tree."""

    def test_label_prefix_mismatch_never_zombie(self) -> None:
        # An instance belonging to a different batch must never be swept.
        inst = _inst("1", "someone-elses-batch-xyz", "stopped", start_date=time.time() - 3600)
        assert (
            _is_zombie(inst, LABEL_PREFIX, tracked_ids={"1"}, r2_sink=None, r2_batch_id="") is False
        )

    def test_tracked_running_is_not_zombie(self) -> None:
        inst = _inst("1", f"{LABEL_PREFIX}-x", "running", start_date=time.time() - 3600)
        assert (
            _is_zombie(inst, LABEL_PREFIX, tracked_ids={"1"}, r2_sink=None, r2_batch_id="") is False
        )

    def test_tracked_stopped_within_grace_is_not_zombie(self) -> None:
        """Regression for the cascading destroy bug seen on 2026-04-20.

        Vast.ai briefly reports cur_state=stopped during container boot. A
        tracked instance younger than the grace period must be spared even
        if the API currently shows it as stopped.
        """
        inst = _inst(
            "42",
            f"{LABEL_PREFIX}-candidate-x",
            "stopped",
            start_date=time.time() - 60,  # 1 min old
        )
        assert (
            _is_zombie(inst, LABEL_PREFIX, tracked_ids={"42"}, r2_sink=None, r2_batch_id="")
            is False
        )

    def test_tracked_stopped_past_grace_is_zombie(self) -> None:
        """Stopped for real: past the grace window with no running worker."""
        inst = _inst(
            "42",
            f"{LABEL_PREFIX}-candidate-x",
            "stopped",
            start_date=time.time() - (_ZOMBIE_GRACE_SECONDS + 60),
        )
        assert (
            _is_zombie(inst, LABEL_PREFIX, tracked_ids={"42"}, r2_sink=None, r2_batch_id="") is True
        )

    def test_tracked_exited_past_grace_is_zombie(self) -> None:
        inst = _inst(
            "42",
            f"{LABEL_PREFIX}-candidate-x",
            "exited",
            start_date=time.time() - (_ZOMBIE_GRACE_SECONDS + 60),
        )
        assert (
            _is_zombie(inst, LABEL_PREFIX, tracked_ids={"42"}, r2_sink=None, r2_batch_id="") is True
        )

    def test_untracked_with_matching_label_is_zombie(self) -> None:
        """Orphans whose label matches our batch prefix but aren't in tracked_ids."""
        inst = _inst("999", f"{LABEL_PREFIX}-orphan", "running", start_date=time.time() - 60)
        assert (
            _is_zombie(inst, LABEL_PREFIX, tracked_ids={"1"}, r2_sink=None, r2_batch_id="") is True
        )

    def test_r2_done_marker_spares_stopped_instance(self) -> None:
        """When R2 confirms the job is done, the sweep should not destroy the instance."""
        inst = _inst(
            "42",
            f"{LABEL_PREFIX}-done-job",
            "stopped",
            start_date=time.time() - (_ZOMBIE_GRACE_SECONDS + 60),
        )

        class _FakeR2:
            def is_job_done(self, _batch_id: str, _job: str) -> bool:
                return True

        assert (
            _is_zombie(
                inst,
                LABEL_PREFIX,
                tracked_ids={"42"},
                r2_sink=_FakeR2(),
                r2_batch_id="batch-xyz",
            )
            is False
        )

    def test_missing_start_date_does_not_grant_grace(self) -> None:
        """Instances missing ``start_date`` fall back to the normal rule."""
        inst = _inst("42", f"{LABEL_PREFIX}-x", "stopped")
        assert (
            _is_zombie(inst, LABEL_PREFIX, tracked_ids={"42"}, r2_sink=None, r2_batch_id="") is True
        )


class TestWithinGracePeriod:
    """Unit tests for the ``_within_grace_period`` helper."""

    def test_recent_start(self) -> None:
        assert _within_grace_period({"start_date": time.time() - 30}) is True

    def test_past_grace(self) -> None:
        assert (
            _within_grace_period({"start_date": time.time() - (_ZOMBIE_GRACE_SECONDS + 60)})
            is False
        )

    def test_missing_field(self) -> None:
        assert _within_grace_period({}) is False

    def test_non_numeric_field(self) -> None:
        assert _within_grace_period({"start_date": "not a number"}) is False
