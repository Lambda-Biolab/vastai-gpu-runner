"""Tests for the zombie sweep classifier in vastai_gpu_runner.orchestrator."""

from __future__ import annotations

from vastai_gpu_runner.orchestrator import _is_zombie


def _inst(iid: str, label: str, status: str) -> dict[str, object]:
    """Build a minimal Vast.ai-API-shaped instance dict for classifier tests."""
    return {"id": iid, "label": label, "cur_state": status}


LABEL_PREFIX = "oralamp-md-abc123"


class TestIsZombie:
    """Covers the ``_is_zombie`` decision tree.

    Policy (2026-04-20 hardening): the sweep is for **orphans only**.
    Tracked instances are always spared because Vast.ai's ``cur_state`` is
    an unreliable indicator of actual container health: confirmed via SSH
    that workers keep running at 80% GPU utilisation while the API
    persistently reports ``stopped`` / ``exited``. The orchestrator's
    collect/destroy flow owns tracked-instance cleanup.
    """

    def test_label_prefix_mismatch_never_zombie(self) -> None:
        # An instance belonging to a different batch must never be swept.
        inst = _inst("1", "someone-elses-batch-xyz", "stopped")
        assert _is_zombie(inst, LABEL_PREFIX, {"1"}, r2_sink=None, r2_batch_id="") is False

    def test_tracked_running_is_not_zombie(self) -> None:
        inst = _inst("1", f"{LABEL_PREFIX}-x", "running")
        assert _is_zombie(inst, LABEL_PREFIX, {"1"}, r2_sink=None, r2_batch_id="") is False

    def test_tracked_stopped_is_not_zombie(self) -> None:
        """Load-bearing regression for the 2026-04-20 cascade.

        Vast.ai reported ``cur_state=stopped`` repeatedly for a tracked
        instance whose OpenMM worker was running at 90% GPU utilisation,
        confirmed via SSH. The sweep must not destroy tracked instances.
        """
        inst = _inst("42", f"{LABEL_PREFIX}-still-running", "stopped")
        assert _is_zombie(inst, LABEL_PREFIX, {"42"}, r2_sink=None, r2_batch_id="") is False

    def test_tracked_exited_is_not_zombie(self) -> None:
        """Even if Vast.ai reports the container as exited, the sweep must
        spare tracked instances. Real exits are harvested by the collect phase."""
        inst = _inst("42", f"{LABEL_PREFIX}-tracked", "exited")
        assert _is_zombie(inst, LABEL_PREFIX, {"42"}, r2_sink=None, r2_batch_id="") is False

    def test_untracked_with_matching_label_is_zombie(self) -> None:
        """Orphan: label matches our prefix but not in tracked_ids."""
        inst = _inst("999", f"{LABEL_PREFIX}-orphan", "running")
        assert _is_zombie(inst, LABEL_PREFIX, {"1"}, r2_sink=None, r2_batch_id="") is True

    def test_untracked_stopped_is_zombie(self) -> None:
        """Orphan with stopped state is still a zombie."""
        inst = _inst("999", f"{LABEL_PREFIX}-orphan", "stopped")
        assert _is_zombie(inst, LABEL_PREFIX, {"1"}, r2_sink=None, r2_batch_id="") is True

    def test_r2_done_marker_path_inert_for_untracked(self) -> None:
        """R2 DONE check returns False for untracked instances (only applies to
        tracked stopped/exited — which the new policy already spares). The R2
        check path is effectively dead for orphans; orphans still get destroyed."""
        inst = _inst("999", f"{LABEL_PREFIX}-orphan", "stopped")

        class _FakeR2:
            @staticmethod
            def is_job_done(_batch_id: str, _job: str) -> bool:
                return True

        assert (
            _is_zombie(
                inst,
                LABEL_PREFIX,
                tracked_ids={"1"},
                r2_sink=_FakeR2(),
                r2_batch_id="batch-xyz",
            )
            is True
        )
