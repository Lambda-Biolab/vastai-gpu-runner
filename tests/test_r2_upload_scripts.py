# pyright: reportPrivateUsage=warning, reportMissingParameterType=warning
"""Tests for the auto-generated R2 uploader scripts.

These run the generated Python scripts directly under a temporary
``boto3`` stub so we can verify the fail-closed semantics — DONE
marker must NOT be published when required uploads fail.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from vastai_gpu_runner.storage.r2 import R2Sink


class _StubClient:
    """Inert boto3 client stand-in for ``R2Sink.__init__`` in the test process."""

    def put_object(self, **kwargs):
        return {}

    def upload_file(self, *args, **kwargs):
        return {}

    def head_object(self, **kwargs):
        return {}

    def list_objects_v2(self, **kwargs):
        return {}

    def delete_objects(self, **kwargs):
        return {}

    def get_paginator(self, *args, **kwargs):
        class _P:
            def paginate(self, **kwargs):
                return []

        return _P()


@pytest.fixture(autouse=True)
def _stub_boto3_client(monkeypatch):
    """Replace ``boto3.client`` with an inert stub so ``R2Sink.__init__`` works
    in the test process even when ``R2_ENDPOINT`` is empty.

    Tests construct ``R2Sink`` instances in the test process; ``R2Sink.__init__``
    eagerly calls ``boto3.client(...)`` with whatever ``R2_ENDPOINT`` env var
    is set. In CI, that env var may be empty, and botocore validates the
    endpoint URL at ``boto3.client`` time. This fixture is autouse so every
    test in this module runs against the stub.

    The actual upload scripts run in subprocesses with a separately-injected
    fake boto3 (see ``_run_script``), so this stub does not affect the
    behaviour we are testing.
    """
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: _StubClient())


# ---------------------------------------------------------------------------
# Fake boto3 — written to a temp module so ``import boto3`` resolves inside
# the generated scripts.
# ---------------------------------------------------------------------------


FAKE_BOTO3_TEMPLATE = """
import json
import os
import sys

CALLS = []
PUT_OBJECT_FAIL = __PUT_OBJECT_FAIL__
UPLOAD_FILE_FAIL = __UPLOAD_FILE_FAIL__
CALLS_LOG = __CALLS_LOG__


class _StubClient:
    def put_object(self, *, Bucket, Key, Body=b""):
        CALLS.append(("put_object", dict(Bucket=Bucket, Key=Key)))
        if PUT_OBJECT_FAIL and Key == PUT_OBJECT_FAIL:
            raise RuntimeError("put_object failed for " + str(Key))
        return {}

    def upload_file(self, Filename, Bucket, Key, **kwargs):
        CALLS.append(("upload_file", dict(Filename=Filename, Bucket=Bucket, Key=Key)))
        if UPLOAD_FILE_FAIL and Key == UPLOAD_FILE_FAIL:
            raise RuntimeError("upload_file failed for " + str(Key))
        return {}

    def head_object(self, *, Bucket, Key):
        for method, kw in reversed(CALLS):
            if method == "put_object" and kw["Key"] == Key:
                return {}
        raise RuntimeError("not found")

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys=1):
        return {"Contents": [], "CommonPrefixes": []}


def _persist_calls() -> None:
    try:
        with open(CALLS_LOG, "w") as f:
            json.dump(CALLS, f)
    except Exception:
        pass


import atexit as _atexit
_atexit.register(_persist_calls)


def client(*args, **kwargs):
    return _StubClient()
"""


def _make_fake_boto3(
    tmp_path: Path,
    *,
    put_object_fail: str = "",
    upload_file_fail: str = "",
    calls_log: Path,
) -> Path:
    """Write a fake ``boto3`` module to tmp_path and return its directory.

    The fake boto3 records all calls to ``calls_log`` as JSON, so tests
    can assert which keys were uploaded and in what order.
    """
    fake_dir = tmp_path / "_fakelib"
    fake_dir.mkdir()
    body = FAKE_BOTO3_TEMPLATE
    body = body.replace("__PUT_OBJECT_FAIL__", repr(put_object_fail))
    body = body.replace("__UPLOAD_FILE_FAIL__", repr(upload_file_fail))
    body = body.replace("__CALLS_LOG__", repr(str(calls_log)))
    (fake_dir / "boto3.py").write_text(body)
    return fake_dir


def _patch_script_to_dump_calls(script: str, calls_log: Path) -> str:
    """Deprecated — persistence now happens via atexit in the fake boto3.

    Kept as a no-op so existing call sites remain readable.
    """
    return script


def _run_script(
    script: str,
    *,
    args: list[str],
    workspace: Path,
    fake_boto3_dir: Path,
) -> subprocess.CompletedProcess[str]:
    """Execute a generated upload script with a fake boto3 on sys.path.

    The generated scripts do ``import boto3`` at module-level, so we
    prepend ``sys.path.insert`` *before* that import — setting only
    PYTHONPATH does not override an installed ``boto3`` in
    site-packages. Also overrides the script's WORKSPACE constant to
    point at *workspace*.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{fake_boto3_dir}{os.pathsep}{existing}" if existing else str(fake_boto3_dir)
    )
    env["R2_ENDPOINT"] = "https://r2.example"
    env.setdefault("R2_ACCESS_KEY_ID", "test-akey")
    env.setdefault("R2_SECRET_ACCESS_KEY", "test-secret-fixture")
    env.setdefault("VASTAI_INSTANCE_ID", "")
    env.setdefault("VASTAI_API_KEY", "")

    prologue = (
        "import sys as _sys\n"
        "_sys.path[:] = [p for p in _sys.path if 'site-packages' not in p "
        "and 'dist-packages' not in p]\n"
        f"_sys.path.insert(0, {str(fake_boto3_dir)!r})\n"
        # Override WORKSPACE so the script reads files from the test
        # directory instead of the hard-coded /workspace.
        f"WORKSPACE_OVERRIDE = {str(workspace)!r}\n"
    )
    full_script = prologue + script.replace(
        'WORKSPACE = "/workspace"',
        "WORKSPACE = WORKSPACE_OVERRIDE",
    )

    return subprocess.run(
        [sys.executable, "-c", full_script, *args],
        capture_output=True,
        text=True,
        cwd=workspace,
        env=env,
        timeout=30,
        check=False,
    )


def _read_calls(calls_log: Path) -> list[tuple[str, dict[str, Any]]]:
    """Load the persisted call log from the fake boto3 module."""
    if not calls_log.exists():
        return []
    raw = calls_log.read_text()
    data = json.loads(raw)
    return [(c[0], c[1]) for c in data]


# ---------------------------------------------------------------------------
# Shard uploader — fail-closed behaviour
# ---------------------------------------------------------------------------


class TestShardUploaderFailClosed:
    def _sink(self) -> R2Sink:
        return R2Sink(bucket="bkt", prefix="project")

    def _calls(self, log: Path) -> list[tuple[str, dict[str, Any]]]:
        return _read_calls(log)

    def _done_calls(self, calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        return [
            kw for method, kw in calls if method == "put_object" and kw["Key"].endswith("/DONE")
        ]

    def test_full_success_publishes_done(
        self,
        tmp_path: Path,
    ) -> None:
        sink = self._sink()
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(tmp_path, calls_log=calls_log)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "outputs" / "pred_a.txt").write_text("hello")
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )

        # First upload the flat file via the no-arg mode so the
        # whole-shard completion marker is written. --done then
        # succeeds with the exitcode + DONE publication.
        upload_all = _run_script(
            script,
            args=[],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert upload_all.returncode == 0, upload_all.stderr
        assert (workspace / "shard_completed").exists()

        result = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        assert result.returncode == 0, result.stderr
        calls = self._calls(calls_log)
        # DONE marker appears at least twice: once from upload_all,
        # once from --done. (The first is per-prediction's
        # upload_all path; we want to ensure --done published at
        # least one fresh DONE.)
        assert len(self._done_calls(calls)) >= 1, calls

    def test_exitcode_upload_failure_omits_done(
        self,
        tmp_path: Path,
    ) -> None:
        sink = self._sink()
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(
            tmp_path,
            upload_file_fail="project/batch1/shard_0/worker.exitcode",
            calls_log=calls_log,
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )
        result = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        assert result.returncode != 0, (
            f"non-zero exit expected on exitcode failure. rc={result.returncode} "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        calls = self._calls(calls_log)
        assert self._done_calls(calls) == [], "DONE marker must NOT be published"
        assert "FAIL" in result.stderr

    def test_required_file_failure_omits_done(
        self,
        tmp_path: Path,
    ) -> None:
        sink = self._sink()
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(
            tmp_path,
            upload_file_fail="project/batch1/shard_0/outputs/pred_a.txt",
            calls_log=calls_log,
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "outputs" / "pred_a.txt").write_text("hello")
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )
        result = _run_script(
            script,
            args=[],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        assert result.returncode != 0
        calls = self._calls(calls_log)
        assert self._done_calls(calls) == []
        assert "FAIL" in result.stderr

    def test_done_marker_publish_failure_returns_non_zero(
        self,
        tmp_path: Path,
    ) -> None:
        sink = self._sink()
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(
            tmp_path,
            put_object_fail="project/batch1/shard_0/DONE",
            calls_log=calls_log,
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )
        result = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        assert result.returncode != 0
        assert "FAIL" in result.stderr

    def test_missing_outputs_dir_is_ok(self, tmp_path: Path) -> None:
        """--done mode with no outputs directory should still succeed when exitcode exists."""
        sink = self._sink()
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(tmp_path, calls_log=calls_log)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        # No outputs dir at all. Worker.exitcode exists.
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )
        result = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        assert result.returncode == 0, result.stderr

    def test_missing_worker_exitcode_omits_done(self, tmp_path: Path) -> None:
        """--done mode without worker.exitcode MUST refuse to publish DONE."""
        sink = self._sink()
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(tmp_path, calls_log=calls_log)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "outputs" / "pred.txt").write_text("hi")
        # NO worker.exitcode — outcome unverifiable.

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )
        result = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        assert result.returncode != 0
        # The sentinel check fires before the exitcode check, so the
        # error message references unresolved failures (flat file in
        # outputs/ without shard_complete) rather than the
        # worker.exitcode absence message. Either is acceptable —
        # what matters is that DONE is not published.
        calls = _read_calls(calls_log)
        done_calls = [
            kw for method, kw in calls if method == "put_object" and kw["Key"].endswith("/DONE")
        ]
        assert done_calls == [], "DONE must NOT be published when exitcode is absent"

    def test_upload_all_missing_worker_exitcode_omits_done(
        self,
        tmp_path: Path,
    ) -> None:
        """upload_all (no args) also refuses to publish DONE without exitcode."""
        sink = self._sink()
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(tmp_path, calls_log=calls_log)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "outputs" / "pred.txt").write_text("hi")
        # NO worker.exitcode.

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )
        result = _run_script(
            script,
            args=[],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        assert result.returncode != 0
        # The shard_complete marker must NOT be present on failure.
        assert not (workspace / "shard_completed").exists()
        calls = _read_calls(calls_log)
        done_calls = [
            kw for method, kw in calls if method == "put_object" and kw["Key"].endswith("/DONE")
        ]
        assert done_calls == []


# ---------------------------------------------------------------------------
# Job uploader — fail-closed behaviour
# ---------------------------------------------------------------------------


class TestJobUploaderFailClosed:
    def _sink(self) -> R2Sink:
        return R2Sink(bucket="bkt", prefix="project")

    def _calls(self, log: Path) -> list[tuple[str, dict[str, Any]]]:
        return _read_calls(log)

    def _done_calls(self, calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        return [
            kw for method, kw in calls if method == "put_object" and kw["Key"].endswith("/DONE")
        ]

    def test_full_success_publishes_done(
        self,
        tmp_path: Path,
    ) -> None:
        sink = self._sink()
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(tmp_path, calls_log=calls_log)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "output").mkdir()
        (workspace / "output" / "checkpoint.txt").write_text("ok")
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_job_upload_script("batch1", "jobA"),
            calls_log,
        )
        result = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        assert result.returncode == 0, result.stderr
        calls = self._calls(calls_log)
        assert len(self._done_calls(calls)) == 1

    def test_required_file_failure_omits_done(
        self,
        tmp_path: Path,
    ) -> None:
        sink = self._sink()
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(
            tmp_path,
            upload_file_fail="project/batch1/jobA/checkpoint.txt",
            calls_log=calls_log,
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "output").mkdir()
        (workspace / "output" / "checkpoint.txt").write_text("ok")
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_job_upload_script("batch1", "jobA"),
            calls_log,
        )
        result = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        assert result.returncode != 0
        calls = self._calls(calls_log)
        assert self._done_calls(calls) == []

    def test_exitcode_failure_omits_done(
        self,
        tmp_path: Path,
    ) -> None:
        sink = self._sink()
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(
            tmp_path,
            upload_file_fail="project/batch1/jobA/worker.exitcode",
            calls_log=calls_log,
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "output").mkdir()
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_job_upload_script("batch1", "jobA"),
            calls_log,
        )
        result = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        assert result.returncode != 0
        calls = self._calls(calls_log)
        assert self._done_calls(calls) == []

    def test_checkpoint_does_not_publish_done(
        self,
        tmp_path: Path,
    ) -> None:
        """Checkpoint mode is best-effort; must NEVER publish DONE."""
        sink = self._sink()
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(tmp_path, calls_log=calls_log)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "output").mkdir()
        (workspace / "output" / "checkpoint.txt").write_text("ok")
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_job_upload_script("batch1", "jobA"),
            calls_log,
        )
        result = _run_script(
            script,
            args=["--checkpoint"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        assert result.returncode == 0, result.stderr
        calls = self._calls(calls_log)
        assert self._done_calls(calls) == [], "checkpoint must not publish DONE"

    def test_checkpoint_failure_is_non_fatal(
        self,
        tmp_path: Path,
    ) -> None:
        sink = self._sink()
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(
            tmp_path,
            upload_file_fail="project/batch1/jobA/checkpoint.txt",
            calls_log=calls_log,
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "output").mkdir()
        (workspace / "output" / "checkpoint.txt").write_text("ok")

        script = _patch_script_to_dump_calls(
            sink.generate_job_upload_script("batch1", "jobA"),
            calls_log,
        )
        result = _run_script(
            script,
            args=["--checkpoint"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        # Best-effort: failure logged as warning, exit 0.
        assert result.returncode == 0, result.stderr
        assert "WARN" in result.stdout

    def test_chunk_no_new_bytes_is_not_failure(
        self,
        tmp_path: Path,
    ) -> None:
        """``_flush_large_file_chunk`` returning ``"none"`` must not be flagged as a failure."""
        sink = self._sink()
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(tmp_path, calls_log=calls_log)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "output").mkdir()
        # Large file does NOT exist yet — _flush returns "none".
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_job_upload_script(
                "batch1",
                "jobA",
                large_file="traj.dcd",
            ),
            calls_log,
        )
        result = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        assert result.returncode == 0, result.stderr
        calls = self._calls(calls_log)
        assert len(self._done_calls(calls)) == 1

    def test_chunk_flush_failure_omits_done(
        self,
        tmp_path: Path,
    ) -> None:
        sink = self._sink()
        calls_log = tmp_path / "calls.json"
        # The chunk key is computed from large_file stem + chunk_index.
        # Index 0, large_file = traj.dcd -> "traj_chunk_000.dcd"
        fake_dir = _make_fake_boto3(
            tmp_path,
            upload_file_fail="project/batch1/jobA/traj_chunk_000.dcd",
            calls_log=calls_log,
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "output").mkdir()
        (workspace / "output" / "traj.dcd").write_text("data")
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_job_upload_script(
                "batch1",
                "jobA",
                large_file="traj.dcd",
            ),
            calls_log,
        )
        result = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        assert result.returncode != 0
        calls = self._calls(calls_log)
        assert self._done_calls(calls) == []


class TestShardPredictionFailClosed:
    """``--prediction`` must omit DONE markers when required uploads fail."""

    def test_failed_prediction_omits_done_markers(
        self,
        tmp_path: Path,
    ) -> None:
        sink = R2Sink(bucket="bkt", prefix="project")
        calls_log = tmp_path / "calls.json"
        # The upload key includes the prediction dir as a sub-prefix.
        fake_dir = _make_fake_boto3(
            tmp_path,
            upload_file_fail="project/batch1/shard_0/outputs/pred/pred.txt",
            calls_log=calls_log,
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "outputs" / "pred").mkdir()
        (workspace / "outputs" / "pred" / "pred.txt").write_text("hello")

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )
        result = _run_script(
            script,
            args=["--prediction", "pred"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        # Required uploads failed; script must exit non-zero and omit BOTH
        # the per-shard and global_markers DONE markers.
        assert result.returncode != 0, result.stderr
        combined = result.stdout + result.stderr
        assert "FAIL" in combined or "failed" in combined
        calls = _read_calls(calls_log)
        marker_puts = [
            kw
            for method, kw in calls
            if method == "put_object"
            and ("markers/pred.done" in kw["Key"] or "global_markers/pred.done" in kw["Key"])
        ]
        assert marker_puts == [], (
            "Failed prediction must NOT publish per-shard or global DONE markers"
        )

    def test_missing_outputs_dir_for_prediction_exits_zero(
        self,
        tmp_path: Path,
    ) -> None:
        """--prediction with no output dir is not a failure (warning, exit 0)."""
        sink = R2Sink(bucket="bkt", prefix="project")
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(tmp_path, calls_log=calls_log)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        # No "missing_pred" directory created.

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )
        result = _run_script(
            script,
            args=["--prediction", "missing_pred"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        assert result.returncode == 0
        assert "no output dir" in result.stdout + result.stderr


class TestShardPredictionSingleSuccess:
    """A successful prediction with exactly one file MUST publish both markers."""

    def test_single_file_success_exits_zero(
        self,
        tmp_path: Path,
    ) -> None:
        sink = R2Sink(bucket="bkt", prefix="project")
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(tmp_path, calls_log=calls_log)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "outputs" / "pred").mkdir()
        (workspace / "outputs" / "pred" / "only.txt").write_text("hi")

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )
        result = _run_script(
            script,
            args=["--prediction", "pred"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )

        # One successful upload must exit 0 and publish BOTH markers.
        assert result.returncode == 0, result.stderr
        combined = result.stdout + result.stderr
        assert "uploaded (1 files)" in combined or "uploaded (1 " in combined
        calls = _read_calls(calls_log)
        marker_puts = [
            kw
            for method, kw in calls
            if method == "put_object"
            and ("markers/pred.done" in kw["Key"] or "global_markers/pred.done" in kw["Key"])
        ]
        assert len(marker_puts) == 2, f"Expected 2 markers (per-shard + global), got {marker_puts}"


class TestShardSequentialPredictionFailure:
    """A failed ``--prediction`` MUST prevent later ``--done`` from publishing DONE.

    This is the critical sequential flow: a transient upload failure
    followed by a successful workload exit MUST NOT let the top-level
    shard DONE marker get published. The local upload-failure
    sentinel persists across script invocations so any subsequent
    ``--done`` (or no-arg) invocation sees it.
    """

    def test_failed_prediction_then_successful_done_omits_shard_done(
        self,
        tmp_path: Path,
    ) -> None:
        sink = R2Sink(bucket="bkt", prefix="project")
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(
            tmp_path,
            upload_file_fail="project/batch1/shard_0/outputs/pred/pred.txt",
            calls_log=calls_log,
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "outputs" / "pred").mkdir()
        (workspace / "outputs" / "pred" / "pred.txt").write_text("hi")
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )

        # First invocation: --prediction fails on the file upload.
        result1 = _run_script(
            script,
            args=["--prediction", "pred"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert result1.returncode != 0
        # The sentinel must exist locally after the failed prediction.
        assert (workspace / "upload_failures.log").exists(), (
            "Failed prediction MUST persist a local upload-failure sentinel"
        )

        # Second invocation: --done MUST refuse to publish the shard
        # DONE marker because the sentinel lists an unresolved failure.
        result2 = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert result2.returncode != 0
        combined = result2.stdout + result2.stderr
        assert (
            "unresolved per-prediction upload failures" in combined
            or "upload_failures.log" in combined
        )

        # The shard DONE marker must NOT appear in the call log.
        calls = _read_calls(calls_log)
        shard_done_calls = [
            kw
            for method, kw in calls
            if method == "put_object"
            and kw["Key"].endswith("/DONE")
            and "/markers/" not in kw["Key"]
            and "/global_markers/" not in kw["Key"]
        ]
        assert shard_done_calls == [], (
            f"Shard DONE must NOT be published when sentinel present; got {shard_done_calls}"
        )

    def test_successful_prediction_clears_sentinel(
        self,
        tmp_path: Path,
    ) -> None:
        """A successful ``--prediction`` after a failed one clears that prediction's line.

        We exercise the genuine retry path: first ``--prediction``
        fails and persists a sentinel line, then we rebuild the fake
        so the upload succeeds, re-run ``--prediction``, and verify
        the sentinel is cleared (and the completion marker is
        written). The subsequent ``--done`` then succeeds.
        """
        sink = R2Sink(bucket="bkt", prefix="project")
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(
            tmp_path,
            upload_file_fail="project/batch1/shard_0/outputs/pred/pred.txt",
            calls_log=calls_log,
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "outputs" / "pred").mkdir()
        (workspace / "outputs" / "pred" / "pred.txt").write_text("hi")
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )

        # First --prediction fails and writes the sentinel.
        r1 = _run_script(
            script,
            args=["--prediction", "pred"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert r1.returncode != 0
        assert (workspace / "upload_failures.log").exists()

        # Now "fix" the network by rebuilding the fake boto3 WITHOUT the
        # failure injection, and retry --prediction successfully.
        import shutil

        shutil.rmtree(fake_dir)
        fake_dir.mkdir()
        (fake_dir / "boto3.py").write_text(
            FAKE_BOTO3_TEMPLATE.replace("__PUT_OBJECT_FAIL__", "''")
            .replace("__UPLOAD_FILE_FAIL__", "''")
            .replace("__CALLS_LOG__", repr(str(calls_log)))
        )

        r2 = _run_script(
            script,
            args=["--prediction", "pred"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert r2.returncode == 0, r2.stderr

        # Sentinel must be cleared AND the completion marker must exist.
        assert not (workspace / "upload_failures.log").exists(), (
            "Sentinel must be cleared after a successful --prediction retry"
        )
        assert (workspace / "prediction_completed" / "pred").exists(), (
            "Completion marker must be written after a successful --prediction retry"
        )

        # Now --done succeeds because the completion marker is in place.
        r3 = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert r2.returncode == 0, r2.stderr

        # Now --done succeeds because the completion marker is in place.
        r3 = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert r3.returncode == 0, r3.stderr
        calls = _read_calls(calls_log)
        shard_done = [
            kw for method, kw in calls if method == "put_object" and kw["Key"].endswith("/DONE")
        ]
        assert len(shard_done) >= 1


class TestShardSequentialHardening:
    """The fail-closed behaviour must hold across every documented failure path.

    The shard DONE marker is authoritative for completion. These
    tests exercise every failure path the ``--done`` mode is meant
    to guard against, including sentinel-write failure, marker
    publication failure, and the no-arg final-upload path.
    """

    def test_sentinel_write_failure_blocks_done(
        self,
        tmp_path: Path,
    ) -> None:
        """If recording the failure sentinel itself fails, --done must still refuse DONE.

        A disk-full or permission-denied failure on the local sentinel
        must not let the orchestrator see a successful shard DONE.
        We simulate the failure by patching the generated script
        so the first sentinel ``open()`` raises. The function's own
        ``except`` branch then attempts the ``SENTINEL_WRITE_FAILED``
        fallback write; if that also fails, ``--done`` must still
        refuse the shard DONE marker (the unreadable-file state
        itself counts as evidence of unresolved failures).
        """
        sink = R2Sink(bucket="bkt", prefix="project")
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(
            tmp_path,
            upload_file_fail="project/batch1/shard_0/outputs/pred/pred.txt",
            calls_log=calls_log,
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "outputs" / "pred").mkdir()
        (workspace / "outputs" / "pred" / "pred.txt").write_text("hi")
        (workspace / "worker.exitcode").write_text("0")

        # Replace the body of _record_upload_failure so BOTH the
        # primary write and the fallback write raise. Either way, the
        # sentinel file does not exist on disk after the call. The
        # function is still expected to return cleanly without
        # raising so the upload_prediction dispatcher can exit
        # cleanly.
        patched_script = sink.generate_upload_script("batch1", 0).replace(
            "def _record_upload_failure(name: str, failed_rel_paths: list[str]) -> None:",
            (
                "def _record_upload_failure(name: str, failed_rel_paths: list[str]) -> None:\n"
                "    raise RuntimeError('disk full')\n"
            ),
            1,
        )
        script = _patch_script_to_dump_calls(patched_script, calls_log)
        patched_script = patched_script.replace(
            "        try:\n            _record_upload_failure(name, failures)",
            (
                "        try:\n"
                "            _record_upload_failure(name, failures)\n"
                "        except Exception as exc:\n"
                "            print(f'INJECTED: {{exc}}', file=sys.stderr)"
            ),
            1,
        )
        # Replace the second occurrence (marker failures path).
        patched_script = patched_script.replace(
            "        try:\n            _record_upload_failure(name, marker_failures)",
            (
                "        try:\n"
                "            _record_upload_failure(name, marker_failures)\n"
                "        except Exception as exc:\n"
                "            print(f'INJECTED: {{exc}}', file=sys.stderr)"
            ),
            1,
        )
        # Replace the third occurrence (completion marker write failure).
        patched_script = patched_script.replace(
            '        try:\n            _record_upload_failure(name, ["<local completion marker>"])',
            (
                "        try:\n"
                '            _record_upload_failure(name, ["<local completion marker>"])\n'
                "        except Exception as exc:\n"
                "            print(f'INJECTED: {{exc}}', file=sys.stderr)"
            ),
            1,
        )
        script = _patch_script_to_dump_calls(patched_script, calls_log)
        patched_script = patched_script.replace(
            "        try:\n            _record_upload_failure(name, failures)",
            (
                "        try:\n"
                "            _record_upload_failure(name, failures)\n"
                "        except Exception as exc:\n"
                "            print(f'INJECTED: {{exc}}', file=sys.stderr)"
            ),
            1,
        )
        # Replace the second occurrence (marker failures path).
        patched_script = patched_script.replace(
            "        try:\n            _record_upload_failure(name, marker_failures)",
            (
                "        try:\n"
                "            _record_upload_failure(name, marker_failures)\n"
                "        except Exception as exc:\n"
                "            print(f'INJECTED: {{exc}}', file=sys.stderr)"
            ),
            1,
        )
        # Replace the third occurrence (completion marker write failure).
        patched_script = patched_script.replace(
            '        try:\n            _record_upload_failure(name, ["<local completion marker>"])',
            (
                "        try:\n"
                '            _record_upload_failure(name, ["<local completion marker>"])\n'
                "        except Exception as exc:\n"
                "            print(f'INJECTED: {{exc}}', file=sys.stderr)"
            ),
            1,
        )
        script = _patch_script_to_dump_calls(patched_script, calls_log)

        r1 = _run_script(
            script,
            args=["--prediction", "pred"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert r1.returncode != 0
        # Either the sentinel file does not exist (every write failed)
        # OR it exists with SENTINEL_WRITE_FAILED. Both states must
        # cause --done to refuse DONE.
        sentinel = workspace / "upload_failures.log"
        if sentinel.exists():
            content = sentinel.read_text()
            assert "SENTINEL_WRITE_FAILED" in content, content

        # --done must refuse the shard DONE marker.
        r2 = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert r2.returncode != 0, r2.stderr
        calls = _read_calls(calls_log)
        shard_done = [
            kw
            for method, kw in calls
            if method == "put_object"
            and kw["Key"].endswith("/DONE")
            and "/markers/" not in kw["Key"]
            and "/global_markers/" not in kw["Key"]
        ]
        assert shard_done == [], (
            f"Shard DONE must NOT be published when sentinel write failed; got {shard_done}"
        )

    def test_marker_publication_failure_blocks_done(
        self,
        tmp_path: Path,
    ) -> None:
        """A failed per-prediction marker publication MUST block --done."""
        sink = R2Sink(bucket="bkt", prefix="project")
        calls_log = tmp_path / "calls.json"
        # Fail the marker put_object (not the file upload).
        fake_dir = _make_fake_boto3(
            tmp_path,
            put_object_fail="project/batch1/shard_0/markers/pred.done",
            calls_log=calls_log,
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "outputs" / "pred").mkdir()
        (workspace / "outputs" / "pred" / "pred.txt").write_text("hi")
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )
        r1 = _run_script(
            script,
            args=["--prediction", "pred"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert r1.returncode != 0
        assert (workspace / "upload_failures.log").exists()

        r2 = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert r2.returncode != 0
        calls = _read_calls(calls_log)
        shard_done = [
            kw
            for method, kw in calls
            if method == "put_object"
            and kw["Key"].endswith("/DONE")
            and "/markers/" not in kw["Key"]
            and "/global_markers/" not in kw["Key"]
        ]
        assert shard_done == [], (
            f"Shard DONE must NOT be published when a marker failed; got {shard_done}"
        )

    def test_failed_no_arg_upload_blocks_done(
        self,
        tmp_path: Path,
    ) -> None:
        """A failed ``upload_all()`` (no-arg) MUST persist a sentinel and block --done."""
        sink = R2Sink(bucket="bkt", prefix="project")
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(
            tmp_path,
            upload_file_fail="project/batch1/shard_0/outputs/pred.txt",
            calls_log=calls_log,
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "outputs" / "pred.txt").write_text("hi")
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )
        # No-arg invocation uploads every file in outputs/ and tries
        # to publish DONE. The file upload must fail and record a
        # sentinel line for the "<no-arg>" pseudo-prediction.
        r1 = _run_script(
            script,
            args=[],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert r1.returncode != 0
        assert (workspace / "upload_failures.log").exists()
        content = (workspace / "upload_failures.log").read_text()
        assert "<no-arg>" in content

        # A subsequent --done MUST refuse the shard DONE marker.
        r2 = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert r2.returncode != 0

    def test_empty_outputs_dir_with_no_predictions_allows_done(
        self,
        tmp_path: Path,
    ) -> None:
        """When ``outputs/`` is empty (no predictions were made), --done succeeds.

        The completion-marker requirement must not penalise a shard
        that produced no predictions at all.
        """
        sink = R2Sink(bucket="bkt", prefix="project")
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(tmp_path, calls_log=calls_log)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )
        result = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert result.returncode == 0, result.stderr


class TestShardWholeShardCompletion:
    """``upload_all()`` must write per-prediction AND shard completion markers.

    A shard that uploads all its outputs via the no-arg mode must
    still leave positive completion evidence for ``--done`` to find.
    """

    def test_successful_no_arg_with_prediction_dirs_writes_markers(
        self,
        tmp_path: Path,
    ) -> None:
        """A no-arg upload of outputs/prediction_a/result.cif must succeed.

        The script writes per-prediction completion markers for every
        subdirectory under outputs/ AND the whole-shard positive
        completion marker. --done can then publish the shard DONE
        marker.
        """
        sink = R2Sink(bucket="bkt", prefix="project")
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(tmp_path, calls_log=calls_log)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "outputs" / "prediction_a").mkdir()
        (workspace / "outputs" / "prediction_a" / "result.cif").write_text("hi")
        (workspace / "outputs" / "prediction_b").mkdir()
        (workspace / "outputs" / "prediction_b" / "result.cif").write_text("hi")
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )
        result = _run_script(
            script,
            args=[],  # no-arg final upload
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert result.returncode == 0, result.stderr
        # Per-prediction completion markers exist.
        assert (workspace / "prediction_completed" / "prediction_a").exists()
        assert (workspace / "prediction_completed" / "prediction_b").exists()
        # Whole-shard positive completion marker exists.
        assert (workspace / "shard_completed").exists()

        # --done must now succeed.
        result2 = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert result2.returncode == 0, result2.stderr
        calls = _read_calls(calls_log)
        shard_done = [
            kw for method, kw in calls if method == "put_object" and kw["Key"].endswith("/DONE")
        ]
        assert len(shard_done) >= 1

    def test_failed_no_arg_then_successful_retry_clears_sentinel(
        self,
        tmp_path: Path,
    ) -> None:
        """A failed no-arg upload must be cleared by a successful retry.

        The first invocation records the failure under ``<no-arg>``;
        the second invocation must clear that entry AND write the
        positive completion markers so ``--done`` can publish.
        """
        sink = R2Sink(bucket="bkt", prefix="project")
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(
            tmp_path,
            upload_file_fail="project/batch1/shard_0/outputs/pred.txt",
            calls_log=calls_log,
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "outputs" / "pred.txt").write_text("hi")
        (workspace / "worker.exitcode").write_text("0")

        script = _patch_script_to_dump_calls(
            sink.generate_upload_script("batch1", 0),
            calls_log,
        )
        # First no-arg fails and writes <no-arg> failure sentinel.
        r1 = _run_script(
            script,
            args=[],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert r1.returncode != 0
        assert (workspace / "upload_failures.log").exists()
        assert "<no-arg>" in (workspace / "upload_failures.log").read_text()

        # Rebuild fake boto3 without the failure injection.
        import shutil

        shutil.rmtree(fake_dir)
        fake_dir.mkdir()
        (fake_dir / "boto3.py").write_text(
            FAKE_BOTO3_TEMPLATE.replace("__PUT_OBJECT_FAIL__", "''")
            .replace("__UPLOAD_FILE_FAIL__", "''")
            .replace("__CALLS_LOG__", repr(str(calls_log)))
        )

        # Retry no-arg: must succeed, clear the failure sentinel, and
        # write the shard completion marker.
        r2 = _run_script(
            script,
            args=[],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert r2.returncode == 0, r2.stderr
        # The <no-arg> failure sentinel must be cleared.
        assert not (workspace / "upload_failures.log").exists()
        # The shard completion marker must be present.
        assert (workspace / "shard_completed").exists()

        # --done must now succeed.
        r3 = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert r3.returncode == 0, r3.stderr
        calls = _read_calls(calls_log)
        shard_done = [
            kw for method, kw in calls if method == "put_object" and kw["Key"].endswith("/DONE")
        ]
        # The third invocation's --done adds at least one more DONE.
        assert len(shard_done) >= 1

    def test_sentinel_write_failure_blocks_done_with_flat_files(
        self,
        tmp_path: Path,
    ) -> None:
        """When the sentinel write itself fails, --done must STILL refuse DONE.

        With only flat files in outputs/ (no prediction directories),
        per-prediction completion markers don't apply. The whole-shard
        completion marker is the only positive proof; the sentinel
        write failure must still block DONE.
        """
        sink = R2Sink(bucket="bkt", prefix="project")
        calls_log = tmp_path / "calls.json"
        fake_dir = _make_fake_boto3(
            tmp_path,
            upload_file_fail="project/batch1/shard_0/outputs/pred.txt",
            calls_log=calls_log,
        )
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "outputs").mkdir()
        (workspace / "outputs" / "pred.txt").write_text("hi")
        (workspace / "worker.exitcode").write_text("0")

        # Patch the generated script so EVERY sentinel write fails.
        # On failure, the script must still refuse DONE.
        patched = sink.generate_upload_script("batch1", 0).replace(
            "def _record_upload_failure(name: str, failed_rel_paths: list[str]) -> None:",
            (
                "def _record_upload_failure(name: str, failed_rel_paths: list[str]) -> None:\n"
                "    raise RuntimeError('disk full')\n"
            ),
            1,
        )
        # Wrap the call sites so the script exits cleanly.
        for old, replacement in [
            (
                "        try:\n            _record_upload_failure(name, failures)",
                (
                    "        try:\n"
                    "            _record_upload_failure(name, failures)\n"
                    "        except Exception as exc:\n"
                    "            print(f'INJECTED: {{exc}}', file=sys.stderr)"
                ),
            ),
            (
                "        try:\n            _record_upload_failure(name, marker_failures)",
                (
                    "        try:\n"
                    "            _record_upload_failure(name, marker_failures)\n"
                    "        except Exception as exc:\n"
                    "            print(f'INJECTED: {{exc}}', file=sys.stderr)"
                ),
            ),
            (
                "        try:\n"
                '            _record_upload_failure(name, [\\"<local completion marker>\\"])',
                (
                    "        try:\n"
                    '            _record_upload_failure(name, [\\"<local completion marker>\\"])\n'
                    "        except Exception as exc:\n"
                    "            print(f'INJECTED: {{exc}}', file=sys.stderr)"
                ),
            ),
        ]:
            patched = patched.replace(old, replacement, 1)

        script = _patch_script_to_dump_calls(patched, calls_log)

        r1 = _run_script(
            script,
            args=[],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert r1.returncode != 0
        # The shard completion marker must NOT exist (file uploads
        # failed before the success path).
        assert not (workspace / "shard_completed").exists()

        # --done must refuse the shard DONE marker.
        r2 = _run_script(
            script,
            args=["--done"],
            workspace=workspace,
            fake_boto3_dir=fake_dir,
        )
        assert r2.returncode != 0, r2.stderr
        calls = _read_calls(calls_log)
        shard_done = [
            kw
            for method, kw in calls
            if method == "put_object"
            and kw["Key"].endswith("/DONE")
            and "/markers/" not in kw["Key"]
            and "/global_markers/" not in kw["Key"]
        ]
        assert shard_done == [], f"Shard DONE must NOT be published; got {shard_done}"
