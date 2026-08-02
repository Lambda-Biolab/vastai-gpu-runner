"""Cloudflare R2 result sink — workers push outputs, orchestrator polls.

R2 is S3-compatible with zero egress fees. Workers upload prediction
outputs after completion so results survive instance self-destruction.
The orchestrator polls for DONE markers instead of SSH.

Credentials are read from a shell-export file (default
``~/.cloud-credentials``) or environment variables.

Usage::

    from vastai_gpu_runner.storage.r2 import R2Sink

    sink = R2Sink(bucket="my-bucket", prefix="project/batches")
    sink.is_shard_done(batch_id, shard_id)
    sink.download_shard(batch_id, shard_id, local_dir)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def load_r2_env(credentials_file: str | Path = "~/.cloud-credentials") -> dict[str, str]:
    """Parse R2 credentials from a shell-export credentials file.

    Args:
        credentials_file: Path to the credentials file. Lines should be
            ``export R2_KEY="value"`` format.

    Returns:
        Dict of R2-prefixed environment variables.
    """
    creds_path = Path(credentials_file).expanduser()
    env: dict[str, str] = {}
    if creds_path.exists():
        for line in creds_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("export R2_"):
                parts = line.replace("export ", "").split("=", 1)
                if len(parts) == 2:
                    env[parts[0]] = parts[1].strip('"').strip("'")
    return env


def get_r2_client(
    credentials_file: str | Path = "~/.cloud-credentials",
):  # type: ignore[no-untyped-def]
    """Create a boto3 S3 client configured for Cloudflare R2.

    Args:
        credentials_file: Path to the credentials file.

    Returns:
        boto3 S3 client.
    """
    import boto3

    env = load_r2_env(credentials_file)
    return boto3.client(
        "s3",
        endpoint_url=env.get("R2_ENDPOINT", os.environ.get("R2_ENDPOINT", "")),
        aws_access_key_id=env.get("R2_ACCESS_KEY_ID", os.environ.get("R2_ACCESS_KEY_ID", "")),
        aws_secret_access_key=env.get(
            "R2_SECRET_ACCESS_KEY", os.environ.get("R2_SECRET_ACCESS_KEY", "")
        ),
        region_name="auto",
    )


class R2Sink:
    """Cloudflare R2 result sink for cloud batch workloads.

    Args:
        bucket: R2 bucket name.
        prefix: Key prefix for all objects (e.g. ``"project/batches"``).
        credentials_file: Path to shell-export credentials file.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        credentials_file: str | Path = "~/.cloud-credentials",
    ) -> None:
        """Initialize R2 sink with bucket, prefix, and credentials."""
        self.bucket = bucket
        self.prefix = prefix
        self._client = get_r2_client(credentials_file)

    # -- Shard operations --------------------------------------------------

    def shard_prefix(self, batch_id: str, shard_id: int) -> str:
        """Return the S3 key prefix for a shard."""
        return f"{self.prefix}/{batch_id}/shard_{shard_id}/"

    def is_shard_done(self, batch_id: str, shard_id: int) -> bool:
        """Check if a shard has uploaded its DONE marker to R2."""
        key = f"{self.shard_prefix(batch_id, shard_id)}DONE"
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self._client.exceptions.ClientError:
            return False

    def count_completed_predictions(self, batch_id: str, shard_id: int) -> int:
        """Count per-prediction .done markers in R2 for progress tracking."""
        prefix = f"{self.shard_prefix(batch_id, shard_id)}markers/"
        count = 0
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            count += sum(1 for obj in page.get("Contents", []) if obj["Key"].endswith(".done"))
        return count

    def download_shard(self, batch_id: str, shard_id: int, local_dir: Path) -> list[str]:
        """Download all outputs for a shard from R2.

        Args:
            batch_id: Batch identifier.
            shard_id: Shard number.
            local_dir: Local directory to download into.

        Returns:
            List of downloaded file paths (relative to local_dir).
        """
        local_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.shard_prefix(batch_id, shard_id)}outputs/"
        keys = self._list_object_keys(prefix)
        return self._download_keys(keys, prefix, local_dir, "shard", batch_id, shard_id)

    def _list_object_keys(self, prefix: str) -> list[tuple[str, str]]:
        """List (key, rel_path) tuples under a prefix.

        Extracted to keep ``download_shard`` and ``download_job``
        under the org complexity threshold (10). The nested loop
        over paginator pages + Contents is the source of the
        extra branches.
        """
        keys: list[tuple[str, str]] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel_path = key[len(prefix) :]
                if rel_path:
                    keys.append((key, rel_path))
        return keys

    def _download_keys(
        self,
        keys: list[tuple[str, str]],
        prefix: str,
        local_dir: Path,
        kind: str,
        batch_id: str,
        unit_id: object,
    ) -> list[str]:
        """Download a list of (key, rel_path) tuples concurrently.

        ``kind`` is ``"shard"`` or ``"job"`` and ``unit_id`` is the
        shard number or job name; used for the log line so the
        context is clear regardless of which downloader is calling.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        for _, rel_path in keys:
            (local_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)

        def _download_one(item: tuple[str, str]) -> str:
            key, rel_path = item
            self._client.download_file(self.bucket, key, str(local_dir / rel_path))
            return rel_path

        downloaded: list[str] = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_download_one, item): item for item in keys}
            for future in as_completed(futures):
                try:
                    downloaded.append(future.result())
                except Exception:
                    _, rel_path = futures[future]
                    logger.warning("R2 download failed: %s", rel_path)

        logger.info(
            "Downloaded %d files from R2 for %s %s (batch %s)",
            len(downloaded),
            kind,
            unit_id,
            batch_id,
        )
        return downloaded

    # -- Global markers (cross-shard coordination) -------------------------

    def global_marker_prefix(self, batch_id: str) -> str:
        """Return the S3 key prefix for batch-wide global markers."""
        return f"{self.prefix}/{batch_id}/global_markers/"

    def prediction_exists(self, batch_id: str, prediction_name: str) -> bool:
        """Check if a prediction's global marker exists (O(1) HEAD request).

        Used by both local and cloud workers to avoid duplicate work.
        Returns False on any R2 error (fail-open).
        """
        key = f"{self.global_marker_prefix(batch_id)}{prediction_name}.done"
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self._client.exceptions.ClientError:
            return False

    def mark_prediction_done(
        self, batch_id: str, prediction_name: str, worker_id: str = "local"
    ) -> None:
        """Write a global marker for a completed prediction."""
        key = f"{self.global_marker_prefix(batch_id)}{prediction_name}.done"
        self._client.put_object(Bucket=self.bucket, Key=key, Body=worker_id.encode())

    def count_global_completed(self, batch_id: str) -> int:
        """Count total completed predictions across all workers."""
        prefix = self.global_marker_prefix(batch_id)
        count = 0
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            count += sum(1 for obj in page.get("Contents", []) if obj["Key"].endswith(".done"))
        return count

    # -- Job operations (1 job = 1 instance, e.g. MD) ----------------------

    def job_prefix(self, batch_id: str, job_name: str) -> str:
        """Return the S3 key prefix for a job-based batch."""
        return f"{self.prefix}/{batch_id}/{job_name}/"

    def is_job_done(self, batch_id: str, job_name: str) -> bool:
        """Check if a job has uploaded its DONE marker."""
        key = f"{self.job_prefix(batch_id, job_name)}DONE"
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self._client.exceptions.ClientError:
            return False

    def download_job(self, batch_id: str, job_name: str, local_dir: Path) -> list[str]:
        """Download all outputs for a job from R2.

        Args:
            batch_id: Batch identifier.
            job_name: Job name.
            local_dir: Local directory to download into.

        Returns:
            List of downloaded relative file paths.
        """
        local_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.job_prefix(batch_id, job_name)
        keys = self._list_object_keys(prefix)
        return self._download_keys(keys, prefix, local_dir, "job", batch_id, job_name)

    # -- Batch management --------------------------------------------------

    def list_batch_shards(self, batch_id: str) -> list[int]:
        """List shard IDs that have DONE markers in R2."""
        prefix = f"{self.prefix}/{batch_id}/"
        done_shards: list[int] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                shard_dir = cp["Prefix"].rstrip("/").split("/")[-1]
                if shard_dir.startswith("shard_"):
                    sid = int(shard_dir.split("_")[1])
                    if self.is_shard_done(batch_id, sid):
                        done_shards.append(sid)
        return sorted(done_shards)

    def cleanup_batch(self, batch_id: str) -> int:
        """Delete all R2 objects for a batch. Returns count of deleted objects."""
        prefix = f"{self.prefix}/{batch_id}/"
        deleted = 0
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if objects:
                self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": objects})
                deleted += len(objects)
        logger.info("Cleaned up %d R2 objects for batch %s", deleted, batch_id)
        return deleted

    # -- Large file chunked upload -----------------------------------------

    def upload_file_chunk(
        self,
        local_file: Path,
        chunk_index: int,
        offset: int,
        size: int,
        r2_prefix: str,
        *,
        filename_stem: str = "data",
        filename_ext: str = "",
    ) -> bool:
        """Upload a byte-range chunk of a large file to R2.

        Reads bytes ``[offset:offset+size]`` from *local_file* and uploads
        as ``{r2_prefix}/{filename_stem}_chunk_{NNN}{filename_ext}``.
        Concatenating chunks in index order reconstructs the original file.

        Args:
            local_file: Path to the full file on disk.
            chunk_index: Zero-based chunk sequence number.
            offset: Byte offset to start reading from.
            size: Number of bytes in this chunk.
            r2_prefix: R2 key prefix for the chunks.
            filename_stem: Base name for chunk files (default ``"data"``).
            filename_ext: File extension including dot (default ``""``).

        Returns:
            True on success, False on failure.
        """
        key = f"{r2_prefix}{filename_stem}_chunk_{chunk_index:03d}{filename_ext}"
        try:
            with open(local_file, "rb") as fh:
                fh.seek(offset)
                data = fh.read(size)
            self._client.put_object(Bucket=self.bucket, Key=key, Body=data)
            logger.debug(
                "Uploaded chunk %d (%d bytes, offset %d)",
                chunk_index,
                size,
                offset,
            )
            return True
        except Exception:
            logger.warning(
                "Chunk upload failed: chunk %d, offset %d, size %d",
                chunk_index,
                offset,
                size,
                exc_info=True,
            )
            return False

    @staticmethod
    def consolidate_chunks(
        local_dir: Path,
        *,
        filename_stem: str = "data",
        filename_ext: str = "",
        output_name: str | None = None,
    ) -> Path | None:
        """Concatenate chunked files into a single file.

        Chunks are sorted by index (``{stem}_chunk_000{ext}``, ...)
        and concatenated byte-for-byte. After concatenation, chunk
        files are deleted.

        Args:
            local_dir: Directory containing downloaded chunk files.
            filename_stem: Base name used when uploading chunks.
            filename_ext: File extension used when uploading chunks.
            output_name: Output filename. Defaults to ``{stem}{ext}``.

        Returns:
            Path to the consolidated file, or None if no chunks found.
        """
        chunks = sorted(local_dir.glob(f"{filename_stem}_chunk_*{filename_ext}"))
        if not chunks:
            return None

        if output_name is None:
            output_name = f"{filename_stem}{filename_ext}"
        traj_path = local_dir / output_name
        total_bytes = 0
        with open(traj_path, "wb") as out:
            for chunk in chunks:
                data = chunk.read_bytes()
                out.write(data)
                total_bytes += len(data)

        logger.info(
            "Consolidated %d chunks -> %s (%d bytes)",
            len(chunks),
            traj_path,
            total_bytes,
        )

        for chunk in chunks:
            chunk.unlink()

        return traj_path

    # -- Upload script generation ------------------------------------------

    def generate_upload_script(
        self,
        batch_id: str,
        shard_id: int,
        workspace: str = "/workspace",
    ) -> str:
        """Generate a Python upload script for shard-based workers.

        The script supports modes: ``--prediction NAME``, ``--done``,
        ``--check``, or no args (upload all).

        Args:
            batch_id: Batch identifier.
            shard_id: Shard index.
            workspace: Worker workspace path.

        Returns:
            Python script as a string.
        """
        return f'''#!/usr/bin/env python3
"""Upload outputs to Cloudflare R2 (auto-generated by vastai-gpu-runner)."""
import argparse
import os
import sys

import boto3

BUCKET = "{self.bucket}"
PREFIX = "{self.prefix}/{batch_id}/shard_{shard_id}/"
BATCH_PREFIX = "{self.prefix}/{batch_id}/"
WORKSPACE = "{workspace}"

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("R2_ENDPOINT", ""),
    aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID", ""),
    aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY", ""),
    region_name="auto",
)


def upload_prediction(name: str) -> tuple[str, int]:
    """Upload one prediction's outputs. Fail closed on partial upload.

    Returns:
        A tuple ``(status, count)``:
            - ``("absent", 0)`` if the prediction output directory is
              missing (warning, not a failure; script exits 0).
            - ``("failed", <partial_count>)`` if any required upload
              (file or marker) failed (failure; script exits non-zero;
              no DONE markers; a local failure sentinel is written so
              subsequent ``--done`` / no-arg invocations also refuse
              DONE).
            - ``("ok", <uploaded_count>)`` on full success. A positive
              completion marker is written atomically so ``--done``
              can require positive proof of completion.

        Using a status string instead of a boolean avoids the
        "0 successful uploads = absent directory" collision when a
        partial failure uploads zero files before any succeed.
    """
    pred_dir = os.path.join(WORKSPACE, "outputs", name)
    if not os.path.isdir(pred_dir):
        print(f"WARN: no output dir for {{name}}", file=sys.stderr)
        return "absent", 0
    failures: list[str] = []
    uploaded = 0
    for root, _dirs, files in os.walk(pred_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, os.path.join(WORKSPACE, "outputs"))
            key = PREFIX + "outputs/" + rel
            try:
                s3.upload_file(local_path, BUCKET, key)
                uploaded += 1
            except Exception as exc:
                failures.append(rel)
                print(f"WARN: upload failed for {{rel}}: {{exc}}", file=sys.stderr)
    if failures:
        # Required prediction file uploads did not complete. Omit both
        # the per-shard and global DONE markers AND persist a local
        # failure sentinel so any subsequent --done / no-arg invocation
        # refuses to publish the top-level shard DONE marker. Without
        # this sentinel, an upload that fails for transient reasons
        # and then succeeds later (via a retry of the worker script)
        # would still let the orchestrator accept an incomplete result
        # set as committed.
        print(
            f"FAIL: {{len(failures)}} prediction file(s) failed; "
            "omitting DONE marker for {{name}}.",
            file=sys.stderr,
        )
        try:
            _record_upload_failure(name, failures)
        except Exception as exc:
            # The recorder itself raised; we still must not let --done
            # publish a false shard DONE. Record a separate global
            # sentinel-write-failure entry as a last resort.
            print(f"FAIL: _record_upload_failure raised: {{exc}}", file=sys.stderr)
        _clear_prediction_success(name)
        return "failed", uploaded
    # File uploads succeeded; now publish the markers. Marker
    # publication failures are also recorded so a later --done
    # cannot mask them.
    marker_failures: list[str] = []
    try:
        s3.put_object(Bucket=BUCKET, Key=PREFIX + f"markers/{{name}}.done", Body=b"")
    except Exception as exc:
        marker_failures.append(f"markers/{{name}}.done")
        print(f"FAIL: marker put failed: {{exc}}", file=sys.stderr)
    try:
        s3.put_object(
            Bucket=BUCKET,
            Key=BATCH_PREFIX + f"global_markers/{{name}}.done",
            Body=f"shard_{shard_id}".encode(),
        )
    except Exception as exc:
        marker_failures.append(f"global_markers/{{name}}.done")
        print(f"FAIL: global marker put failed: {{exc}}", file=sys.stderr)
    if marker_failures:
        # Marker publication failed; record and refuse DONE.
        try:
            _record_upload_failure(name, marker_failures)
        except Exception as exc:
            print(f"FAIL: _record_upload_failure raised: {{exc}}", file=sys.stderr)
        _clear_prediction_success(name)
        return "failed", uploaded
    # All uploads succeeded. Atomically write the positive completion
    # marker on the local filesystem. The --done mode requires this
    # marker for every prediction directory under outputs/, so a
    # missing marker (e.g. from a sentinel write failure) refuses
    # DONE rather than masking the failure.
    if not _record_prediction_success(name):
        try:
            _record_upload_failure(name, ["<local completion marker>"])
        except Exception as exc:
            print(f"FAIL: _record_upload_failure raised: {{exc}}", file=sys.stderr)
        return "failed", uploaded
    _clear_upload_failure(name)
    return "ok", uploaded


def _upload_exitcode() -> bool:
    """Upload worker.exitcode. Returns True only on success.

    The orchestrator reads this file to confirm the workload outcome.
    Failing to upload it — OR the file being absent — MUST prevent
    final ``DONE`` publication. Absence is treated as a failure
    because we cannot verify the workload outcome without it.
    """
    exitcode_path = os.path.join(WORKSPACE, "worker.exitcode")
    if not os.path.exists(exitcode_path):
        print(
            "FAIL: worker.exitcode missing — refusing to publish DONE marker",
            file=sys.stderr,
        )
        return False
    try:
        s3.upload_file(exitcode_path, BUCKET, PREFIX + "worker.exitcode")
        return True
    except Exception as exc:
        print(f"FAIL: worker.exitcode upload failed: {{exc}}", file=sys.stderr)
        return False


def _upload_failures_log_path() -> str:
    """Return the path of the local upload-failure sentinel file."""
    return os.path.join(WORKSPACE, "upload_failures.log")


def _prediction_completed_dir() -> str:
    """Return the directory holding per-prediction completion markers."""
    return os.path.join(WORKSPACE, "prediction_completed")


def _shard_complete_marker_path() -> str:
    """Return the path of the whole-shard positive completion marker.

    This marker is written by ``upload_all()`` (no-arg mode) after
    every required upload succeeds. ``--done`` requires either this
    marker OR per-prediction completion markers for every prediction
    directory under ``outputs/``. The marker is the canonical
    positive proof that the worker successfully published every
    required artifact; absence blocks DONE regardless of whether
    failure-sentinel evidence exists.
    """
    return os.path.join(WORKSPACE, "shard_completed")


def _record_upload_failure(name: str, failed_rel_paths: list[str]) -> None:
    """Append a per-prediction failure line to the sentinel file.

    The sentinel persists on the local filesystem so any later
    ``--done`` / no-arg invocation can refuse to publish the
    top-level shard DONE marker if any per-prediction upload failed.
    """
    log_path = _upload_failures_log_path()
    sentinel_write_failed = False
    try:
        with open(log_path, "a") as fh:
            for rel in failed_rel_paths:
                fh.write(f"{{name}}\\t{{rel}}\\n")
    except Exception as exc:
        sentinel_write_failed = True
        print(f"FAIL: failed to write upload failure sentinel: {{exc}}", file=sys.stderr)

    if sentinel_write_failed:
        # Best-effort fallback so --done still sees evidence of the
        # failure. We write a separate ``SENTINEL_WRITE_FAILED``
        # line; even if THIS write also fails, ``_has_unresolved_upload_failures``
        # treats the unreadable-file state as unresolved.
        try:
            with open(log_path, "a") as fh2:
                fh2.write(f"SENTINEL_WRITE_FAILED\\t{{name}}\\n")
        except Exception:
            pass


def _record_shard_complete() -> bool:
    """Atomically write the whole-shard positive completion marker.

    Returns:
        True on success, False if the marker could not be written.
        A False return signals to callers that ``--done`` must refuse
        to publish the shard DONE marker.
    """
    final_path = _shard_complete_marker_path()
    tmp_path = f"{{final_path}}.tmp"
    try:
        with open(tmp_path, "w") as fh:
            fh.write("ok\\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, final_path)
        return True
    except Exception as exc:
        print(f"FAIL: failed to record shard completion marker: {{exc}}", file=sys.stderr)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return False


def _has_shard_complete_marker() -> bool:
    """Return True iff the whole-shard completion marker exists."""
    return os.path.exists(_shard_complete_marker_path())


def _clear_shard_complete_marker() -> None:
    """Remove the whole-shard completion marker (on failure)."""
    try:
        os.unlink(_shard_complete_marker_path())
    except OSError:
        pass


def _record_prediction_success(name: str) -> bool:
    """Atomically write a positive completion marker for *name*.

    Returns:
        True on success, False if the marker could not be written.
        A False return signals to callers that the ``--done`` path
        must refuse to publish the shard DONE marker.
    """
    marker_dir = _prediction_completed_dir()
    tmp_path = f"{{marker_dir}}/.{{name}}.tmp"
    final_path = f"{{marker_dir}}/{{name}}"
    try:
        os.makedirs(marker_dir, exist_ok=True)
        # Atomic rename so concurrent writers cannot observe a
        # half-written marker.
        with open(tmp_path, "w") as fh:
            fh.write("ok\\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, final_path)
        return True
    except Exception as exc:
        print(f"FAIL: failed to record prediction completion marker: {{exc}}", file=sys.stderr)
        # Best-effort cleanup.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return False


def _clear_upload_failure(name: str) -> None:
    """Remove any prior sentinel line for this prediction on success.

    If the resulting file is empty, the file itself is removed so
    that ``--done`` can detect the absence-of-evidence state cleanly.
    """
    log_path = _upload_failures_log_path()
    if not os.path.exists(log_path):
        return
    try:
        kept: list[str] = []
        with open(log_path) as fh:
            for line in fh:
                parts = line.rstrip("\\n").split("\\t", 1)
                # Keep entries that are NOT for this prediction. The
                # ``SENTINEL_WRITE_FAILED`` marker is global and must
                # never be cleared by a per-prediction success.
                if (
                    len(parts) == 2
                    and parts[0] == name
                    and parts[1] != "SENTINEL_WRITE_FAILED"
                ):
                    continue
                kept.append(line)
        if kept:
            with open(log_path, "w") as fh:
                fh.writelines(kept)
        else:
            os.unlink(log_path)
    except Exception:
        # If we cannot rewrite the sentinel, leave it alone and refuse
        # DONE conservatively in ``_has_unresolved_upload_failures``.
        pass


def _clear_prediction_success(name: str) -> None:
    """Remove the positive completion marker for *name* (on failure)."""
    marker_path = f"{{_prediction_completed_dir()}}/{{name}}"
    try:
        os.unlink(marker_path)
    except OSError:
        pass


def _has_unresolved_upload_failures() -> bool:
    """Return True iff the shard's uploads are NOT provably complete.

    Three sources of evidence must all be absent for completion:

    1. A non-empty failure sentinel. If present, the worker
       observed a failed upload at some point and we refuse DONE
       until that record is cleared by a successful retry.
    2. The whole-shard positive completion marker (written by
       ``upload_all()``). If absent, no ``upload_all()`` ever
       succeeded; we must require positive proof.
    3. Per-prediction completion markers for every prediction
       directory under ``outputs/`` (only checked as a fallback
       when the shard_complete marker is absent).

    Empty ``outputs/``: no work to prove. Non-empty outputs/ with
    neither shard_complete nor per-prediction markers → block.
    """
    # (1) Failure sentinel.
    log_path = _upload_failures_log_path()
    if os.path.exists(log_path):
        try:
            with open(log_path) as fh:
                for line in fh:
                    if line.strip():
                        return True
        except Exception:
            # Treat unreadable sentinel as unresolved.
            return True
    # Empty outputs/ with no failures → no work to prove.
    outputs_dir = os.path.join(WORKSPACE, "outputs")
    outputs_empty = not os.path.isdir(outputs_dir) or not os.listdir(outputs_dir)
    if outputs_empty:
        return False
    # (2) Whole-shard completion marker is the canonical positive
    # proof. If present, the shard is provably complete regardless
    # of per-prediction markers.
    if _has_shard_complete_marker():
        return False
    # (3) No shard_complete but maybe per-prediction markers cover
    # every prediction directory.
    return _missing_prediction_completions() != []


def _missing_prediction_completions() -> list[str]:
    """Return names of prediction directories / flat files lacking completion.

    A prediction that was never invoked by ``--prediction`` has no
    completion marker; a flat file left behind by ``upload_all()``
    only counts as covered if the whole-shard completion marker
    exists. We list all top-level entries (directories AND flat
    files) under ``outputs/`` whose per-prediction marker is
    missing. The caller combines this with the shard_complete
    check; if either positive proof is present we proceed.
    """
    outputs_dir = os.path.join(WORKSPACE, "outputs")
    if not os.path.isdir(outputs_dir):
        return []
    marker_dir = _prediction_completed_dir()
    missing: list[str] = []
    for entry in sorted(os.listdir(outputs_dir)):
        full = os.path.join(outputs_dir, entry)
        # Only directories map 1:1 to per-prediction markers.
        if not os.path.isdir(full):
            # Flat file at the top level: the shard_complete marker
            # is the only positive proof that the worker uploaded
            # it. We add a synthetic "FLAT:<entry>" token so the
            # caller can require shard_complete to be present when
            # any flat files exist.
            if not _has_shard_complete_marker():
                missing.append(f"FLAT:{{entry}}")
            continue
        marker_path = f"{{marker_dir}}/{{entry}}"
        if not os.path.exists(marker_path):
            missing.append(entry)
    return missing


def upload_done_marker() -> int:
    """Upload DONE marker + worker.exitcode. Exit code = upload status.

    Refuses to publish the top-level shard DONE marker if any prior
    per-prediction upload left a local failure sentinel unresolved.
    The sentinel is checked first so a transient per-prediction
    failure cannot be masked by a later successful worker exit.

    Returns:
        0 if all required uploads succeeded; non-zero otherwise. The
        script exits with this return code so the worker treats a
        transport failure as a non-success.
    """
    if _has_unresolved_upload_failures():
        print(
            "FAIL: unresolved per-prediction upload failures in "
            "upload_failures.log — refusing to publish shard DONE.",
            file=sys.stderr,
        )
        return 4
    if not _upload_exitcode():
        # Omit DONE marker — partial publication would let the
        # orchestrator accept an incomplete result set as committed.
        return 1
    try:
        s3.put_object(Bucket=BUCKET, Key=PREFIX + "DONE", Body=b"")
    except Exception as exc:
        print(f"FAIL: DONE marker upload failed: {{exc}}", file=sys.stderr)
        return 2
    return 0


def upload_all() -> int:
    """Upload all outputs, then DONE marker. Returns 0 only on full success.

    Failures are recorded in the local upload-failure sentinel so a
    later ``--done`` invocation cannot mask them. On full success,
    per-prediction completion markers are written for every
    top-level subdirectory under ``outputs/`` AND the whole-shard
    positive completion marker is atomically recorded. Both are
    required for ``--done`` to publish the shard DONE marker.

    A successful invocation also clears any stale ``<no-arg>``
    failure entry so a retry after recovery can complete normally.
    """
    uploaded = 0
    failures: list[str] = []
    outputs_dir = os.path.join(WORKSPACE, "outputs")
    if os.path.isdir(outputs_dir):
        for root, _dirs, files in os.walk(outputs_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel = os.path.relpath(local_path, outputs_dir)
                key = PREFIX + "outputs/" + rel
                try:
                    s3.upload_file(local_path, BUCKET, key)
                    uploaded += 1
                except Exception as exc:
                    failures.append(rel)
                    print(f"WARN: upload failed for {{rel}}: {{exc}}")
    if failures:
        # Required final uploads did not complete; record and omit DONE.
        print(
            f"FAIL: {{len(failures)}} required file(s) failed to upload; "
            "omitting DONE marker.",
            file=sys.stderr,
        )
        _record_upload_failure("<no-arg>", failures)
        _clear_shard_complete_marker()
        return 3
    # All file uploads succeeded. Write per-prediction completion
    # markers for every top-level subdirectory of outputs/ so the
    # next --done has positive proof of every prediction.
    if os.path.isdir(outputs_dir):
        for entry in sorted(os.listdir(outputs_dir)):
            full = os.path.join(outputs_dir, entry)
            if os.path.isdir(full):
                _record_prediction_success(entry)
    # Write the whole-shard positive completion marker.
    shard_marker_written = _record_shard_complete()
    # Clear any stale failure entries from prior failed invocations.
    _clear_upload_failure("<no-arg>")
    if not shard_marker_written:
        print(
            "FAIL: failed to record shard completion marker — refusing DONE.",
            file=sys.stderr,
        )
        return 5
    if _has_unresolved_upload_failures():
        print(
            "FAIL: unresolved per-prediction upload failures in "
            "upload_failures.log — refusing to publish shard DONE.",
            file=sys.stderr,
        )
        return 4
    if not _upload_exitcode():
        _clear_shard_complete_marker()
        return 1
    try:
        s3.put_object(Bucket=BUCKET, Key=PREFIX + "DONE", Body=b"")
    except Exception as exc:
        print(f"FAIL: DONE marker upload failed: {{exc}}", file=sys.stderr)
        _clear_shard_complete_marker()
        return 2
    return 0


def check_prediction(name: str) -> bool:
    key = BATCH_PREFIX + f"global_markers/{{name}}.done"
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


parser = argparse.ArgumentParser()
parser.add_argument("--prediction", help="Upload one prediction by name")
parser.add_argument("--done", action="store_true", help="Upload DONE marker only")
parser.add_argument("--check", nargs="?", const="__connectivity__",
                    help="No arg: R2 connectivity test; with NAME: check done")
args = parser.parse_args()

if args.check == "__connectivity__":
    try:
        s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX, MaxKeys=1)
        print("R2: connectivity OK")
        sys.exit(0)
    except Exception as exc:
        print(f"R2: connectivity FAILED: {{exc}}", file=sys.stderr)
        sys.exit(1)
elif args.check:
    sys.exit(0 if check_prediction(args.check) else 1)
elif args.prediction:
    status, count = upload_prediction(args.prediction)
    if status == "absent":
        # No output dir — not a failure; warn and exit 0.
        print(f"R2: {{args.prediction}} no output dir")
        sys.exit(0)
    if status == "failed":
        # Partial failure — required uploads did not complete.
        print(
            f"R2: {{args.prediction}} failed (partial upload)", file=sys.stderr,
        )
        sys.exit(1)
    print(f"R2: {{args.prediction}} uploaded ({{count}} files)")
elif args.done:
    rc = upload_done_marker()
    if rc == 0:
        print("R2: DONE marker uploaded")
    else:
        print(f"R2: DONE marker upload failed (rc={{rc}})", file=sys.stderr)
    sys.exit(rc)
else:
    rc = upload_all()
    print(f"R2: all uploaded (rc={{rc}})")
    sys.exit(rc)
'''

    def generate_job_upload_script(
        self,
        batch_id: str,
        job_name: str,
        workspace: str = "/workspace",
        *,
        large_file: str = "",
        checkpoint_files: list[str] | None = None,
    ) -> str:
        """Generate an R2 upload script for job-based workers.

        Supports: ``--checkpoint``, ``--done``, or no args (upload all).

        If *large_file* is set (e.g. ``"trajectory.dcd"``), the script
        uses chunked delta uploads for that file to handle files that grow
        continuously during execution. Other checkpoint files are uploaded
        in full on each ``--checkpoint`` call.

        Args:
            batch_id: Batch identifier.
            job_name: Job name.
            workspace: Worker workspace path.
            large_file: Filename for chunked upload (empty = no chunking).
            checkpoint_files: Files to upload on ``--checkpoint``.
                Defaults to all files in output/ if not specified.

        Returns:
            Python script as a string.
        """
        ckpt_list = repr(checkpoint_files) if checkpoint_files else "None"
        return f'''#!/usr/bin/env python3
"""Upload checkpoint/results to Cloudflare R2 (auto-generated by vastai-gpu-runner).

Modes:
    --checkpoint    Chunked large-file upload (delta bytes) + checkpoint files
    --done          Final chunk flush + all output files + DONE marker
    (no args)       Same as --done
"""
import argparse
import json as _json
import os
import sys

import boto3

BUCKET = "{self.bucket}"
PREFIX = "{self.prefix}/{batch_id}/{job_name}/"
WORKSPACE = "{workspace}"
OUTPUT = os.path.join(WORKSPACE, "output")
CHUNK_STATE_FILE = os.path.join(WORKSPACE, "chunk_upload_state.json")
LARGE_FILE = "{large_file}"  # Empty string = no chunked upload
CHECKPOINT_FILES = {ckpt_list}  # None = upload all files in output/

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("R2_ENDPOINT", ""),
    aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID", ""),
    aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY", ""),
    region_name="auto",
)


def _load_chunk_state() -> dict:
    if os.path.exists(CHUNK_STATE_FILE):
        try:
            return _json.loads(open(CHUNK_STATE_FILE).read())
        except Exception:
            pass
    return {{"offset": 0, "chunk_index": 0}}


def _save_chunk_state(state: dict) -> None:
    with open(CHUNK_STATE_FILE, "w") as f:
        f.write(_json.dumps(state))


def _flush_large_file_chunk():
    """Flush the next chunk of the large file.

    Returns:
        - ``"uploaded"`` if a new chunk was uploaded successfully.
        - ``"none"`` if there is no new data to send (no chunk produced).
        - ``"failed"`` if a chunk was attempted but the upload failed.
    """
    if not LARGE_FILE:
        return "none"
    state = _load_chunk_state()
    offset = state["offset"]
    chunk_index = state["chunk_index"]
    file_path = os.path.join(OUTPUT, LARGE_FILE)
    if not os.path.exists(file_path):
        return "none"
    file_size = os.path.getsize(file_path)
    if file_size <= offset:
        return "none"
    chunk_size = file_size - offset
    stem, ext = os.path.splitext(LARGE_FILE)
    chunk_key = PREFIX + f"{{stem}}_chunk_{{chunk_index:03d}}{{ext}}"
    tmp_path = os.path.join(WORKSPACE, "_chunk.tmp")
    try:
        with open(file_path, "rb") as src, open(tmp_path, "wb") as dst:
            src.seek(offset)
            remaining = chunk_size
            while remaining > 0:
                block = src.read(min(remaining, 8 * 1024 * 1024))
                if not block:
                    break
                dst.write(block)
                remaining -= len(block)
        s3.upload_file(tmp_path, BUCKET, chunk_key)
        state["offset"] = file_size
        state["chunk_index"] = chunk_index + 1
        _save_chunk_state(state)
        print(f"  chunk {{chunk_index}}: {{chunk_size}} bytes")
        return "uploaded"
    except Exception as exc:
        print(f"WARN: chunk upload failed: {{exc}}")
        return "failed"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def upload_checkpoint() -> int:
    """Best-effort checkpoint upload. NEVER publishes DONE marker."""
    uploaded = 0
    if _flush_large_file_chunk() == "uploaded":
        uploaded += 1
    files = CHECKPOINT_FILES or (
        [f for f in os.listdir(OUTPUT) if f != LARGE_FILE] if os.path.isdir(OUTPUT) else []
    )
    for fname in files:
        local_path = os.path.join(OUTPUT, fname)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            try:
                s3.upload_file(local_path, BUCKET, PREFIX + fname)
                uploaded += 1
            except Exception as exc:
                print(f"WARN: checkpoint upload failed for {{fname}}: {{exc}}")
    return uploaded


def _upload_exitcode_job() -> bool:
    """Upload worker.exitcode for job-based workers. Required for DONE.

    Absence is treated as a failure because we cannot verify the
    workload outcome without it; final ``DONE`` must not be published.
    """
    exitcode_path = os.path.join(WORKSPACE, "worker.exitcode")
    if not os.path.exists(exitcode_path):
        print(
            "FAIL: worker.exitcode missing — refusing to publish DONE marker",
            file=sys.stderr,
        )
        return False
    try:
        s3.upload_file(exitcode_path, BUCKET, PREFIX + "worker.exitcode")
        return True
    except Exception as exc:
        print(f"FAIL: worker.exitcode upload failed: {{exc}}", file=sys.stderr)
        return False


def _upload_failures_log_path_job() -> str:
    """Return the path of the local upload-failure sentinel file."""
    return os.path.join(WORKSPACE, "upload_failures.log")


def _has_unresolved_upload_failures_job() -> bool:
    """Return True iff the local sentinel lists any unresolved failure."""
    log_path = _upload_failures_log_path_job()
    if not os.path.exists(log_path):
        return False
    try:
        with open(log_path) as fh:
            for line in fh:
                if line.strip():
                    return True
    except Exception:
        # Treat unreadable sentinel as unresolved.
        return True
    # Job uploader publishes all output files in a single --done
    # call; we do not enforce per-prediction completion markers here
    # because the job workload has no notion of "predictions".
    # The failure sentinel alone is sufficient to refuse DONE.
    return False


def upload_done_marker() -> int:
    """Upload DONE marker + worker.exitcode. Returns 0 only on full success.

    Returns:
        0 on success; non-zero on any required upload failure. The
        script exits with this code so the worker treats transport
        failure as a non-success without crashing.
    """
    if _has_unresolved_upload_failures_job():
        print(
            "FAIL: unresolved per-prediction upload failures in "
            "upload_failures.log — refusing to publish job DONE.",
            file=sys.stderr,
        )
        return 4
    if not _upload_exitcode_job():
        return 1
    try:
        s3.put_object(Bucket=BUCKET, Key=PREFIX + "DONE", Body=b"")
    except Exception as exc:
        print(f"FAIL: DONE marker upload failed: {{exc}}", file=sys.stderr)
        return 2
    return 0


def upload_all() -> int:
    """Final upload — chunk flush + all output files + DONE. Fail closed.

    Returns:
        0 on full success; non-zero on any required final upload
        failure. The script exits with this code.
    """
    failures: list[str] = []
    chunk_status = _flush_large_file_chunk()
    if chunk_status == "failed":
        failures.append(LARGE_FILE or "<chunk>")
    if os.path.isdir(OUTPUT):
        for fname in os.listdir(OUTPUT):
            if fname == LARGE_FILE:
                continue
            local_path = os.path.join(OUTPUT, fname)
            if os.path.isfile(local_path):
                try:
                    s3.upload_file(local_path, BUCKET, PREFIX + fname)
                except Exception as exc:
                    failures.append(fname)
                    print(f"WARN: upload failed for {{fname}}: {{exc}}")
    if failures:
        print(
            f"FAIL: {{len(failures)}} required file(s) failed to upload; "
            "omitting DONE marker.",
            file=sys.stderr,
        )
        return 3
    if _has_unresolved_upload_failures_job():
        print(
            "FAIL: unresolved per-prediction upload failures in "
            "upload_failures.log — refusing to publish job DONE.",
            file=sys.stderr,
        )
        return 4
    if not _upload_exitcode_job():
        return 1
    try:
        s3.put_object(Bucket=BUCKET, Key=PREFIX + "DONE", Body=b"")
    except Exception as exc:
        print(f"FAIL: DONE marker upload failed: {{exc}}", file=sys.stderr)
        return 2
    return 0


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", action="store_true")
parser.add_argument("--done", action="store_true")
args = parser.parse_args()

if args.checkpoint:
    n = upload_checkpoint()
    print(f"R2: checkpoint uploaded ({{n}} files)")
elif args.done:
    rc = upload_all()
    print(f"R2: final upload (rc={{rc}})")
    sys.exit(rc)
else:
    rc = upload_all()
    print(f"R2: all uploaded (rc={{rc}})")
    sys.exit(rc)
'''
