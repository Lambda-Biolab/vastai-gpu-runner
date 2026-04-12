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
        from concurrent.futures import ThreadPoolExecutor, as_completed

        local_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.shard_prefix(batch_id, shard_id)}outputs/"

        keys: list[tuple[str, str]] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel_path = key[len(prefix) :]
                if rel_path:
                    keys.append((key, rel_path))

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
            "Downloaded %d files from R2 for shard %d (batch %s)",
            len(downloaded),
            shard_id,
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
        from concurrent.futures import ThreadPoolExecutor, as_completed

        local_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.job_prefix(batch_id, job_name)

        keys: list[tuple[str, str]] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel_path = key[len(prefix) :]
                if rel_path:
                    keys.append((key, rel_path))

        for _, rel_path in keys:
            (local_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)

        def _dl_one(item: tuple[str, str]) -> str:
            key, rel_path = item
            self._client.download_file(self.bucket, key, str(local_dir / rel_path))
            return rel_path

        downloaded: list[str] = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_dl_one, item): item for item in keys}
            for future in as_completed(futures):
                try:
                    downloaded.append(future.result())
                except Exception:
                    _, rel_path = futures[future]
                    logger.warning("R2 download failed: %s", rel_path)

        return downloaded

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


def upload_prediction(name: str) -> int:
    pred_dir = os.path.join(WORKSPACE, "outputs", name)
    if not os.path.isdir(pred_dir):
        print(f"WARN: no output dir for {{name}}")
        return 0
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
                print(f"WARN: upload failed for {{rel}}: {{exc}}")
    s3.put_object(Bucket=BUCKET, Key=PREFIX + f"markers/{{name}}.done", Body=b"")
    s3.put_object(
        Bucket=BUCKET,
        Key=BATCH_PREFIX + f"global_markers/{{name}}.done",
        Body=f"shard_{shard_id}".encode(),
    )
    return uploaded


def upload_done_marker() -> None:
    s3.put_object(Bucket=BUCKET, Key=PREFIX + "DONE", Body=b"")
    exitcode_path = os.path.join(WORKSPACE, "worker.exitcode")
    if os.path.exists(exitcode_path):
        s3.upload_file(exitcode_path, BUCKET, PREFIX + "worker.exitcode")


def upload_all() -> int:
    uploaded = 0
    outputs_dir = os.path.join(WORKSPACE, "outputs")
    for root, _dirs, files in os.walk(outputs_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, outputs_dir)
            key = PREFIX + "outputs/" + rel
            try:
                s3.upload_file(local_path, BUCKET, key)
                uploaded += 1
            except Exception as exc:
                print(f"WARN: upload failed for {{rel}}: {{exc}}")
    upload_done_marker()
    return uploaded


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
    n = upload_prediction(args.prediction)
    print(f"R2: {{args.prediction}} uploaded ({{n}} files)")
elif args.done:
    upload_done_marker()
    print("R2: DONE marker uploaded")
else:
    n = upload_all()
    print(f"R2: all uploaded ({{n}} files)")
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


def _flush_large_file_chunk() -> bool:
    if not LARGE_FILE:
        return False
    state = _load_chunk_state()
    offset = state["offset"]
    chunk_index = state["chunk_index"]
    file_path = os.path.join(OUTPUT, LARGE_FILE)
    if not os.path.exists(file_path):
        return False
    file_size = os.path.getsize(file_path)
    if file_size <= offset:
        return False
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
        return True
    except Exception as exc:
        print(f"WARN: chunk upload failed: {{exc}}")
        return False
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def upload_checkpoint() -> int:
    uploaded = 0
    if _flush_large_file_chunk():
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


def upload_done_marker() -> None:
    s3.put_object(Bucket=BUCKET, Key=PREFIX + "DONE", Body=b"")
    exitcode_path = os.path.join(WORKSPACE, "worker.exitcode")
    if os.path.exists(exitcode_path):
        s3.upload_file(exitcode_path, BUCKET, PREFIX + "worker.exitcode")


def upload_all() -> int:
    uploaded = 0
    if _flush_large_file_chunk():
        uploaded += 1
    if os.path.isdir(OUTPUT):
        for fname in os.listdir(OUTPUT):
            if fname == LARGE_FILE:
                continue
            local_path = os.path.join(OUTPUT, fname)
            if os.path.isfile(local_path):
                try:
                    s3.upload_file(local_path, BUCKET, PREFIX + fname)
                    uploaded += 1
                except Exception as exc:
                    print(f"WARN: upload failed for {{fname}}: {{exc}}")
    upload_done_marker()
    return uploaded


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", action="store_true")
parser.add_argument("--done", action="store_true")
args = parser.parse_args()

if args.checkpoint:
    n = upload_checkpoint()
    print(f"R2: checkpoint uploaded ({{n}} files)")
elif args.done:
    n = upload_all()
    print(f"R2: final upload ({{n}} files + DONE)")
else:
    n = upload_all()
    print(f"R2: all uploaded ({{n}} files)")
'''
