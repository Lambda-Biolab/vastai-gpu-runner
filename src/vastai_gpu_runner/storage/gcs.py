"""Google Cloud Storage artifact sink.

Mirrors the surface of :class:`~vastai_gpu_runner.storage.r2.R2Sink`
so a consumer can swap artifact stores without changing the
orchestrator. Implementations live behind the optional ``gcp``
extra and the Google SDK is imported lazily.

JSON uploads use ``upload_atomic_json`` (stage-then-copy with
overwrite semantics on the final key). Plain ``upload_bytes`` /
``upload_file`` use ``if_generation_match=0`` so two siblings
cannot silently overwrite each other (create-only semantics). The
``upload_cas_write`` / ``read_cas`` pair implements a generation-
preconditioned CAS pattern for controller snapshots: a reader
records the current generation, builds a new value, and the writer
only succeeds if the object is still at the recorded generation.

All CAS and create-only write paths raise :class:`GcsPreconditionFailed` so the
caller has a single, stable exception class to catch. The SDK's
own ``google.api_core.exceptions.PreconditionFailed`` is wrapped
using ``raise X from exc`` so the underlying error is preserved as
``__cause__`` for inspection. Other SDK errors (auth, network,
permissions) propagate unchanged. GCS completion is represented by the
expected object set; this sink does not create or require a ``DONE`` marker.

``upload_atomic_json`` retains its historical name for compatibility. GCS
has no atomic rename operation here: the temporary object and final object
are separate writes, and the final object is the durable completion signal.
"""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path, PurePosixPath
from pathlib import Path as FsPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.cloud import storage

logger = logging.getLogger(__name__)


class GcsPreconditionFailed(Exception):  # noqa: N818 — name is the public contract
    """Raised when a generation precondition fails on a GCS write.

    Stable, provider-neutral exception for the controller snapshot
    CAS pattern. The underlying SDK exception (typically
    ``google.api_core.exceptions.PreconditionFailed`` for create-only
    or in-place CAS mismatch) is preserved as ``__cause__`` so it
    remains inspectable.
    """


def _is_precondition_failure(exc: BaseException) -> bool:
    """Return True iff ``exc`` is the SDK's ``PreconditionFailed`` (or already wrapped)."""
    if isinstance(exc, GcsPreconditionFailed):
        return True
    try:
        from google.api_core.exceptions import PreconditionFailed
    except ImportError:
        # Without google.api_core we cannot verify the type; re-raise
        # any error so we never claim a non-precondition failure is one.
        return False
    return isinstance(exc, PreconditionFailed)


def _resolve_precondition_failure(exc: BaseException) -> GcsPreconditionFailed:
    """Wrap ``exc`` as :class:`GcsPreconditionFailed` if it is a precondition failure.

    Non-precondition failures (auth, network, permissions) are
    re-raised unchanged so callers can handle them with their
    native types. The wrap is a strict ``isinstance`` check against
    the SDK's ``PreconditionFailed`` class — no string matching.
    """
    if not _is_precondition_failure(exc):
        raise exc
    if isinstance(exc, GcsPreconditionFailed):
        return exc
    return GcsPreconditionFailed(str(exc))


class GcsSink:
    """Upload / download artifacts to a GCS bucket.

    Public write surface:

    * ``upload_bytes`` / ``upload_file`` / ``upload_text`` — create-only
      writes (``if_generation_match=0``). Sibling artefacts cannot
      silently overwrite each other; a second upload of the same key
      raises :class:`GcsPreconditionFailed`.
    * ``upload_atomic_json`` — durable, retriable staged JSON write.
      The final key may already exist (overwrite semantics); the temp
      key is cleaned up on success and stale ``.tmp`` files from a
      previous attempt are recovered. It is not an atomic rename.
    * ``upload_cas_write`` — generation-aware CAS update for the
      controller snapshot pattern.

    Public read surface:

    * ``read_cas`` returns ``(data, generation)`` from a single
      generation-aware read. If the object is missing, returns
      ``(b"", None)``. If the underlying object's generation moves
      between the metadata reload and the download, the read fails
      with :class:`GcsPreconditionFailed` so the caller can retry
      with the new generation.

    Operations on a missing bucket raise :class:`KeyError`; the
    caller is responsible for ensuring the bucket exists
    (Terraform / `gcloud buckets create` / equivalent).
    """

    def __init__(
        self,
        bucket_name: str,
        client: storage.Client | None = None,
        chunk_size: int = 40 * 1024 * 1024,
    ) -> None:
        """Construct the sink; the client is created lazily if omitted."""
        self._bucket_name = bucket_name
        self._client = client
        self._chunk_size = chunk_size

    @property
    def bucket_name(self) -> str:
        """Return the configured GCS bucket name."""
        return self._bucket_name

    def _require_client(self) -> storage.Client:
        if self._client is not None:
            return self._client
        from google.cloud import storage

        self._client = storage.Client()
        return self._client

    def _bucket(self) -> Any:
        client = self._require_client()
        try:
            bucket = client.bucket(self._bucket_name)
        except LookupError as exc:
            raise KeyError(f"GCS bucket not found: {self._bucket_name}") from exc
        bucket_exists = getattr(bucket, "exists", None)
        if bucket_exists is None:
            raise KeyError(f"GCS bucket not found: {self._bucket_name}")
        if not bucket_exists():
            raise KeyError(f"GCS bucket not found: {self._bucket_name}")
        return bucket

    def _blob(self, key: str) -> Any:
        return self._bucket().blob(key)

    def upload_file(self, key: str, source: FsPath, *, content_type: str | None = None) -> str:
        """Upload ``source`` to ``key`` with a generation-match=0 precondition.

        Returns the gs:// URI of the uploaded object. The reader should
        verify :attr:`sha256` matches the local hash before treating the
        upload as complete.
        """
        data = source.read_bytes()
        return self.upload_bytes(key, data, content_type=content_type)

    def upload_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> str:
        """Upload ``data`` to ``key`` with a generation-match=0 precondition.

        The ``0`` precondition means the upload only succeeds if the
        object does not yet exist. A second upload of the same key by
        a sibling fails fast (with :class:`GcsPreconditionFailed`)
        rather than overwriting, which keeps artifact-key collisions
        from being silently destructive. Non-precondition SDK errors
        (auth, network, permissions) propagate unchanged.
        """
        blob = self._blob(key)
        blob.chunk_size = self._chunk_size
        try:
            blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
        except Exception as exc:
            raise _resolve_precondition_failure(exc) from exc
        logger.debug("uploaded gs://%s/%s (%d bytes)", self._bucket_name, key, len(data))
        return f"gs://{self._bucket_name}/{key}"

    def upload_text(self, key: str, text: str, *, content_type: str = "text/plain") -> str:
        """Convenience wrapper for text uploads."""
        return self.upload_bytes(key, text.encode("utf-8"), content_type=content_type)

    def _write_bytes_overwrite(
        self, key: str, data: bytes, *, content_type: str | None = None
    ) -> str:
        """Internal helper: write without ``if_generation_match=0`` precondition.

        Used by :meth:`upload_atomic_json` so its final key can be
        overwritten by a later call. The public ``upload_bytes``
        keeps the create-only precondition — callers that need
        cross-sibling isolation use that.
        """
        blob = self._blob(key)
        blob.chunk_size = self._chunk_size
        blob.upload_from_string(data, content_type=content_type)
        return f"gs://{self._bucket_name}/{key}"

    def upload_atomic_json(self, key: str, payload: Any) -> str:
        """Stage a JSON document, then overwrite its durable final key.

        The pattern is:
        1. Clean up any stale ``<key>.tmp`` from a previous crashed
           attempt (best-effort delete; a missing ``.tmp`` is fine).
        2. Write ``<key>.tmp`` with create-only semantics on the
           unique-per-write key.
         3. Write the final ``<key>`` with overwrite semantics. This is
            not an atomic rename; the final key is the completion signal.
        4. Best-effort delete of ``<key>.tmp``.

        Consumers treat a present ``<key>`` without the ``.tmp``
        suffix as the durable signal of completion.
        """
        import json

        tmp_key = f"{key}.tmp"
        body = json.dumps(payload, indent=2, sort_keys=True)
        body_bytes = body.encode("utf-8")
        # Stale tmp recovery: a previous attempt may have left a
        # ``.tmp`` behind (crash, network blip). Clean it up so the
        # create-only write below does not spuriously fail.
        self._delete_best_effort(tmp_key)
        self.upload_bytes(tmp_key, body_bytes, content_type="application/json")
        self._write_bytes_overwrite(key, body_bytes, content_type="application/json")
        self._delete_best_effort(tmp_key)
        return f"gs://{self._bucket_name}/{key}"

    def download_bytes(self, key: str) -> bytes:
        """Download ``key`` and return the raw bytes."""
        return self._blob(key).download_as_bytes()

    def download_to_file(self, key: str, destination: FsPath) -> FsPath:
        """Stream ``key`` to ``destination`` and return the local path."""
        data = self.download_bytes(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return destination

    def list_blobs(self, prefix: str = "") -> list[str]:
        """Return the blob names under ``prefix`` (sorted, with the prefix stripped).

        Empty prefix returns every blob in the bucket. The full key
        is reconstructable as ``f"{prefix}{name}"`` for callers that
        need the original key. The result is sorted lexicographically
        so callers do not depend on the underlying SDK's ordering.
        """
        bucket = self._bucket()
        listed = bucket.list_blobs(prefix=prefix)
        return sorted(blob.name[len(prefix) :] for blob in listed)

    def download_all(self, prefix: str, dest: FsPath) -> list[FsPath]:
        """Download every blob under ``prefix`` into ``dest`` and return the paths.

        The destination directory is created if missing. The local
        layout mirrors the key layout under ``prefix``. Returns the
        list of written paths, in key order.

        Path safety: every blob key is treated as a *relative* path
        under ``dest``. Absolute paths, empty path components, and
        ``../`` traversal segments are rejected with
        :class:`ValueError`. The resolved target path is also
        verified to remain under ``dest`` after symlink resolution
        so a maliciously placed symlink in the bucket cannot escape
        the destination tree.
        """
        bucket = self._bucket()
        listed = list(bucket.list_blobs(prefix=prefix))
        dest_resolved = Path(dest).resolve()
        written: list[FsPath] = []
        for blob in listed:
            rel = blob.name[len(prefix) :]
            if not rel:
                # Skip the prefix itself when it exists as a zero-byte
                # placeholder object.
                continue
            target = self._safe_join_under(dest, dest_resolved, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(target))
            written.append(target)
        return written

    @staticmethod
    def _safe_join_under(dest: FsPath, dest_resolved: Path, rel: str) -> Path:
        """Validate ``rel`` and return a safe target path under ``dest``.

        Rejects:
        * empty relative paths
        * absolute paths (leading ``/``)
        * path components that include ``..`` traversal
        * paths containing NUL bytes
        * resolved targets that escape ``dest`` (e.g. via symlinks)
        """
        if not rel:
            raise ValueError("download_all: empty relative path under prefix")
        if rel.startswith("/"):
            raise ValueError(f"download_all: absolute path in blob key: {rel!r}")
        # PurePosixPath treats '/' as a separator and does not follow
        # Windows conventions, which matches the GCS semantic for keys.
        parts = PurePosixPath(rel).parts
        if any(part == ".." for part in parts):
            raise ValueError(f"download_all: '..' traversal in blob key: {rel!r}")
        if "\x00" in rel:
            raise ValueError(f"download_all: NUL byte in blob key: {rel!r}")
        target = (dest / rel).resolve()
        try:
            target.relative_to(dest_resolved)
        except ValueError as exc:
            raise ValueError(
                f"download_all: resolved target {target} escapes dest {dest_resolved}"
            ) from exc
        return target

    def read_cas(self, key: str) -> tuple[bytes, int | None]:
        """Read ``key`` and return ``(data, generation)`.

        ``generation`` is the live object generation; combine with
        :meth:`upload_cas_write` to perform a compare-and-set update
        on the same object. Returns ``(b"", None)`` when the object
        does not yet exist (consumers treat that as "first writer
        wins").

        The data and generation are returned from a single
        generation-aware read: the metadata ``reload()`` pins the
        generation, then ``download_as_bytes(if_generation_match=gen)``
        reads at that generation. If the generation moves between
        the two steps (a concurrent writer intervened), the
        download fails with :class:`GcsPreconditionFailed` so the
        caller can retry with the new generation.
        """
        blob = self._blob(key)
        # Reload metadata first so the blob's `generation` is fresh.
        # If the object does not exist, `generation` stays None.
        try:
            blob.reload()
        except Exception as exc:  # pragma: no cover - depends on storage client
            try:
                from google.api_core.exceptions import NotFound
            except ImportError:
                raise
            if isinstance(exc, NotFound):
                return b"", None
            raise
        gen = getattr(blob, "generation", None)
        if gen is None:
            return b"", None
        # Read with the generation precondition so the bytes we
        # return are guaranteed to match the generation we report.
        try:
            data = blob.download_as_bytes(if_generation_match=gen)
        except Exception as exc:
            raise _resolve_precondition_failure(exc) from exc
        return data, int(gen)

    def upload_cas_write(
        self,
        key: str,
        data: bytes,
        expected_generation: int | None,
        *,
        content_type: str | None = None,
    ) -> int:
        """CAS upload: succeed only if the live generation matches ``expected_generation``.

        Pass ``expected_generation=0`` to require "object does not yet
        exist" (create-only semantics). On success the new
        generation is returned. On mismatch the SDK raises
        ``PreconditionFailed``; the sink wraps it as
        :class:`GcsPreconditionFailed` so the caller has a single
        stable exception class to catch. Other SDK errors (auth,
        network, permissions) propagate unchanged.

        The Google SDK accepts ``if_generation_match`` for any
        generation value, so a single ``upload_from_string`` call
        covers both the create-only and the in-place CAS paths.
        """
        if expected_generation is None:
            expected_generation = 0
        blob = self._blob(key)
        blob.chunk_size = self._chunk_size
        try:
            blob.upload_from_string(
                data, content_type=content_type, if_generation_match=expected_generation
            )
        except Exception as exc:
            raise _resolve_precondition_failure(exc) from exc
        # CAS rewrites in place by passing the expected generation.
        # Reload to expose the new generation number.
        try:
            blob.reload()
        except Exception:  # pragma: no cover - depends on storage client
            return 0
        new_gen = getattr(blob, "generation", None)
        return int(new_gen) if new_gen is not None else 0

    def exists(self, key: str) -> bool:
        """Return True if ``key`` exists in the configured bucket."""
        try:
            return bool(self._blob(key).exists())
        except Exception:  # pragma: no cover - depends on storage client
            return False

    def sha256(self, key: str) -> str:
        """Return the SHA-256 of ``key``'s contents."""
        data = self.download_bytes(key)
        return hashlib.sha256(data).hexdigest()

    def _delete_best_effort(self, key: str) -> None:
        try:
            self._blob(key).delete()
        except Exception:  # pragma: no cover - best effort
            logger.warning("failed to delete temp artifact gs://%s/%s", self._bucket_name, key)


def upload_json_atomic(sink: GcsSink, key: str, payload: Any) -> str:
    """Thin wrapper kept for compatibility with consumer code paths."""
    return sink.upload_atomic_json(key, payload)


def streaming_upload(sink: GcsSink, key: str, source: io.BufferedIOBase) -> str:
    """Upload a file-like object to ``key`` via streaming I/O."""
    data = source.read()
    return sink.upload_bytes(key, data)
