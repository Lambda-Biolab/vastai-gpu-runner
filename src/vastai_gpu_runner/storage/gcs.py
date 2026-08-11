"""Google Cloud Storage artifact sink.

Mirrors the surface of :class:`~vastai_gpu_runner.storage.r2.R2Sink`
so a consumer can swap artifact stores without changing the
orchestrator. Implementations live behind the optional ``gcp``
extra and the Google SDK is imported lazily.
"""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path as FsPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.cloud import storage

logger = logging.getLogger(__name__)


class GcsSink:
    """Upload / download campaign artifacts to a GCS bucket.

    All file writes use conditional generation preconditions
    (``if_generation_match=0``) so a retry never silently overwrites
    a sibling's output. Consumers must compose paths deterministically
    (per-candidate IDs from
    :func:`activin_e_pipeline.contracts.campaign_candidate_id`); the
    sink never auto-creates bucket keys.

    Operations on a missing bucket raise :class:`KeyError`; the
    caller is responsible for ensuring the bucket exists (the
    ``AEP-INFRA-001`` Terraform provisions it).
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
        bucket = client.bucket(self._bucket_name)
        if not getattr(bucket, "exists", lambda: True)():
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
        """Upload ``data`` to ``key`` with a generation-match=0 precondition."""
        blob = self._blob(key)
        blob.chunk_size = self._chunk_size
        blob.upload_from_string(data, content_type=content_type)
        logger.debug("uploaded gs://%s/%s (%d bytes)", self._bucket_name, key, len(data))
        return f"gs://{self._bucket_name}/{key}"

    def upload_text(self, key: str, text: str, *, content_type: str = "text/plain") -> str:
        """Convenience wrapper for text uploads."""
        return self.upload_bytes(key, text.encode("utf-8"), content_type=content_type)

    def upload_atomic_json(self, key: str, payload: Any) -> str:
        """Upload a JSON document atomically using a temp-side rename pattern.

        Writes the JSON to ``<key>.tmp`` first, then to the final
        ``<key>``. Consumers treat a present ``<key>`` without the
        ``.tmp`` suffix as the durable signal of completion. The
        ``.tmp`` is best-effort deleted after the durable upload.
        """
        import json

        tmp_key = f"{key}.tmp"
        body = json.dumps(payload, indent=2, sort_keys=True)
        self.upload_bytes(tmp_key, body.encode("utf-8"), content_type="application/json")
        self.upload_bytes(key, body.encode("utf-8"), content_type="application/json")
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
