"""Tests for the in-memory fakes shared between GcpBatchRunner and GcsSink.

The fakes in :mod:`vastai_gpu_runner.managed_jobs.gcp_batch` are
reusable across multiple test modules. This file pins their
semantics with explicit tests so the fakes do not silently drift
from the subset of the Google SDK surface they claim to support.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from vastai_gpu_runner.managed_jobs.gcp_batch import FakeGcsClient


def _precondition_failed_type() -> type[BaseException]:
    """Return the exception class the fake raises on a generation mismatch."""
    try:
        from google.api_core.exceptions import PreconditionFailed
    except ImportError:  # pragma: no cover - depends on api_core
        return RuntimeError
    return PreconditionFailed


def test_fake_gcs_client_bucket_lookup_and_create() -> None:
    client = FakeGcsClient()
    client.create_bucket("alpha")
    bucket = client.bucket("alpha")
    assert bucket.exists() is True
    assert client.exists("alpha") is True

    with pytest.raises(LookupError):
        client.bucket("beta")


def test_fake_gcs_client_blob_round_trip() -> None:
    client = FakeGcsClient()
    client.create_bucket("alpha")
    blob = client.bucket("alpha").blob("data/file.txt")
    blob.upload_from_string(b"hello", content_type="text/plain")
    assert blob.exists() is True
    assert blob.download_as_bytes() == b"hello"
    assert client.bucket("alpha").exists("data/file.txt") is True
    assert client.uploads[-1] == ("alpha", "data/file.txt", b"hello", "text/plain", None)


def test_fake_gcs_client_blob_delete_and_exists() -> None:
    client = FakeGcsClient()
    client.create_bucket("alpha")
    blob = client.bucket("alpha").blob("data/file.txt")
    blob.upload_from_string(b"hello")
    assert blob.exists() is True
    blob.delete()
    assert blob.exists() is False
    assert client.bucket("alpha").exists("data/file.txt") is False


def test_fake_gcs_blob_size_and_download_to_file(tmp_path: Path) -> None:
    client = FakeGcsClient()
    client.create_bucket("alpha")
    blob = client.bucket("alpha").blob("data/file.bin")
    blob.upload_from_string(b"hello world")

    assert blob.size == 11
    assert client.bucket("alpha").blob("absent").size is None

    target = tmp_path / "out.bin"
    with target.open("wb") as handle:
        blob.download_to_file(handle)

    assert target.read_bytes() == b"hello world"
    assert client.downloads[-1] == ("alpha", "data/file.bin", None)


def test_fake_gcs_blob_download_to_file_honours_generation() -> None:
    client = FakeGcsClient()
    client.create_bucket("alpha")
    blob = client.bucket("alpha").blob("data/file.bin")
    blob.upload_from_string(b"hello")

    with pytest.raises(_precondition_failed_type(), match="if_generation_match"):
        blob.download_to_file(io.BytesIO(), if_generation_match=999)
