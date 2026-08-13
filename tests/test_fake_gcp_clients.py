"""Tests for the in-memory fakes shared between GcpBatchRunner and GcsSink.

The fakes in :mod:`vastai_gpu_runner.managed_jobs.gcp_batch` are
reusable across multiple test modules. This file pins their
semantics with explicit tests so the fakes do not silently drift
from the subset of the Google SDK surface they claim to support.
"""

from __future__ import annotations

import pytest

from vastai_gpu_runner.managed_jobs.gcp_batch import FakeGcsClient


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
