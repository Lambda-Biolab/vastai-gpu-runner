"""Stable errors raised by managed-job providers."""

from __future__ import annotations

from typing import Any


class ManagedJobError(Exception):
    """Base class for errors in the provider-neutral managed-job contract."""


class ManagedJobTransientError(ManagedJobError):
    """A provider error that is normally safe to retry."""


class ManagedJobPermanentError(ManagedJobError):
    """A provider error that will not be fixed by an immediate retry."""


class ManagedJobConflictError(ManagedJobPermanentError):
    """The requested job conflicts with an existing provider resource."""


ManagedJobAlreadyExistsError = ManagedJobConflictError


class ManagedJobNotFoundError(ManagedJobPermanentError):
    """The requested managed-job resource does not exist."""


def map_gcp_exception(
    exc: BaseException, *, operation: str = "managed-job operation"
) -> ManagedJobError:
    """Map a GCP SDK exception to the stable managed-job error contract."""
    if isinstance(exc, ManagedJobError):
        return exc

    try:
        from google.api_core import exceptions as gcp_exceptions
    except ImportError:
        return ManagedJobPermanentError(f"{operation} failed: {exc}")

    if isinstance(exc, gcp_exceptions.AlreadyExists):
        return ManagedJobConflictError(f"{operation} conflicts with an existing job: {exc}")
    if isinstance(exc, gcp_exceptions.NotFound):
        return ManagedJobNotFoundError(f"{operation} target was not found: {exc}")
    transient_types: tuple[type[Any], ...] = (
        gcp_exceptions.Aborted,
        gcp_exceptions.DeadlineExceeded,
        gcp_exceptions.InternalServerError,
        gcp_exceptions.ResourceExhausted,
        gcp_exceptions.ServiceUnavailable,
        gcp_exceptions.TooManyRequests,
    )
    if isinstance(exc, transient_types):
        return ManagedJobTransientError(f"{operation} failed transiently: {exc}")
    return ManagedJobPermanentError(f"{operation} failed: {exc}")
