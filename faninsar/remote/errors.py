"""Error types for the provider-neutral remote boundary."""

from __future__ import annotations

from typing import NoReturn


class RemoteError(Exception):
    """Base class for errors raised by the remote boundary."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        """Initialize an error with a stable machine-readable reason."""
        self.reason = reason
        super().__init__(message or reason)


def _fail(
    error_type: type[RemoteError], reason: str, message: str | None = None
) -> NoReturn:
    """Raise a remote error while keeping reason construction centralized."""
    raise error_type(reason, message)


class RemoteQueryError(RemoteError):
    """A spatial query or catalog selection is invalid."""


class RemoteAccessError(RemoteError):
    """A catalog, profile, endpoint, or transfer is unavailable."""


class RemoteLimitError(RemoteError):
    """A finite operation budget was exceeded."""


class RemoteIntegrityError(RemoteError):
    """Transferred bytes or a published destination failed validation."""


__all__ = [
    "RemoteAccessError",
    "RemoteError",
    "RemoteIntegrityError",
    "RemoteLimitError",
    "RemoteQueryError",
    "_fail",
]
