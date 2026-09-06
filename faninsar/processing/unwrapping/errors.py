"""Typed errors for spatial phase unwrapping."""

from __future__ import annotations

from faninsar.logging import setup_logger

logger = setup_logger(__name__)


class NoValidSupportError(ValueError):
    """Raised when an unwrap request contains no supported pixel."""


class UnwrapFailedError(RuntimeError):
    """Raised by an orchestration boundary for a failed unwrap result."""

    def __init__(self, message: str, result: object | None = None) -> None:
        """Create an error retaining the failed result for diagnostics."""
        self.result = result
        super().__init__(message)
