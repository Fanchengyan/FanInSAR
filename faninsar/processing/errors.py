"""Typed errors for SAR processing contract violations."""

from __future__ import annotations

from typing import Never

from faninsar.logging import setup_logger

logger = setup_logger(__name__)


class ProcessingContractError(RuntimeError):
    """Base error for invalid processing metadata or state transitions."""


class InvalidProcessingStateError(ProcessingContractError):
    """Raised when a processing stage violates the product state machine."""


class GridMismatchError(ProcessingContractError):
    """Raised when products do not share the required coordinate grid."""


class StageError(ProcessingContractError):
    """Stage failure with stage name, pair identity, and recovery hint.

    Attributes
    ----------
    stage : str
        Name of the failing stage.
    pair : str or None
        Pair identifier when applicable.
    hint : str
        Human-readable recovery or diagnostic hint.

    """

    def __init__(
        self,
        message: str | None = None,
        *,
        stage: str | None = None,
        pair: str | None = None,
        hint: str | None = None,
    ) -> None:
        """Build a StageError, chaining ``__cause__`` when raised from ``except``."""
        self.stage = stage or "unknown"
        self.pair = pair
        self.hint = hint or ""
        parts = [f"stage={self.stage}"]
        if pair:
            parts.append(f"pair={pair}")
        if hint:
            parts.append(hint)
        if message:
            parts.insert(0, message)
        super().__init__(": ".join(parts))


def reject_invalid_state(message: str) -> Never:
    """Log and raise an invalid-state error.

    Parameters
    ----------
    message : str
        Diagnostic explaining the rejected state transition.

    Raises
    ------
    InvalidProcessingStateError
        Always raised after the rejection is logged.

    """
    logger.error("processing state rejected: %s", message)
    raise InvalidProcessingStateError(message)


def reject_grid_mismatch(message: str) -> Never:
    """Log and raise a grid-mismatch error.

    Parameters
    ----------
    message : str
        Diagnostic explaining the incompatible grids.

    Raises
    ------
    GridMismatchError
        Always raised after the rejection is logged.

    """
    logger.error("processing grid rejected: %s", message)
    raise GridMismatchError(message)
