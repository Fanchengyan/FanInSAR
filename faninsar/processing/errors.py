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
