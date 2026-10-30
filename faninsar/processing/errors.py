"""Typed errors for SAR processing contract violations."""

from __future__ import annotations

from typing import Never

from faninsar.logging import reject


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
    reject(f"processing state rejected: {message}", InvalidProcessingStateError)


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
    reject(f"processing grid rejected: {message}", GridMismatchError)


def reject_pair_configuration(message: str) -> Never:
    """Log and reject a legacy pair-shaped public configuration.

    Parameters
    ----------
    message : str
        Migration guidance explaining the Stack configuration boundary.

    Raises
    ------
    RuntimeError
        Always raised after the rejection is logged.

    """
    reject(f"legacy pair configuration rejected: {message}", RuntimeError)
