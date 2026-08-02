"""Typed errors for Sentinel-1 product parsing."""

from __future__ import annotations

from typing import Never

from faninsar.logging import setup_logger

logger = setup_logger(__name__)


class Sentinel1ProductError(RuntimeError):
    """Raised when a Sentinel-1 product cannot be opened safely."""


class UnsupportedPolarizationError(Sentinel1ProductError):
    """Raised when a requested polarization is absent or unsupported."""


def reject_product(message: str) -> Never:
    """Log and raise a product-level error.

    Parameters
    ----------
    message : str
        Diagnostic describing the rejected product state.

    Raises
    ------
    Sentinel1ProductError
        Always raised after logging.

    """
    logger.error("%s", message)
    raise Sentinel1ProductError(message)
