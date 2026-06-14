"""Custom exceptions for FanInSAR frame products."""

from __future__ import annotations


class FrameProductError(Exception):
    """Base exception for frame product errors."""


class FrameGeometryError(FrameProductError):
    """Base exception for frame geometry errors."""


class GridMismatchError(FrameGeometryError):
    """Raised when raster grids do not align."""

    def __init__(self, detail: str = "") -> None:
        """Initialise with optional detail message."""
        msg = "Raster grids do not match."
        if detail:
            msg = f"{msg} {detail}"
        super().__init__(msg)


class COGValidationError(FrameGeometryError):
    """Raised when a raster fails COG validation."""


class MissingGeometryAssetError(FrameGeometryError):
    """Raised when a required geometry asset is missing."""

    def __init__(self, name: str) -> None:
        """Initialise with the missing asset name."""
        super().__init__(f"Geometry asset '{name}' not found or not created.")


class FrameInterferogramError(FrameProductError):
    """Base exception for frame interferogram errors."""


class MissingInterferogramAssetError(FrameInterferogramError):
    """Raised when a required interferogram asset is missing."""

    def __init__(self, pair: str, name: str) -> None:
        """Initialise with the pair and asset names."""
        super().__init__(f"Interferogram asset '{name}' not found for pair '{pair}'.")


class PairNotFoundError(FrameInterferogramError):
    """Raised when a requested pair is not in the collection."""

    def __init__(self, pair: str) -> None:
        """Initialise with the missing pair name."""
        super().__init__(f"Pair '{pair}' not found in the interferogram collection.")


class MetadataError(FrameProductError):
    """Raised when metadata is invalid or missing."""
