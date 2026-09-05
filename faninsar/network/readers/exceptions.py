"""Custom exceptions for FanInSAR Network products."""

from __future__ import annotations


class NetworkProductError(Exception):
    """Base exception for frame product errors."""


class NetworkGeometryError(NetworkProductError):
    """Base exception for frame geometry errors."""


class GridMismatchError(NetworkGeometryError):
    """Raised when raster grids do not align."""

    def __init__(self, detail: str = "") -> None:
        """Initialise with optional detail message."""
        msg = "Raster grids do not match."
        if detail:
            msg = f"{msg} {detail}"
        super().__init__(msg)


class COGValidationError(NetworkGeometryError):
    """Raised when a raster fails COG validation."""


class MissingGeometryAssetError(NetworkGeometryError):
    """Raised when a required geometry asset is missing."""

    def __init__(self, name: str) -> None:
        """Initialise with the missing asset name."""
        super().__init__(f"Geometry asset '{name}' not found or not created.")


class NetworkInterferogramError(NetworkProductError):
    """Base exception for frame interferogram errors."""


class MissingInterferogramAssetError(NetworkInterferogramError):
    """Raised when a required interferogram asset is missing."""

    def __init__(self, pair: str, name: str) -> None:
        """Initialise with the pair and asset names."""
        super().__init__(f"Interferogram asset '{name}' not found for pair '{pair}'.")


class PairNotFoundError(NetworkInterferogramError):
    """Raised when a requested pair is not in the collection."""

    def __init__(self, pair: str) -> None:
        """Initialise with the missing pair name."""
        super().__init__(f"Pair '{pair}' not found in the interferogram collection.")


class MetadataError(NetworkProductError):
    """Raised when metadata is invalid or missing."""
