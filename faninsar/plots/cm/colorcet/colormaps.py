"""Colorcet Colormap Loader with dynamic loading capability."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger

from ..cmap_loader import ColormapLoader  # noqa: TID252

if TYPE_CHECKING:
    from ..enhanced_colormap import EnhancedLinearSegmentedColormap  # noqa: TID252

logger = setup_logger(__name__)


class ColorcetColormaps(ColormapLoader):
    """Colorcet colormap loader with dynamic loading.

    This loader provides access to colorcet colormaps with dynamic loading capability.
    colorcet provides perceptually uniform colormaps.
    """

    def __init__(self) -> None:
        """Initialize colorcet colormap loader."""
        super().__init__(Path(__file__).parent.absolute())

    @property
    def names(self) -> list[str]:
        """Return list of available colorcet colormap names.

        Colorcet stores each colormap as a ``<name>.csv`` mapping file (whose
        first line points to the actual data CSV). We discover names by
        scanning for single-line ``.csv`` files whose stem is *not* a CET
        data file name (those start with ``CET_``).
        """
        if self._names is None:
            self._names = sorted(
                f.stem
                for f in self.data_dir.iterdir()
                if f.suffix == ".csv" and not f.stem.startswith("CET_")
            )
        return self._names

    def _load_colormap_data(self, name: str) -> np.ndarray:
        """Load colorcet colormap data from file.

        Colorcet uses a mapping system where each colormap name maps to a CSV file.

        Parameters
        ----------
        name : str
            Name of the colormap

        Returns
        -------
        np.ndarray
            Numpy array containing colormap data

        """
        mapping_file = self.data_dir / f"{name}.csv"
        if not mapping_file.exists():
            msg = f"Colorcet mapping file not found: {mapping_file}"
            logger.error(msg, stacklevel=2)
            raise FileNotFoundError(msg)

        with mapping_file.open() as f:
            actual_filename = f.readline().strip()

        actual_file = self.data_dir / actual_filename
        if not actual_file.exists():
            msg = f"Colorcet colormap file not found: {actual_file}"
            logger.error(msg, stacklevel=2)
            raise FileNotFoundError(msg)

        return np.loadtxt(actual_file, delimiter=",")


# Create a global instance
_colorcet_colormaps = ColorcetColormaps()

# Export all colormap names for backward compatibility
__all__ = _colorcet_colormaps.__all__


# Create module-level attributes for backward compatibility
def __getattr__(name: str) -> EnhancedLinearSegmentedColormap:
    """Module-level attribute access for colormaps."""
    return getattr(_colorcet_colormaps, name)


def __dir__() -> list[str]:
    """Return list of available module attributes."""
    return _colorcet_colormaps.__all__


# For convenience, also expose the names
names = _colorcet_colormaps.names
