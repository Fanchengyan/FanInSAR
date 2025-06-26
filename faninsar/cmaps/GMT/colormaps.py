"""GMT Colormap Loader with dynamic loading capability."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger

from .. import ColormapLoader  # noqa: TID252

if TYPE_CHECKING:
    from ..enhanced_colormap import EnhancedLinearSegmentedColormap  # noqa: TID252

logger = setup_logger(__name__)


class GMTColormaps(ColormapLoader):
    """GMT colormap loader with dynamic loading.

    This loader provides access to GMT (Generic Mapping Tools) colormaps
    with dynamic loading capability.
    """

    def __init__(self) -> None:
        """Initialize GMT colormap loader."""
        super().__init__(Path(__file__).parent.absolute())
        self._colormap_names = [
            "abyss",
            "bathy",
            "cool",
            "copper",
            "cubhelix",
            "cyclic",
            "dem1",
            "dem2",
            "dem3",
            "dem4",
            "drywet",
            "earth",
            "elevation",
            "etopo1",
            "geo",
            "globe",
            "gray",
            "haxby",
            "hot",
            "inferno",
            "jet",
            "magma",
            "nighttime",
            "no_green",
            "ocean",
            "plasma",
            "polar",
            "rainbow",
            "red2green",
            "relief",
            "seafloor",
            "sealand",
            "seis",
            "split",
            "srtm",
            "terra",
            "topo",
            "turbo",
            "viridis",
            "world",
            "wysiwyg",
        ]

    @property
    def names(self) -> list[str]:
        """Return list of available GMT colormap names."""
        return self._colormap_names

    def _load_colormap_data(self, name: str) -> np.ndarray:
        """Load GMT colormap data from file.

        Parameters
        ----------
        name : str
            Name of the colormap

        Returns
        -------
        np.ndarray
            Numpy array containing colormap data

        """
        cmap_file = self.data_dir / name / f"{name}.txt"
        if not cmap_file.exists():
            msg = f"GMT colormap file not found: {cmap_file}"
            logger.error(msg, stacklevel=2)
            raise FileNotFoundError(msg)

        return np.loadtxt(cmap_file)


# Create a global instance
_gmt_colormaps = GMTColormaps()

# Export all colormap names for backward compatibility
__all__ = _gmt_colormaps.__all__


# Create module-level attributes for backward compatibility
def __getattr__(name: str) -> EnhancedLinearSegmentedColormap:
    """Module-level attribute access for colormaps."""
    return getattr(_gmt_colormaps, name)


def __dir__() -> list[str]:
    """Return list of available module attributes."""
    return _gmt_colormaps.__all__


# For convenience, also expose the names
names = _gmt_colormaps.names
