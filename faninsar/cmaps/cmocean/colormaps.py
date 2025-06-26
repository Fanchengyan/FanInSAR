"""Cmocean Colormap Loader with dynamic loading capability."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger

from .. import ColormapLoader  # noqa: TID252

if TYPE_CHECKING:
    from ..enhanced_colormap import EnhancedLinearSegmentedColormap  # noqa: TID252


logger = setup_logger(__name__)


class CmoceanColormaps(ColormapLoader):
    """Cmocean colormap loader with dynamic loading.

    This loader provides access to cmocean colormaps with dynamic loading capability.
    cmocean provides beautiful colormaps for oceanography.
    """

    def __init__(self) -> None:
        """Initialize cmocean colormap loader."""
        super().__init__(Path(__file__).parent.absolute())
        self._colormap_names = [
            "algae",
            "amp",
            "balance",
            "curl",
            "deep",
            "delta",
            "dense",
            "diff",
            "gray",
            "haline",
            "ice",
            "matter",
            "oxy",
            "phase",
            "rain",
            "solar",
            "speed",
            "tarn",
            "tempo",
            "thermal",
            "topo",
            "turbid",
        ]

    @property
    def names(self) -> list[str]:
        """Return list of available cmocean colormap names."""
        return self._colormap_names

    def _load_colormap_data(self, name: str) -> np.ndarray:
        """Load cmocean colormap data from file.

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
            msg = f"Cmocean colormap file not found: {cmap_file}"
            logger.error(msg, stacklevel=2)
            raise FileNotFoundError(msg)

        return np.loadtxt(cmap_file)


# Create a global instance
_cmocean_colormaps = CmoceanColormaps()

# Export all colormap names for backward compatibility
__all__ = _cmocean_colormaps.__all__


# Create module-level attributes for backward compatibility
def __getattr__(name: str) -> EnhancedLinearSegmentedColormap:
    """Module-level attribute access for colormaps."""
    return getattr(_cmocean_colormaps, name)


def __dir__() -> list[str]:
    """Return list of available module attributes."""
    return _cmocean_colormaps.__all__


# For convenience, also expose the names
names = _cmocean_colormaps.names
