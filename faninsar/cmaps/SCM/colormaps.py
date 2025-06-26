"""SCM Colormap Loader with dynamic loading capability."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger

from .. import ColormapLoader  # noqa: TID252

if TYPE_CHECKING:
    from ..enhanced_colormap import EnhancedLinearSegmentedColormap  # noqa: TID252

logger = setup_logger(__name__)


class SCMColormaps(ColormapLoader):
    """SCM colormap loader with dynamic loading.

    This loader provides access to Scientific Colour Maps (SCM) colormaps
    with dynamic loading capability.

    Attributes
    ----------
    version : str
        Version of the SCM colormap collection

    """

    def __init__(self) -> None:
        """Initialize SCM colormap loader."""
        super().__init__(Path(__file__).parent.absolute())
        self.version = "8.0.1"
        self._colormap_names = [
            "acton",
            "bam",
            "bamako",
            "bamO",
            "batlow",
            "batlowK",
            "batlowW",
            "berlin",
            "bilbao",
            "broc",
            "brocO",
            "buda",
            "bukavu",
            "cork",
            "corkO",
            "davos",
            "devon",
            "fes",
            "glasgow",
            "grayC",
            "hawaii",
            "imola",
            "lajolla",
            "lapaz",
            "lisbon",
            "lipari",
            "managua",
            "navia",
            "nuuk",
            "oleron",
            "oslo",
            "roma",
            "romaO",
            "tofino",
            "tokyo",
            "turku",
            "vanimo",
            "vik",
            "vikO",
        ]

    @property
    def names(self) -> list[str]:
        """Return list of available SCM colormap names."""
        return self._colormap_names

    def _load_colormap_data(self, name: str) -> np.ndarray:
        """Load SCM colormap data from file.

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
            msg = f"SCM colormap file not found: {cmap_file}"
            logger.error(msg, stacklevel=2)
            raise FileNotFoundError(msg)

        return np.loadtxt(cmap_file)


# Create a global instance
_scm_colormaps = SCMColormaps()

# Export all colormap names for backward compatibility
__all__ = _scm_colormaps.__all__


# Create module-level attributes for backward compatibility
def __getattr__(name: str) -> EnhancedLinearSegmentedColormap:
    """Module-level attribute access for colormaps."""
    return getattr(_scm_colormaps, name)


def __dir__() -> list[str]:
    """Return list of available module attributes."""
    return _scm_colormaps.__all__


# For convenience, also expose the names and version
names = _scm_colormaps.names
version = _scm_colormaps.version
