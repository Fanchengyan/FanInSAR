"""SCM Colormap Loader with dynamic loading capability."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..cmap_loader import ColormapLoader  # noqa: TID252

if TYPE_CHECKING:
    from ..enhanced_colormap import EnhancedLinearSegmentedColormap  # noqa: TID252


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
