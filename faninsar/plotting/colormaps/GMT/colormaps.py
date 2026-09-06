"""GMT Colormap Loader with dynamic loading capability."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..cmap_loader import ColormapLoader  # noqa: TID252

if TYPE_CHECKING:
    from ..enhanced_colormap import EnhancedLinearSegmentedColormap  # noqa: TID252


class GMTColormaps(ColormapLoader):
    """GMT colormap loader with dynamic loading.

    This loader provides access to GMT (Generic Mapping Tools) colormaps
    with dynamic loading capability.
    """

    def __init__(self) -> None:
        """Initialize GMT colormap loader."""
        super().__init__(Path(__file__).parent.absolute())


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
