"""Colormap loaders and the canonical unified :data:`cmaps` registry."""

from .cmap_loader import Cmaps, ColormapLoader, cmaps
from .enhanced_colormap import EnhancedLinearSegmentedColormap

__all__ = [
    "Cmaps",
    "ColormapLoader",
    "EnhancedLinearSegmentedColormap",
    "cmaps",
]
