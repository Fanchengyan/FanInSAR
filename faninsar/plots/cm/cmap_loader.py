"""Cmap loader module for dynamic colormap access."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger

from .enhanced_colormap import EnhancedLinearSegmentedColormap

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from os import PathLike


class ColormapLoader:
    """Base class for dynamic colormap loading.

    This class provides the core functionality for loading LinearSegmentedColormap
    objects on-demand from data files, with caching to avoid repeated file I/O.

    By default, it auto-discovers colormaps by scanning ``data_dir`` for
    subdirectories containing a ``.txt`` file whose stem matches the directory
    name (e.g. ``dem1/dem1.txt``). Subclasses may override :pyattr:`names` and
    :pyfunc:`_load_colormap_data` for different file layouts.
    """

    def __init__(self, data_dir: PathLike) -> None:
        """Initialize the colormap loader.

        Parameters
        ----------
        data_dir : str or Path
            Directory containing colormap data files

        """
        self.data_dir = Path(data_dir)
        self._cache: dict[str, EnhancedLinearSegmentedColormap] = {}
        self._names: list[str] | None = None

    @property
    def names(self) -> list[str]:
        """Return list of available colormap names.

        The default implementation scans *data_dir* for subdirectories that
        contain a ``.txt`` file matching the directory name.
        """
        if self._names is None:
            self._names = sorted(
                d.name
                for d in self.data_dir.iterdir()
                if d.is_dir()
                and not d.name.startswith("_")
                and (d / f"{d.name}.txt").exists()
            )
        return self._names

    def _load_colormap_data(self, name: str) -> np.ndarray:
        """Load colormap data from file.

        The default implementation reads ``<data_dir>/<name>/<name>.txt``
        via :pyfunc:`numpy.loadtxt`.

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
            msg = f"Colormap file not found: {cmap_file}"
            logger.error(msg, stacklevel=2)
            raise FileNotFoundError(msg)

        return np.loadtxt(cmap_file)

    @staticmethod
    def _create_colormap(
        name: str, data: np.ndarray
    ) -> EnhancedLinearSegmentedColormap:
        """Create an EnhancedLinearSegmentedColormap from data.

        Parameters
        ----------
        name : str
            Name of the colormap
        data : np.ndarray
            Colormap data array

        Returns
        -------
        EnhancedLinearSegmentedColormap
            Enhanced LinearSegmentedColormap object with additional tools

        """
        return EnhancedLinearSegmentedColormap.from_list(name, data)

    def _get_colormap(self, name: str) -> EnhancedLinearSegmentedColormap:
        """Get a colormap, loading it if necessary.

        Parameters
        ----------
        name : str
            Name of the colormap

        Returns
        -------
        EnhancedLinearSegmentedColormap
            Enhanced LinearSegmentedColormap object with additional tools

        """
        # Check if it's a reversed colormap
        is_reversed = name.endswith("_r")
        base_name = name[:-2] if is_reversed else name

        # Check if base colormap is available
        if base_name not in self.names:
            msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
            logger.error(msg, stacklevel=2)
            raise AttributeError(msg)

        # Check cache first
        if name in self._cache:
            return self._cache[name]

        # Load base colormap if not in cache
        if base_name not in self._cache:
            data = self._load_colormap_data(base_name)
            self._cache[base_name] = self._create_colormap(base_name, data)

        # Create reversed version if needed
        if is_reversed and name not in self._cache:
            reversed_data = self._cache[base_name].to_rgb_array()[::-1]
            self._cache[name] = self._create_colormap(name, reversed_data)

        return self._cache[name]

    def __getattr__(self, name: str) -> EnhancedLinearSegmentedColormap:
        """Dynamic attribute access for colormaps.

        Parameters
        ----------
        name : str
            Name of the colormap

        Returns
        -------
        EnhancedLinearSegmentedColormap
            Enhanced LinearSegmentedColormap object with additional tools

        Raises
        ------
        AttributeError
            If the attribute name starts or ends with underscores (special attributes)
            or if the colormap name is not found

        """
        # Handle special attributes (those starting or ending with underscores)
        # by raising AttributeError immediately to avoid treating them as colormap names
        if name.startswith("_") or name.endswith("_"):
            msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
            raise AttributeError(msg)

        return self._get_colormap(name)

    def __dir__(self) -> list[str]:
        """Return list of available attributes including colormaps."""
        attrs = list(self.__dict__.keys())
        # Add class attributes
        attrs.extend(["names", "_cache", "_names", "data_dir"])
        # Add all colormap names
        attrs.extend(self.names)
        # Add reversed versions
        attrs.extend([f"{name}_r" for name in self.names])
        return sorted(set(attrs))

    @property
    def __all__(self) -> list[str]:
        """Return list of all available colormap names including reversed versions."""
        names = self.names.copy()
        names.extend([f"{name}_r" for name in self.names])
        return names


class Cmaps:
    """Unified colormap class providing access to all colormap collections.

    This class aggregates all colormap loaders and provides a single interface
    to access colormaps from GMT, SCM, cmocean, colorcet, and mintpy collections.

    Attributes
    ----------
    GMT : ColormapLoader
        GMT colormap collection
    SCM : ColormapLoader
        SCM colormap collection
    cmocean : ColormapLoader
        cmocean colormap collection
    colorcet : ColormapLoader
        colorcet colormap collection
    mintpy : ColormapLoader
        mintpy colormap collection

    """

    def __init__(self) -> None:
        """Initialize the unified colormap loader."""
        # Import the individual colormap classes
        from .cmocean.colormaps import _cmocean_colormaps
        from .colorcet.colormaps import _colorcet_colormaps
        from .GMT.colormaps import _gmt_colormaps
        from .mintpy.colormaps import _mintpy_colormaps
        from .SCM.colormaps import _scm_colormaps

        self._loaders: dict[str, ColormapLoader] = {
            "GMT": _gmt_colormaps,
            "SCM": _scm_colormaps,
            "cmocean": _cmocean_colormaps,
            "colorcet": _colorcet_colormaps,
            "mintpy": _mintpy_colormaps,
        }

        # Create a mapping of colormap names to their loaders
        self._colormap_map: dict[str, ColormapLoader] = {}
        for loader in self._loaders.values():
            for cmap_name in loader.names:
                self._colormap_map[cmap_name] = loader
                self._colormap_map[f"{cmap_name}_r"] = loader

    @property
    def GMT(self) -> ColormapLoader:  # noqa: N802
        """Access GMT colormaps.

        Returns
        -------
        ColormapLoader
            GMT colormap loader instance

        """
        return self._loaders["GMT"]

    @property
    def SCM(self) -> ColormapLoader:  # noqa: N802
        """Access SCM colormaps.

        Returns
        -------
        ColormapLoader
            SCM colormap loader instance

        """
        return self._loaders["SCM"]

    @property
    def cmocean(self) -> ColormapLoader:
        """Access cmocean colormaps.

        Returns
        -------
        ColormapLoader
            cmocean colormap loader instance

        """
        return self._loaders["cmocean"]

    @property
    def colorcet(self) -> ColormapLoader:
        """Access colorcet colormaps.

        Returns
        -------
        ColormapLoader
            colorcet colormap loader instance

        """
        return self._loaders["colorcet"]

    @property
    def mintpy(self) -> ColormapLoader:
        """Access mintpy colormaps.

        Returns
        -------
        ColormapLoader
            mintpy colormap loader instance

        """
        return self._loaders["mintpy"]

    def __getattr__(self, name: str) -> EnhancedLinearSegmentedColormap:
        """Dynamic attribute access for colormaps from any collection.

        Parameters
        ----------
        name : str
            Name of the colormap

        Returns
        -------
        EnhancedLinearSegmentedColormap
            Enhanced LinearSegmentedColormap object with additional tools

        """
        if name.startswith("_") or name.endswith("_"):
            msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
            raise AttributeError(msg)

        # Check if it's a reversed colormap
        is_reversed = name.endswith("_r")
        base_name = name[:-2] if is_reversed else name

        # Find the appropriate loader
        loader = self._colormap_map.get(base_name)
        if loader is None:
            msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
            raise AttributeError(msg)
        return getattr(loader, name)

    def __dir__(self) -> list[str]:
        """Return list of available attributes including all colormaps."""
        attrs = ["GMT", "SCM", "cmocean", "colorcet", "mintpy"]
        attrs.extend(self._colormap_map.keys())
        return sorted(set(attrs))

    @property
    def __all__(self) -> list[str]:
        """Return list of all available colormap names."""
        return sorted(self._colormap_map.keys())


# Create a global instance for direct access
cmaps = Cmaps()


# Define all custom colormaps immediately for guaranteed availability
white = "0.95"

# RdGyBu colormap
colors = [
    "#68011f",
    "#bb2832",
    "#e48066",
    "#fbccb4",
    "#ededed",
    "#c2ddec",
    "#6bacd1",
    "#2a71b2",
    "#0d3061",
]
RdGyBu = EnhancedLinearSegmentedColormap.from_list("RdGrBu", colors, N=100)
RdGyBu_r = EnhancedLinearSegmentedColormap.from_list("RdGrBu_r", colors[::-1], N=100)

# GnBu_RdPl colormap
colors = ["#8f07ff", "#d5734a", white, "#0571b0", "#01ef6c"]
GnBu_RdPl = EnhancedLinearSegmentedColormap.from_list("GnBu_RdPl", colors, N=100)
GnBu_RdPl_r = EnhancedLinearSegmentedColormap.from_list(
    "GnBu_RdPl_r", colors[::-1], N=100
)

# WtBuPl colormap
colors = [white, "#0571b0", "#8f07ff", "#d5734a"]
WtBuPl = EnhancedLinearSegmentedColormap.from_list("WtBuPl", colors, N=100)
WtBuPl_r = EnhancedLinearSegmentedColormap.from_list("WtBuPl_r", colors[::-1], N=100)

# WtBuGn colormap
colors = [white, "#0571b0", "#01ef6c"]
WtBuGn = EnhancedLinearSegmentedColormap.from_list("WtBuGn", colors, N=100)
WtBuGn_r = EnhancedLinearSegmentedColormap.from_list("WtBuGn_r", colors[::-1], N=100)

# WtRdPl colormap
colors = [white, "#d5734a", "#8f07ff"]
WtRdPl = EnhancedLinearSegmentedColormap.from_list("WtRdPl", colors, N=100)
WtRdPl_r = EnhancedLinearSegmentedColormap.from_list("WtRdPl_r", colors[::-1], N=100)

# WtHeatRed colormap
colors = [white, "#fff7b3", "#fb9d59", "#aa0526"]
WtHeatRed = EnhancedLinearSegmentedColormap.from_list("WtHeatRed", colors, N=100)
WtHeatRed_r = EnhancedLinearSegmentedColormap.from_list(
    "WtHeatRed_r", colors[::-1], N=100
)

names = cmaps.__all__.copy()
