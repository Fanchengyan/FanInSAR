"""Cmap loader module for dynamic colormap access."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.colors import to_rgb

from faninsar.logging import setup_logger

from .enhanced_colormap import EnhancedLinearSegmentedColormap

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping
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


class _InMemoryColormapLoader(ColormapLoader):
    """Load built-in colormaps from immutable in-memory RGB samples."""

    def __init__(self, colormap_data: Mapping[str, list[str] | np.ndarray]) -> None:
        """Initialize an in-memory loader.

        Parameters
        ----------
        colormap_data : Mapping[str, np.ndarray]
            RGB samples keyed by their public colormap names.

        """
        super().__init__(Path("<built-in-colormaps>"))
        self._colormap_data: dict[str, np.ndarray] = {}
        for name, data in colormap_data.items():
            data_array = np.asarray(data)
            if data_array.ndim == 1:
                rgb_data = np.asarray([to_rgb(color) for color in data_array])
            else:
                rgb_data = np.asarray(data, dtype=float)
            self._colormap_data[name] = rgb_data.copy()
        self._names = sorted(self._colormap_data)

    def _load_colormap_data(self, name: str) -> np.ndarray:
        """Return RGB samples for a built-in colormap."""
        data = self._colormap_data.get(name)
        if data is None:
            msg = f"Built-in colormap not found: {name}"
            logger.error(msg, stacklevel=2)
            raise AttributeError(msg)
        return data

    @staticmethod
    def _create_colormap(
        name: str, data: np.ndarray
    ) -> EnhancedLinearSegmentedColormap:
        """Create a built-in map with the historical 100-sample resolution."""
        return EnhancedLinearSegmentedColormap.from_list(name, data, N=100)

    def _get_colormap(self, name: str) -> EnhancedLinearSegmentedColormap:
        """Load built-in maps while reversing the original control points."""
        if not name.endswith("_r"):
            return super()._get_colormap(name)
        if name in self._cache:
            return self._cache[name]

        base_name = name[:-2]
        if base_name not in self.names:
            msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
            logger.error(msg, stacklevel=2)
            raise AttributeError(msg)
        if base_name not in self._cache:
            self._cache[base_name] = self._create_colormap(
                base_name, self._colormap_data[base_name]
            )
        self._cache[name] = self._create_colormap(
            name, self._colormap_data[base_name][::-1]
        )
        return self._cache[name]


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
            self._register_loader(loader)

    def _register_loader(self, loader: ColormapLoader) -> None:
        """Register a loader's names in the unified lookup table."""
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

    @property
    def names(self) -> list[str]:
        """Return available colormap names without reversed aliases.

        Returns
        -------
        list of str
            Sorted names from all registered collections.

        """
        return sorted({name.removesuffix("_r") for name in self._colormap_map})

    def __dir__(self) -> list[str]:
        """Return list of available attributes including all colormaps."""
        attrs = ["GMT", "SCM", "cmocean", "colorcet", "mintpy", "names"]
        attrs.extend(self._colormap_map.keys())
        return sorted(set(attrs))

    @property
    def __all__(self) -> list[str]:
        """Return list of all available colormap names."""
        return sorted(self._colormap_map.keys())


# Create a global instance for direct access and register project-specific maps.
cmaps = Cmaps()

_WHITE = "0.95"
_BUILTIN_COLORMAPS = {
    "RdGyBu": [
        "#68011f",
        "#bb2832",
        "#e48066",
        "#fbccb4",
        "#ededed",
        "#c2ddec",
        "#6bacd1",
        "#2a71b2",
        "#0d3061",
    ],
    "GnBu_RdPl": ["#8f07ff", "#d5734a", _WHITE, "#0571b0", "#01ef6c"],
    "WtBuPl": [_WHITE, "#0571b0", "#8f07ff", "#d5734a"],
    "WtBuGn": [_WHITE, "#0571b0", "#01ef6c"],
    "WtRdPl": [_WHITE, "#d5734a", "#8f07ff"],
    "WtHeatRed": [_WHITE, "#fff7b3", "#fb9d59", "#aa0526"],
}
_builtin_loader = _InMemoryColormapLoader(_BUILTIN_COLORMAPS)
cmaps._register_loader(_builtin_loader)

names = cmaps.__all__.copy()
