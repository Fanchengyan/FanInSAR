"""Hierarchical mixin for datasets with hierarchical containers (NetCDF, HDF5, Zarr)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faninsar.datasets.xarray_dataset import XarrayDataSpec


class HierarchicalMixin:
    """Mixin providing hierarchical container support for datasets.

    This mixin adds functionality for working with hierarchical file formats
    like NetCDF, HDF5, and Zarr that contain groups and variables.

    Class Attributes
    ----------------
    group : str
        Default group path used for data access. Empty string for root group.
    var : str | None
        Variable name within the group. None to auto-detect.
    x_dim : str | None
        Name of horizontal dimension within the variable. None to auto-detect.
    y_dim : str | None
        Name of vertical dimension within the variable. None to auto-detect.
    pattern_files : str
        Glob pattern for discovering container files in a directory.

    Methods
    -------
    _discover_containers(root_dir)
        Discover container files in the specified directory.
    _normalize_hierarchical_spec(path)
        Create XarrayDataSpec with group/variable info.

    Notes
    -----
    This mixin should be used with XarrayDataset or its subclasses to provide
    hierarchical file handling capabilities.

    Examples
    --------
    >>> class MyHierarchicalDataset(HierarchicalDataset):
    ...     group = "/data/measurements"
    ...     var = "temperature"
    ...     x_dim = "longitude"
    ...     y_dim = "latitude"
    ...     pattern_files = "*.nc"

    """

    # Class attributes to be overridden in subclasses
    group: str = ""
    """Default group path used for data access."""

    var: str | None = None
    """Variable name within the group."""

    x_dim: str | None = None
    """Name of horizontal dimension within the variable."""

    y_dim: str | None = None
    """Name of vertical dimension within the variable."""

    pattern_files: str = "*"
    """Glob pattern for discovering container files."""

    def _discover_containers(self, root_dir: str | Path) -> list[Path]:
        """Discover container files in the specified directory.

        Parameters
        ----------
        root_dir : str | Path
            Root directory for discovery.

        Returns
        -------
        list[Path]
            Discovered container paths sorted alphabetically.

        Notes
        -----
        Uses the ``pattern_files`` class attribute to filter files.

        """
        root = Path(root_dir)
        if not root.exists():
            msg = f"Root directory does not exist: {root}"
            raise FileNotFoundError(msg)

        # Use pattern_files for discovery
        discovered = sorted(root.rglob(self.pattern_files))

        # Filter to only files (not directories)
        return [p for p in discovered if p.is_file()]

    def _normalize_hierarchical_spec(
        self, path: Path, XarrayDataSpec_cls: type
    ) -> XarrayDataSpec:
        """Create XarrayDataSpec with group/variable info from class attributes.

        Parameters
        ----------
        path : Path
            Filesystem path to the container.
        XarrayDataSpec_cls : type
            The XarrayDataSpec class to instantiate.

        Returns
        -------
        XarrayDataSpec
            Specification with group and variable information.

        Notes
        -----
        Uses the ``group`` and ``var`` class attributes to populate the spec.

        """
        return XarrayDataSpec_cls(path=path, group=self.group, variable=self.var)
