"""Hierarchical container integration built on :mod:`xarray`.

Notes
-----
The refactored :class:`HierarchicalDataset` combines :class:`XarrayDataset` with
:class:`HierarchicalMixin` to provide full GeoDataset features (R-tree indexing,
multiple query types) while supporting hierarchical formats like NetCDF, HDF5,
and Zarr.

This architecture enables hierarchical datasets to be combined with temporal
(TimeSeriesDataset) and pair-based (PairDataset) metadata patterns.

See Also
--------
faninsar.data.datasets.xarray_dataset
    Base implementation with Xarray-backed queries and GeoDataset integration.
faninsar.data.datasets.base.hierarchical
    Mixin providing hierarchical container support.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rasterio.enums import Resampling

from faninsar.data.datasets.base.hierarchical import HierarchicalMixin
from faninsar.data.datasets.base.pair import PairDataset
from faninsar.data.datasets.base.timeseries import TimeSeriesDataset
from faninsar.data.datasets.xarray_dataset import XarrayDataset, XarrayDataSpec
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pyproj.crs import CRS

    from faninsar.typing import ResamplingLike

logger = setup_logger(__name__)


class HierarchicalDataset(XarrayDataset, HierarchicalMixin):
    """Raster access for hierarchical container formats (NetCDF/HDF5/Zarr).

    This class combines XarrayDataset's Xarray-backed queries with hierarchical
    container support, providing full GeoDataset features including R-tree spatial
    indexing, multiple query types (points, bbox, polygons), and compatibility
    with the FanInSAR dataset ecosystem.

    Parameters
    ----------
    paths : Iterable[str | Path] | None, optional
        Container file paths to open. If None, files will be auto-discovered
        using the ``pattern_files`` class attribute.
    root_dir : str | Path, optional
        Root directory for auto-discovery when ``paths`` is None.
    crs : CRS | None, optional
        Desired output CRS override. Defaults to the CRS discovered from source
        rasters.
    res : float | tuple[float, float] | None, optional
        Output resolution override. Scalars are interpreted as square pixels.
    nodata : float | int | None, optional
        NoData override forwarded to :class:`XarrayDataset`.
    resampling : Resampling, optional
        Resampling strategy for lazy reprojection. Only
        :class:`~rasterio.enums.Resampling.nearest` is currently supported.
    verbose : bool, optional
        Whether to log warnings when containers cannot be read.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`xarray.open_dataset`.

    Class Attributes
    ----------------
    group : str
        Default group path used for data access.
    var : str | None
        Variable name within the group.
    x_dim : str | None
        Name of horizontal dimension within the variable.
    y_dim : str | None
        Name of vertical dimension within the variable.
    pattern_files : str
        Glob pattern for discovering container files.

    Raises
    ------
    ValueError
        If no paths are provided or containers cannot be read.
    NotImplementedError
        If unsupported feature toggles are requested.

    See Also
    --------
    XarrayDataset
        Parent class providing Xarray-backed queries and GeoDataset integration.
    HierarchicalMixin
        Mixin providing hierarchical container support.

    Examples
    --------
    Basic usage with auto-discovery:

    >>> class MyNetCDFDataset(HierarchicalDataset):
    ...     group = "/data/measurements"
    ...     var = "temperature"
    ...     pattern_files = "*.nc"
    >>> dataset = MyNetCDFDataset(root_dir="data/")
    >>> bbox = BoundingBox(0, 0, 100, 100, crs=dataset.crs)
    >>> data = dataset.boxes_query(bbox)

    With explicit paths:

    >>> paths = [Path("file1.nc"), Path("file2.nc")]
    >>> dataset = MyNetCDFDataset(paths=paths)
    >>> points = Points([(x1, y1), (x2, y2)])
    >>> values = dataset.points_query(points)

    """

    # Subclass attributes to be overridden
    group: str = ""
    """Default group path used for data access."""

    var: str | None = None
    """variable name within the group"""

    x_dim: str | None = None
    """name of horizontal dimension within the variable"""

    y_dim: str | None = None
    """name of vertical dimension within the variable"""

    pattern_files: str = "*"
    """Glob pattern for discovering container files."""

    def __init__(
        self,
        root_dir: str | Path = "data",
        paths: Iterable[str | Path] | None = None,
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        nodata: float | None = None,
        resampling: ResamplingLike = Resampling.nearest,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the hierarchical dataset."""
        # Discover container paths if not provided
        if paths is None:
            paths = self._discover_containers(root_dir)

        # Convert paths to XarrayDataSpec with hierarchical info
        specs = [
            self._normalize_hierarchical_spec(Path(p), XarrayDataSpec) for p in paths
        ]

        if not specs:
            msg = "No container files found or provided."
            logger.error(msg)
            raise ValueError(msg)

        logger.debug(
            "Initialising %s with %d containers", self.__class__.__name__, len(specs)
        )

        # Initialize XarrayDataset (now inherits from GeoDataset)
        # This will also initialize GeoDataset and populate R-tree index
        super().__init__(
            paths=specs,
            out_crs=crs,
            out_res=res,
            out_nodata=nodata,
            group=self.group,
            variable=self.var,
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            resampling=resampling,
            verbose=verbose,
            **kwargs,
        )


class HierarchicalTimeSeriesDataset(HierarchicalDataset, TimeSeriesDataset):
    """Hierarchical containers with time-series support.

    This class combines hierarchical container handling with time-series metadata,
    allowing NetCDF/HDF5/Zarr files to be queried by date ranges and providing
    temporal dimension management.

    Parameters
    ----------
    paths : Iterable[str | Path] | None, optional
        Container file paths to open. If None, files will be auto-discovered.
    root_dir : str | Path, optional
        Root directory for auto-discovery when ``paths`` is None.
    crs : CRS | None, optional
        Desired output CRS override.
    res : float | tuple[float, float] | None, optional
        Output resolution override.
    nodata : float | int | None, optional
        NoData override.
    resampling : Resampling, optional
        Resampling strategy.
    verbose : bool, optional
        Whether to log warnings.
    **kwargs
        Additional keyword arguments forwarded to xarray.open_dataset.

    Class Attributes
    ----------------
    group : str
        Default group path used for data access.
    var : str | None
        Variable name within the group.
    x_dim : str | None
        Name of horizontal dimension.
    y_dim : str | None
        Name of vertical dimension.
    pattern_files : str
        Glob pattern for discovering container files.

    Notes
    -----
    Files must contain parseable date information in their filenames.
    Implement the ``parse_dates`` classmethod to define custom date parsing logic.

    The file dimension is named "date" instead of "file" for time-series datasets.

    Examples
    --------
    >>> class MyTimeSeriesDataset(HierarchicalTimeSeriesDataset):
    ...     group = "/data/measurements"
    ...     var = "temperature"
    ...     pattern_files = "*.nc"
    ...
    ...     @classmethod
    ...     def parse_dates(cls, paths):
    ...         # Parse dates from filenames
    ...         dates = [parse_date_from_filename(p) for p in paths]
    ...         return Acquisition(dates)
    >>>
    >>> dataset = MyTimeSeriesDataset(root_dir="data/")
    >>> # Query specific dates
    >>> dates = Acquisition(["2020-01-01", "2020-01-15"])
    >>> data = dataset.boxes_query(bbox, dates=dates)

    See Also
    --------
    HierarchicalDataset
        Base hierarchical dataset without temporal metadata.
    TimeSeriesDataset
        Time-series functionality for RasterDataset.

    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the hierarchical time-series dataset."""
        # Initialize both parent classes
        super().__init__(*args, **kwargs)

        # Assign dates from files (from TimeSeriesDataset)
        self._assign_dates_from_paths()

    @property
    def file_dim_name(self) -> str:
        """Dimension name for time-series stacking."""
        return "date"


class HierarchicalPairDataset(HierarchicalDataset, PairDataset):
    """Hierarchical containers with InSAR pair support.

    This class combines hierarchical container handling with interferometric pair
    metadata, allowing NetCDF/HDF5/Zarr files containing InSAR data to be queried
    by specific interferometric pairs.

    Parameters
    ----------
    paths : Iterable[str | Path] | None, optional
        Container file paths to open. If None, files will be auto-discovered.
    root_dir : str | Path, optional
        Root directory for auto-discovery when ``paths`` is None.
    crs : CRS | None, optional
        Desired output CRS override.
    res : float | tuple[float, float] | None, optional
        Output resolution override.
    nodata : float | int | None, optional
        NoData override.
    resampling : Resampling, optional
        Resampling strategy.
    verbose : bool, optional
        Whether to log warnings.
    pair_parser : PairParser | None, optional
        Callable that parses file paths into Pairs. When None, parse_pairs is used.
    **kwargs
        Additional keyword arguments forwarded to xarray.open_dataset.

    Class Attributes
    ----------------
    group : str
        Default group path used for data access.
    var : str | None
        Variable name within the group.
    x_dim : str | None
        Name of horizontal dimension.
    y_dim : str | None
        Name of vertical dimension.
    pattern_files : str
        Glob pattern for discovering container files.

    Notes
    -----
    Files must contain parseable pair information in their filenames.
    Implement the ``parse_pairs`` classmethod to define custom pair parsing logic.

    The file dimension is named "pair" instead of "file" for pair datasets.

    Examples
    --------
    >>> class MyInterferogramDataset(HierarchicalPairDataset):
    ...     group = "/science/grids/data/unwrappedPhase"
    ...     var = "unwrappedPhase"
    ...     pattern_files = "*.nc"
    ...
    ...     @classmethod
    ...     def parse_pairs(cls, paths):
    ...         # Parse pairs from filenames
    ...         pairs = [parse_pair_from_filename(p) for p in paths]
    ...         return Pairs(pairs)
    >>>
    >>> dataset = MyInterferogramDataset(root_dir="data/")
    >>> # Query specific pairs
    >>> pairs = Pairs([("2020-01-01", "2020-01-13")])
    >>> data = dataset.boxes_query(bbox, pairs=pairs)

    See Also
    --------
    HierarchicalDataset
        Base hierarchical dataset without pair metadata.
    PairDataset
        Pair functionality for RasterDataset.

    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the hierarchical pair dataset."""
        # Initialize both parent classes
        super().__init__(*args, **kwargs)

        # Assign pairs from files (from PairDataset)
        # Note: This requires implementing parse_pairs in subclass
        self._assign_pairs_from_paths()

    @property
    def file_dim_name(self) -> str:
        """Dimension name for pair stacking."""
        return "pair"
