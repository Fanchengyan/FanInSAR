"""Xarray-backed dataset helpers for FanInSAR.

Notes
-----
This module introduces a lightweight :class:`XarrayDataset` class that uses
Xarray, :mod:`rasterio`, and Dask to read gridded rasters. It targets
single-band GeoTIFF inputs with matching grid alignment, providing both eager
and lazy ``boxes_query`` implementations backed by a custom Dask HighLevelGraph.

Examples
--------
Basic usage:

>>> from pathlib import Path
>>> from rasterio.crs import CRS
>>> from faninsar.datasets.xarray_dataset import XarrayDataset
>>> from faninsar.query.bbox import BoundingBox
>>> dataset = XarrayDataset(paths=[Path("tile.tif")])
>>> bbox = BoundingBox(0, 0, 100, 100, crs=CRS.from_epsg(4326))
>>> data = dataset.boxes_query(bbox)
>>> dataset.close()  # Manually close files

Using context manager:

>>> with XarrayDataset(paths=[Path("tile.tif")]) as dataset:
...     data = dataset.boxes_query(bbox)
# Files are automatically closed when exiting the context

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, TypeAlias, cast

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.types import Resolution
from odc.geo.xr import assign_crs
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds
from tqdm import tqdm

from faninsar.datasets.base.geo import GeoDataset
from faninsar.datasets.geobox import GeoBox
from faninsar.logging import setup_logger
from faninsar.query.bbox import BoundingBox

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from rasterio.windows import Window

    from faninsar.query.points import Points
    from faninsar.query.polygons import Polygons
    from faninsar.typing import ResamplingLike

__all__ = ["FileMetadata", "XarrayDataSpec", "XarrayDataset"]


logger = setup_logger(__name__)

# Common dimension names for automatic inference
_COMMON_X_DIMS = ["x", "lon", "longitude", "east", "easting"]
_COMMON_Y_DIMS = ["y", "lat", "latitude", "north", "northing"]


@dataclass(frozen=True)
class XarrayDataSpec:
    """Specification describing which dataset or variable of Xarray to open.

    Parameters
    ----------
    path : Path
        Filesystem path to the container.
    group : str | None, optional
        Target group to open. ``None`` selects the root group.
    variable : str | None, optional
        Data variable to select from the dataset. ``None`` uses first variable.

    """

    path: Path
    """Filesystem path to the container."""

    group: str | None = None
    """Target group to open. ``None`` selects the root group."""

    variable: str | None = None
    """Data variable to select from the dataset. ``None`` uses first variable."""

    def with_path(self, path: Path) -> XarrayDataSpec:
        """Get a new spec with only path updated."""
        return XarrayDataSpec(path=path, group=self.group, variable=self.variable)

    def with_group(self, group: str | None) -> XarrayDataSpec:
        """Get a new spec with only group updated."""
        return XarrayDataSpec(path=self.path, group=group, variable=self.variable)

    def with_variable(self, variable: str | None) -> XarrayDataSpec:
        """Get a new spec with only variable updated."""
        return XarrayDataSpec(path=self.path, group=self.group, variable=variable)


OpenPathLike: TypeAlias = str | Path | XarrayDataSpec


@dataclass(frozen=True)
class FileMetadata:
    """Metadata describing a single raster file.

    Attributes
    ----------
    path : Path
        Location of the raster on disk.
    y_dim : str
        Name of the y dimension in the Xarray representation.
    x_dim : str
        Name of the x dimension in the Xarray representation.
    geobox : GeoBox
        GeoBox describing the raster grid.
    var_name : str
        Name of the data variable within the opened Xarray dataset.
    group : str | None
        Group selection used when opening the dataset. ``None`` selects the
        root group.
    engine : str | None
        Xarray engine to use when opening the dataset.
    dtype : np.dtype
        Data type of the raster samples.
    nodata : float | int | None
        NoData marker associated with the raster.

    """

    path: Path
    var_name: str
    group: str | None
    y_dim: str
    x_dim: str
    geobox: GeoBox
    dtype: np.dtype
    nodata: float | int | None


@dataclass(frozen=True)
class OutputMetadata:
    """Unified geospatial metadata produced by a GeoBox-based :class:`XarrayDataset`.

    Attributes
    ----------
    geobox : GeoBox
        GeoBox describing the unified output grid.
    dtype : np.dtype
        Data type of the output raster.
    nodata : float | int | None
        NoData marker for the output raster.

    """

    geobox: GeoBox
    dtype: np.dtype
    nodata: float | int | None


class XarrayDataset(GeoDataset):
    """Load and query data using Xarray with GeoDataset features.

    This class integrates Xarray-based lazy loading with the GeoDataset interface,
    providing R-tree spatial indexing, multiple query types, and compatibility with
    the FanInSAR dataset ecosystem.

    Parameters
    ----------
    paths : Iterable[OpenPathLike]
        Input raster specifications. All rasters must share CRS, resolution, and
        pixel grid alignment.
    variable : str | None, optional
        Data variable to select from each dataset. When omitted, the first data
        variable is used.
    group : str | None, optional
        Target group to open within hierarchical containers.
        ``None`` selects the root group.
    x_dim : str | None, optional
        Name of the horizontal dimension. When omitted, inferred automatically.
    y_dim : str | None, optional
        Name of the vertical dimension. When omitted, inferred automatically.
    out_crs : CRS | None, optional
        Target CRS for the dataset. When omitted the first raster's CRS is used.
    out_res : tuple[float, float] | None, optional
        Desired resolution. Defaults to the source resolution.
    out_nodata : float | int | None, optional
        Requested NoData marker. Defaults to the source nodata value.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`xarray.open_dataset`.

    Attributes
    ----------
    crs : CRS
        Output coordinate reference system.
    res : tuple[float, float]
        Output resolution.
    dtype : np.dtype
        Output data type.
    nodata : float | int | None
        Output NoData marker.
    bounds : BoundingBox
        Combined bounds of all rasters in ``out_crs``.
    file_count : int
        Number of input rasters.

    Raises
    ------
    ValueError
        If no paths are provided or if rasters do not share CRS/resolution.

    Notes
    -----
    Bounding-box queries are supported for both eager and lazy execution
    strategies. Additional query types (points, polygons) will be added once
    the lazy graph infrastructure is extended.

    File Management
    ---------------
    By default, opened datasets are kept open and cached for performance.
    Files can be closed manually using the :meth:`close` method, or
    automatically using a context manager:

    >>> with XarrayDataset(paths=[path]) as ds:
    ...     result = ds.boxes_query(bbox)
    # Files are automatically closed when exiting the context

    Files are also automatically closed when the object is destroyed.

    """

    def __init__(
        self,
        *,
        paths: Iterable[OpenPathLike],
        variable: str | None = None,
        group: str | None = None,
        x_dim: str | None = None,
        y_dim: str | None = None,
        out_crs: CRS | None = None,
        out_res: float | tuple[float, float] | None = None,
        out_nodata: float | None = None,
        resampling: ResamplingLike = Resampling.nearest,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Initialise the dataset."""
        # Initialize GeoDataset first (sets up R-tree index)
        super().__init__()

        # Store configuration for dimension and variable inference
        # Must be set before calling _normalise_open_spec
        self._group = group
        self._variable = variable
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._resampling = resampling

        if out_res is not None and isinstance(out_res, (float, int)):
            out_res = (out_res, out_res)

        specs = [self._normalise_open_spec(item) for item in paths]
        if not specs:
            msg = "At least one raster path must be supplied."
            logger.error(msg)
            raise ValueError(msg)

        self._open_specs = specs
        self._open_dataset_kwargs = dict(kwargs)

        # Initialize cache for opened datasets
        self._open_datasets: dict[tuple[Path, str | None], xr.Dataset] = {}

        if verbose:
            specs = tqdm(specs, desc="Scanning files", unit=" files")
        file_metadata = [self._parse_single_file(spec) for spec in specs]

        out_geobox, out_dtype, out_nodata_final = _determine_out_geobox(
            file_metadata=file_metadata,
            out_crs=out_crs,
            out_res=out_res,
            out_nodata=out_nodata,
        )

        self._file_meta_list: list[FileMetadata] = file_metadata
        self._out_meta = OutputMetadata(
            geobox=out_geobox, dtype=out_dtype, nodata=out_nodata_final
        )

        # Set GeoDataset properties from OutputMetadata
        self._crs = self._out_meta.geobox.crs
        resolution = self._out_meta.geobox.resolution
        self._res = (abs(resolution.x), abs(resolution.y))
        self._dtype = self._out_meta.dtype
        self._nodata = self._out_meta.nodata
        self._count = len(file_metadata)

        # Populate R-tree index from file metadata
        for idx, meta in enumerate(file_metadata):
            bb = meta.geobox.boundingbox
            bounds = (bb.left, bb.bottom, bb.right, bb.top)
            self.index.insert(idx, bounds, str(meta.path))

        # Set _valid array (all files are valid in XarrayDataset)
        self._valid = np.ones(len(file_metadata), dtype=bool)

    def get_dataset(self, spec: XarrayDataSpec) -> xr.Dataset:
        """Get or open a dataset from cache.

        Parameters
        ----------
        spec : XarrayDataSpec
            Specification describing which dataset to open. The variable
            spec will be ignored.

        Returns
        -------
        xarray.Dataset
            Opened dataset, either from cache or newly opened.

        Notes
        -----
        Datasets are cached by (path, group) to avoid reopening the same file
        multiple times. The cache is managed by the :meth:`close` method.

        """
        cache_key = (spec.path, spec.group)
        if cache_key not in self._open_datasets:
            open_kwargs = self._build_open_kwargs(spec.group)
            self._open_datasets[cache_key] = xr.open_dataset(spec.path, **open_kwargs)
        return self._open_datasets[cache_key]

    def get_data_array(self, spec: XarrayDataSpec) -> xr.DataArray:
        """Get the data variable from the dataset.

        Parameters
        ----------
        spec : XarrayDataSpec
            Specification indicating which xarray data variable within the group
            in the file to open.

        Returns
        -------
        xarray.DataArray
            Selected data variable from the dataset.

        """
        dataset = self.get_dataset(spec)
        return dataset[spec.variable]

    def close(self) -> None:
        """Close all opened datasets.

        Notes
        -----
        This method closes all datasets that have been opened and cached by this
        instance. After calling this method, subsequent operations will reopen
        datasets as needed.

        """
        if hasattr(self, "_open_datasets"):
            for dataset in self._open_datasets.values():
                dataset.close()
            self._open_datasets.clear()

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Exit context manager and close all datasets."""
        self.close()

    def __del__(self) -> None:
        """Ensure datasets are closed when object is destroyed."""
        self.close()

    def _normalise_open_spec(self, item: OpenPathLike) -> XarrayDataSpec:
        """Return a normalised open specification for ``item``.

        Parameters
        ----------
        item : OpenPathLike
            Candidate specification supplied by the caller.

        Returns
        -------
        XarrayDataSpec
            Normalised specification with resolved path, group, and variable.

        """
        if isinstance(item, XarrayDataSpec):
            return item
        return XarrayDataSpec(
            path=Path(item), group=self._group, variable=self._variable
        )

    def _resolve_dimension_names(
        self,
        data_var: xr.DataArray,
        spec: XarrayDataSpec,
    ) -> tuple[str, str]:
        """Return the ``(y_dim, x_dim)`` tuple for ``data_var``.

        Parameters
        ----------
        data_var : xarray.DataArray
            Candidate data variable selected for metadata extraction.
        spec : XarrayDataSpec
            Specification describing how the dataset was opened.

        Returns
        -------
        tuple[Hashable, Hashable]
            Names of the y and x dimensions.

        Raises
        ------
        ValueError
            If ``data_var`` does not expose at least two dimensions or configured
            dimensions are not found.

        Notes
        -----
        Subclasses may override this method when dimension names should be
        enforced explicitly.

        """
        if len(data_var.dims) < 2:
            msg = (
                f"Data variable selected from {spec} does not expose the "
                "two spatial dimensions required for analysis."
            )
            logger.error(msg)
            raise ValueError(msg)

        # Use configured dimensions if provided
        if self._y_dim is not None and self._x_dim is not None:
            # check if configured dimensions are valid
            if self._y_dim not in data_var.dims:
                msg = (
                    f"Configured y dimension '{self._y_dim}' "
                    f"not found in variable '{data_var.name}'."
                )
                logger.error(msg)
                raise ValueError(msg)
            if self._x_dim not in data_var.dims:
                msg = (
                    f"Configured x dimension '{self._x_dim}' "
                    f"not found in variable '{data_var.name}'."
                )
                logger.error(msg)
                raise ValueError(msg)
            return self._y_dim, self._x_dim

        # Automatic inference if not configured
        dims = _infer_default_dimension_names(data_var)
        if dims is not None:
            return dims

        # using custom function from subclass to get spatial dimensions
        return self.get_dims(spec)

    def get_dims(self, spec: XarrayDataSpec) -> tuple[str, str]:
        """Get spatial dimensions from the dataset.

        Parameters
        ----------
        spec : XarrayDataSpec
            Specification describing how the dataset was opened.

        Returns
        -------
        tuple[str, str]
            Tuple of (y_dim, x_dim) if found, otherwise None.

        """
        msg = "'get_dims' should be implemented in subclasses."
        raise NotImplementedError(msg)

    def get_crs(self, dataset: xr.Dataset, spec: XarrayDataSpec) -> CRS | None:
        """Get CRS from dataset using subclass-specific logic.

        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset to extract CRS from.
        spec : XarrayDataSpec
            Specification describing how the dataset was opened.

        Returns
        -------
        CRS | None
            Coordinate reference system if found, otherwise None.

        Notes
        -----
        Subclasses can override this method to implement custom CRS detection
        logic for specific data formats.

        """
        msg = "'get_crs' should be implemented in subclasses."
        raise NotImplementedError(msg)

    def _infer_crs(
        self,
        dataset: xr.Dataset,
        data_var: xr.DataArray,
        spec: XarrayDataSpec,
    ) -> CRS:
        """Infer CRS from dataset using multiple strategies.

        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset to extract CRS from.
        data_var : xarray.DataArray
            Selected data variable.
        spec : XarrayDataSpec
            Specification describing how the dataset was opened.

        Returns
        -------
        CRS
            Inferred coordinate reference system.

        Notes
        -----
        CRS inference follows this priority:
        1. Dataset/Variable crs properties
        2. Subclass-specific logic via get_crs()
        3. Warning and fallback to EPSG:4326

        """
        # Try dataset and variable attributes first
        crs = dataset.odc.crs or data_var.odc.crs
        if crs is not None:
            return crs

        # Try subclass-specific logic
        try:
            subclass_crs = self.get_crs(dataset, spec)
            if subclass_crs is not None:
                msg = f"CRS successfully inferred from subclass logic for {spec}"
                logger.debug(msg)
                return subclass_crs
        except Exception:
            msg = f"Subclass CRS inference failed for {spec}"
            logger.debug(msg)

        # Fallback with warning
        msg = (
            f"Could not infer CRS from {spec} using any strategy, "
            "falling back to default WGS84. Consider overriding the "
            "'get_crs' method in your subclass."
        )
        logger.warning(msg)
        return CRS.from_epsg(4326)

    def boxes_query(
        self,
        bbox: BoundingBox,
    ) -> xr.DataArray:
        """Extract a bounding box from all rasters and stack along a file axis.

        Parameters
        ----------
        bbox : BoundingBox
            Requested spatial extent. It may use any CRS understood by
            :mod:`pyproj`.

        Returns
        -------
        xarray.DataArray
            3-D array shaped ``(file, y, x)`` populated with data for each
            raster. NoData is used where rasters do not overlap the query.

        Raises
        ------
        NotImplementedError
            If ``out_crs`` differs from the dataset CRS.
        ValueError
            If the rasters are not aligned with the query grid.

        """
        # Create query GeoBox using odc-geo's from_bbox
        query_geobox = GeoBox.from_bbox(
            bbox.to_crs(self.crs).to_tuple(),
            crs=self.crs,
            resolution=Resolution(self.res[0], -self.res[1]),
            tight=True,
        )
        data_array = self._load_bbox_data(query_geobox)

        # Use odc-geo to set CRS
        return assign_crs(data_array, self.crs)

    def points_query(self, points: Points) -> xr.DataArray:
        """Extract values at point locations from all rasters.

        Parameters
        ----------
        points : Points
            Point locations to query. Points will be converted to dataset CRS
            if necessary.

        Returns
        -------
        xarray.DataArray
            2-D array shaped ``(file, point)`` with values at each point location.
            NoData is used where points fall outside raster bounds.

        Notes
        -----
        Uses Xarray's selection methods to interpolate values at point locations.

        """
        # Convert points to dataset CRS
        if points.crs is None:
            msg = f"No CRS specified for points, assuming dataset CRS: {self.crs}"
            logger.warning(msg)
            points_crs = points
        elif points.crs != self.crs:
            points_crs = points.to_crs(self.crs)
        else:
            points_crs = points

        stacked_values: list[np.ndarray] = []

        for meta in self._file_meta_list:
            spec = XarrayDataSpec(meta.path, meta.group, meta.var_name)
            data_var = self.get_data_array(spec)

            # Get x, y coordinates of points
            x_coords = points_crs.x
            y_coords = points_crs.y

            # Sample using Xarray's selection
            values = np.full(len(points_crs), self.nodata, dtype=self.dtype)

            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                try:
                    # Use nearest neighbor selection
                    val = data_var.sel(
                        {meta.x_dim: x, meta.y_dim: y}, method="nearest"
                    ).values
                    if np.ndim(val) == 0:
                        values[i] = val
                    else:
                        # Handle extra dimensions by taking first element
                        values[i] = val.flat[0]
                except (KeyError, IndexError):
                    # Point outside bounds
                    values[i] = self.nodata if self.nodata is not None else np.nan

            stacked_values.append(values)

        # Create DataArray
        result = xr.DataArray(
            np.array(stacked_values),
            dims=("file", "point"),
            coords={
                "file": [spec.path for spec in self._open_specs],
                "point": np.arange(len(points_crs)),
                "x": ("point", points_crs.x),
                "y": ("point", points_crs.y),
            },
            attrs={"crs": str(self.crs), "nodata": self.nodata},
        )

        return assign_crs(result, self.crs)

    def polygons_query(self, polygons: Polygons) -> list[xr.DataArray]:
        """Extract data within polygon boundaries from all rasters.

        Parameters
        ----------
        polygons : Polygons
            Polygon geometries to query. Polygons will be converted to dataset CRS
            if necessary.

        Returns
        -------
        list[xarray.DataArray]
            List of DataArrays, one per polygon. Each array is shaped ``(file, y, x)``
            and contains data clipped to the polygon bounds and masked outside the
            polygon boundary.

        Notes
        -----
        Currently clips to polygon bounding box. Precise polygon masking will be
        added in future updates.

        """
        # Convert polygons to dataset CRS
        if polygons.crs is None:
            logger.warning(
                "No CRS specified for polygons, assuming dataset CRS: %s", self.crs
            )
            polygons_crs = polygons
        elif polygons.crs != self.crs:
            polygons_crs = polygons.to_crs(self.crs)
        else:
            polygons_crs = polygons

        results: list[xr.DataArray] = []

        for poly in polygons_crs.geodataframe.geometry:
            # Get bounding box of polygon
            minx, miny, maxx, maxy = poly.bounds
            bbox = BoundingBox(minx, miny, maxx, maxy, crs=self.crs)

            # Query using bbox (will be masked to polygon in future)
            data = self.boxes_query(bbox)

            # TODO: Add polygon masking here
            # For now, just return the bbox-clipped data
            results.append(data)

        return results

    def _load_bbox_data(self, query_geobox: GeoBox) -> xr.DataArray:
        """Produce eager xarray stacks for ``boxes_query``.

        Parameters
        ----------
        query_geobox : GeoBox
            GeoBox describing the query grid.

        Returns
        -------
        xarray.DataArray
            Stacked array shaped ``(file, y, x)``.

        Notes
        -----
        This path performs immediate IO via :mod:`rasterio` window reads.

        """
        stacked_arrays: list[xr.DataArray] = []

        for meta in self._file_meta_list:
            file_geobox = meta.geobox
            intersection = query_geobox.get_overlap(file_geobox)

            if intersection is not None:
                # Check if warping is needed
                # if not aligned, or query bounds is not fully contained in file
                if not file_geobox.is_aligned(query_geobox):
                    logger.debug("Warping needed...")
                    # Warp path: read source data and reproject
                    intersection_src = intersection.to_crs(file_geobox.crs)
                    src_window = file_geobox.get_window_for_bounds(intersection_src)
                    # Read source data
                    src_da = self._read_file_window(meta, src_window)
                    src_da = assign_crs(src_da, file_geobox.crs)

                    target_array = src_da.odc.reproject(
                        how=GeoBox.from_bbox(
                            intersection.to_tuple(),
                            crs=query_geobox.crs,
                            resolution=query_geobox.resolution,
                            tight=True,
                        ),
                        resampling=self._resampling,
                    )
                else:
                    logger.debug("No warp needed...")
                    # No warp needed: direct copy array from file to target
                    intersection_src = intersection.to_crs(file_geobox.crs)
                    src_window = file_geobox.get_window_for_bounds(intersection_src)
                    target_array = self._read_file_window(meta, src_window)
            else:
                msg = "No overlap between query and file. An empty array is returned."
                logger.warning(msg)
                target_array = xr.DataArray(
                    da.full(
                        (1, query_geobox.height, query_geobox.width),
                        self.nodata,
                        dtype=self.dtype,
                    ),
                    dims=("file", self._y_dim, self._x_dim),
                    coords={
                        "file": [meta.path.name],
                        self._y_dim: query_geobox.y,
                        self._x_dim: query_geobox.x,
                    },
                )
            stacked_arrays.append(target_array)

        da_box = xr.concat(stacked_arrays, dim="file")
        da_box = da_box.assign_coords(file=[spec.path for spec in self._open_specs])
        y_dim, x_dim = query_geobox.dims
        if y_dim not in da_box.dims:
            da_box = da_box.rename({self._y_dim: y_dim})
        if x_dim not in da_box.dims:
            da_box = da_box.rename({self._x_dim: x_dim})
        return da_box

    def _parse_single_file(self, spec: XarrayDataSpec) -> FileMetadata:
        """Extract metadata for a single raster.

        Parameters
        ----------
        spec : XarrayDataSpec
            Specification describing how to open the raster.

        Returns
        -------
        FileMetadata
            Metadata describing the raster.

        Raises
        ------
        ValueError
            If the raster cannot be loaded or metadata is inconsistent.

        """
        dataset = self.get_dataset(spec)
        var_name, data_var = _select_data_variable(dataset, spec)

        y_dim, x_dim = self._resolve_dimension_names(data_var, spec)
        height = int(data_var.sizes[y_dim])
        width = int(data_var.sizes[x_dim])

        transform = dataset.odc.transform
        crs = self._infer_crs(dataset, data_var, spec)
        nodata = data_var.odc.nodata
        dtype = np.dtype(data_var.dtype)

        # Create bounds and GeoBox
        bounds_tuple = array_bounds(height, width, transform)

        geobox = GeoBox.from_bbox(
            bounds_tuple,
            crs=crs,
            shape=(height, width),
            tight=True,
        )

        return FileMetadata(
            path=spec.path,
            y_dim=y_dim,
            x_dim=x_dim,
            geobox=geobox,
            var_name=var_name,
            group=spec.group,
            dtype=dtype,
            nodata=nodata,
        )

    def _build_open_kwargs(self, group: str | None) -> dict[str, object]:
        """Construct keyword arguments for :func:`xarray.open_dataset`.

        Parameters
        ----------
        group : str | None
            Target group to open. ``None`` selects the root group.

        Returns
        -------
        dict[str, object]
            Keyword arguments suitable for :func:`xarray.open_dataset`.

        """
        kwargs = dict(self._open_dataset_kwargs)
        if group is not None:
            kwargs["group"] = group
        return kwargs

    def _read_file_window(
        self,
        meta: FileMetadata,
        window: Window,
    ) -> xr.DataArray:
        """Read a window from a raster via Xarray.

        Parameters
        ----------
        meta : FileMetadata
            Metadata describing the raster.
        window : Window
            Window specified as ``(row_off, col_off, height, width)`` in pixel
            coordinates relative to ``meta``.

        Returns
        -------
        xarray.DataArray
            Extracted data block

        """
        y_range, x_range = window.toranges()
        y_slice = slice(*y_range)
        x_slice = slice(*x_range)

        spec = XarrayDataSpec(meta.path, meta.group, meta.var_name)
        data_var = self.get_data_array(spec)
        subset = data_var.isel({meta.y_dim: y_slice, meta.x_dim: x_slice})
        for dim in tuple(subset.dims):
            if dim not in {meta.y_dim, meta.x_dim}:
                subset = subset.isel({dim: 0})
        return subset.squeeze(drop=True)

    def __repr__(self) -> str:
        """Return a string summary of the dataset."""
        return (
            f"{self.__class__.__name__}(file_count={self.file_count}, "
            f"crs={self.crs}, res={self.res}, dtype={self.dtype})"
        )

    def __len__(self) -> int:
        """Return the number of rasters in the dataset."""
        return self.file_count

    def get_profile(self, bbox: BoundingBox | Literal["roi", "bounds"] = "roi") -> dict:
        """Get profile information for the dataset.

        Parameters
        ----------
        bbox : BoundingBox | Literal["roi", "bounds"], optional
            Bounding box to get profile for. Can be:
            - "roi": Use region of interest (same as bounds for XarrayDataset)
            - "bounds": Use full dataset bounds
            - BoundingBox: Use specific bounding box
            Default is "roi".

        Returns
        -------
        dict
            Profile dictionary with transform, width, height, crs, etc.

        """
        # Handle string literals
        if bbox in {"roi", "bounds"} or bbox is None:
            bbox = self.bounds
        elif not isinstance(bbox, BoundingBox):
            msg = f"bbox must be 'roi', 'bounds', or BoundingBox, got {type(bbox)}"
            raise TypeError(msg)

        geobox = GeoBox.from_bbox(
            bbox.to_tuple(),
            crs=self.crs,
            resolution=Resolution(self.res[0], -self.res[1]),
            tight=True,
        )

        return {
            "transform": geobox.transform,
            "width": geobox.width,
            "height": geobox.height,
            "crs": self.crs,
            "dtype": self.dtype,
            "nodata": self.nodata,
            "count": self.file_count,
        }

    @property
    def width(self) -> int:
        """The output width of the dataset."""
        return self._out_meta.geobox.width

    @property
    def height(self) -> int:
        """The output height of the dataset."""
        return self._out_meta.geobox.height

    @property
    def transform(self) -> Affine:
        """The output transform of the dataset."""
        return self._out_meta.geobox.transform

    @property
    def bounds(self) -> BoundingBox:
        """The output bounds of the dataset."""
        bb = self._out_meta.geobox.boundingbox
        return BoundingBox(
            bb.left, bb.bottom, bb.right, bb.top, crs=self._out_meta.geobox.crs
        )

    @property
    def nbytes(self) -> int:
        """The number of bytes of the allocated output array."""
        geobox = self._out_meta.geobox
        return geobox.width * geobox.height * self.dtype.itemsize * self.file_count

    @property
    def file_meta_list(self) -> list[FileMetadata]:
        """A list of file metadata describing the dataset."""
        return self._file_meta_list

    @property
    def out_meta(self) -> OutputMetadata:
        """Metadata describing the output array."""
        return self._out_meta

    @property
    def file_count(self) -> int:
        """The number of files in the dataset."""
        return len(self._file_meta_list)

    @property
    def files(self) -> pd.DataFrame:
        """Build files DataFrame compatible with RasterDataset.

        This property provides a pandas DataFrame interface compatible with
        RasterDataset's internal structure, enabling TimeSeriesDataset and
        PairDataset to work with XarrayDataset-based hierarchical datasets.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - paths: File paths as strings
            - valid: Boolean array indicating valid files (all True for XarrayDataset)
            - file_crs: CRS of each file
            - file_bounds: Bounds tuple of each file
            - file_res: Resolution tuple of each file
            - file_dtype: Data type of each file
            - file_nodata: NoData value of each file

        Notes
        -----
        This DataFrame is cached after first access. It's regenerated if
        file metadata changes.

        """
        if not hasattr(self, "_cached_files_df"):
            paths = [str(spec.path) for spec in self._open_specs]
            data = {
                "paths": paths,
                "valid": self._valid,
                "file_crs": [meta.geobox.crs for meta in self._file_meta_list],
                "file_bounds": [
                    (
                        meta.geobox.boundingbox.left,
                        meta.geobox.boundingbox.bottom,
                        meta.geobox.boundingbox.right,
                        meta.geobox.boundingbox.top,
                    )
                    for meta in self._file_meta_list
                ],
                "file_res": [
                    (abs(meta.geobox.resolution.x), abs(meta.geobox.resolution.y))
                    for meta in self._file_meta_list
                ],
                "file_dtype": [meta.dtype for meta in self._file_meta_list],
                "file_nodata": [meta.nodata for meta in self._file_meta_list],
            }
            self._cached_files_df = pd.DataFrame(data)
        return self._cached_files_df


def _select_data_variable(
    dataset: xr.Dataset,
    spec: XarrayDataSpec,
) -> tuple[str, xr.DataArray]:
    """Return the primary data variable to analyse.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset opened from disk.
    spec : XarrayDataSpec
        Specification describing how the dataset was opened.

    Returns
    -------
    tuple[str, xarray.DataArray]
        Name of the selected data variable and the corresponding array.

    Raises
    ------
    ValueError
        If the dataset does not contain data variables or the specified
        variable is not found.

    Notes
    -----
    Sub-classes may override this method to implement custom variable
    selection logic (for example, to target a specific band by name).

    Variable selection priority:
    1. spec.variable (per-file specification)
    2. self._variable (dataset-level configuration)
    3. First data variable (automatic fallback)

    """
    if not dataset.data_vars:
        msg = f"Dataset {spec} does not expose any data variables."
        logger.error(msg)
        raise ValueError(msg)

    # Priority 1: Use spec-level variable if provided
    target_variable = spec.variable

    if target_variable is not None:
        # wrong variable name
        if target_variable not in dataset.data_vars:
            msg = (
                f"Variable '{target_variable}' "
                f"not found in dataset {spec}. "
                f"Available variables: {list(dataset.data_vars.keys())}"
            )
            logger.error(msg)
            raise ValueError(msg)
        msg = f"Using configured variable '{target_variable}' from dataset {spec}"
        logger.debug(msg)
        return target_variable, dataset[target_variable]

    # Priority 3: Fallback to first variable
    name = str(next(iter(dataset.data_vars)))
    msg = (
        "No variable configured. "
        f"Using first detected variable '{name}' from dataset {spec}."
    )
    logger.warning(msg)
    return name, dataset[name]


def _infer_default_dimension_names(
    data_var: xr.DataArray,
) -> tuple[str, str] | None:
    """Automatically infer the ``(y_dim, x_dim)`` tuple for ``data_var``.

    Parameters
    ----------
    data_var : xarray.DataArray
        Candidate data variable selected for metadata extraction.

    Returns
    -------
    tuple[str, str] | None
        Names of the y and x dimensions.

    Raises
    ------
    ValueError
        If dimension names cannot be inferred.

    Notes
    -----
    Uses common dimension name patterns for spatial data.

    """
    dims = list(data_var.dims)

    # Try to find x and y dimensions using common names
    x_dim = None
    y_dim = None

    for dim in dims:
        dim_lower = str(dim).lower()
        if x_dim is None and any(name in dim_lower for name in _COMMON_X_DIMS):
            x_dim = cast("str", dim)
        elif y_dim is None and any(name in dim_lower for name in _COMMON_Y_DIMS):
            y_dim = cast("str", dim)

    # If found both, return them
    if x_dim is not None and y_dim is not None:
        return y_dim, x_dim

    # can't find x and y dimensions automatically
    return None


def _determine_out_geobox(
    *,
    file_metadata: Sequence[FileMetadata],
    out_crs: CRS | None,
    out_res: tuple[float, float] | None,
    out_nodata: float | None,
) -> tuple[GeoBox, np.dtype, float | None]:
    """Compute the unified output geoinfo for the dataset.

    Parameters
    ----------
    file_metadata : Sequence[FileMetadata]
        Metadata for all rasters.
    out_crs : CRS | None
        User-supplied CRS override.
    out_res : tuple[float, float] | None
        User-supplied resolution override.
    out_nodata : float | int | None
        User-supplied NoData override.

    Returns
    -------
    tuple[GeoBox, np.dtype, float | None]
        Tuple of (unified GeoBox, output dtype, output nodata).

    Notes
    -----
    User-provided values take precedence. If not provided, the first file's
    values are used as the standard. Bounds are computed after converting
    all files to the output CRS.

    """
    # Extract geoboxes from metadata
    geo_boxes = [meta.geobox for meta in file_metadata]

    # Use user input if provided, otherwise use first file as standard
    out_crs = out_crs if out_crs is not None else geo_boxes[0].crs
    final_nodata = out_nodata if out_nodata is not None else file_metadata[0].nodata

    # Calculate output resolution if not provided
    if out_res is None:
        # If output CRS differs from first file's CRS, calculate appropriate resolution
        if out_crs != geo_boxes[0].crs:
            # Reproject to get resolution in new CRS
            reprojected = geo_boxes[0].to_crs(out_crs)
            out_res = (abs(reprojected.resolution.x), abs(reprojected.resolution.y))
        else:
            # Same CRS, use first file's resolution
            res = geo_boxes[0].resolution
            out_res = (abs(res.x), abs(res.y))

    # Convert all bounds to output CRS for unified boundary calculation
    unified_bounds = []
    for geobox in geo_boxes:
        bb = geobox.boundingbox
        bbox = BoundingBox(bb.left, bb.bottom, bb.right, bb.top, crs=geobox.crs)
        if geobox.crs == out_crs:
            # Same CRS, use bounds directly
            unified_bounds.append(bbox)
        else:
            # Different CRS, convert to output CRS
            unified_bounds.append(bbox.to_crs(out_crs))

    # Calculate unified boundary in output CRS
    left = min(bounds.left for bounds in unified_bounds)
    right = max(bounds.right for bounds in unified_bounds)
    bottom = min(bounds.bottom for bounds in unified_bounds)
    top = max(bounds.top for bounds in unified_bounds)

    out_bounds = BoundingBox(left, bottom, right, top, crs=out_crs)

    # Calculate output dtype from all files
    out_dtype = np.result_type(*(meta.dtype for meta in file_metadata))

    # Create GeoBox using odc-geo's from_bbox
    from odc.geo.types import Resolution

    out_geobox = GeoBox.from_bbox(
        (out_bounds.left, out_bounds.bottom, out_bounds.right, out_bounds.top),
        crs=out_crs,
        resolution=Resolution(out_res[0], -out_res[1]),
        tight=True,
    )

    return out_geobox, out_dtype, final_nodata
