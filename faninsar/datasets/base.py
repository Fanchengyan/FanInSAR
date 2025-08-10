"""Base classes for all :mod:`faninsar` datasets.

The base class RasterDataset in this script is modified from the torchgeo package.
"""

from __future__ import annotations

import contextlib
import functools
import re
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Literal, cast, overload

import numpy as np
import pandas as pd
import pyproj
import rasterio
import rioxarray  # noqa: F401
import shapely
import xarray as xr
from rasterio import features, fill, plot
from rasterio import mask as rio_mask
from rasterio.crs import CRS
from rasterio.dtypes import dtype_ranges, get_minimum_dtype
from rasterio.enums import Resampling
from rasterio.transform import rowcol as tf_rowcol
from rasterio.transform import xy as tf_xy
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.warp import transform as warp_transform
from rtree.index import Index, Property
from shapely import ops
from tqdm import tqdm
from typing_extensions import Self

from faninsar._core import geo_tools
from faninsar._core.geo_tools import Profile, array2kml, array2kmz, geoinfo_from_latlon
from faninsar.logging import setup_logger
from faninsar.query import (
    BBoxesResult,
    BoundingBox,
    GeoQuery,
    Points,
    PointsResult,
    Polygons,
    PolygonsResult,
    QueryResult,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike

    from matplotlib.axes import Axes
    from rasterio.io import DatasetReader
    from rasterio.warp import Affine

    from faninsar._core.sar.acquisition import Acquisition
    from faninsar._core.sar.pairs import Pairs


logger = setup_logger(__name__)

try:
    import dask
    from dask.delayed import delayed

    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    msg = "dask is not installed"

    # Create dummy functions for type checking
    class _DummyDask:
        @staticmethod
        def compute(*args, **kwargs) -> None:  # noqa: ARG004
            """Compute of dask is not installed dummy function for type checking."""
            logger.error(msg)  # noqa: TRY400
            raise ImportError(msg)

    class _DummyDelayed:
        def __call__(self, func) -> None:  # noqa: ANN001, ARG002
            """Delayed of dask is not installed dummy function for type checking."""
            raise ImportError(msg)

    dask = _DummyDask()
    delayed = _DummyDelayed()

__all__ = (
    "GeoDataset",
    "PairDataset",
    "RasterDataset",
    "TimeSeriesDataset",
)

logger = setup_logger(__name__)

lat_names = ["latitude", "lat", "y"]
lon_names = ["longitude", "lon", "x", "long", "lng"]


class GeoDataset(ABC):
    """Abstract base class for all :mod:`faninsar` datasets.

    This class is used to represent a geospatial dataset and provides methods to
    index the dataset and retrieve information about the dataset, such as CRS,
    resolution, data type, no data value, and a bounds.
    """

    # following attributes should be set by the subclass
    _crs: CRS | None = None
    _res: tuple[float, float] = (0.0, 0.0)
    _dtype: np.dtype | None = None
    _count: int = 0
    _roi: BoundingBox | None = None
    _nodata: Any = None
    _valid: np.ndarray

    def __init__(self) -> None:
        """Initialize a new GeoDataset instance."""
        self.index = Index(interleaved=True, properties=Property(dimension=2))

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return f"""\
{self.__class__.__name__} Dataset
    bbox: {self.bounds}
    file count: {len(self)}"""

    def __str__(self) -> str:
        """Return a string representation of the dataset."""
        return self.__repr__()

    def __len__(self) -> int:
        """Return the number of files in the dataset.

        Returns
        -------
            length of the dataset

        """
        return len(self.index)

    def __getstate__(
        self,
    ) -> tuple[dict[str, Any], list[tuple[Any, Any, Any]]]:
        """Define how instances are pickled.

        Returns
        -------
            the state necessary to unpickle the instance

        """
        objects = self.index.intersection(self.index.bounds, objects=True)
        tuples = [(item.id, item.bounds, item.object) for item in objects]
        return self.__dict__, tuples

    def __setstate__(
        self,
        state: tuple[
            dict[Any, Any],
            list[tuple[int, tuple[float, float, float, float, float, float], str]],
        ],
    ) -> None:
        """Define how to unpickle an instance.

        Args:
        ----
            state: the state of the instance when it was pickled

        """
        attrs, tuples = state
        self.__dict__.update(attrs)
        for item in tuples:
            self.index.insert(*item)

    @overload
    def _ensure_query_crs(self, query: BoundingBox) -> BoundingBox: ...

    @overload
    def _ensure_query_crs(self, query: Points) -> Points: ...

    @overload
    def _ensure_query_crs(self, query: Polygons) -> Polygons: ...

    def _ensure_query_crs(
        self,
        query: Points | BoundingBox | Polygons,
    ) -> Points | BoundingBox | Polygons:
        """Ensure that the query has the same CRS as the dataset."""
        if query.crs is None:
            warnings.warn(
                f"No CRS is specified for the {query}, assuming they are in the"
                f" same CRS as the dataset ({self.crs}).",
                stacklevel=2,
            )
        elif query.crs != self.crs:
            query = query.to_crs(self.crs)
        return query

    @property
    def crs(self) -> CRS | None:
        """Coordinate reference system (:term:`CRS`) of the dataset.

        Returns
        -------
            The coordinate reference system (:term:`CRS`).

        """
        return self._crs

    @crs.setter
    def crs(self, new_crs: CRS | str) -> None:
        """Change the coordinate reference system :term:`(CRS)` of a GeoDataset.

        If ``new_crs == self.crs``, does nothing, otherwise updates the R-tree index.

        Parameters
        ----------
        new_crs: CRS or str
            New coordinate reference system :term:`(CRS)`. It can be a CRS object
            or a string, which will be parsed to a CRS object. The string can be
            in any format supported by :meth:`pyproj.crs.CRS.from_user_input`.

        """
        if not isinstance(new_crs, CRS):
            new_crs = CRS.from_user_input(new_crs)
        if new_crs == self.crs:
            return

        if self.crs is not None and len(self) > 0:
            # update the resolution
            profile = self.get_profile("bounds")
            tf, *_ = calculate_default_transform(
                self.crs,
                new_crs,
                profile["width"],
                profile["height"],
                self.bounds[0],
                self.bounds[1],
                self.bounds[2],
                self.bounds[3],
            )
            new_res = (abs(float(tf.a)), abs(float(tf.e)))
            if new_res[0] != self.res[0] or new_res[1] != self.res[1]:
                msg = (
                    "the resolution of the dataset has been changed "
                    f"from {self.res} to {new_res}."
                )
                logger.warning(msg)
                self.res = new_res

            # reproject the index
            new_index = Index(interleaved=True, properties=Property(dimension=2))
            project = pyproj.Transformer.from_crs(
                pyproj.CRS(str(self.crs)),
                pyproj.CRS(str(new_crs)),
                always_xy=True,
            ).transform
            for hit in self.index.intersection(self.index.bounds, objects=True):
                old_xmin, old_xmax, old_ymin, old_ymax = hit.bounds
                old_box = shapely.geometry.box(old_xmin, old_ymin, old_xmax, old_ymax)
                new_box = ops.transform(project, old_box)
                new_bounds = tuple(new_box.bounds)
                new_index.insert(hit.id, new_bounds, hit.object)

            self.index = new_index

        self._crs = new_crs

    @property
    def same_crs(self) -> bool:
        """Whether all files in the dataset have the same CRS with the desired CRS."""
        return self._same_crs

    @property
    def res(self) -> tuple[float, float]:
        """Return the resolution of the dataset.

        Returns
        -------
        res: tuple of floats
            resolution of the dataset in x and y directions.

        """
        return self._res

    @res.setter
    def res(self, new_res: float | tuple[float, float]) -> None:
        """Set the resolution of the dataset.

        Parameters
        ----------
        new_res : float or tuple of floats (x_res, y_res)
            resolution of the dataset . If a float is given, the same resolution
            will be used in both x and y directions.

        """
        if isinstance(new_res, (int, float, np.integer, np.floating)):
            new_res = (float(new_res), float(new_res))
        if len(new_res) != 2:
            msg = f"Resolution must be a float or a tuple of length 2, got {new_res}"
            raise ValueError(
                msg,
            )
        if not all(isinstance(i, float) for i in new_res):
            try:
                new_res = (float(new_res[0]), float(new_res[1]))
            except TypeError as e:
                msg = "Resolution must be a float or a tuple of floats"
                raise TypeError(msg) from e
        self._res = new_res

    @property
    def roi(self) -> BoundingBox:
        """Return the region of interest of the dataset.

        Returns
        -------
        roi: BoundingBox object
            region of interest of the dataset. If None, the bounds of
            entire dataset will be used.

        """
        if self._roi:
            return self._roi
        return self.bounds

    @roi.setter
    def roi(self, new_roi: BoundingBox) -> None:
        """Set the region of interest of the dataset.

        Parameters
        ----------
        new_roi : BoundingBox object, optional
            region of interest of the dataset in the CRS of the dataset. If the
            crs of the new_roi is different from the crs of the dataset, the new_roi
            will be reprojected to the crs of the dataset. If None, the crs of the
            dataset will be used.

        """
        new_roi = self._check_roi(new_roi)

        self._roi = new_roi

    def _check_roi(self, roi: BoundingBox | None) -> BoundingBox:
        """Check the roi and return a valid roi.

        Parameters
        ----------
        roi : BoundingBox object, optional
            region of interest of the dataset in the CRS of the dataset. If the
            crs of the new_roi is different from the crs of the dataset, the new_roi
            will be reprojected to the crs of the dataset. If None, the crs of the
            dataset will be used.

        Returns
        -------
        roi: BoundingBox object
            region of interest of the dataset. If None, the bounds of
            entire dataset will be used.

        """
        if roi is None:
            return self.roi
        if not isinstance(roi, BoundingBox):
            msg = f"roi must be a BoundingBox object, got {type(roi)} instead."
            raise TypeError(msg)
        if roi.crs != self.crs:
            if roi.crs is None:
                roi = BoundingBox(*roi, crs=self.crs)
            else:
                roi = roi.to_crs(self.crs)
        return roi

    @property
    def dtype(self) -> np.dtype | None:
        """Data type of the dataset.

        Returns
        -------
        dtype: numpy.dtype object or None
            data type of the dataset

        """
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: np.dtype) -> None:
        """Set the data type of the dataset.

        Parameters
        ----------
        new_dtype : numpy.dtype
            data type of the dataset

        """
        self._dtype = new_dtype

    @property
    def nodata(self) -> float | None:
        """No data value of the dataset.

        Returns
        -------
        nodata: float or int
            no data value of the dataset

        """
        return self._nodata

    @nodata.setter
    def nodata(self, new_nodata: float) -> None:
        """Set the no data value of the dataset.

        Parameters
        ----------
        new_nodata : float or int
            no data value of the dataset

        """
        self._nodata = new_nodata

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the dataset.

        Returns
        -------
        shape: tuple of ints
            shape of the dataset in (height, width) format

        """
        profile = self.get_profile("bounds")
        return profile["height"], profile["width"]

    @property
    def valid(self) -> np.ndarray:
        """Return a boolean array indicating which files are valid.

        Returns
        -------
        valid: numpy.ndarray
            boolean array indicating which files are valid. True means the file
            is valid and can be read by rasterio, False means the file is invalid.

        """
        return self._valid

    @property
    def bounds(self) -> BoundingBox:
        """Bounds of the overall dataset.

        It is the union of all the files in the dataset.

        Returns
        -------
        bounds: BoundingBox object
            (minx, right, bottom, top) of the dataset

        """
        return BoundingBox(*self.index.bounds, crs=self.crs)

    def _ensure_bbox(
        self,
        bbox: BoundingBox | Literal["roi", "bounds"] = "roi",
    ) -> BoundingBox:
        """Return the bounds of the dataset for the given bounding box type.

        Parameters
        ----------
        bbox : BoundingBox | Literal["roi", "bounds"], optional
            the bounding box used to calculate the bounds of the dataset.
            Default is 'roi'.

        Returns
        -------
        bounds: BoundingBox
            bounds of the dataset for the given bounding box type.

        Raises
        ------
        TypeError: if bbox is not one of {'bounds', 'roi'} or a BoundingBox

        """
        if bbox == "bounds":
            return self.bounds
        if bbox == "roi":
            return self.roi
        if isinstance(bbox, BoundingBox):
            return self._check_roi(bbox)
        msg = f"bbox must be one of ['bounds', 'roi'] or a BoundingBox, but got {bbox}"
        raise TypeError(msg)

    def get_profile(
        self, bbox: BoundingBox | Literal["roi", "bounds"] = "roi"
    ) -> Profile | None:
        """Get profile information of the dataset for the given bounding box type.

        The profile information includes the width, height, transform,
        count, data type, no data value, and CRS of the dataset.

        Parameters
        ----------
        bbox : BoundingBox | Literal["roi", "bounds"], optional
            the bounding box used to calculate the ``width``, ``height``
            and ``transform`` of the dataset for the profile. Default is
            'roi'.

        Returns
        -------
        profile: Profile object or None
            profile of the dataset for the given bounding box type.

        """
        msg = "get_profile method must be implemented in subclass"
        raise NotImplementedError(msg)


class RasterDataset(GeoDataset):
    """A base class for raster datasets.

    Examples
    --------
    >>> from pathlib import Path
    >>> from faninsar.datasets import RasterDataset
    >>> from faninsar.query import BoundingBox, GeoQuery, Points,
    >>> home_dir = Path("./work/data")
    >>> files = list(home_dir.rglob("*unw_phase.tif"))

    initialize a RasterDataset and GeoQuery object

    >>> ds = RasterDataset(paths=files)
    >>> points = Points(
        [(490357, 4283413),
        (491048, 4283411),
        (490317, 4284829)]
        )
    >>> query = GeoQuery(points=points, boxes=[ds.bounds, ds.bounds])

    use the GeoQuery object to index the RasterDataset

    >>> sample = ds[query]

    output the samples shapes:

    >>> print("boxes result shape:", sample.boxes.data.shape)
    boxes result shape: (2, 7, 68, 80)

    >>> print("points result shape:", sample.points.data.shape)
    points result shape: (7, 3)

    of course, you can also use the BoundingBox or Points directly to index the
    RasterDataset. Those two types will be automatically converted to GeoQuery
    object.

    >>> sample = ds[points]
    >>> sample
    {'query': GeoQuery(
        boxes=None
        points=Points(count=3)
    ),
    'boxes': None,
    'points': array([...], dtype=float32)}

    >>> sample = ds[ds.bounds]
    query': GeoQuery(
        boxes=[1 BoundingBox]
        points=None
    ),
    'boxes': array([...], dtype=float32),
    'points': None}

    """

    #: Glob expression used to search for files.
    #:
    #: This expression should be specific enough that it will not pick up files from
    #: other datasets. It should not include a file extension, as the dataset may be in
    #: a different file format than what it was originally downloaded as.
    pattern = "*"

    #: When :attr:`~RasterDataset.separate_files` is True, the following additional
    #: groups are searched for to find other files:
    #:
    #: * ``band``: replaced with requested band name
    filename_regex = ".*"

    #: Date format string used to parse date from filename.
    #:
    #: Not used if :attr:`filename_regex` does not contain a ``date`` group.
    date_format = "%Y%m%d"

    #: Names of all available bands in the dataset
    all_bands: ClassVar[list[str]] = []

    #: Names of RGB bands in the dataset, used for plotting
    rgb_bands: ClassVar[list[str]] = []

    #: Color map for the dataset, used for plotting
    cmap: ClassVar[dict[int, tuple[int, int, int, int]]] = {}

    _same_crs: bool

    def __init__(
        self,
        root_dir: str | PathLike = "data",
        paths: Iterable[str | PathLike] | None = None,
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        dtype: np.dtype | None = None,
        nodata: float | None = None,
        roi: BoundingBox | None = None,
        bands: Iterable[str] | None = None,
        cache: bool = True,
        resampling: Resampling = Resampling.nearest,
        fill_nodata: bool = False,
        verbose: bool = True,
        ds_name: str = "",
        parallel_loading: bool = False,
    ) -> None:
        """Initialize a new raster dataset instance.

        Parameters
        ----------
        root_dir : str or PathLike
            root_dir directory where dataset can be found.
        paths : list of str, optional
            list of file paths to use instead of searching for files in ``root_dir``.
            If None, files will be searched for in ``root_dir``.
        crs : CRS, optional
            the output term:`coordinate reference system (CRS)` of the dataset.
            If None, the CRS of the first file found will be used.
        res : float, optional
            resolution of the output dataset in units of CRS. If None, the resolution
            of the first file found will be used.
        dtype : numpy.dtype, optional
            data type of the output dataset. If None, the data type of the first file
            found will be used.
        nodata : float or int, optional
            no data value of the dataset. If None, the no data value of the first
            file found will be used. This parameter is useful when the no data value
            is not stored in the file.
        roi : BoundingBox, optional
            region of interest to load from the dataset. If None, the union of all files
            bounds in the dataset will be used.
        bands : list of str, optional
            names of bands to return (defaults to all bands)
        cache : bool, optional
            if True, cache file handle to speed up repeated sampling
        resampling : Resampling, optional
            Resampling algorithm used when reading input files.
            Default: `Resampling.nearest`.
        fill_nodata : bool, optional
            Whether to fill holes in the queried data by interpolating them using
            inverse distance weighting method provided by the
            :func:`rasterio.fill.fillnodata`. Default: False.

            .. note::
                This parameter is only used when sampling data using bounding
                boxes or polygons queries, and will not work for points queries.

        verbose : bool, optional
            if True, print verbose output, default: True
        ds_name : str, optional
            name of the dataset. used for printing verbose output, default: ""
        parallel_loading : bool, optional
            Enable parallelized operations (via `dask` when available) both during
            dataset initialization (file discovery and metadata extraction) and
            during queries (lazy/parallel computation). Default: False.

        Raises
        ------
            FileNotFoundError: if no files are found in ``root_dir``

        Examples
        --------
        Following examples show how to use parameters to warp the dataset upon loading.

        .. ref-gallery::
            :tooltip:

            examples/warp/align
            examples/warp/reproject
            examples/warp/resample

        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.bands = bands or self.all_bands
        self.cache = cache
        self.resampling = resampling
        self.fill_nodata = fill_nodata
        self.verbose = verbose
        self.ds_name = ds_name
        self.parallel_loading = bool(parallel_loading)

        if paths is None:
            paths = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for file_path in sorted(self.root_dir.rglob(self.pattern)):
                match = re.match(filename_regex, file_path.name)
                if match is not None:
                    paths.append(str(file_path))
        else:
            paths = [str(p) for p in paths]

        # Scan files and extract metadata
        if self.parallel_loading and HAS_DASK:
            files_df = self._scan_files_with_dask(paths, crs)
        else:
            files_df = self._scan_files_sequential(paths, crs)

        # Store files information
        self._files = files_df

        # Determine final attributes based on user parameters and file metadata
        final_crs, final_res, final_dtype, final_nodata = (
            self._determine_final_attributes(files_df, crs, res, dtype, nodata)
        )

        # Set final attributes
        self.crs = final_crs
        self.res = final_res
        self.dtype = final_dtype
        self.nodata = final_nodata
        self.count = self._count
        self.roi = roi

    @staticmethod
    def _extract_single_file_metadata(
        file_path: str, target_crs: CRS | None = None
    ) -> dict:
        """Extract metadata from a single file.

        Parameters
        ----------
        file_path : str
            path to the file to process
        target_crs : CRS, optional
            Target CRS for coordinate transformation

        Returns
        -------
        dict
            Dictionary containing file metadata with keys:
            - path: file path
            - valid: whether file is readable
            - file_crs: original CRS of the file
            - file_bounds: original bounds of the file
            - file_res: original resolution of the file
            - file_dtype: original data type of the file
            - file_nodata: original nodata value of the file
            - crs: unified CRS (target_crs or file_crs)
            - bounds: bounds in unified CRS
            - res: resolution in unified CRS
            - colormap: colormap if available

        """
        try:
            with rasterio.open(file_path) as src:
                # Extract original file metadata
                file_crs = src.crs
                file_bounds = src.bounds
                file_res = src.res
                file_dtype = src.dtypes[0]
                file_nodata = src.nodata
                file_bounds = src.bounds

                # Extract colormap if available
                colormap = None
                with contextlib.suppress(ValueError):
                    colormap = src.colormap(1)

                # Calculate transformed metadata if target CRS is specified
                transformed_bounds = file_bounds
                transformed_res = file_res
                if target_crs and target_crs != file_crs:
                    with WarpedVRT(src, crs=target_crs) as vrt:
                        transformed_bounds = tuple(vrt.bounds)
                        transformed_res = vrt.res

                return {
                    "paths": file_path,
                    "valid": True,
                    "file_crs": file_crs,
                    "file_bounds": file_bounds,
                    "file_res": file_res,
                    "file_dtype": file_dtype,
                    "file_nodata": file_nodata,
                    "crs": target_crs or file_crs,
                    "bounds": transformed_bounds,
                    "res": transformed_res,
                    "colormap": colormap,
                }
        except Exception as e:
            msg = f"Unable to read {file_path}: \n{e}"
            logger.warning(msg)
            return {
                "paths": file_path,
                "valid": False,
                "file_crs": None,
                "file_bounds": None,
                "file_res": None,
                "file_dtype": None,
                "file_nodata": None,
                "crs": None,
                "bounds": None,
                "res": None,
                "colormap": None,
            }

    def _process_scan_results(self, file_metadata_list: list[dict]) -> pd.DataFrame:
        """Process scan results and create files DataFrame.

        Parameters
        ----------
        file_metadata_list : list[dict]
            List of file metadata dictionaries

        Returns
        -------
        pd.DataFrame
            DataFrame containing file information

        """
        # Create DataFrame from metadata
        files_df = pd.DataFrame(file_metadata_list)

        # Update spatial index for valid files only
        for count, (_, row) in enumerate(files_df[files_df.valid].iterrows()):
            self.index.insert(count, row.bounds, row.paths)

        # Check if any valid files were found
        if count == 0:
            msg = (
                f"No {self.__class__.__name__} data was found in "
                f"`root_dir='{self.root_dir}'`"
            )
            if self.bands:
                msg += f" with `bands={self.bands}`"
            raise FileNotFoundError(msg)

        # Log warning for invalid files
        if not files_df.valid.all():
            invalid_files = files_df[~files_df.valid].paths.astype(str).tolist()
            invalid_files_str = "\n\t".join(invalid_files)
            msg = (
                f"Unable to read {len(invalid_files)} files in "
                f"{self.__class__.__name__} dataset:\n{invalid_files_str}"
            )
            logger.warning(msg)
        # Set internal attributes
        self._count = count
        self._valid = files_df.valid.values

        # Set band indexes
        self.band_indexes = None
        if self.bands:
            if self.all_bands:
                self.band_indexes = [self.all_bands.index(i) + 1 for i in self.bands]
            else:
                msg = (
                    f"{self.__class__.__name__} is missing an `all_bands` "
                    "attribute, so `bands` cannot be specified."
                )
                logger.error(msg)
                raise AssertionError(msg)

        return files_df

    def _scan_files_sequential(
        self, paths: list[str], target_crs: CRS | None = None
    ) -> pd.DataFrame:
        """Scan files sequentially and extract metadata.

        Parameters
        ----------
        paths : list[str]
            List of file paths to scan
        target_crs : CRS, optional
            Target CRS for coordinate transformation

        Returns
        -------
        pd.DataFrame
            DataFrame containing file metadata

        """
        # Add progress bar if verbose
        paths_iter = (
            tqdm(paths, desc="Scanning files", unit=" files") if self.verbose else paths
        )

        # Extract metadata from all files
        file_metadata_list = []
        for path in paths_iter:
            metadata = self._extract_single_file_metadata(path, target_crs)
            file_metadata_list.append(metadata)

        # Process results and return DataFrame
        return self._process_scan_results(file_metadata_list)

    def _scan_files_with_dask(
        self, paths: list[str], target_crs: CRS | None = None
    ) -> pd.DataFrame:
        """Scan files with dask for parallel processing.

        Parameters
        ----------
        paths : list[str]
            List of file paths to scan
        target_crs : CRS, optional
            Target CRS for coordinate transformation

        Returns
        -------
        pd.DataFrame
            DataFrame containing file metadata

        """
        self._check_dask_available()

        # Create delayed tasks for file processing
        delayed_tasks = [
            delayed(self._extract_single_file_metadata)(path, target_crs)
            for path in paths
        ]

        # Add progress bar if verbose
        if self.verbose:
            sequence = tqdm(
                delayed_tasks, desc=f"Scanning {self.ds_name} files", unit=" files"
            )
        else:
            sequence = delayed_tasks

        # Execute all tasks in parallel
        file_metadata_list = list(dask.compute(*sequence))

        # Process results and return DataFrame
        return self._process_scan_results(file_metadata_list)

    def _determine_final_attributes(
        self,
        files_df: pd.DataFrame,
        user_crs: CRS | None,
        user_res: float | tuple[float, float] | None,
        user_dtype: np.dtype | None,
        user_nodata: float | None,
    ) -> tuple[CRS | None, tuple[float, float] | None, np.dtype | None, float | None]:
        """Determine dataset attributes based on user parameters and file metadata.

        Parameters
        ----------
        files_df : pd.DataFrame
            DataFrame containing file metadata
        user_crs : CRS, optional
            User-specified CRS
        user_res : float or tuple[float, float], optional
            User-specified resolution
        user_dtype : np.dtype, optional
            User-specified data type
        user_nodata : float, optional
            User-specified nodata value

        Returns
        -------
        tuple
            Final (crs, res, dtype, nodata) values

        """
        valid_files = files_df[files_df.valid]

        if len(valid_files) == 0:
            return user_crs, user_res, user_dtype, user_nodata

        first_valid = valid_files.iloc[0]

        # Determine final CRS
        final_crs = user_crs if user_crs is not None else first_valid.crs

        # Determine final resolution
        final_res = user_res if user_res is not None else first_valid.res

        # Determine final data type
        final_dtype = user_dtype if user_dtype is not None else first_valid.file_dtype

        # Determine final nodata value
        final_nodata = (
            user_nodata if user_nodata is not None else first_valid.file_nodata
        )

        # Check CRS consistency across files
        self._same_crs = valid_files.file_crs.nunique() == 1

        # Update colormap from first valid file if not already set
        if len(self.cmap) == 0 and first_valid.colormap:
            self.cmap = first_valid.colormap

        return final_crs, final_res, final_dtype, final_nodata

    @staticmethod
    def _check_dask_available() -> None:
        """Ensure dask is available when requested by a call-site."""
        if not HAS_DASK:
            msg = (
                "dask is required for lazy loading and parallel computation. "
                "Please install dask with: pip install 'dask[array]'"
            )
            logger.error(msg, stacklevel=2)
            raise ImportError(msg)

    def __getitem__(
        self,
        query: GeoQuery | Points | BoundingBox | Polygons,
    ) -> QueryResult:
        """Retrieve images values for given query.

        Parameters
        ----------
        query : GeoQuery | Points | BoundingBox | Polygons
            query to index the dataset. It can be :class:`Points`,
            :class:`BoundingBox`, :class:`Polygons`, or a composite
            :class:`GeoQuery` (recommended) object.

        Returns
        -------
        result : QueryResult
            a QueryResult instance containing the results of the various queries.

        """
        query = ensure_geo_query(query)

        paths = self.files[self.files.valid].paths.tolist()
        return self._sample_files(paths, query)

    def _ensure_bands_idx(self, vrt_fh: DatasetReader) -> list[int] | int:
        """Return the proper band indexes to use for the dataset.

        The band indexes is a list of integers if multiple bands are requested,
        otherwise it is an integer.
        """
        bands = self.band_indexes or vrt_fh.indexes
        # If only one band is requested, return a 2D array
        if len(bands) == 1:
            bands = bands[0]
        return bands

    def _ensure_dtype(self, data: np.ndarray) -> np.ndarray:
        """Ensure that the data has the same dtype as the dataset."""
        if data.dtype != self.dtype:
            data = data.astype(self.dtype)
        return data

    def _file_query_points(self, points: Points, vrt_fh: DatasetReader) -> np.ndarray:
        """Return the values of dataset at given points.

        Points that outside the dataset will be masked.
        """
        points = self._ensure_query_crs(points)
        bands_idx = self._ensure_bands_idx(vrt_fh)
        data = np.ma.hstack(list(vrt_fh.sample(points.values, bands_idx, masked=True)))
        return self._ensure_dtype(data)

    def _file_query_bbox(self, bbox: BoundingBox, vrt_fh: DatasetReader) -> np.ndarray:
        """Return the values of the dataset at the given bounding box."""
        bbox = self._ensure_query_crs(bbox)

        win = vrt_fh.window(*bbox)
        bands_idx = self._ensure_bands_idx(vrt_fh)
        out_shape = [
            round((bbox.top - bbox.bottom) / self.res[1]),
            round((bbox.right - bbox.left) / self.res[0]),
        ]
        if isinstance(bands_idx, list):
            out_shape.insert(0, len(bands_idx))

        data = vrt_fh.read(
            out_shape=tuple(out_shape),
            resampling=self.resampling,
            indexes=bands_idx,
            window=win,
            masked=True,
            boundless=self.same_crs,
            # WarpedVRT not supports boundless: https://github.com/rasterio/rasterio/issues/2084
        )

        if data.mask.ndim == 0:
            data = np.ma.masked_array(data.data, data == self.nodata)
        if self.fill_nodata:
            data = fill.fillnodata(data)
        return self._ensure_dtype(data)

    def _file_query_polygons(
        self, polygons: Polygons, vrt_fh: DatasetReader
    ) -> np.ndarray:
        """Return the values of the dataset at the given polygons."""
        polygons = self._ensure_query_crs(polygons)
        bands_idx = self._ensure_bands_idx(vrt_fh)
        mask_params = {
            "filled": False,
            "pad": polygons.pad,
            "all_touched": polygons.all_touched,
            "invert": False,
            "crop": True,
            "indexes": bands_idx,
        }
        rasterize_params = {
            "all_touched": polygons.all_touched,
            "fill": 0,
            "default_value": 1,
        }

        shapes = polygons.geodataframe.geometry.to_list()
        if len(polygons.desired) > 0:
            data_ls = []
            transform_ls = []
            mask_ls = []
            for shp in shapes:
                try:
                    data, out_transform = rio_mask.mask(vrt_fh, [shp], **mask_params)
                    # Create mask for valid data
                    rasterize_params.update(
                        {
                            "out_shape": data.shape
                            if data.ndim == 2
                            else data.shape[1:3],
                            "transform": out_transform,
                        },
                    )
                    mask = features.rasterize([shp], **rasterize_params).astype(bool)
                except ValueError as e:
                    if "Input shapes do not overlap raster" in str(e):
                        # Create empty masked array when polygon doesn't overlap
                        # with raster
                        data = np.ma.array([], dtype=np.float32).reshape(0, 0)
                        out_transform = vrt_fh.transform
                        mask = np.array([], dtype=bool).reshape(0, 0)
                    else:
                        raise

                if self.fill_nodata:
                    data = fill.fillnodata(data)
                    data = np.ma.masked_array(data.data, ~mask)
                data_ls.append(self._ensure_dtype(data))
                transform_ls.append(out_transform)
                mask_ls.append(mask)
        else:
            mask_params.update({"invert": True, "crop": False})
            data, out_transform = rio_mask.mask(vrt_fh, shapes, **mask_params)

            rasterize_params.update(
                {"out_shape": data.shape[1:3], "transform": out_transform},
            )
            mask = features.rasterize(shapes, **rasterize_params).astype(bool)
            if self.fill_nodata:
                data = fill.fillnodata(data)
                data = np.ma.masked_array(data.data, ~mask)
            data_ls = [self._ensure_dtype(data)]
            transform_ls = [out_transform]
            mask_ls = [mask]

        return data_ls, transform_ls, mask_ls

    def _files_query_points(
        self, points: Points, vrt_fhs: Iterable[DatasetReader]
    ) -> np.ndarray:
        """Return the values of the dataset at the given points."""
        data_ls = []
        for vrt_fh in vrt_fhs:
            data = self._file_query_points(points, vrt_fh)
            data_ls.append(data)
        return np.ma.asarray(data_ls)

    def _files_query_bbox(
        self, bbox: BoundingBox, vrt_fhs: Iterable[DatasetReader]
    ) -> np.ndarray:
        """Return the values of the dataset at the given bounding box."""
        data_ls = []
        for vrt_fh in vrt_fhs:
            data = self._file_query_bbox(bbox, vrt_fh)
            data_ls.append(data)
        return np.ma.asarray(data_ls)

    def _files_query_polygons(
        self, polygons: Polygons, vrt_fhs: Iterable[DatasetReader]
    ) -> tuple[list, list[Affine], list[np.ndarray]]:
        """Return the values of the dataset at the given polygons."""
        data_ls_all = []
        transform_ls = []
        mask_ls = []

        for vrt_fh in vrt_fhs:
            data_ls, transform_ls_file, mask_ls_file = self._file_query_polygons(
                polygons, vrt_fh
            )
            data_ls_all.append(data_ls)
            if not transform_ls:  # Only set once, should be same for all files
                transform_ls = transform_ls_file
                mask_ls = mask_ls_file

        # stack the files for each polygon
        n_polygons = len(polygons)
        poly_list = [[] for _ in range(n_polygons)]
        for file_data in data_ls_all:
            for i, poly_i in enumerate(file_data):
                poly_list[i].append(poly_i)

        # Handle arrays with potentially different shapes
        polygons_values = []
        for arr in poly_list:
            try:
                data = np.ma.asarray(arr)
                polygons_values.append(data)
            except ValueError:  # noqa: PERF203
                # If arrays have incompatible shapes, keep as list of individual arrays
                # This happens when different files produce different sized crops for the same polygon  # noqa: E501
                polygons_values.append(arr)

        return polygons_values, transform_ls, mask_ls

    def _ensure_loading_verbose(self, sequence: Iterable) -> Iterable:
        if self.verbose:
            sequence = tqdm(
                sequence, desc=f"Loading {self.ds_name} files", unit=" files"
            )
        return sequence

    def _ensure_saving_verbose(
        self,
        sequence: Iterable,
        ds_name: str,
        unit: str = " files",
    ) -> Iterable:
        if self.verbose:
            sequence = tqdm(sequence, desc=f"Saving {ds_name} files", unit=unit)
        return sequence

    def _safe_close(self, vrt_fhs: DatasetReader) -> None:
        """Close the file handles if not caching."""
        if not self.cache:
            for vrt_fh in vrt_fhs:
                vrt_fh.close()

    def _sample_files(
        self,
        paths: Iterable[str],
        query: GeoQuery,
        parallel_loading: bool | None = None,
    ) -> QueryResult:
        """Sample or retrieve values from the dataset for the given query.

        Parameters
        ----------
        paths : list of str
            list of paths for files to stack
        query : GeoQuery
            a GeoQuery instance containing the desired queries.
        parallel_loading : bool or None, optional
            if True, use dask for lazy loading and parallel computation. If None,
            use the dataset's default parallel_loading setting. Default is None.

        Returns
        -------
        result : QueryResult
            a QueryResult instance containing the results of the various queries.

        """
        # Convert paths to indexes for use with query methods
        paths_list = list(paths)
        valid_paths = self.files[self.files.valid].paths.tolist()
        indexes = [
            valid_paths.index(path) for path in paths_list if path in valid_paths
        ]

        if parallel_loading is None:
            parallel_loading = self.parallel_loading

        if parallel_loading and HAS_DASK:
            # Use deferred computation - build computation graph
            delayed_results = []

            if query.points is not None:
                delayed_results.append(
                    delayed(self._compute_points_query)(query.points, indexes)
                )
            else:
                delayed_results.append(delayed(lambda: None)())

            if query.boxes is not None:
                delayed_results.append(
                    delayed(self._compute_bbox_query)(query.boxes, indexes)
                )
            else:
                delayed_results.append(delayed(lambda: None)())

            if query.polygons is not None:
                delayed_results.append(
                    delayed(self._compute_polygons_query)(query.polygons, indexes)
                )
            else:
                delayed_results.append(delayed(lambda: None)())

            # Single compute call for all queries
            points_result, bbox_result, polygons_result = dask.compute(*delayed_results)
        else:
            # Use immediate computation (original behavior)
            points_result = None
            bbox_result = None
            polygons_result = None

            if query.points is not None:
                points_result = self._compute_points_query(query.points, indexes)
            if query.boxes is not None:
                bbox_result = self._compute_bbox_query(query.boxes, indexes)
            if query.polygons is not None:
                polygons_result = self._compute_polygons_query(query.polygons, indexes)

        return QueryResult(points_result, bbox_result, polygons_result, query)

    def _compute_points_query(
        self, points: Points, indexes: int | list[int] | None = None
    ) -> PointsResult:
        """Compute points query without dask (used by deferred computation)."""
        paths = self._indexes2paths(indexes)
        vrt_fhs = self._paths2vrt_fhs(paths)
        data_ls = self._files_query_points(points, vrt_fhs)

        multi_files = True
        if isinstance(indexes, int):
            if data_ls is not None and hasattr(data_ls, "squeeze"):
                data_ls = data_ls.squeeze(0)
            multi_files = False
        dims, data_ls = parse_1d_dims(data_ls, multi_files)
        return PointsResult({"data": data_ls, "dims": dims})

    def _compute_bbox_query(
        self,
        bbox: BoundingBox | list[BoundingBox],
        indexes: int | list[int] | None = None,
    ) -> BBoxesResult:
        """Compute bbox query without dask (used by deferred computation)."""
        bbox_is_list = isinstance(bbox, list)
        bbox_list = bbox if bbox_is_list else [bbox]
        paths = self._indexes2paths(indexes)

        all_bbox_data = []
        for single_bbox in bbox_list:
            vrt_fhs = self._paths2vrt_fhs(paths)
            data_ls = self._files_query_bbox(single_bbox, vrt_fhs)
            all_bbox_data.append(data_ls)

        # Stack results appropriately
        if bbox_is_list:
            final_data = np.ma.asarray(all_bbox_data)
            if final_data.ndim == 4:
                final_data = final_data.transpose(1, 0, 2, 3)
            elif final_data.ndim == 5:
                final_data = final_data.transpose(1, 0, 2, 3, 4)
        else:
            final_data = all_bbox_data[0]

        multi_files = True
        dims = parse_2d_dims(final_data, multi_files)
        return BBoxesResult({"data": final_data, "dims": dims})

    def _compute_polygons_query(
        self, polygons: Polygons, indexes: int | list[int] | None = None
    ) -> PolygonsResult:
        """Compute polygons query without dask (used by deferred computation)."""
        paths = self._indexes2paths(indexes)
        vrt_fhs = self._paths2vrt_fhs(paths)
        polygons_data, transform_ls, mask_ls = self._files_query_polygons(
            polygons, vrt_fhs
        )

        multi_files = True
        n_polygons = len(polygons)

        if n_polygons == 1:
            polygons_data = [polygons_data[0]]
            if isinstance(polygons_data[0], (np.ma.MaskedArray, np.ndarray)):
                dims = parse_2d_dims(polygons_data[0], multi_files)
            else:
                dims = [("files", "?"), ("height", "?"), ("width", "?")]
        elif len(polygons_data) > 0 and isinstance(
            polygons_data[0], (np.ma.MaskedArray, np.ndarray)
        ):
            dims = parse_2d_dims(polygons_data[0], multi_files)
        else:
            dims = [("files", "?"), ("height", "?"), ("width", "?")]

        return PolygonsResult(
            {
                "data": polygons_data,
                "dims": f"(n_polygons:{n_polygons}, ({format_dims_as_string(dims)}))",
                "transforms": transform_ls,
                "masks": mask_ls,
            }
        )

    @functools.lru_cache(maxsize=128)  # noqa: B019
    def _cached_load_warp_file(self, file_path: str) -> DatasetReader:
        """Return cached version of :meth:`_load_warp_file`.

        Parameters
        ----------
        file_path: str
            file to load and warp

        Returns
        -------
        vrt : DatasetReader
            file handle of warped VRT or original file if no warping is needed

        """
        return self._load_warp_file(file_path)

    def _load_warp_file(self, file_path: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Parameters
        ----------
        file_path: str
            file to load and warp

        Returns
        -------
        vrt : DatasetReader
            file handle of warped VRT or original file if no warping is needed

        """
        src = rasterio.open(file_path)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        return src

    def _indexes2paths(self, indexes: int | Iterable[int] | None) -> list[str]:
        """Convert file indexes to file paths.

        This method is used to convert the indexes of the files in the dataset
        to the file paths. It checks whether the indexes are valid and whether
        the files are valid. If the indexes are None, it returns all the file
        paths in the dataset.

        Parameters
        ----------
        indexes : int or list of int
            indexes of the files to convert

        Returns
        -------
        paths : list of str
            list of file paths

        Raises
        ------
        ValueError
            if the indexes negative or out of range or if the files are invalid

        """
        if isinstance(indexes, int):
            indexes = [indexes]
        if indexes is None:
            indexes = self.files[self.files.valid].index.values

        indexes = np.asarray(indexes)
        if len(indexes) == 0:
            msg = f"No valid files to query. indexes: {indexes}"
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)
        if np.any(indexes < 0):
            msg = f"indexes must be positive integers, got {indexes}"
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)
        if np.any(indexes >= len(self.files)):
            msg = f"indexes must be less than {len(self.files)}, got {indexes}"
            logger.error(msg, stacklevel=2)
            raise ValueError(msg)

        files_used = self.files.iloc[indexes, :]
        if not files_used.valid.all():
            msg = (
                "Following files are invalid and will be ignored: "
                f"{files_used[~files_used.valid]}"
            )
            logger.warning(msg, stacklevel=2)
            files_used = files_used[files_used.valid]

        return files_used.paths.values.tolist()

    def _paths2vrt_fhs(self, paths: Iterable[str]) -> list[DatasetReader]:
        """Convert file paths to file handles.

        Parameters
        ----------
        paths : list of str
            list of file paths

        Returns
        -------
        vrt_fhs : list of DatasetReader
            list of file handles

        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in paths]
        else:
            # load the files without caching
            # this is useful for testing and debugging to avoid caching issues
            vrt_fhs = [self._load_warp_file(fp) for fp in paths]
        return vrt_fhs

    @property
    def count(self) -> int:
        """Number of valid files in the dataset.

        .. Note::

            This is different from the length of the dataset ``len(GeoDataset)``,
            which is the total number of files in the dataset, including invalid
            files that cannot be read by rasterio.

        Returns
        -------
        count: int
            number of valid files in the dataset

        """
        return self._count

    @count.setter
    def count(self, new_count: int) -> None:
        """Set the number of files in the dataset.

        Parameters
        ----------
        new_count : int
            number of files in the dataset

        """
        self._count = int(new_count)

    @property
    def files(self) -> pd.DataFrame:
        """Return a list of all files in the dataset.

        Returns
        -------
            list of all files in the dataset

        """
        return self._files

    def get_profile(
        self,
        bbox: BoundingBox | Literal["roi", "bounds"] = "roi",
    ) -> Profile:
        """Get profile information of dataset for the given bounding box type."""
        bbox = self._ensure_bbox(bbox)
        profile = Profile.from_bounds_res(bbox, self.res)

        profile["count"] = self.count
        profile["dtype"] = self.dtype
        profile["nodata"] = self.nodata
        profile["crs"] = self.crs
        return profile

    def points_query(
        self,
        points: Points,
        indexes: int | list[int] | None = None,
        parallel_loading: bool | None = None,
    ) -> PointsResult:
        """Query the dataset for the given file index and points.

        Parameters
        ----------
        points : Points
            desired points to query.
        indexes : int or list of int or None, optional
            indexes of the files to query. If None, all files in the dataset
            will be used. Default is None.
        parallel_loading : bool or None, optional
            if True, use dask for lazy loading and parallel computation. If None,
            use the dataset's default parallel_loading setting. Default is None.

        Returns
        -------
        result : PointsResult
            a result object containing the results of the query.

        """
        if parallel_loading is None:
            parallel_loading = self.parallel_loading

        paths = self._indexes2paths(indexes)

        if parallel_loading:
            self._check_dask_available()
            # Create delayed tasks for file loading and points query
            delayed_vrt_fhs = [delayed(self._load_warp_file)(fp) for fp in paths]
            delayed_points_data = [
                delayed(self._file_query_points)(points, vrt_fh)
                for vrt_fh in delayed_vrt_fhs
            ]
            # Compute all delayed tasks in parallel
            points_data_list = dask.compute(*delayed_points_data)
            data_ls = np.ma.asarray(points_data_list)
        else:
            # Use original logic for non-dask case
            vrt_fhs = self._paths2vrt_fhs(paths)
            data_ls = self._files_query_points(points, vrt_fhs)

        multi_files = True
        if isinstance(indexes, int):
            if data_ls is not None and hasattr(data_ls, "squeeze"):
                data_ls = data_ls.squeeze(0)
            multi_files = False
        dims, data_ls = parse_1d_dims(data_ls, multi_files)
        return PointsResult({"data": data_ls, "dims": dims})

    def bbox_query(
        self,
        bbox: BoundingBox | list[BoundingBox],
        indexes: int | list[int] | None = None,
        parallel_loading: bool | None = None,
    ) -> BBoxesResult:
        """Query the dataset for the given file index and bounding box(es).

        Parameters
        ----------
        bbox : BoundingBox or list[BoundingBox]
            desired bounding box(es) to query. If a list is provided (even with one
            bbox), the output will have a bbox dimension. If a single bbox (not in a
            list) is provided, no bbox dimension will be added.
        indexes : int or list of int or None, optional
            indexes of the files to query. If None, all files in the dataset
            will be used. Default is None. File dimension will never be automatically
            removed even if it's 1.
        parallel_loading : bool or None, optional
            if True, use dask for lazy loading and parallel computation. If None,
            use the dataset's default parallel_loading setting. Default is None.

        Returns
        -------
        result : BBoxesResult
            a result object containing the results of the query.

        """
        if parallel_loading is None:
            parallel_loading = self.parallel_loading

        bbox_is_list = isinstance(bbox, list)
        # If single bbox, convert to list for processing, but remember original format
        bbox_list = bbox if bbox_is_list else [bbox]

        paths = self._indexes2paths(indexes)

        # Process each bbox separately
        all_bbox_data = []
        for single_bbox in bbox_list:
            if parallel_loading:
                self._check_dask_available()
                # Create delayed tasks for file loading and bbox query
                delayed_vrt_fhs = [delayed(self._load_warp_file)(fp) for fp in paths]
                delayed_bbox_data = [
                    delayed(self._file_query_bbox)(single_bbox, vrt_fh)
                    for vrt_fh in delayed_vrt_fhs
                ]
                # Compute all delayed tasks in parallel
                bbox_data_list = dask.compute(*delayed_bbox_data)
                data_ls = np.ma.asarray(bbox_data_list)
            else:
                vrt_fhs = self._paths2vrt_fhs(paths)
                data_ls = self._files_query_bbox(single_bbox, vrt_fhs)

            all_bbox_data.append(data_ls)

        # Stack results appropriately
        if bbox_is_list:
            # If input was a list, add bbox dimension
            final_data = np.ma.asarray(all_bbox_data)
            # Transpose to get (n_files, n_bboxes, height, width) or
            # (n_files, n_bboxes, n_bands, height, width)
            if final_data.ndim == 4:  # (n_bboxes, n_files, height, width)
                final_data = final_data.transpose(1, 0, 2, 3)
            elif final_data.ndim == 5:  # (n_bboxes, n_files, n_bands, height, width)
                final_data = final_data.transpose(1, 0, 2, 3, 4)
        else:
            # If input was single bbox, no bbox dimension
            final_data = all_bbox_data[0]

        # Never squeeze file dimension, even if it's 1
        multi_files = True
        dims = parse_2d_dims(final_data, multi_files)
        return BBoxesResult({"data": final_data, "dims": dims})

    def polygons_query(
        self,
        polygons: Polygons,
        indexes: int | list[int] | None = None,
        parallel_loading: bool | None = None,
    ) -> PolygonsResult:
        """Query the dataset for the given file index and polygons.

        Parameters
        ----------
        polygons : Polygons
            desired polygons to query. Each polygon will always correspond to a result,
            so there will always be a polygon dimension.
        indexes : int or list of int, optional
            indexes of the files to query. If None, all files in the dataset
            will be used. Default is None. File dimension will never be automatically
            removed even if it's 1.
        parallel_loading : bool or None, optional
            if True, use dask for lazy loading and parallel computation. If None,
            use the dataset's default parallel_loading setting. Default is None.

        Returns
        -------
        result : PolygonsResult
            a result object containing the results of the query.

        """
        if parallel_loading is None:
            parallel_loading = self.parallel_loading

        paths = self._indexes2paths(indexes)

        if parallel_loading:
            self._check_dask_available()
            # Create delayed tasks for file loading and polygons query
            delayed_vrt_fhs = [delayed(self._load_warp_file)(fp) for fp in paths]
            delayed_polygons_data = [
                delayed(self._file_query_polygons)(polygons, vrt_fh)
                for vrt_fh in delayed_vrt_fhs
            ]
            # Compute all delayed tasks in parallel
            polygons_results_list = dask.compute(*delayed_polygons_data)

            # Process polygons results similar to the original method
            data_ls_all = [result[0] for result in polygons_results_list]
            # All should have same transforms
            transform_ls = polygons_results_list[0][1]
            # All should have same masks
            mask_ls = polygons_results_list[0][2]

            # Process the results similar to non-dask version
            # stack the files for each polygon
            n_polygons = len(polygons)
            poly_list = [[] for _ in range(n_polygons)]
            for file_data in data_ls_all:
                for i, poly_i in enumerate(file_data):
                    poly_list[i].append(poly_i)

            # Handle arrays with potentially different shapes
            polygons_values = []
            for arr in poly_list:
                try:
                    data = np.ma.asarray(arr)
                    polygons_values.append(data)
                except ValueError:  # noqa: PERF203
                    # If arrays have incompatible shapes, keep as list
                    polygons_values.append(arr)

            # Apply same logic as non-dask version
            if n_polygons == 1:
                # For single polygon, convert to numpy array if possible
                try:
                    polygons_data_array = np.ma.asarray(polygons_values[0])
                    polygons_data = [
                        polygons_data_array
                    ]  # Wrap in list for polygon dimension
                    dims = parse_2d_dims(polygons_data_array, True)
                except ValueError:
                    polygons_data = [polygons_values[0]]
                    dims = [("files", "?"), ("height", "?"), ("width", "?")]
            else:
                polygons_data = polygons_values
                if len(polygons_data) > 0 and isinstance(
                    polygons_data[0], (np.ma.MaskedArray, np.ndarray)
                ):
                    dims = parse_2d_dims(polygons_data[0], True)
                else:
                    dims = [("files", "?"), ("height", "?"), ("width", "?")]

            return PolygonsResult(
                {
                    "data": polygons_data,
                    "dims": f"(n_polygons:{len(polygons)}, ({format_dims_as_string(dims)}))",  # noqa: E501
                    "transforms": transform_ls,
                    "masks": mask_ls,
                }
            )
        vrt_fhs = self._paths2vrt_fhs(paths)
        polygons_data, transform_ls, mask_ls = self._files_query_polygons(
            polygons, vrt_fhs
        )

        # Never squeeze file dimension, always keep multi_files=True
        multi_files = True

        n_polygons = len(polygons)
        if n_polygons == 1:
            # For single polygon, polygons_data is already a list with one element
            # which is a numpy array with shape (n_files, height, width)
            # We need to wrap it in another list to maintain polygon dimension
            polygons_data = [polygons_data[0]]  # polygons_data[0] is the stacked array
            if isinstance(polygons_data[0], (np.ma.MaskedArray, np.ndarray)):
                dims = parse_2d_dims(polygons_data[0], multi_files)
            else:
                dims = [("files", "?"), ("height", "?"), ("width", "?")]
        # For multiple polygons, polygons_data is already structured correctly
        # Each element in polygons_data corresponds to one polygon
        # Never squeeze file dimension
        elif len(polygons_data) > 0 and isinstance(
            polygons_data[0], (np.ma.MaskedArray, np.ndarray)
        ):
            dims = parse_2d_dims(polygons_data[0], multi_files)
        else:
            dims = [("files", "?"), ("height", "?"), ("width", "?")]

        return PolygonsResult(
            {
                "data": polygons_data,
                "dims": f"(n_polygons:{n_polygons}, ({format_dims_as_string(dims)}))",
                "transforms": transform_ls,
                "masks": mask_ls,
            },
        )

    def query(
        self,
        query: GeoQuery | Points | BoundingBox | Polygons,
        indexes: int | list[int] | None = None,
        parallel_loading: bool | None = None,
    ) -> QueryResult:
        """Retrieve images values for given query.

        This method is an more flexible implementation compared to
        :meth:`__getitem__`, which can retrieve images only for the given pairs.

        Parameters
        ----------
        query : GeoQuery | Points | BoundingBox | Polygons
            query to index the dataset. It can be :class:`Points`,
            :class:`BoundingBox`, :class:`Polygons`, or a composite
            :class:`GeoQuery` (recommended) object.
        indexes : int or list of int or None, optional
            indexes of the files to query. If None, all files in the dataset
            will be used. Default is None.
        parallel_loading : bool or None, optional
            if True, use dask for lazy loading and parallel computation. If None,
            use the dataset's default parallel_loading setting. Default is None.

        Returns
        -------
        result : QueryResult
            a QueryResult instance containing the results of the various queries.

        """
        if parallel_loading is None:
            parallel_loading = self.parallel_loading

        if isinstance(query, Points):
            query = GeoQuery(points=query)
        if isinstance(query, BoundingBox):
            query = GeoQuery(boxes=query)
        if isinstance(query, Polygons):
            query = GeoQuery(polygons=query)

        paths = self._indexes2paths(indexes)
        return self._sample_files(paths, query, parallel_loading=parallel_loading)

    def row_col(
        self,
        xy: Iterable,
        crs: CRS | str | None = None,
        bbox: BoundingBox | Literal["roi", "bounds"] = "roi",
    ) -> np.ndarray:
        """Convert x, y coordinates to row, col in the dataset.

        Parameters
        ----------
        xy: Iterable
            Pairs of x, y coordinates (floats)
        crs: CRS or str, optional
            The CRS of the points. If None, the CRS of the dataset will be used.
            allowed CRS formats are the same as those supported by rasterio.
        bbox : str, one of {'bounds', 'roi'}, optional
            the bounding box used to calculate the ``width``, ``height``
            and ``transform`` of the dataset for the profile. Default is 'roi'.

        Returns
        -------
        row_col: np.ndarray
            row, col in the dataset for the given points(xy)

        """
        xy = np.asarray(xy)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)
        if xy.ndim != 2 or xy.shape[1] != 2:
            msg = f"Expected xy to be an array of shape (n, 2), got {xy.shape}"
            raise ValueError(
                msg,
            )
        if crs is not None:
            crs = CRS.from_user_input(crs)
            if crs != self.crs:
                xs, ys = warp_transform(crs, self.crs, xy[:, 0], xy[:, 1])
                xy = np.column_stack((xs, ys))

        profile = self.get_profile(bbox)

        rows, cols = tf_rowcol(profile["transform"], xy[:, 0], xy[:, 1])
        return np.column_stack((rows, cols)).astype(np.int64)

    def xy(
        self,
        row_col: Iterable,
        crs: CRS | str | None = None,
        bbox: BoundingBox | Literal["roi", "bounds"] = "roi",
    ) -> np.ndarray:
        """Convert row, col in the dataset to x, y coordinates.

        Parameters
        ----------
        row_col: Iterable
            Pairs of row, col in the dataset (floats)
        crs: CRS or str, optional
            The CRS of output points. If None, the CRS of the dataset will be used.
            Can be any of the formats supported by :meth:`pyproj.CRS.from_user_input`.
        bbox : str, one of {'bounds', 'roi'}, optional
            the bounding box used to calculate the ``width``, ``height``
            and ``transform`` of the dataset for the profile. Default is 'roi'.

        Returns
        -------
        xy: np.ndarray
            x, y coordinates in the given CRS (default is the CRS of the dataset)

        """
        row_col = np.asarray(row_col)
        if row_col.ndim == 1:
            row_col = row_col.reshape(1, -1)
        if row_col.ndim != 2 or row_col.shape[1] != 2:
            msg = (
                f"Expected row_col to be an array of shape (n, 2), got {row_col.shape}"
            )
            raise ValueError(
                msg,
            )

        profile = self.get_profile(bbox)

        xs, ys = tf_xy(profile["transform"], row_col[:, 0], row_col[:, 1])

        if crs is not None:
            crs = CRS.from_user_input(crs)
            if crs != self.crs:
                xs, ys = warp_transform(self.crs, crs, xs, ys)
        return np.column_stack((xs, ys))

    def parse_mask(
        self,
        percent: float,
        bbox: BoundingBox | Literal["roi", "bounds"] = "roi",
        seed: int = 0,
    ) -> np.ndarray:
        """Parse the mask of the dataset.

        The mask is a boolean array where True indicates valid data and False
        indicates invalid data, which keeps in line with the GDAL/rasterio strategy.

        Parameters
        ----------
        percent : float
            Percentage (0,1] of files to be used for parsing the mask. The files are
            randomly selected.
        bbox : str, one of {'bounds', 'roi'}, optional
            the desired region of mask. Default is 'roi'.
        seed : int, optional
            Seed for the random number generator. Default is 0.

        """
        # randomly select a subset of files
        idx_all = np.arange(self.count)
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx_all, int(percent * self.count), replace=False)
        paths = self.files.paths[self.valid].values[idx]

        # get the profile of the dataset
        profile = self.get_profile(bbox)
        width, height = profile["width"], profile["height"]
        mask = np.ones((height, width), dtype=bool)

        if self.verbose:
            paths = tqdm(paths, desc="Parsing Mask", unit=" files")
        for path in paths:
            with rasterio.open(path) as src:
                bbox = self._ensure_bbox(bbox)
                win = None if bbox is None else src.window(*bbox)
                mask &= src.read(1, masked=True, window=win).mask
        return ~mask

    def load_mask(
        self,
        mask_path: PathLike,
        bbox: BoundingBox | Literal["roi", "bounds"] = "roi",
    ) -> np.ndarray:
        """Load a mask from a tiff mask file (.msk).

        Parameters
        ----------
        mask_path : str or PathLike
            path to the mask file of tiff format (.msk)
        bbox : str, one of {'bounds', 'roi'}, optional
            the desired region of mask. Default is 'roi'.

        """
        bbox = self._ensure_bbox(bbox)
        profile = self.get_profile(self.bounds)

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        if profile["width"] != mask.shape[1] or profile["height"] != mask.shape[0]:
            msg = (
                f"The shape of the mask {mask.shape} does not match the shape "
                f"of the dataset {(profile['width'], profile['height'])}."
            )
            raise ValueError(
                msg,
            )
        # crop the mask to the desired region
        with rasterio.open(self.files.paths[self.valid].values[0]) as src:
            win = src.window(*bbox)
            return mask[win[0] : win[1], win[2] : win[3]]

    def reproject(
        self,
        new_crs: CRS | str,
        resampling: Resampling = Resampling.nearest,
        nodata: float | None = None,
    ) -> Self:
        """Reproject the dataset to a new CRS.

        Parameters
        ----------
        new_crs : CRS or str
            new coordinate reference system (:term:`CRS`) of the dataset.
            It can be a CRS object or a string, which will be parsed to a
            CRS object. The string can be in any format supported by
            :meth:`pyproj.crs.CRS.from_user_input`.
        resampling : Resampling, optional
            resampling method to use when reprojecting the dataset.
            Default is `Resampling.nearest`.
        nodata : float or int, optional
            no data value of the dataset. If None, the no data value of the
            dataset will be used.

        """
        if not isinstance(new_crs, CRS):
            new_crs = CRS.from_user_input(new_crs)
        if new_crs == self.crs:
            return self

        if nodata is None:
            nodata = self.nodata
        new_bounds: BoundingBox = self.bounds.to_crs(new_crs)
        new_res = (
            abs(new_bounds.right - new_bounds.left) / self.shape[1],
            abs(new_bounds.top - new_bounds.bottom) / self.shape[0],
        )

        return self.__class__(
            root_dir=self.root_dir,
            paths=self.files.paths,
            crs=new_crs,
            res=new_res,
            dtype=self.dtype,
            nodata=nodata,
            roi=new_bounds,
            bands=self.bands,
            cache=self.cache,
            resampling=resampling,
            fill_nodata=self.fill_nodata,
            verbose=self.verbose,
            ds_name=self.ds_name,
        )

    def resample(
        self,
        new_res: float | tuple[float, float],
        resampling: Resampling = Resampling.nearest,
        nodata: float | None = None,
    ) -> Self:
        """Resample the dataset to a new resolution.

        Parameters
        ----------
        new_res : float or tuple of float
            new resolution of the dataset in units of CRS. If a single float is
            provided, it will be used for both x and y dimensions.
        resampling : Resampling, optional
            resampling method to use when resampling the dataset.
            Default is `Resampling.nearest`.
        nodata : float or int, optional
            no data value of the dataset. If None, the no data value of the
            dataset will be used.

        """
        if nodata is None:
            nodata = self.nodata

        return self.__class__(
            root_dir=self.root_dir,
            paths=self.files.paths,
            crs=self.crs,
            res=new_res,
            dtype=self.dtype,
            nodata=nodata,
            roi=self.bounds,
            bands=self.bands,
            cache=self.cache,
            resampling=resampling,
            fill_nodata=self.fill_nodata,
            verbose=self.verbose,
            ds_name=self.ds_name,
        )

    def align_to(
        self,
        other: Self,
        resampling: Resampling = Resampling.nearest,
        nodata: float | None = None,
    ) -> Self:
        """Align the dataset to another dataset.

        Parameters
        ----------
        other : GeoDataset
            dataset to align to
        resampling : Resampling, optional
            resampling method to use when resampling the dataset.
            Default is `Resampling.nearest`.
        nodata : float or int, optional
            no data value of the dataset. If None, the no data value of the
            dataset will be used.

        """
        if nodata is None:
            nodata = self.nodata
        return self.__class__(
            root_dir=self.root_dir,
            paths=self.files.paths,
            crs=other.crs,
            res=other.res,
            dtype=self.dtype,
            nodata=nodata,
            roi=other.bounds,
            bands=self.bands,
            cache=self.cache,
            resampling=resampling,
            fill_nodata=self.fill_nodata,
            verbose=self.verbose,
            ds_name=self.ds_name,
        )

    def show(
        self,
        arr: np.ndarray,
        **kwargs,
    ) -> Axes:
        """Show the array using the dataset's geo information.

        Parameters
        ----------
        arr : np.ndarray
            The array with same shape as the dataset to show. The geo information
            of the dataset will be used to plot the array.
        kwargs : key value pairs, optional
            Additional keyword arguments to pass to the :func:`rasterio.plot.show`
            function.

        Returns
        -------
        ax : Axes
            The axes object of the plot.

        """
        if kwargs is None:
            kwargs = {}
        if "transform" not in kwargs:
            kwargs["transform"] = self.get_profile().transform
        return plot.show(arr, **kwargs)

    def to_tiffs(
        self,
        out_dir: PathLike,
        roi: BoundingBox | None = None,
    ) -> None:
        """Save the dataset to a directory of tiff files for given region of interest.

        Parameters
        ----------
        out_dir : str or PathLike
            path to the directory to save the tiff files
        roi : BoundingBox, optional
            region of interest to save. If None, the roi of the dataset will be used.
            Default is None.

        """
        roi = self._check_roi(roi)

        profile = self.get_profile(roi)
        profile["count"] = 1

        for f in self.files.paths[self.valid]:
            out_file = Path(out_dir) / f.name
            src = self._load_warp_file(f)
            dest_arr = self._file_query_bbox(roi, src).squeeze(0)
            with rasterio.open(out_file, "w", **profile.to_dict()) as dst:
                dst.write(dest_arr, 1)

    def to_netcdf(
        self,
        filename: PathLike,
        roi: BoundingBox | None = None,
    ) -> None:
        """Save the dataset to a netCDF file for given region of interest.

        Parameters
        ----------
        filename : str
            path to the netCDF file to save
        roi : BoundingBox, optional
            region of interest to save. If None, the roi of the dataset will be used.

        """
        if roi is None:
            roi = self.roi

        profile = self.get_profile(roi)
        lat, lon = profile.to_latlon()

        sample = self[roi]

        ds = xr.Dataset(
            {"image": (["band", "lat", "lon"], sample.boxes.data)},
            coords={
                "band": list(range(profile["count"])),
                "lat": lat,
                "lon": lon,
            },
        )
        ds = geo_tools.write_geoinfo_into_ds(
            ds,
            "image",
            crs=self.crs,
            x_dim="lon",
            y_dim="lat",
        )
        ds.to_netcdf(filename)

    def array2tiff(
        self,
        arr: np.ndarray,
        filename: PathLike,
        bounds: BoundingBox | None = None,
        bbox: BoundingBox | None = None,
        band_names: Iterable[str] | None = None,
        arr_type: Literal["data", "mask"] = "data",
        nodata: float | None = None,
        overwrite: bool = False,
    ) -> None:
        """Save a numpy array to a tiff file using the geoinformation of dataset.

        Parameters
        ----------
        arr : numpy.ndarray
            numpy array to save. arr can be a 2D array or a 3D array. If arr is a
            3D array, the first dimension should be the band dimension.
        filename : str or PathLike
            path to the tiff file to save
        bounds : BoundingBox, optional
            the bounds of the arr. Default is None, which means the roi of the
            dataset will be used.
        bbox : BoundingBox, optional
            if specified, the input array will be saved to the given part/bbox of
            dataset. Default is None, which means the array will be saved to the
            entire dataset.
        band_names : Iterable of str, optional
            names of bands to save. Default is None, which will use the band indexes.
        arr_type : str, one of ['data', 'mask'], optional
            type of the array to save. Default is 'data'.
        nodata : float or int, optional
            no data value of the dataset. If None, will automatically parse the
            a proper no data value for the array.
        overwrite : bool, optional
            if True, overwrite the existing file. Default is False, which means
            the array will be saved in append mode (r+ mode).

        """
        # check arr dimension
        if arr.ndim == 2:
            indexes = [1]
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            indexes = [i + 1 for i in range(arr.shape[0])]
        else:
            msg = (
                f"Expected arr to be an array with shape of (n_lat, n_lon) or "
                f"(n_band, n_lat, n_lon), got {arr.shape}"
            )
            raise ValueError(msg)
        # check length of band_names
        if band_names is not None and len(band_names) != arr.shape[0]:
            msg = (
                f"Expected band_names to be of length {arr.shape[0]}, "
                f"got {len(band_names)}"
            )
            raise ValueError(msg)
        # parse profile
        if bounds is None:
            bounds = self.roi
        profile = self.get_profile(bounds)
        profile["count"] = arr.shape[0]
        profile["driver"] = "GTiff"
        profile["dtype"] = get_minimum_dtype(arr)
        profile["nodata"] = get_nodata(arr, nodata, profile["dtype"])
        mode = "w"
        filename = Path(filename)
        if filename.exists() and not overwrite:
            mode = "r+"

        with rasterio.open(filename, mode, **profile.to_dict()) as dst:
            # parse window
            win = None if bbox is None else dst.window(*bbox)

            # write array to tiff
            if arr_type == "mask":
                if arr.shape[0] == 1:
                    arr = arr[0]
                dst.write_mask(arr)
            elif arr_type == "data":
                dst.write(arr, indexes, window=win)
            # update band names
            if band_names is not None:
                dst.descriptions = band_names
                band_names_str = ";".join(band_names)
                band_names_file = filename.with_suffix(".band_name.txt")
                with band_names_file.open("w") as f:
                    f.write(band_names_str)

    def array2kml(
        self,
        arr: np.ndarray,
        out_file: PathLike,
        bounds: BoundingBox | None = None,
        img_kwargs: dict | None = None,
        cbar_kwargs: dict | None = None,
        verbose: bool = True,
    ) -> None:
        """Write a numpy array into a kml file.

        Parameters
        ----------
        arr: numpy.ndarray
            the numpy array to be written into kml file.
        out_file: str or PathLike
            the path of the kml file.
        bounds : BoundingBox, optional
            the bounds of the arr. Default is None, which means the roi of the
            dataset will be used.
        img_kwargs: dict
            the keyword arguments for :func:`matplotlib.pyplot.imshow` function.
        cbar_kwargs: dict
            the keyword arguments for :func:`save_colorbar` function, except for
            the out_file and mappable argument.
        verbose: bool
            whether to print the information of the kml file. Default is verbose.

        """
        if cbar_kwargs is None:
            cbar_kwargs = {}
        if img_kwargs is None:
            img_kwargs = {}
        if bounds is None:
            bounds = self.roi

        wgs84 = CRS.from_epsg(4326)
        if self.crs != wgs84:
            profile = self.get_profile(bounds)
            lat, lon = profile.to_latlon()
            dtype = get_minimum_dtype(arr)
            nodata = get_nodata(arr, None, dtype)

            da = xr.DataArray(arr, coords=[lat, lon], dims=["y", "x"])
            da.rio.set_spatial_dims("x", "y", inplace=True)
            da.rio.write_crs(self.crs, inplace=True)
            da = da.rio.reproject(wgs84, nodata=nodata)
            # update arr and bounds
            arr = da.values
            bounds, *_ = geoinfo_from_latlon(da.y, da.x)
            bounds.set_crs(wgs84)

        array2kml(arr, out_file, bounds, img_kwargs, cbar_kwargs, verbose)

    def array2kmz(
        self,
        arr: np.ndarray,
        out_file: PathLike,
        bounds: BoundingBox | None = None,
        img_kwargs: dict | None = None,
        cbar_kwargs: dict | None = None,
        keep_kml: bool = False,
        verbose: bool = True,
    ) -> None:
        """Write a numpy array into a kmz file.

        Parameters
        ----------
        arr: numpy.ndarray
            the numpy array to be written into kmz file.
        out_file: str or PathLike
            the path of the kmz file.
        bounds : BoundingBox, optional
            the bounds of the arr. Default is None, which means the roi of the
            dataset will be used.
        img_kwargs: dict
            the keyword arguments for :func:`matplotlib.pyplot.imshow` function.
        cbar_kwargs: dict
            the keyword arguments for :func:`save_colorbar` function, except for
            the out_file and mappable argument.
        keep_kml: bool
            whether to keep the kml file. Default is False.
        verbose: bool
            whether to print the information of the kmz file. Default is verbose.

        """
        if cbar_kwargs is None:
            cbar_kwargs = {}
        if img_kwargs is None:
            img_kwargs = {}
        if bounds is None:
            bounds = self.roi
        wgs84 = CRS.from_epsg(4326)
        if self.crs != wgs84:
            profile = self.get_profile(bounds)
            lat, lon = profile.to_latlon()
            dtype = get_minimum_dtype(arr)
            nodata = get_nodata(arr, None, dtype)

            da = xr.DataArray(arr, coords=[lat, lon], dims=["y", "x"])
            da.rio.set_spatial_dims("x", "y", inplace=True)
            da.rio.write_crs(self.crs, inplace=True)
            da = da.rio.reproject(wgs84, nodata=nodata)
            # update arr and bounds
            arr = da.values
            bounds, *_ = geoinfo_from_latlon(da.y, da.x)
            bounds.set_crs(wgs84)

        array2kmz(arr, out_file, bounds, img_kwargs, cbar_kwargs, keep_kml, verbose)


class HierarchicalDataset(GeoDataset):
    """A base class for hierarchical dataset, like h5 and nc files.

    .. note::
        This class is used to load and sample data from a single file. If you
        want to load and sample data from multiple files, you should use
        :class:`MultiHierarchicalDataset`.
    """

    lat_name: str = "lat"
    lon_name: str = "lon"

    def __init__(
        self,
        path: str | Path,
        group: str | None = None,
        roi: BoundingBox | None = None,
    ) -> None:
        super().__init__()
        self._path = Path(path)
        self._group = group
        self._roi = roi
        self._update_geo_info()
        warnings.warn(
            "HierarchicalDataset is still in development and may not work as expected.",
            stacklevel=2,
        )

    def __repr__(self) -> str:
        return self._repr_str

    def _update_geo_info(self) -> None:
        bound, res, shape, crs, ds_info = self._parse_geo_info(self._path)
        self._bound = bound
        self._res = res
        self._crs = crs
        self._shape = shape
        self._lat = ds_info[0]
        self._lon = ds_info[1]
        self._variables = ds_info[2]
        self._repr_str = ds_info[3]

    def _parse_lat_lon_name(self, ds: xr.Dataset) -> tuple[str, str]:
        """Parse the name of the latitude and longitude variables."""
        lat_name = None
        lon_name = None
        if self.lat_name in ds.variables and self.lon_name in ds.variables:
            return None

        for name in ds.variables:
            if name.lower() in lat_names:
                lat_name = name
            if name.lower() in lon_names:
                lon_name = name
        if lat_name is None or lon_name is None:
            msg = (
                "The dataset does not contain latitude and longitude variables. "
                "Please specify the names of the latitude and longitude variables."
            )
            raise ValueError(
                msg,
            )
        return lat_name, lon_name

    def _parse_geo_info(
        self,
        path: str | Path,
    ) -> tuple[BoundingBox, tuple[float, float], tuple[int, int], CRS]:
        """Parse the geoinformation of the dataset."""
        with xr.open_dataset(path) as ds:
            coord_names = self._parse_lat_lon_name(ds)
            if coord_names is not None:
                self.lat_name, self.lon_name = coord_names

            repr_str = ds.__repr__()
            variables = list(ds.variables)
            lat = ds[self.lat_name].values
            lon = ds[self.lon_name].values
            crs = ds.rio.crs

        # parse geo-information
        if crs is None:
            if (
                np.all(lat >= -90)
                and np.all(lat <= 90)
                and np.all(lon >= -180)
                and np.all(lon <= 180)
            ):
                warnings.warn(
                    "No CRS is specified for the dataset, assuming the lat/lon values "
                    "are in the range of WGS84.",
                    stacklevel=2,
                )
                crs = CRS.from_epsg(4326)
            else:
                msg = (
                    "No CRS is specified for the dataset, and the lat/lon values are "
                    "not in the range of WGS84. Please specify the CRS of the dataset"
                    "using the :meth:`set_crs` method later."
                )
                raise ValueError(
                    msg,
                )
        else:
            crs = CRS.from_user_input(ds.rio.crs)
        # parse bound, resolution, shape
        bound, res, shape = geoinfo_from_latlon(lat, lon)
        bound.set_crs(crs)

        return bound, res, shape, crs, (lat, lon, variables, repr_str)

    def __getitem__(self, var: str) -> xr.DataArray | xr.Dataset:
        """Get the variable from the dataset."""
        with xr.open_dataset(self.path, group=self.group) as ds:
            return ds[var]

    def flush_geo_info(self) -> None:
        """Flush the geoinformation of the dataset to the given file."""
        with xr.open_dataset(self.path, group=self.group, mode="a") as ds:
            ds.rio.write_crs(self.crs)
            ds.rio.set_spatial_dims(x_dim=self.lon_name, y_dim=self.lat_name)
        self._update_geo_info()

    def _bbox_query(
        self,
        bbox: BoundingBox,
        variable: str | None = None,
        **kwargs,
    ) -> xr.DataArray | xr.Dataset:
        """Retrieve the data of the dataset for the given bounding box."""
        bbox = self._ensure_query_crs(bbox)
        # get slice for lat/lon values
        if self.lat[0] < self.lat[-1]:
            slice_lat = slice(bbox.bottom, bbox.top)
        else:
            slice_lat = slice(bbox.top, bbox.bottom)
        slice_lon = slice(bbox.left, bbox.right)
        # open and read the dataset
        if variable is None:
            ds = xr.open_dataarray(self.path, group=self.group, **kwargs)
        else:
            ds = xr.open_dataset(self.path, group=self.group, **kwargs)[variable]
        if "y" not in ds.coords or "x" not in ds.coords:
            ds = ds.rename({self.lat_name: "y", self.lon_name: "x"})
        data = ds.sel(y=slice_lat, x=slice_lon)
        # close dataset
        ds.close()

        return data

    def _points_query(
        self,
        points: Points,
        variable: str | None = None,
    ) -> np.ndarray:
        """Return the values of dataset at given points.

        Points that outside the dataset will be masked.
        """

    def _polygons_query(
        self,
        polygons: Polygons,
        variable: str | None = None,
    ) -> np.ndarray:
        """Return the values of the dataset at the given polygons."""

    def query(
        self,
        query: GeoQuery | Points | BoundingBox | Polygons,
        variable: str | None = None,
        **kwargs,
    ) -> QueryResult:
        """Retrieve images values for given query.

        Parameters
        ----------
        query : GeoQuery | Points | BoundingBox | Polygons
            query to index the dataset. It can be :class:`Points`,
            :class:`BoundingBox`, :class:`Polygons`,
            or a composite :class:`GeoQuery` (recommended) object.
        variable : str, optional
            name of the variable to retrieve. If None, all variables will be retrieved.
        **kwargs : dict
            keyword arguments to pass to :meth:`xarray.open_dataarray` if
            variable is None, otherwise to :meth:`xarray.open_dataset`.

        """
        if isinstance(query, Points):
            query = GeoQuery(points=query)
        if isinstance(query, BoundingBox):
            query = GeoQuery(boxes=query)
        if isinstance(query, Polygons):
            query = GeoQuery(polygons=query)

        return self._sample_data(query, variable, **kwargs)

    def _sample_data(
        self,
        query: GeoQuery,
        variable: str | None = None,
        **kwargs,
    ) -> QueryResult:
        """Sample data from the dataset for the given query."""
        # TODO: refine points and polygons query
        # parse points result
        points_result = None
        if query.points is not None:
            points_values = self._points_query(query.points, variable, **kwargs)
            dims, points_result = parse_1d_dims(points_values, multi_files=False)
            points_result = {"data": points_values, "dims": dims}
        # parse bounding boxes result
        boxes_result = None
        if query.boxes is not None:
            if len(query.boxes) == 1:
                boxes_values = self._bbox_query(query.boxes[0], variable, **kwargs)
                dims = parse_2d_dims(boxes_values)
            else:
                boxes_values = [
                    self._bbox_query(bbox, variable, **kwargs) for bbox in query.boxes
                ]
                dims = parse_2d_dims(boxes_values[0], details=False)
                dims = f"boxes:{len(boxes_values)}, ({dims})"
            boxes_result = {"data": boxes_values, "dims": f"({dims})"}
        # parse polygons result
        polygons_result = None
        if query.polygons is not None:
            self._polygons_query(query.polygons, variable)

        return QueryResult(points_result, boxes_result, polygons_result, query)

    def sel(
        self,
        variable: str | None = None,
        **kwargs,
    ) -> xr.DataArray | xr.Dataset:
        """Select a variable from the dataset.

        This method is a wrapper of :meth:`xarray.Dataset.sel` or
        :meth:`xarray.DataArray.sel`.

        Parameters
        ----------
        variable : str, optional
            name of the variable to select. If None, the entire dataset will
            be selected.
        **kwargs : dict
            keyword arguments to pass to :meth:`xarray.Dataset.sel` or
            :meth:`xarray.DataArray.sel`.

        """
        with xr.open_dataset(self.path, group=self.group) as ds:
            return ds.sel(**kwargs) if variable is None else ds[variable].sel(**kwargs)

    def isel(
        self,
        variable: str | None = None,
        **kwargs,
    ) -> xr.DataArray | xr.Dataset:
        """Index a variable from the dataset.

        This method is a wrapper of :meth:`xarray.Dataset.isel` or
        :meth:`xarray.DataArray.isel`.

        Parameters
        ----------
        variable : str, optional
            name of the variable to index. If None, the entire dataset will be indexed.
        **kwargs : dict
            keyword arguments to pass to :meth:`xarray.Dataset.isel` or
            :meth:`xarray.DataArray.isel`.

        """
        with xr.open_dataset(self.path, group=self.group) as ds:
            if variable is None:
                data = ds.isel(**kwargs)
            else:
                data = ds[variable].isel(**kwargs)
        return data

    def set_crs(self, crs: CRS | str) -> None:
        """Set the CRS of the dataset.

        .. note::
            This method is used to set the CRS of the dataset if it is not
            specified in the dataset. If the CRS is already specified in the
            dataset, this method will overwrite the CRS.
        """
        self._crs = CRS.from_user_input(crs)
        self._bounds.set_crs(self._crs)

    @property
    def path(self) -> Path:
        """The path of the dataset."""
        return self._path

    @property
    def group(self) -> str:
        """The group of the dataset."""
        return self._group

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the dataset in (height, width)."""
        return self._shape

    @property
    def bounds(self) -> BoundingBox:
        """The bounds of the dataset."""
        return self._bound

    @property
    def lat(self) -> np.ndarray:
        """The latitudes of the dataset."""
        return self._lat

    @property
    def lon(self) -> np.ndarray:
        """The longitudes of the dataset."""
        return self._lon

    @property
    def variables(self) -> list[str]:
        """The variables of the dataset."""
        return self._variables

    def get_profile(
        self,
        bbox: Literal["roi", "bounds"] | BoundingBox = "roi",
    ) -> Profile | None:
        bbox = self._ensure_bbox(bbox)
        if bbox is None:
            return None
        profile = Profile.from_bounds_res(bbox, self.res)
        profile["crs"] = self.crs
        return profile

    def array2tiff(
        self,
        arr: np.ndarray,
        filename: str | Path,
        bounds: BoundingBox | None = None,
        bbox: BoundingBox | None = None,
        band_names: Sequence[str] | None = None,
        arr_type: Literal["data", "mask"] = "data",
        nodata: float | None = None,
        overwrite: bool = False,
    ) -> None:
        """Save a numpy array to a tiff file using the geoinformation of dataset.

        Parameters
        ----------
        arr : numpy.ndarray
            numpy array to save. arr can be a 2D array or a 3D array. If arr is a
            3D array, the first dimension should be the band dimension.
        filename : str or Path
            path to the tiff file to save
        bounds : BoundingBox, optional
            the bounds of the output dataset. Default is None, which means the
            roi of the dataset will be used.
        bbox : BoundingBox, optional
            if specified, the input array will be saved to the given part/bbox of
            dataset. Default is None, which means the array will be saved to the
            entire dataset.
        band_names : Sequence of str, optional
            names of bands to save. Default is None, which will use the band indexes.
        arr_type : str, one of ['data', 'mask'], optional
            type of the array to save. Default is 'data'.
        nodata : float or int, optional
            no data value of the dataset. If None, will automatically parse the
            a proper no data value for the array.
        overwrite : bool, optional
            if True, overwrite the existing file. Default is False, which means
            the array will be saved in append mode (r+ mode).

        """
        # check arr dimension
        if arr.ndim == 2:
            indexes = [1]
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            indexes = [i + 1 for i in range(arr.shape[0])]
        else:
            msg = (
                f"Expected arr to be an array with shape of (n_lat, n_lon) or "
                f"(n_band, n_lat, n_lon), got {arr.shape}"
            )
            raise ValueError(msg)
        # check length of band_names
        if band_names is not None and len(band_names) != arr.shape[0]:
            msg = (
                "Expected band_names to be of length "
                f"{arr.shape[0]}, got {len(band_names)}"
            )
            raise ValueError(msg)
        # parse profile
        if bounds is None:
            bounds = self.roi
        profile = self.get_profile(bounds)
        profile["count"] = arr.shape[0]
        profile["driver"] = "GTiff"
        profile["dtype"] = get_minimum_dtype(arr)
        if nodata is None:
            if np.issubdtype(arr.dtype, np.floating):
                nodata = np.nan
            else:
                rng = dtype_ranges[profile["dtype"]]
                nodata = rng[1] - 1 if np.any(arr == rng[0]) else rng[0]
        profile["nodata"] = nodata
        mode = "w"
        if Path(filename).exists() and not overwrite:
            mode = "r+"

        dst = rasterio.open(filename, mode, **profile)

        # parse whether to update band names
        desc = np.asarray(dst.descriptions, dtype="str")
        update_tags = False
        if band_names is not None and np.all(desc == "None"):
            update_tags = True

        # parse window
        win = None if bbox is None else dst.window(*bbox)

        # write array to tiff
        if arr_type == "mask":
            dst.write_mask(arr)
        elif arr_type == "data":
            dst.write(arr, indexes, window=win)
            if update_tags:
                for i, name in enumerate(band_names):
                    dst.update_tags(i + 1, NAME=name)
        dst.close()


class MultiHierarchicalDataset(GeoDataset):
    def __init__(self, paths: Sequence[str | Path], **kwargs) -> None:
        pass


class TimeSeriesDataset(RasterDataset, ABC):
    """A base class for time series datasets."""

    _dates: Acquisition

    @property
    def dates(self) -> Acquisition:
        """Return the date for each acquisition in the dataset."""
        return self._dates

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the dataset. *Must be implemented in subclass*."""
        super().__init__(*args, **kwargs)

    @classmethod
    @abstractmethod
    def _parse_dates(cls, paths: Iterable[str | PathLike]) -> Acquisition:
        """Parse dates from filenames. *Must be implemented in subclass*."""

    @classmethod
    def parse_dates(cls, paths: Iterable[str | PathLike]) -> Acquisition:
        """Parse dates from filenames.

        Parameters
        ----------
        paths : list of pathlib.Path
            list of file paths to parse dates

        Returns
        -------
        dates : Acquisition
            dates parsed from filenames

        """
        return cls._parse_dates(paths)

    def query(
        self,
        query: GeoQuery | Points | BoundingBox | Polygons,
        dates: Acquisition | pd.DatetimeIndex | None = None,
    ) -> QueryResult:
        """Retrieve images values for given query.

        This method is an more flexible implementation compared to
        :meth:`__getitem__`, which can retrieve images only for the given pairs.

        Parameters
        ----------
        query : GeoQuery | Points | BoundingBox | Polygons
            query to index the dataset. It can be :class:`Points`,
            :class:`BoundingBox`, :class:`Polygons`, or a composite
            :class:`GeoQuery` (recommended) object.
        dates : Acquisition | pd.DatetimeIndex, optional
            dates to use for the query. If None, all dates will be used.

        Returns
        -------
        result : QueryResult
            a QueryResult instance containing the results of the various queries.

        """
        if isinstance(query, Points):
            query = GeoQuery(points=query)
        if isinstance(query, BoundingBox):
            query = GeoQuery(boxes=query)
        if isinstance(query, Polygons):
            query = GeoQuery(polygons=query)

        mask = self.files.valid
        if dates is not None:
            mask = mask * np.isin(self.dates, dates)

        paths = self.files[mask].paths.tolist()
        return self._sample_files(paths, query)


class PairDataset(RasterDataset):
    """A base class for pair-like (contain two dates for one pair) datasets."""

    _pairs: Pairs

    @property
    def pairs(self) -> Pairs:
        """Return Pairs parsed from filenames."""
        return self._pairs

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the dataset. *Must be implemented in subclass*."""
        super().__init__(*args, **kwargs)

    @classmethod
    @abstractmethod
    def _parse_pairs(cls, paths: Iterable[str | PathLike]) -> Pairs:
        """Parse pairs from filenames. *Must be implemented in subclass*."""

    @classmethod
    def parse_pairs(cls, paths: Iterable[str | PathLike]) -> Pairs:
        """Parse pairs from filenames.

        Parameters
        ----------
        paths : list of str or PathLike
            list of file paths to parse pairs

        Returns
        -------
        pairs : Pairs object
            pairs parsed from filenames

        """
        return cls._parse_pairs(paths)

    def query(
        self,
        query: GeoQuery | Points | BoundingBox | Polygons,
        pairs: Pairs | None = None,
        parallel_loading: bool | None = None,
    ) -> QueryResult:
        """Retrieve images values for given query.

        This method is an more flexible implementation compared to
        :meth:`__getitem__`, which can retrieve images only for the given pairs.

        Parameters
        ----------
        query : GeoQuery | Points | BoundingBox | Polygons
            query to index the dataset. It can be :class:`Points`,
            :class:`BoundingBox`, :class:`Polygons`, or a composite
            :class:`GeoQuery` (recommended) object.
        pairs : Pairs, optional
            pairs to use for the query. If None, all pairs will be used.
        parallel_loading : bool or None, optional
            if True, use dask for lazy loading and parallel computation. If None,
            use the dataset's default parallel_loading setting. Default is None.

        Returns
        -------
        result : QueryResult
            a QueryResult instance containing the results of the various queries.

        """
        if isinstance(query, Points):
            query = GeoQuery(points=query)
        if isinstance(query, BoundingBox):
            query = GeoQuery(boxes=query)
        if isinstance(query, Polygons):
            query = GeoQuery(polygons=query)

        mask = self.files.valid
        if pairs is not None:
            mask = mask * self.pairs.where(pairs, return_type="mask")

        paths = self.files[mask].paths.tolist()
        return self._sample_files(paths, query, parallel_loading)


def get_nodata(
    arr: np.ndarray,
    nodata: float | None,
    dtype: str | np.dtype,
) -> float:
    """Get a proper no data value for the array."""
    if nodata is None:
        if np.issubdtype(arr.dtype, np.floating):
            nodata = np.nan
        else:
            rng = dtype_ranges[str(dtype)]
            nodata = rng[1] if np.any(arr == rng[0]) else rng[0] - 1
    return cast("float", nodata)


def parse_1d_dims(
    values_1d: np.ndarray,
    multi_files: bool = True,
) -> tuple[list[tuple[str, int]], np.ndarray]:
    """Parse the dimensions of 1D array. (used by points).

    Returns
    -------
    dims : list[tuple[str, int]]
        List of (dimension_name, size) tuples for easier analysis.
    values_1d : np.ndarray
        Potentially transposed array.

    """
    if multi_files:
        if values_1d.ndim == 2:
            n_files, n_points = values_1d.shape
            dims = [("files", n_files), ("points", n_points)]
        elif values_1d.ndim == 3:
            n_files, n_points, n_bands = values_1d.shape
            values_1d = values_1d.transpose(0, 2, 1)
            dims = [("files", n_files), ("bands", n_bands), ("points", n_points)]
        else:
            msg = f"values_1d must be 2D or 3D, got {values_1d.ndim}"
            raise ValueError(msg)
    elif values_1d.ndim == 1:
        n_points = values_1d.shape[0]
        dims = [("points", n_points)]
    elif values_1d.ndim == 2:
        n_points, n_bands = values_1d.shape
        values_1d = values_1d.T
        dims = [("bands", n_bands), ("points", n_points)]
    return dims, values_1d


def format_dims_as_string(dims: list[tuple[str, int]] | list[tuple[str, str]]) -> str:
    """Format dims list as string for backward compatibility.

    Parameters
    ----------
    dims : list[tuple[str, int | str]]
        List of (dimension_name, size) tuples.

    Returns
    -------
    str
        Formatted string like "files:2, height:10, width:10"

    """
    return ", ".join([f"{name}:{size}" for name, size in dims])


def parse_2d_dims(
    values_2d: np.ndarray,
    multi_files: bool = True,
) -> list[tuple[str, int]]:
    """Parse the dimensions of 2D array. (used by bbox, polygons).

    Returns
    -------
    dims : list[tuple[str, int]]
        List of (dimension_name, size) tuples for easier analysis.

    """
    if multi_files:
        if values_2d.ndim == 4:
            n_files, n_bands, height, width = values_2d.shape
            dims = [
                ("files", n_files),
                ("bands", n_bands),
                ("height", height),
                ("width", width),
            ]
        elif values_2d.ndim == 3:
            n_files, height, width = values_2d.shape
            dims = [("files", n_files), ("height", height), ("width", width)]
        else:
            msg = f"values_2d must be 3D or 4D, got {values_2d.ndim}"
            raise ValueError(msg)
    elif values_2d.ndim == 3:
        n_bands, height, width = values_2d.shape
        values_2d = values_2d.transpose(1, 2, 0)
        dims = [("bands", n_bands), ("height", height), ("width", width)]
    elif values_2d.ndim == 2:
        height, width = values_2d.shape
        dims = [("height", height), ("width", width)]
    else:
        msg = f"values_2d must be 2D or 3D, got {values_2d.ndim}"
        raise ValueError(msg)
    return dims


def ensure_geo_query(query: GeoQuery | Points | BoundingBox | Polygons) -> GeoQuery:
    """Ensure the query is a GeoQuery object.

    Parameters
    ----------
    query : GeoQuery | Points | BoundingBox | Polygons
        query to ensure

    Returns
    -------
    query : GeoQuery
        the query as a GeoQuery object

    """
    if isinstance(query, GeoQuery):
        return query
    if isinstance(query, Points):
        query = GeoQuery(points=query)
    if isinstance(query, BoundingBox):
        query = GeoQuery(boxes=query)
    if isinstance(query, Polygons):
        query = GeoQuery(polygons=query)
    return query
