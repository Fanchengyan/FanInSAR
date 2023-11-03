"""Base classes for all :mod:`faninsar` datasets. all classes in this script are modified from the torchgeo package."""

import abc
import functools
import glob
import os
import re
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union, overload

import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.merge
import rioxarray
import shapely
import xarray as xr
from rasterio.coords import BoundingBox as BBox
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.transform import rowcol, xy
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.warp import transform as warp_transform
from rtree.index import Index, Property
from shapely import ops
from tqdm import tqdm

from faninsar._core import geo_tools
from faninsar._core.geo_tools import Profile
from faninsar._core.pair_tools import Pairs
from faninsar.query.query import BoundingBox, GeoQuery, Points

__all__ = ("GeoDataset", "RasterDataset")


class GeoDataset(abc.ABC):
    """Abstract base class for all :mod:`faninsar` datasets. This class is used
    to represent a geospatial dataset and provides methods to index the dataset
    and retrieve information about the dataset, such as CRS, resolution, data type,
    no data value, and a bounds.

    .. Note::

        Although this :class:`GeoDataset` class is based on the ``GeoDataset``
        class in the torchgeo package, it has been extensively modified.  When
        using this :class:`GeoDataset` class, you should not make any assumptions
        based on the torchgeo version.
    """

    # following attributes should be set by the subclass
    _crs: Optional[CRS] = None
    _res: tuple[float, float] = (0.0, 0.0)
    _dtype: Optional[np.dtype] = None
    _count: int = 0
    _roi: Optional[BoundingBox] = None
    _nodata: Optional[Union[float, int, Any]] = None
    _valid: np.ndarray

    def __init__(self):
        self.index = Index(interleaved=True, properties=Property(dimension=2))

    def __repr__(self):
        return f"""\
{self.__class__.__name__} Dataset
    bbox: {self.bounds} 
    file count: {len(self)}"""

    def __str__(self):
        return self.__repr__()

    def __len__(self) -> int:
        """Return the number of files in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.index)

    def __getstate__(
        self,
    ) -> tuple[dict[str, Any], list[tuple[Any, Any, Optional[Any]]]]:
        """Define how instances are pickled.

        Returns:
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
            state: the state of the instance when it was pickled
        """
        attrs, tuples = state
        self.__dict__.update(attrs)
        for item in tuples:
            self.index.insert(*item)

    @property
    def crs(self) -> Optional[CRS]:
        """coordinate reference system (:term:`CRS`) of the dataset.

        Returns:
            The coordinate reference system (:term:`CRS`).
        """
        return self._crs

    @crs.setter
    def crs(self, new_crs: Union[CRS, str]) -> None:
        """Change the coordinate reference system :term:`(CRS)` of a GeoDataset.

        If ``new_crs == self.crs``, does nothing, otherwise updates the R-tree index.

        Parameters
        ----------
        new_crs: CRS or str
            New coordinate reference system :term:`(CRS)`. It can be a CRS object
            or a string, which will be parsed to a CRS object. The string can be
            in any format supported by `rasterio.crs.CRS.from_user_input()
            <https://rasterio.readthedocs.io/en/stable/api/rasterio.crs.html#rasterio.crs.CRS.from_user_input>`_.
        """
        if isinstance(new_crs, str):
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
                print(
                    f"Warning: the resolution of the dataset has been changed from {self.res} to {new_res}."
                )
                self.res = new_res

            # reproject the index
            new_index = Index(interleaved=True, properties=Property(dimension=2))
            project = pyproj.Transformer.from_crs(
                pyproj.CRS(str(self.crs)), pyproj.CRS(str(new_crs)), always_xy=True
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
    def res(self) -> tuple[float, float]:
        """Return the resolution of the dataset.

        Returns
        -------
        res: tuple of floats
            resolution of the dataset in x and y directions.
        """
        return self._res

    @res.setter
    def res(self, new_res: Union[float, tuple[float, float]]) -> None:
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
            raise ValueError(
                f"Resolution must be a float or a tuple of length 2, got {new_res}"
            )
        if not all(isinstance(i, float) for i in new_res):
            try:
                new_res = tuple(float(i) for i in new_res)
            except ValueError:
                raise ValueError("Resolution must be a float or a tuple of floats")
        self._res = new_res

    @property
    def roi(self) -> Optional[BoundingBox]:
        """Return the region of interest of the dataset.

        Returns
        -------
        roi: BoundingBox object
            region of interest of the dataset. If None, the bounds of
            entire dataset will be used.
        """
        if self._roi:
            return self._roi
        else:
            return self.bounds

    @roi.setter
    def roi(self, new_roi: Optional[BoundingBox]):
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

    def _check_roi(self, roi: Optional[BoundingBox]) -> BoundingBox:
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
        else:
            if not isinstance(roi, BoundingBox):
                raise TypeError(
                    f"roi must be a BoundingBox object, got {type(roi)} instead."
                )
            if roi.crs != self.crs:
                if roi.crs is None:
                    roi = BoundingBox(*roi, crs=self.crs)
                else:
                    roi = roi.to_crs(self.crs)
            return roi

    @property
    def dtype(self) -> Optional[np.dtype]:
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
    def nodata(self) -> Optional[Union[float, int, Any]]:
        """No data value of the dataset.

        Returns
        -------
        nodata: float or int
            no data value of the dataset
        """
        return self._nodata

    @nodata.setter
    def nodata(self, new_nodata: Union[float, int, Any]) -> None:
        """Set the no data value of the dataset.

        Parameters
        ----------
        new_nodata : float or int
            no data value of the dataset
        """
        self._nodata = new_nodata

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
    def bounds(self) -> BoundingBox:
        """Bounds of the overall dataset. It is the union of all the files in the
        dataset.

        Returns
        -------
        bounds: BoundingBox object
            (minx, right, bottom, top) of the dataset
        """
        return BoundingBox(*self.index.bounds, crs=self.crs)

    def get_profile(
        self, bbox: Union[BoundingBox, Literal["roi", "bounds"]] = "roi"
    ) -> Optional[Profile]:
        """Return the profile information of the dataset for the given bounding
        box type. The profile information includes the width, height, transform,
        count, data type, no data value, and CRS of the dataset.

        Parameters
        ----------
        bbox : str, one of ['bounds', 'roi'], optional
            the bounding box used to calculate the ``width``, ``height``
            and ``transform`` of the dataset for the profile. Default is
            'roi'.

        Returns
        -------
        profile: Profile object or None
            profile of the dataset for the given bounding box type.
        """
        if bbox == "bounds":
            if self.bounds is None:
                return None
            else:
                bounds = self.bounds
        elif bbox == "roi":
            if self.roi is None:
                return None
            else:
                bounds = self.roi
        elif isinstance(bbox, BoundingBox):
            bounds = bbox
        else:
            raise ValueError(
                "bbox must be one of ['bounds', 'roi'] or a "
                f"BoundingBox, but got {bbox}"
            )

        profile = Profile.from_bounds_res(bounds, self.res)
        profile["count"] = self.count
        profile["dtype"] = self.dtype
        profile["nodata"] = self.nodata
        profile["crs"] = self.crs

        return profile


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
    >>> query = GeoQuery(points=points, bbox=[ds.bounds, ds.bounds])

    use the GeoQuery object to index the RasterDataset

    >>> sample = ds[query]

    output the samples shapes:

    >>> print('bbox shape:', sample['bbox'].shape)
    bbox shape: (2, 7, 68, 80)

    >>> print('points shape:', sample['points'].shape)
    points shape: (7, 3)

    of course, you can also use the BoundingBox or Points directly to index the
    RasterDataset. Those two types will be automatically converted to GeoQuery
    object.

    >>> sample = ds[points]
    >>> sample
    {'query': GeoQuery(
        bbox=None
        points=Points(count=3)
    ),
    'bbox': None,
    'points': array([...], dtype=float32)}

    >>> sample = ds[ds.bounds]
    query': GeoQuery(
        bbox=[1 BoundingBox]
        points=None
    ),
    'bbox': array([...], dtype=float32),
    'points': None}
    """

    #: Glob expression used to search for files.
    #:
    #: This expression should be specific enough that it will not pick up files from
    #: other datasets. It should not include a file extension, as the dataset may be in
    #: a different file format than what it was originally downloaded as.
    filename_glob = "*"

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
    all_bands: list[str] = []

    #: Names of RGB bands in the dataset, used for plotting
    rgb_bands: list[str] = []

    #: Color map for the dataset, used for plotting
    cmap: dict[int, tuple[int, int, int, int]] = {}

    def __init__(
        self,
        root_dir: str = "data",
        paths: Optional[Sequence[str]] = None,
        crs: Optional[CRS] = None,
        res: Optional[Union[float, tuple[float, float]]] = None,
        dtype: Optional[np.dtype] = None,
        nodata: Optional[Union[float, int, Any]] = None,
        roi: Optional[BoundingBox] = None,
        bands: Optional[Sequence[str]] = None,
        cache: bool = True,
        resampling=Resampling.nearest,
        verbose: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.

        Parameters
        ----------
        root_dir : str or Path
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
            no data value of the output dataset. If None, the no data value of the first
            file found will be used.
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
        verbose : bool, optional, default: True
            if True, print verbose output

        Raises
        ------
            FileNotFoundError: if no files are found in ``root_dir``
        """
        super().__init__()
        self.root_dir = root_dir
        self.bands = bands or self.all_bands
        self.cache = cache
        self.resampling = resampling
        self.verbose = verbose

        if paths is None:
            paths = []
            pathname = os.path.join(root_dir, "**", self.filename_glob)
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for file_path in glob.iglob(pathname, recursive=True):
                match = re.match(filename_regex, os.path.basename(file_path))
                if match is not None:
                    paths.append(file_path)

        # Populate the dataset index
        count = 0
        files_valid = []
        for file_path in paths:
            try:
                with rasterio.open(file_path) as src:
                    # See if file has a color map
                    if len(self.cmap) == 0:
                        try:
                            self.cmap = src.colormap(1)
                        except ValueError:
                            pass

                    if crs is None:
                        crs = src.crs
                    if res is None:
                        res = src.res
                    if dtype is None:
                        dtype = src.dtypes[0]
                    if nodata is None:
                        nodata = src.nodata

                    with WarpedVRT(src, crs=crs) as vrt:
                        coords = tuple(vrt.bounds)
            except rasterio.errors.RasterioIOError:
                # Skip files that rasterio is unable to read
                files_valid.append(False)
                continue
            else:
                self.index.insert(count, coords, file_path)
                files_valid.append(True)
                count += 1

        if count == 0:
            msg = f"No {self.__class__.__name__} data was found in `root_dir='{self.root_dir}'`"
            if self.bands:
                msg += f" with `bands={self.bands}`"
            raise FileNotFoundError(msg)

        self._files = pd.DataFrame({"paths": paths, "valid": files_valid})
        self._valid = np.array(files_valid)

        if not self._files.valid.all():
            files_invalid = self._files.paths[~self._files.valid].tolist()
            print(
                f"Unable to read {len(files_invalid)} files in {self.__class__.__name__} dataset:"
                "\n\t" + "\n\t".join(files_invalid),
            )

        self.band_indexes = None
        if self.bands:
            if self.all_bands:
                self.band_indexes = [self.all_bands.index(i) + 1 for i in self.bands]
            else:
                msg = (
                    f"{self.__class__.__name__} is missing an `all_bands` "
                    "attribute, so `bands` cannot be specified."
                )
                raise AssertionError(msg)

        self.crs = crs
        self.res = res
        self.dtype = dtype
        self.nodata = nodata
        self.count = count
        self.roi = roi

    def __getitem__(
        self, query: Union[GeoQuery, BoundingBox, Points]
    ) -> dict[str, np.ndarray]:
        """Retrieve image values for given query.

        Parameters
        ----------
        query : Union[GeoQuery, BoundingBox, Points]
            query to index the dataset. It can be a :class:`BoundingBox` or a
            :class:`Points` or a :class:`GeoQuery` (recommended) object.

        Returns
        -------
        sample : dict
            a dictionary containing the image values, corresponding query.
            The keys of the dictionary are:

            * ``query``: a GeoQuery object, the query used to index the dataset
            * ``bbox``: a numpy array of shape (n_bbox, n_file, height, width)
                containing the values of the bounding boxes in the query.
            * ``points``: a numpy array of shape (n_file, n_point) containing the
                values of the points in the query.

            .. Note::

                ``bbox`` and ``points`` will be None if the query does not contain
                bounding boxes or points.
        """
        if isinstance(query, BoundingBox):
            query = GeoQuery(bbox=query)
        if isinstance(query, Points):
            query = GeoQuery(points=query)

        paths = self.files[self.files.valid].paths
        data = self._sample_files(paths, query)

        sample = {"query": query}
        sample.update(data)

        return sample

    def _bbox_query(self, bbox: BoundingBox, vrt_fh) -> np.ndarray:
        """Return the index of files that intersect with the given bounding box."""
        if bbox.crs is None:
            warnings.warn(
                "No CRS is specified for the bounding box, assuming it is in the"
                " same CRS as the dataset."
            )
        elif bbox.crs != self.crs:
            bbox = bbox.to_crs(self.crs)
            
        win = vrt_fh.window(*bbox)
        data = vrt_fh.read(
            out_shape=(
                1,
                round((bbox.top - bbox.bottom) / self.res[1]),
                round((bbox.right - bbox.left) / self.res[0]),
            ),
            resampling=self.resampling,
            indexes=self.band_indexes,
            window=win,
            boundless=True,
            fill_value=self.nodata,
        )

        return data

    def _points_query(self, points: Points, vrt_fh) -> np.ndarray:
        """Return the index of files that intersect with the given points."""
        crs_points = points.crs
        xy = points.values
        if crs_points is None:
            warnings.warn(
                "No CRS is specified for the points, assuming they are in the"
                " same CRS as the dataset."
            )
        else:
            if crs_points != self.crs:
                xs, ys = warp_transform(crs_points, self.crs, xy[:, 0], xy[:, 1])
                xy = np.column_stack((xs, ys))
        data = list(vrt_fh.sample(xy))
        return data

    def _sample_files(
        self, paths: Sequence[str], query: GeoQuery
    ) -> dict[Union[Literal["bbox", "points"], np.ndarray]]:
        """Stack files into a single 3D array.

        Parameters
        ----------
        paths : list of str
            list of paths for files to stack
        query : GeoQuery
            a GeoQuery object containing the BoundingBox(es) and/or Points object

        Returns
        -------
        dest : dict
            a dictionary containing the stacked files. The keys of the dictionary
            are:

            * ``bbox``: a numpy array of shape (n_bbox, n_file, height, width)
                containing the values of the bounding boxes in the query.
            * ``points``: a numpy array of shape (n_file, n_point) containing the
                values of the points in the query.

            .. Note::

                ``bbox`` and ``points`` will be None if the query does not contain
                bounding boxes or points.
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in paths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in paths]

        if self.verbose:
            vrt_fhs = tqdm(vrt_fhs, desc="Loading files", unit=" files")

        files_bbox_list = []
        files_points_list = []
        for vrt_fh in vrt_fhs:
            # Get the bounding boxes values
            bbox_list = []
            if query.bbox is not None:
                for bbox in query.bbox:
                    data = self._bbox_query(bbox, vrt_fh)
                    bbox_list.append(data)
                files_bbox_list.append(bbox_list)
            # Get the points values
            if query.points is not None:
                data = self._points_query(query.points, vrt_fh)
                files_points_list.append(data)

        # Stack the bounding boxes values
        bbox_values = None
        if len(files_bbox_list) > 0:
            bbox_values = (
                np.asarray(files_bbox_list).squeeze(axis=2).transpose(1, 0, 2, 3)
            )

        # Stack the points values
        points_values = None
        if len(files_points_list) > 0:
            points_values = np.asarray(files_points_list).squeeze()

        dest = {"bbox": bbox_values, "points": points_values}
        return dest

    @functools.lru_cache(maxsize=128)
    def _cached_load_warp_file(self, file_path: str) -> DatasetReader:
        """Cached version of :meth:`_load_warp_file`.

        Args:
            file_path: file to load and warp

        Returns:
            file handle of warped VRT
        """
        return self._load_warp_file(file_path)

    def _load_warp_file(self, file_path: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            file_path: file to load and warp

        Returns:
            file handle of warped VRT
        """
        src = rasterio.open(file_path)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        else:
            return src

    @property
    def files(self) -> pd.DataFrame:
        """Return a list of all files in the dataset.

        Returns:
            list of all files in the dataset
        """
        return self._files

    def row_col(
        self,
        xy: Iterable,
        crs: Optional[Union[CRS, str]] = None,
    ) -> np.ndarray:
        """Convert x, y coordinates to row, col in the dataset.

        Parameters
        ----------
        xy: Iterable
            Pairs of x, y coordinates (floats)
        crs: CRS or str, optional
            The CRS of the points. If None, the CRS of the dataset will be used.
            allowed CRS formats are the same as those supported by rasterio.

        Returns
        -------
        row_col: np.ndarray
            row, col in the dataset for the given points(xy)
        """
        xy = np.asarray(xy)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(
                f"Expected xy to be an array of shape (n, 2), got {xy.shape}"
            )
        if crs is not None:
            crs = CRS.from_user_input(crs)
            if crs != self.crs:
                xs, ys = warp_transform(crs, self.crs, xy[:, 0], xy[:, 1])
                xy = np.column_stack((xs, ys))

        profile = self.get_profile("bounds")

        rows, cols = rowcol(profile["transform"], xy[:, 0], xy[:, 1])
        row_col = np.column_stack((rows, cols))
        return row_col

    def xy(
        self,
        row_col: Iterable,
        crs: Optional[Union[CRS, str]] = None,
    ) -> np.ndarray:
        """Convert row, col in the dataset to x, y coordinates.

        Parameters
        ----------
        row_col: Iterable
            Pairs of row, col in the dataset (floats)
        crs: CRS or str, optional
            The CRS of the points. If None, the CRS of the dataset will be used.
            allowed CRS formats are the same as those supported by rasterio.

        Returns
        -------
        xy: np.ndarray
            x, y coordinates in the given CRS (default is the CRS of the dataset)
        """
        row_col = np.asarray(row_col)
        if row_col.ndim == 1:
            row_col = row_col.reshape(1, -1)
        if row_col.ndim != 2 or row_col.shape[1] != 2:
            raise ValueError(
                f"Expected row_col to be an array of shape (n, 2), got {row_col.shape}"
            )

        profile = self.get_profile("bounds")

        xs, ys = xy(profile["transform"], row_col[:, 0], row_col[:, 1])

        if crs is not None:
            crs = CRS.from_user_input(crs)
            if crs != self.crs:
                xs, ys = warp_transform(crs, self.crs, xy[:, 0], xy[:, 1])
        xy = np.column_stack((xs, ys))

        return xy

    def to_tiffs(
        self,
        out_dir: Union[str, Path],
        roi: Optional[BoundingBox] = None,
    ):
        """Save the dataset to a directory of tiff files for given region of interest.

        Parameters
        ----------
        out_dir : str or Path
            path to the directory to save the tiff files
        roi : BoundingBox, optional
            region of interest to save. If None, the roi of the dataset will be used.
        """
        roi = self._check_roi(roi)

        profile = self.get_profile(roi)
        profile["count"] = 1

        for f in self.files.paths[self.valid]:
            out_file = Path(out_dir) / f.name
            with rasterio.open(f) as src:
                dest_arr = self._bbox_query(roi, src).squeeze(0)
            with rasterio.open(out_file, "w", **profile.profile) as dst:
                dst.write(dest_arr, 1)

    def to_netcdf(
        self,
        filename: Union[str, Path],
        roi: Optional[BoundingBox] = None,
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
            {"image": (["band", "lat", "lon"], sample["bbox"])},
            coords={
                "band": list(range(profile["count"])),
                "lat": lat,
                "lon": lon,
            },
        )
        ds = geo_tools.write_geoinfo_into_ds(
            ds, "image", crs=self.crs, x_dim="lon", y_dim="lat"
        )
        ds.to_netcdf(filename)


class InterferogramDataset(RasterDataset):
    """
    A base class for interferogram datasets.

    .. Note::
        The unwrapped interferograms are used to initialize this dataset. The coherence, dem, and mask files can be accessed as attributes ``ds_coh``, ``ds_dem``, and ``ds_mask`` respectively.
    """

    #: Glob expression used to search for files.
    #:
    #: This expression is used to find the interferogram files.
    filename_glob_unw = "*"

    #: This expression is used to find the coherence files.
    filename_glob_coh = "*"

    _pairs: Optional[Pairs] = None

    def __init__(
        self,
        root_dir: str = "data",
        paths_unw: Optional[Sequence[str]] = None,
        paths_coh: Optional[Sequence[str]] = None,
        dem_file: Optional[Union[str, Path]] = None,
        mask_file: Optional[Union[str, Path]] = None,
        crs: Optional[CRS] = None,
        res: Optional[Union[float, tuple[float, float]]] = None,
        dtype: Optional[np.dtype] = None,
        nodata: Optional[Union[float, int, Any]] = None,
        roi: Optional[BoundingBox] = None,
        bands: Optional[Sequence[str]] = None,
        cache: bool = True,
        resampling=Resampling.nearest,
        verbose=True,
    ) -> None:
        """Initialize a new InterferogramDataset instance.

        Parameters
        ----------
        root_dir: str
            root_dir directory where dataset can be found.
        paths_unw: list of str, optional
            list of unwrapped interferogram file paths to use instead of searching
            for files in ``root_dir``. If None, files will be searched for in ``root_dir``.
        paths_coh: list of str, optional
            list of coherence file paths to use instead of searching for files in
            ``root_dir``. If None, files will be searched for in ``root_dir``.
        dem_file: str or Path, optional
            path to the DEM file. If None, no DEM data will be used.
        mask_file: str or Path, optional
            path to the mask file. If None, no Mask data will be used.
        crs: CRS, optional
            the output coordinate reference system term:`(CRS)` of the dataset.
            If None, the CRS of the first file found will be used.
        res: float, optional
            resolution of the output dataset in units of CRS. If None, the resolution
            of the first file found will be used.
        dtype: numpy.dtype, optional
            data type of the output dataset. If None, the data type of the first
            file found will be used.
        nodata: float or int, optional
            no data value of the output dataset. If None, the no data value of the
            first file found will be used.
        roi: BoundingBox, optional
            region of interest to load from the dataset. If None, the union of all
            files bounds in the dataset will be used.
        bands: list of str, optional
            names of bands to return (defaults to all bands)
        cache: bool, optional
            if True, cache file handle to speed up repeated sampling
        resampling: Resampling, optional
            Resampling algorithm used when reading input files.
            Default: `Resampling.nearest`.
        verbose: bool, optional, default: True
            if True, print verbose output.
        """
        root_dir_dir = Path(root_dir)
        self.root_dir_dir = root_dir_dir

        if paths_unw is None:
            paths_unw = np.unique(list(root_dir_dir.rglob(self.filename_glob_unw)))
        if paths_coh is None:
            paths_coh = np.unique(list(root_dir_dir.rglob(self.filename_glob_coh)))

        if len(paths_unw) != len(paths_coh):
            raise ValueError(
                f"Number of interferogram files ({len(paths_unw)}) does not match "
                f"number of coherence files ({len(paths_coh)})"
            )

        # ensure there are no duplicate pairs
        pairs = self.parse_pairs(paths_unw)
        pairs1, index = pairs.sort(inplace=False)
        if len(index) < len(paths_unw):
            deduplicated = "".join(
                [f"\n\t{i.parent.stem}" for i in set(paths_unw) - set(paths_unw[index])]
            )
            warnings.warn(
                f"Duplicate pairs found in dataset, keeping only the first occurrence"
                f"\nDeduplicate pairs: {deduplicated}"
            )
            paths_unw = paths_unw[index]
            paths_coh = paths_coh[index]

        self._pairs = pairs1

        super().__init__(
            root_dir=root_dir,
            paths=paths_unw,
            crs=crs,
            res=res,
            dtype=dtype,
            nodata=nodata,
            roi=roi,
            bands=bands,
            cache=cache,
            resampling=resampling,
            verbose=verbose,
        )

        self.ds_coh = RasterDataset(
            root_dir=root_dir,
            paths=paths_coh,
            crs=self.crs,
            res=self.res,
            dtype=self.dtype,
            nodata=self.nodata,
            roi=self.roi,
            bands=bands,
            cache=cache,
            resampling=resampling,
            verbose=verbose,
        )

        self.ds_dem = None
        if dem_file is not None:
            self.ds_dem = RasterDataset(
                paths=[dem_file],
                crs=self.crs,
                res=self.res,
                dtype=self.dtype,
                nodata=self.nodata,
                roi=self.roi,
                cache=cache,
                resampling=resampling,
                verbose=verbose,
            )

        self.ds_mask = None
        if mask_file is not None:
            self.ds_mask = RasterDataset(
                paths=[mask_file],
                crs=self.crs,
                res=self.res,
                dtype=self.dtype,
                nodata=self.nodata,
                roi=self.roi,
                cache=cache,
                resampling=resampling,
                verbose=verbose,
            )

    def parse_pairs(self, paths: list[Path]) -> Pairs:
        """Used to parse pairs from filenames. Must be implemented in subclass.

        Parameters
        ----------
        paths : list of pathlib.Path
            list of file paths to parse pairs

        Example
        -------
        for the HyP3 dataset, pairs are parsed from the filenames as follows:

        >>> names = [f.name for f in paths]]
        >>> pair_names = ['_'.join(i.split("_")[1:3]) for i in names]

        for the HyP3 dataset, the pair names are the second and third parts of the
        filename, separated by an underscore. After parsing the pair names, the
        :class:`Pairs` object can be created by using the ``from_names`` method.

        >>> pairs = Pairs.from_names(pair_names)

        .. Note::

            * The ``parse_pairs`` method must be implemented in subclass. If you are
              using :class:`InterferogramDataset` directly, you must implement the
              `parse_pairs` method in your code.
            * The ``parse_pairs`` method must return a :class:`Pairs` object.

        Raises
        ------
        NotImplementedError: if not implemented in subclass or directly using
            InterferogramDataset.
        """
        raise NotImplementedError("parse_pairs must be implemented in subclass")

    @property
    def pairs(self) -> Pairs:
        """Return Pairs parsed from filenames."""
        return self._pairs

    def to_netcdf(
        self,
        filename: Union[str, Path],
        roi: Optional[BoundingBox] = None,
        ref_points: Optional[Points] = None,
    ) -> None:
        """Save the dataset to a netCDF file for given region of interest.

        Parameters
        ----------
        filename : str
            path to the netCDF file to save
        roi : BoundingBox, optional
            region of interest to save. If None, the roi of the dataset will be
            used.
        ref_points : Points, optional, default: None
            reference points to save. If None, will keep the original values.
        """
        if roi is None:
            roi = self.roi

        # TODO: 1. add dem and mask 2. using netcdf4 to save the data to avoid the memory issue
        profile = self.get_profile(roi)
        lat, lon = profile.to_latlon()

        query = GeoQuery(bbox=roi, points=ref_points)

        sample_unw = self[query]
        sample_coh = self.coh_dataset[query]

        if ref_points is None:
            unw = sample_unw["bbox"][0]
        else:
            ref_mean = np.nanmean(sample_unw["points"], axis=1)
            unw = sample_unw["bbox"][0] - ref_mean[:, None, None]

        ds = xr.Dataset(
            {
                "unw": (["pair", "lat", "lon"], unw),
                "coh": (["pair", "lat", "lon"], sample_coh["bbox"][0]),
            },
            coords={
                "pair": self.pairs.to_names(),
                "lat": lat,
                "lon": lon,
            },
        )

        ds = geo_tools.write_geoinfo_into_ds(
            ds, ["unw", "coh"], crs=self.crs, x_dim="lon", y_dim="lat"
        )
        ds.to_netcdf(filename)
