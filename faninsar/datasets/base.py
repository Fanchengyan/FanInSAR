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
import shapely
import xarray as xr
from faninsar._core import geo_tools
from faninsar._core.geo_tools import Profile
from faninsar.datasets.query import BoundingBox, GeoQuery, Points
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
        """:term:`coordinate reference system (CRS)` of the dataset.

        Returns:
            The :term:`coordinate reference system (CRS)`.
        """
        return self._crs

    @crs.setter
    def crs(self, new_crs: Union[CRS, str]) -> None:
        """Change the :term:`coordinate reference system (CRS)` of a GeoDataset.

        If ``new_crs == self.crs``, does nothing, otherwise updates the R-tree index.

        Parameters
        ----------
        new_crs: CRS or str
            New :term:`coordinate reference system (CRS)`. It can be a CRS object
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
            return BoundingBox(*self._roi)
        else:
            return self.bounds

    @roi.setter
    def roi(self, new_roi: Union[BoundingBox, Iterable[float]]):
        """Set the region of interest of the dataset.

        Parameters
        ----------
        new_roi : BoundingBox or Iterable
            region of interest of the dataset in the CRS of the dataset.
        """
        if new_roi is None:
            self._roi = None
            return
        if isinstance(new_roi, Iterable):
            new_roi = np.asarray([i for i in new_roi])

        if len(new_roi) != 4:
            raise ValueError(f"ROI must be a tuple of length 4, got {len(new_roi)}")
        if (
            new_roi[0] < self.bounds[0]
            or new_roi[1] < self.bounds[1]
            or new_roi[2] > self.bounds[2]
            or new_roi[3] > self.bounds[3]
        ):
            raise ValueError(
                f"ROI must be within the dataset bounds: {self.bounds}, but got {BoundingBox(*new_roi)}"
            )
        self._roi = BoundingBox(*new_roi)

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
        return BoundingBox(*self.index.bounds)

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
    >>> from faninsar.datasets import BoundingBox, GeoQuery, Points, RasterDataset
    >>> home_dir = Path("./work/data")
    >>> files = list(home_dir.rglob("*unw_phase.tif"))

    initialize a RasterDataset and GeoQuery object

    >>> ds = RasterDataset(file_paths=files)
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
        root: str = "data",
        file_paths: Optional[Sequence[str]] = None,
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
        root : str or Path
            Root directory where dataset can be found.
        file_paths : list of str, optional
            list of file paths to use instead of searching for files in ``root``.
            If None, files will be searched for in ``root``.
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
            FileNotFoundError: if no files are found in ``root``
        """
        super().__init__()
        self.root = root
        self.bands = bands or self.all_bands
        self.cache = cache
        self.resampling = resampling
        self.verbose = verbose

        if file_paths is None:
            file_paths = []
            pathname = os.path.join(root, "**", self.filename_glob)
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for file_path in glob.iglob(pathname, recursive=True):
                match = re.match(filename_regex, os.path.basename(file_path))
                if match is not None:
                    file_paths.append(file_path)

        # Populate the dataset index
        count = 0
        files_valid = []
        for file_path in file_paths:
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
            msg = f"No {self.__class__.__name__} data was found in `root='{self.root}'`"
            if self.bands:
                msg += f" with `bands={self.bands}`"
            raise FileNotFoundError(msg)

        self._files = pd.DataFrame({"file_paths": file_paths, "valid": files_valid})
        if not self._files.valid.all():
            files_invalid = self._files.file_paths[~self._files.valid].tolist()
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
        self.roi = roi
        self.dtype = dtype
        self.nodata = nodata
        self.count = count

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

        file_paths = self.files[self.files.valid].file_paths
        data = self._sample_files(file_paths, query)

        sample = {"query": query}
        sample.update(data)

        return sample

    def _sample_files(
        self, file_paths: Sequence[str], query: GeoQuery
    ) -> dict[Union[Literal["bbox", "points"], np.ndarray]]:
        """Stack files into a single 3D array.

        Parameters
        ----------
        file_paths : list of str
            list of paths for files to stack
        query : GeoQuery
            a GeoQuery object containing the BoundingBox(es) and/or Points object

        Returns
        -------
        dest : numpy.ndarray
            stacked array
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in file_paths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in file_paths]

        if self.verbose:
            vrt_fhs = tqdm(vrt_fhs, desc="Loading files", unit=" files")

        files_bbox_list = []
        files_points_list = []
        for vrt_fh in vrt_fhs:
            # Get the bounding boxes values
            bbox_list = []
            if query.bbox is not None:
                for bbox in query.bbox:
                    win = vrt_fh.window(*bbox)
                    data = vrt_fh.read(
                        out_shape=(
                            1,
                            int((bbox.top - bbox.bottom) / self.res[1]),
                            int((bbox.right - bbox.left) / self.res[0]),
                        ),
                        resampling=self.resampling,
                        indexes=self.band_indexes,
                        window=win,
                        boundless=True,
                        fill_value=self.nodata,
                    )
                    bbox_list.append(data)

                files_bbox_list.append(bbox_list)
            # Get the points values
            if query.points is not None:
                crs_points = query.points.crs
                xy = query.points.values
                if crs_points is None:
                    warnings.warn(
                        "No CRS is specified for the points, assuming they are in the same CRS as the dataset."
                    )
                else:
                    if crs_points != self.crs:
                        xs, ys = warp_transform(
                            crs_points, self.crs, xy[:, 0], xy[:, 1]
                        )
                        xy = np.column_stack((xs, ys))
                files_points_list.append(list(vrt_fh.sample(xy)))

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

    def to_netcdf(
        self,
        filename: str,
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
            {"image": (["band", "lat", "lon"], sample["image"])},
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
    A base class for interferogram datasets. The unwrapped interferogram will be
    used as this dataset's ``image`` data. The ``coherence``, ``dem`` and ``mask``
    data are optional and can be accessed as attributes of the dataset.
    """

    #: Glob expression used to search for files.
    #:
    #: This expression is used to find the interferogram files.
    filename_glob_unw = "*"

    #: This expression is used to find the coherence files.
    filename_glob_coh = "*"

    #: Date format string used to parse date from filename.
    #:
    #: Not used if :attr:`filename_regex` does not contain a ``date`` group.
    date_format = "%Y%m%d"

    #: True if dataset contains imagery, False if dataset contains mask
    is_image = True

    #: Color map for the dataset, used for plotting
    cmap: dict[int, tuple[int, int, int, int]] = {}

    def __init__(
        self,
        root: str = "data",
        file_paths_unw: Optional[Sequence[str]] = None,
        file_paths_coh: Optional[Sequence[str]] = None,
        dem: Optional[Any] = None,
        mask: Optional[Any] = None,
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
        # TODO: add dem and mask support

        Parameters
        ----------
        root: str
            Root directory where dataset can be found.
        file_paths_unw: list of str, optional
            list of unwrapped interferogram file paths to use instead of searching
            for files in ``root``. If None, files will be searched for in ``root``.
        file_paths_coh: list of str, optional
            list of coherence file paths to use instead of searching for files in
            ``root``. If None, files will be searched for in ``root``.
        dem: Any, optional
            DEM data. If None, no DEM data will be used.
        mask: Any, optional
            Mask data. If None, no mask data will be used.
        crs: CRS, optional
            the output term:`coordinate reference system (CRS)` of the dataset.
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
        root_dir = Path(root)
        self.root_dir = root_dir

        if file_paths_unw is None:
            file_paths_unw = sorted(root_dir.rglob(self.filename_glob_unw))
        if file_paths_coh is None:
            file_paths_coh = sorted(root_dir.rglob(self.filename_glob_coh))

        if len(file_paths_unw) != len(file_paths_coh):
            raise ValueError(
                f"Number of interferogram files ({len(file_paths_unw)}) does not match "
                f"number of coherence files ({len(file_paths_coh)})"
            )

        super().__init__(
            root=root,
            file_paths=file_paths_unw,
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

        self.coh_dataset = RasterDataset(
            root=root,
            file_paths=file_paths_coh,
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

    def to_netcdf(
        self,
        filename: str,
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

        # TODO: add dem and mask
        profile = self.get_profile(roi)
        lat, lon = profile.to_latlon()

        ds = xr.Dataset(
            {
                "unw": (["band", "lat", "lon"], self[roi]["image"]),
                "coh": (["band", "lat", "lon"], self.coh_dataset[roi]["image"]),
            },
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
