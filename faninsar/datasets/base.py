"""Base classes for all :mod:`faninsar` datasets. The base class RasterDataset in this script is modified from the torchgeo package."""

from __future__ import annotations

import abc
import functools
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, overload

import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.merge
import shapely
import xarray as xr
from rasterio import features, fill
from rasterio import mask as rio_mask
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.transform import rowcol
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.warp import transform as warp_transform
from rtree.index import Index, Property
from shapely import ops
from tqdm.auto import tqdm

from faninsar._core import geo_tools
from faninsar._core.geo_tools import Profile
from faninsar._core.logger import setup_logger
from faninsar._core.pair_tools import Pairs
from faninsar.query import BoundingBox, GeoQuery, Points, Polygons, QueryResult

__all__ = ("GeoDataset", "RasterDataset", "PairDataset", "ApsDataset")

logger = setup_logger(
    log_name="FanInSAR.datasets.base", log_format="%(levelname)s - %(message)s"
)


class GeoDataset(abc.ABC):
    """Abstract base class for all :mod:`faninsar` datasets. This class is used
    to represent a geospatial dataset and provides methods to index the dataset
    and retrieve information about the dataset, such as CRS, resolution, data type,
    no data value, and a bounds.
    """

    # following attributes should be set by the subclass
    _crs: Optional[CRS] = None
    _res: tuple[float, float] = (0.0, 0.0)
    _dtype: Optional[np.dtype] = None
    _count: int = 0
    _roi: Optional[BoundingBox] = None
    _nodata: Any = None
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
    def crs(self) -> CRS | None:
        """coordinate reference system (:term:`CRS`) of the dataset.

        Returns:
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
                logger.warning(
                    f"the resolution of the dataset has been changed from {self.res} to {new_res}."
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
    def same_crs(self) -> bool:
        """True if all files in the dataset have the same CRS with the
        desired CRS, False otherwise.
        """
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
    def roi(self) -> BoundingBox | None:
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
    def nodata(self) -> float | int | Any | None:
        """No data value of the dataset.

        Returns
        -------
        nodata: float or int
            no data value of the dataset
        """
        return self._nodata

    @nodata.setter
    def nodata(self, new_nodata: float | int | Any) -> None:
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
        self, bbox: BoundingBox | Literal["roi", "bounds"] = "roi"
    ) -> Profile | None:
        """Return the profile information of the dataset for the given bounding
        box type. The profile information includes the width, height, transform,
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
            bounds = self._check_roi(bbox)
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
    >>> query = GeoQuery(points=points, boxes=[ds.bounds, ds.bounds])

    use the GeoQuery object to index the RasterDataset

    >>> sample = ds[query]

    output the samples shapes:

    >>> print('boxes result shape:', sample.boxes.data.shape)
    boxes result shape: (2, 7, 68, 80)

    >>> print('points result shape:', sample.points.data.shape)
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
    all_bands: list[str] = []

    #: Names of RGB bands in the dataset, used for plotting
    rgb_bands: list[str] = []

    #: Color map for the dataset, used for plotting
    cmap: dict[int, tuple[int, int, int, int]] = {}

    def __init__(
        self,
        root_dir: str = "data",
        paths: Sequence[str] | None = None,
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        dtype: np.dtype | None = None,
        nodata: float | int | Any | None = None,
        roi: BoundingBox | None = None,
        bands: Sequence[str] | None = None,
        cache: bool = True,
        resampling=Resampling.nearest,
        fill_nodata: bool = False,
        verbose: bool = True,
        ds_name: str = "",
    ) -> None:
        """Initialize a new raster dataset instance.

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
            no data value of the dataset. If None, the no data value of the first
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

        Raises
        ------
            FileNotFoundError: if no files are found in ``root_dir``
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.bands = bands or self.all_bands
        self.cache = cache
        self.resampling = resampling
        self.fill_nodata = fill_nodata
        self.verbose = verbose
        self.ds_name = ds_name

        if paths is None:
            paths = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for file_path in sorted(self.root_dir.rglob(self.pattern)):
                match = re.match(filename_regex, file_path.name)
                if match is not None:
                    paths.append(file_path)
        else:
            paths = [Path(path) for path in paths]

        # Populate the dataset index
        count = 0
        files_valid = []
        self._same_crs = True
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

                    if crs != src.crs:
                        self._same_crs = False
            except Exception as e:
                # Skip files that rasterio is unable to read
                warnings.warn(f"Unable to read {file_path}: \n--> : {e}", UserWarning)
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
            files_invalid = [str(i) for i in self._files.paths[~self._files.valid]]
            files_invalid_str = "\t" + "\n\t".join(files_invalid)
            warnings.warn(
                f"Unable to read {len(files_invalid)} files in "
                f"{self.__class__.__name__} dataset:\n{files_invalid_str}",
                UserWarning,
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
        self, query: GeoQuery | Points | BoundingBox | Polygons
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
        if isinstance(query, Points):
            query = GeoQuery(points=query)
        if isinstance(query, BoundingBox):
            query = GeoQuery(boxes=query)
        if isinstance(query, Polygons):
            query = GeoQuery(polygons=query)

        paths = self.files[self.files.valid].paths
        result = self._sample_files(paths, query)

        return result

    def _points_query(self, points: Points, vrt_fh) -> np.ndarray:
        """Return the values of dataset at given points. Points that outside the dataset will be masked."""
        points = self._ensure_query_crs(points)
        data = np.ma.hstack(list(vrt_fh.sample(points.values, masked=True)))
        return data

    def _bbox_query(self, bbox: BoundingBox, vrt_fh) -> np.ndarray:
        """Return the values of the dataset at the given bounding box."""
        bbox = self._ensure_query_crs(bbox)

        win = vrt_fh.window(*bbox)
        bands = self.band_indexes or vrt_fh.indexes
        data = vrt_fh.read(
            out_shape=(
                len(bands),
                round((bbox.top - bbox.bottom) / self.res[1]),
                round((bbox.right - bbox.left) / self.res[0]),
            ),
            resampling=self.resampling,
            indexes=self.band_indexes,
            window=win,
            masked=True,
            boundless=self.same_crs,  # boundless=True if self.same_crs else False,
        )

        if data.mask.ndim == 0:
            data = np.ma.masked_array(data.data, data == self.nodata)
        if self.fill_nodata:
            data = fill.fillnodata(data)
        return data

    def _polygons_query(self, polygons: Polygons, vrt_fh) -> np.ndarray:
        """Return the values of the dataset at the given polygons."""
        polygons = self._ensure_query_crs(polygons)

        mask_params = {
            "filled": False,
            "pad": polygons.pad,
            "all_touched": polygons.all_touched,
            "invert": False,
            "crop": True,
        }
        rasterize_params = {
            "all_touched": polygons.all_touched,
            "fill": 0,
            "default_value": 1,
        }

        shapes = polygons.frame.geometry.to_list()
        if len(polygons.desired) > 0:
            data_ls = []
            transform_ls = []
            mask_ls = []
            for shp in shapes:
                data, out_transform = rio_mask.mask(vrt_fh, [shp], **mask_params)

                rasterize_params.update(
                    {"out_shape": data.shape[1:3], "transform": out_transform}
                )
                mask = features.rasterize([shp], **rasterize_params).astype(bool)

                if self.fill_nodata:
                    data = fill.fillnodata(data)
                    data = np.ma.masked_array(data.data, ~mask)
                data_ls.append(data)
                transform_ls.append(out_transform)
                mask_ls.append(mask)
        else:
            mask_params.update({"invert": True, "crop": False})
            data, out_transform = rio_mask.mask(vrt_fh, shapes, **mask_params)

            rasterize_params.update(
                {"out_shape": data.shape[1:3], "transform": out_transform}
            )
            mask = features.rasterize(shapes, **rasterize_params).astype(bool)
            if self.fill_nodata:
                data = fill.fillnodata(data)
                data = np.ma.masked_array(data.data, ~mask)
            data_ls = [data]
            transform_ls = [out_transform]
            mask_ls = [mask]

        return data_ls, transform_ls, mask_ls

    @overload
    def _ensure_query_crs(self, query: BoundingBox) -> BoundingBox: ...

    @overload
    def _ensure_query_crs(self, query: Points) -> Points: ...

    @overload
    def _ensure_query_crs(self, query: Polygons) -> Polygons: ...

    def _ensure_query_crs(
        self, query: Points | BoundingBox | Polygons
    ) -> Points | BoundingBox | Polygons:
        """Ensure that the query has the same CRS as the dataset."""
        if query.crs is None:
            warnings.warn(
                f"No CRS is specified for the {query}, assuming they are in the"
                f" same CRS as the dataset ({self.crs})."
            )
        else:
            if query.crs != self.crs:
                query = query.to_crs(self.crs)
        return query

    def _ensure_loading_verbose(self, sequence: Sequence) -> Sequence:
        if self.verbose:
            sequence = tqdm(
                sequence, desc=f"Loading {self.ds_name} Files", unit=" files"
            )
        return sequence

    def _ensure_saving_verbose(
        self,
        sequence: Sequence,
        ds_name: str,
        unit: str = " files",
    ) -> Sequence:
        if self.verbose:
            sequence = tqdm(sequence, desc=f"Saving {ds_name} Files", unit=unit)
        return sequence

    def _sample_files(self, paths: Sequence[str], query: GeoQuery) -> QueryResult:
        """Sample or retrieve values from the dataset for the given query.

        Parameters
        ----------
        paths : list of str
            list of paths for files to stack
        query : GeoQuery
            a GeoQuery instance containing the desired queries.

        Returns
        -------
        result : QueryResult
            a QueryResult instance containing the results of the various queries.
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in paths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in paths]

        vrt_fhs = self._ensure_loading_verbose(vrt_fhs)

        files_points_list = []
        files_bbox_list = []
        files_polygons_list = []
        for vrt_fh in vrt_fhs:
            # Get the points values
            if query.points is not None:
                data = self._points_query(query.points, vrt_fh)
                files_points_list.append(data)
            # Get the bounding boxes values
            bbox_list = []
            if query.boxes is not None:
                for bbox in query.boxes:
                    data = self._bbox_query(bbox, vrt_fh)
                    bbox_list.append(data)
                files_bbox_list.append(np.ma.asarray(bbox_list))
            # Get the polygons values
            if query.polygons is not None:
                data_ls, transform_ls, mask_ls = self._polygons_query(
                    query.polygons, vrt_fh
                )
                files_polygons_list.append(data_ls)

        # Stack the points values
        points_values = None
        if len(files_points_list) > 0:
            points_values = np.ma.asarray(files_points_list).squeeze()

        # Stack the bounding boxes values
        bbox_values = None
        files_bbox_list = np.ma.asarray(files_bbox_list)
        if len(files_bbox_list) > 0:
            n_band = files_bbox_list.shape[2]
            if n_band == 1:
                bbox_values = files_bbox_list.squeeze(axis=2).transpose(1, 0, 2, 3)
            else:
                bbox_values = files_bbox_list.transpose(1, 0, 2, 3, 4)

        # Stack the polygons values
        polygons_values = None
        if len(files_polygons_list) > 0:
            num_polygons = len(query.polygons)
            arr_list = [[] for _ in range(num_polygons)]
            for data in files_polygons_list:
                for i, d in enumerate(data):
                    arr_list[i].append(d)
            polygons_values = [np.ma.asarray(arr).squeeze(1) for arr in arr_list]

        points_result = (
            None
            if points_values is None
            else {
                "data": points_values,
                "dims": "(n_files, n_point)",
            }
        )
        bbox_result = (
            None
            if bbox_values is None
            else {
                "data": bbox_values,
                "dims": "(n_boxes, (n_files, height, width))",
            }
        )
        polygons_result = (
            None
            if polygons_values is None
            else {
                "data": polygons_values,
                "dims": "(n_polygons, (n_files, height, width))",
                "transforms": transform_ls if polygons_values is not None else None,
                "masks": mask_ls if polygons_values is not None else None,
            }
        )
        result = QueryResult(points_result, bbox_result, polygons_result, query)

        return result

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
        xy: Sequence,
        crs: Optional[CRS | str] = None,
        bbox: BoundingBox | Literal["roi", "bounds"] = "roi",
    ) -> np.ndarray:
        """Convert x, y coordinates to row, col in the dataset.

        Parameters
        ----------
        xy: Sequence
            Pairs of x, y coordinates (floats)
        crs: CRS or str, optional
            The CRS of the points. If None, the CRS of the dataset will be used.
            allowed CRS formats are the same as those supported by rasterio.
        bbox : str, one of ['bounds', 'roi'], optional
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
            raise ValueError(
                f"Expected xy to be an array of shape (n, 2), got {xy.shape}"
            )
        if crs is not None:
            crs = CRS.from_user_input(crs)
            if crs != self.crs:
                xs, ys = warp_transform(crs, self.crs, xy[:, 0], xy[:, 1])
                xy = np.column_stack((xs, ys))

        profile = self.get_profile(bbox)

        rows, cols = rowcol(profile["transform"], xy[:, 0], xy[:, 1])
        row_col = np.column_stack((rows, cols))
        return row_col

    def xy(
        self,
        row_col: Sequence,
        crs: Optional[CRS | str] = None,
        bbox: BoundingBox | Literal["roi", "bounds"] = "roi",
    ) -> np.ndarray:
        """Convert row, col in the dataset to x, y coordinates.

        Parameters
        ----------
        row_col: Sequence
            Pairs of row, col in the dataset (floats)
        crs: CRS or str, optional
            The CRS of the points. If None, the CRS of the dataset will be used.
            allowed CRS formats are the same as those supported by rasterio.
        bbox : str, one of ['bounds', 'roi'], optional
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
            raise ValueError(
                f"Expected row_col to be an array of shape (n, 2), got {row_col.shape}"
            )

        profile = self.get_profile(bbox)

        xs, ys = xy(profile["transform"], row_col[:, 0], row_col[:, 1])

        if crs is not None:
            crs = CRS.from_user_input(crs)
            if crs != self.crs:
                xs, ys = warp_transform(crs, self.crs, xy[:, 0], xy[:, 1])
        xy = np.column_stack((xs, ys))

        return xy

    def to_tiffs(
        self,
        out_dir: str | Path,
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
            src = self._load_warp_file(f)
            dest_arr = self._bbox_query(roi, src).squeeze(0)
            with rasterio.open(out_file, "w", **profile.profile) as dst:
                dst.write(dest_arr, 1)

    def to_netcdf(
        self,
        filename: str | Path,
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
            {"image": (["band", "lat", "lon"], sample.boxes.data)},
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

    def save_arr_to_tiff(
        self,
        arr: np.ndarray,
        filename: str | Path,
        roi: Optional[BoundingBox] = None,
        profile: Optional[Profile] = None,
    ) -> None:
        """Save a numpy array to a tiff file using the geoinformation of the dataset.

        Parameters
        ----------
        arr : numpy.ndarray
            numpy array to save. arr can be a 2D array or a 3D array. If arr is a
            3D array, the first dimension is the band dimension.
        filename : str or Path
            path to the tiff file to save
        roi : BoundingBox, optional
            Region of interest to save. only used if profile is None.
        profile : Profile, optional
            profile of the tiff file. If None, the roi must be specified.
        """
        if arr.ndim == 2:
            indexes = [1]
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            indexes = [i + 1 for i in range(arr.shape[0])]
        else:
            raise ValueError(
                f"Expected arr to be an array of shape (n, m) or (n, m, k), got {arr.shape}"
            )

        if profile is None:
            if roi is None:
                raise ValueError("roi must be specified if profile is None")
            profile = self.get_profile(roi)
            profile["count"] = arr.shape[0]

        with rasterio.open(filename, "w", **profile) as dst:
            dst.write(arr, indexes)


class PairDataset(RasterDataset):
    """A base class for pair datasets."""

    _pairs: Optional[Pairs] = None
    _datetime: Optional[pd.DatetimeIndex] = None

    @classmethod
    @abc.abstractmethod
    def parse_pairs(cls, paths: list[Path]) -> Pairs:
        """Used to parse pairs from filenames. *Must be implemented in subclass*.

        Parameters
        ----------
        paths : list of pathlib.Path
            list of file paths to parse pairs

        Returns
        -------
        pairs : Pairs object
            pairs parsed from filenames

        Example
        -------
        for the HyP3 dataset, pairs are parsed from the filenames as follows:

        >>> names = [f.name for f in paths]]
        >>> pair_names = ['_'.join(i.split("_")[1:3]) for i in names]

        for the HyP3 dataset, the pair names are the second and third parts of the
        filename, separated by an underscore. After parsing the pair names, the
        :class:`Pairs` object can be created by using the ``from_names`` method.

        >>> pairs = Pairs.from_names(pair_names)
        """

    @classmethod
    @abc.abstractmethod
    def parse_datetime(cls, paths: list[Path]) -> pd.DatetimeIndex:
        """Used to parse datetime from filenames. *Must be implemented in subclass*.

        Parameters
        ----------
        paths : list of pathlib.Path
            list of file paths to parse datetime

        Returns
        -------
        datetime : pd.DatetimeIndex
            datetime parsed from filenames
        """

    @property
    def pairs(self) -> Pairs:
        """Return Pairs parsed from filenames."""
        return self._pairs

    @property
    def datetime(self) -> pd.DatetimeIndex:
        """Return the datetime for each pair in the dataset."""
        return self._datetime


class ApsDataset(RasterDataset):
    """
    A base class for aps (atmospheric phase screen) datasets.
    """

    #: This expression is used to find the APS files.
    pattern = "*"

    _pairs = None

    def to_pair_files(
        self,
        out_dir: str | Path,
        pairs: Pairs,
        ref_points: Points,
        roi: Optional[BoundingBox] = None,
        overwrite: bool = False,
        prefix: str = "APS",
    ):
        """Generate aps-pair files for given pairs and reference points.

        Parameters
        ----------
        out_dir : str or Path
            path to the directory to save the aps-pair files
        pairs : Pairs
            pairs to generate aps-pair files
        ref_points : Points
            reference points which values are subtracted for all aps-pair files
        roi : BoundingBox, optional
            region of interest to save. If None, the roi of the dataset will be used.
        overwrite : bool, optional
            if True, overwrite existing files, default: False
        prefix : str, optional
            prefix of the aps-pair files, default: "APS"
        """
        if roi is None:
            roi = self.roi

        profile = self.get_profile(roi)
        profile["count"] = 1
        dates = self.parse_dates(self.files.paths)

        dates_missing = np.setdiff1d(pairs.dates, dates)
        if len(dates_missing) > 0:
            warnings.warn(
                f"Following dates are missing in the {self.ds_name} dataset. \n{dates_missing}"
            )

        df_paths = pd.Series(self.files.paths.values, index=dates)

        mask = ~np.any(np.isin(pairs.values, dates_missing), axis=1)
        pairs = pairs[mask]

        pairs_names = self._ensure_saving_verbose(
            pairs.to_names(), ds_name=f"{self.ds_name} Pair", unit=" pairs"
        )

        for pair_name in pairs_names:
            primary, secondary = pair_name.split("_")
            out_file = Path(out_dir) / f"{prefix}_{pair_name}.tif"
            if out_file.exists() and not overwrite:
                logger.info(f"{out_file.name} already exists, skipping")
                continue
            with rasterio.open(out_file, "w", **profile.profile) as dst:
                src_primary = self._load_warp_file(df_paths[primary])
                src_secondary = self._load_warp_file(df_paths[secondary])
                dest_arr = (
                    self._bbox_query(roi, src_primary).squeeze(0)
                    - self._bbox_query(roi, src_secondary).squeeze(0)
                    - (
                        self._points_query(ref_points, src_primary)
                        - self._points_query(ref_points, src_secondary)
                    ).mean()
                )

                dst.write(dest_arr, 1)

    @classmethod
    @abc.abstractmethod
    def parse_dates(cls, paths: Optional[Sequence[str]] = None) -> pd.DatetimeIndex:
        """Used to parse acquisition dates from filenames. *Must be implemented
        in subclass*.

        Parameters
        ----------
        paths : list of pathlib.Path
            list of file paths to parse datetime

        Returns
        -------
        datetime : pd.DatetimeIndex
            datetime parsed from filenames
        """


class ApsPairs(PairDataset):
    """
    A dataset manages the data of APS pairs.
    """

    #: This expression is used to find the GACOSPairs files.
    pattern = "*.tif"

    def __init__(
        self,
        root_dir: str = "data",
        paths: Optional[Sequence[str]] = None,
        crs: Optional[CRS] = None,
        res: Optional[float | tuple[float, float]] = None,
        dtype: Optional[np.dtype] = None,
        nodata: Optional[float | int | Any] = None,
        roi: Optional[BoundingBox] = None,
        bands: Optional[Sequence[str]] = None,
        cache: bool = True,
        resampling=Resampling.nearest,
        fill_nodata: bool = False,
        verbose: bool = True,
        ds_name: str = "",
    ) -> None:
        """Initialize a new ApsPairs instance.

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

        Raises
        ------
            FileNotFoundError: if no files are found in ``root_dir``
        """
        super().__init__(
            root_dir=root_dir,
            paths=paths,
            crs=crs,
            res=res,
            dtype=dtype,
            nodata=nodata,
            roi=roi,
            bands=bands,
            cache=cache,
            resampling=resampling,
            fill_nodata=fill_nodata,
            verbose=verbose,
            ds_name=ds_name,
        )
        self._pairs = self.parse_pairs(self.files.paths[self.valid])
        self._datetime = self.parse_datetime(self.files.paths[self.valid])

    @classmethod
    def parse_pairs(cls, paths: list[Path]) -> Pairs:
        """Parse pairs from a list of APS-pair file paths."""
        names = [Path(f).stem for f in paths]
        pair_names = ["_".join(i.split("_")[1:3]) for i in names]
        pairs = Pairs.from_names(pair_names)
        return pairs

    @classmethod
    def parse_datetime(cls, paths: list[Path]) -> pd.DatetimeIndex:
        f"""Parse datetime from a list of {cls.__class__.__name__} file paths."""
        names = [Path(f).stem for f in paths]
        pair_names = ["_".join(i.split("_")[1:3]) for i in names]
        date_names = np.unique([i.split("_") for i in pair_names])
        return pd.DatetimeIndex(date_names)

    @property
    def dates(self):
        """Return the dates of the dataset."""
        return self._datetime
