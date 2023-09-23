"""Base classes for all :mod:`faninsar` datasets. all classes in this script are modified from the torchgeo package."""

import abc
import functools
import glob
import os
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import (Any, Callable, Literal, Optional, Tuple, Union, cast,
                    overload)

import fiona
import fiona.transform
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.merge
import shapely
from rasterio.coords import BoundingBox as BBox
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.warp import transform as warp_transform
from rtree.index import Index, Property
from shapely import ops
from tqdm import tqdm

from faninsar.utils.geo_tools import Profile

__all__ = ("BoundingBox", "GeoDataset", "RasterDataset")


@dataclass(frozen=True)
class BoundingBox:
    """Data class for indexing interferogram data using a spatial bounding box.

    .. Note::

        The :class:`BoundingBox` class a modified version of the ``BoundingBox``
        class from the torchgeo package. The modifications are:

        * date bounds are removed
        * the bounds is changed to (left, bottom, right, top) which is the same as
            rasterio :class:`rasterio.coords.BoundingBox`
        * added :meth:`to_rasterio_bounds` method to convert the bounding box to a
            rasterio bounds tuple
    """

    #: western boundary
    left: float
    #: southern boundary
    bottom: float
    #: eastern boundary
    right: float
    #: northern boundary
    top: float

    def __post_init__(self) -> None:
        """Validate the arguments passed to :meth:`__init__`.

        Raises:
            ValueError: if bounding box is invalid
                (left > right, bottom > top)

        .. versionadded:: 0.2
        """
        if self.left > self.right:
            raise ValueError(
                f"Bounding box is invalid: 'left={self.left}' > 'right={self.right}'"
            )
        if self.bottom > self.top:
            raise ValueError(
                f"Bounding box is invalid: 'bottom={self.bottom}' > 'top={self.top}'"
            )

    # https://github.com/PyCQA/pydocstyle/issues/525
    @overload
    def __getitem__(self, key: int) -> float:  # noqa: D105
        pass

    @overload
    def __getitem__(self, key: slice) -> list[float]:  # noqa: D105
        pass

    def __getitem__(self, key: Union[int, slice]) -> Union[float, list[float]]:
        """Index the (left, bottom, right,  top) tuple.

        Args:
            key: integer or slice object

        Returns:
            the value(s) at that index

        Raises:
            IndexError: if key is out of bounds
        """
        return [self.left, self.bottom, self.right, self.top][key]

    def __iter__(self) -> Iterator[float]:
        """Container iterator.

        Returns:
            iterator object that iterates over all objects in the container
        """
        yield from [self.left, self.bottom, self.right, self.top]

    def __contains__(self, other: 'BoundingBox') -> bool:
        """Whether or not other is within the bounds of this bounding box.

        Args:
            other: another bounding box

        Returns:
            True if other is within this bounding box, else False
        """
        return (
            (self.left <= other.left <= self.right)
            and (self.left <= other.right <= self.right)
            and (self.bottom <= other.bottom <= self.top)
            and (self.bottom <= other.top <= self.top)
        )

    def __or__(self, other: 'BoundingBox') -> 'BoundingBox':
        """The union operator.

        Args:
            other: another bounding box

        Returns:
            the minimum bounding box that contains both self and other
        """
        return BoundingBox(
            min(self.left, other.left),
            max(self.right, other.right),
            min(self.bottom, other.bottom),
            max(self.top, other.top)
        )

    def __and__(self, other: 'BoundingBox') -> 'BoundingBox':
        """The intersection operator.

        Args:
            other: another bounding box

        Returns:
            the intersection of self and other

        Raises:
            ValueError: if self and other do not intersect
        """
        try:
            return BoundingBox(
                max(self.left, other.left),
                min(self.right, other.right),
                max(self.bottom, other.bottom),
                min(self.top, other.top),
            )
        except ValueError:
            raise ValueError(
                f"Bounding boxes {self} and {other} do not overlap")

    @property
    def area(self) -> float:
        """Area of bounding box.

        Area is defined as spatial area.

        Returns: 
            area
        """
        return (self.right - self.left) * (self.top - self.bottom)

    def to_dict(self) -> dict[str, float]:
        """Convert the bounding box to a dictionary.

        Returns:
            dictionary with keys 'left', 'bottom', 'right', 'top'
        """
        return {'left': self.left, 'bottom': self.bottom, 'right': self.right, 'top': self.top}

    def intersects(self, other: 'BoundingBox') -> bool:
        """Whether or not two bounding boxes intersect.

        Args:
            other: another bounding box

        Returns:
            True if bounding boxes intersect, else False
        """
        return (
            self.left <= other.right
            and self.right >= other.left
            and self.bottom <= other.top
            and self.top >= other.bottom
        )

    def split(
        self, proportion: float, horizontal: bool = True
    ) -> tuple['BoundingBox', 'BoundingBox']:
        """Split BoundingBox in two.

        Args:
            proportion: split proportion in range (0,1)
            horizontal: whether the split is horizontal or vertical

        Returns:
            A tuple with the resulting BoundingBoxes

        .. versionadded:: 0.5
        """
        if not (0.0 < proportion < 1.0):
            raise ValueError("Input proportion must be between 0 and 1.")

        if horizontal:
            w = self.right - self.left
            splitx = self.left + w * proportion
            bbox1 = BoundingBox(
                self.left, splitx, self.bottom, self.top
            )
            bbox2 = BoundingBox(
                splitx, self.right, self.bottom, self.top
            )
        else:
            h = self.top - self.bottom
            splity = self.bottom + h * proportion
            bbox1 = BoundingBox(
                self.left, self.right, self.bottom, splity
            )
            bbox2 = BoundingBox(
                self.left, self.right, splity, self.top
            )

        return bbox1, bbox2


class GeoDataset(abc.ABC):
    """Abstract base class for all :mod:`faninsar` datasets. This class is used to
    represent a geospatial dataset and provides methods to index the dataset and
    retrieve information about the dataset, such as CRS, resolution, and bounds.

    .. Note::

        Although this :class:`GeoDataset` class is based on the ``GeoDataset`` class in the torchgeo package, it has been extensively modified to the point where the two classes are completely distinct. When using this :class:`GeoDataset` class, you should not make any assumptions based on the torchgeo version, as the implementations differ significantly.
    """

    # following attributes should be set by the subclass
    _crs: Optional[CRS] = None
    _res: Tuple[float, float] = (0.0, 0.0)
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
    size: {len(self)}"""

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
            profile = self.get_profile('bounds')
            tf, *_ = calculate_default_transform(
                self.crs,
                new_crs,
                profile['width'],
                profile['height'],
                self.bounds[0],
                self.bounds[1],
                self.bounds[2],
                self.bounds[3]
            )
            new_res = (abs(float(tf.a)), abs(float(tf.e)))
            if (new_res[0] != self.res[0]
                    or new_res[1] != self.res[1]):
                print(
                    f"Warning: the resolution of the dataset has been changed from {self.res} to {new_res}."
                )
                self.res = new_res

            # reproject the index
            new_index = Index(
                interleaved=True,
                properties=Property(dimension=2)
            )
            project = pyproj.Transformer.from_crs(
                pyproj.CRS(str(self.crs)),
                pyproj.CRS(str(new_crs)),
                always_xy=True
            ).transform
            for hit in self.index.intersection(self.index.bounds, objects=True):
                old_xmin, old_xmax, old_ymin, old_ymax = hit.bounds
                old_box = shapely.geometry.box(
                    old_xmin, old_ymin, old_xmax, old_ymax)
                new_box = ops.transform(project, old_box)
                new_bounds = tuple(new_box.bounds)
                new_index.insert(hit.id, new_bounds, hit.object)

            self.index = new_index

        self._crs = new_crs

    @property
    def res(self) -> Tuple[float, float]:
        """Return the resolution of the dataset.

        Returns
        -------
        res: tuple of floats
            resolution of the dataset in x and y directions.
        """
        return self._res

    @res.setter
    def res(self, new_res: Union[float, Tuple[float, float]]) -> None:
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
                raise ValueError(
                    "Resolution must be a float or a tuple of floats"
                )
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
        if isinstance(new_roi, Iterable):
            new_roi = np.asarray([i for i in new_roi])
        if len(new_roi) != 4:
            raise ValueError(
                f"ROI must be a tuple of length 4, got {len(new_roi)}"
            )
        if (new_roi[0] < self.bounds[0]
                or new_roi[1] < self.bounds[1]
                or new_roi[2] > self.bounds[2]
                or new_roi[3] > self.bounds[3]):
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
        self,
        bbox: Literal['roi', 'bounds'] = 'roi'
    ) -> Optional[Profile]:
        """Return the profile of the dataset

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
        if bbox == 'bounds':
            if self.bounds is None:
                return None
            else:
                bounds = self.bounds
        elif bbox == 'roi':
            if self.roi is None:
                return None
            else:
                bounds = self.roi
        else:
            raise ValueError(
                f"bbox must be one of ['bounds', 'roi'], but got {bbox}"
            )

        profile = Profile.from_bounds_res(bounds, self.res)
        profile["count"] = self.count
        profile["dtype"] = self.dtype
        profile["nodata"] = self.nodata
        profile["crs"] = self.crs

        return profile


class RasterDataset(GeoDataset):
    """A base class for raster datasets.

    .. Note::

        This class is a modified version of the ``RasterDataset`` class from the 
        torchgeo package. The modifications are:

        * the ``__init__`` method is modified to accept more parameters for the dataset
        * add :property:`files` property to return a DataFrame of all files in the dataset, including invalid files and information about whether the file is valid
        * add :meth:`sample` method to sample values from the dataset for given points
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

    #: True if dataset contains imagery, False if dataset contains mask
    is_image = True

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
        res: Optional[Union[float, Tuple[float, float]]] = None,
        dtype: Optional[np.dtype] = None,
        nodata: Optional[Union[float, int, Any]] = None,
        rio: Optional[BoundingBox] = None,
        bands: Optional[Sequence[str]] = None,
        cache: bool = True,
        resampling=Resampling.nearest,
        verbose: bool = False,
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
        verbose : bool, optional
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

        self._files = pd.DataFrame(
            {"file_paths": file_paths, "valid": files_valid}
        )
        if not self._files.valid.all():
            files_invalid = self._files.file_paths[~self._files.valid].tolist()
            print(
                f"Unable to read {len(files_invalid)} files in {self.__class__.__name__} dataset:"
                "\n\t" + "\n\t".join(files_invalid),
            )

        self.band_indexes = None
        if self.bands:
            if self.all_bands:
                self.band_indexes = [
                    self.all_bands.index(i) + 1 for i in self.bands
                ]
            else:
                msg = (
                    f"{self.__class__.__name__} is missing an `all_bands` "
                    "attribute, so `bands` cannot be specified."
                )
                raise AssertionError(msg)

        self.crs = crs
        self.res = res
        self.rio = rio
        self.dtype = dtype
        self.nodata = nodata
        self.count = count

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (left, right, bottom, top) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        file_paths = self.files[self.files.valid].file_paths
        data = self._stack_files(file_paths, query)

        sample = {"crs": self.crs, "bbox": query}

        if self.is_image:
            sample["image"] = data
        else:
            sample["mask"] = data

        return sample

    def _stack_files(
        self,
        file_paths: Sequence[str],
        query: BoundingBox,
    ) -> np.ndarray:
        """Stack files into a single 3D array.

        Parameters
        ----------
        file_paths : list of str
            list of paths for files to stack
        query : BoundingBox
            bounding box to index

        Returns
        -------
        dest : numpy.ndarray
            stacked array
        tf : affine.Affine
            affine transform of the stacked array
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in file_paths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in file_paths]

        if self.verbose:
            vrt_fhs = tqdm(
                vrt_fhs,
                desc="Loading files",
                unit=" files"
            )

        data_list = []
        for vrt_fh in vrt_fhs:
            win = vrt_fh.window(*query)
            data = vrt_fh.read(
                out_shape=(
                    1,
                    int((query.top - query.bottom) / self.res[1]),
                    int((query.right - query.left) / self.res[0]),
                ),
                resampling=self.resampling,
                indexes=self.band_indexes,
                window=win,
                boundless=True,
                fill_value=self.nodata,
            )
            data_list.append(data)
        dest = np.concatenate(data_list, axis=0)

        print(win)

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

    def sample(
        self,
        xy: Iterable,
        crs: Optional[Union[CRS, str]] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Sample values from dataset for given points.

        Parameters
        ----------
        xy: Iterable
            Pairs of x, y coordinates (floats)
        crs: CRS or str, optional
            The CRS of the points. If None, the CRS of the dataset will be used.
            allowed CRS formats are the same as those supported by rasterio.
        verbose: bool, optional
            If True, show progress

        Returns:
            array of values at those points
        """
        xy = np.asarray(xy)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(
                f"Expected xy to be an array of shape (n, 2), got {xy.shape}"
            )
        if crs is not None:
            crs = CRS.from_user_input(crs)
            if crs != self.crs:
                xs, ys = warp_transform(crs, self.crs, xy[:, 0], xy[:, 1])
                xy = np.column_stack((xs, ys))

        files = self.files[self.files.valid].file_paths

        if verbose:
            files = tqdm(
                files,
                desc="Sampling values",
                unit=" files"
            )

        values = []
        for file_path in files:
            with rasterio.open(file_path) as src:
                values.append(list(src.sample(xy)))

        return np.asarray(values).squeeze()
