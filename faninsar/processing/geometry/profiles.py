"""Raster profile model built on geospatial grid metadata."""

from __future__ import annotations

import pprint
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
from rasterio import Affine, transform
from rasterio.profiles import Profile as RasterioProfile

from faninsar.io.file_tools import load_metas
from faninsar.logging import setup_logger

from .coordinates import xy_from_transform
from .grids import GeoGrid, GeoGridMixin

if TYPE_CHECKING:
    from os import PathLike

    from numpy.typing import ArrayLike

    from faninsar.data.query.bbox import BoundingBox
    from faninsar.core.types import CrsLike

logger = setup_logger(__name__)

DEFAULT_KEYS_Profile = [
    "height",
    "width",
    "transform",
    "crs",
    "nodata",
    "count",
    "driver",
    "dtype",
]


class Profile(GeoGridMixin, MutableMapping[str, Any]):
    """A class to manage the profile of a raster image.

    .. note::
        the :attr:`height`, :attr:`width`, :attr:`transform` and :attr:`crs`
        are the basic parameters for a warp process.

    Parameters
    ----------
    height: int
        The height of the raster image in pixels.
    width: int
        The width of the raster image in pixels.
    transform: Affine
        The affine transformation matrix that maps pixel coordinates to spatial
        coordinates.
    crs: CrsLike | None
        The coordinate reference system of the raster image. Could be any type
        that :meth:`pyproj.CRS.from_user_input` accepts. Default is None (unset).
    nodata: float | None
        The nodata value of the raster image. If not set, it will be None.
    count: int
        The count of bands of the raster image. Default is 1.
    driver: str
        The driver of the raster image. Default is "GTiff".
    dtype: str | np.dtype | None
        The dtype of the raster image. Default is None. If not set, it will be
        determined by the data array when writing to raster file.
    kwargs: dict[str, Any]
        Other keyword arguments for :class:`rasterio.profiles.Profile` class.

    """

    def __init__(
        self,
        height: int,
        width: int,
        transform: Affine,
        crs: CrsLike | None = None,
        nodata: float | None = None,
        count: int = 1,
        driver: str = "GTiff",
        dtype: str | np.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a raster profile."""
        self.shape = (int(height), int(width))
        self.transform = transform
        self.crs = crs
        self.nodata: float | None = None
        self.nodata = nodata
        self.count = count
        self.driver = driver
        self.dtype = dtype
        self.kwargs = {} if kwargs is None else dict(kwargs)

        for key, value in self.kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """Get the value of the key."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set the value of the key."""
        if key not in DEFAULT_KEYS_Profile:
            self.kwargs[key] = value
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        """Delete a non-default profile item."""
        if key in DEFAULT_KEYS_Profile:
            msg = f"Cannot delete required profile key: {key}"
            logger.error(msg)
            raise KeyError(msg)
        if key not in self.kwargs:
            msg = f"{key!r} is not a stored profile metadata key"
            logger.error(msg)
            raise KeyError(msg)
        self.kwargs.pop(key)
        if hasattr(self, key):
            delattr(self, key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over profile keys."""
        return iter(self.to_dict())

    def __len__(self) -> int:
        """Return the number of profile items."""
        return len(self.to_dict())

    def __repr__(self) -> str:
        """Get the string representation of the Profile."""
        info = self.to_dict()
        info["crs"] = self.crs.to_string() if self.crs else None
        repr_str = f" {pprint.pformat(info, indent=2, sort_dicts=False).strip('{}')}"
        return f"Profile(\n{repr_str}\n)"

    @property
    def geogrid(self) -> GeoGrid:
        """Get the GeoGrid object from the profile."""
        return GeoGrid(self.transform, self.shape, self.crs)

    @property
    def nodata(self) -> float | None:
        """The nodata value of the raster image."""
        return self._nodata

    @nodata.setter
    def nodata(self, value: float | None) -> None:
        """Set the nodata value of the raster image."""
        if value is None:
            self._nodata = None
            return
        if not isinstance(value, (int, float, np.integer, np.floating)):
            msg = f"nodata must be a numeric value or None, but got {value!r}"
            logger.error(msg)
            raise TypeError(msg)
        self._nodata = float(value)

    @property
    def count(self) -> int:
        """The count of bands of the raster image."""
        return self._count

    @count.setter
    def count(self, value: int) -> None:
        """Set the count of bands of the raster image."""
        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
            msg = f"count must be an integer, but got {value!r}"
            logger.exception(msg)
            raise TypeError(msg) from exc
        if count < 1:
            msg = f"count must be greater than 0, but got {count}"
            logger.error(msg)
            raise ValueError(msg)
        self._count = count

    @property
    def driver(self) -> str:
        """The driver of the raster image."""
        return self._driver

    @driver.setter
    def driver(self, value: str) -> None:
        """Set the driver of the raster image."""
        if not isinstance(value, str):
            msg = f"driver must be a string, but got {value!r}"
            logger.error(msg)
            raise TypeError(msg)
        self._driver = value

    @property
    def dtype(self) -> str | np.dtype | None:
        """The dtype of the raster image."""
        return self._dtype

    @dtype.setter
    def dtype(self, value: str | np.dtype | None) -> None:
        """Set the dtype of the raster image."""
        if value is not None and not isinstance(value, (str, np.dtype)):
            msg = f"dtype must be a string, numpy.dtype, or None, but got {value!r}"
            logger.error(msg)
            raise TypeError(msg)
        self._dtype = value

    @property
    def kwargs(self) -> dict[str, Any]:
        """Other keyword arguments for rasterio profile metadata."""
        return self._kwargs

    @kwargs.setter
    def kwargs(self, value: dict[str, Any]) -> None:
        """Set other keyword arguments for rasterio profile metadata."""
        if not isinstance(value, dict):
            msg = f"kwargs must be a dictionary, but got {value!r}"
            logger.error(msg)
            raise TypeError(msg)
        self._kwargs = dict(value)

    @staticmethod
    def _split_profile(profile: dict) -> tuple[dict, dict]:
        """Split the profile into default keys and other keys."""
        kwargs = {}
        profile_new = {}
        for key, value in profile.items():
            if key not in DEFAULT_KEYS_Profile:
                kwargs[key] = value
            else:
                profile_new[key] = value
        return profile_new, kwargs

    @classmethod
    def from_geogrid(cls, geogrid: GeoGrid, **kwargs: Any) -> Profile:
        """Create a Profile object from a GeoGrid object.

        Parameters
        ----------
        geogrid : GeoGrid
            GeoGrid object providing the shared geometry information.
        **kwargs : Any
            Additional profile metadata such as ``nodata``, ``count``,
            ``driver``, ``dtype``, or other rasterio profile options.

        Returns
        -------
        Profile
            Profile object created from the given GeoGrid and metadata.

        """
        profile = {
            "height": geogrid.height,
            "width": geogrid.width,
            "transform": geogrid.transform,
            "crs": geogrid.crs,
        }
        profile.update(kwargs)
        profile, kwargs_extra = cls._split_profile(profile)
        return cls(**profile, **kwargs_extra)

    @classmethod
    def from_raster_file(cls, raster_file: PathLike, **kwargs: Any) -> Profile:
        """Create a Profile object from a raster file.

        Parameters
        ----------
        raster_file : PathLike
            Raster file used to initialize the profile.
        **kwargs : Any
            Additional profile metadata. Values in ``kwargs`` override metadata
            loaded from the raster file.

        """
        with rasterio.open(raster_file) as ds:
            profile = dict(ds.profile.copy())
        profile.update(kwargs)
        # split the profile into default keys and other keys
        profile, kwargs = cls._split_profile(profile)
        return cls(**profile, **kwargs)

    @classmethod
    def from_ascii_header_file(
        cls,
        ascii_file: PathLike,
        **kwargs: Any,
    ) -> Profile:
        """Create a Profile object from an ascii header file.

        The ascii header file is the metadata of a binary. More information can
        be found at: https://desktop.arcgis.com/zh-cn/arcmap/latest/manage-data/raster-and-images/esri-ascii-raster-format.htm.

        Example of an ascii header file
        -------------------------------
        ::

            ncols         43200
            nrows         18000
            xllcorner     -180.000000
            yllcorner     -60.000000
            cellsize      0.008333
            nodata_value  -9999
        """
        dict_common = load_metas(
            ascii_file,
            keys=["ncols", "nrows", "cellsize", "nodata_value"],
            line_end=10,
        )
        if (
            dict_common["ncols"] is None
            or dict_common["nrows"] is None
            or dict_common["cellsize"] is None
        ):
            msg = "ncols, nrows and cellsize must be set in the ascii file"
            raise ValueError(msg)
        # convert to rasterio profile format
        width, height = int(dict_common["ncols"]), int(dict_common["nrows"])
        cell_size = float(dict_common["cellsize"])
        nodata = (
            eval(dict_common["nodata_value"]) if dict_common["nodata_value"] else None
        )

        # get the coordinates of left and bottom corner
        dict_corner = load_metas(
            ascii_file,
            keys=["xllcorner", "yllcorner"],
            line_end=10,
        )
        if (
            dict_corner["xllcorner"] is not None
            and dict_corner["yllcorner"] is not None
        ):
            left = float(dict_corner["xllcorner"])
            bottom = float(dict_corner["yllcorner"])
        else:
            dict_center = load_metas(
                ascii_file,
                keys=["xllcenter", "yllcenter"],
                line_end=10,
            )
            if dict_center["xllcenter"] is None or dict_center["yllcenter"] is None:
                msg = (
                    "xllcenter and yllcenter or xllcorner and yllcorner"
                    "must be set in the ascii file"
                )
                raise ValueError(msg)

            left = float(dict_center["xllcenter"]) - cell_size / 2
            bottom = float(dict_center["yllcenter"]) - cell_size / 2

        # pixel left lower corner to pixel left upper corner (rasterio transform)
        top = bottom + (height + 1) * cell_size
        # get affine transform
        tf = transform.from_origin(left, top, cell_size, cell_size)
        geogrid = GeoGrid(tf, (height, width))
        profile_kwargs = {"nodata": nodata}
        profile_kwargs.update(kwargs)
        return cls.from_geogrid(geogrid, **profile_kwargs)

    @classmethod
    def from_xy(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        crs: CrsLike = "WGS84",
        **kwargs: Any,
    ) -> Profile:
        """Create a Profile object from x and y coordinates.

        Parameters
        ----------
        x, y : ArrayLike
            X and Y coordinates of pixel centers.
        crs : CrsLike, optional
            Coordinate reference system of the coordinates. Default is
            ``"WGS84"``.
        **kwargs : Any
            Additional profile metadata such as ``nodata``, ``count``,
            ``driver``, ``dtype``, or other rasterio profile options.

        Returns
        -------
        Profile
            Profile object created from x and y coordinates.

        """
        geogrid = GeoGrid.from_xy(x, y, crs=crs)
        return cls.from_geogrid(geogrid, **kwargs)

    @classmethod
    def from_profile_file(cls, profile_file: PathLike, **kwargs: Any) -> Profile:
        """Create a Profile object from a profile file.

        Parameters
        ----------
        profile_file : PathLike
            Profile file used to initialize the profile.
        **kwargs : Any
            Additional profile metadata. Values in ``kwargs`` override metadata
            loaded from the profile file.

        """
        profile = eval(Path(profile_file).read_text(encoding="utf-8"))
        profile.update(kwargs)
        profile, kwargs = cls._split_profile(profile)
        return cls(**profile, **kwargs)

    @classmethod
    def from_bounds(
        cls,
        bounds: tuple[float, float, float, float] | BoundingBox,
        res: float | tuple[float, float],
        crs: CrsLike | None = None,
        **kwargs: Any,
    ) -> Profile:
        """Create a Profile object from bounds and resolution.

        Parameters
        ----------
        bounds : tuple of float (left/W, bottom/S, right/E, top/N)
            The bounds of the raster file.
        res : float or tuple of float (x_res, y_res)
            The resolution of the raster file. If a float is provided,
            the x_res and y_res will be the same.
        crs : CrsLike | None, optional
            The coordinate reference system of the raster file.
        **kwargs : Any
            Additional profile metadata such as ``nodata``, ``count``,
            ``driver``, ``dtype``, or other rasterio profile options.

        Returns
        -------
        Profile : Profile
            A Profile object only with width, height and transform.

        """
        if isinstance(res, (int, float, np.integer, np.floating)):
            res = (float(res), float(res))
        geogrid = GeoGrid.from_bounds(bounds, res=res, crs=crs)
        return cls.from_geogrid(geogrid, **kwargs)

    def copy(self) -> Profile:
        """Return a copy of the Profile object."""
        profile, kwargs = self._split_profile(self.to_dict())
        return Profile(**profile, **kwargs)

    def to_dict(self) -> dict:
        """Convert the Profile object to a python :class:`dict`."""
        profile = {key: getattr(self, key) for key in DEFAULT_KEYS_Profile}
        profile.update(self.kwargs)
        return profile

    def to_file(self, out_file: PathLike) -> None:
        """Write the profile into a file.

        .. tip::
            - The profile will be written into a file with the same name and
            suffix ".profile".
            - You can load the profile by :meth:`Profile.from_profile_file`.

        Parameters
        ----------
        out_file : str or Path
            The file to be written. The profile will be written into a file with
            the same name and suffix ".profile".

        """
        out_file = Path(out_file)
        if out_file.suffix != ".profile":
            out_file = out_file.parent / (out_file.name + ".profile")
        with out_file.open("w") as f:
            f.write(str(self.to_dict()))

    def to_rasterio_profile(self) -> RasterioProfile:
        """Convert the Profile object to a rasterio profile."""
        return RasterioProfile(data=self.to_dict())

    def get_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the x and y coordinates from profile data.

        .. note::
            The pixel location for the x and y coordinates is the
            "PixelIsArea" Raster Space, which means the pixel location
            is the center of the pixel. See `Raster Space <https://web.archive.org/web/20160326194152/http://remotesensing.org/geotiff/spec/geotiff2.5.html#2.5.2>`_
            for more details.
        """
        return xy_from_transform(self.transform, self.width, self.height)
