"""A module to provide some useful tools for geospatial data processing."""

from __future__ import annotations

import pprint
import zipfile
from collections.abc import Callable, Iterable, Iterator, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from lxml import etree
from matplotlib import ticker
from pykml.factory import KML_ElementMaker as KML
from pyproj.crs import CRS
from rasterio import Affine, dtypes, transform
from rasterio.io import MemoryFile
from rasterio.profiles import Profile as RasterioProfile
from rasterio.transform import array_bounds, rowcol, xy
from rasterio.warp import Resampling, calculate_default_transform, reproject
from tqdm import tqdm

from faninsar._core.file_tools import load_metas
from faninsar.logging import setup_logger
from faninsar.query.bbox import BoundingBox

if TYPE_CHECKING:
    from os import PathLike
    from typing import Self

    from matplotlib.cm import ScalarMappable
    from numpy.typing import ArrayLike
    from odc.geo import GeoBox

    from faninsar.typing import CrsLike

logger = setup_logger(__name__)


OFFSET_LOCATIONS: dict[
    Literal["center", "ul", "ur", "ll", "lr"], tuple[float, float]
] = {
    "center": (0.5, 0.5),
    "ul": (0, 0),
    "ur": (1, 0),
    "ll": (0, 1),
    "lr": (1, 1),
}


def _offset_from_loc(
    loc: Literal["center", "ul", "ur", "ll", "lr"],
) -> tuple[float, float]:
    """Get the offset from pixel location."""
    if loc not in OFFSET_LOCATIONS:
        msg = f"loc should be one of {tuple(OFFSET_LOCATIONS.keys())}, but got {loc}"
        logger.error(msg)
        raise ValueError(msg)
    return OFFSET_LOCATIONS[loc]


def _ensure_bounds_in_wgs84(
    bounds: tuple[float, ...],
) -> tuple[float, float, float, float]:
    """Ensure the bounds are in WGS84 coordinate system."""
    if len(bounds) != 4:
        msg = (
            f"bounds should be a tuple of (west, south, east, north), but got {bounds}"
        )
        logger.error(msg)
        raise ValueError(msg)
    west, south, east, north = bounds
    if west < -180 or east > 180 or south < -90 or north > 90:
        msg = (
            "bounds should be in WGS84 coordinate system, "
            f"but got [{west}, {south}, {east}, {north}]"
        )
        raise ValueError(msg)
    return west, south, east, north


def save_colorbar(
    out_file: PathLike,
    mappable: ScalarMappable,
    figsize: tuple[float, float] = (0.18, 3.6),
    label: str | None = None,
    nbins: int | None = None,
    alpha: float = 0.5,
    **kwargs,
) -> None:
    """Save the colorbar to a file.

    Parameters
    ----------
    out_file: str or Path
        the path of the output colorbar file.
    mappable: ScalarMappable
        the ScalarMappable object to be used for the colorbar.
    figsize: tuple
        the size of the colorbar figure. Default is (0.18, 3.6).
    label: str
        the label of the colorbar. Default is None.
    nbins: int
        the number of bins of the colorbar. Default is None.
    alpha: float
        the transparency of the colorbar figure. Default is 0.5.
    kwargs: dict
        the keyword arguments for :func:`matplotlib.pyplot.colorbar` function.

    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    kwargs.update({"fraction": 1})
    cbar = fig.colorbar(mappable, ax=ax, **kwargs)

    # update colorbar label and ticks
    if label:
        cbar.set_label(label, fontsize=12)
    if nbins:
        cbar.locator = ticker.MaxNLocator(nbins=nbins)
        cbar.update_ticks()

    cbar.ax.tick_params(which="both", labelsize=12)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(alpha)
    fig.savefig(out_file, bbox_inches="tight", dpi=300)


def array2kml(
    arr: np.ndarray,
    out_file: PathLike,
    bounds: tuple[float, float, float, float] | BoundingBox,
    img_kwargs: dict | None = None,
    cbar_kwargs: dict | None = None,
    verbose: bool = True,
) -> None:
    """Write a numpy array into a kml file.

    Parameters
    ----------
    arr: numpy.ndarray
        the numpy array to be written into kml file.
    out_file: str or Path
        the path of the kml file.
    bounds: tuple or BoundingBox
        the bounds of image in [west, south, east, north] order in WGS84.
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
    wgs84 = CRS.from_user_input("EPSG:4326")
    if not isinstance(bounds, BoundingBox) and isinstance(bounds, Iterable):
        bounds = _ensure_bounds_in_wgs84(tuple(bounds))
        bounds = BoundingBox(*bounds, crs=wgs84)
    if bounds.crs != CRS.from_user_input(wgs84):
        bounds = bounds.to_crs(wgs84)

    out_file = Path(out_file)
    if out_file.suffix != ".kml":
        out_file = out_file.parent / (out_file.stem + ".kml")
    img_file = out_file.parent / (out_file.stem + ".png")
    cbar_file = out_file.parent / (out_file.stem + "_cbar.png")

    # plot image
    figsize = (arr.shape[1] / 100, arr.shape[0] / 100)
    plt.figure(figsize=figsize)
    im = plt.imshow(arr, **img_kwargs)
    plt.axis("off")
    plt.savefig(img_file, bbox_inches="tight", pad_inches=0, dpi=100, transparent=True)
    plt.close()

    # plot colorbar
    save_colorbar(cbar_file, im, **cbar_kwargs)

    # write kml file
    kml_doc = KML.Document()
    img_overlay = KML.GroundOverlay(
        KML.Icon(
            KML.href(img_file.name),
            KML.viewBoundScale(1),  # 设置 viewBoundScale 为 1
            KML.scale(1),  # 设置 scale 为 1),
            KML.size(
                x=str(arr.shape[1]),
                y=str(arr.shape[0]),
                xunits="pixels",
                yunits="pixels",
            ),
        ),
        KML.LatLonBox(
            KML.north(bounds[3]),
            KML.south(bounds[1]),
            KML.east(bounds[2]),
            KML.west(bounds[0]),
        ),
    )
    kml_doc.append(img_overlay)

    # colorbar overlay
    cbar_overlay = KML.ScreenOverlay(
        KML.name("Color bar"),
        KML.Icon(KML.href(cbar_file.name)),
        KML.overlayXY(x="1", y="0", xunits="fraction", yunits="fraction"),
        KML.screenXY(x="1", y="0", xunits="fraction", yunits="fraction"),
        KML.size(x="0", y="500", xunits="pixel", yunits="pixel"),
    )
    kml_doc.append(cbar_overlay)

    kml = KML.kml(kml_doc)
    Path(out_file).write_text(
        etree.tostring(kml, pretty_print=True).decode("utf8"), encoding="utf-8"
    )
    if verbose:
        info = f"write kml file to {out_file}"
        logger.info(info)


def array2kmz(
    arr: np.ndarray,
    out_file: PathLike,
    bounds: tuple[float, float, float, float] | BoundingBox,
    img_kwargs: dict | None = None,
    cbar_kwargs: dict | None = None,
    keep_kml: bool = False,
    verbose: bool = True,
) -> None:
    """Write a numpy array into a kmz file.

    Parameters
    ----------
    arr: numpy.ndarray
        the numpy array to be written into kml file.
    out_file: str or Path
        the path of the kmz file.
    bounds: tuple or BoundingBox
        the bounds of image in [west, south, east, north] order in WGS84
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
    out_file = Path(out_file)
    if out_file.suffix != ".kmz":
        out_file = out_file.parent / (out_file.stem + ".kmz")
    img_file = out_file.parent / (out_file.stem + ".png")
    cbar_file = out_file.parent / (out_file.stem + "_cbar.png")

    kml_file = out_file.parent / (out_file.stem + ".kml")
    array2kml(arr, kml_file, bounds, img_kwargs, cbar_kwargs, verbose=False)
    with zipfile.ZipFile(out_file, "w") as kmz:
        kmz.write(kml_file, kml_file.name)
        kmz.write(img_file, img_file.name)
        kmz.write(cbar_file, cbar_file.name)
    if not keep_kml:
        img_file.unlink()
        cbar_file.unlink()
        kml_file.unlink()
    if verbose:
        info = f"write kmz file to {out_file}"
        logger.info(info)


def transform_from_xy(
    x: ArrayLike,
    y: ArrayLike,
    *,
    loc: Literal["center", "ul", "ur", "ll", "lr"] = "center",
) -> Affine:
    """Get the :class:`rasterio.Affine` from x and y coordinates.

    Parameters
    ----------
    x, y: ArrayLike
        x and y coordinates
    loc: Literal["center", "ul", "ur", "ll", "lr"], optional
        The pixel location that the coordinates refer to. Supported values are
        "center", "ul", "ur", "ll", and "lr". Default is "center".

    """
    west, south, east, north = (
        np.nanmin(x),
        np.nanmin(y),
        np.nanmax(x),
        np.nanmax(y),
    )
    width, height = len(x), len(y)

    xsize = (east - west) / width
    ysize = (north - south) / height

    offset = _offset_from_loc(loc)

    return transform.from_origin(
        west - offset[0] * xsize,  # center to left
        north + offset[1] * ysize,  # center to top
        xsize,
        ysize,
    )


def bound_from_xy(
    x: ArrayLike,
    y: ArrayLike,
    *,
    loc: Literal["center", "ul", "ur", "ll", "lr"] = "center",
    crs: CrsLike = "WGS84",
) -> BoundingBox:
    """Get the bounds from x and y coordinates.

    Parameters
    ----------
    x, y: ArrayLike
        x and y coordinates
    loc: Literal["center", "ul", "ur", "ll", "lr"], optional
        The pixel location that the coordinates refer to. Supported values are
        "center", "ul", "ur", "ll", and "lr". Default is "center".
    crs: CrsLike, optional
        the coordinate reference system. Could be any type that accepted by
        :meth:`pyproj.CRS.from_user_input`. Default is "WGS84".

    """
    width, height = len(x), len(y)
    tf = transform_from_xy(x, y, loc=loc)
    left, top = tf * (0, 0)
    right, bottom = tf * (width, height)
    return BoundingBox(left, bottom, right, top, crs=crs)


def geoinfo_from_xy(
    x: ArrayLike,
    y: ArrayLike,
    *,
    crs: CrsLike = "WGS84",
    loc: Literal["center", "ul", "ur", "ll", "lr"] = "center",
) -> tuple[BoundingBox, Affine, tuple, tuple]:
    """Evaluate the geoinformation from x and y coordinates.

    Parameters
    ----------
    x, y: numpy.ndarray or list
        x and y coordinates
    loc: Literal["center", "ul", "ur", "ll", "lr"], optional
        The pixel location that the coordinates refer to. Supported values are
        "center", "ul", "ur", "ll", and "lr". Default is "center".
    crs: CrsLike, optional
        the coordinate reference system. Could be any type that accepted by
        :meth:`pyproj.CRS.from_user_input`. Default is "WGS84".

    Returns
    -------
    bounds: BoundingBox
        the bounding box of the raster.
    transform: Affine
        the affine transform of the raster.
    res: tuple[xsize, ysize]
        the resolution of the raster
    shape: tuple[height, width]
        the shape of the raster

    """
    tf = transform_from_xy(x, y, loc=loc)
    res = (abs(tf.a), abs(tf.e))

    width, height = len(x), len(y)
    shape = (height, width)

    left, top = tf * (0, 0)
    right, bottom = tf * (width, height)
    bounds = BoundingBox(left, bottom, right, top, crs=crs)

    return bounds, tf, res, shape


def xy_from_transform(
    tf: Affine | None,
    width: int,
    height: int,
    *,
    loc: Literal["center", "ul", "ur", "ll", "lr"] = "center",
) -> tuple[np.ndarray, np.ndarray]:
    """Get the x and y coordinates from transform and shape.

    Parameters
    ----------
    tf: Affine | None
        the transform of the raster. If tf is None, the x and y coordinates will
        be range(width) and range(height).
    width, height: int
        the width and height of the raster
    loc: Literal["center", "ul", "ur", "ll", "lr"], optional
        the pixel location that the coordinates refer to. Supported values are
        "center", "ul", "ur", "ll", and "lr". Default is "center".

    Returns
    -------
    x, y: numpy.ndarray

    """
    if tf is None:
        return np.arange(width), np.arange(height)
    offset = _offset_from_loc(loc)
    x = tf.xoff + tf.a * (np.arange(width) + offset[0])
    y = tf.yoff + tf.e * (np.arange(height) + offset[1])
    return x, y


def xy_from_profile(profile: RasterioProfile) -> tuple[np.ndarray, np.ndarray]:
    """Get the x and y coordinates from rasterio profile data.

    Parameters
    ----------
    profile: Profile
        the profile data of rasterio dataset. It can be get from
        rasterio.open().profile

    Returns
    -------
    x, y: numpy.ndarray

    """
    tf = profile["transform"]
    width = profile["width"]
    height = profile["height"]
    return xy_from_transform(tf, width, height)


@overload
def write_geoinfo_into_ds(
    ds: xr.DataArray,
    var: None = None,
    crs: CrsLike = "EPSG:4326",
    x_dim: str = "lon",
    y_dim: str = "lat",
) -> xr.DataArray: ...
@overload
def write_geoinfo_into_ds(
    ds: xr.Dataset,
    var: str | tuple | list,
    crs: CrsLike = "EPSG:4326",
    x_dim: str = "lon",
    y_dim: str = "lat",
) -> xr.Dataset: ...
def write_geoinfo_into_ds(
    ds: xr.DataArray | xr.Dataset,
    var: str | tuple | list | None = None,
    crs: CrsLike = "EPSG:4326",
    x_dim: str = "lon",
    y_dim: str = "lat",
) -> xr.DataArray | xr.Dataset:
    """Write geoinformation in to the given xr DataArray or DataSet.

    Parameters
    ----------
    ds: xarray DataArray or DataSet object
        data to be written into geoinfo.If type of ds is DataSet,
        var should be set
    var: str, tuple or list
        variables that need to be added geoinformation
    crs: CrsLike
        the coordinate reference system. Could be any type that
        :meth:`rasterio.crs.CRS.from_user_input` accepts.
    x_dim: str
        the coordinate name that presents the x dimension
    y_dim: str
        the coordinate name that presents the y dimension

    """
    if isinstance(ds, xr.DataArray):
        ds = ds.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
        ds = ds.rio.write_crs(crs)
    elif isinstance(var, str):
        ds[var] = ds[var].rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
        ds[var] = ds[var].rio.write_crs(crs)
    elif isinstance(var, (tuple, list)):
        for var_ in var:
            ds[var_] = ds[var_].rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
            ds[var_] = ds[var_].rio.write_crs(crs)
    elif var is None:
        msg = "Detected type of ds is a xr.Dataset. var must be set"
        raise TypeError(msg)
    else:
        msg = "var type must be one of [str,tuple,list]"
        raise TypeError(msg)
    return ds


def write_geoinfo_into_nc(
    nc_file: PathLike,
    var: str | tuple | list | None = None,
    crs: CrsLike = "EPSG:4326",
    x_dim: str = "lon",
    y_dim: str = "lat",
    encode_time: bool = False,
) -> None:
    """Write geoinformation in to the given nc file.

    This make it could be opened with geoinformation in QGIS directly.

    Parameters
    ----------
    nc_file: str or pathlib.Path object
        the path of nc file
    var: str, tuple or list
        variables that need to be added geoinformation
    crs: CrsLike
        the coordinate reference system. Could be any type that
        :meth:`rasterio.crs.CRS.from_user_input` accepts.
    x_dim: str
        the coordinate name that presents the x dimension
    y_dim: str
        the coordinate name that presents the y dimension
    encode_time: bool
        whether to encode the time since "2000-01-01 00:00:00" if
        "time" coordinate is exists. Default is False.

    """
    ds = xr.load_dataset(nc_file)

    ds = write_geoinfo_into_ds(ds, var, crs, x_dim, y_dim)

    encode = {}
    if encode_time:
        if "time" in ds:
            encode.update({"time": {"units": "days since 2000-01-01 00:00:00"}})
        else:
            info = (
                f'there is no "time" dimension in {nc_file}, '
                "encoding process will be ignored"
            )
            logger.warning(info)
    ds.to_netcdf(nc_file, encoding=encode)


def match_to_raster(
    src_arr: np.ndarray,
    src_profile: Profile,
    dst_profile: Profile,
    algorithm: Resampling = Resampling.nearest,
) -> np.ndarray:
    """Match the source raster to the destination raster.

    Parameters
    ----------
    src_arr: numpy.ndarray
        the source array to be matched.
    src_profile: Profile
        the profile of the source raster.
    dst_profile: Profile
        the profile of the destination raster.
    algorithm: Resampling
        the resampling algorithm. Default is Resampling.nearest.

    Returns
    -------
    numpy.ndarray
        the matched array.

    """
    src_crs = src_profile["crs"]
    src_tf = src_profile["transform"]
    dst_height = dst_profile["height"]
    dst_width = dst_profile["width"]
    dst_crs = dst_profile["crs"]
    dst_tf = dst_profile["transform"]
    nodata = dst_profile["nodata"]

    if src_arr.ndim == 2:
        indexes = 1
        src_n_band = 1
        src_height, src_width = src_arr.shape
    elif src_arr.ndim == 3:
        src_n_band, src_height, src_width = src_arr.shape
        indexes = np.arange(1, src_n_band + 1).tolist()
    else:
        msg = "dimension of src_arr must be 2 or 3"
        raise ValueError(msg)
    with (
        MemoryFile() as memfile,
        memfile.open(
            driver="GTiff",
            count=src_n_band,
            height=src_height,
            width=src_width,
            dtype="float32",
            crs=src_crs,
            transform=src_tf,
        ) as src,
    ):
        src.write(src_arr, indexes)

        with (
            MemoryFile() as memfile1,
            memfile1.open(
                driver="GTiff",
                count=src_n_band,
                height=dst_height,
                width=dst_width,
                dtype="float32",
                crs=dst_crs,
                transform=dst_tf,
            ) as dst,
        ):
            if indexes == 1:
                indexes = [1]
            for i in tqdm(indexes, desc="matching raster"):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst.transform,
                    dst_crs=dst.crs,
                    resampling=algorithm,
                    dst_nodata=nodata,
                )
            arr_dst = dst.read(indexes)
    return arr_dst


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


class GeoGridMixin:
    """A class to manage GeoGrid information of a raster image.

    A GeoGrid is a fixed pixel grid reference system for geospatial data. It
    represents a grid system with fixed pixel resolution, alignment, and coordinate
    reference system (CRS). Different spatial extents can be derived from the
    same grid, ensuring consistent pixel grid alignment across all views.
    """

    def _refresh_bounds(self) -> None:
        """Refresh cached bounds after geometry updates."""
        if (
            hasattr(self, "_transform")
            and hasattr(self, "_shape")
            and hasattr(self, "_crs")
        ):
            self._bounds = self._parse_bounds()

    def _parse_bounds(self) -> BoundingBox:
        """Parse the bounds from geogrid data."""
        west, south, east, north = array_bounds(self.height, self.width, self.transform)
        return BoundingBox(west, south, east, north, crs=self.crs)

    @property
    def transform(self) -> Affine:
        """The transform of raster image."""
        return self._transform

    @transform.setter
    def transform(self, value: Affine) -> None:
        """Set the transform of raster image."""
        self._transform = value
        self._refresh_bounds()

    @property
    def north_up(self) -> bool:
        """Whether the raster image is north-up."""
        return self.transform.e < 0

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of raster image in (height, width) order."""
        return self._shape

    @shape.setter
    def shape(self, value: tuple[int, int]) -> None:
        """Set the shape of raster image in (height, width) order."""
        if len(value) != 2:
            msg = "shape must be a tuple of (height, width)"
            logger.error(msg)
            raise ValueError(msg)
        self._shape = (int(value[0]), int(value[1]))
        self._refresh_bounds()

    @property
    def res(self) -> tuple[float, float]:
        """The resolution of raster image in x and y direction."""
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def width(self) -> int:
        """The width of raster image."""
        return self.shape[1]

    @width.setter
    def width(self, value: int) -> None:
        """Set the width of raster image."""
        self._shape = (self.shape[0], int(value))
        self._refresh_bounds()

    @property
    def height(self) -> int:
        """The height of raster image."""
        return self.shape[0]

    @height.setter
    def height(self, value: int) -> None:
        """Set the height of raster image."""
        self._shape = (int(value), self.shape[1])
        self._refresh_bounds()

    @property
    def crs(self) -> CRS | None:
        """The coordinate reference system of raster image."""
        return self._crs

    @crs.setter
    def crs(self, value: CrsLike | None) -> None:
        """Set the coordinate reference system of raster image."""
        self._crs = CRS.from_user_input(value) if value is not None else None
        self._refresh_bounds()

    @property
    def bounds(self) -> BoundingBox:
        """The bounds of the GeoGrid in (west, south, east, north) order."""
        return self._bounds

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """The extent of the GeoGrid in (west, east, south, north) order."""
        b = self.bounds
        return (b.left, b.right, b.bottom, b.top)


class GeoGrid(GeoGridMixin):
    """A fixed pixel grid reference system for geospatial data.

    GeoGrid represents a grid system with fixed resolution, pixel alignment,
    and coordinate reference system (CRS). Different spatial extents can be
    derived from the same grid, ensuring consistent pixel grid alignment
    across all views.

    Parameters
    ----------
    transform: Affine
        Affine transformation matrix mapping pixel coordinates to map coordinates.
    shape: tuple[int, int]
        Grid dimensions as (height, width) in pixels.
    crs: CrsLike, optional
        Coordinate reference system. Accepts any type compatible with
        :meth:`pyproj.CRS.from_user_input`. Default is None (unset).

    """

    def __init__(
        self,
        transform: Affine,
        shape: tuple[int, int],
        crs: CrsLike | None = None,
    ) -> None:
        """Initialize the GeoBox class."""
        self._transform = transform
        self.crs = crs
        self.shape = shape

    def __repr__(self) -> str:
        """Get the string representation of the GeoGrid."""
        info = {
            "bounds": self.bounds.to_tuple(),
            "transform": self.transform,
            "shape": self.shape,
            "crs": self.crs.to_string() if self.crs else None,
        }
        repr_str = f" {pprint.pformat(info, indent=2, sort_dicts=False).strip('{}')}"
        return f"GeoGrid(\n{repr_str}\n)"

    @classmethod
    def from_bounds(
        cls,
        bounds: BoundingBox | tuple[float, float, float, float],
        *,
        res: float | tuple[float, float] | None = None,
        shape: tuple[int, int] | None = None,
        crs: CrsLike | None = None,
    ) -> Self:
        """Create a GeoGrid from bounds and shape.

        Parameters
        ----------
        bounds: BoundingBox or tuple[float, float, float, float]
            the bounds of the raster image in [west, south, east, north] order.
            The bounds will be converted to the :param:`crs` if crs is not None,
            otherwise the crs of bounds will be used.
        res: float or tuple[float, float] | None, optional
            the resolution of the raster image in x and y direction. If res is a
            single float, it will be used for both x and y direction. If None,
            the resolution will be calculated from bounds and shape.
        shape: tuple[int, int] | None, optional
            the shape of the raster image in (height, width) order, which will
            be ignored if res is not None.
        crs: CrsLike | None, optional
            the coordinate reference system of the raster image. Could be any
            type that :meth:`pyproj.CRS.from_user_input` accepts. If None, the
            crs of bounds will be used.

        Returns
        -------
        GeoGrid
             the GeoGrid object created from bounds and shape.

        Raises
        ------
        ValueError
            if neither res nor shape is provided

        """
        bounds, crs = format_bounds_and_crs(bounds, crs)
        left, bottom, right, top = bounds
        if res is not None:
            if isinstance(res, (int, float)):
                res = (res, res)
            xsize = abs(float(res[0]))
            ysize = abs(float(res[1]))
            width = int(np.ceil((right - left) / xsize))
            height = int(np.ceil((top - bottom) / ysize))
            shape = (height, width)
        elif shape is not None:
            width, height = shape[1], shape[0]
            xsize = (right - left) / width
            ysize = (top - bottom) / height
        else:
            msg = "either res or shape must be provided"
            logger.error(msg)
            raise ValueError(msg)
        tf = transform.from_origin(left, top, xsize, ysize)

        return cls(tf, shape, crs)

    @classmethod
    def from_xy(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        crs: CrsLike = "WGS84",
        loc: Literal["center", "ul", "ur", "ll", "lr"] = "center",
    ) -> Self:
        """Create a GeoGrid from x and y coordinates.

        Parameters
        ----------
        x, y: ArrayLike
            x and y coordinates of pixel centers in raster image.
        crs: CrsLike, optional
            the coordinate reference system of the input coordinates. Could be
            any type that accepted by :meth:`pyproj.CRS.from_user_input`.
            Default is "WGS84".
        loc: str, optional
            the location of the coordinates in pixel. It can be "center", "ul",
            "ur", "ll" or "lr". Default is "center".

        Returns
        -------
        GeoGrid
            the GeoGrid object created from x and y coordinates.

        """
        bounds, tf, _, shape = geoinfo_from_xy(x, y, crs=crs, loc=loc)
        return cls(tf, shape, bounds.crs)

    def to_crs(
        self,
        crs: CrsLike,
        *,
        res: float | tuple[float, float] | None = None,
        shape: tuple[int, int] | None = None,
    ) -> GeoGrid:
        """Get a new GeoGrid reprojected to the destination CRS.

        Parameters
        ----------
        crs: CrsLike
            the destination coordinate reference system. Could be any type that
            :meth:`pyproj.CRS.from_user_input` accepts.
        res: float | tuple[float, float] | None, optional
            Target resolution, in units of target coordinate reference system.
            Default is None.
        shape: tuple[x resolution, y resolution] | None, optional
            the shape of the new GeoGrid in (height, width) order. Cannot be set
            if res is not None. Default is None.

        Returns
        -------
        GeoGrid
            New GeoGrid reprojected to the destination CRS with aligned pixel grid.

        """
        left, bottom, right, top = self.bounds
        dst_width, dst_height = None, None
        if shape is not None:
            dst_width, dst_height = shape[1], shape[0]
        tf, width, height = calculate_default_transform(
            self.crs,
            crs,
            self.width,
            self.height,
            left,
            bottom,
            right,
            top,
            dst_width=dst_width,
            dst_height=dst_height,
            resolution=res,
        )

        # Create a new GeoGrid with the same resolution and shape but new bounds and CRS
        return GeoGrid(tf, (height, width), crs)

    def to_view(self, roi: BoundingBox) -> GeoGrid:
        """Create a new GeoGrid with the view focused on the roi region.

        The new GeoGrid shares the same pixel resolution and alignment as the
        original, only view window (bounds) change.

        Parameters
        ----------
        roi: BoundingBox
            Region of interest in BoundingBox format. The bounds will be
            converted to the GeoGrid's CRS if needed.

        Returns
        -------
        GeoGrid
            New GeoGrid windowed to the roi region with aligned pixel grid.

        """
        roi, _ = format_bounds_and_crs(roi, self.crs)
        if roi == self.bounds:
            return self
        xsize = abs(float(self.res[0]))
        ysize = abs(float(self.res[1]))

        left, bottom, right, top = roi
        rows, cols = rowcol(
            self.transform,
            [left, right, right, left],
            [top, top, bottom, bottom],
            op=float,
        )
        w, n = self.transform * (min(cols), min(rows))
        tf = transform.from_origin(w, n, xsize, ysize)
        shape = (max(rows) - min(rows), max(cols) - min(cols))

        return GeoGrid(tf, shape, self.crs)

    def get_xy(
        self, loc: Literal["center", "ul", "ur", "ll", "lr"] = "center"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the x and y coordinates of pixel centers of the raster image."""
        return xy_from_transform(self.transform, self.width, self.height, loc=loc)

    def to_geobox(self) -> GeoBox:
        """Convert the GeoGrid to an odc.geo.GeoBox."""
        from odc.geo import GeoBox

        return GeoBox(self.shape, self.transform, self.crs)

    def row_col(
        self, x: ArrayLike, y: ArrayLike, op: Callable | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the row and column indices for the given x and y coordinates.

        Parameters
        ----------
        x, y: ArrayLike
            x and y coordinates to be converted to row and column indices.
        op: Callable, optional
            Function to convert fractional pixels to whole numbers (floor,
            ceiling, round). If None, numpy.floor will be used. Default is None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            row and column indices corresponding to the input x and y coordinates.

        """
        return rowcol(self.transform, x, y, op=op)

    def xy(
        self,
        row: ArrayLike,
        col: ArrayLike,
        offset: Literal["center", "ul", "ur", "ll", "lr"] = "center",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the x and y coordinates for the given row and column indices.

        Parameters
        ----------
        row, col: ArrayLike
            row and column indices to be converted to x and y coordinates.
        offset: str, optional
            Determines if the returned coordinates are for the center of the
            pixel or for a corner. It can be "center", "ul", "ll" or "lr".
            Default is "center".

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            x and y coordinates corresponding to the input row and column indices.

        """
        return xy(self.transform, row, col, offset=offset)


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
        **kwargs: dict[str, Any],
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
        return cls(**profile, kwargs=kwargs_extra)

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
        return cls(**profile, kwargs=kwargs)

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
        return cls(**profile, kwargs=kwargs)

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
        return Profile(**profile, kwargs=kwargs)

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


class GeoDataFormatConverter:
    """A class to convert data format between raster and binary.

    Examples
    --------
    ::

        >>> from pathlib import Path
        >>> from data_tool import GeoDataFormatConverter
        >>> phase_file = Path("phase.tif")
        >>> amplitude_file = Path("amplitude.tif")
        >>> binary_file = Path("phase.int")

        load/add raster and convert to binary

        >>> gfc = GeoDataFormatConverter()
        >>> gfc.load_raster(phase_file)
        >>> gfc.add_band_from_raster(amplitude_file)
        >>> gfc.to_binary(binary_file)

        load binary file

        >>> gfc.load_binary(binary_file)
        >>> print(gfc.arr.shape)

    """

    def __init__(self) -> None:
        """Initialize the GeoDataFormatConverter class."""
        self.arr: np.ndarray | None = None
        self.profile: RasterioProfile | None = None

    @property
    def _profile_str(self) -> str:
        return pprint.pformat(self.profile, sort_dicts=False)

    def __str__(self) -> str:
        """Return the string representation of the class."""
        return f"DataConverter: \n{self._profile_str}"

    def __repr__(self) -> str:
        """Return the string representation of the class."""
        return str(self)

    def _load_raster(
        self,
        raster_file: PathLike,
    ) -> tuple[np.ndarray, RasterioProfile]:
        """Load a raster file into the data array."""
        with rasterio.open(raster_file) as ds:
            arr = ds.read()
            profile = ds.profile.copy()
        return arr, profile

    def load_binary(
        self,
        binary_file: PathLike,
        order: Literal["BSQ", "BIP", "BIL"] = "BSQ",
        dtype: str | np.dtype = "auto",
    ) -> None:
        """Load a binary file into the data array.

        Parameters
        ----------
        binary_file : str or Path
            The binary file to be loaded. the binary file should be with a profile
            file with the same name.
        order : str, one of ['BSQ', 'BIP', 'BIL']
            The order of the data array. 'BSQ' for band sequential, 'BIP' for band
            interleaved by pixel, 'BIL' for band interleaved by line.
            Default is 'BSQ'.
            More details can be found at:
            https://desktop.arcgis.com/zh-cn/arcmap/latest/manage-data/raster-and-images/bil-bip-and-bsq-raster-files.htm
        dtype : str or numpy.dtype
            The dtype of the array. If 'auto', the minimum dtype will be used.
            Default is 'auto'.

        """
        binary_profile_file = str(binary_file) + ".profile"
        if not Path(binary_profile_file).exists():
            msg = f"{binary_profile_file} not found"
            raise FileNotFoundError(msg)

        with Path(binary_profile_file).open(encoding="utf-8") as f:
            profile = eval(f.read())

        # todo: auto detect dtype by shape
        if dtype == "auto":
            dtype = "float32"

        arr = np.fromfile(binary_file, dtype=dtype)
        if order == "BSQ":
            arr = arr.reshape(profile["count"], profile["height"], profile["width"])
        elif order == "BIP":
            arr = arr.reshape(
                profile["height"],
                profile["width"],
                profile["count"],
            ).transpose(2, 0, 1)
        elif order == "BIL":
            arr = arr.reshape(
                profile["height"],
                profile["count"],
                profile["width"],
            ).transpose(1, 0, 2)
        else:
            msg = f"order should be one of ['BSQ', 'BIP', 'BIL'], but got {order}"
            raise ValueError(msg)

        if "dtype" not in profile:
            profile["dtype"] = dtypes.get_minimum_dtype(arr)

        self.arr = arr
        self.profile = profile

    def load_raster(self, raster_file: PathLike) -> None:
        """Load a raster file into the data array.

        Parameters
        ----------
        raster_file : str or Path
            The raster file to be loaded. raster format should be supported by gdal.
            More details can be found at: https://gdal.org/drivers/raster/index.html

        """
        self.arr, self.profile = self._load_raster(raster_file)

    def to_binary(
        self,
        out_file: PathLike,
        order: Literal["BSQ", "BIP", "BIL"] = "BSQ",
    ) -> None:
        """Write the data array into a binary file.

        Parameters
        ----------
        out_file : str or Path
            The binary file to be written. the binary file will be with a profile
            file with the same name.
        order : str, one of ['BSQ', 'BIP', 'BIL']
            The order of the data array. 'BSQ' for band sequential, 'BIP' for
            band interleaved by pixel, 'BIL' for band interleaved by line.
            Default is 'BSQ'.
            More details can be found at:
            https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/bil-bip-and-bsq-raster-files.htm

        """
        if self.arr is None:
            msg = "data array is not set yet"
            raise AttributeError(msg)

        if order == "BSQ":
            arr = self.arr
        elif order == "BIL":
            arr = np.transpose(self.arr, (1, 2, 0))
        elif order == "BIP":
            arr = np.transpose(self.arr, (1, 0, 2))

        # write data into a binary file
        (arr.astype("float32").tofile(out_file))

        # write profile into a file with the same name
        out_profile_file = str(out_file) + ".profile"
        Path(out_profile_file).write_text(self._profile_str, encoding="utf-8")

    def to_raster(self, out_file: PathLike, driver: str = "GTiff") -> None:
        """Write the data array into a raster file.

        Parameters
        ----------
        out_file : str or Path
            The raster file to be written.
        driver : str
            The driver to be used to write the raster file.
            More details can be found at: https://gdal.org/drivers/raster/index.html

        """
        if self.profile is None:
            msg = "profile is not set yet"
            raise AttributeError(msg)

        if self.arr is None:
            msg = "data array is not set yet"
            raise AttributeError(msg)

        self.profile.update({"driver": driver})
        with rasterio.open(out_file, "w", **self.profile) as ds:
            bands = range(1, self.profile["count"] + 1)
            ds.write(self.arr, bands)

    def add_band(self, arr: np.ndarray) -> None:
        """Add a band to the data array.

        Parameters
        ----------
        arr : 2D or 3D numpy.ndarray
            The array to be added. The shape of the array should be (height, width)
            or (band, height, width).

        """
        if self.arr is None:
            msg = "data array is not set yet"
            raise AttributeError(msg)

        if not isinstance(arr, np.ndarray):
            try:
                arr = np.array(arr)
            except Exception as e:
                msg = "arr can not be converted to numpy array"
                raise TypeError(msg) from e

        if len(arr.shape) == 2:
            arr = np.concatenate((self.arr, arr[None, :, :]), axis=0)
        if len(arr.shape) == 3:
            arr = np.concatenate((self.arr, arr), axis=0)

        self.update_arr(arr)

    def add_band_from_raster(self, raster_file: PathLike) -> None:
        """Add band to the data array from a raster file.

        Parameters
        ----------
        raster_file : str or Path
            The raster file to be added. raster format should be supported by gdal.
            More details can be found at: https://gdal.org/drivers/raster/index.html

        """
        arr, _profile = self._load_raster(raster_file)
        self.add_band(arr)

    # def add_band_from_binary(self, binary_file: PathLike) -> None:
    #     """Add band to the data array from a binary file.

    #     Parameters
    #     ----------
    #     binary_file : str or Path
    #         The binary file to be added. the binary file should be with a profile
    #         file with the same name.

    #     """
    #     arr, profile = self._load_binary(binary_file)
    #     self.add_band(arr)

    def update_arr(
        self,
        arr: np.ndarray,
        dtype: str = "auto",
        nodata: float | Literal["auto"] = "auto",
        error_if_nodata_invalid: bool = True,
    ) -> None:
        """Update the data array.

        Parameters
        ----------
        arr : numpy.ndarray
            The array to be updated. The profile will be updated accordingly.
        dtype : str or numpy.dtype
            The dtype of the array. If 'auto', the minimum dtype will be used.
            Default is 'auto'.
        nodata : float | Literal["auto"] = "auto"
            The nodata value of the array. If 'auto', the nodata value will be
            set to the nodata value of the profile if valid, otherwise None.
            Default is 'auto'.
        error_if_nodata_invalid : bool
            Whether to raise error if nodata is out of dtype range. Default is True.

        """
        self.arr = arr

        if self.profile is None:
            msg = "profile is not set yet"
            raise AttributeError(msg)

        # update profile info
        self.profile["count"] = arr.shape[0]
        self.profile["height"] = arr.shape[1]
        self.profile["width"] = arr.shape[2]

        if dtype == "auto":
            self.profile["dtype"] = dtypes.get_minimum_dtype(arr)
        else:
            if not dtypes.check_dtype(dtype):
                msg = f"dtype {dtype} is not supported"
                raise ValueError(msg)
            self.profile["dtype"] = dtype

        if nodata == "auto":
            nodata = self.profile["nodata"]
            error_if_nodata_invalid = False

        if nodata is None:
            self.profile["nodata"] = None
        else:
            dtype_ranges = dtypes.dtype_ranges[self.profile["dtype"]]
            if dtypes.in_dtype_range(nodata, self.profile["dtype"]):
                self.profile["nodata"] = nodata
            elif error_if_nodata_invalid:
                msg = f"nodata {nodata} is out of dtype range {dtype_ranges}"
                raise ValueError(
                    msg,
                )
            else:
                logger.warning(
                    "nodata is out of dtype range, nodata will be set to None",
                )
                self.profile["nodata"] = None


def format_bounds_and_crs(
    bounds: BoundingBox | tuple[float, float, float, float],
    crs: CrsLike | None = None,
) -> tuple[BoundingBox, CRS | None]:
    """Get the formatted bounds and crs from the input."""
    if not isinstance(bounds, BoundingBox):
        left, bottom, right, top = bounds
        bounds = BoundingBox(left, bottom, right, top, crs=crs)
    if crs is not None:
        crs = CRS.from_user_input(crs)
        if bounds.crs is not None and bounds.crs != crs:
            bounds = bounds.to_crs(crs)
    else:
        crs = bounds.crs
    return bounds, crs
