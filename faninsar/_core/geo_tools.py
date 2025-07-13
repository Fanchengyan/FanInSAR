"""A module to provide some useful tools for geospatial data processing."""

from __future__ import annotations

import pprint
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from lxml import etree
from matplotlib import ticker
from pykml.factory import KML_ElementMaker as KML
from pyproj import CRS
from rasterio import dtypes, transform
from rasterio.io import MemoryFile
from rasterio.profiles import Profile as RasterioProfile
from rasterio.warp import Resampling, reproject
from tqdm import tqdm

from faninsar.logging import setup_logger
from faninsar.query.bbox import BoundingBox

from .file_tools import load_metas

if TYPE_CHECKING:
    from os import PathLike

    from matplotlib.cm import ScalarMappable

    from faninsar.typing import CrsLike

logger = setup_logger(__name__)


def _ensure_bounds_in_wgs84(
    bounds: tuple[float, float, float, float],
) -> None:
    """Ensure the bounds are in WGS84 coordinate system."""
    west, south, east, north = bounds
    if west < -180 or east > 180 or south < -90 or north > 90:
        msg = (
            "bounds should be in WGS84 coordinate system, "
            f"but got [{west}, {south}, {east}, {north}]"
        )
        raise ValueError(msg)


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
    if isinstance(bounds, tuple):
        _ensure_bounds_in_wgs84(bounds)
        bounds = BoundingBox(*bounds, crs="EPSG:4326")
    if bounds.crs != CRS.from_user_input("EPSG:4326"):
        msg = "bounds should be in WGS84 coordinate system"
        raise ValueError(msg)

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
    with Path(out_file).open("w", encoding="utf-8") as f:
        f.write(etree.tostring(kml, pretty_print=True).decode("utf8"))
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


def bound_from_latlon(
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[float, float, float, float]:
    """Get the bounds from latitude and longitude."""
    west, south, east, north = (
        np.nanmin(lon),
        np.nanmin(lat),
        np.nanmax(lon),
        np.nanmax(lat),
    )
    return west, south, east, north


def geoinfo_from_latlon(
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[BoundingBox, tuple, tuple]:
    """Get the geoinformation from latitude and longitude.

    Parameters
    ----------
    lat, lon: numpy.ndarray or list
        latitudes and longitudes

    Returns
    -------
    bounds: BoundingBox
        the bounding box of the raster.

        .. note:: the crs is not set yet.
    res: tuple[xsize, ysize]
        the resolution of the raster
    shape: tuple[height, width]
        the shape of the raster

    """
    west, south, east, north = bound_from_latlon(lat, lon)
    width, height = len(lon), len(lat)

    xsize = (east - west) / (width - 1)
    ysize = (north - south) / (height - 1)
    bounds = BoundingBox(west, south, east, north)
    res = (xsize, ysize)
    shape = (height, width)
    return bounds, res, shape


def transform_from_latlon(
    lat: np.ndarray,
    lon: np.ndarray,
) -> Affine:
    """Get the :class:`rasterio.Affine` from latitude and longitude.

    .. note::
        The pixel location will shift from center to upper-left corner.

    Parameters
    ----------
    lat, lon: numpy.ndarray or list
        latitudes and longitudes

    """
    (west, north), (xsize, ysize), _ = geoinfo_from_latlon(lat, lon)

    return transform.from_origin(
        west - 0.5 * xsize,  # center to left
        north + 0.5 * ysize,  # center to top
        xsize,
        ysize,
    )


def latlon_from_profile(profile: RasterioProfile) -> tuple[np.ndarray, np.ndarray]:
    """Get the latitude and longitude from rasterio profile data.

    Parameters
    ----------
    profile: Profile
        the profile data of rasterio dataset. It can be get from
        rasterio.open().profile

    Returns
    -------
    lat, lon: numpy.ndarray

    """
    tf = profile["transform"]
    width = profile["width"]
    height = profile["height"]
    lon = tf.xoff + tf.a * np.arange(width) + tf.a * 0.5
    lat = tf.yoff + tf.e * np.arange(height) + tf.e * 0.5
    return lat, lon


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
        for _var in var:
            ds[_var] = ds[_var].rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
            ds[_var] = ds[_var].rio.write_crs(crs)
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
                "encoding process will be ignored",
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
    with MemoryFile() as memfile, memfile.open(
        driver="GTiff",
        count=src_n_band,
        height=src_height,
        width=src_width,
        dtype="float32",
        crs=src_crs,
        transform=src_tf,
    ) as src:
        src.write(src_arr, indexes)

        with MemoryFile() as memfile1, memfile1.open(
            driver="GTiff",
            count=src_n_band,
            height=dst_height,
            width=dst_width,
            dtype="float32",
            crs=dst_crs,
            transform=dst_tf,
        ) as dst:
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
        with Path(out_profile_file).open("w", encoding="utf-8") as f:
            f.write(self._profile_str)

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
        arr, profile = self._load_raster(raster_file)
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


@dataclass
class Profile:
    """A class to manage the profile of a raster image.

    .. note::
        the :attr:`height`, :attr:`width`, :attr:`transform` and :attr:`crs`
        are the basic parameters for a warp process.
    """

    #: The height (number of rows) of the raster image.
    height: float

    #: The width (number of columns) of the raster image.
    width: float

    #: The transform of the raster image. The transform is a instance of
    #: :class:`rasterio.Affine` representing an affine transformation matrix.
    #:
    #: .. note::
    #:      The Raster Space of transform is in "PixelIsPoint" Raster Space, which
    #:      means the pixel location is at the upper-left corner of pixels.
    #:      More details can be found at: `Raster Space <https://web.archive.org/web/20160326194152/http://remotesensing.org/geotiff/spec/geotiff2.5.html#2.5.2>`_
    transform: Affine

    #: The coordinate reference system of the raster image. If not set, it will be None.
    crs: CrsLike | None = None

    #: The nodata value of the raster image. If not set, it will be None.
    nodata: float | None = None

    #: The count of bands of the raster image. Default is 1.
    count: int = 1

    #: The driver of the raster image. Default is "GTiff".
    driver: str = "GTiff"

    #: The dtype of the raster image. Default is None.
    dtype: str | np.dtype | None = None

    #: Other keyword arguments for :class:`rasterio.profiles.Profile` class.
    kwargs: dict = field(repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Post initialization."""
        self._bounds = self._parse_bounds()
        self._res = (self.transform.a, self.transform.e)
        if self.crs is not None:
            self.crs = CRS.from_user_input(self.crs)
        for key in self.kwargs:
            setattr(self, key, self.kwargs[key])
        self.crs = cast("CRS", self.crs)

    def __getitem__(self, key: str) -> Any:
        """Get the value of the key."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set the value of the key."""
        setattr(self, key, value)

    def _parse_bounds(self) -> tuple[float, float, float, float]:
        """Parse the bounds from profile data."""
        tf = self.transform
        width = self.width
        height = self.height
        left = tf.c
        top = tf.f
        right = left + width * tf.a
        bottom = top + height * tf.e
        return left, bottom, right, top

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

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """The bounds in [west, south, east, north] order."""
        return self._bounds

    @property
    def res(self) -> tuple[float, float]:
        """The resolution in x and y direction."""
        return self._res

    @classmethod
    def from_raster_file(cls, raster_file: PathLike) -> Profile:
        """Create a Profile object from a raster file."""
        with rasterio.open(raster_file) as ds:
            profile = dict(ds.profile.copy())
        # split the profile into default keys and other keys
        profile, kwargs = cls._split_profile(profile)

        return cls(**profile, kwargs=kwargs)

    @classmethod
    def from_ascii_header_file(cls, ascii_file: PathLike) -> Profile:
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

        return cls(width=width, height=height, transform=tf, nodata=nodata)

    @classmethod
    def from_profile_file(cls, profile_file: PathLike) -> Profile:
        """Create a Profile object from a profile file."""
        with Path(profile_file).open(encoding="utf-8") as f:
            profile = eval(f.read())
        profile, kwargs = cls._split_profile(profile)
        return cls(**profile, kwargs=kwargs)

    @classmethod
    def from_bounds_res(
        cls,
        bounds: tuple[float, float, float, float] | BoundingBox,
        res: float | tuple[float, float],
    ) -> Profile:
        """Create a Profile object from bounds and resolution.

        Parameters
        ----------
        bounds : tuple of float (left/W, bottom/S, right/E, top/N)
            The bounds of the raster file.
        res : float or tuple of float (x_res, y_res)
            The resolution of the raster file. If a float is provided,
            the x_res and y_res will be the same.

        Returns
        -------
        Profile : Profile
            A Profile object only with width, height and transform.

        """
        if isinstance(res, (int, float, np.integer, np.floating)):
            res = (float(res), float(res))
        dst_w, dst_s, dst_e, dst_n = bounds
        width = round((dst_e - dst_w) / res[0])
        height = round((dst_n - dst_s) / res[1])
        tf = Affine.translation(dst_w, dst_n) * Affine.scale(res[0], -res[1])

        profile = {"width": width, "height": height, "transform": tf}
        return cls(**profile)

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

    def to_latlon(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the latitude and longitude from profile data.

        .. note::
            The pixel location for the latitude and longitude is the
            "PixelIsArea" Raster Space, which means the pixel location
            is the center of the pixel. See `Raster Space <https://web.archive.org/web/20160326194152/http://remotesensing.org/geotiff/spec/geotiff2.5.html#2.5.2>`_
            for more details.
        """
        tf = self.transform
        width = self.width
        height = self.height
        lon = tf.xoff + tf.a * np.arange(width) + tf.a * 0.5
        lat = tf.yoff + tf.e * np.arange(height) + tf.e * 0.5
        return lat, lon
