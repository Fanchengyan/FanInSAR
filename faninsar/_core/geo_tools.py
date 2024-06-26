from __future__ import annotations

import pprint
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio import Affine, dtypes, transform
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from tqdm.auto import tqdm

from ..query.bbox import BoundingBox
from .logger import setup_logger

logger = setup_logger(
    log_name="FanInSAR.geo_tools", log_format="%(levelname)s - %(message)s"
)


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
        self.arr: np.ndarray = None
        self.profile: dict = None

    @property
    def _profile_str(self):
        return pprint.pformat(self.profile, sort_dicts=False)

    def __str__(self) -> str:
        return f"DataConverter: \n{self._profile_str}"

    def __repr__(self) -> str:
        return str(self)

    def _load_raster(self, raster_file: str | Path):
        """Load a raster file into the data array."""
        with rasterio.open(raster_file) as ds:
            arr = ds.read()
            profile = ds.profile.copy()
        return arr, profile

    def load_binary(
        self,
        binary_file: str | Path,
        order: Literal["BSQ", "BIP", "BIL"] = "BSQ",
        dtype="auto",
    ):
        """Load a binary file into the data array.

        Parameters
        ----------
        binary_file : str or Path
            The binary file to be loaded. the binary file should be with a profile
            file with the same name.
        order : str, one of ['BSQ', 'BIP', 'BIL']
            The order of the data array. 'BSQ' for band sequential, 'BIP' for
            band interleaved by pixel, 'BIL' for band interleaved by line.
            Default is 'BSQ'.
            More details can be found at: https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/bil-bip-and-bsq-raster-files.htm
        """
        binary_profile_file = str(binary_file) + ".profile"
        if not Path(binary_profile_file).exists():
            raise FileNotFoundError(f"{binary_profile_file} not found")

        with open(binary_profile_file, "r") as f:
            profile = eval(f.read())

        # todo: auto detect dtype by shape
        if dtype == "auto":
            dtype = np.float32

        arr = np.fromfile(binary_file, dtype=dtype)
        if order == "BSQ":
            arr = arr.reshape(profile["count"], profile["height"], profile["width"])
        elif order == "BIP":
            arr = arr.reshape(
                profile["height"], profile["width"], profile["count"]
            ).transpose(2, 0, 1)
        elif order == "BIL":
            arr = arr.reshape(
                profile["height"], profile["count"], profile["width"]
            ).transpose(1, 0, 2)
        else:
            raise ValueError(
                "order should be one of ['BSQ', 'BIP', 'BIL']," f" but got {order}"
            )

        if "dtype" not in profile:
            profile["dtype"] = dtypes.get_minimum_dtype(arr)

        self.arr = arr
        self.profile = profile

    def load_raster(self, raster_file: str | Path):
        """Load a raster file into the data array.

        Parameters
        ----------
        raster_file : str or Path
            The raster file to be loaded. raster format should be supported by gdal.
            See: https://gdal.org/drivers/raster/index.html
        """
        self.arr, self.profile = self._load_raster(raster_file)

    def to_binary(
        self, out_file: str | Path, order: Literal["BSQ", "BIP", "BIL"] = "BSQ"
    ):
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
            More details can be found at: https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/bil-bip-and-bsq-raster-files.htm
        """
        if order == "BSQ":
            arr = self.arr
        elif order == "BIL":
            arr = np.transpose(self.arr, (1, 2, 0))
        elif order == "BIP":
            arr = np.transpose(self.arr, (1, 0, 2))

        # write data into a binary file
        (arr.astype(np.float32).tofile(out_file))

        # write profile into a file with the same name
        out_profile_file = str(out_file) + ".profile"
        with open(out_profile_file, "w") as f:
            f.write(self._profile_str)

    def to_raster(self, out_file: str | Path, driver="GTiff"):
        """Write the data array into a raster file.

        Parameters
        ----------
        out_file : str or Path
            The raster file to be written.
        driver : str
            The driver to be used to write the raster file.
            More details can be found at: https://gdal.org/drivers/raster/index.html
        """
        self.profile.update({"driver": driver})
        with rasterio.open(out_file, "w", **self.profile) as ds:
            bands = range(1, self.profile["count"] + 1)
            ds.write(self.arr, bands)

    def add_band(self, arr: np.ndarray):
        """Add a band to the data array.

        Parameters
        ----------
        arr : 2D or 3D numpy.ndarray
            The array to be added. The shape of the array should be (height, width) or (band, height, width).
        """
        if not isinstance(arr, np.ndarray):
            try:
                arr = np.array(arr)
            except:
                raise TypeError("arr can not be converted to numpy array")

        if len(arr.shape) == 2:
            arr = np.concatenate((self.arr, arr[None, :, :]), axis=0)
        if len(arr.shape) == 3:
            arr = np.concatenate((self.arr, arr), axis=0)

        self.update_arr(arr)

    def add_band_from_raster(self, raster_file: str | Path):
        """Add band to the data array from a raster file.

        Parameters
        ----------
        raster_file : str or Path
            The raster file to be added. raster format should be supported by gdal.
            More details can be found at: https://gdal.org/drivers/raster/index.html
        """
        arr, profile = self._load_raster(raster_file)
        self.add_band(arr)

    def add_band_from_binary(self, binary_file: str | Path):
        """Add band to the data array from a binary file.

        Parameters
        ----------
        binary_file : str or Path
            The binary file to be added. the binary file should be with a profile
            file with the same name.
        """
        arr, profile = self._load_binary(binary_file)
        self.add_band(arr)

    def update_arr(
        self,
        arr: np.ndarray,
        dtype: str = "auto",
        nodata: Any | Literal["auto"] = "auto",
        error_if_nodata_invalid: bool = True,
    ):
        """update the data array.

        Parameters
        ----------
        arr : numpy.ndarray
            The array to be updated. The profile will be updated accordingly.
        dtype : str or numpy.dtype
            The dtype of the array. If 'auto', the minimum dtype will be used.
            Default is 'auto'.
        nodata : Any | Literal["auto"] = "auto"
            The nodata value of the array. If 'auto', the nodata value will be
            set to the nodata value of the profile if valid, otherwise None.
            Default is 'auto'.
        error_if_nodata_invalid : bool
            Whether to raise error if nodata is out of dtype range. Default is True.
        """
        self.arr = arr
        if not hasattr(self, "profile"):
            raise AttributeError("profile is not set yet")

        # update profile info
        self.profile["count"] = arr.shape[0]
        self.profile["height"] = arr.shape[1]
        self.profile["width"] = arr.shape[2]

        if dtype == "auto":
            self.profile["dtype"] = dtypes.get_minimum_dtype(arr)
        else:
            if not dtypes.check_dtype(dtype):
                raise ValueError(f"dtype {dtype} is not supported")
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
            else:
                if error_if_nodata_invalid:
                    raise ValueError(
                        f"nodata {nodata} is out of dtype range {dtype_ranges}"
                    )
                else:
                    logger.warning(
                        "nodata is out of dtype range, " "nodata will be set to None"
                    )
                    self.profile["nodata"] = None


class Profile:
    """a class to manage the profile of a raster file.
    The profile is the metadata of the raster file and
    can be recognized by rasterio package"""

    def __init__(self, profile: dict = None) -> None:
        self.profile = profile

    def __str__(self) -> str:
        return pprint.pformat(self.profile, sort_dicts=False)

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, key):
        return self.profile[key]

    def __setitem__(self, key, value):
        self.profile[key] = value

    def __contains__(self, key):
        return key in self.profile

    def __iter__(self):
        return iter(self.profile)

    def __len__(self):
        return len(self.profile)

    def __delitem__(self, key):
        del self.profile[key]

    def __eq__(self, other):
        return self.profile == other.profile

    def __ne__(self, other):
        return self.profile != other.profile

    @classmethod
    def from_raster_file(cls, raster_file: str | Path):
        """Create a Profile object from a raster file."""
        with rasterio.open(raster_file) as ds:
            profile = ds.profile.copy()
        return cls(profile)

    @classmethod
    def from_ascii_header_file(cls, ascii_file: str | Path):
        """Create a Profile object from an ascii header file.
        The ascii header file is the metadata of a binary.
        More information can be found at:
        https://desktop.arcgis.com/zh-cn/arcmap/latest/manage-data/raster-and-images/esri-ascii-raster-format.htm

        Example of an ascii header file
        -------------------------------
        ::

            ncols         43200
            nrows         18000
            xllcorner     -180.000000
            yllcorner     -60.000000
            cellsize      0.008333
            NODATA_value  -9999
        """
        df = pd.read_csv(ascii_file, sep="\s+", header=None, index_col=0)
        df.index = df.index.str.lower()

        width = int(df.loc["ncols", 1])
        height = int(df.loc["nrows", 1])
        cell_size = float(df.loc["cellsize", 1])
        try:
            left = float(df.loc["xllcorner", 1])
            bottom = float(df.loc["yllcorner", 1])
        except:
            left = float(df.loc["xllcenter", 1]) - cell_size / 2
            bottom = float(df.loc["yllcenter", 1]) - cell_size / 2

        # pixel left lower corner to pixel left upper corner (rasterio transform)
        top = bottom + (height + 1) * cell_size

        tf = transform.from_origin(left, top, cell_size, cell_size)

        nodata = None
        if "nodata_value" in df.index:
            nodata = float(df.loc["nodata_value", 1])

        profile = {
            "width": width,
            "height": height,
            "transform": tf,
            "count": 1,
            "nodata": nodata,
        }

        return cls(profile)

    @classmethod
    def from_profile_file(cls, profile_file: str | Path):
        """Create a Profile object from a profile file."""
        with open(profile_file, "r") as f:
            profile = eval(f.read())
        return cls(profile)

    @classmethod
    def from_bounds_res(
        cls,
        bounds: Tuple[float, float, float, float],
        res: float | Tuple[float, float],
    ):
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
        width = int((dst_e - dst_w) / res[0])
        height = int((dst_n - dst_s) / res[1])
        tf = Affine.translation(dst_w, dst_n) * Affine.scale(res[0], -res[1])

        profile = {"width": width, "height": height, "transform": tf}
        return cls(profile)

    def to_file(self, file: str | Path):
        """Write the profile into a file."""
        file = Path(file)
        if file.suffix != ".profile":
            file = file.parent / (file.name + ".profile")
        with open(file, "w") as f:
            f.write(str(self))

    def to_latlon(self) -> Tuple[np.ndarray, np.ndarray]:
        """get the latitude and longitude from profile data"""
        tf = self.profile["transform"]
        width = self.profile["width"]
        height = self.profile["height"]
        lon = tf.xoff + tf.a * np.arange(width) + tf.a * 0.5
        lat = tf.yoff + tf.e * np.arange(height) + tf.e * 0.5
        return lat, lon


def bound_from_latlon(
    lat: np.ndarray, lon: np.ndarray
) -> Tuple[float, float, float, float]:
    """get the bounds from latitude and longitude."""
    west, south, east, north = (
        np.nanmin(lon),
        np.nanmin(lat),
        np.nanmax(lon),
        np.nanmax(lat),
    )
    return west, south, east, north


def geoinfo_from_latlon(
    lat: np.ndarray, lon: np.ndarray
) -> tuple[BoundingBox, tuple, tuple]:
    """get the geoinformation from latitude and longitude.

    Parameters
    ----------
    lat, lon: numpy.ndarray or list
        latitudes and longitudes

    Returns
    --------
    bound: BoundingBox
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
    bound = BoundingBox(west, south, east, north)
    res = (xsize, ysize)
    shape = (height, width)
    return bound, res, shape


def transform_from_latlon(lat, lon) -> Affine:
    """get the rasterio.transform from latitude and longitude.
    the pixel location will shift from center to upper-left corner

    Parameters
    ----------
    lat, lon: numpy.ndarray or list
        latitudes and longitudes
    """
    west, north, xsize, ysize, _, _ = geoinfo_from_latlon(lat, lon)

    tf = transform.from_origin(
        west - 0.5 * xsize,  # center to left
        north + 0.5 * ysize,  # center to top
        xsize,
        ysize,
    )
    return tf


def latlon_from_profile(profile: Profile) -> np.ndarray:
    """get the latitude and longitude from rasterio profile data

    Parameters
    ----------
    profile: Profile
        the profile data of rasterio dataset. It can be get from
        rasterio.open().profile

    Returns
    --------
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
    vars: Optional[str | Tuple | List] = None,
    crs: Any = "EPSG:4326",
    x_dim: str = "lon",
    y_dim: str = "lat",
):
    """write geoinformation in to the given xr DataArray or DataSet.

    Parameters:
    -----------
    ds: xarray DataArray or DataSet object
        data to be written into geoinfo.If type of ds is DataSet,
        vars should be set
    vars: str, tuple or list
        variables that need to be added geoinformation
    crs: str, int, dict or rasterio.crs.CRS object
        the coordinate reference system. Could be any type that
        :meth:`rasterio.crs.CRS.from_user_input` accepts.
    x_dim/y_dim: str
        the coordinate name that presents the x/y dimension
    """
    if isinstance(ds, xr.DataArray):
        ds = ds.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
        ds = ds.rio.write_crs(crs)
    else:
        if isinstance(vars, str):
            ds[vars] = ds[vars].rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
            ds[vars] = ds[vars].rio.write_crs(crs)
        elif isinstance(vars, (tuple, list)):
            for var in vars:
                ds[var] = ds[var].rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
                ds[var] = ds[var].rio.write_crs(crs)
        elif vars is None:
            raise TypeError("Detected type of ds is a xr.Dataset." " vars must be set")
        else:
            raise TypeError("vars type must be one of [str,tuple,list]")
    return ds


def write_geoinfo_into_nc(
    nc_file: str | Path,
    vars: Optional[str | Tuple | List] = None,
    crs: Any = "EPSG:4326",
    x_dim: str = "lon",
    y_dim: str = "lat",
    encode_time: bool = False,
):
    """write geoinformation in to the given nc file and making it could be
    opened with geoinformation in QGIS directly.

    Parameters:
    -----------
    nc_file: str or pathlib.Path object
        the path of nc file
    vars: str, tuple or list
        variables that need to be added geoinformation
    x_dim/y_dim: str
        the coordinate name that presents the x/y dimension
    crs: str, int, dict or rasterio.crs.CRS object
        the coordinate reference system. Could be any type that
        :meth:`rasterio.crs.CRS.from_user_input` accepts.
    encode_time: bool
        whether to encode the time since "2000-01-01 00:00:00" if
        "time" coordinate is exists. Default is False.
    """
    ds = xr.load_dataset(nc_file)

    ds = write_geoinfo_into_ds(ds, vars, crs, x_dim, y_dim)

    encode = {}
    if encode_time:
        if "time" in ds:
            encode.update({"time": {"units": "days since 2000-01-01 00:00:00"}})
        else:
            logger.warning(
                f'there is no "time" dimension in {nc_file}, '
                "encoding process will be ignored"
            )
    ds.to_netcdf(nc_file, encoding=encode)


def match_to_raster(
    src_arr,
    src_profile,
    dst_profile,
    resampleAlg=Resampling.nearest,
):
    """match the source raster to the destination raster.
    
    Parameters
    ----------
    src_arr: numpy.ndarray
        the source array to be matched.
    src_profile: dict
        the profile of the source raster.
    dst_profile: dict
        the profile of the destination raster.
    resampleAlg: Resampling
        the resampling algorithm. Default is Resampling.nearest.    
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
        raise ValueError("dimension of src_arr must be 2 or 3")
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            count=src_n_band,
            height=src_height,
            width=src_width,
            dtype="float32",
            crs=src_crs,
            transform=src_tf,
        ) as src:
            src.write(src_arr, indexes)

            with MemoryFile() as memfile1:
                with memfile1.open(
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
                            resampling=resampleAlg,
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
        self.arr: np.ndarray = None
        self.profile: dict = None

    @property
    def _profile_str(self):
        return pprint.pformat(self.profile, sort_dicts=False)

    def __str__(self) -> str:
        return f"DataConverter: \n{self._profile_str}"

    def __repr__(self) -> str:
        return str(self)

    def _load_raster(self, raster_file: str | Path):
        """Load a raster file into the data array."""
        with rasterio.open(raster_file) as ds:
            arr = ds.read()
            profile = ds.profile.copy()
        return arr, profile

    def load_binary(
        self,
        binary_file: str | Path,
        order: Literal["BSQ", "BIP", "BIL"] = "BSQ",
        dtype="auto",
    ):
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
        """
        binary_profile_file = str(binary_file) + ".profile"
        if not Path(binary_profile_file).exists():
            raise FileNotFoundError(f"{binary_profile_file} not found")

        with open(binary_profile_file, "r") as f:
            profile = eval(f.read())

        # todo: auto detect dtype by shape
        if dtype == "auto":
            dtype = np.float32

        arr = np.fromfile(binary_file, dtype=dtype)
        if order == "BSQ":
            arr = arr.reshape(profile["count"], profile["height"], profile["width"])
        elif order == "BIP":
            arr = arr.reshape(
                profile["height"], profile["width"], profile["count"]
            ).transpose(2, 0, 1)
        elif order == "BIL":
            arr = arr.reshape(
                profile["height"], profile["count"], profile["width"]
            ).transpose(1, 0, 2)
        else:
            raise ValueError(
                "order should be one of ['BSQ', 'BIP', 'BIL']," f" but got {order}"
            )

        if "dtype" not in profile:
            profile["dtype"] = dtypes.get_minimum_dtype(arr)

        self.arr = arr
        self.profile = profile

    def load_raster(self, raster_file: str | Path):
        """Load a raster file into the data array.

        Parameters
        ----------
        raster_file : str or Path
            The raster file to be loaded. raster format should be supported by gdal.
            More details can be found at: https://gdal.org/drivers/raster/index.html
        """
        self.arr, self.profile = self._load_raster(raster_file)

    def to_binary(
        self, out_file: str | Path, order: Literal["BSQ", "BIP", "BIL"] = "BSQ"
    ):
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
        if order == "BSQ":
            arr = self.arr
        elif order == "BIL":
            arr = np.transpose(self.arr, (1, 2, 0))
        elif order == "BIP":
            arr = np.transpose(self.arr, (1, 0, 2))

        # write data into a binary file
        (arr.astype(np.float32).tofile(out_file))

        # write profile into a file with the same name
        out_profile_file = str(out_file) + ".profile"
        with open(out_profile_file, "w") as f:
            f.write(self._profile_str)

    def to_raster(self, out_file: str | Path, driver="GTiff"):
        """Write the data array into a raster file.

        Parameters
        ----------
        out_file : str or Path
            The raster file to be written.
        driver : str
            The driver to be used to write the raster file.
            More details can be found at: https://gdal.org/drivers/raster/index.html
        """
        self.profile.update({"driver": driver})
        with rasterio.open(out_file, "w", **self.profile) as ds:
            bands = range(1, self.profile["count"] + 1)
            ds.write(self.arr, bands)

    def add_band(self, arr: np.ndarray):
        """Add a band to the data array.

        Parameters
        ----------
        arr : 2D or 3D numpy.ndarray
            The array to be added. The shape of the array should be (height, width)
            or (band, height, width).
        """
        if not isinstance(arr, np.ndarray):
            try:
                arr = np.array(arr)
            except:
                raise TypeError("arr can not be converted to numpy array")

        if len(arr.shape) == 2:
            arr = np.concatenate((self.arr, arr[None, :, :]), axis=0)
        if len(arr.shape) == 3:
            arr = np.concatenate((self.arr, arr), axis=0)

        self.update_arr(arr)

    def add_band_from_raster(self, raster_file: str | Path):
        """Add band to the data array from a raster file.

        Parameters
        ----------
        raster_file : str or Path
            The raster file to be added. raster format should be supported by gdal.
            More details can be found at: https://gdal.org/drivers/raster/index.html
        """
        arr, profile = self._load_raster(raster_file)
        self.add_band(arr)

    def add_band_from_binary(self, binary_file: str | Path):
        """Add band to the data array from a binary file.

        Parameters
        ----------
        binary_file : str or Path
            The binary file to be added. the binary file should be with a profile
            file with the same name.
        """
        arr, profile = self._load_binary(binary_file)
        self.add_band(arr)

    def update_arr(
        self,
        arr: np.ndarray,
        dtype: str = "auto",
        f: Any | Literal["auto"] = "auto",
        error_if_nodata_invalid: bool = True,
    ):
        """update the data array.

        Parameters
        ----------
        arr : numpy.ndarray
            The array to be updated. The profile will be updated accordingly.
        dtype : str or numpy.dtype
            The dtype of the array. If 'auto', the minimum dtype will be used.
            Default is 'auto'.
        nodata : Any | Literal["auto"] = "auto"
            The nodata value of the array. If 'auto', the nodata value will be
            set to the nodata value of the profile if valid, otherwise None.
            Default is 'auto'.
        error_if_nodata_invalid : bool
            Whether to raise error if nodata is out of dtype range. Default is True.
        """
        self.arr = arr
        if not hasattr(self, "profile"):
            raise AttributeError("profile is not set yet")

        # update profile info
        self.profile["count"] = arr.shape[0]
        self.profile["height"] = arr.shape[1]
        self.profile["width"] = arr.shape[2]

        if dtype == "auto":
            self.profile["dtype"] = dtypes.get_minimum_dtype(arr)
        else:
            if not dtypes.check_dtype(dtype):
                raise ValueError(f"dtype {dtype} is not supported")
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
            else:
                if error_if_nodata_invalid:
                    raise ValueError(
                        f"nodata {nodata} is out of dtype range {dtype_ranges}"
                    )
                else:
                    logger.warning(
                        "nodata is out of dtype range, " "nodata will be set to None"
                    )
                    self.profile["nodata"] = None


class Profile:
    """a class to manage the profile of a raster file. The profile is the metadata
    of the raster file and can be recognized by rasterio package"""

    def __init__(self, profile: dict = None) -> None:
        self.profile = profile

    def __str__(self) -> str:
        return pprint.pformat(self.profile, sort_dicts=False)

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, key):
        return self.profile[key]

    def __setitem__(self, key, value):
        self.profile[key] = value

    def __contains__(self, key):
        return key in self.profile

    def __iter__(self):
        return iter(self.profile)

    def __len__(self):
        return len(self.profile)

    def __delitem__(self, key):
        del self.profile[key]

    def __eq__(self, other):
        return self.profile == other.profile

    def __ne__(self, other):
        return self.profile != other.profile

    def keys(self):
        return self.profile.keys()

    def values(self):
        return self.profile.values()

    def items(self):
        return self.profile.items()

    def get(self, key, default=None):
        return self.profile.get(key, default)

    def update(self, other: dict):
        self.profile.update(other)

    @classmethod
    def from_raster_file(cls, raster_file: str | Path):
        """Create a Profile object from a raster file."""
        with rasterio.open(raster_file) as ds:
            profile = ds.profile.copy()
        return cls(profile)

    @classmethod
    def from_ascii_header_file(cls, ascii_file: str | Path):
        """Create a Profile object from an ascii header file. The ascii header
        file is the metadata of a binary. More information can be found at:
        https://desktop.arcgis.com/zh-cn/arcmap/latest/manage-data/raster-and-images/esri-ascii-raster-format.htm

        Example of an ascii header file
        -------------------------------
        ::

            ncols         43200
            nrows         18000
            xllcorner     -180.000000
            yllcorner     -60.000000
            cellsize      0.008333
            NODATA_value  -9999
        """
        df = pd.read_csv(ascii_file, sep="\s+", header=None, index_col=0)
        df.index = df.index.str.lower()

        width = int(df.loc["ncols", 1])
        height = int(df.loc["nrows", 1])
        cell_size = float(df.loc["cellsize", 1])
        try:
            left = float(df.loc["xllcorner", 1])
            bottom = float(df.loc["yllcorner", 1])
        except:
            left = float(df.loc["xllcenter", 1]) - cell_size / 2
            bottom = float(df.loc["yllcenter", 1]) - cell_size / 2

        # pixel left lower corner to pixel left upper corner (rasterio transform)
        top = bottom + (height + 1) * cell_size

        tf = transform.from_origin(left, top, cell_size, cell_size)

        nodata = None
        if "nodata_value" in df.index:
            nodata = float(df.loc["nodata_value", 1])

        profile = {
            "width": width,
            "height": height,
            "transform": tf,
            "count": 1,
            "nodata": nodata,
        }

        return cls(profile)

    @classmethod
    def from_profile_file(cls, profile_file: str | Path):
        """Create a Profile object from a profile file."""
        with open(profile_file, "r") as f:
            profile = eval(f.read())
        return cls(profile)

    @classmethod
    def from_bounds_res(
        cls,
        bounds: Tuple[float, float, float, float],
        res: float | Tuple[float, float],
    ):
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
        return cls(profile)

    def to_file(self, file: str | Path):
        """Write the profile into a file."""
        file = Path(file)
        if file.suffix != ".profile":
            file = file.parent / (file.name + ".profile")
        with open(file, "w") as f:
            f.write(str(self))

    def to_latlon(self) -> Tuple[np.ndarray, np.ndarray]:
        """get the latitude and longitude from profile data"""
        tf = self.profile["transform"]
        width = self.profile["width"]
        height = self.profile["height"]
        lon = tf.xoff + tf.a * np.arange(width) + tf.a * 0.5
        lat = tf.yoff + tf.e * np.arange(height) + tf.e * 0.5
        return lat, lon
