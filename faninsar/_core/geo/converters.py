"""Converters between raster files and raw binary raster data."""

from __future__ import annotations

import pprint
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import rasterio
from rasterio import dtypes

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from os import PathLike

    from rasterio.profiles import Profile as RasterioProfile

logger = setup_logger(__name__)


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
