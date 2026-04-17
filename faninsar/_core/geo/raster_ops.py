"""Raster reprojection and matching operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject
from tqdm import tqdm

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from .profiles import Profile

logger = setup_logger(__name__)


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
