"""Raster I/O helpers for FanInSAR frame products."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from faninsar._core.geo import Profile
from faninsar.datasets.geogrid import GeoGrid
from faninsar.logging import setup_logger

from .exceptions import COGValidationError

if TYPE_CHECKING:
    from os import PathLike

logger = setup_logger(__name__)


def read_geogrid(raster_path: str | PathLike) -> GeoGrid:
    """Read the GeoGrid from a raster file."""
    with rasterio.open(raster_path) as ds:
        return GeoGrid.from_rio(ds)


# Backwards-compat alias — older code may import read_geobox.
read_geobox = read_geogrid


def read_profile(raster_path: str | PathLike) -> Profile:
    """Read the Profile from a raster file."""
    return Profile.from_raster_file(Path(raster_path))


def reproject_to_geogrid(
    src_path: str | PathLike,
    dst_geogrid: GeoGrid,
    resampling: Resampling = Resampling.bilinear,
    dst_dtype: str | np.dtype = "float32",
    dst_nodata: float | None = -9999.0,
) -> np.ndarray:
    """Reproject a source raster file to match a destination GeoGrid.

    Parameters
    ----------
    src_path : path-like
        Path to the source raster file.
    dst_geogrid : GeoGrid
        Target grid to reproject onto.
    resampling : Resampling
        Resampling algorithm. Default is bilinear.
    dst_dtype : str or numpy.dtype
        Output data type.
    dst_nodata : float or None
        NoData value for the output.

    Returns
    -------
    numpy.ndarray
        Reprojected array with shape ``(height, width)``.

    """
    dst_height, dst_width = dst_geogrid.shape
    dst_transform = dst_geogrid.transform
    dst_crs = dst_geogrid.crs

    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata

        dst_arr = np.full(
            (dst_height, dst_width),
            dst_nodata if dst_nodata is not None else 0,
            dtype=np.dtype(dst_dtype),
        )

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
        )

    return dst_arr


# Backwards-compat alias.
reproject_to_geobox = reproject_to_geogrid


def write_cog(
    arr: np.ndarray,
    dst_path: str | PathLike,
    geogrid: GeoGrid,
    *,
    nodata: float | None = -9999.0,
    dtype: str | np.dtype = "float32",
    compress: str = "DEFLATE",
    blocksize: int = 512,
    overwrite: bool = False,
) -> None:
    """Write a 2-D array as a COG or COG-like GeoTIFF.

    Parameters
    ----------
    arr : numpy.ndarray
        2-D array to write.
    dst_path : path-like
        Output file path.
    geogrid : GeoGrid
        GeoGrid describing the grid (CRS, transform, shape).
    nodata : float or None
        NoData value.
    dtype : str or numpy.dtype
        Output data type.
    compress : str
        Compression algorithm.
    blocksize : int
        Tile/block size for COG.
    overwrite : bool
        If True, overwrite existing files.

    Raises
    ------
    FileExistsError
        If the output exists and overwrite is False.

    """
    dst_path = Path(dst_path)
    if dst_path.exists() and not overwrite:
        msg = f"Output file already exists: {dst_path}"
        logger.info(msg)
        raise FileExistsError(msg)

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = arr.shape[-2], arr.shape[-1]
    dst_crs = geogrid.crs
    dst_transform = geogrid.transform

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "dtype": dtype,
        "width": width,
        "height": height,
        "count": 1,
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": blocksize,
        "blockysize": blocksize,
        "compress": compress,
    }

    # Try COG driver first, fall back to GTiff
    try:
        profile["driver"] = "COG"
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(arr.astype(np.dtype(dtype)), 1)
    except Exception:
        logger.warning("COG driver unavailable, falling back to tiled GeoTIFF")
        profile["driver"] = "GTiff"
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(arr.astype(np.dtype(dtype)), 1)
            dst.build_overviews([2, 4, 8], Resampling.nearest)


def validate_alignment(
    raster_path: str | PathLike,
    reference_geogrid: GeoGrid,
    tol: float = 1e-6,
) -> bool:
    """Validate that a raster is aligned with a reference GeoGrid.

    Parameters
    ----------
    raster_path : path-like
        Path to the raster to validate.
    reference_geogrid : GeoGrid
        The reference grid.
    tol : float
        Numerical tolerance for alignment check.

    Returns
    -------
    bool
        True if aligned, False otherwise.

    """
    file_geogrid = read_geogrid(raster_path)
    return file_geogrid.is_aligned(reference_geogrid, tol=tol)


def validate_cog(raster_path: str | PathLike) -> bool:
    """Check that a raster has COG-like properties.

    Parameters
    ----------
    raster_path : path-like
        Path to the raster to validate.

    Returns
    -------
    bool
        True if the file has tiled, compressed, and overview properties.

    Raises
    ------
    COGValidationError
        If the file does not have COG-like properties.

    """
    path = Path(raster_path)
    if not path.exists():
        msg = f"File does not exist: {path}"
        raise COGValidationError(msg)

    try:
        from rio_cogeo.cogeo import cog_validate

        is_valid, _, _ = cog_validate(str(path))
    except ImportError:
        is_valid = None

    if is_valid is not None:
        return is_valid

    # Fallback: manual validation
    with rasterio.open(path) as ds:
        if not ds.profile.get("tiled", False):
            msg = f"Raster is not tiled: {path}"
            raise COGValidationError(msg)
        if ds.profile.get("compress") is None:
            logger.warning("Raster has no compression: %s", path)

    return True
