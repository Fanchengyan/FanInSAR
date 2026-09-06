"""Raster I/O helpers for FanInSAR Network products."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from faninsar.data.datasets.geogrid import GeoGrid
from faninsar.logging import setup_logger
from faninsar.processing.geometry.profiles import Profile

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


def reproject_phase_to_geogrid(
    src_path: str | PathLike,
    dst_geogrid: GeoGrid,
    *,
    resampling: Resampling = Resampling.lanczos,
    dst_nodata: float = -9999.0,
) -> np.ndarray:
    """Reproject a wrapped-phase raster using complex averaging.

    Wrapped phase is a cyclic quantity on [-pi, pi]. Naive bilinear
    resampling produces nonsensical values across 2*pi wrap boundaries: a
    pixel at +3.1 rad next to one at -3.1 rad (both near pi) averages to ~0
    instead of ~pi. This function resamples the *complex representation*
    ``exp(i*phi)`` (the kernel is applied to the real and imaginary parts
    independently), then converts back via ``np.angle``, which correctly
    handles wraps.

    The default kernel is Lanczos (windowed sinc), the production choice
    for resampling complex / wrapped-phase SAR data: bilinear's ``sinc²``
    response attenuates in-band signal and leaks residual aliasing, which
    smears fine phase texture. A sinc-family kernel preserves phase
    statistics. Callers may pass ``Resampling.bilinear`` for already
    heavily multilooked phase screens whose bandwidth is well below the
    output grid Nyquist, where bilinear is an acceptable approximation.

    Parameters
    ----------
    src_path : path-like
        Path to the source wrapped-phase raster (radians).
    dst_geogrid : GeoGrid
        Target grid to reproject onto.
    resampling : Resampling
        Resampling algorithm applied to the complex field. Default
        ``Resampling.lanczos``; ``Resampling.bilinear`` is acceptable only
        for already-multilooked phase screens.
    dst_nodata : float
        NoData value for the output.

    Returns
    -------
    numpy.ndarray
        Reprojected wrapped-phase array ``(height, width)`` in radians on
        ``[-pi, pi]``.

    """
    dst_height, dst_width = dst_geogrid.shape
    dst_transform = dst_geogrid.transform
    dst_crs = dst_geogrid.crs

    with rasterio.open(src_path) as src:
        src_phi = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata

    # Build the complex field, masking nodata / invalid phase values to zero
    # magnitude so they contribute nothing to the average.
    invalid = ~np.isfinite(src_phi)
    if src_nodata is not None:
        invalid |= src_phi == src_nodata
    complex_src = np.where(
        invalid, 0.0 + 0.0j, np.exp(1j * src_phi)
    ).astype(np.complex64)

    complex_dst = np.zeros((dst_height, dst_width), dtype=np.complex64)
    reproject(
        source=complex_src,
        destination=complex_dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=None,
        dst_nodata=None,
    )

    out = np.full(
        (dst_height, dst_width),
        dst_nodata,
        dtype=np.float32,
    )
    # Where the averaged complex vector has meaningful magnitude, take its
    # angle; otherwise leave as nodata (low-coherence / empty regions).
    magnitude = np.abs(complex_dst)
    valid = magnitude > 1e-6
    out[valid] = np.angle(complex_dst[valid]).astype(np.float32)
    return out


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
