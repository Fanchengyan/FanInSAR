"""Raster write primitives: COG writing, COG validation, grid reporting.

This module owns the low-level raster operations that several domains need:
writing one array as a Cloud Optimized GeoTIFF, validating that a file is a
COG, reading the grid of an existing raster, and reprojecting a source raster
onto a destination grid. It deliberately depends on nothing in
``faninsar.network`` or ``faninsar.stack``; ``faninsar.data`` is imported only
lazily where a concrete :class:`~faninsar.data.datasets.geogrid.GeoGrid` has to
be constructed.

See Also
--------
faninsar.io.tiff : Bounded remote TIFF inspection and window decoding.
faninsar.io.format : Serialization format tags.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from faninsar.logging import reject, setup_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from os import PathLike

logger = setup_logger(__name__)

__all__ = [
    "RasterGridSpec",
    "read_geobox",
    "read_geogrid",
    "read_profile",
    "reproject_phase_to_geogrid",
    "reproject_to_geogrid",
    "validate_alignment",
    "validate_cog",
    "write_cog",
]


@runtime_checkable
class RasterGridSpec(Protocol):
    """Minimum grid surface the raster helpers need.

    ``shape`` is a two-element sequence rather than ``tuple[int, int]`` because
    :class:`~faninsar.data.datasets.geogrid.GeoGrid` inherits odc-geo's
    ``Shape2d``, which is a ``Sequence[int]``. Two-element length is required at
    runtime.
    """

    @property
    def crs(self) -> object:
        """Return the target coordinate reference system."""
        ...

    @property
    def transform(self) -> object:
        """Return the affine transform of the top-left corner."""
        ...

    @property
    def shape(self) -> Sequence[int]:
        """Return ``(height, width)``."""
        ...


def _grid_shape(geogrid: RasterGridSpec) -> tuple[int, int]:
    """Return a validated two-element ``(height, width)`` from a grid."""
    shape = tuple(int(value) for value in geogrid.shape)
    if len(shape) != 2 or any(size <= 0 for size in shape):
        reject(f"grid shape must be two positive integers; got {shape!r}", ValueError)
    return shape[0], shape[1]


def read_geogrid(raster_path: str | PathLike) -> Any:
    """Read the :class:`~faninsar.data.datasets.geogrid.GeoGrid` of a raster."""
    from faninsar.data.datasets.geogrid import GeoGrid

    with rasterio.open(raster_path) as ds:
        return GeoGrid.from_rio(ds)


# Backwards-compat alias — older code may call this read_geobox.
read_geobox = read_geogrid


def read_profile(raster_path: str | PathLike) -> Any:
    """Read the geometry profile of an existing raster file."""
    from faninsar.processing.geometry.profiles import Profile

    return Profile.from_raster_file(Path(raster_path))


def reproject_to_geogrid(
    src_path: str | PathLike,
    dst_geogrid: RasterGridSpec,
    resampling: Resampling = Resampling.bilinear,
    dst_dtype: str | np.dtype = "float32",
    dst_nodata: float | None = -9999.0,
) -> np.ndarray:
    """Reproject a source raster onto a destination grid.

    Parameters
    ----------
    src_path : path-like
        Path to the source raster.
    dst_geogrid : RasterGridSpec
        Target grid.
    resampling : rasterio.enums.Resampling
        Resampling algorithm.
    dst_dtype : str or numpy.dtype
        Output data type.
    dst_nodata : float or None
        NoData value for the output.

    Returns
    -------
    numpy.ndarray
        Reprojected array with shape ``(height, width)``.

    """
    dst_height, dst_width = _grid_shape(dst_geogrid)
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


def reproject_phase_to_geogrid(
    src_path: str | PathLike,
    dst_geogrid: RasterGridSpec,
    *,
    resampling: Resampling = Resampling.lanczos,
    dst_nodata: float = -9999.0,
) -> np.ndarray:
    """Reproject wrapped phase by averaging its complex representation.

    Wrapped phase is cyclic: bilinear averaging across a ``2*pi`` wrap boundary
    yields roughly zero instead of roughly ``pi``. This resamples
    ``exp(i*phi)`` and then takes ``np.angle``, which is wrap-correct.

    Parameters
    ----------
    src_path : path-like
        Source wrapped-phase raster in radians.
    dst_geogrid : RasterGridSpec
        Target grid.
    resampling : rasterio.enums.Resampling
        Kernel applied to the complex field. ``Resampling.bilinear`` is only
        acceptable for heavily multilooked phase screens.
    dst_nodata : float
        Output NoData value.

    Returns
    -------
    numpy.ndarray
        Reprojected wrapped phase in radians on ``[-pi, pi]``.

    """
    dst_height, dst_width = _grid_shape(dst_geogrid)
    dst_transform = dst_geogrid.transform
    dst_crs = dst_geogrid.crs

    with rasterio.open(src_path) as src:
        src_phi = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata

    invalid = ~np.isfinite(src_phi)
    if src_nodata is not None:
        invalid |= src_phi == src_nodata
    complex_src = np.where(invalid, 0.0 + 0.0j, np.exp(1j * src_phi)).astype(
        np.complex64
    )

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

    out = np.full((dst_height, dst_width), dst_nodata, dtype=np.float32)
    magnitude = np.abs(complex_dst)
    valid = magnitude > 1e-6
    out[valid] = np.angle(complex_dst[valid]).astype(np.float32)
    return out


def _mask_plane(
    array: np.ndarray,
    mask: np.ndarray | bool | None,
    shape: tuple[int, int],
) -> np.ndarray | None:
    """Return one dataset-wide boolean validity plane, or ``None``.

    ``True`` derives validity from finiteness: for a single band the pixel is
    valid when the sample is finite; for several bands it is valid only when
    every band is finite, because a GeoTIFF mask applies to the whole dataset.
    An explicit plane must already be ``(height, width)``.

    """
    if mask is None or mask is False:
        return None
    if mask is True:
        finite = np.isfinite(array)
        return finite if array.ndim == 2 else np.all(finite, axis=0)
    plane = np.asarray(mask)
    if plane.dtype != np.bool_:
        reject(
            f"explicit mask must be boolean; got dtype {plane.dtype}",
            TypeError,
        )
    if plane.shape != shape:
        reject(
            f"explicit mask shape {plane.shape} does not match the raster "
            f"shape {shape}; a GeoTIFF mask is dataset-wide",
            ValueError,
        )
    return plane


def _resolved_profile(path: Path) -> dict[str, Any]:
    """Return the effective writer settings of one finalized COG.

    The requested profile is not the written profile: GDAL may clamp block
    sizes, and overview resampling lives in the ``IMAGE_STRUCTURE`` domain
    rather than in ``profile``. Reopening the file is the only way to report
    what a reader will actually see.

    """
    with rasterio.open(path) as dataset:
        profile = dict(dataset.profile)
        structure = dataset.tags(ns="IMAGE_STRUCTURE")
        block_shapes = dataset.block_shapes
        overviews = dataset.overviews(1) if dataset.count else []
    resolved: dict[str, Any] = {
        "driver": profile.get("driver"),
        "dtype": profile.get("dtype"),
        "count": profile.get("count"),
        "compress": profile.get("compress"),
        "nodata": (
            None
            if profile.get("nodata") is None
            else str(profile["nodata"])
            if np.isnan(profile["nodata"])
            else profile["nodata"]
        ),
        "block_shapes": [list(shape) for shape in block_shapes],
        "overviews": list(overviews),
        "overview_resampling": structure.get("OVERVIEW_RESAMPLING"),
    }
    return resolved


def write_cog(
    array: np.ndarray,
    path: str | PathLike,
    grid: RasterGridSpec | None = None,
    *,
    nodata: float | None = -9999.0,
    dtype: str | np.dtype | None = None,
    mask: np.ndarray | bool | None = None,
    band_descriptions: Sequence[str] | None = None,
    tags: Mapping[str, object] | None = None,
    band_tags: Sequence[Mapping[str, object]] | None = None,
    compress: str = "DEFLATE",
    blocksize: int = 512,
    overview_resampling: str = "average",
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Write an array as a Cloud Optimized GeoTIFF.

    Validity is embedded as one internal GeoTIFF mask, so no ``.msk`` sidecar
    is ever produced and the payload stays a single file. The write is staged
    in the destination directory and published by rename, so an interrupted
    write never leaves a partial COG at the final path.

    Parameters
    ----------
    array : numpy.ndarray
        ``(y, x)`` or ``(band, y, x)`` samples.
    path : path-like
        Destination file.
    grid : RasterGridSpec, optional
        Destination grid. ``None`` writes an unreferenced COG, which is valid
        COG and is what a radar-domain product needs.
    nodata : float or None
        NoData value, written alongside the mask because some clients read only
        one of the two.
    dtype : str or numpy.dtype, optional
        Output dtype; defaults to the input dtype.
    mask : numpy.ndarray or bool, optional
        ``True`` derives validity from finiteness, an explicit plane must be
        boolean ``(height, width)``, ``None`` writes no mask.
    band_descriptions : sequence of str, optional
        One description per band, so a single-band COG shows a name rather than
        ``Band 1``.
    tags : mapping, optional
        Dataset-level tags.
    band_tags : sequence of mapping, optional
        Per-band tags, one mapping per band.
    compress : str
        Compression algorithm.
    blocksize : int
        Tile size; GDAL may clamp it to the driver's grid.
    overview_resampling : str
        Resampling used to build overviews.
    overwrite : bool
        Replace an existing destination.

    Returns
    -------
    tuple of (pathlib.Path, dict)
        The written path and its resolved writer profile, which may differ from
        the requested one.

    Raises
    ------
    FileExistsError
        If the destination exists, including as a dangling symlink, and
        ``overwrite`` is false.
    ValueError
        If the array is not 2-D or 3-D, or the mask does not match the raster.

    Examples
    --------
    >>> write_cog(array, "out/phase.cog.tif", grid, mask=True)  # doctest: +SKIP

    """
    values = np.asarray(array)
    if values.ndim not in {2, 3}:
        reject(
            f"write_cog accepts a 2-D or 3-D array; got shape {values.shape}",
            ValueError,
        )

    destination = Path(path)
    if destination.is_symlink() and not destination.exists():
        reject(f"destination is a dangling symlink: {destination}", FileExistsError)
    if destination.exists() and not overwrite:
        logger.info("output file already exists: %s", destination)
        reject(f"output file already exists: {destination}", FileExistsError)

    height, width = values.shape[-2], values.shape[-1]
    band_count = 1 if values.ndim == 2 else values.shape[0]
    if band_descriptions is not None and len(band_descriptions) != band_count:
        reject(
            f"band_descriptions has {len(band_descriptions)} entries for "
            f"{band_count} bands",
            ValueError,
        )
    if band_tags is not None and len(band_tags) != band_count:
        reject(
            f"band_tags has {len(band_tags)} entries for {band_count} bands",
            ValueError,
        )

    plane = _mask_plane(values, mask, (height, width))
    out_dtype = np.dtype(dtype) if dtype is not None else values.dtype

    profile: dict[str, Any] = {
        "driver": "COG",
        "dtype": out_dtype.name,
        "width": width,
        "height": height,
        "count": band_count,
        "nodata": nodata,
        "compress": compress,
        "blocksize": blocksize,
        "overview_resampling": overview_resampling,
    }
    if grid is not None:
        profile["crs"] = grid.crs
        profile["transform"] = grid.transform

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        f".{destination.name}.{__import__('uuid').uuid4().hex}.tmp"
    )
    try:
        with rasterio.open(staging, "w", **profile) as dataset:
            written = np.asarray(values, dtype=out_dtype)
            if values.ndim == 2:
                dataset.write(written, 1)
            else:
                dataset.write(written)
            if plane is not None:
                dataset.write_mask(plane.astype(np.uint8) * 255)
            if tags:
                dataset.update_tags(
                    **{str(key): str(value) for key, value in tags.items()}
                )
            if band_descriptions:
                for index, label in enumerate(band_descriptions, start=1):
                    dataset.set_band_description(index, str(label))
            if band_tags:
                for index, mapping in enumerate(band_tags, start=1):
                    dataset.update_tags(
                        index,
                        **{str(key): str(value) for key, value in mapping.items()},
                    )
        resolved = _resolved_profile(staging)
        staging.replace(destination)
    except Exception:
        staging.unlink(missing_ok=True)
        raise
    return destination, resolved


def validate_cog(raster_path: str | PathLike) -> tuple[bool, tuple[str, ...]]:
    """Check whether a raster is a Cloud Optimized GeoTIFF.

    Parameters
    ----------
    raster_path : path-like
        File to inspect.

    Returns
    -------
    tuple of (bool, tuple of str)
        Whether the file is a valid COG, and the validator's error messages.
        The messages are the useful part of a negative result, so they are
        returned rather than discarded.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    rasterio.errors.RasterioIOError
        If the file is not a readable raster.

    """
    path = Path(raster_path)
    if not path.exists():
        reject(f"file does not exist: {path}", FileNotFoundError)

    from rio_cogeo.cogeo import cog_validate

    is_valid, errors, _warnings = cog_validate(str(path))
    return bool(is_valid), tuple(str(error) for error in errors)


def validate_alignment(
    raster_path: str | PathLike,
    reference_geogrid: RasterGridSpec,
    tol: float = 1e-6,
) -> bool:
    """Validate that a raster is aligned with a reference grid.

    Parameters
    ----------
    raster_path : path-like
        Path to the raster to validate.
    reference_geogrid : RasterGridSpec
        The reference grid.
    tol : float
        Numerical tolerance for the alignment check.

    Returns
    -------
    bool
        True if aligned, False otherwise.

    """
    return read_geogrid(raster_path).is_aligned(reference_geogrid, tol=tol)
