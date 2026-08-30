"""Nearest-neighbour projection of a geo mask onto the radar grid (D2).

PROPOSAL-0039 "Radar projection": a boolean water/removed mask lives on a
geographic grid while interferogram support masks live on the radar
``(azimuth, range)`` grid.  This module projects the mask into radar
coordinates with **nearest-neighbour only** semantics:

- **geo mode** — a dense :class:`~faninsar.processing.pipeline.geo_lut.\
Geo2RdrLUT` is available (geo-coregistered runs).  Mask values are scattered
at the LUT's ``az_full``/``rg_full`` radar indices (``rint`` nearest) and
radar pixels not covered by any LUT cell are filled with the value of the
nearest covered pixel.
- **radar mode** — no dense LUT is available.  ``run_geo2rdr`` runs chunked
  over the mask grid rows (the
  :func:`faninsar.processing.pipeline.production.\
_apply_geo_topographic_phase_chunked` memmap + watchdog pattern), scattering
  converged in-bounds radar indices into the radar plane.

The full-resolution boolean plane caches under ``<cache_dir>/<cache_key>.npy``
(one file per ``(vector-layer digest, buffer, resolution, DEM identity)``
identity, digested by the caller through
:func:`radar_projection_cache_key`), so an unchanged configuration never
re-projects.  A ``multilook`` window reduces the cached full-resolution plane
with ``any()``: a look is water/removed when **any** contributing
full-resolution pixel is water.

Array conventions follow :mod:`faninsar.processing.masking.mask`: the input
mask product is uint8 with ``1`` = removed (water) and ``255`` = invalid where
no data exists; invalid cells never remove data.  The returned plane is
boolean with ``True`` = removed.

Governing proposal: PROPOSAL-0039 (radar projection, cache identity,
multilook ``any()`` reduction); PROPOSAL-0038 consumes the projected plane as
a support input at the IFG/unwrap seams.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from affine import Affine

    from faninsar.processing.pipeline.geo_lut import Geo2RdrLUT

logger = setup_logger(__name__)

__all__ = [
    "project_mask_to_radar",
    "radar_projection_cache_key",
]

#: Default geo2rdr chunk height (mask grid rows per tile) in radar mode.
DEFAULT_CHUNK_SIZE = 256


def radar_projection_cache_key(
    *,
    vector_digest: str,
    buffer_km: float,
    resolution_m: float | None,
    dem_identity: str,
) -> str:
    """Digest the radar-projection cache identity (PROPOSAL-0039).

    The key folds exactly the four identity components pinned by the
    proposal — vector-layer digest, land buffer, mask grid resolution, and
    DEM identity — so any configuration change projects a fresh plane while
    an unchanged configuration reuses the cached one.

    Parameters
    ----------
    vector_digest : str
        Digest of the vector mask layer (e.g. the ``WaterLayer.identity`` of
        the masking manager or any caller-defined layer digest).
    buffer_km : float
        Land buffer width in kilometres used to build the mask.
    resolution_m : float or None
        Mask grid resolution override in metres (``None`` records the DEM /
        native grid resolution).
    dem_identity : str
        Stable DEM identity string (path, height, or sampler name).

    Returns
    -------
    str
        Lowercase SHA-256 hexdigest of the canonical identity payload.

    """
    payload = {
        "vector_digest": str(vector_digest),
        "buffer_km": float(buffer_km),
        "resolution_m": None if resolution_m is None else float(resolution_m),
        "dem_identity": str(dem_identity),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_mask_plane(mask: np.ndarray | str | Path) -> np.ndarray:
    """Read the mask product into a uint8 plane (1 = removed, 255 = invalid)."""
    if isinstance(mask, (str, Path)):
        import rasterio

        with rasterio.open(Path(mask)) as dataset:
            plane = np.asarray(dataset.read(1))
    else:
        plane = np.asarray(mask)
    if plane.ndim != 2:
        message = "mask plane must be two-dimensional"
        logger.error(message)
        raise ValueError(message)
    if plane.dtype == np.bool_:
        return plane.astype(np.uint8)
    if plane.dtype.kind not in "ui":
        message = f"mask plane must be an integer or boolean raster, got {plane.dtype}"
        logger.error(message)
        raise ValueError(message)
    return plane.astype(np.uint8, copy=False)


def _multilook_any(plane: np.ndarray, multilook: tuple[int, int]) -> np.ndarray:
    """Reduce a boolean plane per look window with ``any()`` (tails kept).

    A look is removed when any contributing full-resolution pixel is removed;
    edge windows keep their partial support (the
    :func:`faninsar.processing.interferometry.pair.form_interferogram` ceil
    convention).
    """
    az_looks, rg_looks = (int(multilook[0]), int(multilook[1]))
    if az_looks < 1 or rg_looks < 1:
        message = f"multilook factors must be >= 1, got {multilook}"
        logger.error(message)
        raise ValueError(message)
    if az_looks == 1 and rg_looks == 1:
        return np.ascontiguousarray(plane, dtype=bool)
    height, width = plane.shape
    out_h = (height + az_looks - 1) // az_looks
    out_w = (width + rg_looks - 1) // rg_looks
    padded = np.zeros((out_h * az_looks, out_w * rg_looks), dtype=bool)
    padded[:height, :width] = plane
    blocks = padded.reshape(out_h, az_looks, out_w, rg_looks)
    return np.any(blocks, axis=(1, 3))


def _scatter_lut_mask(
    water_geo: np.ndarray,
    lut: Geo2RdrLUT,
    full_radar_shape: tuple[int, int],
) -> np.ndarray:
    """Scatter the geo mask at the dense LUT indices and nearest-fill holes.

    The mask plane must share the LUT's geographic grid exactly (its
    ``shape``).  Valid LUT cells land on radar pixel
    ``(rint(az_full), rint(rg_full))``; a radar pixel is water when **any**
    contributing geo cell is water.  Radar pixels covered by no LUT cell are
    filled with the value of the nearest covered pixel (nearest fill; an
    entirely uncovered radar frame projects to all-False — no mask
    information, nothing removed).
    """
    from scipy.ndimage import distance_transform_edt

    height, width = (int(full_radar_shape[0]), int(full_radar_shape[1]))
    if water_geo.shape != lut.valid.shape:
        message = (
            f"mask plane shape {water_geo.shape} does not match the LUT "
            f"geographic grid {lut.valid.shape}"
        )
        logger.error(message)
        raise ValueError(message)
    water = np.asarray(water_geo) == 1
    az_values = np.asarray(lut.az_full, dtype=np.float64)
    rg_values = np.asarray(lut.rg_full, dtype=np.float64)
    finite = (
        np.isfinite(az_values)
        & np.isfinite(rg_values)
        & np.asarray(lut.valid, dtype=bool)
    )
    # Replace non-finite entries before the integer cast (they are excluded
    # by ``finite`` anyway) so the cast never warns on NaN.
    az_index = np.rint(np.where(finite, az_values, 0.0)).astype(np.int64)
    rg_index = np.rint(np.where(finite, rg_values, 0.0)).astype(np.int64)
    in_bounds = (
        finite
        & (az_index >= 0)
        & (az_index < height)
        & (rg_index >= 0)
        & (rg_index < width)
    )
    out = np.zeros((height, width), dtype=bool)
    covered = np.zeros((height, width), dtype=bool)
    if np.any(in_bounds):
        rows = az_index[in_bounds]
        cols = rg_index[in_bounds]
        np.logical_or.at(out, (rows, cols), water[in_bounds])
        covered[rows, cols] = True
    if bool(covered.all()):
        return out
    if not bool(covered.any()):
        logger.debug(
            "radar mask projection: no LUT cell covered the radar frame; "
            "the projected mask is empty (nothing removed)"
        )
        return out
    _, nearest = distance_transform_edt(~covered, return_indices=True)
    return out[nearest[0], nearest[1]]


def _project_radar_mode(
    water_geo: np.ndarray,
    *,
    geometry: Any,
    mask_transform: Affine,
    dem: Any | None,
    device: str,
    chunk_size: int,
    watchdog: Any | None,
    output_plane: np.ndarray,
) -> None:
    """Run chunked ``run_geo2rdr`` over the mask grid rows (radar mode).

    Mirrors the ``_apply_geo_topographic_phase_chunked`` pattern: row tiles of
    ``chunk_size`` mask rows, a disk-backed output plane when caching, page
    release after every tile, and one watchdog sample per completed tile.
    Mask cell centres map through ``run_geo2rdr`` to full-resolution radar
    indices; converged in-bounds results scatter with ``rint`` nearest (a
    radar pixel is water when any contributing mask cell is water).  Without
    a ``dem`` sampler the projection runs at ellipsoidal height zero — the
    documented radar-mode approximation; the dense-LUT geo mode is the
    DEM-exact path.
    """
    from faninsar.processing.geometry.prepare_production import run_geo2rdr
    from faninsar.processing.memory import release_memmap_pages

    grid_height, grid_width = water_geo.shape
    water = np.asarray(water_geo) == 1
    rows_index, cols_index = np.indices((grid_height, grid_width), dtype=np.float64)
    longitudes, latitudes = mask_transform * (cols_index + 0.5, rows_index + 0.5)
    for row_start in range(0, grid_height, chunk_size):
        row_stop = min(row_start + chunk_size, grid_height)
        rows = slice(row_start, row_stop)
        height_chunk = np.zeros((row_stop - row_start, grid_width), dtype=np.float64)
        if dem is not None:
            height_chunk = np.asarray(
                dem.sample(latitudes[rows], longitudes[rows]), dtype=np.float64
            )
        result = run_geo2rdr(
            geometry,
            latitudes[rows],
            longitudes[rows],
            height_chunk,
            device=device,
        )
        azimuth = np.asarray(result.azimuth_index, dtype=np.float64)
        range_index = np.asarray(result.range_index, dtype=np.float64)
        converged = (
            np.asarray(result.converged, dtype=bool)
            & np.isfinite(azimuth)
            & np.isfinite(range_index)
        )
        azimuth_int = np.rint(np.where(converged, azimuth, 0.0)).astype(np.int64)
        range_int = np.rint(np.where(converged, range_index, 0.0)).astype(np.int64)
        height_img, width_img = output_plane.shape
        in_bounds = (
            converged
            & (azimuth_int >= 0)
            & (azimuth_int < height_img)
            & (range_int >= 0)
            & (range_int < width_img)
        )
        if np.any(in_bounds):
            np.logical_or.at(
                output_plane,
                (azimuth_int[in_bounds], range_int[in_bounds]),
                water[rows][in_bounds],
            )
        if isinstance(output_plane, np.memmap):
            output_plane.flush()
            release_memmap_pages(output_plane)
        if watchdog is not None:
            watchdog.sample(f"mask_radar_projection:{row_start}:{row_stop}")


def _cache_path(cache_dir: str | Path | None, cache_key: str | None) -> Path | None:
    """Resolve the projection cache file path (both parts required)."""
    if cache_key is None and cache_dir is None:
        return None
    if cache_key is None or cache_dir is None:
        message = "radar-projection cache requires both cache_key and cache_dir"
        logger.error(message)
        raise ValueError(message)
    key = str(cache_key)
    if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
        message = (
            "radar-projection cache_key must be a lowercase SHA-256 hexdigest "
            f"(see radar_projection_cache_key); got {cache_key!r}"
        )
        logger.error(message)
        raise ValueError(message)
    return Path(cache_dir) / f"{key}.npy"


def project_mask_to_radar(
    mask: np.ndarray | str | Path,
    *,
    full_radar_shape: tuple[int, int],
    lut: Geo2RdrLUT | None = None,
    geometry: Any | None = None,
    mask_transform: Affine | None = None,
    dem: Any | None = None,
    device: str = "cpu",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    multilook: tuple[int, int] = (1, 1),
    cache_key: str | None = None,
    cache_dir: str | Path | None = None,
    watchdog: Any | None = None,
) -> np.ndarray:
    """Project a geographic mask onto the full-resolution radar grid.

    Parameters
    ----------
    mask : numpy.ndarray, str, or pathlib.Path
        The mask product on its geographic grid: a uint8 array with
        ``1`` = removed (water), ``0`` = keep, ``255`` = invalid (invalid
        cells never remove data), or a path to such a single-band GeoTIFF.
    full_radar_shape : tuple of int
        Full-resolution radar image shape ``(azimuth, range)``.
    lut : Geo2RdrLUT, optional
        Dense geographic-to-radar lookup table (geo mode).  The mask plane
        must share the LUT's geographic grid shape exactly.
    geometry : object, optional
        Reference-scene radar geometry model (radar mode).  Exactly one of
        ``lut`` and ``geometry`` must be supplied.
    mask_transform : affine.Affine, optional
        Affine transform of the mask grid (required in radar mode; it maps
        ``(col, row)`` pixel coordinates to geodetic degrees).
    dem : object, optional
        Optional DEM sampler (``sample(latitude, longitude)``) used by the
        radar-mode geo2rdr solve; ``None`` runs at ellipsoidal height zero.
    device : str, optional
        Numerical device for the radar-mode geo2rdr solve.
    chunk_size : int, optional
        Mask grid rows per radar-mode tile (default 256).
    multilook : tuple of int, optional
        ``(azimuth, range)`` look factors.  The cached full-resolution plane
        is reduced with ``any()`` per look window (default ``(1, 1)`` keeps
        full resolution).
    cache_key : str, optional
        SHA-256 identity from :func:`radar_projection_cache_key`.  Required
        together with ``cache_dir`` to enable caching.
    cache_dir : path-like, optional
        Directory receiving ``<cache_key>.npy`` (one boolean plane per
        ``(vector digest, buffer, resolution, DEM identity)`` identity).
    watchdog : object, optional
        Memory guard with ``sample(label)`` invoked after every radar-mode
        tile (the production chunked-stage pattern).

    Returns
    -------
    numpy.ndarray
        Boolean plane with ``True`` = water/removed, at the multilooked
        radar shape (``ceil(full_radar_shape / multilook)``; the full shape
        for ``(1, 1)``).

    Raises
    ------
    ValueError
        If the mode selection, plane shape, cache key, or multilook factors
        are invalid.

    """
    if (lut is None) == (geometry is None):
        message = (
            "project_mask_to_radar requires exactly one projection source: "
            "a dense Geo2RdrLUT (geo mode) or a radar geometry model "
            "(radar mode)"
        )
        logger.error(message)
        raise ValueError(message)
    if lut is not None and lut.valid.shape != tuple(
        int(size) for size in lut.az_full.shape
    ):
        # Defensive: the LUT planes must agree before any scatter.
        message = "Geo2RdrLUT planes disagree on the geographic grid shape"
        logger.error(message)
        raise ValueError(message)

    height, width = (int(full_radar_shape[0]), int(full_radar_shape[1]))
    if height < 1 or width < 1:
        message = f"full_radar_shape must be positive, got {full_radar_shape!r}"
        logger.error(message)
        raise ValueError(message)

    cache_file = _cache_path(cache_dir, cache_key)
    full_plane: np.ndarray | None = None
    if cache_file is not None and cache_file.is_file():
        full_plane = np.load(cache_file, mmap_mode="r")
        if full_plane.shape != (height, width) or full_plane.dtype != np.bool_:
            message = (
                f"radar-projection cache {cache_file} holds "
                f"{full_plane.shape}/{full_plane.dtype}; expected "
                f"{(height, width)}/bool — the cache identity is stale"
            )
            logger.error(message)
            raise ValueError(message)
        logger.info("radar-projection cache hit: %s", cache_file)
        return _multilook_any(np.asarray(full_plane), multilook)

    water_geo = (_load_mask_plane(mask) == 1).astype(bool)
    if lut is not None:
        full_plane = _scatter_lut_mask(water_geo, lut, (height, width))
        if cache_file is not None:
            _atomic_cache_save(cache_file, full_plane)
        logger.info(
            "Projected mask to radar (geo mode): %d/%d pixels removed",
            int(np.count_nonzero(full_plane)),
            full_plane.size,
        )
        return _multilook_any(full_plane, multilook)

    if mask_transform is None:
        message = "radar mode requires the mask grid transform"
        logger.error(message)
        raise ValueError(message)
    if chunk_size < 1:
        message = f"chunk_size must be >= 1, got {chunk_size}"
        logger.error(message)
        raise ValueError(message)
    staged: np.memmap | None = None
    staging_path: Path | None = None
    try:
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            staged, staging_path = _staging_memmap(cache_file, (height, width))
            full_plane = staged
        else:
            full_plane = np.zeros((height, width), dtype=bool)
        _project_radar_mode(
            water_geo,
            geometry=geometry,
            mask_transform=mask_transform,
            dem=dem,
            device=device,
            chunk_size=chunk_size,
            watchdog=watchdog,
            output_plane=full_plane,
        )
    except BaseException:
        # Never publish a partially projected plane under the cache identity.
        if staged is not None:
            del staged
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)
        raise
    if staging_path is not None:
        # The staging plane is a raw boolean dump; publish it under the
        # cache identity as a proper .npy (header-carrying, memmap-loadable).
        staged.flush()
        del staged
        raw = np.memmap(staging_path, mode="r", dtype=bool, shape=(height, width))
        _atomic_cache_save(cache_file, raw)
        del raw
        staging_path.unlink(missing_ok=True)
        full_plane = np.ascontiguousarray(np.load(cache_file, mmap_mode="r"))
    logger.info(
        "Projected mask to radar (radar mode): %d/%d pixels removed",
        int(np.count_nonzero(full_plane)),
        full_plane.size,
    )
    return _multilook_any(np.asarray(full_plane), multilook)


def _staging_memmap(cache_file: Path, shape: tuple[int, int]) -> tuple[np.memmap, Path]:
    """Create a disk-backed staging plane beside the final cache file."""
    staging_path = cache_file.with_name(
        f".{cache_file.name}.{os.getpid()}-{secrets.token_hex(8)}.tmp"
    )
    return np.memmap(staging_path, mode="w+", dtype=bool, shape=shape), staging_path


def _atomic_cache_save(cache_file: Path, plane: np.ndarray) -> None:
    """Publish one full-resolution cache plane (temp file + rename)."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_name(
        f".{cache_file.name}.{os.getpid()}-{secrets.token_hex(8)}.tmp"
    )
    try:
        with temporary.open("wb") as stream:
            np.save(stream, np.ascontiguousarray(plane, dtype=bool))
        temporary.replace(cache_file)
    finally:
        temporary.unlink(missing_ok=True)
