"""Nearest-neighbour projection of a canonical mask onto the radar grid.

PROPOSAL-0040: a canonical tri-state mask lives on a
geographic grid while interferogram support masks live on the radar
``(azimuth, range)`` grid.  This module projects the mask into radar
coordinates with **nearest-neighbour only** semantics:

- **geo mode** — a dense :class:`~faninsar.processing.pipeline.geo_lut.\
Geo2RdrLUT` is available (geo-coregistered runs).  Mask values are scattered
at the LUT's ``az_full``/``rg_full`` radar indices (``rint`` nearest) and
radar pixels not covered by any LUT cell remain invalid (``255``).
- **radar mode** — no dense LUT is available.  ``run_geo2rdr`` runs chunked
  over the mask grid rows (the
  :func:`faninsar.processing.pipeline.production.\
_apply_geo_topographic_phase_chunked` memmap + watchdog pattern), scattering
  converged in-bounds radar indices into the radar plane.

The full-resolution uint8 plane caches under ``<cache_dir>/<cache_key>.npy``
using the materialized source mask, grids, projection model, DEM, and target
validity as its identity.  A ``multilook`` request is a nearest-neighbour view
of that plane; labels are never aggregated into a new class.

Array conventions follow :mod:`faninsar.processing.masking.mask`: ``0`` = keep,
``1`` = removed (water), and ``255`` = invalid.  The returned plane is always
canonical contiguous uint8.

Governing proposal: PROPOSAL-0040; PROPOSAL-0038 consumes the projected plane
as a support input at the IFG/unwrap seams.
"""

from __future__ import annotations

import hashlib
import json
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
KEEP = np.uint8(0)
EXCLUDED = np.uint8(1)
INVALID = np.uint8(255)


def radar_projection_cache_key(
    *,
    vector_digest: str | None = None,
    buffer_km: float | None = None,
    resolution_m: float | None = None,
    dem_identity: object | None = None,
    source_mask_identity: object | None = None,
    master_grid: object | None = None,
    target_grid: object | None = None,
    reference_scene: object | None = None,
    lut: object | None = None,
    geometry: object | None = None,
    dem: object | None = None,
    target_validity: object | None = None,
) -> str:
    """Digest every projection identity observable at this seam.

    The materialized source mask, master and target grids, reference scene,
    geometry/LUT, DEM, and final target validity form the cache identity.
    ``vector_digest`` and ``dem_identity`` are retained as input aliases for
    callers transitioning to those canonical identities.  Buffer and
    resolution settings do not identify a materialized product and are
    intentionally ignored.

    Parameters
    ----------
    vector_digest, dem_identity : object, optional
        Legacy aliases for ``source_mask_identity`` and ``dem``.
    buffer_km, resolution_m : float, optional
        Legacy settings ignored by the hard-cutover identity.
    source_mask_identity, master_grid, target_grid, reference_scene : object, optional
        Materialized source, source/target grids, and reference-scene inputs.
    lut, geometry, dem, target_validity : object, optional
        Projection model inputs and explicit target validity plane.

    Returns
    -------
    str
        Lowercase SHA-256 hexdigest of the canonical identity payload.

    """
    if source_mask_identity is None:
        source_mask_identity = vector_digest
    if dem is None:
        dem = dem_identity
    payload = {
        "proposal": "PROPOSAL-0040",
        "source_mask_identity": _identity(source_mask_identity),
        "master_grid": _identity(master_grid),
        "target_grid": _identity(target_grid),
        "reference_scene": _identity(reference_scene),
        "lut": _identity(lut),
        "geometry": _identity(geometry),
        "dem": _identity(dem),
        "target_validity": _identity(target_validity),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _identity(value: object) -> object:  # noqa: PLR0911
    """Return a stable representation for materialization identity inputs."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        path = value.expanduser().resolve()
        try:
            stat = path.stat()
            return {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        except OSError:
            return {"path": str(path)}
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "dtype": str(array.dtype),
            "shape": array.shape,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    identity = getattr(value, "identity", None)
    if identity is not None and not callable(identity):
        return {"type": type(value).__qualname__, "identity": _identity(identity)}
    if (
        hasattr(value, "crs")
        and hasattr(value, "transform")
        and hasattr(value, "shape")
    ):
        result: dict[str, object] = {
            "type": type(value).__qualname__,
            "crs": str(value.crs),
            "transform": tuple(value.transform),
            "shape": tuple(value.shape),
        }
        for attribute in ("bounds", "validity"):
            if hasattr(value, attribute):
                result[attribute] = _identity(getattr(value, attribute))
        return result
    if hasattr(value, "__dict__"):
        return {
            "type": type(value).__qualname__,
            "attributes": {
                key: _identity(item)
                for key, item in sorted(vars(value).items())
                if not key.startswith("_")
            },
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _load_mask_plane(mask: object) -> tuple[np.ndarray, object | None, object | None]:
    """Read and validate a canonical plane, grid, and source identity."""
    source_grid = getattr(mask, "grid", None)
    source_identity = getattr(mask, "identity", None)
    if hasattr(mask, "data") and source_identity is not None:
        plane = np.asarray(mask.data)
    elif isinstance(mask, (str, Path)):
        path = Path(mask)
        import rasterio

        with rasterio.open(path) as dataset:
            plane = np.asarray(dataset.read(1))
            source_grid = {
                "crs": str(dataset.crs),
                "transform": dataset.transform,
                "shape": (dataset.height, dataset.width),
            }
        source_identity = _identity(path)
    else:
        plane = np.asarray(mask)
        source_identity = _identity(plane)
    if plane.ndim != 2:
        message = "mask plane must be two-dimensional"
        logger.error(message)
        raise ValueError(message)
    if plane.dtype == np.bool_:
        result = plane.astype(np.uint8)
    elif plane.dtype != np.uint8:
        message = f"mask plane must be bool or uint8, got {plane.dtype}"
        logger.error(message)
        raise TypeError(message)
    else:
        result = np.ascontiguousarray(plane)
    if np.any(~np.isin(result, np.array([0, 1, 255], dtype=np.uint8))):
        message = "mask plane may contain only canonical labels 0, 1, and 255"
        logger.error(message)
        raise ValueError(message)
    return result, source_grid, source_identity


def _nearest_label_view(plane: np.ndarray, multilook: tuple[int, int]) -> np.ndarray:
    """Return a nearest-neighbour view of a canonical hard-label plane."""
    az_looks, rg_looks = (int(multilook[0]), int(multilook[1]))
    if az_looks < 1 or rg_looks < 1:
        message = f"multilook factors must be >= 1, got {multilook}"
        logger.error(message)
        raise ValueError(message)
    if az_looks == 1 and rg_looks == 1:
        return np.ascontiguousarray(plane, dtype=np.uint8)
    height, width = plane.shape
    out_h = (height + az_looks - 1) // az_looks
    out_w = (width + rg_looks - 1) // rg_looks
    # Select the pixel nearest to each output pixel centre.  This preserves
    # the original hard label and has deterministic edge behaviour for odd
    # and even look factors alike.
    azimuth = np.minimum(
        np.arange(out_h) * az_looks + (az_looks - 1) // 2, height - 1
    )
    range_index = np.minimum(
        np.arange(out_w) * rg_looks + (rg_looks - 1) // 2, width - 1
    )
    return np.ascontiguousarray(plane[np.ix_(azimuth, range_index)], dtype=np.uint8)


def _scatter_candidates(
    output: np.ndarray,
    best_distance: np.ndarray,
    labels: np.ndarray,
    azimuth: np.ndarray,
    range_index: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Assign nearest hard-label candidates into an output plane."""
    finite = (
        np.asarray(valid, dtype=bool) & np.isfinite(azimuth) & np.isfinite(range_index)
    )
    height, width = output.shape
    azimuth_int = np.rint(np.where(finite, azimuth, 0.0)).astype(np.int64)
    range_int = np.rint(np.where(finite, range_index, 0.0)).astype(np.int64)
    in_bounds = (
        finite
        & (azimuth_int >= 0)
        & (azimuth_int < height)
        & (range_int >= 0)
        & (range_int < width)
    )
    if not np.any(in_bounds):
        return
    rows = azimuth_int[in_bounds]
    cols = range_int[in_bounds]
    candidate_distance = (azimuth[in_bounds] - rows) ** 2 + (
        range_index[in_bounds] - cols
    ) ** 2
    take = candidate_distance < best_distance[rows, cols]
    if np.any(take):
        selected_labels = np.asarray(labels)[in_bounds]
        output[rows[take], cols[take]] = selected_labels[take]
        best_distance[rows[take], cols[take]] = candidate_distance[take]


def _scatter_lut_mask(
    labels_geo: np.ndarray,
    lut: Geo2RdrLUT,
    full_radar_shape: tuple[int, int],
) -> np.ndarray:
    """Scatter canonical labels through a dense LUT; holes remain invalid."""
    if labels_geo.shape != lut.valid.shape:
        message = (
            f"mask plane shape {labels_geo.shape} does not match the LUT "
            f"geographic grid {lut.valid.shape}"
        )
        logger.error(message)
        raise ValueError(message)
    height, width = (int(full_radar_shape[0]), int(full_radar_shape[1]))
    output = np.full((height, width), INVALID, dtype=np.uint8)
    best_distance = np.full((height, width), np.inf, dtype=np.float64)
    az_values = np.asarray(lut.az_full, dtype=np.float64)
    rg_values = np.asarray(lut.rg_full, dtype=np.float64)
    _scatter_candidates(
        output,
        best_distance,
        labels_geo.ravel(),
        az_values.ravel(),
        rg_values.ravel(),
        np.asarray(lut.valid, dtype=bool).ravel(),
    )
    return output


def _project_radar_mode(
    labels_geo: np.ndarray,
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

    grid_height, grid_width = labels_geo.shape
    rows_index, cols_index = np.indices((grid_height, grid_width), dtype=np.float64)
    longitudes, latitudes = mask_transform * (cols_index + 0.5, rows_index + 0.5)
    best_distance = np.full(output_plane.shape, np.inf, dtype=np.float64)
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
        _scatter_candidates(
            output_plane,
            best_distance,
            labels_geo[rows].ravel(),
            np.asarray(result.azimuth_index, dtype=np.float64).ravel(),
            np.asarray(result.range_index, dtype=np.float64).ravel(),
            np.asarray(result.converged, dtype=bool).ravel(),
        )
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
    mask: object,
    *,
    full_radar_shape: tuple[int, int] | None = None,
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
    master_grid: object | None = None,
    target_grid: object | None = None,
    reference_scene: object | None = None,
    target_validity: object | None = None,
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
        ``(azimuth, range)`` nearest-neighbour view factors (default
        ``(1, 1)`` keeps full resolution).  No labels are aggregated.
    cache_key : str, optional
        SHA-256 identity from :func:`radar_projection_cache_key`.  Required
        together with ``cache_dir`` to enable caching.
    cache_dir : path-like, optional
        Directory receiving ``<cache_key>.npy`` for the canonical projection
        identity.
    watchdog : object, optional
        Memory guard with ``sample(label)`` invoked after every radar-mode
        tile (the production chunked-stage pattern).
    master_grid, target_grid, reference_scene : object, optional
        Source/target grid and reference-scene identity inputs.
    target_validity : array-like, optional
        Final target validity plane; false cells are always ``255``.

    Returns
    -------
    numpy.ndarray
        Canonical uint8 plane at the nearest-neighbour radar shape
        (``ceil(full_radar_shape / multilook)``; the full shape for
        ``(1, 1)``).

    Raises
    ------
    ValueError
        If the mode selection, plane shape, cache key, or multilook factors
        are invalid.

    """
    if (lut is None) == (geometry is None):
        message = "project_mask_to_radar requires exactly one projection source"
        logger.error(message)
        raise ValueError(message)
    labels, source_grid, source_identity = _load_mask_plane(mask)
    if (
        master_grid is not None
        and tuple(getattr(master_grid, "shape", labels.shape)) != labels.shape
    ):
        message = f"mask plane shape {labels.shape} does not match master grid"
        logger.error(message)
        raise ValueError(message)
    if mask_transform is None and source_grid is not None:
        mask_transform = getattr(source_grid, "transform", None)
        if mask_transform is None and isinstance(source_grid, dict):
            mask_transform = source_grid.get("transform")
    if lut is not None and lut.valid.shape != tuple(
        int(size) for size in lut.az_full.shape
    ):
        # Defensive: the LUT planes must agree before any scatter.
        message = "Geo2RdrLUT planes disagree on the geographic grid shape"
        logger.error(message)
        raise ValueError(message)

    if full_radar_shape is None:
        if target_grid is not None and hasattr(target_grid, "shape"):
            full_radar_shape = tuple(int(size) for size in target_grid.shape)
        elif lut is not None:
            full_radar_shape = tuple(int(size) for size in lut.full_radar_shape)
        else:
            message = "full_radar_shape or target_grid is required"
            logger.error(message)
            raise ValueError(message)
    height, width = (int(full_radar_shape[0]), int(full_radar_shape[1]))
    if height < 1 or width < 1:
        message = f"full_radar_shape must be positive, got {full_radar_shape!r}"
        logger.error(message)
        raise ValueError(message)
    if geometry is not None and mask_transform is None:
        message = "radar mode requires the mask grid transform"
        logger.error(message)
        raise ValueError(message)
    if chunk_size < 1:
        message = f"chunk_size must be >= 1, got {chunk_size}"
        logger.error(message)
        raise ValueError(message)

    validity = target_validity
    if validity is None and target_grid is not None:
        validity = getattr(target_grid, "validity", None)
    validity_array = (
        np.ones((height, width), dtype=bool)
        if validity is None
        else np.asarray(validity, dtype=bool)
    )
    if validity_array.shape != (height, width):
        message = (
            f"target validity shape {validity_array.shape} does not match "
            f"{(height, width)}"
        )
        logger.error(message)
        raise ValueError(message)
    if cache_key is None and cache_dir is not None:
        cache_key = radar_projection_cache_key(
            source_mask_identity=source_identity,
            master_grid=master_grid if master_grid is not None else source_grid,
            target_grid=target_grid,
            reference_scene=reference_scene,
            lut=lut,
            geometry=geometry,
            dem=dem,
            target_validity=validity_array,
            dem_identity=dem,
        )
    cache_file = _cache_path(cache_dir, cache_key)
    full_plane: np.ndarray | None = None
    if cache_file is not None and cache_file.is_file():
        full_plane = np.asarray(np.load(cache_file))
        if full_plane.shape != (height, width) or full_plane.dtype != np.uint8:
            message = f"radar-projection cache {cache_file} is not canonical uint8"
            logger.error(message)
            raise ValueError(message)
        if np.any(~np.isin(full_plane, np.array([0, 1, 255], dtype=np.uint8))):
            message = (
                f"radar-projection cache {cache_file} contains non-canonical labels"
            )
            logger.error(message)
            raise ValueError(message)
    elif lut is not None:
        full_plane = _scatter_lut_mask(labels, lut, (height, width))
    else:
        full_plane = np.full((height, width), INVALID, dtype=np.uint8)
        _project_radar_mode(
            labels,
            geometry=geometry,
            mask_transform=mask_transform,
            dem=dem,
            device=device,
            chunk_size=chunk_size,
            watchdog=watchdog,
            output_plane=full_plane,
        )
    full_plane[~validity_array] = INVALID
    if cache_file is not None and not cache_file.is_file():
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, np.ascontiguousarray(full_plane, dtype=np.uint8))
    return _nearest_label_view(full_plane, multilook)
