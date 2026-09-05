"""Chunked NISAR RSLC scene provider (PROPOSAL-0035).

An explicit ``nisar_window`` retains the bounded smoke path.  When the window
is omitted, the provider finds the common radar coverage and publishes it as
deterministic scene tiles.  Geographic production projects the same bounded
radar tiles into local windows of one caller-supplied geographic grid.  Stack
remains responsible for lifecycle, markers, alignment identity, and all
downstream interferometric products.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.coordinates import GeoGrid, RadarGrid
from faninsar.processing.errors import (
    InvalidProcessingStateError,
    reject_invalid_state,
    reject_pair_configuration,
)
from faninsar.processing.slc import RadarSLC
from faninsar.stack.provider import (
    SourceHandle,
    UnsupportedStackCapabilityError,
)
from faninsar.stack.scene_store import (
    CoregisteredSceneStore,
    scene_grid_identity,
    write_scene_unit,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from faninsar.processing.contracts import SLCProduct
    from faninsar.processing.dem import DEM
    from faninsar.stack.provider import SceneProductionCallback

logger = setup_logger(__name__)


def _source_digest(path: Path) -> tuple[str, str]:
    """Return canonical source id and content SHA-256 for one RSLC path."""
    source_id = str(path.expanduser().resolve(strict=False))
    digest = hashlib.sha256()
    try:
        with Path(source_id).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except (FileNotFoundError, NotADirectoryError):
        digest.update(f"missing:{source_id}".encode())
    except OSError as error:
        logger.exception("NISAR source cannot be snapshotted: %s", source_id)
        reject_invalid_state(
            f"NISAR source cannot be snapshotted: {source_id}: {error}"
        )
    return source_id, digest.hexdigest()


def _dem_identity(dem: object) -> str:
    """Return a stable, human-readable identity for a geometry DEM."""
    qualified_name = f"{type(dem).__module__}.{type(dem).__qualname__}"
    height = getattr(dem, "height_m", None)
    if height is not None:
        try:
            return f"{qualified_name}:height_m={float(height):.17g}"
        except (TypeError, ValueError):
            return f"{qualified_name}:height_m={height!r}"
    for name in ("content_digest", "source_digest", "sha256"):
        digest = getattr(dem, name, None)
        if isinstance(digest, str) and digest:
            return f"{qualified_name}:{name}={digest}"
    path = getattr(dem, "path", None)
    if path is not None:
        resolved = Path(path).expanduser().resolve(strict=False)
        try:
            stat = resolved.stat()
        except OSError:
            return f"{qualified_name}:path={resolved}"
        return (
            f"{qualified_name}:path={resolved}:size={stat.st_size}:"
            f"mtime_ns={stat.st_mtime_ns}"
        )
    return qualified_name


def _contiguous_geometry_input(value: object) -> object:
    """Materialize one geometry input as a contiguous one-dimensional lane.

    Geometry results may be views (for example, a one-pixel result produced by
    a broadcasted native output).  The public geometry validator intentionally
    rejects those views before native dispatch.  Keep this normalization at
    the NISAR provider seam so callers and the Sentinel-1 path retain their
    existing array contracts.

    Parameters
    ----------
    value : object
        NumPy array or Torch tensor returned by the geometry operation.

    Returns
    -------
    object
        A same-dtype, one-dimensional contiguous value.  Torch tensors remain
        on their original device.

    """
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value).reshape(-1)
    contiguous = getattr(value, "contiguous", None)
    if callable(contiguous):
        normalized = contiguous()
        reshape = getattr(normalized, "reshape", None)
        return reshape(-1) if callable(reshape) else normalized
    return np.ascontiguousarray(np.asarray(value)).reshape(-1)


@dataclass(frozen=True, slots=True)
class NisarPairState:
    """Minimal state consumed by :meth:`Stack.coregister_scenes`.

    The bounded provider does not estimate residual offsets.  Explicit zero
    values make that fact part of the state rather than leaving marker
    generation to infer missing fields.
    """

    pair_id: str
    range_shift_px: float = 0.0
    azimuth_shift_px: float = 0.0
    esd_azimuth_shift_px: float = 0.0
    amplitude_residual_rg_px: float = 0.0
    stage_timings_s: dict[str, float] | None = None
    coregistration_timings_s: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class _DenseRadarMapping:
    """Secondary source coordinates for one Reference radar tile."""

    azimuth: np.ndarray
    range_index: np.ndarray
    valid: np.ndarray
    source_bounds: tuple[int, int, int, int] | None


def _apply_range_offset_flatten(
    secondary: np.ndarray,
    secondary_range_index: np.ndarray,
    *,
    primary_col_origin: int,
    range_spacing_m: float,
    wavelength_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the NISAR ellipsoidal range-offset phase to one aligned tile.

    The NISAR RIFG convention removes the geometric screen from the complex
    interferogram.  Since scene stores retain the aligned secondary SLC and
    form ``reference * conj(secondary)`` later, the inverse-conjugate screen
    is applied to the secondary here.  The resulting interferogram therefore
    carries ``exp(-1j * phase)`` with the ISCE3 range-offset convention
    ``phase = 4*pi*range_spacing/wavelength*(secondary-reference)``.

    Parameters
    ----------
    secondary : numpy.ndarray
        Aligned secondary SLC tile.
    secondary_range_index : numpy.ndarray
        Secondary fractional range coordinates for each reference pixel.
    primary_col_origin : int
        Full-grid range origin of the reference tile.
    range_spacing_m, wavelength_m : float
        Radar range spacing and wavelength.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Flattened secondary tile and applied phase screen in radians.

    """
    if secondary.ndim != 2 or secondary_range_index.shape != secondary.shape:
        reject_invalid_state("range-offset flatten inputs must be matching 2-D tiles")
    primary_range = (
        primary_col_origin + np.arange(secondary.shape[1], dtype=np.float64)[None, :]
    )
    range_offset = np.asarray(secondary_range_index, dtype=np.float64) - primary_range
    phase = (4.0 * np.pi * range_spacing_m / wavelength_m) * range_offset
    flattened = np.asarray(secondary, dtype=np.complex64) * np.exp(1j * phase)
    return flattened.astype(np.complex64, copy=False), phase.astype(np.float32)


def _sanitize_phase_screen(phase: np.ndarray) -> np.ndarray:
    """Replace invalid phase-screen lanes with neutral finite phase.

    Parameters
    ----------
    phase : numpy.ndarray
        Phase-screen values in radians.

    Returns
    -------
    numpy.ndarray
        Finite ``float32`` phase-screen values, with invalid lanes set to
        zero radians.

    """
    return np.where(np.isfinite(phase), phase, 0.0).astype(np.float32, copy=False)


def _window(value: object, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Validate a row/column crop against one source shape."""
    if value is None:
        mission = "NISAR RSLC Stack"
        capability = "scene-production"
        reason = (
            "bounded scene production requires an explicit nisar_window; "
            "full-scene promotion is unsupported"
        )
        raise UnsupportedStackCapabilityError(
            mission,
            capability,
            reason,
        )
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        reject_invalid_state(
            "NISAR provider window must be (row_start, row_stop, col_start, col_stop)"
        )
    try:
        row_start, row_stop, col_start, col_stop = (int(item) for item in value)
    except (TypeError, ValueError) as error:
        reject_invalid_state(f"NISAR provider window is not integral: {error}")
    if not (
        0 <= row_start < row_stop <= shape[0] and 0 <= col_start < col_stop <= shape[1]
    ):
        reject_invalid_state(
            f"NISAR provider window {(row_start, row_stop, col_start, col_stop)} "
            f"is outside source shape {shape}"
        )
    return row_start, row_stop, col_start, col_stop


def _tile_shape(value: object) -> tuple[int, int]:
    """Validate the full-scene tile shape."""
    if value is None:
        return (2048, 2048)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        reject_invalid_state("NISAR tile shape must be (rows, cols)")
    try:
        rows, cols = (int(item) for item in value)
    except (TypeError, ValueError) as error:
        reject_invalid_state(f"NISAR tile shape is not integral: {error}")
    if rows < 1 or cols < 1:
        reject_invalid_state("NISAR tile dimensions must be positive")
    return rows, cols


def _iter_tiles(
    bounds: tuple[int, int, int, int],
    tile_shape: tuple[int, int],
) -> Iterator[tuple[int, int, int, int]]:
    """Yield deterministic row-major tiles within ``bounds``."""
    row_start, row_stop, col_start, col_stop = bounds
    tile_rows, tile_cols = tile_shape
    for tile_row_start in range(row_start, row_stop, tile_rows):
        for tile_col_start in range(col_start, col_stop, tile_cols):
            yield (
                tile_row_start,
                min(tile_row_start + tile_rows, row_stop),
                tile_col_start,
                min(tile_col_start + tile_cols, col_stop),
            )


def _full_primary_bounds(
    primary_product: SLCProduct,
    secondary_product: SLCProduct,
    *,
    device: str,
    dem: DEM | None,
    height_m: float | None,
) -> tuple[int, int, int, int]:
    """Return the full Primary grid; invalid Secondary coverage is masked."""
    if not isinstance(primary_product.grid, RadarGrid) or not isinstance(
        secondary_product.grid, RadarGrid
    ):
        reject_invalid_state("NISAR full-scene overlap requires radar products")
    _ = device, dem, height_m
    return (0, primary_product.grid.shape[0], 0, primary_product.grid.shape[1])


def _full_stack_reference_bounds(
    products: Mapping[str, SLCProduct],
    reference: str | None = None,
    *,
    device: str,
    dem: DEM | None,
    height_m: float | None,
) -> tuple[int, int, int, int]:
    """Intersect every acquisition onto one stable Reference radar grid."""
    primary_product = products[reference]
    if not isinstance(primary_product.grid, RadarGrid):
        reject_invalid_state("NISAR full Stack Reference must use a radar grid")
    _ = products, device, dem, height_m
    return (0, primary_product.grid.shape[0], 0, primary_product.grid.shape[1])


def _numpy_geometry(value: object, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Move one geometry field to a contiguous NumPy array."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _dense_secondary_mapping(
    primary_product: SLCProduct,
    secondary_product: SLCProduct,
    bounds: tuple[int, int, int, int],
    *,
    device: str,
    dem: DEM,
    lanczos_a: int = 4,
) -> _DenseRadarMapping:
    """Map every Reference tile pixel into the secondary radar grid."""
    from faninsar.processing.geometry import RadarGeometryModel
    from faninsar.processing.geometry.prepare_production import (
        run_geo2rdr,
        run_rdr2geo,
    )

    if not isinstance(primary_product.grid, RadarGrid) or not isinstance(
        secondary_product.grid, RadarGrid
    ):
        reject_invalid_state("NISAR dense coregistration requires radar products")
    row_start, row_stop, col_start, col_stop = bounds
    azimuth, range_index = np.meshgrid(
        np.arange(row_start, row_stop, dtype=np.float64),
        np.arange(col_start, col_stop, dtype=np.float64),
        indexing="ij",
    )
    primary_model = RadarGeometryModel.from_radar_grid(
        primary_product.grid, primary_product.orbit
    )
    secondary_model = RadarGeometryModel.from_radar_grid(
        secondary_product.grid, secondary_product.orbit
    )
    tile_shape = azimuth.shape
    ground = run_rdr2geo(
        primary_model,
        azimuth.reshape(-1),
        range_index.reshape(-1),
        dem,
        device=device,
        doppler_tol_hz=0.1,
    )
    ground_valid = _numpy_geometry(ground.converged, dtype=np.dtype(bool)).reshape(
        tile_shape
    )
    latitude = _numpy_geometry(ground.latitude_deg, dtype=np.dtype(np.float64)).reshape(
        tile_shape
    )
    longitude = _numpy_geometry(
        ground.longitude_deg, dtype=np.dtype(np.float64)
    ).reshape(tile_shape)
    height = _numpy_geometry(ground.height_m, dtype=np.dtype(np.float64)).reshape(
        tile_shape
    )
    ground_valid &= np.isfinite(latitude) & np.isfinite(longitude) & np.isfinite(height)
    mapped = run_geo2rdr(
        secondary_model,
        np.where(ground_valid, latitude, 0.0).reshape(-1),
        np.where(ground_valid, longitude, 0.0).reshape(-1),
        np.where(ground_valid, height, 0.0).reshape(-1),
        device=device,
        doppler_tol_hz=0.1,
    )
    secondary_azimuth = _numpy_geometry(
        mapped.azimuth_index, dtype=np.dtype(np.float64)
    ).reshape(tile_shape)
    secondary_range = _numpy_geometry(
        mapped.range_index, dtype=np.dtype(np.float64)
    ).reshape(tile_shape)
    valid = (
        ground_valid
        & _numpy_geometry(mapped.converged, dtype=np.dtype(bool)).reshape(tile_shape)
        & np.isfinite(secondary_azimuth)
        & np.isfinite(secondary_range)
        & (secondary_azimuth >= 0.0)
        & (secondary_azimuth <= secondary_product.grid.shape[0] - 1.0)
        & (secondary_range >= 0.0)
        & (secondary_range <= secondary_product.grid.shape[1] - 1.0)
    )
    if not np.any(valid):
        finite_latitude = latitude[np.isfinite(latitude)]
        finite_longitude = longitude[np.isfinite(longitude)]
        finite_azimuth = secondary_azimuth[np.isfinite(secondary_azimuth)]
        finite_range = secondary_range[np.isfinite(secondary_range)]
        logger.warning(
            "NISAR dense tile has no secondary coverage: "
            "primary_bounds=%s ground_valid_count=%d "
            "mapped_converged_count=%d "
            "latitude_range=%s longitude_range=%s "
            "mapped_azimuth_range=%s mapped_range_range=%s "
            "secondary_grid_shape=%s",
            bounds,
            int(np.count_nonzero(ground_valid)),
            int(
                np.count_nonzero(
                    _numpy_geometry(mapped.converged, dtype=np.dtype(bool))
                )
            ),
            None
            if finite_latitude.size == 0
            else (float(np.min(finite_latitude)), float(np.max(finite_latitude))),
            None
            if finite_longitude.size == 0
            else (
                float(np.min(finite_longitude)),
                float(np.max(finite_longitude)),
            ),
            None
            if finite_azimuth.size == 0
            else (float(np.min(finite_azimuth)), float(np.max(finite_azimuth))),
            None
            if finite_range.size == 0
            else (float(np.min(finite_range)), float(np.max(finite_range))),
            secondary_product.grid.shape,
        )
        return _DenseRadarMapping(
            azimuth=np.zeros_like(secondary_azimuth),
            range_index=np.zeros_like(secondary_range),
            valid=valid,
            source_bounds=None,
        )
    halo = int(lanczos_a) + 1
    source_row_start = max(0, int(np.floor(np.min(secondary_azimuth[valid]))) - halo)
    source_row_stop = min(
        secondary_product.grid.shape[0],
        int(np.ceil(np.max(secondary_azimuth[valid]))) + halo + 1,
    )
    source_col_start = max(0, int(np.floor(np.min(secondary_range[valid]))) - halo)
    source_col_stop = min(
        secondary_product.grid.shape[1],
        int(np.ceil(np.max(secondary_range[valid]))) + halo + 1,
    )
    return _DenseRadarMapping(
        azimuth=secondary_azimuth - source_row_start,
        range_index=secondary_range - source_col_start,
        valid=valid,
        source_bounds=(
            source_row_start,
            source_row_stop,
            source_col_start,
            source_col_stop,
        ),
    )


def _lanczos_source_coverage(
    samples: np.ndarray,
    azimuth: np.ndarray,
    range_index: np.ndarray,
    *,
    lanczos_a: int = 4,
) -> np.ndarray:
    """Return destinations whose complete Lanczos source support is valid."""
    from scipy.ndimage import map_coordinates, minimum_filter

    source_valid = (
        np.isfinite(samples.real) & np.isfinite(samples.imag) & (np.abs(samples) > 0.0)
    )
    support_width = 2 * int(lanczos_a) + 1
    full_support = minimum_filter(
        source_valid.astype(np.uint8),
        size=support_width,
        mode="constant",
        cval=0,
    )
    sampled = map_coordinates(
        full_support,
        [azimuth, range_index],
        order=0,
        mode="constant",
        cval=0,
    )
    return np.asarray(sampled, dtype=bool)


def _geocode_aligned_radar_tile(
    primary_product: SLCProduct,
    reference: np.ndarray,
    secondary: np.ndarray,
    radar_bounds: tuple[int, int, int, int],
    target: GeoGrid,
    *,
    device: str,
    dem: DEM,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-geocode aligned complex tiles through Geo2Rdr + Lanczos."""
    from faninsar.processing.geometry import RadarGeometryModel
    from faninsar.processing.merge.grid import GeoGridSpec
    from faninsar.processing.pipeline.geo_lut import build_geo2rdr_lut
    from faninsar.processing.pipeline.geo_resample import (
        resample_complex_at_coordinates,
    )

    if not isinstance(primary_product.grid, RadarGrid):
        reject_invalid_state("NISAR Geo coregistration requires a radar product")
    a, b, c, d, e, f = target.transform
    if b != 0.0 or d != 0.0:
        reject_invalid_state("NISAR Geo coregistration requires a north-up grid")
    grid = GeoGridSpec(
        crs=target.crs,
        transform=(c, a, 0.0, f, 0.0, e),
        width=target.shape[1],
        height=target.shape[0],
        resolution_m=(abs(a), abs(e)),
    )
    geometry = RadarGeometryModel.from_radar_grid(
        primary_product.grid,
        primary_product.orbit,
    )
    lut = build_geo2rdr_lut(
        geometry=geometry,
        grid=grid,
        full_radar_shape=primary_product.grid.shape,
        dem=dem,
        device=device,
    )
    row_start, _row_stop, col_start, _col_stop = radar_bounds
    local_azimuth = np.asarray(lut.az_full, dtype=np.float64) - row_start
    local_range = np.asarray(lut.rg_full, dtype=np.float64) - col_start
    valid = np.asarray(lut.valid, dtype=bool)
    primary_geo, primary_valid = resample_complex_at_coordinates(
        reference,
        local_azimuth,
        local_range,
        valid=valid,
        device=device,
    )
    secondary_geo, secondary_valid = resample_complex_at_coordinates(
        secondary,
        local_azimuth,
        local_range,
        valid=valid,
        device=device,
    )
    primary_valid &= _lanczos_source_coverage(
        reference,
        local_azimuth,
        local_range,
    )
    secondary_valid &= _lanczos_source_coverage(
        secondary,
        local_azimuth,
        local_range,
    )
    primary_geo[~primary_valid] = np.complex64(np.nan + 1j * np.nan)
    secondary_geo[~secondary_valid] = np.complex64(np.nan + 1j * np.nan)
    return primary_geo, secondary_geo


def _geo_tile_for_radar_crop(
    product: SLCProduct,
    bounds: tuple[int, int, int, int],
    target: GeoGrid,
    *,
    device: str,
    dem: DEM,
) -> tuple[GeoGrid, int, int]:
    """Map radar-crop control points to a bounded target-grid window."""
    from faninsar.processing.geometry import RadarGeometryModel
    from faninsar.processing.geometry.prepare_production import run_rdr2geo

    if not isinstance(product.grid, RadarGrid):
        reject_invalid_state("NISAR Geo tiling requires a radar product")
    row_start, row_stop, col_start, col_stop = bounds
    azimuth_axis = np.linspace(row_start, row_stop - 1, num=3, dtype=np.float64)
    range_axis = np.linspace(col_start, col_stop - 1, num=3, dtype=np.float64)
    azimuth_grid, range_grid = np.meshgrid(
        azimuth_axis,
        range_axis,
        indexing="ij",
    )
    model = RadarGeometryModel.from_radar_grid(product.grid, product.orbit)
    ground = run_rdr2geo(
        model,
        azimuth_grid.reshape(-1),
        range_grid.reshape(-1),
        dem,
        device=device,
        doppler_tol_hz=0.1,
    )
    valid = np.asarray(ground.converged, dtype=bool)
    latitude = np.asarray(ground.latitude_deg, dtype=np.float64)
    longitude = np.asarray(ground.longitude_deg, dtype=np.float64)
    valid &= np.isfinite(latitude) & np.isfinite(longitude)
    if not np.any(valid):
        reject_invalid_state("NISAR radar tile has no converged geographic corners")
    a, b, c, d, e, f = target.transform
    if b != 0.0 or d != 0.0:
        reject_invalid_state("chunked NISAR Geo production requires a north-up grid")
    normalized_crs = target.crs.upper().replace(" ", "")
    if normalized_crs in {"EPSG:4326", "OGC:CRS84", "CRS84"}:
        x_coordinates = longitude[valid]
        y_coordinates = latitude[valid]
    else:
        try:
            from pyproj import Transformer

            transformer = Transformer.from_crs(
                "EPSG:4326",
                target.crs,
                always_xy=True,
            )
            x_coordinates, y_coordinates = transformer.transform(
                longitude[valid],
                latitude[valid],
            )
        except Exception as error:
            logger.exception("NISAR tile coordinates cannot be projected")
            reject_invalid_state(
                f"NISAR geo_grid CRS cannot project WGS84 coordinates: {error}"
            )
        x_coordinates = np.asarray(x_coordinates, dtype=np.float64)
        y_coordinates = np.asarray(y_coordinates, dtype=np.float64)
    finite = np.isfinite(x_coordinates) & np.isfinite(y_coordinates)
    if not np.any(finite):
        reject_invalid_state("NISAR radar tile has no finite projected coordinates")
    columns = (x_coordinates[finite] - c) / a
    rows = (y_coordinates[finite] - f) / e
    # Two pixels retain the nearest-neighbour support at tile boundaries.
    target_row_start = max(0, int(np.floor(np.min(rows))) - 2)
    target_row_stop = min(target.shape[0], int(np.ceil(np.max(rows))) + 3)
    target_col_start = max(0, int(np.floor(np.min(columns))) - 2)
    target_col_stop = min(target.shape[1], int(np.ceil(np.max(columns))) + 3)
    if target_row_start >= target_row_stop or target_col_start >= target_col_stop:
        reject_invalid_state("NISAR radar tile lies outside the configured geo_grid")
    local = GeoGrid(
        shape=(
            target_row_stop - target_row_start,
            target_col_stop - target_col_start,
        ),
        crs=target.crs,
        transform=(
            a,
            b,
            c + target_col_start * a + target_row_start * b,
            d,
            e,
            f + target_col_start * d + target_row_start * e,
        ),
    )
    return local, target_row_start, target_col_start


def _scene_tile_exists(
    root: Path,
    *,
    tag: str,
    date_id: str,
    reference_id: str,
    domain: str,
    grid_shape: tuple[int, int],
    grid_identity: str,
    row_origin: int,
    col_origin: int,
    shape: tuple[int, int],
    resume_identity: str,
) -> bool:
    """Return whether one prior tile is complete and matches this request."""
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return False
    store = CoregisteredSceneStore.open(root)
    if (
        store.date_id != date_id
        or store.reference_id != reference_id
        or store.domain != domain
        or store.grid_shape != grid_shape
        or store.grid_identity != grid_identity
    ):
        reject_invalid_state("persisted NISAR tile store does not match this request")
    unit = store.unit_map().get(tag)
    if unit is None:
        return False
    if (
        unit.row_origin != row_origin
        or unit.col_origin != col_origin
        or unit.shape != shape
    ):
        reject_invalid_state(f"persisted NISAR tile {tag!r} has different placement")
    if (unit.phase_state or {}).get("resume_identity") != resume_identity:
        reject_invalid_state(f"persisted NISAR tile {tag!r} has a stale identity")
    store.read(tag)
    return True


def _tile_resume_identity(payload: Mapping[str, object]) -> str:
    """Hash the scientific and tiling inputs for one resumable tile."""
    try:
        encoded = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        reject_invalid_state(f"NISAR tile identity is not canonical: {error}")
    return hashlib.sha256(encoded).hexdigest()


def _radar_crop(
    product: SLCProduct,
    array: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> RadarSLC:
    """Attach one native crop to a matching radar product contract."""
    if not isinstance(product.grid, RadarGrid):
        reject_invalid_state("NISAR RSLC provider requires radar-grid products")
    row_start, row_stop, col_start, col_stop = bounds
    grid = replace(
        product.grid,
        shape=(row_stop - row_start, col_stop - col_start),
        starting_slant_range_m=(
            product.grid.starting_slant_range_m
            + col_start * product.grid.range_spacing_m
        ),
        sensing_start=product.grid.sensing_start
        + timedelta(seconds=row_start * product.grid.azimuth_time_interval_s),
    )
    descriptor = replace(product.samples, shape=grid.shape)
    return RadarSLC(
        product=replace(product, grid=grid, samples=descriptor),
        samples=np.asarray(array, dtype=np.complex64),
    )


def _shared_radar_window(
    reference: RadarGrid,
    secondary: RadarGrid,
    primary_bounds: tuple[int, int, int, int],
    *,
    primary_product: SLCProduct | None = None,
    secondary_product: SLCProduct | None = None,
    device: str = "cpu",
    dem: DEM | None = None,
    height_m: float | None = None,
) -> tuple[int, int, int, int]:
    """Map a reference crop onto the secondary radar grid physically.

    NISAR acquisitions can have different zero-Doppler starts and slant-range
    origins.  A common pixel index is therefore not a common physical target;
    this mapping keeps the target range/time and only then selects secondary
    indices.
    """
    row_start, row_stop, col_start, col_stop = primary_bounds
    target_time = reference.sensing_start + timedelta(
        seconds=row_start * reference.azimuth_time_interval_s
    )
    target_range = (
        reference.starting_slant_range_m + col_start * reference.range_spacing_m
    )
    secondary_row_start = round(
        (target_time - secondary.sensing_start).total_seconds()
        / secondary.azimuth_time_interval_s
    )
    secondary_col_start = round(
        (target_range - secondary.starting_slant_range_m) / secondary.range_spacing_m
    )
    rows = row_stop - row_start
    cols = col_stop - col_start
    secondary_bounds = (
        secondary_row_start,
        secondary_row_start + rows,
        secondary_col_start,
        secondary_col_start + cols,
    )
    if primary_product is None or secondary_product is None:
        # Explicit degraded mode for callers that only have grid metadata.
        # A normalized NISAR pair always supplies products and therefore uses
        # the physical geometry seam below, even when this seed is in bounds.
        _window(secondary_bounds, secondary.shape)
        return secondary_bounds
    return _geometry_shared_radar_window(
        primary_product,
        secondary_product,
        primary_bounds,
        device=device,
        dem=dem,
        height_m=height_m,
    )


def _geometry_shared_radar_window(
    primary_product: SLCProduct,
    secondary_product: SLCProduct,
    primary_bounds: tuple[int, int, int, int],
    *,
    device: str,
    dem: DEM | None = None,
    height_m: float | None = None,
) -> tuple[int, int, int, int]:
    """Map a bounded crop through the shared Radar→Geo→Radar geometry seam."""
    from faninsar.processing.dem import ConstantDEM
    from faninsar.processing.geometry import RadarGeometryModel
    from faninsar.processing.geometry.prepare_production import (
        run_geo2rdr,
        run_rdr2geo,
    )

    if not isinstance(primary_product.grid, RadarGrid) or not isinstance(
        secondary_product.grid, RadarGrid
    ):
        reject_invalid_state("NISAR geometry crop mapping requires radar products")
    primary_bounds = _window(primary_bounds, primary_product.grid.shape)
    if dem is None and height_m is None:
        reject_invalid_state(
            "NISAR geometry crop mapping requires an explicit DEM or height"
        )
    if dem is not None and height_m is not None:
        reject_invalid_state(
            "NISAR geometry crop mapping cannot combine DEM and height inputs"
        )
    if height_m is not None:
        try:
            resolved_height = float(height_m)
        except (TypeError, ValueError) as error:
            reject_invalid_state(f"NISAR geometry mapping height is invalid: {error}")
        if not np.isfinite(resolved_height):
            reject_invalid_state("NISAR geometry mapping height must be finite")
        dem = ConstantDEM(resolved_height)
    row_start, row_stop, col_start, col_stop = primary_bounds
    center_row = np.array(
        [(row_start + row_stop - 1) / 2.0],
        dtype=np.float64,
    )
    center_col = np.array(
        [(col_start + col_stop - 1) / 2.0],
        dtype=np.float64,
    )
    primary_model = RadarGeometryModel.from_radar_grid(
        primary_product.grid,
        primary_product.orbit,
    )
    secondary_model = RadarGeometryModel.from_radar_grid(
        secondary_product.grid,
        secondary_product.orbit,
    )
    try:
        ground = run_rdr2geo(
            primary_model,
            center_row,
            center_col,
            dem,
            device=device,
            doppler_tol_hz=0.1,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        logger.exception("NISAR reference crop geometry mapping failed")
        reject_invalid_state(f"NISAR reference crop rdr2geo failed: {error}")
    converged = getattr(ground, "converged", None)
    converged_values = (
        np.asarray(converged, dtype=bool).reshape(-1)
        if converged is not None
        else np.array([], dtype=bool)
    )
    if not converged_values.size or not bool(converged_values[0]):
        reject_invalid_state("NISAR reference crop target did not converge in rdr2geo")
    try:
        ground_latitude = _contiguous_geometry_input(ground.latitude_deg)
        ground_longitude = _contiguous_geometry_input(ground.longitude_deg)
        ground_height = _contiguous_geometry_input(ground.height_m)
    except AttributeError as error:
        reject_invalid_state(
            f"NISAR reference crop rdr2geo result is missing coordinates: {error}"
        )
    try:
        mapped = run_geo2rdr(
            secondary_model,
            ground_latitude,
            ground_longitude,
            ground_height,
            device=device,
            doppler_tol_hz=0.1,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        logger.exception("NISAR secondary crop geometry mapping failed")
        reject_invalid_state(f"NISAR secondary crop geo2rdr failed: {error}")
    mapped_converged = getattr(mapped, "converged", None)
    mapped_converged_values = (
        np.asarray(mapped_converged, dtype=bool).reshape(-1)
        if mapped_converged is not None
        else np.array([], dtype=bool)
    )
    if not mapped_converged_values.size or not bool(mapped_converged_values[0]):
        reject_invalid_state("NISAR shared crop target did not converge in geo2rdr")
    mapped_azimuth = np.asarray(
        getattr(mapped, "azimuth_index", np.array([])), dtype=np.float64
    ).reshape(-1)
    mapped_range = np.asarray(
        getattr(mapped, "range_index", np.array([])), dtype=np.float64
    ).reshape(-1)
    if (
        mapped_azimuth.size == 0
        or mapped_range.size == 0
        or not np.isfinite(mapped_azimuth[0])
        or not np.isfinite(mapped_range[0])
    ):
        reject_invalid_state(
            "NISAR shared crop target returned non-finite radar indices in geo2rdr"
        )
    rows = row_stop - row_start
    cols = col_stop - col_start
    secondary_row_start = round(float(mapped_azimuth[0])) - rows // 2
    secondary_col_start = round(float(mapped_range[0])) - cols // 2
    secondary_bounds = (
        secondary_row_start,
        secondary_row_start + rows,
        secondary_col_start,
        secondary_col_start + cols,
    )
    try:
        return _window(secondary_bounds, secondary_product.grid.shape)
    except InvalidProcessingStateError as error:
        reject_invalid_state(
            "NISAR mapped secondary radar crop lies outside the source grid; "
            f"full-window rejection: {error}"
        )


def _geo_target(value: object) -> GeoGrid:
    """Normalize a public ``GeoGrid`` or ``GeoGridSpec`` target."""
    if isinstance(value, GeoGrid):
        return value
    if value is None:
        reject_invalid_state(
            "NISAR geographic processing requires an explicit geo_grid target"
        )
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = (getattr(value, "height", 0), getattr(value, "width", 0))
    crs = getattr(value, "crs", None)
    transform = getattr(value, "transform", None)
    if not crs or transform is None:
        reject_invalid_state("NISAR geo_grid must expose crs, transform, and shape")
    try:
        # Canonical ``GridSpec`` stores an affine in GDAL order while the
        # processing ``GeoGrid`` contract stores ``(dx, 0, x0, 0, dy, y0)``.
        from affine import Affine

        affine = Affine(*tuple(transform))
        x0, dx, y0, dy = affine.c, affine.a, affine.f, affine.e
        return GeoGrid(
            shape=(int(shape[0]), int(shape[1])),
            crs=str(crs),
            transform=(float(dx), 0.0, float(x0), 0.0, float(dy), float(y0)),
        )
    except (TypeError, ValueError, IndexError) as error:
        reject_invalid_state(f"invalid NISAR geo_grid target: {error}")


def _provider_window(options: Mapping[str, Any], configured: object) -> object:
    """Resolve the explicit NISAR crop option without changing Stack options."""
    for key in ("nisar_window", "rslc_window", "window"):
        if key in options:
            return options[key]
    return configured


def _geometry_inputs(
    options: Mapping[str, Any],
    *,
    configured_dem: DEM | None,
    configured_height: float | None,
) -> tuple[DEM | None, float | None]:
    """Resolve one DEM source by explicit-call then configured precedence."""
    option_dem = options.get("dem")
    if option_dem is not None:
        return option_dem, None
    option_height = options.get("height_m", options.get("height"))
    if option_height is not None:
        return None, option_height
    if configured_dem is not None:
        return configured_dem, None
    return None, configured_height


def make_nisar_scene_provider(
    *,
    sensor: Any,
    handles: Mapping[Path, Any],
    products: Mapping[str, SLCProduct],
    lineage: Mapping[str, str],
    reference: str,
    channel: tuple[str, str],
    configured_window: object = None,
    configured_tile_shape: object = None,
    configured_dem: DEM | None = None,
    configured_height: float | None = None,
    flatten_stage: str = "coregistration",
    admission_lineage: Mapping[str, Mapping[str, object]] | None = None,
    **legacy: object,
) -> SceneProductionCallback:
    """Build a callback that publishes one bounded or full NISAR pair scene.

    Parameters
    ----------
    sensor : object
        Normalized sensor exposing ``read_slc_window``.
    handles : mapping of pathlib.Path to object
        Lazy reader handles opened during stack construction.
    products, lineage : mappings
        Date-keyed normalized products and source paths.
    reference : str
        Stack Reference date used in scene manifests.
    channel : tuple of str
        Admitted ``(frequency, polarization)`` pair.
    configured_window : sequence of int, optional
        Default crop as ``(row_start, row_stop, col_start, col_stop)``.
        Omitting it enables chunked full-scene production.
    configured_tile_shape : sequence of int, optional
        Full-scene tile dimensions. Defaults to ``(2048, 2048)``.
    configured_dem : DEM, optional
        Callback-level DEM used when a scene-production call omits ``dem``.
    configured_height : float, optional
        Callback-level constant height used when a scene-production call omits
        both ``dem`` and ``height``.
    flatten_stage : {"coregistration", "interferogram"}, optional
        Stage at which the NISAR range-offset phase screen is applied.
    admission_lineage : mapping, optional
        Date-keyed trusted pre-open metadata.  It is copied into the scene
        manifest so source identity and policy survive stack publication.
    **legacy : object
        Removed keyword arguments. The old ``master`` keyword is rejected with
        a migration error.

    Returns
    -------
    SceneProductionCallback
        Provider callback accepted by :class:`StackSceneProvider`.

    """
    if "master" in legacy:
        reject_pair_configuration(
            "make_nisar_scene_provider no longer accepts 'master'; use 'reference'"
        )
    if legacy:
        reject_invalid_state(f"unsupported NISAR provider options: {sorted(legacy)}")
    if reference is None:
        reject_invalid_state("NISAR provider Reference date is required")
    if flatten_stage not in {"coregistration", "interferogram"}:
        reject_invalid_state(
            "NISAR flatten_stage must be 'coregistration' or 'interferogram'"
        )

    path_dates = {Path(path): date_id for date_id, path in lineage.items()}
    admission_lineage = dict(admission_lineage or {})
    admitted_sources = {
        date_id: _source_digest(Path(path)) for date_id, path in lineage.items()
    }

    def produce_pair(
        primary_path: SourceHandle,
        secondary_path: SourceHandle,
        *,
        output_dir: Path,
        options: Mapping[str, Any],
    ) -> NisarPairState:
        def one_path(
            value: SourceHandle | Path | tuple[Path, ...],
            label: str,
        ) -> Path:
            """Resolve one source path from a logical acquisition payload."""
            sources = SourceHandle._from_source(value)._resolve()
            if len(sources) != 1:
                reject_invalid_state(
                    f"NISAR provider requires one RSLC path for {label}"
                )
            return sources[0]

        primary_source = one_path(primary_path, "primary")
        secondary_source = one_path(secondary_path, "secondary")
        primary_date = path_dates.get(primary_source)
        secondary_date = path_dates.get(secondary_source)
        if primary_date is None or secondary_date is None:
            reject_invalid_state("NISAR provider received an unadmitted source path")
        primary_product = products[primary_date]
        secondary_product = products[secondary_date]
        requested_flatten_stage = str(options.get("flatten_stage", flatten_stage))
        if requested_flatten_stage != flatten_stage:
            reject_invalid_state(
                "NISAR provider flatten_stage differs from its admitted configuration"
            )
        mapping_dem, mapping_height = _geometry_inputs(
            options,
            configured_dem=configured_dem,
            configured_height=configured_height,
        )
        for date_id, source_path in (
            (primary_date, primary_source),
            (secondary_date, secondary_source),
        ):
            current = _source_digest(Path(source_path))
            if current != admitted_sources[date_id]:
                reject_invalid_state(
                    "NISAR RSLC source path or content changed after admission"
                )
        if not isinstance(primary_product.grid, RadarGrid) or not isinstance(
            secondary_product.grid, RadarGrid
        ):
            reject_invalid_state("NISAR provider requires normalized radar products")
        if primary_product.grid.wavelength_m != secondary_product.grid.wavelength_m:
            reject_invalid_state("NISAR pair radar wavelengths do not match")
        domain = str(options.get("coregistration_grid", "radar")).lower()
        if mapping_dem is None and mapping_height is None:
            reject_invalid_state(
                "NISAR geometry crop mapping requires an explicit DEM or height "
                "in provider options"
            )
        if mapping_dem is not None and mapping_height is not None:
            reject_invalid_state(
                "NISAR geometry crop mapping cannot combine DEM and height inputs"
            )
        if mapping_dem is None:
            from faninsar.processing.dem import ConstantDEM

            mapping_dem = ConstantDEM(float(mapping_height))
        dem_identity = _dem_identity(mapping_dem)
        if domain not in {"radar", "geo"}:
            reject_invalid_state(
                f"NISAR provider does not support coordinate domain {domain!r}"
            )
        device = str(options.get("device", "cpu"))
        requested_window = _provider_window(options, configured_window)
        full_scene = requested_window is None
        if full_scene:
            resolved_tile_shape = _tile_shape(configured_tile_shape)
            if primary_date == reference:
                bounds = _full_stack_reference_bounds(
                    products,
                    reference,
                    device=device,
                    dem=mapping_dem,
                    height_m=mapping_height,
                )
            else:
                bounds = _full_primary_bounds(
                    primary_product,
                    secondary_product,
                    device=device,
                    dem=mapping_dem,
                    height_m=mapping_height,
                )
            tiles = tuple(_iter_tiles(bounds, resolved_tile_shape))
        else:
            resolved_tile_shape = None
            source_shape = (
                min(primary_product.grid.shape[0], secondary_product.grid.shape[0]),
                min(primary_product.grid.shape[1], secondary_product.grid.shape[1]),
            )
            bounds = _window(requested_window, source_shape)
            tiles = (bounds,)
        if domain == "radar":
            grid_shape = (
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
            )
            grid_identity = scene_grid_identity("radar", grid_shape)
            target = None
        else:
            target = _geo_target(options.get("geo_grid"))
            grid_shape = target.shape
            grid_identity = scene_grid_identity(
                "geo",
                grid_shape,
                {"crs": target.crs, "transform": list(target.transform)},
            )
        scenes = output_dir / "scenes"
        pair_claimed = np.zeros(grid_shape, dtype=bool) if domain == "geo" else None
        pair_valid_pixels = 0
        for tile_index, tile_bounds in enumerate(tiles):
            dense_mapping: _DenseRadarMapping | None = None
            if full_scene:
                dense_mapping = _dense_secondary_mapping(
                    primary_product,
                    secondary_product,
                    tile_bounds,
                    device=device,
                    dem=mapping_dem,
                )
                secondary_bounds = dense_mapping.source_bounds
            else:
                secondary_bounds = _shared_radar_window(
                    primary_product.grid,
                    secondary_product.grid,
                    tile_bounds,
                    primary_product=primary_product,
                    secondary_product=secondary_product,
                    device=device,
                    dem=mapping_dem,
                    height_m=mapping_height,
                )
            row_start, row_stop, col_start, col_stop = tile_bounds
            if secondary_bounds is None:
                sec_row_start = sec_row_stop = sec_col_start = sec_col_stop = 0
            else:
                sec_row_start, sec_row_stop, sec_col_start, sec_col_stop = (
                    secondary_bounds
                )
            tag = "NISAR_b0" if not full_scene else f"NISAR_b{tile_index:06d}"
            if domain == "radar":
                local_target = None
                row_origin = row_start - bounds[0]
                col_origin = col_start - bounds[2]
                tile_output_shape = (row_stop - row_start, col_stop - col_start)
            else:
                assert target is not None
                if full_scene:
                    local_target, row_origin, col_origin = _geo_tile_for_radar_crop(
                        primary_product,
                        tile_bounds,
                        target,
                        device=device,
                        dem=mapping_dem,
                    )
                else:
                    local_target = target
                    row_origin = 0
                    col_origin = 0
                tile_output_shape = local_target.shape
            resume_payload = {
                "schema": "nisar_scene_tile_v2",
                "primary_date": primary_date,
                "secondary_date": secondary_date,
                "reference_id": reference,
                "domain": domain,
                "tag": tag,
                "tile_bounds": list(tile_bounds),
                "tile_shape": (
                    None if resolved_tile_shape is None else list(resolved_tile_shape)
                ),
                "grid_shape": list(grid_shape),
                "grid_identity": grid_identity,
                "channel": list(channel),
                "dem_identity": dem_identity,
                "device": device,
                "runtime": {
                    "python": ".".join(str(item) for item in sys.version_info[:3]),
                    "numpy": np.__version__,
                },
                "geometry": (
                    "per_pixel_rdr2geo_geo2rdr_lanczos4"
                    if full_scene
                    else "bounded_window"
                ),
                "secondary_source_bounds": (
                    None if secondary_bounds is None else list(secondary_bounds)
                ),
                "range_offset_flatten": "nisar_ellipsoidal_v1",
                "flatten_stage": requested_flatten_stage,
            }
            resume_identity = _tile_resume_identity(resume_payload)
            if _scene_tile_exists(
                scenes,
                tag=tag,
                date_id=secondary_date,
                reference_id=reference,
                domain=domain,
                grid_shape=grid_shape,
                grid_identity=grid_identity,
                row_origin=row_origin,
                col_origin=col_origin,
                shape=tile_output_shape,
                resume_identity=resume_identity,
            ):
                existing = CoregisteredSceneStore.open(scenes)
                existing_reference, existing_secondary, _ = existing.read(tag)
                if domain == "geo":
                    row_slice = slice(row_origin, row_origin + tile_output_shape[0])
                    col_slice = slice(col_origin, col_origin + tile_output_shape[1])
                    assert pair_claimed is not None
                    primary_valid = (
                        np.isfinite(existing_reference.real)
                        & np.isfinite(existing_reference.imag)
                        & (np.abs(existing_reference) > 0.0)
                    )
                    secondary_valid = (
                        np.isfinite(existing_secondary.real)
                        & np.isfinite(existing_secondary.imag)
                        & (np.abs(existing_secondary) > 0.0)
                    )
                    if not np.array_equal(primary_valid, secondary_valid):
                        reject_invalid_state(
                            f"persisted NISAR Geo tile {tag!r} has asymmetric ownership"
                        )
                    pair_claimed[row_slice, col_slice] |= primary_valid
                    pair_valid_pixels += int(np.count_nonzero(primary_valid))
                elif full_scene:
                    pair_valid_pixels += int(
                        np.count_nonzero(
                            np.isfinite(existing_reference.real)
                            & np.isfinite(existing_reference.imag)
                            & np.isfinite(existing_secondary.real)
                            & np.isfinite(existing_secondary.imag)
                            & (np.abs(existing_reference) > 0.0)
                            & (np.abs(existing_secondary) > 0.0)
                        )
                    )
                continue
            primary_window = (
                slice(row_start, row_stop),
                slice(col_start, col_stop),
            )
            try:
                primary_samples = sensor.read_slc_window(
                    handles[primary_source],
                    primary_window,
                    frequency=channel[0],
                    polarization=channel[1],
                )
                if secondary_bounds is not None:
                    secondary_samples = sensor.read_slc_window(
                        handles[secondary_source],
                        (
                            slice(sec_row_start, sec_row_stop),
                            slice(sec_col_start, sec_col_stop),
                        ),
                        frequency=channel[0],
                        polarization=channel[1],
                    )
                else:
                    secondary_samples = None
            except KeyError as error:
                reject_invalid_state(f"NISAR source handle is unavailable: {error}")
            primary_radar = _radar_crop(primary_product, primary_samples, tile_bounds)
            if full_scene:
                assert dense_mapping is not None
                if secondary_samples is None:
                    secondary_array = np.full(
                        primary_radar.samples.shape,
                        np.nan + 1j * np.nan,
                        dtype=np.complex64,
                    )
                    dense_valid = dense_mapping.valid
                else:
                    from faninsar.processing.pipeline.geo_resample import (
                        resample_complex_at_coordinates,
                    )

                    secondary_array, dense_valid = resample_complex_at_coordinates(
                        np.asarray(secondary_samples, dtype=np.complex64),
                        dense_mapping.azimuth,
                        dense_mapping.range_index,
                        valid=dense_mapping.valid,
                        device=device,
                    )
                    dense_valid &= _lanczos_source_coverage(
                        np.asarray(secondary_samples, dtype=np.complex64),
                        dense_mapping.azimuth,
                        dense_mapping.range_index,
                    )
                    secondary_array[~dense_valid] = np.complex64(np.nan + 1j * np.nan)
                dense_valid &= (
                    np.isfinite(primary_radar.samples.real)
                    & np.isfinite(primary_radar.samples.imag)
                    & (np.abs(primary_radar.samples) > 0.0)
                )
                secondary_array[~dense_valid] = np.complex64(np.nan + 1j * np.nan)
                primary_array = np.where(
                    dense_valid,
                    primary_radar.samples,
                    np.complex64(np.nan + 1j * np.nan),
                ).astype(np.complex64, copy=False)
                aligned_product = replace(
                    secondary_product,
                    grid=primary_radar.grid,
                    orbit=primary_product.orbit,
                    samples=replace(
                        secondary_product.samples,
                        shape=primary_radar.grid.shape,
                    ),
                )
                secondary_radar = RadarSLC(
                    product=aligned_product,
                    samples=np.asarray(secondary_array, dtype=np.complex64),
                )
                primary_radar = RadarSLC(
                    product=primary_radar.product,
                    samples=np.asarray(primary_array, dtype=np.complex64),
                )
            else:
                assert secondary_samples is not None
                assert secondary_bounds is not None
                secondary_radar = _radar_crop(
                    secondary_product,
                    secondary_samples,
                    secondary_bounds,
                )
                primary_array = primary_radar.samples
                secondary_array = secondary_radar.samples
            if full_scene:
                assert dense_mapping is not None
                if secondary_bounds is None:
                    # A full-scene tile can lie outside the secondary swath.
                    # Keep the tile in the common manifest as an explicit
                    # all-NaN/no-overlap tile; do not invent a source window or
                    # feed zero-filled samples to the Lanczos resampler.
                    secondary_range_index = np.broadcast_to(
                        float(col_start)
                        + np.arange(secondary_array.shape[1], dtype=np.float64)[
                            None, :
                        ],
                        secondary_array.shape,
                    )
                else:
                    secondary_range_index = np.asarray(
                        dense_mapping.range_index, dtype=np.float64
                    ) + float(sec_col_start)
            else:
                secondary_range_index = np.broadcast_to(
                    float(sec_col_start)
                    + np.arange(secondary_array.shape[1], dtype=np.float64)[None, :],
                    secondary_array.shape,
                )
            raw_secondary_array = np.asarray(secondary_array, dtype=np.complex64)
            flattened_secondary, range_offset_phase = _apply_range_offset_flatten(
                raw_secondary_array,
                secondary_range_index,
                primary_col_origin=col_start,
                range_spacing_m=float(primary_product.grid.range_spacing_m),
                wavelength_m=float(primary_product.grid.wavelength_m),
            )
            if requested_flatten_stage == "interferogram" and domain == "radar":
                # Invalid dense-mapping lanes are represented by non-finite
                # source coordinates.  They remain excluded by the sample
                # mask below, but the persisted screen must still be a
                # finite float32 payload.  Zero is the neutral phase on
                # those lanes and matches the Geo path's policy.
                range_offset_phase = _sanitize_phase_screen(range_offset_phase)
            # Coregistration stores the inverse-conjugate screen on the
            # secondary.  Interferogram-stage flattening stores raw secondary
            # samples and persists the exact screen for the IFG boundary.
            secondary_array = (
                flattened_secondary
                if requested_flatten_stage == "coregistration"
                else raw_secondary_array
            )
            if domain == "radar":
                pass
            else:
                assert local_target is not None
                if full_scene:
                    primary_array, secondary_array = _geocode_aligned_radar_tile(
                        primary_product,
                        np.asarray(primary_array, dtype=np.complex64),
                        np.asarray(secondary_array, dtype=np.complex64),
                        tile_bounds,
                        local_target,
                        device=device,
                        dem=mapping_dem,
                    )
                    if requested_flatten_stage == "interferogram":
                        phase_factor = np.exp(1j * range_offset_phase).astype(
                            np.complex64,
                            copy=False,
                        )
                        _, phase_geo = _geocode_aligned_radar_tile(
                            primary_product,
                            phase_factor,
                            phase_factor,
                            tile_bounds,
                            local_target,
                            device=device,
                            dem=mapping_dem,
                        )
                        range_offset_phase = np.angle(phase_geo).astype(
                            np.float32,
                            copy=False,
                        )
                        range_offset_phase = _sanitize_phase_screen(range_offset_phase)
                else:
                    primary_array = primary_radar.rdr2geo(
                        device=device,
                        dem=mapping_dem,
                        geo_grid=local_target,
                        use_cache=False,
                    ).samples
                    secondary_array = secondary_radar.rdr2geo(
                        device=device,
                        dem=mapping_dem,
                        geo_grid=local_target,
                        use_cache=False,
                    ).samples
                    if requested_flatten_stage == "interferogram":
                        phase_radar = RadarSLC(
                            product=primary_radar.product,
                            samples=np.exp(1j * range_offset_phase).astype(
                                np.complex64,
                                copy=False,
                            ),
                        )
                        phase_geo = phase_radar.rdr2geo(
                            device=device,
                            dem=mapping_dem,
                            geo_grid=local_target,
                            use_cache=False,
                        ).samples
                        range_offset_phase = np.angle(phase_geo).astype(
                            np.float32,
                            copy=False,
                        )
                        range_offset_phase = _sanitize_phase_screen(range_offset_phase)
                assert pair_claimed is not None
                row_slice = slice(row_origin, row_origin + tile_output_shape[0])
                col_slice = slice(col_origin, col_origin + tile_output_shape[1])
                primary_valid = (
                    np.isfinite(primary_array.real)
                    & np.isfinite(primary_array.imag)
                    & (np.abs(primary_array) > 0.0)
                )
                secondary_valid = (
                    np.isfinite(secondary_array.real)
                    & np.isfinite(secondary_array.imag)
                    & (np.abs(secondary_array) > 0.0)
                )
                pair_owner = (
                    primary_valid
                    & secondary_valid
                    & ~pair_claimed[row_slice, col_slice]
                )
                primary_array = np.where(
                    pair_owner,
                    primary_array,
                    np.complex64(np.nan + 1j * np.nan),
                )
                secondary_array = np.where(
                    pair_owner,
                    secondary_array,
                    np.complex64(np.nan + 1j * np.nan),
                )
                pair_claimed[row_slice, col_slice] |= pair_owner
            write_scene_unit(
                scenes,
                date_id=secondary_date,
                reference_id=reference,
                domain=domain,
                tag=tag,
                primary=np.asarray(primary_array, dtype=np.complex64),
                secondary=np.asarray(secondary_array, dtype=np.complex64),
                row_origin=row_origin,
                col_origin=col_origin,
                grid_shape=grid_shape,
                wavelength_m=primary_product.grid.wavelength_m,
                grid_identity=grid_identity,
                scientific_lineage=(
                    {
                        "stage": "nisar_rslc_tile"
                        if full_scene
                        else "nisar_rslc_window",
                        "source": str(primary_source),
                        "source_id": admitted_sources[primary_date][0],
                        "source_digest": admitted_sources[primary_date][1],
                        "admission_policy": admission_lineage.get(primary_date, {}).get(
                            "policy"
                        ),
                        "channel": f"{channel[0]}/{channel[1]}",
                        "lineage": "primary",
                        "dem_identity": dem_identity,
                        "window": f"{row_start}:{row_stop},{col_start}:{col_stop}",
                    },
                    {
                        "stage": "nisar_rslc_tile"
                        if full_scene
                        else "nisar_rslc_window",
                        "source": str(secondary_source),
                        "source_id": admitted_sources[secondary_date][0],
                        "source_digest": admitted_sources[secondary_date][1],
                        "admission_policy": admission_lineage.get(
                            secondary_date, {}
                        ).get("policy"),
                        "channel": f"{channel[0]}/{channel[1]}",
                        "lineage": "secondary",
                        "dem_identity": dem_identity,
                        "range_offset_flatten": "nisar_ellipsoidal_v1",
                        "flatten_stage": requested_flatten_stage,
                        "window": (
                            "no_intersection"
                            if secondary_bounds is None
                            else (
                                f"{sec_row_start}:{sec_row_stop},"
                                f"{sec_col_start}:{sec_col_stop}"
                            )
                        ),
                    },
                ),
                phase_state={
                    "resume_identity": resume_identity,
                    "coverage_policy": (
                        "joint_first_valid_row_major_v1"
                        if domain == "geo"
                        else "nan_mask_v1"
                    ),
                    "geometry_method": resume_payload["geometry"],
                    "range_offset_flatten": "nisar_ellipsoidal_v1",
                    "flatten_stage": requested_flatten_stage,
                    "range_offset_phase_sign": "ifg_exp_minus_j_phase",
                    "range_offset_phase_model": (
                        "phase=4*pi*range_spacing/wavelength*(secondary-reference)"
                    ),
                    "range_offset_phase_array_digest": hashlib.sha256(
                        np.ascontiguousarray(
                            range_offset_phase, dtype=np.float32
                        ).tobytes()
                    ).hexdigest(),
                    "range_offset_phase_rms_rad": float(np.nanstd(range_offset_phase)),
                    "primary_valid_pixels": int(
                        np.sum(
                            np.isfinite(primary_array.real)
                            & np.isfinite(primary_array.imag)
                            & (np.abs(primary_array) > 0.0)
                        )
                    ),
                    "secondary_valid_pixels": int(
                        np.sum(
                            np.isfinite(secondary_array.real)
                            & np.isfinite(secondary_array.imag)
                            & (np.abs(secondary_array) > 0.0)
                        )
                    ),
                    "pair_valid_pixels": int(
                        np.sum(
                            np.isfinite(primary_array.real)
                            & np.isfinite(primary_array.imag)
                            & np.isfinite(secondary_array.real)
                            & np.isfinite(secondary_array.imag)
                            & (np.abs(primary_array) > 0.0)
                            & (np.abs(secondary_array) > 0.0)
                        )
                    ),
                },
                phase_screen=(
                    range_offset_phase
                    if requested_flatten_stage == "interferogram"
                    else None
                ),
                validate_existing_payloads=not full_scene,
            )
            if full_scene:
                pair_valid_pixels += int(
                    np.sum(
                        np.isfinite(primary_array.real)
                        & np.isfinite(primary_array.imag)
                        & np.isfinite(secondary_array.real)
                        & np.isfinite(secondary_array.imag)
                        & (np.abs(primary_array) > 0.0)
                        & (np.abs(secondary_array) > 0.0)
                    )
                )
        if full_scene:
            if domain == "geo":
                has_pair_coverage = pair_claimed is not None and bool(
                    np.any(pair_claimed)
                )
            else:
                has_pair_coverage = pair_valid_pixels > 0
            if not has_pair_coverage:
                reject_invalid_state(
                    "NISAR full-scene pair has no physical secondary overlap; "
                    "all dense radar-to-geo-to-radar lanes were invalid"
                )
        return NisarPairState(
            pair_id=f"{primary_date}_{secondary_date}",
            stage_timings_s={},
            coregistration_timings_s={},
        )

    return produce_pair


__all__ = ["NisarPairState", "make_nisar_scene_provider"]
