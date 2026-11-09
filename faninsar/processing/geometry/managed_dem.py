"""One authoritative WGS84 ellipsoidal DEM for Stack and acquisition."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from affine import Affine

from faninsar.logging import reject, setup_logger
from faninsar.processing.errors import ProcessingContractError
from faninsar.processing.geometry.dem_view import (
    geodetic_crop_grid,
    is_geodetic_crs,
    native_spacing_deg,
)
from faninsar.processing.geometry.grids import GridSpec

if TYPE_CHECKING:
    from faninsar.processing.geometry.api import RasterDEM, SourceDEM, VerticalDatum
    from faninsar.processing.geometry.resources import ResourceBudget

logger = setup_logger(__name__)

INTERPOLATION_MARGIN_M = 2000.0
_METRES_PER_DEGREE = 111_000.0
COVERAGE_TOLERANCE = 1.0e-8
_KERNEL_MARGIN_PX = 4


class DEMCoverageInsufficientError(ProcessingContractError):
    """Raised when a live DEM does not cover the currently required bounds."""

    def __init__(
        self,
        message: str,
        *,
        required_bounds: tuple[float, float, float, float],
        actual_bounds: tuple[float, float, float, float] | None = None,
        path: Path | None = None,
    ) -> None:
        """Record the uncovered request and the raster that failed it."""
        super().__init__(message)
        self.required_bounds = tuple(float(value) for value in required_bounds)
        self.actual_bounds = (
            None
            if actual_bounds is None
            else tuple(float(value) for value in actual_bounds)
        )
        self.path = None if path is None else Path(path)


def interpolation_margin_deg(
    spacing_deg: float, *, margin_px: int = _KERNEL_MARGIN_PX
) -> float:
    """Return the larger of kernel-pixel and 2 km geographic margins."""
    spacing = float(spacing_deg)
    if not np.isfinite(spacing) or spacing <= 0.0:
        message = "DEM spacing must be a finite positive number"
        reject(message, ValueError)
    return max(float(margin_px) * spacing, INTERPOLATION_MARGIN_M / _METRES_PER_DEGREE)


def expand_bounds(
    bounds: tuple[float, float, float, float],
    margin_deg: float,
) -> tuple[float, float, float, float]:
    """Expand WGS84 bounds by ``margin_deg`` on every side."""
    west, south, east, north = (float(value) for value in bounds)
    margin = float(margin_deg)
    return (west - margin, south - margin, east + margin, north + margin)


def union_bounds(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Return the axis-aligned union of two WGS84 envelopes."""
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def bounds_cover(
    actual: tuple[float, float, float, float],
    required: tuple[float, float, float, float],
    *,
    tolerance: float = COVERAGE_TOLERANCE,
) -> bool:
    """Return whether ``actual`` covers ``required`` in the same CRS."""
    return (
        actual[0] <= required[0] + tolerance
        and actual[1] <= required[1] + tolerance
        and actual[2] >= required[2] - tolerance
        and actual[3] >= required[3] - tolerance
    )


def wgs84_bounds(grid: GridSpec) -> tuple[float, float, float, float]:
    """Return ``(west, south, east, north)`` of ``grid`` in EPSG:4326."""
    from faninsar.processing.geometry.dem_view import _geodetic_bounds

    return _geodetic_bounds(grid)


def read_raster_header(path: Path) -> tuple[GridSpec, str]:
    """Read CRS/transform/shape/datum without loading band samples."""
    import rasterio

    with rasterio.open(path) as dataset:
        grid = GridSpec(
            crs=dataset.crs,
            transform=dataset.transform,
            height=dataset.height,
            width=dataset.width,
        )
        datum = dataset.tags().get("dem_vertical_datum") or "ellipsoidal"
    return grid, str(datum)


def _try_read_raster_header(path: Path) -> tuple[GridSpec, str] | None:
    """Return header metadata, or ``None`` when the raster cannot be opened."""
    try:
        return read_raster_header(path)
    except OSError:
        logger.warning("Managed DEM %s is unreadable; rematerializing", path)
        return None


def raster_covers_bounds(
    dem_or_path: RasterDEM | Path,
    required_bounds: tuple[float, float, float, float],
) -> bool:
    """Return whether a raster header covers ``required_bounds`` in WGS84."""
    if isinstance(dem_or_path, Path):
        grid, _datum = read_raster_header(dem_or_path)
    else:
        grid = dem_or_path.grid
    actual = grid.bounds if is_geodetic_crs(grid.crs) else wgs84_bounds(grid)
    return bounds_cover(actual, required_bounds)


def require_dem_coverage(
    dem: RasterDEM,
    required_bounds: tuple[float, float, float, float],
    *,
    path: Path | None = None,
) -> None:
    """Raise :class:`DEMCoverageInsufficientError` when ``dem`` is too small."""
    if raster_covers_bounds(dem, required_bounds):
        return
    actual = (
        dem.grid.bounds if is_geodetic_crs(dem.grid.crs) else wgs84_bounds(dem.grid)
    )
    message = (
        "DEM bounds "
        f"{tuple(float(value) for value in actual)} do not cover the required "
        f"extent {tuple(float(value) for value in required_bounds)}"
    )
    raise DEMCoverageInsufficientError(
        message,
        required_bounds=required_bounds,
        actual_bounds=actual,
        path=path if path is not None else getattr(dem, "path", None),
    )


def is_canonical_managed_dem(grid: GridSpec, datum: str) -> bool:
    """Return whether ``grid`` and ``datum`` already match the managed contract."""
    return is_geodetic_crs(grid.crs) and str(datum).strip().lower() == "ellipsoidal"


def grow_geodetic_grid(
    existing: GridSpec,
    extra_bounds: tuple[float, float, float, float],
) -> GridSpec:
    """Expand ``existing`` onto its pixel lattice to cover ``extra_bounds``."""
    if not is_geodetic_crs(existing.crs):
        message = "managed DEM growth requires an EPSG:4326 raster"
        reject(message, ValueError)
    affine = Affine(*existing.transform)
    px = abs(float(affine.a))
    py = abs(float(affine.e))
    origin_west = float(affine.c)
    origin_north = float(affine.f)
    west, south, east, north = union_bounds(existing.bounds, extra_bounds)
    west_shift = max(0, int(np.ceil((origin_west - west) / px - 1.0e-9)))
    north_shift = max(0, int(np.ceil((north - origin_north) / py - 1.0e-9)))
    new_west = origin_west - west_shift * px
    new_north = origin_north + north_shift * py
    width = max(
        existing.width + west_shift,
        int(np.ceil((east - new_west) / px - 1.0e-9)),
        6,
    )
    height = max(
        existing.height + north_shift,
        int(np.ceil((new_north - south) / py - 1.0e-9)),
        6,
    )
    return GridSpec(
        "EPSG:4326",
        Affine(px, 0.0, new_west, 0.0, -py, new_north),
        height=height,
        width=width,
    )


def geographic_source_grid(
    bounds: tuple[float, float, float, float],
    spacing_deg: float,
) -> GridSpec:
    """Build a native-spacing EPSG:4326 grid covering ``bounds`` plus kernel pad."""
    spacing = float(spacing_deg)
    padded = expand_bounds(bounds, interpolation_margin_deg(spacing))
    west, south, east, north = padded
    width = max(6, int(np.ceil((east - west) / spacing - 1.0e-9)))
    height = max(6, int(np.ceil((north - south) / spacing - 1.0e-9)))
    coverage = GridSpec(
        "EPSG:4326",
        Affine(spacing, 0.0, west, 0.0, -spacing, north),
        height=height,
        width=width,
    )
    return geodetic_crop_grid(coverage, spacing)


def _replace_managed_raster(
    raster: RasterDEM,
    managed_path: Path,
    scratch_dir: Path,
) -> RasterDEM:
    """Write ``raster`` in scratch and replace ``managed_path`` only on success."""
    from faninsar.processing.geometry.api import RasterDEM

    scratch_dir.mkdir(parents=True, exist_ok=True)
    temporary = scratch_dir / f".{managed_path.stem}.expand.tmp{managed_path.suffix}"
    if temporary.exists():
        temporary.unlink()
    try:
        raster.save(temporary)
        managed_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(managed_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return RasterDEM(path=managed_path, vertical_datum=raster.vertical_datum)


def _load_managed_raster(path: Path, vertical_datum: str | None = None) -> RasterDEM:
    from faninsar.processing.geometry.api import RasterDEM

    datum: VerticalDatum = "ellipsoidal"
    if vertical_datum is not None:
        datum = vertical_datum  # type: ignore[assignment]
    else:
        _grid, tagged = read_raster_header(path)
        del _grid
        datum = tagged  # type: ignore[assignment]
    return RasterDEM(path=path, vertical_datum=datum)


def _header_wgs84_bounds(grid: GridSpec) -> tuple[float, float, float, float]:
    """Return WGS84 bounds for a header grid that may still be projected."""
    return grid.bounds if is_geodetic_crs(grid.crs) else wgs84_bounds(grid)


def _normalize_managed_raster(
    *,
    path: Path,
    live: RasterDEM | None,
    header_grid: GridSpec,
    header_datum: str,
    source: SourceDEM | None,
    required_bounds: tuple[float, float, float, float] | None,
    check_coverage: bool,
    cache_dir: Path | None,
    budget: ResourceBudget | None,
    scratch: Path,
    product: str | None,
) -> RasterDEM:
    """Rewrite a noncanonical raster onto EPSG:4326 ellipsoidal ``dem.tif`` once."""
    existing_bounds = _header_wgs84_bounds(header_grid)
    covers = required_bounds is None or bounds_cover(existing_bounds, required_bounds)
    if check_coverage and required_bounds is not None and not covers and source is None:
        message = (
            "DEM does not cover the required bounds and no source recipe can expand it"
        )
        raise DEMCoverageInsufficientError(
            message,
            required_bounds=required_bounds,
            actual_bounds=existing_bounds,
            path=path,
        )

    if source is not None:
        bounds = existing_bounds
        if check_coverage and required_bounds is not None:
            bounds = union_bounds(existing_bounds, required_bounds)
        spacing = native_spacing_deg(product or source.product)
        grid = geographic_source_grid(bounds, spacing)
        if cache_dir is not None and source.cache_dir is None:
            source.cache_dir = Path(cache_dir)
        logger.info("Normalizing managed DEM onto an EPSG:4326 ellipsoidal grid")
        raster = source.to_raster(grid, vertical_datum="ellipsoidal", budget=budget)
        return _replace_managed_raster(raster, path, scratch)

    raster = live if live is not None else _load_managed_raster(path, header_datum)
    if is_geodetic_crs(raster.crs):
        grid = raster.grid
    else:
        grid = geographic_source_grid(existing_bounds, native_spacing_deg(product))
    converted = raster.to_raster(grid, vertical_datum="ellipsoidal", budget=budget)
    return _replace_managed_raster(converted, path, scratch)


def resolve_managed_wgs84_dem(
    *,
    managed_path: Path,
    source: SourceDEM | None = None,
    existing: RasterDEM | None = None,
    required_bounds: tuple[float, float, float, float] | None = None,
    check_coverage: bool = False,
    cache_dir: Path | None = None,
    budget: ResourceBudget | None = None,
    scratch_dir: Path | None = None,
    product: str | None = None,
) -> RasterDEM:
    """Return the canonical ``dem.tif`` raster, creating or growing it as needed.

    Parameters
    ----------
    managed_path : pathlib.Path
        Workspace ``dem.tif`` path. This is the only default managed raster.
    source : SourceDEM, optional
        Recipe used when the raster is missing or must grow.
    existing : RasterDEM, optional
        Live raster already owned by the caller.
    required_bounds : tuple of float, optional
        Currently required WGS84 ``(west, south, east, north)``.
    check_coverage : bool, optional
        When false, an existing managed file is reused without a coverage
        check. When true, ``required_bounds`` is compared to the raster
        header and insufficient coverage grows the same file.
    cache_dir, budget, scratch_dir, product
        Source cache, resource budget, atomic-write scratch, and product
        name used for native spacing.

    Returns
    -------
    RasterDEM
        EPSG:4326 ellipsoidal raster covering the current request.

    Raises
    ------
    DEMCoverageInsufficientError
        If coverage is insufficient and no source can supply terrain.
    ValueError, FileNotFoundError
        If a new DEM is requested without an extent or source.

    """
    path = Path(managed_path)
    scratch = Path(scratch_dir) if scratch_dir is not None else path.parent / "tmp"
    live = existing
    header_grid: GridSpec | None = None
    header_datum: str | None = None
    if live is not None:
        header_grid = live.grid
        header_datum = live.vertical_datum
    elif path.is_file():
        header = _try_read_raster_header(path)
        if header is not None:
            header_grid, header_datum = header

    if header_grid is not None and header_datum is not None:
        if not is_canonical_managed_dem(header_grid, header_datum):
            return _normalize_managed_raster(
                path=path,
                live=live,
                header_grid=header_grid,
                header_datum=header_datum,
                source=source,
                required_bounds=required_bounds,
                check_coverage=check_coverage,
                cache_dir=cache_dir,
                budget=budget,
                scratch=scratch,
                product=product,
            )
        actual = _header_wgs84_bounds(header_grid)
        covered = required_bounds is not None and bounds_cover(actual, required_bounds)
        if not check_coverage or required_bounds is None or covered:
            if not check_coverage or required_bounds is None:
                logger.info("Reusing managed DEM %s without a coverage check", path)
            else:
                logger.info("Reusing managed DEM %s; explicit ROI is covered", path)
            if live is not None:
                return live
            return _load_managed_raster(path, header_datum)

    if required_bounds is None and not path.is_file() and live is None:
        message = "shared DEM has no usable extent"
        reject(message, FileNotFoundError)

    if source is None:
        if (
            live is not None
            and is_geodetic_crs(live.crs)
            and (required_bounds is None or raster_covers_bounds(live, required_bounds))
        ):
            return live
        message = (
            "DEM does not cover the required bounds and no source recipe can expand it"
        )
        actual = None
        if live is not None:
            actual = (
                live.grid.bounds
                if is_geodetic_crs(live.grid.crs)
                else wgs84_bounds(live.grid)
            )
        elif path.is_file():
            header, _datum = read_raster_header(path)
            actual = (
                header.bounds if is_geodetic_crs(header.crs) else wgs84_bounds(header)
            )
        raise DEMCoverageInsufficientError(
            message,
            required_bounds=required_bounds
            if required_bounds is not None
            else (0.0, 0.0, 0.0, 0.0),
            actual_bounds=actual,
            path=path,
        )

    spacing = native_spacing_deg(product or source.product)
    if live is not None and is_geodetic_crs(live.crs):
        extra = expand_bounds(
            required_bounds if required_bounds is not None else live.grid.bounds,
            interpolation_margin_deg(abs(float(live.grid.transform[0]))),
        )
        grid = grow_geodetic_grid(live.grid, extra)
    elif path.is_file():
        header_pair = _try_read_raster_header(path)
        if header_pair is not None and is_geodetic_crs(header_pair[0].crs):
            header = header_pair[0]
            extra = expand_bounds(
                required_bounds if required_bounds is not None else header.bounds,
                interpolation_margin_deg(abs(float(header.transform[0]))),
            )
            grid = grow_geodetic_grid(header, extra)
        else:
            if required_bounds is None:
                message = "shared DEM has no usable extent"
                reject(message, FileNotFoundError)
            grid = geographic_source_grid(required_bounds, spacing)
    else:
        if required_bounds is None:
            message = "shared DEM has no usable extent"
            reject(message, FileNotFoundError)
        grid = geographic_source_grid(required_bounds, spacing)

    if cache_dir is not None and source.cache_dir is None:
        source.cache_dir = Path(cache_dir)
    logger.info("Materializing managed WGS84 DEM onto %s", grid.crs)
    raster = source.to_raster(grid, vertical_datum="ellipsoidal", budget=budget)
    return _replace_managed_raster(raster, path, scratch)
