"""Private lon/lat DEM view shared by rdr2geo backends.

Public RasterDEM objects stay on their caller grid (often UTM).  Radar
geometry kernels sample a north-up EPSG:4326 raster whose affine is
degrees.  This module is the single place that builds that execution view.
It is not a second user-facing DEM product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from affine import Affine

from faninsar.logging import setup_logger
from faninsar.processing.geometry.grids import GridSpec

if TYPE_CHECKING:
    from faninsar.processing.geometry.api import DEM, RasterDEM

logger = setup_logger(__name__)

_GLO30_SPACING_DEG = 1.0 / 3600.0
_GLO90_SPACING_DEG = 1.0 / 1200.0
_NATIVE_SPACING_DEG = {
    "glo30": _GLO30_SPACING_DEG,
    "auto": _GLO30_SPACING_DEG,
    "nasadem": _GLO30_SPACING_DEG,
    "nisar-glo30": _GLO30_SPACING_DEG,
    "srtm": _GLO30_SPACING_DEG,
    "srtm-skadi": _GLO30_SPACING_DEG,
    "glo90": _GLO90_SPACING_DEG,
}
_VIEW_TABLE: dict[str, LonLatDEMView] = {}
_VIEW_TABLE_LIMIT = 8


@dataclass(frozen=True, slots=True)
class LonLatDEMView:
    """Regular north-up WGS84 raster consumed by rdr2geo kernels.

    Attributes
    ----------
    values : numpy.ndarray
        Height samples, shape ``(height, width)``, float64.
    latitude_start_deg, longitude_start_deg : float
        Affine origin (northing of the first row, easting of the first
        column) in degrees.
    latitude_spacing_deg, longitude_spacing_deg : float
        Pixel spacing in degrees.  Latitude spacing is negative for a
        north-up grid.
    height_min_m, height_max_m : float
        Finite height bounds used by native kernels.

    """

    values: np.ndarray
    latitude_start_deg: float
    longitude_start_deg: float
    latitude_spacing_deg: float
    longitude_spacing_deg: float
    height_min_m: float
    height_max_m: float


def is_geodetic_crs(crs: object) -> bool:
    """Return whether ``crs`` is a lon/lat geographic CRS."""
    text = str(crs).strip().upper().replace(" ", "")
    return text in {"EPSG:4326", "OGC:CRS84"}


def native_spacing_deg(product: object | None) -> float:
    """Return the native geographic spacing of a DEM product.

    Parameters
    ----------
    product : str or None
        Registry product name such as ``glo30`` or ``glo90``.

    Returns
    -------
    float
        Pixel size in degrees.  Unknown products fall back to GLO-30's
        1 arc-second grid so a geographic crop still exists.

    """
    key = str(product or "glo30").strip().lower()
    if ":" in key:
        key = key.split(":", 1)[0]
    return _NATIVE_SPACING_DEG.get(key, _GLO30_SPACING_DEG)


def geodetic_crop_grid(
    coverage: GridSpec,
    spacing_deg: float,
    extra_bounds: tuple[float, float, float, float] | None = None,
) -> GridSpec:
    """Build a native-spacing EPSG:4326 grid covering ``coverage``.

    The lon/lat bounding box of ``coverage`` is snapped outward onto the
    source pixel lattice.  Pixel size is the native DEM spacing, not the
    Stack product ``resolution_m``.  ``extra_bounds`` expands that box so a
    processing-domain crop can cover a working frame larger than the public
    product grid.

    Parameters
    ----------
    coverage : GridSpec
        Public Stack grid, typically projected.
    spacing_deg : float
        Positive geographic pixel size in degrees.
    extra_bounds : tuple of float, optional
        Additional WGS84 ``(west, south, east, north)`` envelope unioned
        with ``coverage`` before snapping.

    Returns
    -------
    GridSpec
        North-up EPSG:4326 grid with at least a 6x6 kernel support.

    Raises
    ------
    ValueError
        If ``spacing_deg`` is not a finite positive number, or
        ``extra_bounds`` is present but not four finite ordered values.

    """
    spacing = float(spacing_deg)
    if not np.isfinite(spacing) or spacing <= 0.0:
        message = "geographic DEM spacing must be a finite positive number"
        logger.error(message)
        raise ValueError(message)
    west, south, east, north = _geodetic_bounds(coverage)
    if extra_bounds is not None:
        try:
            extra_west, extra_south, extra_east, extra_north = (
                float(value) for value in extra_bounds
            )
        except (TypeError, ValueError) as error:
            message = "extra_bounds must contain four finite WGS84 values"
            logger.exception(message)
            raise ValueError(message) from error
        extra = (extra_west, extra_south, extra_east, extra_north)
        if len(extra_bounds) != 4 or not np.all(np.isfinite(extra)):
            message = "extra_bounds must contain four finite WGS84 values"
            logger.error(message)
            raise ValueError(message)
        if extra_west >= extra_east or extra_south >= extra_north:
            message = "extra_bounds must be ordered west, south, east, north"
            logger.error(message)
            raise ValueError(message)
        west = min(west, extra_west)
        south = min(south, extra_south)
        east = max(east, extra_east)
        north = max(north, extra_north)
    inverse = 1.0 / spacing
    west_s = float(np.floor(west * inverse + 1.0e-9) * spacing)
    north_s = float(np.ceil(north * inverse - 1.0e-9) * spacing)
    width = max(6, int(np.ceil((east - west_s) / spacing - 1.0e-9)))
    height = max(6, int(np.ceil((north_s - south) / spacing - 1.0e-9)))
    return GridSpec(
        "EPSG:4326",
        Affine(spacing, 0.0, west_s, 0.0, -spacing, north_s),
        height=height,
        width=width,
    )


def lonlat_dem_view(dem: DEM | None) -> LonLatDEMView:
    """Return the cached lon/lat execution view for ``dem``.

    Parameters
    ----------
    dem : DEM or None
        Public sampler.  ``None`` is treated as a 0 m constant field.

    Returns
    -------
    LonLatDEMView
        Kernel-facing geographic raster.  Projected public rasters are
        sampled at lon/lat centres; cells outside the projected footprint
        are filled with 0 m so native kernels keep a finite bucket.

    Raises
    ------
    TypeError
        If ``dem`` is not a supported DEM type.
    ValueError
        If a raster view is smaller than 6x6 or has no finite samples.

    """
    from faninsar.processing.geometry.api import ConstantDEM, RasterDEM

    sampler = ConstantDEM(0.0) if dem is None else dem
    key = _view_key(sampler)
    cached = _VIEW_TABLE.get(key)
    if cached is not None:
        return cached
    if isinstance(sampler, ConstantDEM):
        view = _constant_view(float(sampler.height))
    elif isinstance(sampler, RasterDEM):
        view = _raster_view(sampler)
    else:
        message = f"unsupported DEM type for rdr2geo: {type(sampler).__name__}"
        logger.error(message)
        raise TypeError(message)
    if len(_VIEW_TABLE) >= _VIEW_TABLE_LIMIT:
        oldest = next(iter(_VIEW_TABLE))
        _VIEW_TABLE.pop(oldest, None)
    _VIEW_TABLE[key] = view
    return view


def clear_lonlat_dem_views() -> None:
    """Drop cached execution views.  Intended for tests."""
    _VIEW_TABLE.clear()


def _view_key(dem: object) -> str:
    """Return a stable cache key for one public DEM."""
    identity = getattr(dem, "identity", None)
    if isinstance(identity, str) and identity:
        return identity
    kind = type(dem).__name__
    height = getattr(dem, "height", 0.0)
    datum = getattr(dem, "vertical_datum", "ellipsoidal")
    return f"{kind}:{height}:{datum}"


def _geodetic_bounds(
    grid: GridSpec,
) -> tuple[float, float, float, float]:
    """Return ``(west, south, east, north)`` of ``grid`` in EPSG:4326."""
    west, south, east, north = (float(value) for value in grid.bounds)
    if is_geodetic_crs(grid.crs):
        return west, south, east, north
    from pyproj import Transformer

    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    longitudes, latitudes = transformer.transform(
        [west, east, west, east],
        [south, south, north, north],
    )
    return (
        float(np.min(longitudes)),
        float(np.min(latitudes)),
        float(np.max(longitudes)),
        float(np.max(latitudes)),
    )


def _constant_view(height: float) -> LonLatDEMView:
    """Build a coarse global raster so the six-point stencil stays in bounds."""
    rows, cols = 80, 160
    values = np.full((rows, cols), float(height), dtype=np.float64)
    return LonLatDEMView(
        values=values,
        latitude_start_deg=90.0,
        longitude_start_deg=-180.0,
        latitude_spacing_deg=-180.0 / float(rows - 1),
        longitude_spacing_deg=360.0 / float(cols - 1),
        height_min_m=float(height),
        height_max_m=float(height),
    )


def _raster_view(dem: RasterDEM) -> LonLatDEMView:
    """Materialize a geographic kernel raster from a public RasterDEM."""
    source = dem
    if not is_geodetic_crs(source.crs):
        source = _projected_to_geodetic(dem)
    values = np.asarray(source.array, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 6:
        message = "rdr2geo DEM must be at least 6x6"
        logger.error(message)
        raise ValueError(message)
    finite = np.isfinite(values)
    if not np.any(finite):
        message = "rdr2geo DEM has no finite samples"
        logger.error(message)
        raise ValueError(message)
    affine = Affine(*source.grid.transform)
    return LonLatDEMView(
        values=values,
        latitude_start_deg=float(affine.f),
        longitude_start_deg=float(affine.c),
        latitude_spacing_deg=float(affine.e),
        longitude_spacing_deg=float(affine.a),
        height_min_m=float(np.min(values[finite])),
        height_max_m=float(np.max(values[finite])),
    )


def _projected_to_geodetic(dem: RasterDEM) -> RasterDEM:
    """Sample a projected public raster onto its lon/lat bounding box.

    This path is the fallback when Stack did not materialize a native
    geographic crop from the source tiles.  Corner triangles of the lon/lat
    box fall outside the projected footprint and are filled with 0 m for
    kernel finite-bucket requirements; that fill does not belong on disk.
    """
    from faninsar.processing.geometry.api import RasterDEM

    west, south, east, north = _geodetic_bounds(dem.grid)
    view_grid = GridSpec(
        "EPSG:4326",
        Affine(
            (east - west) / dem.width,
            0.0,
            west,
            0.0,
            -(north - south) / dem.height,
            north,
        ),
        shape=dem.shape,
    )
    view_x, view_y = view_grid.xy_pixel_centers()
    view_values = np.asarray(
        dem.sample(np.asarray(view_y), np.asarray(view_x)),
        dtype=np.float32,
    )
    void_count = int(np.count_nonzero(~np.isfinite(view_values)))
    view_values[~np.isfinite(view_values)] = 0.0
    if void_count:
        logger.info(
            "geometry DEM view filled %d out-of-footprint cells with 0 m",
            void_count,
        )
    return RasterDEM(
        array=view_values,
        grid=view_grid,
        vertical_datum=dem.vertical_datum,
        provenance={"kind": "private-geometry-view"},
    )
