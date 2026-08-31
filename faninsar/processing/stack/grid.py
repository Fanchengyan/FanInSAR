"""Authoritative automatic projected-grid selection for Stack."""

# The project uses descriptive boundary errors in this small integration
# module; the exception-message lint rules are intentionally disabled here.
# ruff: noqa: EM101, TRY003

from __future__ import annotations

import numpy as np
from affine import Affine
from pyproj import Transformer

from faninsar._core.geo.grids import GridSpec
from faninsar.logging import setup_logger
from faninsar.processing.dem.resources import ResourceBudget, preflight_grid
from faninsar.processing.dem.seam import ExplicitAntimeridianError

logger = setup_logger(__name__)


def _geometry_bounds(roi: object) -> tuple[float, float, float, float]:
    """Extract finite WGS84 bounds from common ROI representations."""
    if roi is None:
        raise ValueError("automatic Stack grid selection requires an ROI")
    roi_crs = getattr(roi, "crs", None)
    if roi_crs is not None:
        try:
            if str(roi_crs) not in {"EPSG:4326", "WGS 84"} and hasattr(
                roi, "to_crs"
            ):
                roi = roi.to_crs("EPSG:4326")
        except (TypeError, ValueError):
            raise ValueError("ROI CRS cannot be transformed to EPSG:4326") from None
    if hasattr(roi, "total_bounds"):
        values = tuple(float(value) for value in roi.total_bounds)
    elif hasattr(roi, "geodataframe"):
        values = tuple(float(value) for value in roi.geodataframe.total_bounds)
    elif hasattr(roi, "bounds"):
        values = tuple(float(value) for value in roi.bounds)
    else:
        try:
            values = tuple(float(value) for value in roi)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise TypeError("ROI must expose WGS84 bounds") from error
    if len(values) != 4 or not np.all(np.isfinite(values)):
        raise ValueError("ROI bounds must contain four finite values")
    west, south, east, north = values
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError("ROI longitude must be within [-180, 180]")
    if south >= north:
        raise ValueError("ROI latitude bounds must be ordered and non-empty")
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise ValueError("ROI latitude must be within [-90, 90]")
    return values


def _unwrap_bounds(west: float, east: float) -> tuple[float, float, bool]:
    """Return the shortest continuous longitude interval and seam flag."""
    raw_width = east - west
    if raw_width < 0:
        raw_width += 360
    if raw_width > 180:
        return east, west + 360, True
    return west, west + raw_width, west > east


def _centre_and_interval(
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float, bool]:
    """Compute deterministic antimeridian-aware center and unwrapped bounds."""
    west, south, east, north = bounds
    if west == -180 and east == 180:
        return 0.0, south, north, 360.0, False
    start, stop, seam = _unwrap_bounds(west, east)
    center = (start + stop) / 2
    center = ((center + 180) % 360) - 180
    return center, south, north, stop - start, seam


def _epsg_for_center(longitude: float, latitude: float) -> str:
    """Select UTM or UPS using the accepted latitude cut-offs."""
    if latitude >= 84:
        return "EPSG:32661"
    if latitude < -80:
        return "EPSG:32761"
    zone = min(60, max(1, int((longitude + 180) // 6) + 1))
    return f"EPSG:{32600 + zone if latitude >= 0 else 32700 + zone}"


def _coerce_budget(budget: object | None) -> ResourceBudget | None:
    """Adapt Stack's process budget to the DEM cell/byte preflight seam."""
    if budget is None or isinstance(budget, ResourceBudget):
        return budget
    cells = getattr(budget, "max_decoded_bytes", None)
    output = getattr(budget, "max_decoded_bytes", None)
    fetch = getattr(budget, "max_encoded_bytes", None)
    temporary = getattr(budget, "max_temporary_bytes", None)
    if all(
        isinstance(value, int) and value > 0
        for value in (cells, output, fetch, temporary)
    ):
        return ResourceBudget(
            max_cells=max(1, cells // 4),
            max_output_bytes=output,
            max_fetch_bytes=fetch,
            max_temporary_bytes=temporary,
        )
    raise TypeError("unsupported Stack resource budget")


def automatic_grid(
    roi: object,
    *,
    resolution_m: float = 30.0,
    margin_m: float = 0.0,
    budget: object | None = None,
    explicit: bool = False,
) -> GridSpec:
    """Build Stack's deterministic UTM/UPS grid from a WGS84 ROI.

    Explicit grids should be passed directly to :func:`resolve_stack_grid`.
    The automatic path warns for seams, projection boundaries, and large
    extents, while continuing with the center-selected CRS.
    """
    if not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("resolution_m must be a finite positive number")
    if not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError("margin_m must be finite and non-negative")
    bounds = _geometry_bounds(roi)
    center_lon, south, north, lon_width, seam = _centre_and_interval(bounds)
    if explicit and seam:
        raise ExplicitAntimeridianError(
            "explicit Stack ROI crossing the antimeridian is unsupported"
        )
    center_lat = (south + north) / 2
    crs = _epsg_for_center(center_lon, center_lat)
    if seam:
        logger.warning("automatic Stack ROI crosses the antimeridian; continuing")
    if lon_width > 6:
        logger.warning("automatic Stack ROI crosses a UTM zone; continuing")
    if south < -80 < north or south < 84 < north:
        logger.warning("automatic Stack ROI crosses a UTM/UPS boundary; continuing")

    longitudes = np.array(
        [center_lon + ((x - center_lon + 180) % 360 - 180) for x in bounds[::2]]
    )
    latitudes = np.array([south, north])
    mesh_lon, mesh_lat = np.meshgrid(longitudes, latitudes)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = transformer.transform(mesh_lon.ravel(), mesh_lat.ravel())
    x_min, x_max = float(np.min(xs)) - margin_m, float(np.max(xs)) + margin_m
    y_min, y_max = float(np.min(ys)) - margin_m, float(np.max(ys)) + margin_m
    width = max(1, int(np.ceil((x_max - x_min) / resolution_m)))
    height = max(1, int(np.ceil((y_max - y_min) / resolution_m)))
    x_max = x_min + width * resolution_m
    y_min = y_max - height * resolution_m
    longest = max(x_max - x_min, y_max - y_min)
    if longest > 1_000_000:
        logger.warning("automatic Stack projected ROI exceeds 1000 km; continuing")
    preflight_grid(height, width, budget=_coerce_budget(budget))
    return GridSpec(
        crs=crs,
        transform=Affine(resolution_m, 0, x_min, 0, -resolution_m, y_max),
        height=height,
        width=width,
    )


def resolve_stack_grid(
    grid: GridSpec | str,
    *,
    roi: object | None = None,
    resolution_m: float = 30.0,
    budget: object | None = None,
    explicit_roi: bool = False,
) -> GridSpec:
    """Resolve an explicit GridSpec or the automatic UTM/UPS policy."""
    if isinstance(grid, GridSpec):
        preflight_grid(grid.height, grid.width, budget=_coerce_budget(budget))
        return grid
    if str(grid).strip().lower() != "auto":
        raise ValueError("Stack grid must be a GridSpec or 'auto'")
    return automatic_grid(
        roi,
        resolution_m=resolution_m,
        budget=budget,
        explicit=explicit_roi,
    )


__all__ = ["automatic_grid", "resolve_stack_grid"]
