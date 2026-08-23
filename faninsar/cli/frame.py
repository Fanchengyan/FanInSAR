"""``faninsar frame`` — process one or more frames, sub-swaths, or an ROI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    import numpy as np

    from faninsar.query import BoundingBox

logger = setup_logger(__name__)


def _as_path_list(value: str) -> list[str]:
    """Split a comma-separated path list, dropping empty entries."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_roi(value: str | None) -> BoundingBox | None:
    """Parse ``lon_min,lat_min,lon_max,lat_max`` into a BoundingBox."""
    if value is None:
        return None
    from faninsar.query import BoundingBox

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        message = "--roi must be lon_min,lat_min,lon_max,lat_max in EPSG:4326"
        raise SystemExit(message)
    try:
        left, bottom, right, top = (float(part) for part in parts)
    except ValueError as exc:
        message = "--roi must be four numbers: lon_min,lat_min,lon_max,lat_max"
        raise SystemExit(message) from exc
    if left >= right or bottom >= top:
        message = "--roi needs left < right and bottom < top"
        raise SystemExit(message)
    return BoundingBox(left, bottom, right, top, crs="EPSG:4326")


def _burst_body(body: str, swath: str) -> list[int] | range | str:
    """Parse one per-swath burst body into a list, range, or ``"all"``."""
    if body == "all":
        return "all"
    if ":" in body:
        start, _, stop = body.partition(":")
        if not (start.strip().isdigit() and stop.strip().isdigit()):
            message = f"invalid burst range {body!r} for {swath}; expected start:stop"
            raise SystemExit(message)
        return range(int(start), int(stop))
    parts = [part.strip() for part in body.split(",") if part.strip()]
    if not parts or not all(part.isdigit() for part in parts):
        message = f"invalid burst list {body!r} for {swath}"
        raise SystemExit(message)
    return [int(part) for part in parts]


def _parse_burst_selection(
    value: str | None,
) -> dict[str, list[int] | range | str] | None:
    """Parse ``IW1:0,1,2,IW2:2:5,IW3:all`` into per-swath selections."""
    if value is None:
        return None
    result: dict[str, list[int] | range | str] = {}
    current_swath: str | None = None
    for raw in value.split(","):
        segment = raw.strip()
        if not segment:
            continue
        if ":" in segment:
            swath, _, body = segment.partition(":")
            swath, body = swath.strip(), body.strip()
            if not swath or not body:
                message = f"invalid --bursts entry {raw!r}"
                raise SystemExit(message)
            current_swath = swath
            result[swath] = _burst_body(body, swath)
        elif current_swath is not None:
            existing = result[current_swath]
            added = _burst_body(segment, current_swath)
            if not isinstance(existing, list) or not isinstance(added, list):
                message = f"cannot extend {current_swath} selection with {segment!r}"
                raise SystemExit(message)
            existing.extend(added)
        else:
            message = "--bursts must start with a swath entry like IW1:0"
            raise SystemExit(message)
    return result


def _resolve_dem_path(value: str, output: Path) -> Path:
    """Resolve a --dem argument to a path.

    Absolute/relative paths pass through; a bare file name is placed under
    <output>/dem/.
    """
    path = Path(value)
    if path.is_absolute() or path.parent != Path():
        return path
    return output / "dem" / path


def _cli_dem_bounds(
    roi: BoundingBox | None,
    reference: list[str],
) -> tuple[float, float, float, float]:
    """Return EPSG:4326 bounds for the CLI DEM build.

    Uses ROI bounds or the union of every reference SAFE burst radar quad
    buffered by 2 km.
    """
    if roi is not None:
        return (
            float(roi.left),
            float(roi.bottom),
            float(roi.right),
            float(roi.top),
        )
    from faninsar.missions.sentinel1.safe import open_safe_product
    from faninsar.processing.pipeline.geo_lut import burst_geo_quad_lonlat
    from faninsar.processing.pipeline.production import (
        DEM_BOUNDS_BUFFER_M,
        _quad_bounds_with_buffer_m,
        _radar_model,
    )

    quads: list[np.ndarray] = []
    for path in reference:
        product = open_safe_product(path)
        for swath_item in product.swaths:
            shape = (swath_item.lines_per_burst, swath_item.samples_per_burst)
            for burst in swath_item.bursts:
                geometry = _radar_model(
                    swath_item,
                    burst,
                    shape=shape,
                    row0=burst.index * swath_item.lines_per_burst,
                    col0=0,
                )
                quad = burst_geo_quad_lonlat(
                    geometry=geometry,
                    radar_shape=shape,
                    dem=None,
                )
                if quad is not None:
                    quads.append(quad)
    return _quad_bounds_with_buffer_m(quads, DEM_BOUNDS_BUFFER_M)


def run_frame_cli(
    *,
    reference: str,
    secondary: str,
    output: str,
    dem: str | None = None,
    reference_orbit: str | None = None,
    secondary_orbit: str | None = None,
    swaths: str = "IW1,IW2,IW3",
    bursts: str | None = None,
    roi: str | None = None,
    az_looks: int = 2,
    rg_looks: int = 10,
    goldstein: float = 0.5,
    device: str = "cpu",
    dem_source: str | None = None,
) -> int:
    """Run the unified pair production pipeline from the command line.

    ``reference``/``secondary`` may be comma-separated lists of SAFE products
    spanning consecutive frames along the same pass. ``--roi`` takes
    ``lon_min,lat_min,lon_max,lat_max`` (EPSG:4326) and selects the bursts
    intersecting it, overriding ``--swaths``/``--bursts``.

    Returns
    -------
    int
        Exit code (0 on success).

    Raises
    ------
    ValueError
        When ``dem_source`` names an unknown or unwired provider (the CLI
        converts this to a non-zero exit with the failure message).

    """
    from faninsar.processing.geometry.dem import RasterDEM
    from faninsar.processing.pipeline import run_pair
    from faninsar.processing.pipeline.production import resolve_auto_dem

    roi_box = _parse_roi(roi)
    dem_sampler = None
    if dem is not None:
        dem_path = _resolve_dem_path(dem, Path(output))
        if not dem_path.exists():
            bounds = _cli_dem_bounds(roi_box, _as_path_list(reference))
            # Shared datum-aware wrap rule; never wraps ellipsoidal sources.
            dem_sampler = resolve_auto_dem(
                bounds,
                output_dir=Path(output),
                geoid_correction=True,
                dem_source=dem_source,
                output_name=dem_path.name,
            )
        else:
            dem_sampler = RasterDEM(path=dem_path, interpolation="biquintic")

    state = run_pair(
        _as_path_list(reference),
        _as_path_list(secondary),
        output_dir=Path(output),
        dem=dem_sampler,
        roi=roi_box,
        swaths=tuple(name.strip() for name in swaths.split(",") if name.strip()),
        bursts=_parse_burst_selection(bursts),
        multilook=(az_looks, rg_looks),
        goldstein_alpha=goldstein,
        device=device,
        dem_source=dem_source,
        reference_orbit_path=(
            _as_path_list(reference_orbit) if reference_orbit else None
        ),
        secondary_orbit_path=(
            _as_path_list(secondary_orbit) if secondary_orbit else None
        ),
    )
    assert state.complex_ifg is not None
    logger.info("merged frame: %s", state.complex_ifg.shape)
    logger.info("timings: %s", state.stage_timings_s)
    return 0
