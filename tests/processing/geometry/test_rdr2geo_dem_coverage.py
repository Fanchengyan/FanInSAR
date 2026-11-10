"""Public rdr2geo reports DEM extent shortage without treating nodata as coverage."""

# The repository intentionally keeps tests as a namespace package.
# ruff: noqa: INP001

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from affine import Affine

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.processing.geometry import ConstantDEM, RadarGeometryModel, RasterDEM
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.geometry.grids import GridSpec
from faninsar.processing.geometry.managed_dem import (
    DEMCoverageInsufficientError,
    union_bounds,
)
from faninsar.processing.geometry.prepare_production import (
    _rdr2geo_dem_required_bounds,
    run_rdr2geo,
)


def _synthetic_model() -> RadarGeometryModel:
    epoch = datetime(2016, 12, 7, 11, 18, 0, tzinfo=UTC)
    radius = 7_071_000.0
    omega = 0.001
    vectors = []
    for index in range(21):
        t = 10.0 * index
        angle = omega * t
        vectors.append(
            OrbitStateVector(
                time=epoch + timedelta(seconds=t),
                position_m=(
                    radius * np.cos(angle),
                    0.0,
                    radius * np.sin(angle),
                ),
                velocity_m_s=(
                    -radius * omega * np.sin(angle),
                    0.0,
                    radius * omega * np.cos(angle),
                ),
            )
        )
    orbit = OrbitMetadata(
        reference_frame="ITRF",
        source="synthetic",
        vectors=tuple(vectors),
    )
    grid = RadarGrid(
        shape=(64, 128),
        starting_slant_range_m=800_000.0,
        range_spacing_m=2.3,
        sensing_start=vectors[5].time,
        azimuth_time_interval_s=0.002,
        wavelength_m=0.0555,
        look_direction="right",
    )
    return RadarGeometryModel.from_radar_grid(grid, orbit)


def _full_radar_query() -> tuple[np.ndarray, np.ndarray]:
    azimuth, slant = np.meshgrid(
        np.arange(64, dtype=np.float64),
        np.arange(128, dtype=np.float64),
        indexing="ij",
    )
    return azimuth, slant


def _frame_envelope(
    model: RadarGeometryModel, azimuth: np.ndarray, slant: np.ndarray
) -> tuple[float, float, float, float]:
    full = run_rdr2geo(model, azimuth, slant, None, device="cpu", max_iter=40)
    lat = np.asarray(full.latitude_deg)
    lon = np.asarray(full.longitude_deg)
    good = np.isfinite(lat) & np.isfinite(lon)
    assert int(good.sum()) > 0
    return (
        float(lon[good].min()),
        float(lat[good].min()),
        float(lon[good].max()),
        float(lat[good].max()),
    )


def _geographic_dem(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    spacing: float = 0.05,
    height: float = 123.0,
    interior_nodata: bool = False,
) -> RasterDEM:
    width = max(8, int(np.ceil((east - west) / spacing)))
    height_px = max(8, int(np.ceil((north - south) / spacing)))
    array = np.full((height_px, width), height, dtype=np.float32)
    if interior_nodata:
        array[3:5, 3:5] = np.nan
    grid = GridSpec(
        crs="EPSG:4326",
        transform=Affine(spacing, 0.0, west, 0.0, -spacing, north),
        height=height_px,
        width=width,
    )
    return RasterDEM(array=array, grid=grid, vertical_datum="ellipsoidal")


def test_full_frame_rdr2geo_raises_when_dem_extent_covers_only_half() -> None:
    """A live RasterDEM that omits half the query envelope is a coverage error."""
    model = _synthetic_model()
    azimuth, slant = _full_radar_query()
    west, south, east, north = _frame_envelope(model, azimuth, slant)
    mid_lat = (south + north) / 2.0
    partial = _geographic_dem(west - 0.05, south - 0.05, east + 0.05, mid_lat)
    with pytest.raises(DEMCoverageInsufficientError) as info:
        run_rdr2geo(model, azimuth, slant, partial, device="cpu", max_iter=40)
    error = info.value
    required = error.required_bounds
    assert required[1] <= south
    assert required[3] >= north
    actual = error.actual_bounds
    assert actual is not None
    assert actual[3] < north


def test_covering_dem_with_source_nodata_does_not_raise_coverage() -> None:
    """Nodata inside a covering raster stays numerical NaN, not a coverage error."""
    model = _synthetic_model()
    azimuth, slant = _full_radar_query()
    west, south, east, north = _frame_envelope(model, azimuth, slant)
    dem = _geographic_dem(
        west - 2.0,
        south - 2.0,
        east + 2.0,
        north + 2.0,
        interior_nodata=True,
    )
    run_rdr2geo(model, azimuth, slant, dem, device="cpu", max_iter=40)


def _solver_kwargs() -> dict[str, float | int | str]:
    return {
        "device": "cpu",
        "max_iter": 40,
        "range_tol_m": 0.01,
        "doppler_tol_hz": 0.1,
    }


def test_extracted_demand_matches_rdr2geo_coverage_error() -> None:
    """Shared demand bounds are the same envelope run_rdr2geo reports."""
    model = _synthetic_model()
    azimuth, slant = _full_radar_query()
    west, south, east, north = _frame_envelope(model, azimuth, slant)
    partial = _geographic_dem(
        west - 0.05, south - 0.05, east + 0.05, (south + north) / 2.0
    )
    demand = _rdr2geo_dem_required_bounds(
        model, azimuth, slant, partial, **_solver_kwargs()
    )
    with pytest.raises(DEMCoverageInsufficientError) as info:
        run_rdr2geo(model, azimuth, slant, partial, device="cpu", max_iter=40)
    assert demand == info.value.required_bounds


def test_broadcast_crop_views_match_full_window_demand() -> None:
    """Zero-stride crop views request the same DEM as a materialized window."""
    model = _synthetic_model()
    azimuth, slant = _full_radar_query()
    west, south, east, north = _frame_envelope(model, azimuth, slant)
    dem = _geographic_dem(west - 0.05, south - 0.05, east + 0.05, (south + north) / 2.0)
    row0, row1, col0, col1 = 8, 24, 16, 48
    window_az = azimuth[row0:row1, col0:col1]
    window_rg = slant[row0:row1, col0:col1]
    rows = np.arange(row0, row1, dtype=np.float64)[:, None]
    cols = np.arange(col0, col1, dtype=np.float64)[None, :]
    edge_az, edge_rg = np.broadcast_arrays(rows, cols)
    kwargs = _solver_kwargs()
    full = _rdr2geo_dem_required_bounds(model, window_az, window_rg, dem, **kwargs)
    edge = _rdr2geo_dem_required_bounds(model, edge_az, edge_rg, dem, **kwargs)
    assert full is not None
    assert edge == full
    assert edge_az.strides[1] == 0
    assert edge_rg.strides[0] == 0


def test_constant_dem_has_no_rdr2geo_coverage_demand() -> None:
    """Non-raster samplers skip DEM extent preflight."""
    model = _synthetic_model()
    azimuth, slant = _full_radar_query()
    demand = _rdr2geo_dem_required_bounds(
        model, azimuth, slant, ConstantDEM(0.0), **_solver_kwargs()
    )
    assert demand is None


def test_union_of_northern_and_western_window_demands_covers_both_edges() -> None:
    """Two disjoint radar windows require the union, not the first window alone."""
    model = _synthetic_model()
    azimuth, slant = _full_radar_query()
    west, south, _east, _north = _frame_envelope(model, azimuth, slant)
    tiny = _geographic_dem(west, south, west + 0.001, south + 0.001)
    kwargs = _solver_kwargs()
    northern = _rdr2geo_dem_required_bounds(
        model, azimuth[-4:, :], slant[-4:, :], tiny, **kwargs
    )
    western = _rdr2geo_dem_required_bounds(
        model, azimuth[:, :8], slant[:, :8], tiny, **kwargs
    )
    assert northern is not None
    assert western is not None
    combined = union_bounds(northern, western)
    assert combined[0] <= min(northern[0], western[0])
    assert combined[1] <= min(northern[1], western[1])
    assert combined[2] >= max(northern[2], western[2])
    assert combined[3] >= max(northern[3], western[3])
    assert combined != northern
    assert combined != western
