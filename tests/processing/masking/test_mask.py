"""Slice A geometry-core tests for the general mask operator (PROPOSAL-0039).

Covers TDD-plan items 1-8 against ``faninsar/processing/masking/mask.py``:

1. ``MaskSampler`` runtime-checkable protocol (RasterMask/VectorMask satisfy).
2. ``RasterMask`` extraction (``excluded_values`` / ``threshold`` / ``invert``,
   out-of-extent -> valid, NoData -> invalid) and nearest-neighbour resampling
   onto an exact target grid.
3. ``VectorMask`` rasterization on an exact grid with the ``all_touched``
   tie-break pinned on coastline cells.
4. ``MaskOperator`` union / intersection / invert composition.
5. UTM planar land buffer: accuracy vs a fine geodesic reference at 70 N
   (port of ``round6_utm_design_audit.py`` C), ``buffer_km=0`` identity,
   zone-edge distortion gate for a 10-deg-wide ROI (port of round6 D).
6. Padded fetch band: the round-5 neighbor-tile blocker regression and the
   zero-buffer identity.
7. Antimeridian seam guard: the 11 executed cases of
   ``round5_seam_guard.py`` re-expressed on the padded band.
8. ``boundary is None`` guard (shapely 2.x GeometryCollection).

All tests are offline: no network access, fixtures are written locally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pyproj
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon, box

from faninsar.processing.masking.mask import (
    MaskOperator,
    MaskSampler,
    RasterMask,
    VectorMask,
    _utm_crs,
    antimeridian_seam_guard,
    buffer_land_utm_km,
    padded_fetch_band,
    rasterize_to_grid,
    resample_mask_to_grid,
    snap_band,
)

if TYPE_CHECKING:
    from pathlib import Path

GEOD = pyproj.Geod(ellps="WGS84")


def _write_uint8_tif(
    path: Path,
    values: np.ndarray,
    transform: object,
    *,
    nodata: int | None = None,
) -> Path:
    """Write one single-band uint8 EPSG:4326 GeoTIFF fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profile: dict[str, object] = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)
    return path


def _disk_sum(geom: object, radius_m: float, disk_pts: int, step_m: float) -> object:
    """Port of the executed prototype disk-sum (round5/round6)."""
    import shapely

    disks: list[Polygon] = []
    parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
    step_deg = step_m / 111132.0
    for part in parts:
        boundary = part.boundary
        if boundary is None or boundary.is_empty:
            continue
        length = boundary.length
        if length <= 0:
            continue
        n = math.ceil(length / step_deg) + 1
        pts = [boundary.interpolate(i * (length / n)) for i in range(n)]
        for pt in pts:
            circle = [
                (
                    GEOD.fwd(pt.x, pt.y, az, radius_m)[0],
                    GEOD.fwd(pt.x, pt.y, az, radius_m)[1],
                )
                for az in np.linspace(0.0, 360.0, disk_pts, endpoint=False)
            ]
            disks.append(Polygon(circle))
    if not disks:
        return geom
    return shapely.union_all([geom, *disks])


def _geodesic_buffer_fine(geom: object, radius_m: float) -> object:
    """High-fidelity geodesic reference: step = radius/10, 360-point disks."""
    return _disk_sum(geom, radius_m, 360, radius_m * 0.1)


@dataclass
class _LongitudeKeepMask:
    """Fake sampler keeping longitudes strictly below a threshold (True=keep)."""

    keep_below_lon: float

    def sample(self, latitude_deg: object, longitude_deg: object) -> np.ndarray:
        """Return the boolean keep plane for the requested coordinates."""
        lat = np.asarray(latitude_deg, dtype=np.float64)
        lon = np.asarray(longitude_deg, dtype=np.float64)
        lat_b, lon_b = np.broadcast_arrays(lat, lon)
        del lat_b
        return np.asarray(lon_b < self.keep_below_lon, dtype=bool)


# ---------------------------------------------------------------------------
# 1. MaskSampler protocol
# ---------------------------------------------------------------------------


class _NotASampler:
    """Object without a ``sample`` method (negative protocol check)."""


def test_mask_sampler_protocol(tmp_path: Path) -> None:
    """RasterMask, VectorMask, and MaskOperator satisfy MaskSampler."""
    path = _write_uint8_tif(
        tmp_path / "tiny.tif",
        np.zeros((2, 2), dtype="uint8"),
        from_origin(0.0, 1.0, 0.5, 0.5),
    )
    raster_mask = RasterMask(path)
    vector_mask = VectorMask(geometries=box(0.0, 0.0, 0.5, 0.5))
    operator = MaskOperator.union(raster_mask, vector_mask)
    assert isinstance(raster_mask, MaskSampler)
    assert isinstance(vector_mask, MaskSampler)
    assert isinstance(operator, MaskSampler)
    assert not isinstance(_NotASampler(), MaskSampler)
    # samplers return boolean keep planes
    kept = vector_mask.sample(np.array([0.25, 0.75]), np.array([0.25, 0.75]))
    assert kept.dtype == bool


# ---------------------------------------------------------------------------
# 2. RasterMask extraction + nearest-neighbour exact-grid resampling
# ---------------------------------------------------------------------------


def _class_raster(tmp_path: Path) -> Path:
    """10x10 classified raster: west half class 80, one NoData cell."""
    values = np.zeros((10, 10), dtype="uint8")
    values[:, :5] = 80
    values[0, 0] = 255
    transform = from_origin(10.0, 2.0, 0.1, 0.1)
    return _write_uint8_tif(tmp_path / "classes.tif", values, transform, nodata=255)


def test_raster_mask_extraction(tmp_path: Path) -> None:
    """excluded_values / threshold / invert extract to boolean keep planes."""
    path = _class_raster(tmp_path)
    mask = RasterMask(path, excluded_values=frozenset({80}))
    lats = np.array([1.95, 1.85, 1.85, 0.95, 1.75])
    lons = np.array([10.05, 10.15, 10.55, 10.25, 12.0])
    kept = mask.sample(lats, lons)
    assert kept.dtype == bool
    # NoData cell -> invalid (not kept); class 80 -> removed; class 0 -> kept;
    # out-of-extent cells -> valid (kept), the user's exclusion intent never
    # silently deletes data.
    np.testing.assert_array_equal(kept, np.array([False, False, True, True, True]))

    inverted = RasterMask(path, excluded_values=frozenset({80}), invert=True)
    kept_inv = inverted.sample(lats, lons)
    # invert flips the predicate only: class 80 kept, class 0 removed; the
    # structural fill rules (NoData invalid, out-of-extent valid) hold.
    np.testing.assert_array_equal(kept_inv, np.array([False, True, False, True, True]))

    # threshold mode on a continuous occurrence-style raster
    occurrence = np.zeros((10, 10), dtype="uint8")
    occurrence[:5, :] = 100
    occurrence[5:, :] = 20
    occurrence[9, 9] = 255
    occ_path = _write_uint8_tif(
        tmp_path / "occurrence.tif",
        occurrence,
        from_origin(10.0, 2.0, 0.1, 0.1),
        nodata=255,
    )
    threshold_mask = RasterMask(occ_path, threshold=50)
    occ_kept = threshold_mask.sample(
        np.array([1.75, 1.25, 1.05]), np.array([10.25, 10.25, 10.95])
    )
    # occurrence 100 -> removed; occurrence 20 -> kept; NoData -> invalid
    np.testing.assert_array_equal(occ_kept, np.array([False, True, False]))

    inverted_threshold = RasterMask(occ_path, threshold=50, invert=True)
    occ_kept_inv = inverted_threshold.sample(
        np.array([1.75, 1.25, 1.05]), np.array([10.25, 10.25, 10.95])
    )
    np.testing.assert_array_equal(occ_kept_inv, np.array([True, False, False]))


def test_raster_mask_boolean_default(tmp_path: Path) -> None:
    """With no knobs, a boolean-mask raster excludes its nonzero cells."""
    values = np.zeros((4, 4), dtype="uint8")
    values[1:3, 1:3] = 1
    path = _write_uint8_tif(
        tmp_path / "boolean.tif", values, from_origin(0.0, 1.0, 0.25, 0.25)
    )
    mask = RasterMask(path)
    kept = mask.sample(np.array([0.875, 0.625, 0.125]), np.array([0.125, 0.625, 0.625]))
    np.testing.assert_array_equal(kept, np.array([True, False, True]))


def test_raster_mask_resample_exact_grid(tmp_path: Path) -> None:
    """Nearest-neighbour resample onto an exact target grid (uint8 0/1/255)."""
    path = _class_raster(tmp_path)
    transform = from_origin(10.0, 2.0, 0.1, 0.1)
    mask = RasterMask(path, excluded_values=frozenset({80}))

    expected = np.zeros((10, 10), dtype="uint8")
    expected[:, :5] = 1
    expected[0, 0] = 255  # NoData -> invalid
    out = resample_mask_to_grid(mask, transform, (10, 10))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, expected)

    # target grid shifted west by one source pixel: lon 9..10 is outside the
    # raster extent -> valid (0); lon 10..11 maps onto the source columns.
    west = from_origin(9.0, 2.0, 0.1, 0.1)
    out_west = resample_mask_to_grid(mask, west, (10, 20))
    np.testing.assert_array_equal(out_west[:, :10], np.zeros((10, 10), "uint8"))
    np.testing.assert_array_equal(out_west[:, 10:], expected)

    # nearest-neighbour only: target offset by 0.06 deg shifts every lookup
    # one source column east (no fractional values are ever invented).
    offset = from_origin(10.06, 2.0, 0.1, 0.1)
    out_off = resample_mask_to_grid(mask, offset, (10, 9))
    np.testing.assert_array_equal(out_off, expected[:, 1:])


# ---------------------------------------------------------------------------
# 3. VectorMask rasterization
# ---------------------------------------------------------------------------


def test_vector_mask_rasterize() -> None:
    """GeoDataFrame and shapely geometries rasterize identically (uint8 0/1)."""
    transform = from_origin(0.0, 2.0, 0.1, 0.1)
    shape = (20, 20)
    water = box(0.5, 0.5, 1.5, 1.5)

    frame = gpd.GeoDataFrame({"name": ["lake"]}, geometry=[water], crs="EPSG:4326")
    from_frame = VectorMask(geometries=frame).rasterize(transform, shape)
    from_shapely = VectorMask(geometries=water).rasterize(transform, shape)
    assert from_frame.dtype == np.uint8
    np.testing.assert_array_equal(from_frame, from_shapely)

    expected = np.zeros(shape, dtype="uint8")
    expected[5:15, 5:15] = 1
    np.testing.assert_array_equal(from_shapely, expected)

    # sampler view: inside the water polygons is removed (True=keep outside)
    kept = VectorMask(geometries=frame).sample(
        np.array([1.0, 0.2]), np.array([1.0, 1.8])
    )
    np.testing.assert_array_equal(kept, np.array([False, True]))


def test_vector_mask_rasterize_all_touched_tie_break() -> None:
    """all_touched=False burns centers inside; True burns every touched cell."""
    transform = from_origin(0.0, 2.0, 0.1, 0.1)
    shape = (20, 20)
    coastline = box(0.02, 0.02, 1.02, 1.02)
    vector_mask = VectorMask(geometries=coastline)

    center = vector_mask.rasterize(transform, shape, all_touched=False)
    touched = vector_mask.rasterize(transform, shape, all_touched=True)
    assert int(center.sum()) == 100  # centers 0.05..0.95 -> rows/cols 10..19/0..9
    assert int(touched.sum()) == 121  # cells 9..19 x 0..10 touched by the edges
    assert center[10, 10] == 0  # center (0.95, 1.05) lies outside the coastline
    assert touched[10, 10] == 1  # but its cell is touched by the edge


def test_rasterize_to_grid_nodata_plane() -> None:
    """rasterize_to_grid returns uint8 0/1 with 255 where no data exists."""
    transform = from_origin(0.0, 2.0, 0.1, 0.1)
    validity = np.ones((20, 20), dtype=bool)
    validity[16:, :] = False  # e.g. DEM NoData rows
    out = rasterize_to_grid(
        [box(0.5, 0.5, 1.5, 1.5)], transform, (20, 20), validity=validity
    )
    assert out.dtype == np.uint8
    assert out[10, 10] == 1
    assert out[0, 0] == 0
    assert np.all(out[16:, :] == 255)

    # without a validity plane no 255 is produced
    plain = rasterize_to_grid(box(0.5, 0.5, 1.5, 1.5), transform, (20, 20))
    assert int(plain.sum()) == 100
    assert not np.any(plain == 255)


# ---------------------------------------------------------------------------
# 4. MaskOperator composition
# ---------------------------------------------------------------------------


def test_mask_operator_composition() -> None:
    """union/intersection/invert compose the REMOVED regions of masks."""
    a = _LongitudeKeepMask(0.5)  # removes lon >= 0.5
    b = _LongitudeKeepMask(0.8)  # removes lon >= 0.8
    lats = np.zeros(4)
    lons = np.array([0.2, 0.6, 0.9, -1.0])

    union = MaskOperator.union(a, b)
    # union of removed regions: a cell is kept only if EVERY mask keeps it
    np.testing.assert_array_equal(
        union.sample(lats, lons), np.array([True, False, False, True])
    )

    intersection = MaskOperator.intersection(a, b)
    # intersection of removed regions: kept if ANY mask keeps it
    np.testing.assert_array_equal(
        intersection.sample(lats, lons), np.array([True, True, False, True])
    )

    inverted = MaskOperator.invert(a)
    np.testing.assert_array_equal(
        inverted.sample(lats, lons), np.array([False, True, True, False])
    )

    # operators nest and still satisfy the sampler protocol
    nested = MaskOperator.union(MaskOperator.invert(a), b)
    assert isinstance(nested, MaskSampler)
    np.testing.assert_array_equal(
        nested.sample(lats, lons), np.array([False, True, False, False])
    )


def test_mask_operator_validation() -> None:
    """Empty compositions raise a structured ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        MaskOperator.union()
    with pytest.raises(ValueError, match="exactly one"):
        MaskOperator.invert(_LongitudeKeepMask(0.5), _LongitudeKeepMask(0.8))
    with pytest.raises(ValueError, match="exactly one"):
        MaskOperator.invert()


# ---------------------------------------------------------------------------
# 5. UTM planar land buffer
# ---------------------------------------------------------------------------


def test_buffer_land_utm_km_geodesic_accuracy_at_70n() -> None:
    """UTM planar buffer matches the fine geodesic reference within 0.5%."""
    square = box(0.0, 70.0, 0.05, 70.05)
    buffered = buffer_land_utm_km(square, 1.0, zone_lon=0.025, zone_lat=70.025)
    reference = _geodesic_buffer_fine(square, 1000.0)
    area_utm = abs(GEOD.geometry_area_perimeter(buffered)[0])
    area_reference = abs(GEOD.geometry_area_perimeter(reference)[0])
    diff_pct = 100.0 * abs(area_utm - area_reference) / area_reference
    assert diff_pct < 0.5


def test_buffer_land_utm_km_zero_buffer_identity() -> None:
    """buffer_km=0 is the identity (UTM round-trip noise only)."""
    lake = box(10.0, 40.0, 10.5, 40.5)
    out = buffer_land_utm_km(lake, 0.0, zone_lon=10.25, zone_lat=40.25)
    assert out.symmetric_difference(lake).area < 1e-12


def test_buffer_land_utm_km_zone_edge_distortion_within_half_percent() -> None:
    """Zone selection (EPSG:326xx/327xx) and the 10-deg-ROI distortion gate."""
    assert _utm_crs(104.0, 1.0) == "EPSG:32648"
    assert _utm_crs(104.0, -1.0) == "EPSG:32748"
    to_zone = pyproj.Transformer.from_crs(
        "EPSG:4326", _utm_crs(104.0, 1.0), always_xy=True
    )
    p1 = to_zone.transform(99.0, 1.0)
    p2 = to_zone.transform(109.0, 1.0)
    _, _, d_geod = GEOD.inv(99.0, 1.0, 109.0, 1.0)
    scale = math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / d_geod
    assert abs(scale - 1.0) < 0.005


def test_buffer_land_utm_km_negative_buffer_raises() -> None:
    """A negative buffer is rejected before any geometry work."""
    with pytest.raises(ValueError, match="buffer_km must be >= 0"):
        buffer_land_utm_km(box(0.0, 0.0, 1.0, 1.0), -1.0, zone_lon=0.5, zone_lat=0.5)


# ---------------------------------------------------------------------------
# 6. Padded fetch band
# ---------------------------------------------------------------------------


def test_padded_fetch_band_neighbor_tile_regression() -> None:
    """Round-5 blocker: the padded band fetches the neighbor tile [90, 100)."""
    roi = (100.0, 0.5, 104.0, 2.0)
    band = padded_fetch_band(roi, 1.0, zone_lon=102.0, zone_lat=1.25)
    snapped = snap_band(band, 10.0)
    assert snapped[0] <= 90.0 + 1e-9

    # unpadded contrast: the raw ROI band stops exactly at the tile edge
    unpadded = snap_band(roi, 10.0)
    assert unpadded[0] == pytest.approx(100.0)

    # water inside the previously unfetched tile: buffered by 1 km it reaches
    # the ROI and the strip just east of lon 100 is masked on the 30 m grid.
    water = box(99.9985, 0.9, 99.9995, 1.1)
    buffered = buffer_land_utm_km(water, 1.0, zone_lon=102.0, zone_lat=1.25)
    px = 30.0 / 111_320.0
    transform = from_origin(100.0, 2.0, px, px)
    height = int(1.5 / px) + 1
    grid = rasterize_to_grid([buffered], transform, (height, 400))
    c1 = int((100.0085 - 100.0) / px)
    assert int(grid[:, 0:c1].sum()) > 0


def test_padded_fetch_band_padding_folds_into_identity_inputs() -> None:
    """The padded band carries the padding (cache-identity input); 0 km is id."""
    roi = (100.0, 0.5, 104.0, 2.0)
    zero = padded_fetch_band(roi, 0.0, zone_lon=102.0, zone_lat=1.25)
    assert zero == pytest.approx(roi, abs=1e-9)

    padded = padded_fetch_band(roi, 1.0, zone_lon=102.0, zone_lat=1.25)
    assert padded[0] < roi[0]
    assert padded[1] < roi[1]
    assert padded[2] > roi[2]
    assert padded[3] > roi[3]


def test_snap_band_tile_grid() -> None:
    """snap_band snaps outwards onto the tile grid (shared by guard+fetcher)."""
    assert snap_band((100.0, 0.5, 104.0, 2.0), 10.0) == (
        100.0,
        0.0,
        110.0,
        10.0,
    )
    assert snap_band((100.5, 0.5, 103.5, 1.5), 10.0) == (100.0, 0.0, 110.0, 10.0)
    assert snap_band((178.0, 8.0, 179.9, 10.0), 10.0)[2] == 180.0


# ---------------------------------------------------------------------------
# 7. Antimeridian seam guard (11 executed cases of round5_seam_guard.py)
# ---------------------------------------------------------------------------


SEAM_CASES: list[tuple[str, tuple[float, float, float, float], float, bool]] = [
    # (name, raw_bounds, tile_size_deg, expected_ok)
    ("normal_roi", (100.0, 30.0, 120.0, 40.0), 10.0, True),
    ("wrapped_evader_bbox", (-178.0, 8.0, 178.0, 10.0), 10.0, False),
    ("vertex_straddle", (-179.5, 8.0, 179.5, 9.0), 10.0, False),
    ("exact_180_hemisphere", (-90.0, 0.0, 90.0, 10.0), 10.0, True),
    ("near_seam_east_only_R4_crash", (178.0, 8.0, 179.9, 10.0), 10.0, False),
    ("near_seam_west_only", (-179.9, 8.0, -178.0, 10.0), 10.0, False),
    ("in_seam_tile_band", (165.0, 8.0, 175.0, 10.0), 10.0, False),
    ("tile_band_short_of_seam", (155.0, 8.0, 169.0, 10.0), 10.0, True),
    ("unwrapped_lon_0_360", (175.0, 8.0, 185.0, 10.0), 10.0, False),
    ("worldcover_tile_band", (177.5, 8.0, 179.0, 9.0), 3.0, False),
    ("worldcover_safe", (100.0, 30.0, 102.0, 32.0), 3.0, True),
]


def _snapped_padded_band(
    raw_bounds: tuple[float, float, float, float],
    tile_size_deg: float,
    buffer_km: float = 1.0,
) -> tuple[float, float, float, float]:
    """snap(padded_fetch_band(raw_bounds)) — the band handed to the guard."""
    min_lon, min_lat, max_lon, max_lat = raw_bounds
    band = padded_fetch_band(
        raw_bounds,
        buffer_km,
        zone_lon=(min_lon + max_lon) / 2.0,
        zone_lat=(min_lat + max_lat) / 2.0,
    )
    return snap_band(band, tile_size_deg)


@pytest.mark.parametrize(
    ("name", "raw_bounds", "tile_size_deg", "expected_ok"),
    SEAM_CASES,
    ids=[case[0] for case in SEAM_CASES],
)
def test_antimeridian_seam_guard(
    name: str,
    raw_bounds: tuple[float, float, float, float],
    tile_size_deg: float,
    expected_ok: bool,
) -> None:
    """The 11 executed seam-guard cases pass on the padded band."""
    del name
    band = _snapped_padded_band(raw_bounds, tile_size_deg)
    ok, reason = antimeridian_seam_guard(raw_bounds, padded_band=band)
    assert ok is expected_ok
    assert isinstance(reason, str)
    assert reason
    if not expected_ok:
        assert reason != "ok"


def test_antimeridian_seam_guard_reasons() -> None:
    """Each guard condition reports its own fail-closed reason."""
    # (c) unwrapped / out-of-range longitude
    ok, reason = antimeridian_seam_guard(
        (175.0, 8.0, 185.0, 10.0), padded_band=(0.0, 0.0, 1.0, 1.0)
    )
    assert not ok
    assert "out-of-range" in reason
    ok, reason = antimeridian_seam_guard(
        (-185.0, 8.0, -175.0, 10.0), padded_band=(0.0, 0.0, 1.0, 1.0)
    )
    assert not ok
    assert "out-of-range" in reason
    # (a) planar wrap evader
    ok, reason = antimeridian_seam_guard(
        (-178.0, 8.0, 178.0, 10.0), padded_band=(0.0, 0.0, 1.0, 1.0)
    )
    assert not ok
    assert "span" in reason
    # (b) padded band reaches the +/-180 seam (exactly and within tolerance)
    ok, reason = antimeridian_seam_guard(
        (100.0, 8.0, 101.0, 10.0), padded_band=(99.0, 0.0, 180.0, 20.0)
    )
    assert not ok
    assert "seam" in reason
    ok, reason = antimeridian_seam_guard(
        (100.0, 8.0, 101.0, 10.0), padded_band=(99.0, 0.0, 179.9999999995, 20.0)
    )
    assert not ok
    assert "seam" in reason
    # passing case
    ok, reason = antimeridian_seam_guard(
        (100.0, 8.0, 101.0, 10.0), padded_band=(99.0, 0.0, 102.0, 20.0)
    )
    assert ok
    assert reason == "ok"


def test_antimeridian_seam_guard_rejects_round4_crash_input() -> None:
    """The round-4 near-seam crash input is rejected before any buffering."""
    crash_bounds = (178.0, 8.0, 179.9, 10.0)
    band = _snapped_padded_band(crash_bounds, 10.0)
    ok, reason = antimeridian_seam_guard(crash_bounds, padded_band=band)
    assert ok is False
    assert "seam" in reason


# ---------------------------------------------------------------------------
# 8. boundary-is-None guard
# ---------------------------------------------------------------------------


def test_boundary_none_guard() -> None:
    """Shapely 2.x GeometryCollection.boundary is None never crashes buffers."""
    collection = GeometryCollection([box(0.0, 70.0, 0.05, 70.05)])
    assert collection.boundary is None
    buffered = buffer_land_utm_km(collection, 1.0, zone_lon=0.025, zone_lat=70.025)
    assert not buffered.is_empty

    empty = GeometryCollection([])
    assert empty.boundary is None
    assert buffer_land_utm_km(empty, 1.0, zone_lon=0.025, zone_lat=70.025).is_empty

    # Polygon / MultiPolygon paths buffer every part
    multi = MultiPolygon([box(0.0, 70.0, 0.05, 70.05), box(0.2, 70.0, 0.25, 70.05)])
    buffered_multi = buffer_land_utm_km(multi, 1.0, zone_lon=0.025, zone_lat=70.025)
    assert buffered_multi.area > multi.area

    # the guard also holds for point members inside a collection
    point_collection = GeometryCollection([box(0.0, 40.0, 0.5, 40.5), Point(1.0, 40.2)])
    buffered_points = buffer_land_utm_km(
        point_collection, 1.0, zone_lon=0.5, zone_lat=40.25
    )
    assert not buffered_points.is_empty
