"""Tests for the multi-source DEM registry (PROPOSAL-0030)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from faninsar.processing.geometry.dem_sources import (
    AUTO_SOURCE_NAME,
    get_dem_source,
    list_dem_sources,
)
from faninsar.query import BoundingBox


def _bounds(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> BoundingBox:
    """Build a geographic BoundingBox."""
    return BoundingBox(min_lon, min_lat, max_lon, max_lat)


def test_registry_metadata_for_wired_sources() -> None:
    """The five wired sources expose the pinned registry metadata."""
    glo30 = get_dem_source("copernicus-30")
    assert glo30.resolution_m == pytest.approx(1 / 3600)
    assert glo30.vertical_datum == "egm2008"
    assert glo30.derived is False
    assert glo30.product_kind == "dsm"
    assert glo30.auth == "none"
    assert glo30.layout_id == "copernicus-cog-stem"
    assert glo30.default_base_url == "https://copernicus-dem-30m.s3.amazonaws.com"

    glo90 = get_dem_source("copernicus-90")
    assert glo90.resolution_m == pytest.approx(1 / 1200)
    assert glo90.vertical_datum == "egm2008"
    assert glo90.derived is False
    assert glo90.product_kind == "dsm"
    assert glo90.auth == "none"
    assert glo90.layout_id == "copernicus-cog-stem"
    assert glo90.default_base_url == "https://copernicus-dem-90m.s3.amazonaws.com"

    skadi = get_dem_source("srtm-skadi")
    assert skadi.resolution_m == pytest.approx(1 / 3600)
    assert skadi.vertical_datum == "egm96"
    assert skadi.derived is False
    assert skadi.product_kind == "dsm"
    assert skadi.auth == "none"
    assert skadi.layout_id == "skadi-hgt-gz"
    assert skadi.default_base_url == "https://elevation-tiles-prod.s3.amazonaws.com"

    terrain = get_dem_source("terrain-tiles")
    assert terrain.resolution_m == pytest.approx(360.0 / (256 * 2**12))
    assert terrain.vertical_datum == "mixed-derived"
    assert terrain.derived is True
    assert terrain.product_kind == "merged-derived"
    assert terrain.auth == "none"
    assert terrain.layout_id == "terrain-zxy"
    assert terrain.default_base_url == "https://elevation-tiles-prod.s3.amazonaws.com"


def test_product_group_modifiers_for_wired_sources() -> None:
    """Method/modifier product-group fields pin the v1 values."""
    expectations = {
        "copernicus-30": ("dsm", "radar-interferometric"),
        "copernicus-90": ("dsm", "radar-interferometric"),
        AUTO_SOURCE_NAME: ("dsm", "radar-interferometric"),
        "srtm-skadi": ("dsm", "radar-interferometric"),
        "terrain-tiles": ("merged-derived", "composite"),
    }
    for name, (kind, method) in expectations.items():
        source = get_dem_source(name)
        assert source.product_kind == kind, name
        assert source.method == method, name
        assert source.hydro_conditioned is False, name
        assert source.void_filled is False, name
        assert source.auth == "none", name
    # dtm / topo-bathy are reserved for future registry entries.
    assert "dtm" not in {get_dem_source(n).product_kind for n in list_dem_sources()}
    assert "topo-bathy" not in {
        get_dem_source(n).product_kind for n in list_dem_sources()
    }


def test_auto_source_is_selection_only() -> None:
    """Auto is a selection-level source without its own transport layout."""
    auto = get_dem_source(AUTO_SOURCE_NAME)
    assert auto.name == "auto"
    assert auto.fallback == "copernicus-90"
    bounds = tiles_bounds()
    tiles = auto.tiles(bounds)
    assert [tile.remote_url for tile in tiles] == [
        tile.remote_url for tile in get_dem_source("copernicus-30").tiles(bounds)
    ]


def tiles_bounds() -> BoundingBox:
    """Bounds shared by the auto/comparison enumeration assertion."""
    return _bounds(45.2, 42.8, 45.4, 42.9)


def test_list_dem_sources_returns_all_five() -> None:
    """list_dem_sources exposes exactly the five wired names."""
    assert set(list_dem_sources()) == {
        "copernicus-30",
        "copernicus-90",
        "auto",
        "srtm-skadi",
        "terrain-tiles",
    }


def test_get_dem_source_fail_closed_lists_valid_names() -> None:
    """An unknown source name fails closed listing the valid names."""
    with pytest.raises(ValueError, match="nasadem") as excinfo:
        get_dem_source("nasadem")
    message = str(excinfo.value)
    for name in (
        "copernicus-30",
        "copernicus-90",
        "auto",
        "srtm-skadi",
        "terrain-tiles",
    ):
        assert name in message


def test_glo30_tile_url_layout() -> None:
    """GLO-30 keeps the PROPOSAL-0013 stem layout and per-tile URL shape."""
    glo30 = get_dem_source("copernicus-30")
    tiles = glo30.tiles(_bounds(100.2, 38.2, 100.4, 38.4))
    assert len(tiles) == 1
    tile = tiles[0]
    assert tile.cache_relative_path == (
        "N38_E100/Copernicus_DSM_COG_10_N38_00_E100_00_DEM/"
        "Copernicus_DSM_COG_10_N38_00_E100_00_DEM.tif"
    )
    # Remote S3 key: stem/stem.tif directly under the bucket root.
    assert tile.remote_url == (
        "https://copernicus-dem-30m.s3.amazonaws.com/"
        "Copernicus_DSM_COG_10_N38_00_E100_00_DEM/"
        "Copernicus_DSM_COG_10_N38_00_E100_00_DEM.tif"
    )
    assert tile.minimum_bytes >= 1 << 20


def test_glo90_tile_url_stem_at_bucket_root() -> None:
    """GLO-90 mirrors the GLO-30 COG stem at the bucket root (no dir prefix).

    Regression pin for the R1 feasibility finding: the withheld cell N38E045
    exists on copernicus-dem-90m as Copernicus_DSM_COG_30_... directly under
    the bucket root.
    """
    glo90 = get_dem_source("copernicus-90")
    tiles = glo90.tiles(_bounds(45.2, 38.2, 45.4, 38.4))
    assert len(tiles) == 1
    tile = tiles[0]
    assert tile.cache_relative_path == (
        "N38_E045/Copernicus_DSM_COG_30_N38_00_E045_00_DEM/"
        "Copernicus_DSM_COG_30_N38_00_E045_00_DEM.tif"
    )
    assert tile.remote_url == (
        "https://copernicus-dem-90m.s3.amazonaws.com/"
        "Copernicus_DSM_COG_30_N38_00_E045_00_DEM/"
        "Copernicus_DSM_COG_30_N38_00_E045_00_DEM.tif"
    )
    assert "COG_30/" not in tile.remote_url


def test_skadi_tile_url_layout() -> None:
    """Skadi enumerates skadi/{NS}{YY}/{NS}{YY}{EW}{XXX}.hgt.gz paths."""
    skadi = get_dem_source("srtm-skadi")
    tiles = skadi.tiles(_bounds(94.2, 34.2, 94.4, 34.4))
    assert len(tiles) == 1
    tile = tiles[0]
    assert tile.cache_relative_path == "skadi/N34/N34E094.hgt.gz"
    assert tile.remote_url == (
        "https://elevation-tiles-prod.s3.amazonaws.com/skadi/N34/N34E094.hgt.gz"
    )
    # An all-void ocean 1-degree tile gzips to tens of KB; floor stays small.
    assert tile.minimum_bytes < 1 << 20
    assert tile.expected_decompressed_bytes == 3601 * 3601 * 2
    assert tile.raster_open_path(Path("cache") / tile.cache_relative_path) == (
        f"/vsigzip/{Path('cache') / 'skadi' / 'N34' / 'N34E094.hgt.gz'}"
    )


def test_terrain_tiles_z12_xyz_orientation() -> None:
    """terrain-tiles pins XYZ z12 orientation at a known cell.

    Madrid (40.5 N, 3.5 W) at zoom 12 must be x=2008, y=1543 in slippy-map
    XYZ orientation (y measured from the top/north pole).
    """
    terrain = get_dem_source("terrain-tiles")
    tiles = terrain.tiles(_bounds(-3.51, 40.49, -3.50, 40.50))
    assert len(tiles) == 1
    tile = tiles[0]
    assert tile.cache_relative_path == "geotiff/12/2008/1543.tif"
    assert tile.remote_url == (
        "https://elevation-tiles-prod.s3.amazonaws.com/geotiff/12/2008/1543.tif"
    )
    assert tile.minimum_bytes > 1024


def test_tile_enumeration_degree_boundaries() -> None:
    """Enumeration includes every intersecting cell including boundaries."""
    glo30 = get_dem_source("copernicus-30")
    tiles = glo30.tiles(_bounds(99.9, 37.9, 101.1, 39.1))
    names = {tile.cache_relative_path for tile in tiles}
    assert names == {
        "N37_E099/Copernicus_DSM_COG_10_N37_00_E099_00_DEM/"
        "Copernicus_DSM_COG_10_N37_00_E099_00_DEM.tif",
        "N37_E100/Copernicus_DSM_COG_10_N37_00_E100_00_DEM/"
        "Copernicus_DSM_COG_10_N37_00_E100_00_DEM.tif",
        "N37_E101/Copernicus_DSM_COG_10_N37_00_E101_00_DEM/"
        "Copernicus_DSM_COG_10_N37_00_E101_00_DEM.tif",
        "N38_E099/Copernicus_DSM_COG_10_N38_00_E099_00_DEM/"
        "Copernicus_DSM_COG_10_N38_00_E099_00_DEM.tif",
        "N38_E100/Copernicus_DSM_COG_10_N38_00_E100_00_DEM/"
        "Copernicus_DSM_COG_10_N38_00_E100_00_DEM.tif",
        "N38_E101/Copernicus_DSM_COG_10_N38_00_E101_00_DEM/"
        "Copernicus_DSM_COG_10_N38_00_E101_00_DEM.tif",
        "N39_E099/Copernicus_DSM_COG_10_N39_00_E099_00_DEM/"
        "Copernicus_DSM_COG_10_N39_00_E099_00_DEM.tif",
        "N39_E100/Copernicus_DSM_COG_10_N39_00_E100_00_DEM/"
        "Copernicus_DSM_COG_10_N39_00_E100_00_DEM.tif",
        "N39_E101/Copernicus_DSM_COG_10_N39_00_E101_00_DEM/"
        "Copernicus_DSM_COG_10_N39_00_E101_00_DEM.tif",
    }
    southern = get_dem_source("copernicus-90").tiles(_bounds(-71.5, -3.5, -70.5, -2.5))
    assert any("S03_W072" in tile.cache_relative_path for tile in southern)


def test_coverage_fail_closed_outside_skadi_latitude() -> None:
    """Skadi coverage fails closed outside 56 S to 60 N."""
    skadi = get_dem_source("srtm-skadi")
    assert skadi.coverage(_bounds(94.2, 34.2, 94.4, 34.4)) is None
    inside_message = skadi.coverage(_bounds(94.2, 59.9, 94.4, 60.1))
    assert inside_message is not None
    assert "60" in inside_message or "latitude" in inside_message.lower()
    south_message = skadi.coverage(_bounds(10.0, -57.0, 11.0, -56.5))
    assert south_message is not None


def test_base_url_override_rejects_non_https() -> None:
    """Non-https base overrides are rejected for every concrete source."""

    class _Recorder:
        def __init__(self) -> None:
            self.warnings: list[str] = []

    recorder = _Recorder()
    for name in ("copernicus-30", "copernicus-90", "srtm-skadi", "terrain-tiles"):
        source: Any = get_dem_source(name)
        with pytest.raises(ValueError, match="https"):
            source.with_base_url(
                "http://mirror.example.test", warn=recorder.warnings.append
            )
        overridden = source.with_base_url(
            "https://mirror.example.test", warn=recorder.warnings.append
        )
        # The override binds to the copy; the registry entry is untouched.
        assert overridden.base_url == "https://mirror.example.test"
        assert get_dem_source(name).default_base_url == source.default_base_url
        assert recorder.warnings


def test_transport_boundary_guard_rejects_traversal_paths(tmp_path: Path) -> None:
    """Cache-relative paths containing .. or absolute components are rejected."""
    from faninsar.processing.geometry.dem_sources import validate_cache_relative_path

    validate_cache_relative_path("skadi/N34/N34E094.hgt.gz", tmp_path)
    for bad in ("../escape.tif", "/abs/path.tif", "a/../../b.tif"):
        with pytest.raises(ValueError, match=r"\.\.|absolute"):
            validate_cache_relative_path(bad, tmp_path)


def test_auto_falls_back_to_copernicus_90_per_cell() -> None:
    """The auto selection enumerates primary tiles carrying a fallback marker."""
    auto = get_dem_source(AUTO_SOURCE_NAME)
    bounds = _bounds(45.2, 38.2, 45.4, 38.4)
    primary_tiles = get_dem_source("copernicus-30").tiles(bounds)
    fallback_tiles = get_dem_source("copernicus-90").tiles(bounds)
    assert auto.fallback == "copernicus-90"
    assert [t.remote_url for t in primary_tiles] != [
        t.remote_url for t in fallback_tiles
    ]
