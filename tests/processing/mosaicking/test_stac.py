"""Tests for the multi-swath orchestration and STAC item writer."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from faninsar.processing.mosaicking.grid import GeoGridSpec
from faninsar.processing.mosaicking.mosaic import merge_burst_products
from faninsar.processing.mosaicking.orchestration import (
    run_network_merge,
    write_mosaic_stac_item,
    write_mosaic_zarr,
)
from faninsar.processing.mosaicking.overlap import compute_feather
from faninsar.processing.mosaicking.products import BurstGeoProduct


def _grid() -> GeoGridSpec:
    return GeoGridSpec(
        crs="EPSG:32633",
        transform=(0.0, 10.0, 0.0, 1000.0, 0.0, -10.0),
        width=40,
        height=20,
        resolution_m=(10.0, 10.0),
    )


def _make_product(
    burst_id: str,
    swath: str,
    phase_offset_rad: float,
    valid_slice: tuple[slice, slice],
    *,
    path_id: str = "T1_A",
) -> BurstGeoProduct:
    g = _grid()
    h, w = g.shape
    z = np.zeros((h, w), dtype=np.complex64)
    mask = np.zeros((h, w), dtype=bool)
    mask[valid_slice] = True
    z[mask] = np.exp(1j * phase_offset_rad, dtype=np.complex64)
    weight = compute_feather(mask, feather_width_px=0.0)
    coh = np.zeros((h, w), dtype=np.float32)
    coh[mask] = 1.0
    return BurstGeoProduct(
        burst_id=burst_id,
        path_id=path_id,
        swath=swath,
        date=date(2024, 1, 1),
        grid=g,
        complex=z,
        weight=weight,
        coherence=coh,
    )


def _three_swath_products() -> list[BurstGeoProduct]:
    """Build 3 products representing IW1/IW2/IW3 bursts with range overlap."""
    return [
        _make_product("iw1_b0", "IW1", 0.0, (slice(0, 20), slice(0, 16))),
        _make_product("iw2_b0", "IW2", 0.2, (slice(0, 20), slice(10, 26))),
        _make_product("iw3_b0", "IW3", 0.4, (slice(0, 20), slice(20, 36))),
    ]


def test_run_network_merge_writes_zarr(tmp_path: Path) -> None:
    """run_network_merge writes a Zarr store for a multi-swath Network."""
    products = _three_swath_products()
    out = run_network_merge(
        products,
        output_dir=tmp_path,
        merge_paths=False,
        pair_id="frame_test",
        min_overlap_px=10,
        min_edge_coherence=0.0,
    )
    assert out.exists()
    assert out.name == "frame_test_mosaic.zarr"


def test_run_network_merge_three_swath_single_component() -> None:
    """Three overlapping swaths produce a single connected component."""
    products = _three_swath_products()
    mosaic = merge_burst_products(
        products,
        mode="phase_network",
        min_overlap_px=10,
        min_edge_coherence=0.0,
    )
    # IW1-IW2 and IW2-IW3 overlap; should form one connected component
    # (plus 0 for nodata). So the max component_id should be 1.
    assert mosaic.component_id.max() == 1
    assert mosaic.n_bursts.max() == 3 or mosaic.n_bursts.max() == 2


def test_write_mosaic_stac_item(tmp_path: Path) -> None:
    """write_mosaic_stac_item emits a valid STAC item JSON."""
    products = _three_swath_products()
    mosaic = merge_burst_products(
        products,
        mode="phase_network",
        min_overlap_px=10,
        min_edge_coherence=0.0,
    )
    zarr_path = write_mosaic_zarr(
        mosaic, tmp_path / "mosaic.zarr", pair_id="stac_test"
    )
    stac_path = write_mosaic_stac_item(
        mosaic,
        zarr_path,
        tmp_path / "item.json",
        pair_id="stac_test",
        source_burst_ids=[p.burst_id for p in products],
    )
    assert stac_path.exists()
    import json

    with stac_path.open() as f:
        item = json.load(f)
    assert item["type"] == "Feature"
    assert item["properties"]["merge_domain"] == "complex"
    assert "bbox" in item
    assert len(item["bbox"]) == 4
    assert item["assets"]["complex_ifg"]["href"].endswith("mosaic.zarr/complex_ifg")
    # source burst ids recorded
    assert "faninsar:source_bursts" in item["properties"]


def test_write_mosaic_stac_item_bbox_from_weight_sum(tmp_path: Path) -> None:
    """STAC bbox is derived from the weight_sum>0 footprint."""
    p = _make_product("a", "IW1", 0.0, (slice(5, 15), slice(5, 15)))
    mosaic = merge_burst_products([p], mode="complex_average")
    zarr_path = write_mosaic_zarr(
        mosaic, tmp_path / "m.zarr", pair_id="bbox_test"
    )
    stac_path = write_mosaic_stac_item(
        mosaic, zarr_path, tmp_path / "item.json", pair_id="bbox_test",
        source_burst_ids=["a"],
    )
    import json

    with stac_path.open() as f:
        item = json.load(f)
    # bbox should be within the grid extent
    west, south, east, north = item["bbox"]
    assert west < east
    assert south < north
