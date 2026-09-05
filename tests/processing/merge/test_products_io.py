"""Tests for the merge pipeline wiring and Zarr/STAC writers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import zarr

from faninsar.processing.merge.grid import GeoGridSpec
from faninsar.processing.merge.overlap import compute_feather
from faninsar.processing.merge.path_catalog import PathCatalog
from faninsar.processing.merge.orchestration import (
    run_multi_burst_pair_merge,
    write_mosaic_zarr,
)
from faninsar.processing.merge.products import BurstGeoProduct, MosaicProduct


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
    phase_offset_rad: float,
    valid_slice: tuple[slice, slice],
    *,
    path_id: str = "T1_A",
    look: str = "ascending",
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
        swath="IW1",
        date=date(2024, 1, 1),
        grid=g,
        complex=z,
        weight=weight,
        coherence=coh,
        look_direction=look,
    )


def test_write_mosaic_zarr_round_trip(tmp_path: Path) -> None:
    """write_mosaic_zarr writes arrays and grid metadata."""
    p_i = _make_product("a", 0.0, (slice(0, 20), slice(0, 20)))
    p_j = _make_product("b", 0.5, (slice(0, 20), slice(10, 30)))
    products = [p_i, p_j]
    from faninsar.processing.merge.mosaic import merge_burst_products

    mosaic = merge_burst_products(
        products, mode="phase_network", min_overlap_px=10, min_edge_coherence=0.0
    )
    out = write_mosaic_zarr(mosaic, tmp_path / "mosaic.zarr", pair_id="test_pair")
    assert out.exists()
    root = zarr.open_group(str(out), mode="r")
    assert "complex_ifg" in root
    assert "coherence" in root
    assert "weight_sum" in root
    assert "n_bursts" in root
    assert "component_id" in root
    assert root.attrs["pair_id"] == "test_pair"
    assert root.attrs["merge_domain"] == "complex"


def test_run_multi_burst_pair_merge_returns_mosaic() -> None:
    """run_multi_burst_pair_merge wires geocode → merge for in-memory products."""
    p_i = _make_product("a", 0.0, (slice(0, 20), slice(0, 16)))
    p_j = _make_product("b", 0.5, (slice(0, 20), slice(10, 26)))
    mosaic = run_multi_burst_pair_merge(
        [p_i, p_j],
        mode="phase_network",
        min_overlap_px=10,
        min_edge_coherence=0.0,
    )
    assert isinstance(mosaic, MosaicProduct)
    assert mosaic.complex.shape == _grid().shape
    assert mosaic.network is not None


def test_path_catalog_groups_by_path_id() -> None:
    """PathCatalog groups burst products by relative orbit + asc/desc."""
    products = [
        _make_product("a1", 0.0, (slice(0, 20), slice(0, 10)), path_id="T1_A"),
        _make_product("a2", 0.0, (slice(0, 20), slice(5, 15)), path_id="T1_A"),
        _make_product("b1", 0.0, (slice(0, 20), slice(0, 10)), path_id="T2_A"),
        _make_product("c1", 0.0, (slice(0, 20), slice(0, 10)), path_id="T1_D"),
    ]
    catalog = PathCatalog.from_products(products)
    assert "T1_A" in catalog.paths
    assert "T2_A" in catalog.paths
    assert "T1_D" in catalog.paths
    assert len(catalog.paths["T1_A"]) == 2
    assert len(catalog.paths["T2_A"]) == 1


def test_path_catalog_same_path_only_filter() -> None:
    """PathCatalog.same_path returns only products for one path."""
    products = [
        _make_product("a1", 0.0, (slice(0, 20), slice(0, 10)), path_id="T1_A"),
        _make_product("b1", 0.0, (slice(0, 20), slice(0, 10)), path_id="T2_A"),
    ]
    catalog = PathCatalog.from_products(products)
    same = catalog.same_path("T1_A")
    assert len(same) == 1
    assert same[0].path_id == "T1_A"


def test_path_catalog_look_direction_groups() -> None:
    """PathCatalog separates ascending and descending within a path."""
    products = [
        _make_product("a1", 0.0, (slice(0, 20), slice(0, 10)), path_id="T1_A", look="ascending"),
        _make_product("d1", 0.0, (slice(0, 20), slice(0, 10)), path_id="T1_D", look="descending"),
    ]
    catalog = PathCatalog.from_products(products)
    asc = catalog.by_look_direction("ascending")
    desc = catalog.by_look_direction("descending")
    assert all(p.look_direction == "ascending" for p in asc)
    assert all(p.look_direction == "descending" for p in desc)
    assert len(asc) == 1
    assert len(desc) == 1
