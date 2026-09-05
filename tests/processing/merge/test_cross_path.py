"""Tests for cross-path merge: asc/desc isolation and multi-component handling."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from faninsar.processing.merge.grid import GeoGridSpec
from faninsar.processing.merge.overlap import compute_feather
from faninsar.processing.merge.mosaic import merge_burst_products
from faninsar.processing.merge.orchestration import run_multi_path_pair_merge
from faninsar.processing.merge.path_catalog import PathCatalog
from faninsar.processing.merge.products import BurstGeoProduct


def _grid() -> GeoGridSpec:
    return GeoGridSpec(
        crs="EPSG:32633",
        transform=(0.0, 10.0, 0.0, 1000.0, 0.0, -10.0),
        width=60,
        height=20,
        resolution_m=(10.0, 10.0),
    )


def _make_product(
    burst_id: str,
    path_id: str,
    look: str,
    phase_offset_rad: float,
    valid_slice: tuple[slice, slice],
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


def test_asc_desc_not_phase_linked_by_default() -> None:
    """Ascending and descending bursts share the grid but are not phase-linked."""
    p_asc = _make_product("a1", "T1_A", "ascending", 0.0, (slice(0, 20), slice(0, 30)))
    p_desc = _make_product("d1", "T1_D", "descending", 1.0, (slice(0, 20), slice(20, 60)))
    mosaic = merge_burst_products(
        [p_asc, p_desc],
        mode="phase_network",
        min_overlap_px=10,
        min_edge_coherence=0.0,
        allow_asc_desc_phase_link=False,
    )
    # Two components: ascending and descending are not phase-linked
    assert mosaic.component_id.max() >= 2
    # The overlap region (cols 20-29) still has both bursts contributing
    overlap = (p_asc.weight > 0) & (p_desc.weight > 0)
    assert overlap.any()
    # In the overlap, phases are NOT aligned (different components)
    # The mosaic complex in the overlap is a mix of unaligned phases.
    # Just check the mosaic is finite there.
    assert np.isfinite(mosaic.complex[overlap]).all()


def test_asc_desc_phase_linked_when_opt_in() -> None:
    """When allow_asc_desc_phase_link=True, asc/desc are phase-linked."""
    p_asc = _make_product("a1", "T1_A", "ascending", 0.0, (slice(0, 20), slice(0, 30)))
    p_desc = _make_product("d1", "T1_D", "descending", 0.5, (slice(0, 20), slice(20, 60)))
    mosaic = merge_burst_products(
        [p_asc, p_desc],
        mode="phase_network",
        min_overlap_px=10,
        min_edge_coherence=0.0,
        allow_asc_desc_phase_link=True,
    )
    # One component: ascending and descending are phase-linked
    assert mosaic.component_id.max() == 1


def test_cross_path_same_direction_links() -> None:
    """Two ascending paths with overlap form a single component."""
    p1 = _make_product("a1", "T1_A", "ascending", 0.0, (slice(0, 20), slice(0, 30)))
    p2 = _make_product("a2", "T2_A", "ascending", 0.3, (slice(0, 20), slice(20, 60)))
    mosaic = run_multi_path_pair_merge(
        [p1, p2],
        min_overlap_px=10,
        min_edge_coherence=0.0,
    )
    assert mosaic.component_id.max() == 1
    assert mosaic.network is not None
    assert mosaic.network.n_edges == 1


def test_cross_path_non_overlapping_two_components() -> None:
    """Two non-overlapping paths produce two separate components."""
    p1 = _make_product("a1", "T1_A", "ascending", 0.0, (slice(0, 20), slice(0, 10)))
    p2 = _make_product("a2", "T2_A", "ascending", 1.0, (slice(0, 20), slice(50, 60)))
    mosaic = run_multi_path_pair_merge(
        [p1, p2],
        min_overlap_px=10,
        min_edge_coherence=0.0,
    )
    assert mosaic.component_id.max() >= 2
    assert mosaic.network is not None


def test_path_catalog_cross_path_pairs() -> None:
    """PathCatalog.cross_path_pairs lists all path-id combinations."""
    products = [
        _make_product("a", "T1_A", "ascending", 0.0, (slice(0, 20), slice(0, 10))),
        _make_product("b", "T2_A", "ascending", 0.0, (slice(0, 20), slice(0, 10))),
        _make_product("c", "T3_A", "ascending", 0.0, (slice(0, 20), slice(0, 10))),
    ]
    catalog = PathCatalog.from_products(products)
    pairs = catalog.cross_path_pairs()
    assert ("T1_A", "T2_A") in pairs
    assert ("T1_A", "T3_A") in pairs
    assert ("T2_A", "T3_A") in pairs
    assert len(pairs) == 3


def test_same_path_only_policy_rejects_cross_path_edges() -> None:
    """same_path_only builds no edges between different paths."""
    p1 = _make_product("a1", "T1_A", "ascending", 0.0, (slice(0, 20), slice(0, 30)))
    p2 = _make_product("a2", "T2_A", "ascending", 0.3, (slice(0, 20), slice(20, 60)))
    mosaic = merge_burst_products(
        [p1, p2],
        mode="phase_network",
        min_overlap_px=10,
        min_edge_coherence=0.0,
        path_policy="same_path_only",
    )
    # No edges -> two components
    assert mosaic.component_id.max() >= 2
    assert mosaic.network is not None
    assert mosaic.network.n_edges == 0