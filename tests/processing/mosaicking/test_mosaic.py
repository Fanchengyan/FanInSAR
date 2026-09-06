"""Tests for the weighted mosaic and unwrap-input rejection."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from faninsar.processing.mosaicking.grid import GeoGridSpec
from faninsar.processing.mosaicking.mosaic import merge_burst_products
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
    phase_offset_rad: float,
    valid_slice: tuple[slice, slice],
    *,
    path_id: str = "T1_A",
    look: str = "ascending",
    phase_domain: str = "complex",
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
        phase_domain=phase_domain,  # type: ignore[arg-type]
    )


def test_merge_rejects_unwrapped_input_by_default() -> None:
    """allow_unwrapped_merge=False rejects unwrapped products."""
    p_i = _make_product("a", 0.0, (slice(0, 20), slice(0, 20)))
    p_j = _make_product(
        "b", 0.0, (slice(0, 20), slice(5, 25)), phase_domain="unwrapped"
    )
    with pytest.raises(ValueError, match="unwrapped"):
        merge_burst_products([p_i, p_j])


def test_merge_accepts_unwrapped_when_opt_in() -> None:
    """allow_unwrapped_merge=True permits unwrapped inputs with a warning."""
    p_i = _make_product("a", 0.0, (slice(0, 20), slice(0, 20)))
    p_j = _make_product(
        "b", 0.0, (slice(0, 20), slice(5, 25)), phase_domain="unwrapped"
    )
    mosaic = merge_burst_products(
        [p_i, p_j], allow_unwrapped_merge=True, mode="complex_average"
    )
    assert mosaic.complex.shape == _grid().shape


def test_merge_complex_average_preserves_phase_where_single_burst() -> None:
    """complex_average mode keeps the phase where only one burst contributes."""
    p_i = _make_product("a", 0.7, (slice(0, 20), slice(0, 16)))
    mosaic = merge_burst_products([p_i], mode="complex_average")
    # where valid, phase should be ~0.7
    mask = p_i.weight > 0
    phases = np.angle(mosaic.complex[mask])
    assert np.allclose(phases, 0.7, atol=1e-5)


def test_merge_phase_network_aligns_overlapping_bursts() -> None:
    """phase_network mode aligns overlapping bursts to a common phase."""
    p_i = _make_product("a", 0.0, (slice(0, 20), slice(0, 16)))
    p_j = _make_product("b", 0.5, (slice(0, 20), slice(10, 26)))
    mosaic = merge_burst_products(
        [p_i, p_j],
        mode="phase_network",
        min_overlap_px=10,
        min_edge_coherence=0.0,
    )
    # After alignment, both bursts should have ~same phase in the overlap.
    overlap = (p_i.weight > 0) & (p_j.weight > 0)
    phase_overlap = np.angle(mosaic.complex[overlap])
    # The overlap phase should be close to the reference phase (0.0)
    assert np.std(phase_overlap) < 1e-3


def test_merge_phase_network_vs_complex_average_overlap_coh() -> None:
    """phase_network yields higher overlap coherence than complex_average."""
    p_i = _make_product("a", 0.0, (slice(0, 20), slice(0, 20)))
    p_j = _make_product("b", 0.8, (slice(0, 20), slice(10, 30)))
    mosaic_avg = merge_burst_products(
        [p_i, p_j],
        mode="complex_average",
        min_overlap_px=10,
        min_edge_coherence=0.0,
    )
    mosaic_net = merge_burst_products(
        [p_i, p_j],
        mode="phase_network",
        min_overlap_px=10,
        min_edge_coherence=0.0,
    )
    overlap = (p_i.weight > 0) & (p_j.weight > 0)
    # |mean of aligned complex| should be higher for the network mosaic
    mag_avg = np.abs(mosaic_avg.complex[overlap]).mean()
    mag_net = np.abs(mosaic_net.complex[overlap]).mean()
    assert mag_net > mag_avg


def test_merge_outputs_component_id_and_weight_sum() -> None:
    """MosaicProduct carries component_id, weight_sum, and n_bursts."""
    p_i = _make_product("a", 0.0, (slice(0, 20), slice(0, 16)))
    p_j = _make_product("b", 0.5, (slice(0, 20), slice(10, 26)))
    mosaic = merge_burst_products(
        [p_i, p_j],
        mode="phase_network",
        min_overlap_px=10,
        min_edge_coherence=0.0,
    )
    assert mosaic.component_id.shape == _grid().shape
    assert mosaic.weight_sum.shape == _grid().shape
    assert mosaic.n_bursts.shape == _grid().shape
    assert mosaic.n_bursts.dtype == np.uint8
    # in the overlap, n_bursts == 2
    overlap = (p_i.weight > 0) & (p_j.weight > 0)
    assert np.all(mosaic.n_bursts[overlap] == 2)


def test_merge_disjoint_bursts_produce_two_components() -> None:
    """Two non-overlapping bursts are separate components, no error."""
    p_i = _make_product("a", 0.0, (slice(0, 20), slice(0, 10)))
    p_j = _make_product("b", 1.0, (slice(0, 20), slice(30, 40)))
    mosaic = merge_burst_products(
        [p_i, p_j],
        mode="phase_network",
        min_overlap_px=10,
        min_edge_coherence=0.0,
    )
    assert mosaic.component_id.max() >= 1
    assert mosaic.n_bursts.sum() > 0


def test_merge_nodata_where_no_burst_contributes() -> None:
    """Pixels with no contributing burst are zero in the mosaic."""
    p_i = _make_product("a", 0.0, (slice(0, 20), slice(0, 10)))
    mosaic = merge_burst_products([p_i], mode="complex_average")
    assert mosaic.complex[19, 39] == 0.0
    assert mosaic.weight_sum[19, 39] == 0.0
