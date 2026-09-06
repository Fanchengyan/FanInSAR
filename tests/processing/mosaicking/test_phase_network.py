"""Tests for the burst overlap phase network and least-squares adjustment."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from faninsar.processing.mosaicking.grid import GeoGridSpec
from faninsar.processing.mosaicking.overlap import compute_feather
from faninsar.processing.mosaicking.phase_network import (
    MergeGraph,
    estimate_edge,
    estimate_edges,
    solve_network,
)
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
    coherence_val: float = 1.0,
) -> BurstGeoProduct:
    """Build a synthetic burst product with a constant phase offset."""
    g = _grid()
    h, w = g.shape
    z = np.zeros((h, w), dtype=np.complex64)
    mask = np.zeros((h, w), dtype=bool)
    mask[valid_slice] = True
    z[mask] = np.exp(1j * phase_offset_rad, dtype=np.complex64)
    weight = compute_feather(mask, feather_width_px=0.0)
    coh = np.full((h, w), coherence_val, dtype=np.float32)
    coh[~mask] = 0.0
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


def test_estimate_edge_recovers_known_phase_offset() -> None:
    """estimate_edge recovers a constant phase offset between two bursts.

    The plan convention is Δφ_ij = arg(Σ w_i w_j z_i z_j*) = φ_i - φ_j.
    """
    p_i = _make_product("a", 0.3, (slice(0, 20), slice(0, 20)))
    p_j = _make_product("b", 0.9, (slice(0, 20), slice(5, 25)))
    edge = estimate_edge(p_i, p_j, min_overlap_px=10, min_edge_coherence=0.0)
    assert edge is not None
    # Δφ = φ_i - φ_j = 0.3 - 0.9 = -0.6
    assert edge.dphi_rad == pytest.approx(-0.6, abs=1e-3)
    assert edge.coherence > 0.99
    assert edge.overlap_px >= 10


def test_estimate_edge_returns_none_when_overlap_too_small() -> None:
    """No edge when overlap below min_overlap_px."""
    p_i = _make_product("a", 0.0, (slice(0, 20), slice(0, 5)))
    p_j = _make_product("b", 0.0, (slice(0, 20), slice(35, 40)))
    edge = estimate_edge(p_i, p_j, min_overlap_px=100, min_edge_coherence=0.0)
    assert edge is None


def test_estimate_edges_builds_chain() -> None:
    """estimate_edges builds edges for a 3-burst azimuth chain."""
    products = [
        _make_product("a", 0.0, (slice(0, 20), slice(0, 16))),
        _make_product("b", 0.5, (slice(0, 20), slice(10, 26))),
        _make_product("c", 1.0, (slice(0, 20), slice(20, 36))),
    ]
    graph = estimate_edges(products, min_overlap_px=10, min_edge_coherence=0.0)
    assert isinstance(graph, MergeGraph)
    assert len(graph.nodes) == 3
    # a-b and b-c overlap, a-c do not
    edge_keys = {frozenset((e.i, e.j)) for e in graph.edges}
    assert frozenset((0, 1)) in edge_keys
    assert frozenset((1, 2)) in edge_keys
    assert frozenset((0, 2)) not in edge_keys


def test_solve_network_recovers_phases_to_1e5() -> None:
    """LS adjustment recovers per-burst phases from synthetic Δφ edges."""
    # 3-node chain with known phases
    products = [
        _make_product("a", 0.0, (slice(0, 20), slice(0, 16))),
        _make_product("b", 0.5, (slice(0, 20), slice(10, 26))),
        _make_product("c", 1.0, (slice(0, 20), slice(20, 36))),
    ]
    graph = estimate_edges(products, min_overlap_px=10, min_edge_coherence=0.0)
    result = solve_network(graph, reference_node=0)
    # φ_hat relative to ref=0; expected ~ [0, 0.5, 1.0] rad
    assert result.phi_hat[0] == pytest.approx(0.0, abs=1e-5)
    assert result.phi_hat[1] == pytest.approx(0.5, abs=1e-3)
    assert result.phi_hat[2] == pytest.approx(1.0, abs=1e-3)
    assert result.rms_residual < 1e-3


def test_solve_network_disconnected_components_independent() -> None:
    """Two disconnected components each get their own reference."""
    products = [
        _make_product("a", 0.0, (slice(0, 20), slice(0, 10))),
        _make_product("b", 0.4, (slice(0, 20), slice(4, 14))),
        _make_product("c", 2.0, (slice(0, 20), slice(30, 40))),
    ]
    graph = estimate_edges(products, min_overlap_px=10, min_edge_coherence=0.0)
    # a-b connected, c isolated
    result = solve_network(graph, reference_node=0)
    assert len(set(result.component_id)) >= 2
    # isolated node has phi_hat = 0 (its own reference)
    c_index = [n.burst_id for n in graph.nodes].index("c")
    assert result.phi_hat[c_index] == pytest.approx(0.0, abs=1e-6)


def test_solve_network_residuals_reported() -> None:
    """Per-edge residuals are reported."""
    products = [
        _make_product("a", 0.0, (slice(0, 20), slice(0, 16))),
        _make_product("b", 0.5, (slice(0, 20), slice(10, 26))),
    ]
    graph = estimate_edges(products, min_overlap_px=10, min_edge_coherence=0.0)
    result = solve_network(graph, reference_node=0)
    assert result.edge_residuals is not None
    assert len(result.edge_residuals) == len(graph.edges)
    assert result.rms_residual >= 0.0
    assert result.range_slope_hat.shape == result.phi_hat.shape


def _make_product_with_range_ramp(
    burst_id: str,
    phase0: float,
    slope: float,
    valid_slice: tuple[slice, slice],
) -> BurstGeoProduct:
    """Burst product with φ = phase0 + slope * column."""
    g = _grid()
    h, w = g.shape
    z = np.zeros((h, w), dtype=np.complex64)
    mask = np.zeros((h, w), dtype=bool)
    mask[valid_slice] = True
    cols = np.arange(w, dtype=np.float64)[None, :]
    phase = np.broadcast_to(phase0 + slope * cols, (h, w)).copy()
    z[mask] = np.exp(1j * phase[mask]).astype(np.complex64)
    weight = compute_feather(mask, feather_width_px=0.0)
    coh = np.full((h, w), 1.0, dtype=np.float32)
    coh[~mask] = 0.0
    return BurstGeoProduct(
        burst_id=burst_id,
        path_id="T1_A",
        swath="IW1",
        date=date(2024, 1, 1),
        grid=g,
        complex=z,
        weight=weight,
        coherence=coh,
        look_direction="ascending",
    )


def test_estimate_edge_recovers_range_slope() -> None:
    """Column-wise fit recovers a known residual range slope between bursts."""
    # Relative: i has slope 0, j has slope 0.02 → edge slope = 0 - 0.02 = -0.02
    p_i = _make_product_with_range_ramp("a", 0.0, 0.0, (slice(0, 20), slice(0, 25)))
    p_j = _make_product_with_range_ramp("b", 0.5, 0.02, (slice(0, 20), slice(10, 35)))
    edge = estimate_edge(p_i, p_j, min_overlap_px=10, min_edge_coherence=0.0)
    assert edge is not None
    assert edge.range_slope_rad_per_px == pytest.approx(-0.02, abs=2e-3)


def test_solve_network_and_mosaic_remove_range_seam() -> None:
    """Range-linear network + mosaic removes a cross-swath range ramp seam.

    Regression for fan_radar east IW2–IW3 stitch jump: constant-only Δφ
    leaves a multi-radian range residual; slope unknowns absorb it.
    """
    from faninsar.processing.mosaicking.mosaic import merge_burst_products

    # Two side-by-side "swaths" with a relative range ramp in the overlap.
    left = _make_product_with_range_ramp(
        "IW2", 0.0, 0.0, (slice(0, 20), slice(0, 22))
    )
    right = _make_product_with_range_ramp(
        "IW3", 0.3, 0.025, (slice(0, 20), slice(12, 40))
    )
    mosaic = merge_burst_products(
        [left, right],
        mode="phase_network",
        min_overlap_px=10,
        min_edge_coherence=0.0,
    )
    z = mosaic.complex
    # In the right-only region the residual range ramp should be gone
    # (aligned to left's zero slope / zero phase).
    right_only = z[:, 30:38]
    valid = np.abs(right_only) > 0.5
    assert int(valid.sum()) > 20
    phases = np.angle(right_only[valid])
    # After alignment, phases should be near zero (left reference).
    assert float(np.std(phases)) < 0.15
    assert abs(float(np.mean(phases))) < 0.25
    # Overlap should not show a large left-right step.
    mid = z[:, 14:20]
    mid_ok = np.abs(mid) > 0.5
    assert float(np.std(np.angle(mid[mid_ok]))) < 0.2


def test_edge_dphi_converted_to_frame_centre_with_slope() -> None:
    """dphi_rad is the intercept at frame centre when a range slope is fit.

    Circular mean over the overlap is the value near x0; mosaic applies
    slope*(x - x_ref).  Without converting dphi to x_ref, residual seam
    offsets of slope*(x0 - x_ref) remain (full-frame multi-radian jumps).
    """
    # Width 40 → x_ref = 19.5.  Overlap ~ cols 10..25 → x0 ~ 17.5.
    # Relative slope -0.02 → correction slope*(x_ref - x0) ≈ -0.04.
    p_i = _make_product_with_range_ramp("a", 0.0, 0.0, (slice(0, 20), slice(0, 25)))
    p_j = _make_product_with_range_ramp("b", 0.0, 0.02, (slice(0, 20), slice(10, 35)))
    edge = estimate_edge(p_i, p_j, min_overlap_px=10, min_edge_coherence=0.0)
    assert edge is not None
    assert edge.range_slope_rad_per_px == pytest.approx(-0.02, abs=2e-3)
    # At frame centre the pure-ramp Δφ should be slope*(x_ref - 0) for i
    # minus slope_j*(x_ref) ... relative b=-0.02, zero constant at x=0 on both
    # → Δφ(x)= -0.02*x, so at x_ref=19.5 → dphi_ref ≈ -0.39.
    x_ref = 0.5 * (40 - 1)
    expected = -0.02 * x_ref
    assert edge.dphi_rad == pytest.approx(float(np.angle(np.exp(1j * expected))), abs=0.08)
