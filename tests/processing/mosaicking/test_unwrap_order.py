"""Regression test: post-unwrap merge residual ≫ complex merge residual.

This locks the plan §4.3 timing rule: merging per-burst unwrapped phases
is fragile because independent 2π ambiguities inject integer-cycle jumps
into the overlap Δψ estimate, while complex-domain Δφ is invariant to
per-burst global phase constants and therefore stable.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from faninsar.processing.mosaicking.grid import GeoGridSpec
from faninsar.processing.mosaicking.mosaic import merge_burst_products
from faninsar.processing.mosaicking.overlap import (
    compute_feather,
    overlap_mask,
)
from faninsar.processing.mosaicking.phase_network import estimate_edge
from faninsar.processing.mosaicking.products import BurstGeoProduct


def _grid() -> GeoGridSpec:
    return GeoGridSpec(
        crs="EPSG:32633",
        transform=(0.0, 10.0, 0.0, 1000.0, 0.0, -10.0),
        width=60,
        height=20,
        resolution_m=(10.0, 10.0),
    )


def _two_bursts_with_overlap(
    phase_i: float,
    phase_j: float,
) -> tuple[BurstGeoProduct, BurstGeoProduct]:
    """Build two complex bursts with a known phase offset and overlap."""
    g = _grid()
    h, w = g.shape
    mask_i = np.zeros((h, w), dtype=bool)
    mask_j = np.zeros((h, w), dtype=bool)
    mask_i[:, 0:40] = True
    mask_j[:, 20:60] = True

    z_i = np.zeros((h, w), dtype=np.complex64)
    z_j = np.zeros((h, w), dtype=np.complex64)
    z_i[mask_i] = np.exp(1j * phase_i, dtype=np.complex64)
    z_j[mask_j] = np.exp(1j * phase_j, dtype=np.complex64)

    w_i = compute_feather(mask_i, feather_width_px=0.0)
    w_j = compute_feather(mask_j, feather_width_px=0.0)
    coh_i = np.zeros((h, w), dtype=np.float32)
    coh_j = np.zeros((h, w), dtype=np.float32)
    coh_i[mask_i] = 1.0
    coh_j[mask_j] = 1.0

    p_i = BurstGeoProduct(
        burst_id="a", path_id="T1_A", swath="IW1", date=date(2024, 1, 1),
        grid=g, complex=z_i, weight=w_i, coherence=coh_i,
    )
    p_j = BurstGeoProduct(
        burst_id="b", path_id="T1_A", swath="IW1", date=date(2024, 1, 1),
        grid=g, complex=z_j, weight=w_j, coherence=coh_j,
    )
    return p_i, p_j


def _complex_merge_residual(p_i, p_j) -> float:
    """RMS residual of a complex-domain merge."""
    mosaic = merge_burst_products(
        [p_i, p_j],
        mode="phase_network",
        min_overlap_px=10,
        min_edge_coherence=0.0,
    )
    assert mosaic.network is not None
    return float(mosaic.network.rms_residual)


def _post_unwrap_merge_residual(p_i, p_j, k_j_func) -> float:
    """RMS residual when burst j has spatially-varying 2π ambiguities.

    We simulate per-burst unwrap by adding an integer 2π offset to burst j's
    phase. The key pathology (plan §4.3) is that different parts of the
    overlap can carry *different* integer ambiguities (because each burst's
    unwrap reference and branch cuts differ), so the overlap Δψ is not a
    single constant — it has step discontinuities that the median cannot
    repair. ``k_j_func(row)`` returns the integer ambiguity for a given
    overlap row index.
    """
    overlap = overlap_mask(p_i.weight, p_j.weight, threshold=0.0)
    psi_i = np.angle(p_i.complex[overlap])
    # Build a spatially-varying integer ambiguity across the overlap
    overlap_rows = np.where(overlap.any(axis=1))[0]
    k_j = np.zeros_like(psi_i, dtype=np.float64)
    flat_idx = 0
    for r in overlap_rows:
        n_cols = int(overlap[r].sum())
        k = k_j_func(int(r))
        k_j[flat_idx:flat_idx + n_cols] = k
        flat_idx += n_cols
    psi_j = np.angle(p_j.complex[overlap]) + 2.0 * np.pi * k_j
    # Unwrapped-domain difference: the 2π step discontinuity survives here.
    dpsi = psi_i - psi_j
    median_offset = np.median(dpsi)
    residual = dpsi - median_offset
    return float(np.sqrt(np.mean(residual**2)))


def test_unwrap_order_post_unwrap_residual_far_larger_than_complex() -> None:
    """Complex merge residual is near-zero; post-unwrap merge is large."""
    p_i, p_j = _two_bursts_with_overlap(phase_i=0.0, phase_j=0.3)

    complex_residual = _complex_merge_residual(p_i, p_j)
    # Complex merge should be essentially exact (constant phase offset)
    assert complex_residual < 1e-3

    # Inject a spatially-varying 1-cycle (2π) unwrap ambiguity on burst j:
    # top half of the overlap has k=0, bottom half has k=1. This simulates
    # the branch-cut / reference inconsistency described in plan §4.3.
    post_unwrap_residual = _post_unwrap_merge_residual(
        p_i, p_j, k_j_func=lambda r: 0 if r < 10 else 1
    )
    assert post_unwrap_residual > 10.0 * complex_residual
    # And in absolute terms, the post-unwrap residual is non-trivial
    assert post_unwrap_residual > 0.1


def test_complex_merge_invariant_to_global_phase_shift() -> None:
    """A global 2π*k shift on one burst does not change the complex residual."""
    p_i0, p_j0 = _two_bursts_with_overlap(phase_i=0.0, phase_j=0.3)
    p_i1, p_j1 = _two_bursts_with_overlap(phase_i=2.0 * np.pi, phase_j=0.3)
    r0 = _complex_merge_residual(p_i0, p_j0)
    r1 = _complex_merge_residual(p_i1, p_j1)
    assert r0 == pytest.approx(r1, abs=1e-9)
