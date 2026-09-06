"""Tests for the unified merge-methods gateway (methods.merge_bursts).

Covers all selectable methods listed in ``MergeMethod`` and confirms:
 - perfectly-aligned bursts are recovered by every alignment-bearing method,
 - a known constant offset is removed,
 - a range ramp is removed by ``faninsar_network`` / ``faninsar_weighted``
   (the only methods that estimate a slope),
 - legacy ``merge_burst_products(mode=...)`` maps onto the new gateway.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from faninsar.processing.mosaicking.grid import GeoGridSpec
from faninsar.processing.mosaicking.methods import (
    NON_REFERENCE_METHODS,
    MergeMethod,
    merge_bursts,
)
from faninsar.processing.mosaicking.mosaic import merge_burst_products
from faninsar.processing.mosaicking.overlap import compute_feather
from faninsar.processing.mosaicking.products import BurstGeoProduct


def _grid(width: int = 40, height: int = 20) -> GeoGridSpec:
    return GeoGridSpec(
        crs="EPSG:32633",
        transform=(0.0, 10.0, 0.0, 1000.0, 0.0, -10.0),
        width=width,
        height=height,
        resolution_m=(10.0, 10.0),
    )


def _make_product(
    burst_id: str,
    phase_offset_rad: float = 0.0,
    valid_slice: tuple[slice, slice] = (slice(0, 20), slice(0, 40)),
    *,
    z_field: np.ndarray | None = None,
    coh_val: float = 1.0,
    path_id: str = "T1_A",
    look: str = "ascending",
) -> BurstGeoProduct:
    """Build a synthetic burst product with a constant-phase field.

    If ``z_field`` is provided it is used directly (e.g. for range-ramp
    scenarios); otherwise the burst is a unit-modulus disk at
    ``phase_offset_rad`` on the valid region.
    """
    g = _grid()
    h, w = g.shape
    mask = np.zeros((h, w), dtype=bool)
    mask[valid_slice] = True
    if z_field is None:
        z = np.zeros((h, w), dtype=np.complex64)
        z[mask] = np.exp(1j * phase_offset_rad, dtype=np.complex64)
    else:
        z = z_field.astype(np.complex64)
    weight = compute_feather(mask, feather_width_px=0.0)
    coh = np.zeros((h, w), dtype=np.float32)
    coh[mask] = coh_val
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


ALL_METHODS = [
    "complex_average",
    "faninsar_network",
    "isce2_avg",
    "isce2_top",
    "insardev_equal",
    "insardev_ramp",
    "insardev_weighted",
    "faninsar_weighted",
    "gmtsar_cut",
]

# Methods that perform an inter-burst phase alignment (constant offset).
ALIGNING_METHODS = [m for m in ALL_METHODS if m not in ("complex_average", "gmtsar_cut")]


@pytest.mark.parametrize("method", ALIGNING_METHODS)
def test_aligned_bursts_recovered(method: MergeMethod) -> None:
    """Two bursts with the SAME phase merge to |z|~1 everywhere with alignment."""
    p_i = _make_product("a", 0.0, (slice(0, 20), slice(0, 20)))
    p_j = _make_product("b", 0.0, (slice(0, 20), slice(10, 30)))
    mosaic = merge_bursts(
        [p_i, p_j], method=method, min_overlap_px=10, min_edge_coherence=0.0
    )
    overlap = (p_i.weight > 0) & (p_j.weight > 0)
    assert overlap.sum() > 50
    mag = np.abs(mosaic.complex[overlap]).mean()
    # Aligning methods should drive the overlap to coherent |z| close to 1.
    assert mag > 0.85, f"{method}: overlap |z|={mag:.3f} expected > 0.85"


@pytest.mark.parametrize("method", ALIGNING_METHODS)
def test_constant_phase_offset_removed(method: MergeMethod) -> None:
    """A 0.8 rad constant offset between bursts is removed by aligning methods."""
    p_i = _make_product("a", 0.0, (slice(0, 20), slice(0, 20)))
    p_j = _make_product("b", 0.8, (slice(0, 20), slice(10, 30)))
    mosaic = merge_bursts(
        [p_i, p_j], method=method, min_overlap_px=10, min_edge_coherence=0.0
    )
    overlap = (p_i.weight > 0) & (p_j.weight > 0)
    mag = np.abs(mosaic.complex[overlap]).mean()
    assert mag > 0.85, f"{method}: overlap |z|={mag:.3f} expected > 0.85"


def test_range_ramp_removed_by_faninsar_network() -> None:
    """``faninsar_network`` (const + range-slope LS) removes a range-ramp seam.

    Mirrors ``test_solve_network_and_mosaic_remove_range_seam`` but drives the
    new gateway directly: a 0.025 rad/col ramp on the right burst is absorbed
    by the slope unknowns of ``solve_network``.
    """
    g = _grid()
    h, w = g.shape
    cols = np.arange(w, dtype=np.float64)
    left = _make_product("lw", phase_offset_rad=0.0, valid_slice=(slice(0, 20), slice(0, 22)))
    # Right burst: phase = 0.3 + 0.025 * col (relative to left's zero).
    z_right = np.zeros((h, w), dtype=np.complex64)
    mask_r = np.zeros((h, w), dtype=bool)
    mask_r[slice(0, 20), slice(12, 40)] = True
    for c in range(w):
        if mask_r[:, c].any():
            z_right[:, c][mask_r[:, c]] = np.exp(
                1j * (0.3 + 0.025 * c), dtype=np.complex64
            )
    right = _make_product("rw", z_field=z_right, valid_slice=(slice(0, 20), slice(12, 40)))
    mosaic = merge_bursts(
        [left, right], method="faninsar_network",
        min_overlap_px=10, min_edge_coherence=0.0,
    )
    z = mosaic.complex
    # Right-only region (cols 30..38): phase should be near left's 0 reference.
    right_only = z[:, 30:38]
    valid = np.abs(right_only) > 0.5
    assert int(valid.sum()) > 20
    phases = np.angle(right_only[valid])
    assert float(np.std(phases)) < 0.15
    assert abs(float(np.mean(phases))) < 0.25


def test_range_ramp_not_removed_when_no_slope_method() -> None:
    """``insardev_ramp`` skips small-x-extent ramps (x_range < 100 guard).

    Confirm that with a narrow overlap (x_extent < 100 px) the insardev_ramp
    estimator does NOT pick up the range ramp — leaving a column-dependent
    residual that ``faninsar_network`` (which always estimates a slope) absorbs.
    The comparison is *relative*: the residual std under insardev_ramp must
    exceed the residual std under faninsar_network, which is the observed-
    data consequence of the x_range > 100 guard in InSAR.dev ``fit``.
    """
    g = _grid()
    h, w = g.shape
    left = _make_product("lw", phase_offset_rad=0.0, valid_slice=(slice(0, 20), slice(0, 22)))
    z_right = np.zeros((h, w), dtype=np.complex64)
    mask_r = np.zeros((h, w), dtype=bool)
    mask_r[slice(0, 20), slice(12, 40)] = True
    for c in range(w):
        if mask_r[:, c].any():
            z_right[:, c][mask_r[:, c]] = np.exp(
                1j * (0.3 + 0.025 * c), dtype=np.complex64
            )
    right = _make_product("rw", z_field=z_right, valid_slice=(slice(0, 20), slice(12, 40)))

    def _resid_std(method: MergeMethod) -> float:
        mosaic = merge_bursts(
            [left, right], method=method,
            min_overlap_px=10, min_edge_coherence=0.0,
        )
        z = mosaic.complex
        right_only = z[:, 30:38]
        valid = np.abs(right_only) > 0.5
        return float(np.std(np.angle(right_only[valid])))

    std_rramp = _resid_std("insardev_ramp")
    std_fsnet = _resid_std("faninsar_network")
    # insardev_ramp MUST leave a larger residual than faninsar_network
    # because it skips the slope. Quantitative values depend on the synthetic
    # ramp magnitude; only the ordering is asserted.
    assert std_rramp > std_fsnet, (
        f"insardev_ramp residual {std_rramp:.4f} should exceed "
        f"faninsar_network residual {std_fsnet:.4f} (insardev_ramp skips slope)"
    )


def test_legacy_mode_phase_network_maps_to_faninsar_network() -> None:
    """Legacy ``mode='phase_network'`` delegates to ``faninsar_network``.

    Ensures the wrong-sign ``insardev_ramp`` default does NOT silently
    break legacy callers expecting range-slope LS behaviour.
    """
    p_i = _make_product("a", 0.0, (slice(0, 20), slice(0, 20)))
    p_j = _make_product("b", 0.5, (slice(0, 20), slice(10, 26)))
    legacy = merge_burst_products(
        [p_i, p_j], mode="phase_network", min_overlap_px=10, min_edge_coherence=0.0
    )
    direct = merge_bursts(
        [p_i, p_j], method="faninsar_network",
        min_overlap_px=10, min_edge_coherence=0.0,
    )
    assert np.allclose(legacy.complex, direct.complex, atol=1e-6)


def test_legacy_mode_complex_average_maps_to_complex_average() -> None:
    p_i = _make_product("a", 0.0, (slice(0, 20), slice(0, 20)))
    p_j = _make_product("b", 0.8, (slice(0, 20), slice(10, 30)))
    legacy = merge_burst_products(
        [p_i, p_j], mode="complex_average", min_overlap_px=10, min_edge_coherence=0.0
    )
    direct = merge_bursts(
        [p_i, p_j], method="complex_average",
        min_overlap_px=10, min_edge_coherence=0.0,
    )
    assert np.allclose(legacy.complex, direct.complex, atol=1e-6)


def test_non_reference_methods_set_is_documented() -> None:
    """``faninsar_weighted`` must be flagged as not grounded in any reference."""
    assert "faninsar_weighted" in NON_REFERENCE_METHODS
    assert "insardev_ramp" not in NON_REFERENCE_METHODS
    assert "isce2_avg" not in NON_REFERENCE_METHODS


@pytest.mark.parametrize("method", ALL_METHODS)
def test_output_shapes_and_dtypes(method: MergeMethod) -> None:
    """Every method emits a MosaicProduct with the expected shape/dtype."""
    p_i = _make_product("a", 0.0, (slice(0, 20), slice(0, 16)))
    p_j = _make_product("b", 0.5, (slice(0, 20), slice(10, 26)))
    g = _grid()
    mosaic = merge_bursts(
        [p_i, p_j], method=method, min_overlap_px=10, min_edge_coherence=0.0
    )
    assert mosaic.complex.shape == g.shape
    assert mosaic.complex.dtype == np.complex64
    assert mosaic.weight_sum.shape == g.shape
    assert mosaic.n_bursts.dtype == np.uint8
    assert mosaic.component_id.shape == g.shape


def test_empty_products_raises() -> None:
    with pytest.raises(ValueError):
        merge_bursts([], method="insardev_ramp")
    with pytest.raises(ValueError):
        merge_burst_products([], mode="phase_network")


@pytest.mark.parametrize("method", ALL_METHODS)
def test_single_product_passes_through(method: MergeMethod) -> None:
    """A single burst is the identity for all methods (no overlap possible)."""
    p = _make_product("a", 0.4, (slice(0, 20), slice(0, 20)))
    mosaic = merge_bursts([p], method=method)
    assert np.allclose(mosaic.complex[p.weight > 0], p.complex[p.weight > 0])
