"""Tests for topographic phase computation and removal."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from faninsar.processing.contracts import OrbitMetadata, OrbitStateVector
from faninsar.processing.coordinates import RadarGrid
from faninsar.processing.dem import ConstantDEM
from faninsar.processing.geometry import RadarGeometryModel
from faninsar.processing.interferometry.flatten import (
    compute_topographic_phase,
    estimate_residual_topographic_scale,
    remove_topographic_phase,
)


def _orbit_and_grid() -> tuple[OrbitMetadata, RadarGrid]:
    epoch = datetime(2016, 12, 7, 11, 18, 0, tzinfo=UTC)
    # Realistic low-Earth orbit (~700 km altitude) for well-conditioned geometry
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
    return orbit, grid


def test_remove_topographic_phase_on_synthetic_dem() -> None:
    """Topographic phase removal recovers flat-Earth phase on synthetic data."""
    orbit, grid = _orbit_and_grid()
    model_ref = RadarGeometryModel.from_radar_grid(grid, orbit)

    # Create a secondary model with a small baseline shift in z
    vectors_shifted = [
        OrbitStateVector(
            time=vec.time,
            position_m=(
                vec.position_m[0],
                vec.position_m[1],
                vec.position_m[2] + 100.0,
            ),
            velocity_m_s=vec.velocity_m_s,
        )
        for vec in orbit.vectors
    ]
    orbit_sec = OrbitMetadata(
        reference_frame=orbit.reference_frame,
        source=orbit.source + "-shifted",
        vectors=tuple(vectors_shifted),
    )
    model_sec = RadarGeometryModel.from_radar_grid(grid, orbit_sec)

    # Small grid for fast testing
    az_idx = np.array([[0.0, 1.0], [2.0, 3.0]])
    rg_idx = np.array([[10.0, 20.0], [30.0, 40.0]])

    dem = ConstantDEM(height=50.0)

    topo_phase = compute_topographic_phase(
        model_ref,
        model_sec,
        az_idx,
        rg_idx,
        dem=dem,
        device="cpu",
    )

    # Create a synthetic interferogram where the phase equals the topographic phase
    complex_ifg = np.exp(1j * topo_phase)

    # Remove topographic phase
    flat = remove_topographic_phase(complex_ifg, topo_phase)

    # Where converged, the flattened interferogram should have phase near 0
    converged = np.isfinite(topo_phase)
    if np.any(converged):
        np.testing.assert_allclose(np.angle(flat[converged]), 0.0, atol=1e-6)
    else:
        pytest.skip("no converged pixels for synthetic flattening test")


def test_remove_topographic_phase_changes_phase_when_dem_varies() -> None:
    """Flattening must actually change interferogram phase when DEM varies."""
    orbit, grid = _orbit_and_grid()
    model_ref = RadarGeometryModel.from_radar_grid(grid, orbit)

    vectors_shifted = [
        OrbitStateVector(
            time=vec.time,
            position_m=(
                vec.position_m[0],
                vec.position_m[1],
                vec.position_m[2] + 100.0,
            ),
            velocity_m_s=vec.velocity_m_s,
        )
        for vec in orbit.vectors
    ]
    orbit_sec = OrbitMetadata(
        reference_frame=orbit.reference_frame,
        source=orbit.source + "-shifted",
        vectors=tuple(vectors_shifted),
    )
    model_sec = RadarGeometryModel.from_radar_grid(grid, orbit_sec)

    az_idx = np.array([[0.0, 1.0], [2.0, 3.0]])
    rg_idx = np.array([[10.0, 20.0], [30.0, 40.0]])

    # Varying DEM heights
    dem = ConstantDEM(height=0.0)
    topo_flat = compute_topographic_phase(
        model_ref, model_sec, az_idx, rg_idx, dem=dem, device="cpu"
    )

    dem = ConstantDEM(height=100.0)
    topo_hill = compute_topographic_phase(
        model_ref, model_sec, az_idx, rg_idx, dem=dem, device="cpu"
    )

    conv_flat = np.isfinite(topo_flat)
    conv_hill = np.isfinite(topo_hill)
    conv_both = conv_flat & conv_hill

    if not np.any(conv_both):
        pytest.skip("no common converged pixels for DEM variation test")

    # Phase must differ when DEM changes
    assert not np.allclose(topo_flat[conv_both], topo_hill[conv_both], atol=1e-3)


def test_compute_topographic_phase_non_converged_are_nan() -> None:
    """Pixels outside orbit coverage produce NaN topographic phase."""
    orbit, grid = _orbit_and_grid()
    model_ref = RadarGeometryModel.from_radar_grid(grid, orbit)
    model_sec = model_ref

    # Huge azimuth index far outside orbit coverage
    az_idx = np.array([1.0e9])
    rg_idx = np.array([10.0])
    dem = ConstantDEM(height=0.0)

    topo_phase = compute_topographic_phase(
        model_ref,
        model_sec,
        az_idx,
        rg_idx,
        dem=dem,
        device="cpu",
    )

    assert np.all(np.isnan(topo_phase))


def test_estimate_residual_phase_screen_from_unwrapped_removes_ramp() -> None:
    """Unwrapped poly screen recovers a known multi-fringe range ramp."""
    from faninsar.processing.interferometry.flatten import (
        apply_residual_phase_screen_to_products,
        remove_residual_phase_screen,
    )

    height, width = 48, 128
    rg = np.arange(width, dtype=np.float64)[None, :]
    az = np.arange(height, dtype=np.float64)[:, None]
    true_screen = 0.03 * rg - 0.01 * az + 1.2e-5 * rg**2
    rng = np.random.default_rng(0)
    noise = 0.15 * rng.standard_normal((height, width))
    unw = (true_screen + noise).astype(np.float32)
    ifg = np.exp(1j * unw).astype(np.complex64)
    coh = np.full((height, width), 0.8, dtype=np.float32)

    unw_c, z_c, est, span, applied = apply_residual_phase_screen_to_products(
        unwrapped_phase=unw,
        complex_ifg=ifg,
        coherence=coh,
        range_degree=2,
        azimuth_degree=1,
        min_span_rad=0.5,
    )
    assert applied
    assert span > 2.0
    assert float(np.std(unw_c)) < float(np.std(unw)) * 0.35
    assert z_c is not None
    col = np.angle(z_c.mean(axis=0))
    residual_p2p = float(np.ptp(np.unwrap(col)))
    assert residual_p2p < 1.5
    # Screen should itself be removable from the original complex field.
    flat = remove_residual_phase_screen(ifg, est)
    assert float(np.std(np.angle(flat))) < 1.0


def test_residual_screen_uses_largest_conncomp_and_rejects_runaway() -> None:
    """Multi-component offsets must not invent a huge global poly screen."""
    from faninsar.processing.interferometry.flatten import (
        apply_residual_phase_screen_to_products,
        estimate_residual_phase_screen_from_unwrapped,
    )

    height, width = 40, 80
    rg = np.arange(width, dtype=np.float64)[None, :]
    true_ramp = 0.02 * rg
    unw = np.tile(true_ramp, (height, 1)).astype(np.float32)
    # Second component has a large constant offset (unwrap island).
    conn = np.ones((height, width), dtype=np.int32)
    conn[:, width // 2 :] = 2
    unw[:, width // 2 :] += 80.0
    coh = np.full((height, width), 0.9, dtype=np.float32)

    screen = estimate_residual_phase_screen_from_unwrapped(
        unw,
        coherence=coh,
        connected_components=conn,
        range_degree=1,
        azimuth_degree=0,
    )
    # Fit only on component 1 (left half); screen span should track the ramp.
    left = screen[:, : width // 2]
    assert float(np.ptp(left)) < 5.0

    # Without conncomp restriction a global fit on the offset field is huge.
    global_screen = estimate_residual_phase_screen_from_unwrapped(
        unw,
        coherence=coh,
        range_degree=1,
        azimuth_degree=0,
    )
    assert float(np.ptp(global_screen)) > 40.0

    # Without conncomp, span > max_span is rejected (multi-component risk).
    unw_c, z_c, _scr, span, applied = apply_residual_phase_screen_to_products(
        unwrapped_phase=unw,
        complex_ifg=np.exp(1j * unw).astype(np.complex64),
        coherence=coh,
        range_degree=1,
        azimuth_degree=0,
        min_span_rad=1.0,
        max_span_rad=40.0,
    )
    assert span > 40.0
    assert not applied
    np.testing.assert_array_equal(unw_c, unw)
    assert z_c is not None
    np.testing.assert_allclose(np.angle(z_c), np.angle(np.exp(1j * unw)), atol=1e-5)


def test_estimate_tiled_residual_height_screen_removes_local_height_residual() -> None:
    """Tiled residual recovers a constant height residual within each range band."""
    from faninsar.processing.interferometry.flatten import (
        estimate_tiled_residual_height_screen,
    )

    height, width = 48, 96
    xx = np.arange(width, dtype=np.float64)[None, :]
    hgt = np.broadcast_to(1000.0 + 20.0 * xx, (height, width)).copy()
    # Pure height residual (constant scale) — tiles should recover near-zero residual.
    true = 0.005 * hgt
    rng = np.random.default_rng(0)
    unw = (true + 0.1 * rng.standard_normal((height, width))).astype(np.float32)
    coh = np.full((height, width), 0.95, dtype=np.float32)
    conn = np.ones((height, width), dtype=np.int32)

    screen = estimate_tiled_residual_height_screen(
        unw,
        hgt,
        coherence=coh,
        connected_components=conn,
        n_az_tiles=2,
        n_rg_tiles=4,
        overlap_frac=0.25,
        min_tile_samples=20,
    )
    corr = unw - screen.astype(np.float32)
    assert float(np.std(corr)) < 0.5
    assert float(np.std(corr)) < float(np.std(unw)) * 0.25


def test_estimate_residual_height_poly_screen_removes_dem_shaped_residual() -> None:
    """Height-correlated residual (DEM × baseline error) is removed by the fit."""
    from faninsar.processing.interferometry.flatten import (
        estimate_residual_height_poly_screen,
        remove_residual_phase_screen,
    )

    height, width = 40, 80
    yy, xx = np.mgrid[0:height, 0:width]
    hgt = (800.0 + 12.0 * xx + 3.0 * yy).astype(np.float64)
    # Residual proportional to height plus mild range ramp.
    true = 0.008 * hgt + 0.01 * xx
    rng = np.random.default_rng(2)
    unw = (true + 0.15 * rng.standard_normal((height, width))).astype(np.float32)
    coh = np.full((height, width), 0.85, dtype=np.float32)
    conn = np.ones((height, width), dtype=np.int32)

    screen = estimate_residual_height_poly_screen(
        unw,
        hgt,
        coherence=coh,
        connected_components=conn,
        range_degree=1,
        azimuth_degree=0,
    )
    corr = unw - screen.astype(np.float32)
    assert float(np.std(corr)) < float(np.std(unw)) * 0.35
    ifg = np.exp(1j * unw).astype(np.complex64)
    flat = remove_residual_phase_screen(ifg, screen)
    assert float(np.std(np.angle(flat))) < 1.0


def test_residual_screen_applies_large_legitimate_ramp_with_conncomp() -> None:
    """A multi-cycle residual on a single component must be corrected.

    Regression for IW1_b3 / IW2_b3 geo bursts: true residual spans of
    hundreds of radians were skipped by a hard max_span gate even though
    the poly reduced std from ~38 to ~1.5 rad.
    """
    from faninsar.processing.interferometry.flatten import (
        apply_residual_phase_screen_to_products,
    )

    height, width = 48, 96
    rg = np.arange(width, dtype=np.float64)[None, :]
    az = np.arange(height, dtype=np.float64)[:, None]
    # ~60 rad az + multi-fringe range residual (like geo IW1_b3).
    true = 0.04 * rg + 1.2 * az
    rng = np.random.default_rng(1)
    unw = (true + 0.2 * rng.standard_normal((height, width))).astype(np.float32)
    coh = np.full((height, width), 0.85, dtype=np.float32)
    conn = np.ones((height, width), dtype=np.int32)
    ifg = np.exp(1j * unw).astype(np.complex64)

    unw_c, z_c, _scr, span, applied = apply_residual_phase_screen_to_products(
        unwrapped_phase=unw,
        complex_ifg=ifg,
        coherence=coh,
        connected_components=conn,
        range_degree=2,
        azimuth_degree=1,
        min_span_rad=1.0,
        max_span_rad=40.0,
    )
    assert span > 40.0
    assert applied
    assert float(np.std(unw_c)) < float(np.std(unw)) * 0.2
    assert z_c is not None


def test_estimate_residual_topographic_scale_recovers_known_scale() -> None:
    """Grid search recovers a planted residual topo scale on high-coherence pixels."""
    height, width = 48, 64
    az = np.arange(height, dtype=np.float64)[:, None]
    rg = np.arange(width, dtype=np.float64)[None, :]
    topo = (0.05 * rg + 0.02 * az).astype(np.float64)
    true_scale = 0.65
    rng = np.random.default_rng(7)
    noise = 0.05 * rng.standard_normal((height, width))
    phase = true_scale * topo + noise
    complex_ifg = np.exp(1j * phase).astype(np.complex64)
    coherence = np.full((height, width), 0.9, dtype=np.float32)

    scale, residual_rms = estimate_residual_topographic_scale(
        complex_ifg,
        topo,
        coherence=coherence,
        coh_thr=0.2,
        search=(-1.5, 1.5),
        n_grid=61,
    )
    assert scale == pytest.approx(true_scale, abs=0.05)
    assert residual_rms < 0.15

    corrected = remove_topographic_phase(complex_ifg, scale * topo)
    residual = np.angle(corrected)
    assert float(np.std(residual)) < float(np.std(phase)) * 0.5


def test_estimate_residual_topographic_scale_returns_zero_when_all_masked() -> None:
    """No valid pixels → scale 0 and NaN residual RMS."""
    topo = np.ones((8, 8), dtype=np.float64)
    ifg = np.exp(1j * topo).astype(np.complex64)
    coh = np.zeros((8, 8), dtype=np.float32)
    scale, residual_rms = estimate_residual_topographic_scale(
        ifg,
        topo,
        coherence=coh,
        coh_thr=0.5,
    )
    assert scale == 0.0
    assert residual_rms != residual_rms  # NaN
