"""Regression tests for original-domain secondary restore after deramped resample."""

from __future__ import annotations

import numpy as np

from faninsar.processing.coregistration import resample_complex_deramped_reramp
from faninsar.processing.coregistration.tops.deramp import (
    TOPSCarrierModel,
    deramp,
    reramp,
    restore_original_domain_secondary,
    tops_carrier_phase,
)
from faninsar.processing.interferometry.pair import form_interferogram


def _carrier(*, fm0: float = -2000.0, dc0: float = 50.0) -> TOPSCarrierModel:
    """Build a carrier with strong azimuth FM so wrong restore is detectable."""
    return TOPSCarrierModel(
        radar_frequency_hz=5.405e9,
        slant_range_time0_s=0.0053,
        range_sampling_rate_hz=64.348e6,
        azimuth_time_interval_s=0.002055556,
        doppler_centroid_hz=(dc0, 0.02),
        doppler_t0_s=0.0,
        fm_rate_hz_s=(fm0, 5.0),
        fm_t0_s=0.0,
        burst_sensing_time_s=0.0,
        burst_start_slant_range_time_s=0.0053,
        azimuth_steering_rate_hz_s=6500.0,
    )


def test_deramped_single_remap_matches_integer_shifted_interferogram() -> None:
    """Deramped remap with analytical reramp is exact at integer coordinates."""
    height, width = 64, 128
    az = np.arange(height, dtype=np.float64)[:, None]
    rg = np.arange(width, dtype=np.float64)[None, :]
    geo_phase = 0.04 * rg + 0.01 * az

    carrier_ref = _carrier(fm0=-1800.0, dc0=40.0)
    carrier_sec = _carrier(fm0=-4000.0, dc0=400.0)
    phi_ref = tops_carrier_phase(carrier_ref, height, width, dtype=np.float64)
    phi_sec = tops_carrier_phase(carrier_sec, height, width, dtype=np.float64)

    ref_orig = np.exp(1j * (0.5 * geo_phase + phi_ref)).astype(np.complex64)
    sec_orig = np.exp(1j * (-0.5 * geo_phase + phi_sec)).astype(np.complex64)

    off_rg, off_az = 3.0, 2.0
    ref_d = deramp(ref_orig, carrier_ref)
    sec_d = deramp(sec_orig, carrier_sec)
    sec_d_shifted = np.roll(
        np.roll(sec_d, int(off_az), axis=0),
        int(off_rg),
        axis=1,
    )
    sec_d_shifted[: int(off_az), :] = 0
    sec_d_shifted[:, : int(off_rg)] = 0
    raw_ifg = form_interferogram(ref_d, sec_d_shifted, multilook=(2, 4))
    sec_resamp = resample_complex_deramped_reramp(
        sec_d,
        secondary_carrier=carrier_sec,
        range_offset_px=off_rg,
        azimuth_offset_px=off_az,
        output_carrier=carrier_ref,
    )
    ref_out = reramp(ref_d, carrier_ref)
    good_ifg = form_interferogram(ref_out, sec_resamp, multilook=(2, 4))

    mask = (np.abs(raw_ifg.complex_ifg) > 0.1) & (np.abs(good_ifg.complex_ifg) > 0.1)
    dph = np.angle(raw_ifg.complex_ifg * np.conj(good_ifg.complex_ifg))
    assert mask.any()
    assert float(np.sqrt(np.mean(dph[mask] ** 2))) < 0.15
    assert float(np.abs(np.mean(np.exp(1j * dph[mask])))) > 0.95

    sec_wrong = reramp(sec_d_shifted, carrier_sec)
    bad_ifg = form_interferogram(ref_out, sec_wrong, multilook=(2, 4))
    dph_bad = np.angle(raw_ifg.complex_ifg * np.conj(bad_ifg.complex_ifg))
    assert float(np.abs(np.mean(np.exp(1j * dph_bad[mask])))) < 0.5
    assert float(np.sqrt(np.mean(dph_bad[mask] ** 2))) > 0.8


def test_deramped_remap_fractional_shift_phase_error_bound() -> None:
    """Fractional offsets keep residual phase error small on band-limited signals.

    Dual-modes root-cause analysis showed ~1 rad dual_std gap is consistent with
    O(0.002) px phase error after carrier mishandling. Analytical reramp at
    fractional source coords must keep phase error well below that on a pure
    geometric phase field.
    """
    height, width = 96, 160
    az = np.arange(height, dtype=np.float64)[:, None]
    rg = np.arange(width, dtype=np.float64)[None, :]
    geo_phase = 0.03 * rg + 0.008 * az

    carrier_ref = _carrier(fm0=-1800.0, dc0=40.0)
    carrier_sec = _carrier(fm0=-3500.0, dc0=300.0)
    phi_ref = tops_carrier_phase(carrier_ref, height, width, dtype=np.float64)
    phi_sec = tops_carrier_phase(carrier_sec, height, width, dtype=np.float64)

    ref_orig = np.exp(1j * (0.5 * geo_phase + phi_ref)).astype(np.complex64)
    # Secondary is geometrically shifted by a known fractional offset on the
    # deramped field; analytic phase without carrier is the truth reference.
    off_rg, off_az = 2.35, 1.4
    sec_geo = np.exp(1j * (-0.5 * geo_phase)).astype(np.complex64)
    # Build secondary in original domain at unshifted grid, then deramp+resample.
    sec_orig = (sec_geo * np.exp(1j * phi_sec)).astype(np.complex64)

    ref_d = deramp(ref_orig, carrier_ref)
    sec_d = deramp(sec_orig, carrier_sec)
    sec_resamp = resample_complex_deramped_reramp(
        sec_d,
        secondary_carrier=carrier_sec,
        range_offset_px=off_rg,
        azimuth_offset_px=off_az,
        output_carrier=carrier_ref,
    )
    ref_out = reramp(ref_d, carrier_ref)
    ifg = form_interferogram(ref_out, sec_resamp, multilook=(1, 1))

    # Expected geometric ifg phase after perfect coreg: geo_phase (ref 0.5 - sec -0.5).
    # Because we apply offset on secondary (source = out - offset), the secondary
    # geometric phase at output (r,c) comes from source (r-off_az, c-off_rg).
    rows = az + 0.0
    cols = rg + 0.0
    src_r = rows - off_az
    src_c = cols - off_rg
    # Interior valid for fractional sampling
    interior = (
        (src_r > 4)
        & (src_r < height - 5)
        & (src_c > 4)
        & (src_c < width - 5)
    )
    expected = 0.5 * geo_phase - (-0.5) * (
        0.03 * src_c + 0.008 * src_r
    )
    # After form_interferogram (ref * conj(sec)) in original domain, carrier
    # should cancel if reramp uses ref carrier on both — residual ≈ geo only.
    dph = np.angle(ifg.complex_ifg * np.exp(-1j * expected))
    mask = interior & (np.abs(ifg.complex_ifg) > 0.5)
    assert mask.any()
    rmse = float(np.sqrt(np.mean(dph[mask] ** 2)))
    # Bound well below the historical ~1 rad formation gap.
    assert rmse < 0.25


def test_restore_original_domain_is_unit_magnitude() -> None:
    """Restore multiplies by a pure phase factor (amplitude preserved)."""
    height, width = 32, 48
    carrier = _carrier()
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(height, width)) + 1j * rng.normal(size=(height, width))
    sec = noise.astype(np.complex64)
    amp0 = np.abs(sec)
    rg_off = np.full((height, width), 1.5, dtype=np.float64)
    az_off = np.full((height, width), -0.5, dtype=np.float64)
    out = restore_original_domain_secondary(sec, carrier, rg_off, az_off)
    # Interior pixels (away from map_coordinates edges) keep amplitude.
    interior = np.s_[2:-2, 2:-2]
    np.testing.assert_allclose(
        np.abs(out[interior]), amp0[interior], rtol=1e-5, atol=1e-5
    )
