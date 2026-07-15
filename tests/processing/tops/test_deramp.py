"""Tests for TOPS deramp/reramp operators."""

from __future__ import annotations

import numpy as np

from faninsar.processing.tops import (
    TOPSCarrierModel,
    deramp,
    deramp_reramp_roundtrip_error,
    reramp,
    tops_carrier_phase,
)


def _model() -> TOPSCarrierModel:
    return TOPSCarrierModel(
        radar_frequency_hz=5.405e9,
        slant_range_time0_s=0.00533,
        range_sampling_rate_hz=6.434e7,
        azimuth_time_interval_s=0.002055,
        doppler_centroid_hz=(100.0, -50.0, 10.0),
        doppler_t0_s=0.00534,
        fm_rate_hz_s=(-2300.0, 4.5e5, -7.9e7),
        fm_t0_s=0.00533,
        burst_sensing_time_s=0.0,
    )


def test_deramp_reramp_roundtrip_within_tolerance() -> None:
    """Deramp then reramp recovers complex samples to 1e-6 relative error."""
    model = _model()
    rng = np.random.default_rng(0)
    samples = (rng.normal(size=(32, 64)) + 1j * rng.normal(size=(32, 64))).astype(
        np.complex64
    )
    # Apply a synthetic carrier then remove/restore it.
    phase = tops_carrier_phase(model, 32, 64)
    modulated = (samples * np.exp(1j * phase)).astype(np.complex64)
    error = deramp_reramp_roundtrip_error(modulated, model)
    assert error <= 1e-6


def test_deramp_centers_spectrum_energy() -> None:
    """Deramping a pure TOPS carrier reduces low-frequency residual phase."""
    model = _model()
    phase = tops_carrier_phase(model, 32, 64)
    carrier = np.exp(1j * phase).astype(np.complex64)
    deramped = deramp(carrier, model)
    # Ideal deramp of pure carrier yields near-constant complex ones.
    residual = np.angle(deramped * np.conjugate(deramped[0, 0]))
    assert float(np.max(np.abs(residual))) < 1e-5
    restored = reramp(deramped, model)
    np.testing.assert_allclose(restored, carrier, rtol=0, atol=1e-5)
