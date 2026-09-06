"""Tests for ESD azimuth shift estimation."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.coregistration.esd import ESDResult, estimate_azimuth_shift_esd


def _make_dual_tone_reference(
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a complex image with two azimuth tones at the look centres.

    The lower and upper spectral looks each capture one tone, so the
    differential interferogram has a constant phase proportional to the
    azimuth shift.  This is an idealised signal for testing ESD.
    """
    n_az, n_rg = shape
    i = np.arange(n_az)[:, None]
    # Tones at the centre frequencies of the lower and upper looks
    f_lower = -0.225
    f_upper = +0.225
    tone_lower = np.exp(2j * np.pi * f_lower * i)
    tone_upper = np.exp(2j * np.pi * f_upper * i)
    noise = 0.01 * (rng.normal(size=(n_az, n_rg)) + 1j * rng.normal(size=(n_az, n_rg)))
    return ((tone_lower + tone_upper) + noise).astype(np.complex64)


@pytest.mark.parametrize("injected_shift", [-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5])
def test_esd_recovers_injected_azimuth_shift(injected_shift: float) -> None:
    """ESD recovers a synthetic sub-pixel azimuth shift within 0.05 px."""
    rng = np.random.default_rng(42)
    n_az, n_rg = 256, 64
    reference = _make_dual_tone_reference((n_az, n_rg), rng)

    # Apply a pure azimuth shift via linear phase ramp in the azimuth spectrum
    f_az = np.fft.fftfreq(n_az, d=1.0)[:, None]
    phase_ramp = np.exp(-2j * np.pi * f_az * injected_shift)
    secondary = np.fft.ifft(np.fft.fft(reference, axis=0) * phase_ramp, axis=0).astype(
        np.complex64
    )

    result = estimate_azimuth_shift_esd(reference, secondary)
    assert isinstance(result, ESDResult)
    assert result.coherence > 0.85
    assert result.azimuth_shift_px == pytest.approx(injected_shift, abs=0.05)


def test_esd_zero_shift_gives_near_zero() -> None:
    """Identical arrays should yield a shift near zero."""
    rng = np.random.default_rng(7)
    reference = _make_dual_tone_reference((128, 32), rng)
    result = estimate_azimuth_shift_esd(reference, reference)
    assert result.azimuth_shift_px == pytest.approx(0.0, abs=0.05)
    assert result.coherence > 0.95


def test_esd_invalid_inputs_raises() -> None:
    """Mismatched shapes or non-complex inputs raise ValueError."""
    a = np.zeros((10, 10), dtype=np.complex64)
    b = np.zeros((10, 11), dtype=np.complex64)
    from faninsar.processing.errors import InvalidProcessingStateError

    with pytest.raises(InvalidProcessingStateError, match="matching 2-D arrays"):
        estimate_azimuth_shift_esd(a, b)
    c = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(InvalidProcessingStateError, match="complex-valued"):
        estimate_azimuth_shift_esd(a, c)
