"""Tests for range split-spectrum bandpass filtering."""

from __future__ import annotations

import math

import pytest
import torch

from faninsar.processing.atmosphere.split_spectrum import split_range_spectrum

FS = 32.0e6
ROWS = 8


def _tone(cols: int, offset_hz: float, fs: float) -> torch.Tensor:
    n = torch.arange(cols, dtype=torch.float64)
    phase = 2.0 * math.pi * (offset_hz * n / fs)
    wave = torch.exp(1j * phase).to(torch.complex64)
    return wave.unsqueeze(0).repeat(ROWS, 1)


class TestSplitRangeSpectrum:
    def test_inband_tone_passes(self) -> None:
        cols = 512
        plateau_offset = -5.0e6  # inside the Tukey plateau of [-14, +4] MHz
        slc = _tone(cols, plateau_offset, FS)

        result = split_range_spectrum(
            slc,
            range_sampling_rate_hz=FS,
            low_offset_hz=-14.0e6,
            high_offset_hz=+4.0e6,
        )
        in_power = float((result.data.abs() ** 2).sum())
        src_power = float((slc.abs() ** 2).sum())
        assert in_power / src_power > 0.90

    def test_out_of_band_tone_suppressed(self) -> None:
        cols = 512
        far_offset = +13.0e6  # outside the requested upper edge (+4 MHz)
        slc = _tone(cols, far_offset, FS)

        result = split_range_spectrum(
            slc,
            range_sampling_rate_hz=FS,
            low_offset_hz=-14.0e6,
            high_offset_hz=+4.0e6,
        )
        out_power = float((result.data.abs() ** 2).sum())
        src_power = float((slc.abs() ** 2).sum())
        ratio = out_power / max(src_power, 1e-30)
        assert ratio < 5e-3

    def test_shape_and_dtype_preserved(self) -> None:
        slc = torch.randn(ROWS, 256, dtype=torch.float32).to(torch.complex64)
        result = split_range_spectrum(
            slc,
            range_sampling_rate_hz=FS,
            low_offset_hz=-9e6,
            high_offset_hz=9e6,
        )
        assert result.data.shape == slc.shape
        assert result.data.dtype == torch.complex64
        assert result.bandwidth == pytest.approx(18e6)
        assert result.center_frequency == pytest.approx(0.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_non_positive_sampling_rate(self, bad: float) -> None:
        with pytest.raises(ValueError, match="range_sampling_rate_hz"):
            split_range_spectrum(
                torch.zeros(2, 8, dtype=torch.complex64),
                range_sampling_rate_hz=bad,
                low_offset_hz=-1e6,
                high_offset_hz=1e6,
            )

    def test_rejects_unknown_window(self) -> None:
        with pytest.raises(ValueError, match="window_function"):
            split_range_spectrum(
                torch.zeros(2, 8, dtype=torch.complex64),
                range_sampling_rate_hz=FS,
                low_offset_hz=-1e6,
                high_offset_hz=1e6,
                window_function="hamming",
            )
