"""Tests for IonosphereEstimationConfig caller-input validation."""

from __future__ import annotations

import pytest

from faninsar.processing.atmosphere.config import IonosphereEstimationConfig


class TestIonosphereEstimationConfig:
    """Physical-sanity contract of the caller-supplied configuration."""

    def test_accepts_fine_mode_thirds_split(
        self, fine_mode_config: IonosphereEstimationConfig
    ) -> None:
        assert fine_mode_config.effective_split_hz == pytest.approx(2 * 28e6 / 3)
        assert fine_mode_config.degraded is False

    @pytest.mark.parametrize("bad", [-1.0, 0.0])
    def test_rejects_non_positive_carrier(self, bad: float) -> None:
        with pytest.raises(ValueError, match="f0 must be positive"):
            IonosphereEstimationConfig(f0=bad, freq_low=1.0, freq_high=2.0)

    def test_rejects_non_monotonic_roles(self) -> None:
        with pytest.raises(ValueError, match="freq_low < f0 < freq_high"):
            IonosphereEstimationConfig(f0=1257.5e6, freq_low=1257.5e6, freq_high=1266e6)

    def test_rejects_equal_subband_centers(self) -> None:
        with pytest.raises(ValueError, match="freq_low < f0 < freq_high"):
            IonosphereEstimationConfig(
                f0=1257.5e6,
                freq_low=1257.4e6,
                freq_high=1257.4e6,
            )

    def test_narrow_band_requires_explicit_degradation(self) -> None:
        with pytest.raises(ValueError, match="degraded=True"):
            IonosphereEstimationConfig(
                f0=1257.5e6,
                freq_low=1257.498e6,
                freq_high=1257.502e6,
                min_bandwidth_hz=14e6,
            )

    def test_degraded_requires_reason_code(self) -> None:
        with pytest.raises(ValueError, match="degradation_reason"):
            IonosphereEstimationConfig(
                f0=1257.5e6,
                freq_low=1257.498e6,
                freq_high=1257.502e6,
                min_bandwidth_hz=14e6,
                degraded=True,
            )
