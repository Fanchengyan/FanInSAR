"""Carrier factory physical centre tests (crop-aware TOPS carrier)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

from faninsar.processing.coregistration.tops import TOPSCarrierModel
from faninsar.processing.coregistration.tops.carrier import carrier_from_swath
from faninsar.processing.errors import InvalidProcessingStateError


def _swath(lines_per_burst: int = 1488) -> tuple[MagicMock, MagicMock]:
    """Minimal mock swath feeding the real carrier factory."""
    now = datetime(2020, 1, 10, tzinfo=UTC)
    burst = MagicMock()
    burst.index = 3
    burst.lines = lines_per_burst
    burst.sensing_time = now
    burst.azimuth_time = now
    burst.azimuth_anx_time_s = 2976.0
    vector = MagicMock()
    vector.time = now + timedelta(seconds=3.0)
    vector.velocity_m_s = (7500.0, 900.0, 500.0)
    dc = MagicMock()
    dc.reference_range_m = 800000.0
    dc.reference_time_s = 0.0
    dc.coefficients_hz = (100.0, -50.0, 10.0)
    swath = MagicMock()
    swath.swath = "IW1"
    swath.slant_range_time_s = 0.005334
    swath.range_sampling_rate_hz = 64_345_238.1
    swath.azimuth_time_interval_s = 0.0020555563
    swath.radar_frequency_hz = 5.405e9
    swath.azimuth_steering_rate_rad_s = 0.0016
    swath.azimuth_fm_rate = [(now, 0.005334, (-2300.0, 4.5e5, -7.9e7))]
    swath.doppler_centroid = [dc]
    swath.orbit.vectors = [vector]
    swath.bursts = (burst,)
    return swath, burst


@pytest.mark.parametrize(
    ("first_azimuth_line", "lines"),
    [(370, 1488), (364, 1488), (0, 1488)],
)
def test_factory_physical_centre_from_first_azimuth_line(
    first_azimuth_line: int, lines: int
) -> None:
    """The factory carries the mid-burst row of the loaded array."""
    swath, burst = _swath(lines_per_burst=lines)
    carrier = carrier_from_swath(swath, burst, first_azimuth_line=first_azimuth_line)
    assert carrier.azimuth_centre_row == pytest.approx(
        float(burst.lines // 2 - first_azimuth_line)
    )


def test_factory_without_first_azimuth_line_keeps_legacy_none() -> None:
    """No crop origin means no model centre; legacy derivation stays active."""
    swath, burst = _swath()
    carrier = carrier_from_swath(swath, burst, first_range_sample=0)
    assert carrier.azimuth_centre_row is None


def test_model_centre_rejects_non_finite() -> None:
    """The physical centre must be a finite row index."""
    with pytest.raises(InvalidProcessingStateError):
        TOPSCarrierModel(
            radar_frequency_hz=5.405e9,
            slant_range_time0_s=0.00533,
            range_sampling_rate_hz=6.434e7,
            azimuth_time_interval_s=0.002055,
            doppler_centroid_hz=(100.0, -50.0, 10.0),
            doppler_t0_s=0.00534,
            fm_rate_hz_s=(-2300.0, 4.5e5, -7.9e7),
            fm_t0_s=0.00533,
            burst_sensing_time_s=0.0,
            burst_start_slant_range_time_s=0.00533,
            azimuth_steering_rate_hz_s=6500.0,
            azimuth_centre_row=float("nan"),
        )


def test_real_layout_centres_geo_vs_full() -> None:
    """Geo crops 370/364 give 374/380; the full burst keeps 744."""
    swath, burst = _swath()
    assert burst.lines // 2 - 370 == 374
    assert burst.lines // 2 - 364 == 380
    assert burst.lines // 2 - 0 == 744
    for row0, expected in ((370, 374.0), (364, 380.0), (0, 744.0)):
        carrier = carrier_from_swath(swath, burst, first_azimuth_line=row0)
        assert carrier.azimuth_centre_row == pytest.approx(expected)
