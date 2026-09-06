"""Tests for remote burst extraction (ASF SAFE via ``requests``).

These tests require network access and a valid ``~/.netrc`` entry for
``urs.earthdata.nasa.gov``.  They are marked ``@slow``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.missions.s1 import extract_remote_burst

if TYPE_CHECKING:
    from faninsar.missions.s1.io import BurstArray

# This granule was used for all remote validation tests.
_REMOTE_URL = (
    "https://datapool.asf.alaska.edu/SLC/SA/"
    "S1A_IW_SLC__1SDV_20240630T234903_20240630T234921_054559_06A3FE_1884.zip"
)


@pytest.mark.slow
@pytest.mark.network
class TestRemoteExtract:
    """Smoke tests against a real ASF SAFE product."""

    @pytest.fixture(scope="class")
    def burst(self) -> BurstArray:
        """Extract the middle burst of IW1 VV for all tests."""
        return extract_remote_burst(_REMOTE_URL, "IW1", "VV", burst_index=2)

    def test_burst_shape_nonzero(self, burst: BurstArray) -> None:
        """The burst has 2-D samples with the expected number of azimuth lines."""
        assert burst.samples.ndim == 2
        assert burst.samples.shape[0] == 1498  # lines_per_burst
        # Width is the valid-sample envelope (not necessarily samples_per_burst)
        assert 0 < burst.samples.shape[1] <= 20843
        assert burst.samples.dtype == np.complex64

    def test_burst_has_valid_samples(self, burst: BurstArray) -> None:
        """The majority of samples should be non-zero."""
        nz = float(np.count_nonzero(burst.samples))
        ratio = nz / float(burst.samples.size)
        assert ratio > 0.8, f"valid sample ratio {ratio:.3f} < 0.8"

    def test_burst_has_valid_mask(self, burst: BurstArray) -> None:
        """Valid mask covers most pixels."""
        assert burst.valid_mask.dtype == np.bool_
        assert float(np.mean(burst.valid_mask)) > 0.8

    def test_burst_index_and_placement(self, burst: BurstArray) -> None:
        """Placement metadata is set."""
        assert burst.burst_index == 2
        assert burst.row0 == 2 * 1498
        assert burst.col0 >= 0
