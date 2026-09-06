"""Real SAFE tests using local scenes as fixtures for the production workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from faninsar.missions.s1 import open_safe_product, read_burst_window

SLC_ROOT = Path("/Volumes/DATA2/TEST_sentinel-1/sentinel-slc")
SCENES = sorted(SLC_ROOT.glob("S1A_IW_SLC*.zip")) if SLC_ROOT.exists() else []


@pytest.mark.skipif(len(SCENES) < 1, reason="local S1 ZIP corpus unavailable")
def test_read_burst_window_from_real_safe() -> None:
    """Read a non-empty complex window from a real SAFE measurement."""
    product = open_safe_product(SCENES[0])
    window = read_burst_window(
        product.swath("IW1"),
        burst_index=0,
        height=64,
        width=64,
    )
    assert window.samples.shape == (64, 64)
    assert np.iscomplexobj(window.samples)
    assert float(np.mean(np.abs(window.samples) > 0)) > 0.5
