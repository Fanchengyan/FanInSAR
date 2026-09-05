"""Focused Stack/DEM seams for PROPOSAL-0041."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from faninsar.processing.dem import GridSpec
from faninsar.processing.dem.datum import convert_heights
from faninsar.processing.dem.seam import ExplicitAntimeridianError
from faninsar.stack.grid import automatic_grid, resolve_stack_grid


class _ConstantGeoid:
    def __init__(self, value: float) -> None:
        self.value = value

    def sample(self, latitude_deg: np.ndarray, longitude_deg: np.ndarray) -> np.ndarray:
        return np.full(np.broadcast(latitude_deg, longitude_deg).shape, self.value)


def test_datum_graph_converts_between_geoids_at_target_centres() -> None:
    """Cross-geoid conversion uses source plus and target minus undulation."""
    result = convert_heights(
        np.array([[100.0, np.nan]]),
        np.array([[10.0, 10.0]]),
        np.array([[45.0, 45.0]]),
        "egm96",
        "egm2008",
        samplers={"egm96": _ConstantGeoid(12.0), "egm2008": _ConstantGeoid(7.0)},
    )
    np.testing.assert_allclose(result[0, 0], 105.0)
    assert np.isnan(result[0, 1])


def test_automatic_grid_selects_utm_and_ups() -> None:
    """Center policy selects the expected UTM, north UPS, and south UPS."""
    assert automatic_grid((10.0, 40.0, 11.0, 41.0)).crs == "EPSG:32632"
    assert automatic_grid((10.0, 84.0, 11.0, 85.0)).crs == "EPSG:32661"
    assert automatic_grid((10.0, -85.0, 11.0, -84.0)).crs == "EPSG:32761"


def test_explicit_grid_wins_and_explicit_seam_fails() -> None:
    """An explicit grid bypasses auto CRS, while explicit seam ROI fails."""
    grid = GridSpec(
        crs="EPSG:3857",
        transform=Affine(30.0, 0, 0, 0, -30.0, 300.0),
        height=10,
        width=10,
    )
    assert resolve_stack_grid(grid, roi=(170.0, 10.0, -170.0, 11.0)).crs == "EPSG:3857"
    with pytest.raises(ExplicitAntimeridianError):
        resolve_stack_grid(
            "auto", roi=(170.0, 10.0, -170.0, 11.0), explicit_roi=True
        )
