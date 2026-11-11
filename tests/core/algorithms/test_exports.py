"""Explicit, lazy public exports without expanding FanInSAR's root API."""

from __future__ import annotations

import faninsar
from faninsar.core import algorithms


def test_toolbox_publishes_only_accepted_operations_and_mathematical_results():
    """Discovery names are stable without loading absent/inactive family kernels."""
    expected = {
        "interpolate2d",
        "block_reduce2d",
        "filter2d",
        "polyval",
        "fit_surface2d",
        "least_squares",
        "PolynomialSurface",
        "LeastSquaresResult",
    }
    assert set(algorithms.__all__) == expected
    assert expected <= set(dir(algorithms))
    assert expected.isdisjoint(faninsar.__all__)
