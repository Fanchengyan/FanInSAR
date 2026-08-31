"""Tests for the shared immutable DEM grid contract."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from faninsar.processing.dem import GridSpec


def test_grid_spec_is_canonical_and_immutable() -> None:
    validity = np.ones((2, 3), dtype=bool)
    grid = GridSpec(
        crs="EPSG:4326",
        transform=Affine(0.1, 0, 10, 0, -0.1, 2),
        height=2,
        width=3,
        bounds=(10, 1.8, 10.3, 2),
        validity=validity,
    )
    validity[0, 0] = False
    assert grid.crs == "EPSG:4326"
    assert grid.shape == (2, 3)
    assert grid.bounds == (10.0, 1.8, 10.3, 2.0)
    assert grid.validity is not None
    assert bool(grid.validity[0, 0])
    with pytest.raises((AttributeError, ValueError)):
        grid.validity[0, 0] = False


def test_grid_spec_rejects_invalid_bounds_and_transform() -> None:
    with pytest.raises(ValueError):
        GridSpec(
            crs="EPSG:4326",
            transform=Affine(1, 0, 0, 0, -1, 2),
            height=2,
            width=2,
            bounds=(0, 0, 3, 2),
        )
    with pytest.raises(ValueError):
        GridSpec(
            crs="EPSG:4326",
            transform=Affine(1, 1, 0, 0, -1, 2),
            height=2,
            width=2,
        )
