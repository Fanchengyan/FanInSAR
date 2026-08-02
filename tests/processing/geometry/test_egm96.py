"""Tests for the EGM96 geoid undulation model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.processing.geometry.egm96 import (
    EGM96Geoid,
    _geoid_undulation_point,
    default_egm96_path,
    load_egm96_coefficients,
)

if TYPE_CHECKING:
    from pathlib import Path

    from faninsar.processing.geometry.egm96 import _EGM96Coefficients


@pytest.fixture(scope="module")
def coefficients() -> _EGM96Coefficients | None:
    """Return EGM96 coefficients, skipping when the data file is absent."""
    path = default_egm96_path()
    if path is None:
        pytest.skip("EGM96 coefficient file not available")
    return load_egm96_coefficients(path)


def test_geoid_undulation_matches_isce2_corner_heights(
    coefficients: _EGM96Coefficients | None,
) -> None:
    """Undulation near Qinghai matches ISCE2's sampled geoid corner range.

    ISCE2's verifyDEM logged corner geoid heights of -44.6 to -50.4 m for
    the GLO-30 DEM covering 98-100.5 E, 37-38.5 N.  The model must produce
    values in that band with a systematic (geoid) signature, not near zero.
    """
    latitudes = np.array([36.85, 36.85, 38.65, 38.65])
    longitudes = np.array([100.65, 97.85, 97.85, 100.65])
    values = np.array(
        [
            _geoid_undulation_point(float(lat), float(lon), coefficients)
            for lat, lon in zip(latitudes, longitudes, strict=True)
        ]
    )
    assert np.all(values < -40.0)
    assert np.all(values > -55.0)
    assert float(np.std(values)) > 1.0


def test_geoid_sampler_interpolates_bilinear() -> None:
    """The grid sampler returns smooth undulation values via bilinear fit."""
    if default_egm96_path() is None:
        pytest.skip("EGM96 coefficient file not available")
    geoid = EGM96Geoid()
    latitudes = np.linspace(37.5, 38.0, 6)
    longitudes = np.linspace(98.5, 99.5, 6)
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    values = geoid.sample(lat_grid, lon_grid)
    assert values.shape == lat_grid.shape
    assert np.all(np.isfinite(values))
    assert float(np.max(np.abs(np.diff(values, axis=0)))) < 1.0
    assert float(np.max(np.abs(np.diff(values, axis=1)))) < 1.0


def test_missing_coefficient_file_raises(tmp_path: Path) -> None:
    """A missing coefficient file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_egm96_coefficients(tmp_path / "missing.dat")
