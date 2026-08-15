"""Tests for raster DEM sampling and cache reuse."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from affine import Affine

from faninsar.processing.geometry import RasterDEM, torch_kernels
from faninsar.processing.geometry.dem import _natural_spline_six

if TYPE_CHECKING:
    from pathlib import Path


def test_raster_dem_reuses_loaded_height_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse the decoded DEM band across repeated coordinate batches."""
    path = tmp_path / "dem.tif"
    path.touch()
    raster = SimpleNamespace(
        nodata=None,
        transform=Affine.identity(),
        read_count=0,
    )

    def read(band: int) -> np.ndarray:
        assert band == 1
        raster.read_count += 1
        return np.arange(16, dtype=np.float32).reshape(4, 4)

    raster.read = read
    monkeypatch.setattr(RasterDEM, "_open", lambda _self: raster)
    dem = RasterDEM(path, interpolation="bilinear")
    latitude = np.array([1.25])
    longitude = np.array([1.5])

    first = dem.sample(latitude, longitude)
    second = dem.sample(latitude, longitude)

    assert raster.read_count == 1
    assert np.array_equal(first, second)


def test_raster_dem_bilinear_uses_pixel_area_convention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bilinear sampling uses the pixel-area convention (matches ISCE2)."""
    path = tmp_path / "dem.tif"
    path.touch()
    raster = SimpleNamespace(
        nodata=None,
        transform=Affine.translation(10.0, 20.0) * Affine.scale(2.0, -2.0),
        read=lambda _band: np.arange(16, dtype=np.float32).reshape(4, 4),
    )
    monkeypatch.setattr(RasterDEM, "_open", lambda _self: raster)
    dem = RasterDEM(path, interpolation="bilinear")

    # Pixel-area convention: integer lon/lat indexes the lower edge of the
    # containing pixel, so a point at a pixel corner bilinearly blends the
    # four neighbours (no half-pixel centre shift).
    sampled = dem.sample(
        latitude_deg=np.array([19.0, 17.0]),
        longitude_deg=np.array([11.0, 13.0]),
    )

    assert np.allclose(sampled, np.array([2.5, 7.5]))

    # A point exactly on a pixel boundary belongs to the next pixel.
    corner = dem.sample(
        latitude_deg=np.array([18.0]),
        longitude_deg=np.array([12.0]),
    )
    assert np.allclose(corner, np.array([5.0]))


def test_isce_six_sample_spline_preserves_linear_surfaces() -> None:
    """The local ISCE2 spline must exactly reproduce a linear height profile."""
    samples = 20.0 + 3.0 * np.arange(6, dtype=np.float64)
    fractions = np.array([0.0, 0.25, 0.75, 1.0])
    tiled = np.broadcast_to(samples, (fractions.size, samples.size))

    interpolated = _natural_spline_six(tiled, fractions)

    np.testing.assert_allclose(interpolated, 23.0 + 3.0 * fractions)


def test_torch_spline_weights_match_recursive_float64_reference() -> None:
    """Closed-form six-point weights preserve the recursive spline values."""
    torch = pytest.importorskip("torch")
    values = torch.tensor(
        [
            [0.5, 1.25, -2.0, 4.0, 3.5, -1.0],
            [10.0, -4.0, 2.0, 8.0, 0.25, 5.0],
            [-3.0, 7.0, 11.0, -2.5, 6.0, 1.0],
        ],
        dtype=torch.float64,
    )
    fractions = torch.tensor([0.0, 0.37, 1.0], dtype=torch.float64)

    second = [torch.zeros_like(fractions) for _ in range(6)]
    recurrence = [torch.zeros_like(fractions) for _ in range(6)]
    for index in range(1, 5):
        denominator = recurrence[index - 1] / 2.0 + 2.0
        recurrence[index] = -0.5 / denominator
        second[index] = (
            3.0 * (values[:, index + 1] - 2.0 * values[:, index] + values[:, index - 1])
            - second[index - 1] / 2.0
        ) / denominator
    for index in range(4, 0, -1):
        second[index] = recurrence[index] * second[index + 1] + second[index]
    recursive = values[:, 1] + fractions * (
        values[:, 2]
        - values[:, 1]
        - second[1] / 3.0
        - second[2] / 6.0
        + fractions * (second[1] / 2.0 + fractions * (second[2] - second[1]) / 6.0)
    )
    weights = torch_kernels._spline_six_weights(fractions)
    weighted = torch.sum(values * weights, dim=-1)

    torch.testing.assert_close(weighted, recursive, rtol=1.0e-12, atol=1.0e-12)
