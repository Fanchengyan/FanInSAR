"""Tests for raster DEM sampling and cache reuse."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from affine import Affine

from faninsar.processing.dem import ConstantDEM
from faninsar.processing.geometry import torch_kernels
from faninsar.processing.geometry.dem import (
    GeoidAdjustedDEM,
    RasterDEM,
    _natural_spline_six,
    clone_raster_dem,
    pin_dem_sampler_device,
)

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


def test_cpu_and_cuda_biquintic_sample_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA RasterDEM.sample matches the CPU NumPy oracle (PROPOSAL-0032 §1)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for DEM identity")
    path = tmp_path / "dem.tif"
    path.touch()
    heights = np.arange(64, dtype=np.float32).reshape(8, 8)
    raster = SimpleNamespace(
        nodata=None,
        transform=Affine.identity(),
        read=lambda _band: heights.copy(),
    )
    monkeypatch.setattr(RasterDEM, "_open", lambda _self: raster)
    cpu = RasterDEM(path, interpolation="biquintic", device="cpu")
    gpu = RasterDEM(path, interpolation="biquintic", device="cuda")
    rng = np.random.default_rng(32)
    latitude = rng.uniform(1.5, 5.5, size=32)
    longitude = rng.uniform(1.5, 5.5, size=32)
    cpu_h = cpu.sample(latitude, longitude)
    gpu_h = gpu.sample(latitude, longitude)
    np.testing.assert_allclose(gpu_h, cpu_h, atol=1.0e-9)
    np.testing.assert_equal(np.isnan(gpu_h), np.isnan(cpu_h))


def test_cuda_biquintic_reuses_device_height_tensor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second CUDA sample must not re-upload the full DEM raster."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for DEM tensor reuse")
    path = tmp_path / "dem.tif"
    path.touch()
    raster = SimpleNamespace(
        nodata=None,
        transform=Affine.identity(),
        read=lambda _band: np.arange(36, dtype=np.float32).reshape(6, 6),
    )
    monkeypatch.setattr(RasterDEM, "_open", lambda _self: raster)
    dem = RasterDEM(path, interpolation="biquintic", device="cuda")
    latitude = np.array([2.2, 3.4])
    longitude = np.array([2.3, 3.1])
    first = dem.sample(latitude, longitude)
    token = dem._height_tensor
    assert token is not None
    second = dem.sample(latitude, longitude)
    assert dem._height_tensor is token
    np.testing.assert_allclose(first, second)


def test_admit_cpu_identity_is_cpu() -> None:
    """Explicit cpu does not require a second auto admission."""
    from faninsar.processing.geometry.dem import admit_dem_device_identity

    assert admit_dem_device_identity("cpu") == "cpu"


def test_pickle_missing_device_restores_cpu(tmp_path: Path) -> None:
    """Pre-change pickles without device restore the historical CPU identity."""
    path = tmp_path / "dem.tif"
    path.touch()
    dem = RasterDEM(path, interpolation="bilinear", device="cpu")
    state = dem.__getstate__()
    del state["device"]
    restored = RasterDEM.__new__(RasterDEM)
    restored.__setstate__(state)
    assert restored.device == "cpu"


def test_bilinear_on_cuda_identity_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bilinear has no CUDA identity and must not silently stay on NumPy."""
    path = tmp_path / "dem.tif"
    path.touch()
    dem = RasterDEM(path, interpolation="bilinear", device="cpu")
    monkeypatch.setattr(RasterDEM, "_resolved_identity", lambda _self: "cuda")
    with pytest.raises(ValueError, match="no identity"):
        _ = dem.sample(np.array([0.0]), np.array([0.0]))


def test_mps_identity_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-CPU, non-CUDA identities have no DEM sampler."""
    path = tmp_path / "dem.tif"
    path.touch()
    dem = RasterDEM(path, interpolation="biquintic", device="cpu")
    monkeypatch.setattr(RasterDEM, "_resolved_identity", lambda _self: "mps")
    with pytest.raises(ValueError, match="no identity"):
        _ = dem.sample(np.array([0.0]), np.array([0.0]))


def test_zero_chunk_points_fail_closed(tmp_path: Path) -> None:
    """sample_chunk_points <= 0 is rejected at construction."""
    path = tmp_path / "dem.tif"
    path.touch()
    with pytest.raises(ValueError, match="sample_chunk_points"):
        _ = RasterDEM(path, sample_chunk_points=0)


def test_pin_dem_sampler_device_walks_inner_sampler(tmp_path: Path) -> None:
    """pin_dem_sampler_device binds RasterDEM and nested GeoidAdjustedDEM."""
    path = tmp_path / "dem.tif"
    path.touch()
    raster = RasterDEM(path, interpolation="bilinear", device="cpu")
    assert pin_dem_sampler_device(raster, "cpu") is raster

    moved = pin_dem_sampler_device(raster, "cuda")
    assert moved is not raster
    assert isinstance(moved, RasterDEM)
    assert moved.device == "cuda"
    assert raster.device == "cpu"

    wrapped = GeoidAdjustedDEM(raster, ConstantDEM(0.0))
    pinned = pin_dem_sampler_device(wrapped, "cuda")
    assert isinstance(pinned, GeoidAdjustedDEM)
    assert pinned is not wrapped
    assert pinned.orthometric_dem is not raster
    assert isinstance(pinned.orthometric_dem, RasterDEM)
    assert pinned.orthometric_dem.device == "cuda"
    assert pinned.geoid is wrapped.geoid


def test_clone_raster_dem_copies_device_and_sample_chunk_points(
    tmp_path: Path,
) -> None:
    """clone_raster_dem copies device identity and sample_chunk_points."""
    path = tmp_path / "dem.tif"
    path.touch()
    dem = RasterDEM(
        path,
        interpolation="bilinear",
        device="cpu",
        sample_chunk_points=1234,
    )
    cloned = clone_raster_dem(dem)
    assert cloned is not dem
    assert cloned.device == "cpu"
    assert cloned.sample_chunk_points == 1234
    assert cloned.interpolation == "bilinear"
    assert cloned.path == dem.path

    retargeted = clone_raster_dem(dem, device="cuda")
    assert retargeted.device == "cuda"
    assert retargeted.sample_chunk_points == 1234


def test_cpu_biquintic_sample_on_eight_by_eight_ramp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU biquintic sampling interpolates an 8x8 linear ramp."""
    path = tmp_path / "dem.tif"
    path.touch()
    ramp = np.arange(64, dtype=np.float32).reshape(8, 8)
    raster = SimpleNamespace(
        nodata=None,
        transform=Affine.identity(),
        read=lambda _band: ramp,
    )
    monkeypatch.setattr(RasterDEM, "_open", lambda _self: raster)
    dem = RasterDEM(path, interpolation="biquintic", device="cpu")
    sampled = dem.sample(
        latitude_deg=np.array([2.25]),
        longitude_deg=np.array([2.5]),
    )
    assert sampled.shape == (1,)
    assert np.isfinite(sampled).all()
    np.testing.assert_allclose(sampled, np.array([20.5]), atol=1.0e-6)


def test_numpy_and_torch_natural_spline_six_match() -> None:
    """NumPy and Torch six-sample splines agree on a small random window."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    window = rng.standard_normal((5, 6))
    fractions = rng.random(5)
    numpy_out = _natural_spline_six(window, fractions)
    torch_out = torch_kernels._natural_spline_six(
        torch.as_tensor(window, dtype=torch.float64),
        torch.as_tensor(fractions, dtype=torch.float64),
    )
    np.testing.assert_allclose(numpy_out, torch_out.detach().cpu().numpy(), atol=1.0e-12)
