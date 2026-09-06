"""Tests that dask-torch resample_complex uploads the source SLC once."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec

import numpy as np
import pytest

from faninsar.processing.coregistration.offsets import resample_complex
from faninsar.processing.resampling import lanczos_resample
from faninsar.processing.resampling_torch import lanczos_resample_torch

torch = pytest.importorskip("torch")


def test_resample_complex_dask_torch_single_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a multi-tile dask-torch resample, source H2D happens once.

    Forces the P7 resident path via torch CPU (no CUDA required) and counts
    2-D source uploads through ``torch.from_numpy``. Row tiles must only move
    coordinates, not re-upload the full SLC.
    """
    from faninsar.processing import resampling_torch as rt

    rng = np.random.default_rng(7)
    height, width = 20, 16
    row_chunk = 4
    n_tiles = (height + row_chunk - 1) // row_chunk
    assert n_tiles > 1

    real = rng.normal(size=(height, width))
    imag = rng.normal(size=(height, width))
    samples = (real + 1j * imag).astype(np.complex64)
    rg = np.full(samples.shape, 0.35, dtype=np.float64)
    az = np.full(samples.shape, -0.25, dtype=np.float64)

    # Exercise the resident-tensor path without requiring CUDA.
    monkeypatch.setattr(
        rt,
        "_resolve_torch_device",
        lambda _device: torch.device("cpu"),
    )

    upload_count = {"n": 0}
    real_from_numpy = torch.from_numpy

    def counting_from_numpy(arr: np.ndarray) -> object:
        tensor = real_from_numpy(arr)
        # Count full-source uploads only (2-D, same shape as samples).
        if getattr(arr, "ndim", 0) == 2 and arr.shape == samples.shape:
            upload_count["n"] += 1
        return tensor

    monkeypatch.setattr(torch, "from_numpy", counting_from_numpy)

    out = resample_complex(
        samples,
        range_offset_px=rg,
        azimuth_offset_px=az,
        row_chunk=row_chunk,
        executor="torch",
        device="cpu",
    )

    assert out.shape == samples.shape
    assert out.dtype == samples.dtype
    assert upload_count["n"] == 1, (
        f"expected 1 source upload, got {upload_count['n']} (row tiles={n_tiles})"
    )


def test_resample_complex_torch_matches_public_api() -> None:
    """Coregistration and public API use the same Torch Lanczos kernel."""
    rng = np.random.default_rng(11)
    samples = (rng.normal(size=(24, 20)) + 1j * rng.normal(size=(24, 20))).astype(
        np.complex64
    )
    rg = np.full(samples.shape, 0.4, dtype=np.float64)
    az = np.full(samples.shape, -0.3, dtype=np.float64)

    actual = resample_complex(
        samples,
        range_offset_px=rg,
        azimuth_offset_px=az,
        row_chunk=5,
        executor="torch",
        device="cpu",
    )
    rows, columns = np.mgrid[: samples.shape[0], : samples.shape[1]]
    coordinates = np.vstack([(rows - az).ravel(), (columns - rg).ravel()])
    expected = lanczos_resample(samples, coordinates, device="cpu").reshape(
        samples.shape
    )

    max_delta = float(np.max(np.abs(actual - expected)))
    assert max_delta < 1e-5, f"max|Δ|={max_delta}"


def test_resample_complex_torch_cpu_does_not_use_numpy_dask() -> None:
    """Keep the portable CPU executor on the unified Torch kernel."""
    rng = np.random.default_rng(19)
    samples = (rng.normal(size=(24, 20)) + 1j * rng.normal(size=(24, 20))).astype(
        np.complex64
    )
    assert find_spec("faninsar.processing.resampling_dask") is None
    resample_complex(
        samples,
        range_offset_px=0.4,
        azimuth_offset_px=-0.3,
        row_chunk=12,
        executor="torch",
        device="cpu",
    )

@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available",
)
def test_lanczos_mps_accepts_fractional_coordinates() -> None:
    """Run fractional Lanczos coordinates on Apple MPS without float64."""
    rng = np.random.default_rng(23)
    samples = (rng.normal(size=(18, 16)) + 1j * rng.normal(size=(18, 16))).astype(
        np.complex64
    )
    coordinates = np.vstack(
        [
            rng.uniform(1.0, 16.0, 128),
            rng.uniform(1.0, 14.0, 128),
        ]
    )

    expected = lanczos_resample(samples, coordinates, device="cpu")
    actual = lanczos_resample_torch(
        samples,
        coordinates,
        device="mps",
        chunk_size=64,
    )

    assert np.max(np.abs(actual - expected)) < 2e-5


def test_torch_cpu_concurrent_resamples_keep_sources_isolated() -> None:
    """Keep concurrent Torch CPU source tensors isolated across pair tasks."""
    rng = np.random.default_rng(29)
    coordinates = np.vstack(
        [
            rng.uniform(4.0, 59.0, 80_000),
            rng.uniform(4.0, 59.0, 80_000),
        ]
    )
    first = np.full((64, 64), 1.0 + 2.0j, dtype=np.complex64)
    second = np.full((64, 64), -3.0 + 0.5j, dtype=np.complex64)

    def resample(samples: np.ndarray) -> np.ndarray:
        return lanczos_resample_torch(
            samples,
            coordinates,
            device="cpu",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(resample, first)
        second_future = pool.submit(resample, second)
        first_output = first_future.result()
        second_output = second_future.result()

    assert np.allclose(first_output, 1.0 + 2.0j)
    assert np.allclose(second_output, -3.0 + 0.5j)
