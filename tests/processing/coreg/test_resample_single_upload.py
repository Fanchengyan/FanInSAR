"""Tests that dask-torch resample_complex uploads the source SLC once."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.coreg.offsets import resample_complex

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
        executor="dask-torch",
        device="cpu",
    )

    assert out.shape == samples.shape
    assert out.dtype == samples.dtype
    assert upload_count["n"] == 1, (
        f"expected 1 source upload, got {upload_count['n']} (row tiles={n_tiles})"
    )


def test_resample_complex_dask_torch_matches_serial() -> None:
    """dask-torch path stays bit-close to the serial NumPy Lanczos path."""
    rng = np.random.default_rng(11)
    samples = (rng.normal(size=(24, 20)) + 1j * rng.normal(size=(24, 20))).astype(
        np.complex64
    )
    rg = np.full(samples.shape, 0.4, dtype=np.float64)
    az = np.full(samples.shape, -0.3, dtype=np.float64)

    serial = resample_complex(
        samples,
        range_offset_px=rg,
        azimuth_offset_px=az,
        row_chunk=5,
        executor="serial",
    )
    dask_torch = resample_complex(
        samples,
        range_offset_px=rg,
        azimuth_offset_px=az,
        row_chunk=5,
        executor="dask-torch",
        device="cpu",
    )

    max_delta = float(np.max(np.abs(dask_torch - serial)))
    assert max_delta < 1e-5, f"max|Δ|={max_delta}"
