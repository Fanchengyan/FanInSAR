"""Tests for residual azimuth phase ramp estimation and removal."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from faninsar.processing.interferometry.flatten import (
    azimuth_ramp_device_kwargs,
    estimate_residual_azimuth_ramp,
    remove_azimuth_phase_ramp,
)
from faninsar.processing.pipeline.production import stage_flatten


def _scalar_numpy_oracle(
    ifg: np.ndarray,
    reference: np.ndarray,
    *,
    coherence: np.ndarray | None = None,
    coh_thr: float = 0.2,
    search: tuple[float, float] = (-0.5, 0.5),
    n_grid: int = 201,
) -> float:
    """Original per-candidate nansum loop used as an independent oracle."""
    phase = np.angle(ifg)
    mask = np.isfinite(phase) & np.isfinite(reference) & (np.abs(ifg) > 0)
    if coherence is not None:
        coherence_array = np.asarray(coherence)
        mask = mask & np.isfinite(coherence_array) & (coherence_array >= coh_thr)
        weights = np.where(mask, coherence_array, 0.0)
    else:
        weights = mask.astype(np.float64)
    if not np.any(mask):
        return 0.0
    azimuth = np.arange(phase.shape[0], dtype=np.float64)[:, None]
    residual = phase - reference
    best_candidate = 0.0
    best_score = -1.0
    for candidate in np.linspace(search[0], search[1], int(n_grid)):
        score = float(
            np.abs(np.nansum(weights * np.exp(1j * (residual - candidate * azimuth))))
        )
        if score > best_score:
            best_score = score
            best_candidate = float(candidate)
    return best_candidate


def _masked_ramp_field(
    height: int = 64,
    width: int = 96,
    coefficient: float = -0.17,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic residual-ramp field with invalid weighted pixels."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    reference = 0.03 * xx + 0.01 * yy
    ifg = np.exp(1j * (reference + coefficient * yy)).astype(np.complex64)
    coherence = np.ones((height, width), dtype=np.float32)
    ifg[0, 0] = np.nan + 1j * np.nan
    ifg[10, 11] = 0.0 + 0.0j
    reference[20, 21] = np.nan
    coherence[30, 31] = np.nan
    coherence[40, 41] = 0.1
    return ifg, reference, coherence


def test_estimate_and_remove_linear_azimuth_ramp() -> None:
    """Known linear az ramp on top of reference phase is recovered and removed."""
    height, width = 40, 80
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    # Smooth geometric-like reference (range ramp + mild 2D).
    ref_phase = 0.08 * xx + 0.01 * yy * (xx / width)
    true_c = -0.17
    ph = np.angle(np.exp(1j * (ref_phase + true_c * yy)))
    ifg = np.exp(1j * ph).astype(np.complex64)
    coh = np.ones((height, width), dtype=np.float32)

    est = estimate_residual_azimuth_ramp(ifg, ref_phase, coherence=coh)
    assert abs(est - true_c) < 0.02

    fixed = remove_azimuth_phase_ramp(ifg, est)
    residual = np.angle(fixed * np.exp(-1j * ref_phase))
    assert float(np.sqrt(np.mean(residual**2))) < 0.1
    assert float(np.abs(np.mean(np.exp(1j * residual)))) > 0.95


def test_remove_azimuth_ramp_zero_is_noop() -> None:
    """Zero ramp coefficient leaves the array unchanged."""
    ifg = np.ones((8, 8), dtype=np.complex64) * (1 + 1j)
    out = remove_azimuth_phase_ramp(ifg, 0.0)
    np.testing.assert_array_equal(out, ifg)


def test_vectorized_numpy_matches_scalar_oracle() -> None:
    """The row-reduction NumPy path preserves the original coefficient contract."""
    ifg, reference, coherence = _masked_ramp_field()
    expected = _scalar_numpy_oracle(ifg, reference, coherence=coherence)
    actual = estimate_residual_azimuth_ramp(
        ifg,
        reference,
        coherence=coherence,
        executor="numpy",
    )
    assert actual == expected


def test_numpy_all_invalid_returns_zero() -> None:
    """An empty validity mask returns the documented zero sentinel."""
    ifg = np.zeros((8, 12), dtype=np.complex64)
    reference = np.full(ifg.shape, np.nan, dtype=np.float64)
    assert estimate_residual_azimuth_ramp(ifg, reference) == 0.0


def test_torch_ramp_matches_numpy_on_masked_field() -> None:
    """Eager Torch CPU scoring matches the portable NumPy coefficient."""
    pytest.importorskip("torch")
    ifg, reference, coherence = _masked_ramp_field()
    expected = estimate_residual_azimuth_ramp(
        ifg,
        reference,
        coherence=coherence,
        executor="numpy",
    )
    actual = estimate_residual_azimuth_ramp(
        ifg,
        reference,
        coherence=coherence,
        executor="torch",
        device="cpu",
        candidate_chunk=7,
    )
    assert actual == expected


def test_torch_ramp_all_invalid_returns_zero() -> None:
    """Torch retains the all-invalid zero sentinel."""
    pytest.importorskip("torch")
    ifg = np.zeros((8, 12), dtype=np.complex64)
    reference = np.full(ifg.shape, np.nan, dtype=np.float64)
    actual = estimate_residual_azimuth_ramp(
        ifg,
        reference,
        executor="torch",
        device="cpu",
    )
    assert actual == 0.0


def test_torch_ramp_first_candidate_wins_exact_tie() -> None:
    """Exact score ties keep the lowest-index candidate on both executors."""
    pytest.importorskip("torch")
    tied = np.ones((1, 4), dtype=np.complex128)
    reference = np.zeros_like(tied.real)
    kwargs = {"search": (1.0, 2.0), "n_grid": 2}
    numpy_result = estimate_residual_azimuth_ramp(
        tied, reference, executor="numpy", **kwargs
    )
    torch_result = estimate_residual_azimuth_ramp(
        tied, reference, executor="torch", device="cpu", candidate_chunk=1, **kwargs
    )
    assert numpy_result == torch_result == 1.0


def test_torch_ramp_rejects_nonpositive_candidate_chunk() -> None:
    """Torch candidate workspace sizes must be positive."""
    with pytest.raises(ValueError, match="candidate_chunk"):
        estimate_residual_azimuth_ramp(
            np.ones((4, 4), dtype=np.complex64),
            np.zeros((4, 4), dtype=np.float64),
            executor="torch",
            candidate_chunk=0,
        )


def test_torch_ramp_rejects_unavailable_cuda() -> None:
    """Explicit CUDA fails closed when the device is absent."""
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this host")
    with pytest.raises(RuntimeError, match="CUDA"):
        estimate_residual_azimuth_ramp(
            np.ones((4, 4), dtype=np.complex64),
            np.zeros((4, 4), dtype=np.float64),
            executor="torch",
            device="cuda",
        )


def test_torch_ramp_uses_shared_device_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch scoring resolves the Stack device through the shared helper."""
    pytest.importorskip("torch")
    import torch

    from faninsar.processing import torch_kernels

    seen: list[object] = []
    real_resolve = torch_kernels.resolve_torch_device

    def _capture(device: object) -> torch.device:
        seen.append(device)
        return real_resolve("cpu")

    monkeypatch.setattr(torch_kernels, "resolve_torch_device", _capture)
    ifg, reference, coherence = _masked_ramp_field()
    estimate_residual_azimuth_ramp(
        ifg,
        reference,
        coherence=coherence,
        executor="torch",
        device="cpu",
        n_grid=5,
    )
    assert seen == ["cpu"]


def test_azimuth_ramp_device_kwargs_forward_stack_device() -> None:
    """A Stack device selects Torch on CPU/CUDA and NumPy on MPS."""
    assert azimuth_ramp_device_kwargs("cuda") == {
        "executor": "torch",
        "device": "cuda",
    }
    assert azimuth_ramp_device_kwargs("cpu") == {
        "executor": "torch",
        "device": "cpu",
    }
    assert azimuth_ramp_device_kwargs("auto") == {
        "executor": "torch",
        "device": "auto",
    }
    assert azimuth_ramp_device_kwargs("mps") == {"executor": "numpy"}


def test_stage_flatten_forwards_device_into_ramp() -> None:
    """Production flatten must pass the Stack device into ramp scoring."""
    source = inspect.getsource(stage_flatten)
    assert "azimuth_ramp_device_kwargs(device)" in source
    assert source.count("azimuth_ramp_device_kwargs(device)") == 2
