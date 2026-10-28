"""Regression tests for calculate_u device handling and loop corrections.

The server failure reproduced here: with NaN-bearing phases and the
default device (``None``), ``calculate_u`` used to keep its tensors on
the CPU while ``batch_lstsq`` resolved ``None`` to the GPU, and the
correction assignment mixed the two devices and raised
``Expected all tensors to be on the same device``.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pytest
import torch

import faninsar as fis
from faninsar import NSBAS, Loops, parse_device


def _loops_case() -> tuple:
    """Return pairs, loops, loop-pair indices, unw, and NaN-bearing phases."""
    names = [
        "20170101_20170113",
        "20170101_20170125",
        "20170101_20170206",
        "20170113_20170125",
        "20170113_20170206",
        "20170125_20170206",
    ]
    pairs = fis.Pairs.from_names(names)
    loops = pairs.build_loops(edge_days=12)
    idx = pairs.where(loops.pairs)
    unw = np.arange(12, dtype=np.float64).reshape(6, 2)
    unw_used = unw[idx].copy()
    unw_used[0, 1] = np.nan  # NaN -> batch_lstsq branch, like real data
    return pairs, loops, idx, unw, unw_used


def _reference_u(loops: Loops, unw_used: np.ndarray) -> np.ndarray:
    """Independent NumPy implementation of the calculate_u contract."""
    c = loops.loop_matrix().astype(np.float64)
    diag_idx = loops.pairs.where(loops.diagonal_pairs)
    cc = c[:, diag_idx]
    closure = c @ unw_used
    u = np.zeros_like(unw_used)
    for j in range(unw_used.shape[1]):
        rows = ~np.isnan(closure[:, j])
        if not rows.any():
            u[diag_idx, j] = np.nan
            continue
        uc, *_ = np.linalg.lstsq(cc[rows], closure[rows, j], rcond=1e-10)
        u[diag_idx, j] = uc / (2 * np.pi)
    return u


def _available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def _dtype_for(device: str) -> torch.dtype:
    return torch.float32 if device == "mps" else torch.float64


class TestCalculateU:
    """Device and correctness regression tests for calculate_u."""

    def test_matches_numpy_reference_on_cpu(self) -> None:
        """CPU results match an independent NumPy reference."""
        _, loops, _, _, unw_used = _loops_case()
        u = NSBAS.calculate_u(loops, unw_used, device="cpu")
        np.testing.assert_allclose(
            u, _reference_u(loops, unw_used), rtol=1e-9, atol=1e-9, equal_nan=True
        )
        # the all-NaN pixel is NaN on the diagonal pairs and zero elsewhere
        diag_idx = loops.pairs.where(loops.diagonal_pairs)
        other = np.setdiff1d(np.arange(u.shape[0]), diag_idx)
        assert np.isnan(u[diag_idx, 1]).all()
        assert (u[other, 1] == 0).all()
        assert np.isfinite(u[:, 0]).all()

    def test_consistent_phases_give_zero_correction(self) -> None:
        """A perfectly consistent network needs no correction."""
        pairs, loops, *_ = _loops_case()
        increments = np.array([[1.0, 0.5], [-0.5, 2.0], [0.3, -1.0]])
        unw_consistent = pairs.sbas_matrix().astype(np.float64) @ increments
        u = NSBAS.calculate_u(loops, unw_consistent, device="cpu")
        np.testing.assert_allclose(u, 0.0, atol=1e-10)

    def test_default_device_matches_reference(self) -> None:
        """The server call (device=None, NaN input) must not mix devices."""
        _, loops, _, _, unw_used = _loops_case()
        resolved = str(parse_device(None))
        dtype = _dtype_for(resolved)
        try:
            u_default = NSBAS.calculate_u(loops, unw_used, dtype=dtype)
            u_tensor = NSBAS.calculate_u(
                loops, unw_used, dtype=dtype, return_numpy=False
            )
        except NotImplementedError as exc:  # pragma: no cover
            pytest.skip(f"{resolved} lacks required operators: {exc}")
        assert str(u_tensor.device).startswith(resolved)
        np.testing.assert_allclose(
            u_default,
            _reference_u(loops, unw_used),
            rtol=1e-5,
            atol=1e-5,
            equal_nan=True,
        )

    @pytest.mark.parametrize("device", _available_devices())
    def test_explicit_device_matches_reference(self, device: str) -> None:
        """Explicit devices agree with the NumPy reference."""
        _, loops, _, _, unw_used = _loops_case()
        try:
            u_explicit = NSBAS.calculate_u(
                loops, unw_used, device=device, dtype=_dtype_for(device)
            )
            u_tensor = NSBAS.calculate_u(
                loops,
                unw_used,
                device=device,
                dtype=_dtype_for(device),
                return_numpy=False,
            )
        except NotImplementedError as exc:  # pragma: no cover
            pytest.skip(f"{device} lacks required operators: {exc}")
        assert str(u_tensor.device).startswith(device)
        np.testing.assert_allclose(
            u_explicit,
            _reference_u(loops, unw_used),
            rtol=1e-5,
            atol=1e-5,
            equal_nan=True,
        )

    def test_notebook_flow_on_default_device(self) -> None:
        """The documented round-and-correct recipe works end to end."""
        _, loops, idx, unw, unw_used = _loops_case()
        resolved = str(parse_device(None))
        try:
            u_t = NSBAS.calculate_u(
                loops, unw_used, dtype=_dtype_for(resolved), return_numpy=False
            )
        except NotImplementedError as exc:  # pragma: no cover
            pytest.skip(f"{resolved} lacks required operators: {exc}")
        u = np.zeros_like(unw, dtype=np.float32)
        u[idx] = np.round(u_t.cpu().numpy())
        unw_c = unw - 2 * np.pi * u
        assert unw_c.shape == unw.shape
