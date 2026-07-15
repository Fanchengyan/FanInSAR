"""1D temporal / network IRLS phase unwrapping (numpy in, numpy out).

Orthogonal to 2D spatial unwrap and to batch least-squares inversion.
Default device is CPU (small n_pairs x n_intervals systems; R2.5 / O1D-1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from faninsar._core.device import parse_device
from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.unwrap._incidence import build_incidence_matrix

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faninsar import Pairs

logger = setup_logger(__name__)

_TWO_PI: float = 2.0 * np.pi
_SINGULAR_COND: float = 1.0e12

@dataclass(frozen=True, slots=True)
class TemporalUnwrapResult:
    """Unwrapped pair phases, integer 2π corrections, and IRLS diagnostics."""

    phase_unw: np.ndarray
    corrections_k: np.ndarray
    iterations: int
    converged: bool
    method: str
    device: str


def _to_tensor(
    arr: np.ndarray,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = torch.float64 if device.type == "cpu" else torch.float32
    return torch.as_tensor(arr, dtype=dtype, device=device)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _cleanup_gpu(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _solve_weighted_batch(
    a_t: torch.Tensor,
    phi: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Solve ``(A^T W A) x = A^T W phi`` per pixel; singular pixels -> NaN."""
    atwa = torch.einsum("pi,pb,pj->bij", a_t, weight, a_t)
    atwb = torch.einsum("pi,pb->bi", a_t, weight * phi)

    n_int = a_t.shape[1]
    batch = phi.shape[1]
    x_out = torch.full(
        (n_int, batch),
        float("nan"),
        dtype=phi.dtype,
        device=phi.device,
    )

    try:
        cond = torch.linalg.cond(atwa)
    except RuntimeError:
        cond = torch.full((batch,), float("inf"), dtype=phi.dtype, device=phi.device)

    diag_mass = torch.diagonal(atwa, dim1=-2, dim2=-1).abs().sum(dim=-1)
    valid = torch.isfinite(cond) & (cond < _SINGULAR_COND) & (diag_mass > 0)

    n_valid = int(valid.sum().item())
    if n_valid == 0:
        logger.warning(
            "temporal IRLS: all %s pixels in batch are singular / zero-weight",
            batch,
        )
        return x_out

    atwa_v = atwa[valid]
    atwb_v = atwb[valid]
    try:
        x_v = torch.linalg.solve(atwa_v, atwb_v)
    except RuntimeError:
        logger.warning(
            "temporal IRLS: torch.linalg.solve failed; falling back to lstsq",
        )
        try:
            x_v = torch.linalg.lstsq(atwa_v, atwb_v.unsqueeze(-1)).solution.squeeze(-1)
        except RuntimeError:
            logger.exception("temporal IRLS: lstsq fallback also failed for batch")
            return x_out

    x_out[:, valid] = x_v.T
    n_singular = batch - n_valid
    if n_singular > 0:
        logger.warning(
            "temporal IRLS: %s / %s pixels singular -> NaN",
            n_singular,
            batch,
        )
    return x_out


def unwrap_temporal_irls(
    phase_stack: np.ndarray,
    pair_dates: Pairs | Sequence[tuple[str, str]] | np.ndarray,
    *,
    weights: np.ndarray | None = None,
    wrapped_input: bool = True,
    max_iter: int = 10,
    tol: float = 1e-3,
    epsilon: float = 1e-3,
    device: str = "cpu",
    batch_pixels: int = 50000,
) -> TemporalUnwrapResult:
    """Unwrap a pair-phase stack with 1D temporal/network IRLS.

    Parameters
    ----------
    phase_stack : numpy.ndarray
        Pair phases with shape ``(n_pairs, H, W)`` or ``(n_pairs, n_points)``.
    pair_dates : Pairs or sequence of (primary, secondary) or ndarray
        Temporal pair definitions aligned with axis 0 of ``phase_stack``.
        When a :class:`~faninsar.Pairs` instance is given, its
        :meth:`~faninsar.Pairs.sbas_matrix` is reused.
    weights : numpy.ndarray, optional
        Per-pair weights matching ``phase_stack``. Defaults to ones. NaN
        samples in ``phase_stack`` are zero-weighted.
    wrapped_input : bool, optional
        If ``True`` (default), residuals are wrapped via ``atan2`` each
        iteration. If ``False``, residuals are treated as already-unwrapped.
        There is no ``"auto"`` mode.
    max_iter : int, optional
        Maximum IRLS iterations per pixel batch.
    tol : float, optional
        Relative weight-change / step-size tolerance for convergence.
    epsilon : float, optional
        Stability floor for L1-type weights ``W = 1 / (|r| + eps)``.
    device : str, optional
        Torch device in ``{"cpu", "cuda", "auto"}``. Default ``"cpu"``
        (preferred). Explicit ``"cuda"`` without a GPU raises.
    batch_pixels : int, optional
        Number of spatial samples processed per batch.

    Returns
    -------
    TemporalUnwrapResult
        Unwrapped pair phases, integer corrections, and diagnostics.

    Notes
    -----
    Gauss-Newton IRLS: ``x <- x + dx`` with ``A dx ~ r`` and
    ``r = wrap(phi - A x)`` (or plain residual). Final
    ``k = round((A x - phi) / 2pi)``, ``phi_unw = phi + 2pi k``.
    Torch is internal; public contract is numpy in / numpy out.

    """
    phase = np.asarray(phase_stack, dtype=np.float64)
    if phase.ndim < 2:
        reject_invalid_state(
            "phase_stack must have shape (n_pairs, H, W) or (n_pairs, n_points)",
        )

    n_pairs = phase.shape[0]
    spatial_shape = phase.shape[1:]
    n_points = int(np.prod(spatial_shape))
    phase_flat = phase.reshape(n_pairs, n_points)

    a_mat, _pair_ids = build_incidence_matrix(pair_dates)
    if a_mat.shape[0] != n_pairs:
        reject_invalid_state(
            f"pair_dates rows ({a_mat.shape[0]}) must match phase_stack "
            f"axis-0 ({n_pairs})",
        )

    if weights is None:
        weight_flat = np.ones_like(phase_flat, dtype=np.float64)
    else:
        weight_arr = np.asarray(weights, dtype=np.float64)
        if weight_arr.shape != phase.shape:
            reject_invalid_state("weights must match phase_stack shape")
        weight_flat = weight_arr.reshape(n_pairs, n_points)

    invalid = ~np.isfinite(phase_flat)
    if invalid.any():
        weight_flat = weight_flat.copy()
        weight_flat[invalid] = 0.0
        phase_flat = phase_flat.copy()
        phase_flat[invalid] = 0.0

    torch_device = parse_device(device)
    dtype = torch.float64 if torch_device.type == "cpu" else torch.float32
    a_t = _to_tensor(a_mat, torch_device, dtype=dtype)

    phase_unw_flat = np.full_like(phase_flat, np.nan, dtype=np.float64)
    corrections_flat = np.full_like(phase_flat, np.nan, dtype=np.float64)
    last_iterations = 0
    last_converged = False

    if batch_pixels < 1:
        reject_invalid_state("batch_pixels must be >= 1")

    for start in range(0, n_points, batch_pixels):
        stop = min(start + batch_pixels, n_points)
        batch = stop - start

        phi = _to_tensor(phase_flat[:, start:stop], torch_device, dtype=dtype)
        base_w = _to_tensor(weight_flat[:, start:stop], torch_device, dtype=dtype)
        w_irls = base_w / epsilon
        prev_w = w_irls.clone()
        # Gauss-Newton IRLS: x <- x + dx, A dx ~ r
        x = torch.zeros((a_mat.shape[1], batch), dtype=dtype, device=torch_device)
        converged = False
        iterations = 0

        for it in range(1, max_iter + 1):
            iterations = it
            recon = a_t @ x
            res = phi - recon
            if wrapped_input:
                res = torch.atan2(torch.sin(res), torch.cos(res))
            dx = _solve_weighted_batch(a_t, res, w_irls)
            x = x + dx
            w_irls = base_w / (res.abs() + epsilon)

            delta_w = float(
                (
                    torch.linalg.vector_norm(w_irls - prev_w)
                    / (torch.linalg.vector_norm(prev_w) + 1e-12)
                ).item(),
            )
            dx_finite = dx[:, torch.isfinite(dx).all(dim=0)]
            delta_x = (
                float(torch.linalg.vector_norm(dx_finite).item())
                if dx_finite.numel() > 0
                else 0.0
            )
            prev_w = w_irls.clone()
            if delta_w < tol or delta_x < tol:
                converged = True
                break

        recon_final = a_t @ x
        k = torch.round((recon_final - phi) / _TWO_PI)
        phi_unw = phi + _TWO_PI * k

        bad = torch.isnan(x).any(dim=0)
        if bad.any():
            phi_unw[:, bad] = float("nan")
            k[:, bad] = float("nan")

        phase_unw_flat[:, start:stop] = _to_numpy(phi_unw)
        corrections_flat[:, start:stop] = _to_numpy(k)
        last_iterations = iterations
        last_converged = converged

        del phi, base_w, w_irls, prev_w, x, recon_final, k, phi_unw, dx
        _cleanup_gpu(torch_device)

    return TemporalUnwrapResult(
        phase_unw=phase_unw_flat.reshape(n_pairs, *spatial_shape),
        corrections_k=corrections_flat.reshape(n_pairs, *spatial_shape),
        iterations=last_iterations,
        converged=last_converged,
        method="temporal_irls",
        device=str(torch_device),
    )
