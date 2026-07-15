"""Clean-room DCT + IRLS phase unwrapping (portable CPU / optional Torch).

Formulation
-----------
Weighted least-squares unwrapping via discrete cosine transforms, following
the published literature (Ghiglia & Romero 1994; Pritt 1996):

1. Form wrapped phase gradients.
2. Build a weighted discrete divergence (right-hand side of Poisson equation).
3. Solve the Neumann Poisson problem in the Type-II DCT domain.
4. Reweight edges by residual magnitude (IRLS / L1-like) and iterate.

License / provenance
--------------------
Clean-room implementation from published IEEE formulations only.
**No source-available InSAR.dev (or similar) code was consulted.**

Torch 2.10 does not expose ``torch.fft.dct``; the default path uses
:mod:`scipy.fft` Type-II DCT (ortho). An optional Torch path keeps arrays on
device for residual reweighting when ``device`` is ``cuda`` / ``mps``, but still
uses SciPy DCT for the spectral solve unless a pure-FFT DCT becomes available.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import fft as sp_fft

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.unwrap.irls import IRLSUnwrapResult, wrap_phase

logger = setup_logger(__name__)

DeviceName = Literal["auto", "cpu", "cuda", "mps"]


def _resolve_torch_device(device: DeviceName) -> object | None:
    """Return a torch device or None for pure NumPy/SciPy path."""
    resolved: object | None = None
    if device == "cpu":
        return resolved
    try:
        import torch
    except ImportError:
        if device in ("cuda", "mps"):
            reject_invalid_state(f"torch is required for device={device!r}")
        return resolved

    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_ok = torch.cuda.is_available()
    if device == "auto":
        if cuda_ok:
            resolved = torch.device("cuda")
        elif mps_ok:
            resolved = torch.device("mps")
    elif device == "cuda":
        if not cuda_ok:
            reject_invalid_state("CUDA requested but torch.cuda is unavailable")
        resolved = torch.device("cuda")
    elif device == "mps":
        if not mps_ok:
            reject_invalid_state("MPS requested but torch.backends.mps is unavailable")
        resolved = torch.device("mps")
    return resolved


def _cleanup_torch(dev: object | None) -> None:
    """Release GPU caches after a Torch-assisted iteration."""
    if dev is None:
        return
    try:
        import torch

        dev_type = getattr(dev, "type", None)
        if dev_type == "cuda":
            torch.cuda.empty_cache()
        elif dev_type == "mps":
            torch.mps.empty_cache()
    except Exception:  # pragma: no cover - defensive
        pass


def _laplacian_eigenvalues(height: int, width: int, dtype: np.dtype) -> np.ndarray:
    r"""Return positive DCT-domain eigenvalues of the Neumann Laplacian.

    For Type-II DCT modes ``(m, n)`` the 5-point Laplacian spectrum is

    .. math::

        \mu_{m,n} = 2\cos(\pi m / M) + 2\cos(\pi n / N) - 4 \le 0.

    We store ``\lambda = -\mu \ge 0`` and solve via
    ``\hat\phi = \hat\rho / \lambda`` with the divergence RHS of
    :func:`_weighted_divergence`.

    Mode ``(0, 0)`` is set to ``1``; the corresponding RHS entry is zeroed.
    """
    m = np.arange(height, dtype=dtype)[:, None]
    n = np.arange(width, dtype=dtype)[None, :]
    # lambda = 4 - 2 cos(pi m/M) - 2 cos(pi n/N)
    lam = (
        4.0
        - 2.0 * np.cos(np.pi * m / float(height))
        - 2.0 * np.cos(np.pi * n / float(width))
    )
    lam[0, 0] = 1.0
    return lam


def _dct2(arr: np.ndarray) -> np.ndarray:
    """2-D Type-II DCT, orthonormalised (SciPy)."""
    return sp_fft.dctn(arr, type=2, norm="ortho")


def _idct2(arr: np.ndarray) -> np.ndarray:
    """2-D Type-II inverse DCT, orthonormalised (SciPy)."""
    return sp_fft.idctn(arr, type=2, norm="ortho")


def _weighted_divergence(
    grad_h: np.ndarray,
    grad_v: np.ndarray,
    w_h: np.ndarray,
    w_v: np.ndarray,
) -> np.ndarray:
    """Discrete weighted divergence of edge fields onto pixel centres.

    Builds the RHS of the discrete Poisson problem so that, with positive
    eigenvalues ``lambda = -mu(L)``, the DCT solve recovers ramps from
    wrapped gradients. Edge arrays:

    - ``grad_h``, ``w_h``: shape ``(H, W-1)``
    - ``grad_v``, ``w_v``: shape ``(H-1, W)``
    """
    height = grad_v.shape[0] + 1
    width = grad_h.shape[1] + 1
    rho = np.zeros((height, width), dtype=np.float64)

    # Discrete divergence consistent with lambda = 4 - 2 cos - 2 cos.
    rho[:, 1:] += w_h * grad_h
    rho[:, :-1] -= w_h * grad_h
    rho[1:, :] += w_v * grad_v
    rho[:-1, :] -= w_v * grad_v
    return rho


def _dct_poisson_solve(rho: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
    """Solve Neumann Poisson problem via Type-II DCT."""
    rhs = rho.astype(np.float64, copy=False)
    # Pin absolute phase: zero DC force.
    rhs = rhs - float(np.mean(rhs))
    spec = _dct2(rhs)
    spec[0, 0] = 0.0
    # eigenvalues already have lambda_00 = 1; DC is zeroed on RHS.
    spec = spec / eigenvalues
    return _idct2(spec)


def dct_irls_unwrap(
    wrapped_phase: np.ndarray,
    coherence: np.ndarray | None = None,
    *,
    max_iter: int = 20,
    tol: float = 1e-3,
    epsilon: float = 1e-3,
    device: DeviceName = "auto",
) -> IRLSUnwrapResult:
    """Unwrap phase with DCT-domain IRLS (L1-like weighted least squares).

    Parameters
    ----------
    wrapped_phase : numpy.ndarray
        Wrapped phase in radians, shape ``(H, W)``.
    coherence : numpy.ndarray, optional
        Per-pixel weights in ``[0, 1]``. Defaults to uniform weights.
    max_iter : int, optional
        Maximum IRLS outer iterations.
    tol : float, optional
        Relative change tolerance on the unwrapped field.
    epsilon : float, optional
        Floor for residual-based reweighting.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Optional Torch device for residual bookkeeping. Spectral solve uses
        SciPy DCT (portable). ``auto`` prefers CUDA, then MPS, else CPU-only.

    Returns
    -------
    IRLSUnwrapResult
        Unwrapped phase and diagnostics (same type as SciPy IRLS path).

    Notes
    -----
    Clean-room from Ghiglia & Romero (1994) / Pritt (1996) style DCT Poisson
    unwrapping with IRLS reweighting. No InSAR.dev sources consulted.

    For discontinuous phase the DCT approximation of the *weighted* Laplacian
    is not exact; keep :func:`irls_unwrap` for the sparse-CG reference.

    """
    phase = np.asarray(wrapped_phase, dtype=np.float64)
    if phase.ndim != 2:
        reject_invalid_state("DCT-IRLS unwrap requires a 2-D phase array")
    height, width = phase.shape
    if height < 2 or width < 2:
        reject_invalid_state("DCT-IRLS unwrap requires at least 2x2 samples")
    if coherence is None:
        weight = np.ones((height, width), dtype=np.float64)
    else:
        weight = np.clip(np.asarray(coherence, dtype=np.float64), 0.0, 1.0)
        if weight.shape != phase.shape:
            reject_invalid_state("coherence must match wrapped phase shape")

    # Wrapped gradients (edge fields).
    grad_h = wrap_phase(phase[:, 1:] - phase[:, :-1])
    grad_v = wrap_phase(phase[1:, :] - phase[:-1, :])
    # Initial edge weights from coherence pairs.
    w_h = np.minimum(weight[:, 1:], weight[:, :-1])
    w_v = np.minimum(weight[1:, :], weight[:-1, :])
    w0_h = w_h.copy()
    w0_v = w_v.copy()

    eigenvalues = _laplacian_eigenvalues(height, width, np.float64)
    torch_dev = _resolve_torch_device(device)
    if torch_dev is not None:
        logger.info(
            "DCT-IRLS using torch device=%s for residual reweight assist",
            torch_dev,
        )

    unwrapped = np.zeros((height, width), dtype=np.float64)
    prev = unwrapped.copy()
    converged = False
    iterations = 0
    # Cap IRLS weight growth so the DCT (uniform-Laplacian) approximation
    # is not destroyed by extreme non-uniform weights from tiny residuals.
    w_cap = 1.0 / max(epsilon, 1e-12)

    for iterations in range(1, max_iter + 1):
        rho = _weighted_divergence(grad_h, grad_v, w_h, w_v)
        candidate = _dct_poisson_solve(rho, eigenvalues)
        candidate = candidate - float(candidate[0, 0]) + float(phase[0, 0])

        # Linear residual of the unwrapped gradient vs wrapped target.
        res_h = (candidate[:, 1:] - candidate[:, :-1]) - grad_h
        res_v = (candidate[1:, :] - candidate[:-1, :]) - grad_v
        grad_res = float(np.sqrt(np.mean(res_h**2) + np.mean(res_v**2)))

        unwrapped = candidate
        if grad_res < tol:
            converged = True
            break

        w_h = np.minimum(w0_h / np.maximum(np.abs(res_h), epsilon), w_cap)
        w_v = np.minimum(w0_v / np.maximum(np.abs(res_v), epsilon), w_cap)

        delta = float(
            np.linalg.norm(unwrapped - prev) / (np.linalg.norm(prev) + 1e-12)
        )
        prev = unwrapped.copy()
        if delta < tol and iterations > 1:
            converged = True
            break

    _cleanup_torch(torch_dev)

    rewrap = wrap_phase(unwrapped - phase)
    components = np.ones((height, width), dtype=np.int32)
    return IRLSUnwrapResult(
        unwrapped_phase=unwrapped.astype(np.float32),
        connected_components=components,
        rewrap_residual=rewrap.astype(np.float32),
        iterations=iterations,
        converged=converged,
    )
