"""Clean-room IRLS L1 phase unwrapping (NumPy/SciPy reference)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class IRLSUnwrapResult:
    """Unwrapped phase and quality diagnostics from the IRLS solver."""

    unwrapped_phase: np.ndarray
    connected_components: np.ndarray
    rewrap_residual: np.ndarray
    iterations: int
    converged: bool


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Wrap phase into ``(-π, π]``."""
    return np.angle(np.exp(1j * np.asarray(phase, dtype=np.float64)))


def _build_difference_operators(
    height: int,
    width: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Build sparse horizontal and vertical first-difference operators."""
    n = height * width
    # horizontal differences: pixel (i,j+1) - (i,j)
    h_rows = []
    h_cols = []
    h_data = []
    idx = 0
    for row in range(height):
        for col in range(width - 1):
            p0 = row * width + col
            p1 = p0 + 1
            h_rows.extend([idx, idx])
            h_cols.extend([p0, p1])
            h_data.extend([-1.0, 1.0])
            idx += 1
    h_op = sparse.csr_matrix((h_data, (h_rows, h_cols)), shape=(idx, n))

    v_rows = []
    v_cols = []
    v_data = []
    idx = 0
    for row in range(height - 1):
        for col in range(width):
            p0 = row * width + col
            p1 = p0 + width
            v_rows.extend([idx, idx])
            v_cols.extend([p0, p1])
            v_data.extend([-1.0, 1.0])
            idx += 1
    v_op = sparse.csr_matrix((v_data, (v_rows, v_cols)), shape=(idx, n))
    return h_op, v_op


def irls_unwrap(
    wrapped_phase: np.ndarray,
    coherence: np.ndarray | None = None,
    *,
    max_iter: int = 20,
    tol: float = 1e-3,
    epsilon: float = 1e-3,
) -> IRLSUnwrapResult:
    """Unwrap phase with iterative reweighted least squares (L1-like).

    The algorithm minimises a weighted Laplacian residual between the
    unwrapped phase gradients and the wrapped phase gradients, reweighting
    by inverse gradient residual magnitude each iteration (IRLS for L1).

    Parameters
    ----------
    wrapped_phase : numpy.ndarray
        Wrapped phase in radians.
    coherence : numpy.ndarray, optional
        Weights in ``[0, 1]``. Defaults to uniform weights.
    max_iter : int, optional
        Maximum IRLS iterations.
    tol : float, optional
        Relative change tolerance for convergence.
    epsilon : float, optional
        Stability floor for IRLS weights.

    Returns
    -------
    IRLSUnwrapResult
        Unwrapped phase, trivial connected-component label, rewrap residual,
        iteration count and convergence flag.

    Notes
    -----
    Clean-room implementation from the published IRLS / weighted-least-squares
    unwrapping formulation. No source-available InSAR.dev code is consulted.

    """
    phase = np.asarray(wrapped_phase, dtype=np.float64)
    if phase.ndim != 2:
        reject_invalid_state("IRLS unwrap requires a 2-D phase array")
    height, width = phase.shape
    if height < 2 or width < 2:
        reject_invalid_state("IRLS unwrap requires at least 2x2 samples")
    if coherence is None:
        weight = np.ones_like(phase, dtype=np.float64)
    else:
        weight = np.clip(np.asarray(coherence, dtype=np.float64), 0.0, 1.0)
        if weight.shape != phase.shape:
            reject_invalid_state("coherence must match wrapped phase shape")

    h_op, v_op = _build_difference_operators(height, width)
    # wrapped gradients
    grad_h = wrap_phase(phase[:, 1:] - phase[:, :-1]).ravel()
    grad_v = wrap_phase(phase[1:, :] - phase[:-1, :]).ravel()
    # edge weights from coherence pairs
    w_h = np.minimum(weight[:, 1:], weight[:, :-1]).ravel()
    w_v = np.minimum(weight[1:, :], weight[:-1, :]).ravel()

    unwrapped = phase.copy().ravel()
    # pin first pixel
    unwrapped[0] = phase.ravel()[0]
    converged = False
    iterations = 0
    prev = unwrapped.copy()

    for iterations in range(1, max_iter + 1):
        # residual gradients
        res_h = (h_op @ unwrapped) - grad_h
        res_v = (v_op @ unwrapped) - grad_v
        irls_h = w_h / np.maximum(np.abs(res_h), epsilon)
        irls_v = w_v / np.maximum(np.abs(res_v), epsilon)

        # Build normal equations A^T W A x = A^T W b
        wh = sparse.diags(irls_h)
        wv = sparse.diags(irls_v)
        ata = h_op.T @ wh @ h_op + v_op.T @ wv @ v_op
        atb = h_op.T @ wh @ grad_h + v_op.T @ wv @ grad_v
        # pin first pixel with a strong diagonal prior
        ata = ata.tolil()
        ata[0, 0] = ata[0, 0] + 1.0e6
        ata = ata.tocsr()
        atb = atb.copy()
        atb[0] = atb[0] + 1.0e6 * phase.ravel()[0]

        solution, info = cg(ata, atb, x0=unwrapped, maxiter=200, rtol=1e-6)
        if info != 0:
            logger.warning("IRLS CG did not fully converge at iteration %s", iterations)
        unwrapped = np.asarray(solution, dtype=np.float64)
        # keep absolute offset aligned to first wrapped sample
        unwrapped = unwrapped - unwrapped[0] + phase.ravel()[0]
        delta = float(np.linalg.norm(unwrapped - prev) / (np.linalg.norm(prev) + 1e-12))
        prev = unwrapped.copy()
        if delta < tol:
            converged = True
            break

    unwrapped_2d = unwrapped.reshape(height, width)
    rewrap = wrap_phase(unwrapped_2d - phase)
    components = np.ones((height, width), dtype=np.int32)
    return IRLSUnwrapResult(
        unwrapped_phase=unwrapped_2d.astype(np.float32),
        connected_components=components,
        rewrap_residual=rewrap.astype(np.float32),
        iterations=iterations,
        converged=converged,
    )
