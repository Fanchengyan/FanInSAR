"""DCT-preconditioned IRLS phase unwrapping on CPU, CUDA, or MPS."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

logger = setup_logger(__name__)

DeviceName = Literal["auto", "cpu", "cuda", "mps"]


@dataclass(frozen=True, slots=True)
class IRLSUnwrapResult:
    """Unwrapped phase and quality diagnostics from the IRLS solver."""

    unwrapped_phase: np.ndarray
    connected_components: np.ndarray
    rewrap_residual: np.ndarray
    iterations: int
    converged: bool


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Wrap phase into ``[-π, π]``."""
    values = np.asarray(phase)
    return np.arctan2(np.sin(values), np.cos(values))


def _resolve_device(device: DeviceName) -> str:
    """Resolve an explicit Torch device name."""
    try:
        import torch
    except ImportError:
        message = "IRLS requires PyTorch; install FanInSAR with core dependencies"
        logger.exception(message)
        raise ImportError(message) from None

    match device:
        case "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        case "cuda":
            if not torch.cuda.is_available():
                reject_invalid_state("CUDA requested but torch.cuda is unavailable")
            return "cuda"
        case "mps":
            if not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                reject_invalid_state(
                    "MPS requested but torch.backends.mps is unavailable"
                )
            return "mps"
        case "cpu":
            return "cpu"


def _size_ordered_components(
    valid: np.ndarray,
    minimum_size: int,
) -> np.ndarray:
    """Label valid components by descending pixel count."""
    labels, count = ndimage.label(valid)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    ordered = sorted(
        (label for label in range(1, count + 1) if sizes[label] >= minimum_size),
        key=lambda label: int(sizes[label]),
        reverse=True,
    )
    output = np.zeros(labels.shape, dtype=np.int32)
    for output_label, input_label in enumerate(ordered, start=1):
        output[labels == input_label] = output_label
    return output


def _align_component_offsets(
    unwrapped: np.ndarray,
    wrapped: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """Align each disconnected solution to its wrapped circular mean."""
    aligned = unwrapped.copy()
    for label in range(1, int(components.max(initial=0)) + 1):
        mask = components == label
        offset = np.angle(np.mean(np.exp(1j * (wrapped[mask] - aligned[mask]))))
        aligned[mask] += float(offset)
    aligned[components == 0] = np.nan
    return aligned


def _torch_irls(
    phase: np.ndarray,
    weight: np.ndarray,
    valid: np.ndarray,
    *,
    device: str,
    max_iter: int,
    tol: float,
    cg_max_iter: int,
    cg_tol: float,
    epsilon: float,
) -> tuple[np.ndarray, int, bool]:
    """Solve weighted L1 phase unwrapping with DCT-preconditioned CG."""
    try:
        import torch

        torch_dct = importlib.import_module("torch_dct")
    except ImportError:
        message = "IRLS requires torch-dct; install FanInSAR with core dependencies"
        logger.exception(message)
        raise ImportError(message) from None

    torch_device = torch.device(device)
    dct = torch_dct.dct
    idct = torch_dct.idct
    phi = torch.as_tensor(
        np.where(valid, phase, 0.0),
        dtype=torch.float32,
        device=torch_device,
    )
    quality = torch.as_tensor(
        np.where(valid, weight, 0.0),
        dtype=torch.float32,
        device=torch_device,
    )
    valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=torch_device)
    height, width = phase.shape

    target_x = torch.zeros_like(phi)
    target_y = torch.zeros_like(phi)
    target_x[:, :-1] = torch.atan2(
        torch.sin(phi[:, 1:] - phi[:, :-1]),
        torch.cos(phi[:, 1:] - phi[:, :-1]),
    )
    target_y[:-1, :] = torch.atan2(
        torch.sin(phi[1:, :] - phi[:-1, :]),
        torch.cos(phi[1:, :] - phi[:-1, :]),
    )

    edge_x = torch.zeros_like(phi)
    edge_y = torch.zeros_like(phi)
    edge_x[:, :-1] = 0.5 * (quality[:, :-1] + quality[:, 1:])
    edge_y[:-1, :] = 0.5 * (quality[:-1, :] + quality[1:, :])
    edge_x[:, :-1] *= valid_tensor[:, :-1] & valid_tensor[:, 1:]
    edge_y[:-1, :] *= valid_tensor[:-1, :] & valid_tensor[1:, :]

    row = torch.arange(height, dtype=torch.float64)
    col = torch.arange(width, dtype=torch.float64)
    eigenvalues = (
        4.0
        - 2.0 * torch.cos(torch.pi * row[:, None] / height)
        - 2.0 * torch.cos(torch.pi * col[None, :] / width)
    ).to(dtype=torch.float32, device=torch_device)
    eigenvalues[0, 0] = 1.0

    def dct2(values: torch.Tensor) -> torch.Tensor:
        return dct(dct(values, norm="ortho").T, norm="ortho").T

    def idct2(values: torch.Tensor) -> torch.Tensor:
        return idct(idct(values, norm="ortho").T, norm="ortho").T

    def divergence(
        gradient_x: torch.Tensor,
        gradient_y: torch.Tensor,
        weights_x: torch.Tensor,
        weights_y: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(gradient_x)
        flux_x = weights_x[:, :-1] * gradient_x[:, :-1]
        flux_y = weights_y[:-1, :] * gradient_y[:-1, :]
        output[:, :-1] -= flux_x
        output[:, 1:] += flux_x
        output[:-1, :] -= flux_y
        output[1:, :] += flux_y
        return output

    def apply_operator(
        values: torch.Tensor,
        weights_x: torch.Tensor,
        weights_y: torch.Tensor,
    ) -> torch.Tensor:
        gradient_x = torch.zeros_like(values)
        gradient_y = torch.zeros_like(values)
        gradient_x[:, :-1] = values[:, 1:] - values[:, :-1]
        gradient_y[:-1, :] = values[1:, :] - values[:-1, :]
        return divergence(gradient_x, gradient_y, weights_x, weights_y)

    def precondition(residual: torch.Tensor) -> torch.Tensor:
        spectrum = dct2(residual)
        spectrum[0, 0] = 0.0
        return idct2(spectrum / eigenvalues)

    def conjugate_gradient(
        rhs: torch.Tensor,
        weights_x: torch.Tensor,
        weights_y: torch.Tensor,
        initial: torch.Tensor,
    ) -> torch.Tensor:
        solution = initial.clone()
        residual = rhs - apply_operator(solution, weights_x, weights_y)
        preconditioned = precondition(residual)
        direction = preconditioned.clone()
        residual_dot = torch.sum(residual * preconditioned)
        rhs_norm = max(float(torch.linalg.vector_norm(rhs)), 1.0)
        for _ in range(cg_max_iter):
            operator_direction = apply_operator(direction, weights_x, weights_y)
            denominator = torch.sum(direction * operator_direction)
            if abs(float(denominator)) < 1e-12:
                break
            step = residual_dot / denominator
            solution += step * direction
            residual -= step * operator_direction
            if float(torch.linalg.vector_norm(residual)) <= cg_tol * rhs_norm:
                break
            next_preconditioned = precondition(residual)
            next_dot = torch.sum(residual * next_preconditioned)
            direction = next_preconditioned + (next_dot / residual_dot) * direction
            preconditioned = next_preconditioned
            residual_dot = next_dot
        return solution

    initial_rhs = divergence(target_x, target_y, edge_x, edge_y)
    solution = precondition(initial_rhs)
    valid_float = valid_tensor.to(torch.float32)
    valid_count = torch.sum(valid_float).clamp_min(1.0)
    solution -= torch.sum(solution * valid_float) / valid_count
    converged = False
    completed_iterations = 0

    for iteration in range(1, max_iter + 1):
        completed_iterations = iteration
        previous = solution
        residual_x = torch.zeros_like(solution)
        residual_y = torch.zeros_like(solution)
        residual_x[:, :-1] = solution[:, 1:] - solution[:, :-1] - target_x[:, :-1]
        residual_y[:-1, :] = solution[1:, :] - solution[:-1, :] - target_y[:-1, :]
        weights_x = edge_x / torch.sqrt(residual_x.square() + epsilon**2)
        weights_y = edge_y / torch.sqrt(residual_y.square() + epsilon**2)
        weights_x.clamp_(1e-6, 1e6)
        weights_y.clamp_(1e-6, 1e6)
        rhs = divergence(target_x, target_y, weights_x, weights_y)
        solution = conjugate_gradient(rhs, weights_x, weights_y, solution)
        solution -= torch.sum(solution * valid_float) / valid_count
        relative_change = torch.linalg.vector_norm(solution - previous) / (
            torch.linalg.vector_norm(solution) + 1e-10
        )
        if float(relative_change) < tol:
            converged = True
            break

    result = solution.detach().cpu().numpy().astype(np.float32)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    return result, completed_iterations, converged


def irls_unwrap(
    wrapped_phase: np.ndarray,
    coherence: np.ndarray | None = None,
    *,
    device: DeviceName = "auto",
    max_iter: int = 50,
    tol: float = 1e-2,
    cg_max_iter: int = 10,
    cg_tol: float = 1e-3,
    epsilon: float = 1e-2,
    conncomp_size: int = 30,
) -> IRLSUnwrapResult:
    """Unwrap a 2-D phase field with DCT-preconditioned weighted IRLS.

    Parameters
    ----------
    wrapped_phase : numpy.ndarray
        Wrapped phase in radians. Non-finite pixels are excluded.
    coherence : numpy.ndarray, optional
        Quality weights in ``[0, 1]``.
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Torch execution device.
    max_iter, tol : int, float, optional
        IRLS iteration limit and relative convergence tolerance.
    cg_max_iter, cg_tol : int, float, optional
        Inner preconditioned-CG iteration limit and tolerance.
    epsilon : float, optional
        Smooth L1 approximation scale.
    conncomp_size : int, optional
        Minimum retained valid connected-component size.

    Returns
    -------
    IRLSUnwrapResult
        Unwrapped phase, component labels, and solver diagnostics.

    Notes
    -----
    The solver follows the DCT-preconditioned weighted least-squares framework
    of Ghiglia and Romero (1994), with smooth-L1 IRLS edge reweighting.

    """
    phase = np.asarray(wrapped_phase, dtype=np.float32)
    if phase.ndim != 2 or min(phase.shape) < 2:
        reject_invalid_state("IRLS unwrap requires a 2-D array of at least 2x2")
    if coherence is None:
        quality = np.ones(phase.shape, dtype=np.float32)
    else:
        quality = np.asarray(coherence, dtype=np.float32)
        if quality.shape != phase.shape:
            reject_invalid_state("coherence must match wrapped phase shape")
        quality = np.clip(quality, 0.0, 1.0)

    valid = np.isfinite(phase) & np.isfinite(quality)
    components = _size_ordered_components(valid, conncomp_size)
    valid = components > 0
    if not np.any(valid):
        logger.error(
            "IRLS input has no valid component with at least %s pixels",
            conncomp_size,
        )
        reject_invalid_state("IRLS input has no retained valid connected component")

    resolved_device = _resolve_device(device)
    logger.info(
        "IRLS device=%s shape=%s valid=%s components=%s",
        resolved_device,
        phase.shape,
        int(np.count_nonzero(valid)),
        int(components.max()),
    )
    solution, iterations, converged = _torch_irls(
        phase,
        quality,
        valid,
        device=resolved_device,
        max_iter=max_iter,
        tol=tol,
        cg_max_iter=cg_max_iter,
        cg_tol=cg_tol,
        epsilon=epsilon,
    )
    solution = _align_component_offsets(solution, phase, components)
    residual = wrap_phase(solution - phase)
    residual[~valid] = np.nan
    return IRLSUnwrapResult(
        unwrapped_phase=solution.astype(np.float32),
        connected_components=components,
        rewrap_residual=residual.astype(np.float32),
        iterations=iterations,
        converged=converged,
    )
