"""DCT-preconditioned IRLS phase unwrapping on CPU, CUDA, or MPS."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import ndimage

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.unwrap.common import SpatialUnwrapper, SpatialUnwrapResult
from faninsar.processing.unwrap.errors import NoValidSupportError

logger = setup_logger(__name__)


def _raise_contract(message: str) -> None:
    """Log and raise a public input contract error."""
    logger.error("spatial unwrap contract rejected: %s", message)
    raise ValueError(message)


if TYPE_CHECKING:
    import torch

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
    """Wrap phase into the half-open interval ``[-π, π)``.

    NumPy input retains the historical array helper behavior. The new spatial
    interface passes Torch tensors and receives a tensor on the same device.
    """
    if _is_torch_tensor(phase):
        import torch

        return torch.remainder(phase + torch.pi, 2.0 * torch.pi) - torch.pi
    values = np.asarray(phase)
    return np.remainder(values + np.pi, 2.0 * np.pi) - np.pi


def _is_torch_tensor(value: object) -> bool:
    """Return whether a value is a Torch tensor without importing Torch early."""
    return value.__class__.__module__.split(".", 1)[0] == "torch"


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


class SpatialIRLS(SpatialUnwrapper):
    """Torch-native spatial phase unwrapping using smooth-L1 IRLS.

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of outer IRLS iterations.
    tol : float, optional
        Relative outer stopping tolerance.
    cg_max_iter : int, optional
        Maximum PCG iterations for each component and outer iteration.
    cg_tol : float, optional
        Relative PCG tolerance.
    cg_atol : float, optional
        Absolute PCG tolerance in phase-radian units.
    epsilon : float, optional
        Positive smooth-L1 scale in radians.

    Notes
    -----
    The numerical state stays in Torch and on the input device. Components are
    formed from finite supported pixels joined by positive-quality edges; a
    zero coherence endpoint therefore cuts an edge but remains supported.
    The solver follows the DCT-preconditioned weighted least-squares framework
    of Ghiglia and Romero (1994), with smooth-L1 IRLS edge reweighting.

    """

    def __init__(
        self,
        *,
        max_iter: int = 50,
        tol: float = 1.0e-2,
        cg_max_iter: int = 10,
        cg_tol: float = 1.0e-3,
        cg_atol: float = 0.0,
        epsilon: float = 1.0e-2,
    ) -> None:
        """Validate bounded solver parameters."""
        import math

        if not isinstance(max_iter, int) or max_iter < 1:
            _raise_contract("max_iter must be an integer >= 1")
        if not isinstance(cg_max_iter, int) or cg_max_iter < 1:
            _raise_contract("cg_max_iter must be an integer >= 1")
        if not math.isfinite(tol) or tol < 0.0:
            _raise_contract("tol must be finite and non-negative")
        if not math.isfinite(cg_tol) or cg_tol < 0.0:
            _raise_contract("cg_tol must be finite and non-negative")
        if not math.isfinite(cg_atol) or cg_atol < 0.0:
            _raise_contract("cg_atol must be finite and non-negative")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            _raise_contract("epsilon must be finite and positive")
        self.max_iter = max_iter
        self.tol = float(tol)
        self.cg_max_iter = cg_max_iter
        self.cg_tol = float(cg_tol)
        self.cg_atol = float(cg_atol)
        self.epsilon = float(epsilon)

    def unwrap(
        self,
        wrapped_phase: torch.Tensor,
        *,
        coherence: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> SpatialUnwrapResult:
        """Unwrap one 2-D Torch phase tensor on its existing device.

        Parameters
        ----------
        wrapped_phase : torch.Tensor
            Wrapped phase in radians, with shape ``(azimuth, range)``.
        coherence : torch.Tensor, optional
            Finite values in ``[0, 1]``. Nonfinite values remove pixel support;
            zero values retain pixels while cutting incident edges.
        valid_mask : torch.Tensor, optional
            Boolean authoritative support mask with the same shape.

        Returns
        -------
        SpatialUnwrapResult
            Same-device phase, mask, component labels, anchors, and diagnostics.

        Raises
        ------
        NoValidSupportError
            If no pixel remains supported.
        ValueError
            If tensor shape, dtype, device, or coherence contract is invalid.

        """
        import torch

        self._validate_input(wrapped_phase, coherence, valid_mask)
        phase = wrap_phase(wrapped_phase)
        support = torch.isfinite(wrapped_phase)
        if valid_mask is not None:
            support = support & valid_mask
        if coherence is not None:
            finite_coherence = torch.isfinite(coherence)
            if torch.any(coherence[finite_coherence] < 0) or torch.any(
                coherence[finite_coherence] > 1
            ):
                _raise_contract("coherence finite values must be within [0, 1]")
            support = support & finite_coherence

        labels, anchors, horizontal, vertical = self._graph(support, coherence)
        if not torch.any(support):
            logger.error("SpatialIRLS input has no valid support")
            raise NoValidSupportError

        output = torch.full_like(phase, torch.nan)
        output[support] = phase[support]
        active_count = int(horizontal[2].sum().item() + vertical[2].sum().item())
        if active_count == 0:
            return self._result(output, support, labels, anchors, True, 0, 0, 0.0, None)

        initial_norm = self._residual_norm(output, phase, horizontal, vertical)
        target_norm = self._target_norm(phase, horizontal, vertical)
        if initial_norm <= self.tol * max(target_norm, 1.0):
            return self._result(
                output, support, labels, anchors, True, 0, 0, initial_norm, None
            )

        total_pcg = 0
        completed_outer = 0
        failure: str | None = None
        for outer in range(1, self.max_iter + 1):
            previous = output.clone()
            for component in range(int(anchors.numel())):
                output, pcg_used, reason = self._solve_component(
                    output,
                    phase,
                    labels == component,
                    anchors[component],
                    horizontal,
                    vertical,
                    coherence,
                )
                total_pcg += pcg_used
                if reason is not None:
                    failure = reason
                    completed_outer = outer - 1
                    break
            if failure is not None:
                break
            completed_outer = outer
            change = torch.linalg.vector_norm((output - previous)[support])
            baseline = torch.linalg.vector_norm(previous[support]).clamp_min(1.0)
            if not bool(torch.isfinite(change) & torch.isfinite(baseline)):
                failure = "nonfinite_state"
                break
            if float(change / baseline) <= self.tol:
                return self._result(
                    output,
                    support,
                    labels,
                    anchors,
                    True,
                    completed_outer,
                    total_pcg,
                    self._residual_norm(output, phase, horizontal, vertical),
                    None,
                )
        if failure is None:
            failure = "outer_iteration_limit"
        residual = (
            float("inf")
            if failure == "nonfinite_state"
            else self._residual_norm(output, phase, horizontal, vertical)
        )
        return self._result(
            output,
            support,
            labels,
            anchors,
            False,
            completed_outer,
            total_pcg,
            residual,
            failure,
        )

    @staticmethod
    def _validate_input(
        phase: torch.Tensor,
        coherence: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
    ) -> None:
        """Validate the public tensor boundary."""
        import torch

        if not isinstance(phase, torch.Tensor) or phase.ndim != 2:
            _raise_contract("wrapped_phase must be a 2-D torch tensor")
        if not (phase.is_floating_point() or phase.is_complex()):
            _raise_contract("wrapped_phase must have a floating dtype")
        if phase.is_complex():
            _raise_contract("wrapped_phase must contain phase values, not complex data")
        for name, value in (("coherence", coherence), ("valid_mask", valid_mask)):
            if value is not None and not isinstance(value, torch.Tensor):
                _raise_contract(f"{name} must be a torch tensor")
            if value is not None and value.shape != phase.shape:
                _raise_contract(f"{name} must match wrapped_phase shape")
            if value is not None and value.device != phase.device:
                _raise_contract(f"{name} must be on the wrapped_phase device")
        if valid_mask is not None and valid_mask.dtype is not torch.bool:
            _raise_contract("valid_mask must have dtype torch.bool")
        if coherence is not None and not coherence.is_floating_point():
            _raise_contract("coherence must have a floating dtype")

    @staticmethod
    def _graph(
        support: torch.Tensor,
        coherence: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        """Build row-major components and positive-quality edge masks."""
        import torch

        height, width = support.shape
        h_src = support[:, :-1]
        h_dst = support[:, 1:]
        v_src = support[:-1, :]
        v_dst = support[1:, :]
        if coherence is None:
            h_active = h_src & h_dst
            v_active = v_src & v_dst
        else:
            h_active = h_src & h_dst & (coherence[:, :-1] > 0) & (coherence[:, 1:] > 0)
            v_active = v_src & v_dst & (coherence[:-1, :] > 0) & (coherence[1:, :] > 0)

        parent = list(range(height * width))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for row, col in zip(*torch.where(h_active), strict=True):
            index = int(row) * width + int(col)
            union(index, index + 1)
        for row, col in zip(*torch.where(v_active), strict=True):
            index = int(row) * width + int(col)
            union(index, index + width)

        labels = torch.full_like(support, -1, dtype=torch.int64)
        root_to_label: dict[int, int] = {}
        anchors_list: list[torch.Tensor] = []
        for index in range(height * width):
            row, col = divmod(index, width)
            if not bool(support[row, col]):
                continue
            root = find(index)
            label = root_to_label.setdefault(root, len(root_to_label))
            labels[row, col] = label
            if label == len(anchors_list):
                anchors_list.append(torch.tensor(index, device=support.device))
        anchors = (
            torch.stack(anchors_list)
            if anchors_list
            else torch.empty(0, dtype=torch.int64, device=support.device)
        )
        return labels, anchors, (h_src, h_dst, h_active), (v_src, v_dst, v_active)

    def _solve_component(
        self,
        state: torch.Tensor,
        phase: torch.Tensor,
        component: torch.Tensor,
        anchor: torch.Tensor,
        horizontal: tuple[torch.Tensor, ...],
        vertical: tuple[torch.Tensor, ...],
        coherence: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int, str | None]:
        """Run one component's weighted PCG update."""
        import torch

        _, _, h_active = horizontal
        _, _, v_active = vertical
        active_h = h_active & component[:, :-1] & component[:, 1:]
        active_v = v_active & component[:-1, :] & component[1:, :]
        qh = active_h.to(state.dtype)
        qv = active_v.to(state.dtype)
        if coherence is not None:
            qh = torch.sqrt(coherence[:, :-1] * coherence[:, 1:]) * active_h
            qv = torch.sqrt(coherence[:-1, :] * coherence[1:, :]) * active_v
        target_h = torch.zeros_like(state)
        target_v = torch.zeros_like(state)
        phase_h = torch.where(
            active_h,
            wrap_phase(
                torch.nan_to_num(phase[:, 1:]) - torch.nan_to_num(phase[:, :-1])
            ),
            torch.zeros_like(phase[:, :-1]),
        )
        phase_v = torch.where(
            active_v,
            wrap_phase(
                torch.nan_to_num(phase[1:, :]) - torch.nan_to_num(phase[:-1, :])
            ),
            torch.zeros_like(phase[:-1, :]),
        )
        target_h[:, :-1] = phase_h
        target_v[:-1, :] = phase_v
        free = component.flatten().clone()
        free[anchor] = False
        free_index = torch.where(free)[0]
        if free_index.numel() == 0:
            return state, 0, None

        def divergence(weight_h: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(state)
            flux_h = weight_h[:, :-1]
            flux_v = weight_v[:-1, :]
            result[:, :-1] -= flux_h
            result[:, 1:] += flux_h
            result[:-1, :] -= flux_v
            result[1:, :] += flux_v
            return result

        def apply(
            values: torch.Tensor,
            weight_h: torch.Tensor,
            weight_v: torch.Tensor,
        ) -> torch.Tensor:
            grad_h = torch.zeros_like(values)
            grad_v = torch.zeros_like(values)
            grad_h[:, :-1] = values[:, 1:] - values[:, :-1]
            grad_v[:-1, :] = values[1:, :] - values[:-1, :]
            return divergence(weight_h * grad_h, weight_v * grad_v)

        initial_h = torch.where(
            active_h,
            state[:, 1:] - state[:, :-1] - target_h[:, :-1],
            torch.zeros_like(state[:, :-1]),
        )
        initial_v = torch.where(
            active_v,
            state[1:, :] - state[:-1, :] - target_v[:-1, :],
            torch.zeros_like(state[:-1, :]),
        )
        weights_h = torch.zeros_like(state)
        weights_v = torch.zeros_like(state)
        weights_h[:, :-1] = torch.where(
            active_h,
            qh / torch.sqrt(initial_h.square() + self.epsilon**2),
            torch.zeros_like(initial_h),
        )
        weights_v[:-1, :] = torch.where(
            active_v,
            qv / torch.sqrt(initial_v.square() + self.epsilon**2),
            torch.zeros_like(initial_v),
        )
        rhs = divergence(weights_h * target_h, weights_v * target_v)
        current = state.clone()
        current_flat = current.flatten()
        current_flat[anchor] = phase.flatten()[anchor]

        anchor_state = torch.zeros_like(state)
        anchor_state.flatten()[anchor] = phase.flatten()[anchor]
        rhs = rhs - apply(anchor_state, weights_h, weights_v)
        rhs_flat = rhs.flatten()[free_index]

        def operator_free(values: torch.Tensor) -> torch.Tensor:
            full = torch.zeros_like(state).flatten()
            full[free_index] = values
            full = full.reshape_as(state)
            return apply(full, weights_h, weights_v).flatten()[free_index]

        solution = current_flat[free_index].clone()
        residual = rhs_flat - operator_free(solution)
        if not bool(torch.isfinite(residual).all()):
            return state, 0, "nonfinite_state"
        direction = residual.clone()
        rr = torch.dot(residual, residual)
        cg_limit = max(
            self.cg_atol,
            self.cg_tol * max(float(torch.linalg.vector_norm(rhs_flat)), 1.0),
        )
        if float(torch.sqrt(rr)) <= cg_limit:
            current_flat[free_index] = solution
            return current_flat.reshape_as(state), 0, None
        used = 0
        for used in range(1, self.cg_max_iter + 1):
            adirection = operator_free(direction)
            denominator = torch.dot(direction, adirection)
            if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
                return state, used, "pcg_breakdown"
            step = rr / denominator
            solution = solution + step * direction
            residual = residual - step * adirection
            if not bool(
                torch.isfinite(solution).all() and torch.isfinite(residual).all()
            ):
                return state, used, "nonfinite_state"
            next_rr = torch.dot(residual, residual)
            if float(torch.sqrt(next_rr)) <= max(
                self.cg_atol,
                self.cg_tol * max(float(torch.linalg.vector_norm(rhs_flat)), 1.0),
            ):
                current_flat[free_index] = solution
                return current_flat.reshape_as(state), used, None
            direction = residual + (next_rr / rr) * direction
            rr = next_rr
        return state, used, "inner_iteration_limit"

    @staticmethod
    def _residual_norm(
        state: torch.Tensor,
        phase: torch.Tensor,
        horizontal: tuple[torch.Tensor, ...],
        vertical: tuple[torch.Tensor, ...],
    ) -> float:
        """Return global active-edge wrapped-gradient residual norm."""
        import torch

        h_active = horizontal[2]
        v_active = vertical[2]
        rh = state[:, 1:] - state[:, :-1] - wrap_phase(phase[:, 1:] - phase[:, :-1])
        rv = state[1:, :] - state[:-1, :] - wrap_phase(phase[1:, :] - phase[:-1, :])
        values = torch.cat((rh[h_active], rv[v_active]))
        return float(torch.linalg.vector_norm(values)) if values.numel() else 0.0

    @staticmethod
    def _target_norm(
        phase: torch.Tensor,
        horizontal: tuple[torch.Tensor, ...],
        vertical: tuple[torch.Tensor, ...],
    ) -> float:
        """Return the norm used for initial convergence scaling."""
        import torch

        h_active, v_active = horizontal[2], vertical[2]
        values = torch.cat(
            (
                wrap_phase(phase[:, 1:] - phase[:, :-1])[h_active],
                wrap_phase(phase[1:, :] - phase[:-1, :])[v_active],
            )
        )
        return float(torch.linalg.vector_norm(values)) if values.numel() else 0.0

    @staticmethod
    def _result(
        phase: torch.Tensor,
        valid_mask: torch.Tensor,
        labels: torch.Tensor,
        anchors: torch.Tensor,
        converged: bool,
        iterations: int,
        pcg_iterations: int,
        residual_norm: float,
        failure_reason: str | None,
    ) -> SpatialUnwrapResult:
        """Construct the checked public result."""
        import torch

        references = phase.flatten()[anchors]
        phase = phase.clone()
        phase[~valid_mask] = torch.nan
        return SpatialUnwrapResult(
            phase=phase,
            valid_mask=valid_mask,
            component_labels=labels,
            reference_values=references,
            converged=converged,
            iterations=iterations,
            pcg_iterations=pcg_iterations,
            residual_norm=residual_norm,
            failure_reason=failure_reason,
        )
