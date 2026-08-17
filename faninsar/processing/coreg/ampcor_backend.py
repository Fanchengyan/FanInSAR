# ruff: noqa: EM101, EM102, TRY003, TRY301, TRY300

"""Prepared Torch, compiled, and native Ampcor energy candidates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Literal

from faninsar.logging import setup_logger
from faninsar.processing.geometry.native_v2 import (
    NativeBackend,
    NativeBuilder,
    NativeBuildRequest,
    NativeOperation,
    PreparationStatus,
)

logger = setup_logger(__name__)

AmpcorBackend = Literal["eager", "compile", "native"]
EnergyExecutor = Callable[[object], object]
_NATIVE_ABI = "faninsar.ampcor_prefix_energy.v1"
_NCC_NATIVE_ABI = "faninsar.ampcor_ncc_postprocess.v1"
_NCC_PERFORMANCE_SHAPES = frozenset({(17, 17), (33, 33)})


def canonical_torch_device(device: str) -> str:
    """Return a concrete device identity, including the CUDA ordinal."""
    import torch

    resolved = torch.device(device)
    if resolved.type == "cuda":
        index = resolved.index
        if index is None:
            index = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
        resolved = torch.device("cuda", index)
    return str(resolved)


def _centered_sample(input_shape: tuple[int, int, int], device: str) -> object:
    """Create a deterministic centered float64 qualification sample."""
    import torch

    sample = torch.arange(
        int(input_shape[0] * input_shape[1] * input_shape[2]),
        dtype=torch.float64,
        device=device,
    ).reshape(input_shape)
    return sample - sample.mean(dim=(-2, -1), keepdim=True)


class AmpcorCandidateError(RuntimeError):
    """Raised when a prepared candidate violates its execution contract."""


class AmpcorNccCandidateError(RuntimeError):
    """Raised when the experimental native NCC candidate rejects a call."""


def ampcor_ncc_postprocess_reference(
    correlation: object,
    energy: object,
    *,
    search_az: int,
    search_rg: int,
    subpixel: bool,
    snr_threshold: float,
    max_abs_residual: float,
) -> tuple[object, object, object, object, object]:
    """Evaluate the native NCC postprocess contract with Torch float64.

    Parameters
    ----------
    correlation, energy : torch.Tensor
        Contiguous float64 CUDA or CPU tensors with shape ``[batch, height,
        width]``. ``energy`` is the non-negative local-search energy surface.
    search_az, search_rg : int
        Half-widths used to convert the peak index into an offset.
    subpixel : bool
        Apply the same three-point parabolic refinement as the Torch path.
    snr_threshold, max_abs_residual : float
        Inclusive validity thresholds.

    Returns
    -------
    tuple[torch.Tensor, ...]
        ``(d_rg, d_az, snr, surface_edge, valid)`` vectors. ``surface_edge``
        marks a peak on the outer correlation-surface edge; it is distinct
        from the public threshold-near boundary oracle mask. ``valid`` applies
        the inclusive cull.

    Raises
    ------
    ValueError
        If the tensors or scalar parameters violate the ABI.

    """
    import torch

    if not torch.is_tensor(correlation) or not torch.is_tensor(energy):
        raise ValueError("NCC postprocess requires Torch tensors")
    if correlation.dtype is not torch.float64 or energy.dtype is not torch.float64:
        raise ValueError("NCC postprocess requires float64 tensors")
    if (
        correlation.dim() != 3
        or energy.dim() != 3
        or not correlation.is_contiguous()
        or not energy.is_contiguous()
        or tuple(correlation.shape) != tuple(energy.shape)
    ):
        raise ValueError("NCC postprocess requires matching contiguous rank-3 tensors")
    if correlation.device != energy.device:
        raise ValueError("NCC postprocess tensors must share a device")
    batch, height, width = (int(value) for value in correlation.shape)
    if batch < 1 or height < 1 or width < 1:
        raise ValueError("NCC postprocess dimensions must be positive")
    if search_az < 0 or search_rg < 0:
        raise ValueError("NCC search half-widths must be non-negative")
    if 2 * search_az + 1 != height or 2 * search_rg + 1 != width:
        raise ValueError("NCC surface shape does not match search half-widths")

    ncc = correlation / torch.sqrt(torch.clamp(energy, min=1e-12))
    peak_flat = torch.argmax(ncc.reshape(batch, -1), dim=1)
    peak_az = torch.div(peak_flat, width, rounding_mode="floor")
    peak_rg = peak_flat.remainder(width)
    peak_value = ncc.reshape(batch, -1).gather(1, peak_flat[:, None]).squeeze(1)
    rows = torch.arange(height, device=ncc.device)[None, :, None]
    columns = torch.arange(width, device=ncc.device)[None, None, :]
    near_peak = (
        (rows >= peak_az[:, None, None] - 1)
        & (rows <= peak_az[:, None, None] + 1)
        & (columns >= peak_rg[:, None, None] - 1)
        & (columns <= peak_rg[:, None, None] + 1)
    )
    sidelobe = torch.where(near_peak, torch.full_like(ncc, torch.nan), ncc.abs())
    sidelobe_mean = torch.nanmean(sidelobe, dim=(-2, -1))
    snr = torch.where(
        torch.isfinite(sidelobe_mean) & (sidelobe_mean > 0),
        peak_value / sidelobe_mean,
        torch.full_like(peak_value, torch.nan),
    )
    az_shift = peak_az.to(torch.float64) - search_az
    rg_shift = peak_rg.to(torch.float64) - search_rg
    if subpixel:
        interior = (
            (peak_az > 0)
            & (peak_az < height - 1)
            & (peak_rg > 0)
            & (peak_rg < width - 1)
        )
        safe_az = peak_az.clamp(1, height - 2)
        safe_rg = peak_rg.clamp(1, width - 2)
        indices = torch.arange(batch, device=ncc.device)
        az_left = ncc[indices, safe_az - 1, safe_rg]
        az_center = ncc[indices, safe_az, safe_rg]
        az_right = ncc[indices, safe_az + 1, safe_rg]
        rg_left = ncc[indices, safe_az, safe_rg - 1]
        rg_right = ncc[indices, safe_az, safe_rg + 1]
        az_denom = az_left - 2 * az_center + az_right
        rg_denom = rg_left - 2 * az_center + rg_right
        az_sub = torch.where(
            az_denom.abs() < 1e-12,
            torch.zeros_like(az_denom),
            0.5 * (az_left - az_right) / az_denom,
        )
        rg_sub = torch.where(
            rg_denom.abs() < 1e-12,
            torch.zeros_like(rg_denom),
            0.5 * (rg_left - rg_right) / rg_denom,
        )
        az_shift += torch.where(interior, az_sub, torch.zeros_like(az_sub))
        rg_shift += torch.where(interior, rg_sub, torch.zeros_like(rg_sub))
    surface_edge = (
        (peak_az == 0)
        | (peak_az == height - 1)
        | (peak_rg == 0)
        | (peak_rg == width - 1)
    )
    valid = (
        torch.isfinite(snr)
        & torch.isfinite(rg_shift)
        & torch.isfinite(az_shift)
        & (snr >= snr_threshold)
        & (rg_shift.abs() <= max_abs_residual)
        & (az_shift.abs() <= max_abs_residual)
    )
    return rg_shift, az_shift, snr, surface_edge, valid


@dataclass(frozen=True, slots=True)
class AmpcorNccCandidate:
    """Prepared experimental native CUDA NCC/peak candidate."""

    device: str
    search_shape: tuple[int, int]
    executor: Callable[..., tuple[object, object, object, object, object]]
    runtime_profile: str = ""
    source_digest: str = ""
    abi_version: str = _NCC_NATIVE_ABI
    prepared: bool = True
    correctness_qualified: bool = True
    performance_eligible: bool = True
    native_module: object | None = None

    def execute(
        self,
        correlation: object,
        energy: object,
        *,
        subpixel: bool,
        snr_threshold: float,
        max_abs_residual: float,
    ) -> tuple[object, object, object, object, object]:
        """Execute the prepared native candidate without compilation."""
        import torch

        if not self.prepared or not self.correctness_qualified:
            raise AmpcorNccCandidateError("NCC candidate is not prepared")
        if not torch.is_tensor(correlation) or not torch.is_tensor(energy):
            raise AmpcorNccCandidateError("NCC candidate requires Torch tensors")
        if correlation.dim() != 3 or energy.dim() != 3:
            raise AmpcorNccCandidateError("NCC candidate requires rank-3 input")
        if (
            str(correlation.device) != self.device
            or correlation.device != energy.device
        ):
            raise AmpcorNccCandidateError("NCC candidate device does not match input")
        expected = (correlation.shape[0], *self.search_shape)
        if tuple(correlation.shape) != expected or tuple(energy.shape) != expected:
            raise AmpcorNccCandidateError("NCC candidate input shape does not match")
        if correlation.dtype is not torch.float64 or energy.dtype is not torch.float64:
            raise AmpcorNccCandidateError("NCC candidate requires float64 input")
        if not correlation.is_contiguous() or not energy.is_contiguous():
            raise AmpcorNccCandidateError("NCC candidate requires contiguous input")
        try:
            result = self.executor(
                correlation,
                energy,
                bool(subpixel),
                float(snr_threshold),
                float(max_abs_residual),
            )
        except AmpcorNccCandidateError:
            raise
        except Exception as error:
            raise AmpcorNccCandidateError("NCC candidate execution failed") from error
        if not isinstance(result, tuple) or len(result) != 5:
            raise AmpcorNccCandidateError("NCC candidate returned invalid outputs")
        for index, value in enumerate(result):
            if not torch.is_tensor(value) or value.shape != (correlation.shape[0],):
                raise AmpcorNccCandidateError("NCC candidate returned invalid shape")
            expected_dtype = torch.bool if index >= 3 else torch.float64
            if value.dtype is not expected_dtype or value.device != correlation.device:
                raise AmpcorNccCandidateError(
                    "NCC candidate returned invalid dtype/device"
                )
        return result


def native_workspace_bytes(
    input_shape: tuple[int, int, int], window_shape: tuple[int, int]
) -> int:
    """Return a checked conservative native prefix/output packet in bytes."""
    batch, height, width = input_shape
    window_az, window_rg = window_shape
    if (
        len(input_shape) != 3
        or any(value < 1 for value in input_shape)
        or len(window_shape) != 2
        or any(value < 1 for value in window_shape)
        or window_az > height
        or window_rg > width
    ):
        raise ValueError("native Ampcor shapes are invalid")
    output = batch * (height - window_az + 1) * (width - window_rg + 1)
    prefix = batch * (height + 1) * (width + 1)
    input_values = batch * height * width
    return 8 * (input_values + prefix + output)


def torch_integral_energy(
    secondary_centered: object, window_az: int, window_rg: int
) -> object:
    """Compute rectangular float64 local energy with a Torch prefix image.

    Parameters
    ----------
    secondary_centered : torch.Tensor
        Centered, contiguous float64 tensor with shape ``[batch, height, width]``.
    window_az, window_rg : int
        Rectangular output window dimensions.

    Returns
    -------
    torch.Tensor
        Local energy for every valid window.

    """
    import torch

    squared = secondary_centered * secondary_centered
    row_cumulative = torch.cumsum(squared, dim=-1)
    leading_column = torch.zeros(
        (*secondary_centered.shape[:-1], 1),
        dtype=torch.float64,
        device=secondary_centered.device,
    )
    row_cumulative = torch.cat((leading_column, row_cumulative), dim=-1)
    integral = torch.cumsum(row_cumulative, dim=-2)
    leading_row = torch.zeros(
        (secondary_centered.shape[0], 1, secondary_centered.shape[-1] + 1),
        dtype=torch.float64,
        device=secondary_centered.device,
    )
    integral = torch.cat((leading_row, integral), dim=-2)
    bottom_right = integral[:, window_az:, window_rg:]
    top_right = integral[:, :-window_az, window_rg:]
    bottom_left = integral[:, window_az:, :-window_rg]
    top_left = integral[:, :-window_az, :-window_rg]
    return bottom_right - top_right - bottom_left + top_left


@dataclass(frozen=True, slots=True)
class AmpcorEnergyCandidate:
    """One prepared same-device Ampcor energy implementation."""

    backend: AmpcorBackend
    device: str
    window_shape: tuple[int, int]
    executor: EnergyExecutor
    workspace_bytes: int = 0
    input_shape: tuple[int, int, int] | None = None
    runtime_profile: str = ""
    source_digest: str = ""
    abi_version: str = "faninsar.ampcor_prefix_energy.v1"
    input_contract: str = "centered-f64"
    allow_partial_batch: bool = False
    prepared: bool = True
    correctness_qualified: bool = True
    performance_eligible: bool = True
    reason: str = ""
    native_module: object | None = None

    def __post_init__(self) -> None:
        """Validate the immutable candidate identity and resource packet."""
        if self.backend not in ("eager", "compile", "native"):
            raise ValueError(f"unsupported Ampcor backend: {self.backend!r}")
        if len(self.window_shape) != 2 or any(value < 1 for value in self.window_shape):
            raise ValueError("Ampcor window_shape must contain positive dimensions")
        if isinstance(self.workspace_bytes, bool) or self.workspace_bytes < 0:
            raise ValueError("Ampcor workspace_bytes must be non-negative")
        object.__setattr__(self, "device", canonical_torch_device(self.device))

    def execute(self, secondary_centered: object) -> object:
        """Execute this already-prepared candidate without compilation."""
        if not self.prepared or not self.correctness_qualified:
            raise AmpcorCandidateError(
                self.reason or "Ampcor candidate is not prepared"
            )
        try:
            import torch

            if not torch.is_tensor(secondary_centered):
                raise AmpcorCandidateError("Ampcor candidate requires a Torch tensor")
            if str(secondary_centered.device) != self.device:
                raise AmpcorCandidateError(
                    "Ampcor candidate device does not match input"
                )
            if secondary_centered.dtype is not torch.float64:
                raise AmpcorCandidateError("Ampcor candidate requires float64 input")
            if secondary_centered.dim() != 3 or not secondary_centered.is_contiguous():
                raise AmpcorCandidateError(
                    "Ampcor candidate requires contiguous rank-3 input"
                )
            partial_batch = False
            if self.input_shape is not None:
                actual_shape = tuple(secondary_centered.shape)
                expected_shape = self.input_shape
                spatial_match = actual_shape[1:] == expected_shape[1:]
                partial_batch = (
                    actual_shape[0] < expected_shape[0] and self.allow_partial_batch
                )
                batch_match = (
                    actual_shape[0] <= expected_shape[0]
                    if self.allow_partial_batch
                    else actual_shape[0] == expected_shape[0]
                )
                if not spatial_match or not batch_match:
                    raise AmpcorCandidateError(
                        "Ampcor candidate input shape does not match"
                    )
            centered_mean = secondary_centered.mean(dim=(-2, -1)).abs().max()
            input_valid = torch.isfinite(secondary_centered).all() & (
                centered_mean <= 1e-10
            )
            if not bool(input_valid.item()):
                raise AmpcorCandidateError("Ampcor native ABI requires centered input")
            dispatch_input = secondary_centered
            if partial_batch and self.backend == "compile":
                padding = torch.zeros(
                    (
                        self.input_shape[0] - secondary_centered.shape[0],
                        *secondary_centered.shape[1:],
                    ),
                    dtype=secondary_centered.dtype,
                    device=secondary_centered.device,
                )
                dispatch_input = torch.cat((secondary_centered, padding), dim=0)
            result = self.executor(dispatch_input)
            if partial_batch:
                result = result[: secondary_centered.shape[0]]
            if not torch.is_tensor(result):
                raise AmpcorCandidateError(
                    "Ampcor candidate returned a non-Tensor result"
                )
            expected_shape = (
                secondary_centered.shape[0],
                secondary_centered.shape[1] - self.window_shape[0] + 1,
                secondary_centered.shape[2] - self.window_shape[1] + 1,
            )
            if tuple(result.shape) != expected_shape:
                raise AmpcorCandidateError("Ampcor candidate returned an invalid shape")
            if result.dtype is not torch.float64 or str(result.device) != self.device:
                raise AmpcorCandidateError(
                    "Ampcor candidate returned invalid dtype/device"
                )
            output_valid = torch.isfinite(result).all() & (result >= 0).all()
            if not bool(output_valid.item()):
                raise AmpcorCandidateError("Ampcor candidate returned invalid energy")
            return result
        except AmpcorCandidateError:
            raise
        except Exception as error:
            raise AmpcorCandidateError("Ampcor candidate execution failed") from error


class AmpcorBackendRegistry:
    """Exact-key registry for prepared Ampcor energy candidates."""

    def __init__(self) -> None:
        """Create an empty process-local registry."""
        self._candidates: dict[tuple[object, ...], AmpcorEnergyCandidate] = {}
        self._ncc_candidates: dict[tuple[object, ...], AmpcorNccCandidate] = {}

    @staticmethod
    def _key(candidate: AmpcorEnergyCandidate) -> tuple[object, ...]:
        """Return the complete candidate identity used for lookup."""
        return (
            candidate.device,
            candidate.backend,
            candidate.window_shape,
            candidate.input_shape,
            candidate.runtime_profile,
            candidate.source_digest,
            candidate.abi_version,
            candidate.input_contract,
            candidate.allow_partial_batch,
        )

    def register(self, candidate: AmpcorEnergyCandidate) -> None:
        """Atomically publish a prepared candidate in its exact slot."""
        if not candidate.prepared:
            raise ValueError("Ampcor registry accepts prepared candidates only")
        self._candidates[self._key(candidate)] = candidate

    def register_ncc(self, candidate: AmpcorNccCandidate) -> None:
        """Atomically publish an experimental prepared NCC candidate."""
        if not candidate.prepared:
            raise ValueError("Ampcor registry accepts prepared candidates only")
        self._ncc_candidates[
            (
                candidate.device,
                candidate.search_shape,
                candidate.runtime_profile,
                candidate.source_digest,
                candidate.abi_version,
            )
        ] = candidate

    def get_ncc(
        self,
        device: str,
        search_shape: tuple[int, int],
        *,
        runtime_profile: str = "",
        source_digest: str = "",
        abi_version: str = _NCC_NATIVE_ABI,
    ) -> AmpcorNccCandidate | None:
        """Return an exact experimental NCC candidate, if qualified."""
        candidate = self._ncc_candidates.get(
            (
                canonical_torch_device(device),
                search_shape,
                runtime_profile,
                source_digest,
                abi_version,
            )
        )
        return (
            candidate
            if (
                candidate is not None
                and candidate.correctness_qualified
                and candidate.performance_eligible
            )
            else None
        )

    def get(
        self,
        device: str,
        backend: AmpcorBackend,
        window_shape: tuple[int, int],
        input_shape: tuple[int, int, int] | None = None,
        runtime_profile: str = "",
        source_digest: str = "",
        abi_version: str = "faninsar.ampcor_prefix_energy.v1",
        input_contract: str = "centered-f64",
        allow_partial_batch: bool = False,
    ) -> AmpcorEnergyCandidate | None:
        """Return an eligible exact candidate, or ``None`` for eager fallback."""
        device = canonical_torch_device(device)
        candidate = self._candidates.get(
            (
                device,
                backend,
                window_shape,
                input_shape,
                runtime_profile,
                source_digest,
                abi_version,
                input_contract,
                allow_partial_batch,
            )
        )
        if candidate is None or not candidate.correctness_qualified:
            return None
        return candidate


def eager_ampcor_candidate(
    *, device: str, window_shape: tuple[int, int], workspace_bytes: int = 0
) -> AmpcorEnergyCandidate:
    """Create the portable same-device Torch eager energy candidate."""
    window_az, window_rg = window_shape
    return AmpcorEnergyCandidate(
        backend="eager",
        device=canonical_torch_device(device),
        window_shape=window_shape,
        executor=lambda secondary: torch_integral_energy(
            secondary, window_az, window_rg
        ),
        workspace_bytes=workspace_bytes,
        runtime_profile="torch-eager",
        abi_version=_NATIVE_ABI,
        input_contract="centered-f64",
        allow_partial_batch=True,
    )


def prepare_ampcor_compile(
    *,
    device: str,
    window_shape: tuple[int, int],
    input_shape: tuple[int, int, int],
) -> AmpcorEnergyCandidate:
    """Compile only the pure Tensor integral-energy operation.

    ``input_shape`` is mandatory so compilation and its first call happen in
    preparation, never on the public dispatch path.
    """
    import torch

    if len(input_shape) != 3 or any(value < 1 for value in input_shape):
        raise ValueError("input_shape must contain three positive dimensions")
    resolved = torch.device(canonical_torch_device(device))
    window_az, window_rg = window_shape
    if window_az > input_shape[1] or window_rg > input_shape[2]:
        raise ValueError("Ampcor window does not fit input_shape")
    compiled = torch.compile(
        lambda secondary: torch_integral_energy(secondary, window_az, window_rg),
        dynamic=False,
    )
    sample = _centered_sample(input_shape, str(resolved))
    compiled_result = compiled(sample)
    eager_result = torch_integral_energy(sample, window_az, window_rg)
    try:
        torch.testing.assert_close(
            compiled_result, eager_result, rtol=1e-10, atol=1e-12
        )
    except Exception as error:
        raise RuntimeError("Ampcor compile candidate failed eager parity") from error
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    return AmpcorEnergyCandidate(
        backend="compile",
        device=str(resolved),
        window_shape=window_shape,
        executor=compiled,
        input_shape=input_shape,
        runtime_profile=f"torch-{torch.__version__}-{resolved}",
        source_digest=sha256(b"torch_integral_energy.v1").hexdigest(),
        abi_version=_NATIVE_ABI,
        input_contract="centered-f64",
        allow_partial_batch=True,
    )


def prepare_ampcor_native(
    *,
    device: Literal["cpu", "cuda"],
    window_shape: tuple[int, int],
    source_root: str | Path,
    build_dir: str | Path,
    input_shape: tuple[int, int, int],
    workspace_bytes: int | None = None,
    compiler: str | None = None,
    libomp_root: str | Path | None = None,
) -> AmpcorEnergyCandidate:
    """Explicitly build and load one native candidate through P25 Builder.

    The extension loader is reachable only from this preparation function;
    candidate execution has no compiler or loader access.
    """
    import torch
    from torch.utils import cpp_extension

    backend = NativeBackend(device)
    request = NativeBuildRequest(
        NativeOperation.AMPCOR_PREFIX_ENERGY,
        backend,
        source_root=Path(source_root),
        compiler=compiler,
        libomp_root=Path(libomp_root) if libomp_root is not None else None,
    )
    builder = NativeBuilder()
    plan = builder.plan(request)
    if not plan.supported:
        raise RuntimeError(
            plan.unsupported_reason or "Ampcor native backend unsupported"
        )
    module_holder: dict[str, object] = {}

    def build(plan_to_build: object) -> Path:
        """Build the extension through the explicit P25 preparation callback."""
        plan_value = plan_to_build
        module = cpp_extension.load(
            name=plan_value.extension_name,
            sources=[str(source) for source in plan_value.sources],
            extra_cflags=list(plan_value.compile_flags),
            extra_cuda_cflags=(
                list(plan_value.compile_flags) if backend is NativeBackend.CUDA else []
            ),
            extra_ldflags=list(plan_value.link_flags),
            extra_include_paths=[str(path) for path in plan_value.include_dirs],
            build_directory=str(Path(build_dir)),
            with_cuda=backend is NativeBackend.CUDA,
            verbose=False,
        )
        module_holder["module"] = module
        return Path(getattr(module, "__file__", build_dir))

    prepared = NativeBuilder().prepare(request, build=build)
    if prepared.status is not PreparationStatus.PREPARED:
        raise RuntimeError(prepared.reason or "Ampcor native preparation failed")
    module = module_holder["module"]
    source_abi = getattr(module, "native_source_abi", None)
    if source_abi is None or source_abi() != _NATIVE_ABI:
        raise RuntimeError("Ampcor native source ABI mismatch")
    symbol = (
        "ampcor_prefix_energy_cuda"
        if backend is NativeBackend.CUDA
        else "ampcor_prefix_energy_cpu"
    )
    entry = getattr(module, symbol)
    required_workspace = native_workspace_bytes(input_shape, window_shape)
    if workspace_bytes is not None and workspace_bytes < required_workspace:
        raise ValueError("native workspace_bytes is below the checked packet")
    prepared = replace(
        prepared,
        _entry_point=lambda secondary: entry(secondary, *window_shape),
    )
    sample = _centered_sample(input_shape, canonical_torch_device(device))
    native_result = prepared.dispatch(sample)
    eager_result = torch_integral_energy(sample, *window_shape)
    try:
        torch.testing.assert_close(native_result, eager_result, rtol=1e-10, atol=1e-12)
    except Exception as error:
        raise RuntimeError("Ampcor native candidate failed eager parity") from error
    return AmpcorEnergyCandidate(
        backend="native",
        device=canonical_torch_device(device),
        window_shape=window_shape,
        executor=lambda secondary: prepared.dispatch(secondary),
        workspace_bytes=required_workspace,
        input_shape=input_shape,
        runtime_profile=(
            f"native-{canonical_torch_device(device)}-torch-{torch.__version__}"
            f"-cuda-{torch.version.cuda or 'none'}-runtime-"
            f"{plan.runtime_name or 'cuda'}"
        ),
        abi_version=_NATIVE_ABI,
        input_contract="centered-f64",
        allow_partial_batch=True,
        source_digest=sha256(
            b"".join(path.read_bytes() for path in plan.sources)
        ).hexdigest(),
        native_module=module,
    )


def prepare_ampcor_ncc_native(
    *,
    device: Literal["cuda"],
    search_shape: tuple[int, int],
    source_root: str | Path,
    build_dir: str | Path,
    compiler: str | None = None,
) -> AmpcorNccCandidate:
    """Build and qualify the experimental CUDA NCC postprocess candidate.

    The candidate is intentionally separate from :mod:`offsets`; callers must
    explicitly prepare and execute it. Compilation and module loading happen
    only here, never from a public Ampcor dispatch path.
    """
    import torch
    from torch.utils import cpp_extension

    if len(search_shape) != 2 or any(value < 1 for value in search_shape):
        raise ValueError("search_shape must contain positive dimensions")
    if search_shape[0] < 3 or search_shape[1] < 3:
        raise ValueError("NCC postprocess requires a 3x3 or larger surface")
    resolved_device = canonical_torch_device(device)
    if not resolved_device.startswith("cuda:"):
        raise ValueError("NCC postprocess native candidate requires CUDA")
    Path(build_dir).mkdir(parents=True, exist_ok=True)
    request = NativeBuildRequest(
        NativeOperation.AMPCOR_NCC_POSTPROCESS,
        NativeBackend.CUDA,
        source_root=Path(source_root),
        compiler=compiler,
    )
    builder = NativeBuilder()
    plan = builder.plan(request)
    if not plan.supported:
        raise RuntimeError(plan.unsupported_reason or "NCC native backend unsupported")
    module_holder: dict[str, object] = {}

    def build(plan_to_build: object) -> Path:
        """Build the CUDA extension during explicit candidate preparation."""
        plan_value = plan_to_build
        module = cpp_extension.load(
            name=plan_value.extension_name,
            sources=[str(source) for source in plan_value.sources],
            extra_cflags=list(plan_value.compile_flags),
            extra_cuda_cflags=list(plan_value.compile_flags),
            extra_ldflags=list(plan_value.link_flags),
            extra_include_paths=[str(path) for path in plan_value.include_dirs],
            build_directory=str(Path(build_dir)),
            with_cuda=True,
            verbose=False,
        )
        module_holder["module"] = module
        return Path(getattr(module, "__file__", build_dir))

    prepared = builder.prepare(request, build=build)
    if prepared.status is not PreparationStatus.PREPARED:
        raise RuntimeError(prepared.reason or "NCC native preparation failed")
    module = module_holder["module"]
    source_abi = getattr(module, "native_source_abi", None)
    if source_abi is None or source_abi() != _NCC_NATIVE_ABI:
        raise RuntimeError("NCC native source ABI mismatch")
    entry = getattr(module, "ampcor_ncc_postprocess_cuda", None)
    if entry is None:
        raise RuntimeError("NCC native module has no postprocess entry point")

    height, width = search_shape
    sample_corr = _centered_sample((1, height, width), resolved_device)
    sample_energy = torch.ones_like(sample_corr)
    native_result = entry(
        sample_corr,
        sample_energy,
        height // 2,
        width // 2,
        True,
        0.0,
        1e9,
    )
    reference_result = ampcor_ncc_postprocess_reference(
        sample_corr,
        sample_energy,
        search_az=height // 2,
        search_rg=width // 2,
        subpixel=True,
        snr_threshold=0.0,
        max_abs_residual=1e9,
    )
    for native_value, reference_value in zip(
        native_result, reference_result, strict=True
    ):
        if native_value.dtype is torch.bool:
            torch.testing.assert_close(native_value, reference_value)
        else:
            torch.testing.assert_close(
                native_value, reference_value, rtol=1e-10, atol=1e-12
            )
    torch.cuda.synchronize(torch.device(resolved_device))
    source_digest = sha256(
        b"".join(path.read_bytes() for path in plan.sources)
    ).hexdigest()

    def execute_native(
        correlation: object,
        energy: object,
        subpixel: bool,
        snr_threshold: float,
        max_abs_residual: float,
    ) -> tuple[object, object, object, object, object]:
        """Call the fixed-shape native entry point."""
        return entry(
            correlation,
            energy,
            height // 2,
            width // 2,
            subpixel,
            snr_threshold,
            max_abs_residual,
        )

    return AmpcorNccCandidate(
        device=resolved_device,
        search_shape=search_shape,
        executor=execute_native,
        runtime_profile=(
            f"native-ncc-{resolved_device}-torch-{torch.__version__}-cuda-"
            f"{torch.version.cuda or 'none'}"
        ),
        source_digest=source_digest,
        performance_eligible=search_shape in _NCC_PERFORMANCE_SHAPES,
        native_module=module,
    )


__all__ = [
    "AmpcorBackend",
    "AmpcorBackendRegistry",
    "AmpcorCandidateError",
    "AmpcorEnergyCandidate",
    "AmpcorNccCandidate",
    "AmpcorNccCandidateError",
    "ampcor_ncc_postprocess_reference",
    "canonical_torch_device",
    "eager_ampcor_candidate",
    "native_workspace_bytes",
    "prepare_ampcor_compile",
    "prepare_ampcor_native",
    "prepare_ampcor_ncc_native",
    "torch_integral_energy",
]
