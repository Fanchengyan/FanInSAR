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
            if self.input_shape is not None:
                actual_shape = tuple(secondary_centered.shape)
                expected_shape = self.input_shape
                spatial_match = actual_shape[1:] == expected_shape[1:]
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
            result = self.executor(secondary_centered)
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
        allow_partial_batch=False,
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


__all__ = [
    "AmpcorBackend",
    "AmpcorBackendRegistry",
    "AmpcorCandidateError",
    "AmpcorEnergyCandidate",
    "canonical_torch_device",
    "eager_ampcor_candidate",
    "native_workspace_bytes",
    "prepare_ampcor_compile",
    "prepare_ampcor_native",
    "torch_integral_energy",
]
