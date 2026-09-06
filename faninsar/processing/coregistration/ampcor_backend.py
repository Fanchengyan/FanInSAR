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
_NCC_OPERATION = NativeOperation.AMPCOR_NCC_POSTPROCESS.value


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


class AmpcorNccPreDispatchError(AmpcorNccCandidateError):
    """Raised when a native NCC input is ineligible before dispatch."""


class AmpcorNccExecutionError(AmpcorNccCandidateError):
    """Raised when native NCC entry or output validation fails."""


@dataclass(frozen=True, slots=True)
class AmpcorNccWorkspaceMeasurement:
    """Measured native NCC allocation and launch metadata.

    The measured packet is produced during explicit candidate preparation. It
    is intentionally immutable so public workspace admission cannot silently
    replace a measured native allocation with a shape-only estimate.
    """

    operation: str
    search_shape: tuple[int, int]
    batch_size: int
    runtime_profile: str
    source_digest: str
    abi_version: str
    candidate_generation: str
    peak_allocated_bytes: int
    output_bytes: int
    global_scratch_bytes: int
    shared_memory_bytes: int
    launch_blocks: int
    threads_per_block: int

    def __post_init__(self) -> None:
        """Validate all measured packet fields."""
        if (
            not self.operation
            or len(self.search_shape) != 2
            or any(value < 1 for value in self.search_shape)
        ):
            message = "Ampcor NCC workspace identity is invalid"
            logger.error(message)
            raise ValueError(message)
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
            or not self.runtime_profile
            or not self.source_digest
            or not self.abi_version
            or not self.candidate_generation
        ):
            message = "Ampcor NCC workspace identity is incomplete"
            logger.error(message)
            raise ValueError(message)
        fields = (
            self.peak_allocated_bytes,
            self.output_bytes,
            self.global_scratch_bytes,
            self.shared_memory_bytes,
            self.launch_blocks,
            self.threads_per_block,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in fields
        ):
            message = "Ampcor NCC workspace measurements must be integers"
            logger.error(message)
            raise TypeError(message)
        if any(value < 0 for value in fields) or self.launch_blocks < 1:
            message = "Ampcor NCC workspace measurements are invalid"
            logger.error(message)
            raise ValueError(message)

    @property
    def admitted_bytes(self) -> int:
        """Return the measured packet charged to workspace admission."""
        return (
            max(self.peak_allocated_bytes, self.output_bytes)
            + self.global_scratch_bytes
            + self.shared_memory_bytes
        )


@dataclass(frozen=True, slots=True)
class AmpcorNccRuntimeProfile:
    """Runtime identity attested when an NCC candidate is prepared.

    The profile deliberately contains no hostname or process identity. It
    records the runtime that prepared a candidate so a later lookup can
    refuse a stale binary without pinning public Ampcor to one lab SKU.
    """

    device_name: str
    compute_capability: tuple[int, int]
    torch_version: str
    cuda_runtime: str
    memory_bytes: int
    source_digest: str
    abi_version: str

    def canonical(self) -> str:
        """Return a stable, human-readable profile identity."""
        capability = ".".join(str(part) for part in self.compute_capability)
        return (
            f"{self.device_name}|sm_{capability}|torch-{self.torch_version}|"
            f"cuda-{self.cuda_runtime}|memory-{self.memory_bytes}|"
            f"source-{self.source_digest}|abi-{self.abi_version}"
        )


@dataclass(frozen=True, slots=True)
class _AmpcorNccQualifiedRecord:
    """Native NCC shapes and source identity for optional dispatch."""

    qualified_combinations: frozenset[tuple[tuple[int, int], int]]
    source_digest: str
    abi_version: str


_NCC_QUALIFIED_RECORDS = (
    _AmpcorNccQualifiedRecord(
        qualified_combinations=frozenset(
            {
                ((17, 17), 16),
                ((33, 33), 32),
            }
        ),
        source_digest=(
            "f88a7b883ada00b63677b60cb20965c3a875900596c5642bc4c22226f05dab62"
        ),
        abi_version=_NCC_NATIVE_ABI,
    ),
)
_NCC_PERFORMANCE_COMBINATIONS = frozenset(
    combination
    for record in _NCC_QUALIFIED_RECORDS
    for combination in record.qualified_combinations
)


def _current_ncc_runtime_profile(
    device: str, *, source_digest: str, abi_version: str
) -> AmpcorNccRuntimeProfile | None:
    """Probe the current CUDA device without relying on host naming."""
    import torch

    resolved = torch.device(canonical_torch_device(device))
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(resolved)
    return AmpcorNccRuntimeProfile(
        device_name=str(properties.name),
        compute_capability=tuple(
            int(part) for part in torch.cuda.get_device_capability(resolved)
        ),
        torch_version=str(torch.__version__),
        cuda_runtime=str(torch.version.cuda or ""),
        memory_bytes=int(properties.total_memory),
        source_digest=source_digest,
        abi_version=abi_version,
    )


def _current_ncc_source_digest() -> str:
    """Hash the current NCC source set once for one registry acquisition."""
    source_root = Path(__file__).resolve().parents[1] / "geometry" / "native_v2"
    sources = (
        source_root / "ampcor_ncc_cuda_bindings.cpp",
        source_root / "ampcor_ncc_postprocess_cuda.cu",
    )
    try:
        return sha256(b"".join(source.read_bytes() for source in sources)).hexdigest()
    except OSError:
        message = "unable to hash the current native NCC source set"
        logger.exception(message)
        return ""


def _is_qualified_ncc_profile(
    profile: AmpcorNccRuntimeProfile | None,
    *,
    search_shape: tuple[int, int],
    batch_size: int,
    source_digest: str,
    abi_version: str,
) -> bool:
    """Return whether the native NCC kernel identity matches a recorded shape."""
    if profile is None:
        return False
    return any(
        (search_shape, batch_size) in record.qualified_combinations
        and profile.source_digest == source_digest == record.source_digest
        and profile.abi_version == abi_version == record.abi_version
        for record in _NCC_QUALIFIED_RECORDS
    )


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
        message = "NCC postprocess requires Torch tensors"
        logger.error(message)
        raise ValueError(message)
    if correlation.dtype is not torch.float64 or energy.dtype is not torch.float64:
        message = "NCC postprocess requires float64 tensors"
        logger.error(message)
        raise ValueError(message)
    if (
        correlation.dim() != 3
        or energy.dim() != 3
        or not correlation.is_contiguous()
        or not energy.is_contiguous()
        or tuple(correlation.shape) != tuple(energy.shape)
    ):
        message = "NCC postprocess requires matching contiguous rank-3 tensors"
        logger.error(message)
        raise ValueError(message)
    if correlation.device != energy.device:
        message = "NCC postprocess tensors must share a device"
        logger.error(message)
        raise ValueError(message)
    batch, height, width = (int(value) for value in correlation.shape)
    if batch < 1 or height < 1 or width < 1:
        message = "NCC postprocess dimensions must be positive"
        logger.error(message)
        raise ValueError(message)
    if search_az < 0 or search_rg < 0:
        message = "NCC search half-widths must be non-negative"
        logger.error(message)
        raise ValueError(message)
    if 2 * search_az + 1 != height or 2 * search_rg + 1 != width:
        message = "NCC surface shape does not match search half-widths"
        logger.error(message)
        raise ValueError(message)

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


@dataclass(slots=True)
class AmpcorNccCandidate:
    """Prepared experimental native CUDA NCC/peak candidate."""

    device: str
    search_shape: tuple[int, int]
    executor: Callable[..., tuple[object, object, object, object, object]]
    runtime_profile: str = ""
    source_digest: str = ""
    abi_version: str = _NCC_NATIVE_ABI
    profile: AmpcorNccRuntimeProfile | None = None
    prepared: bool = True
    correctness_qualified: bool = True
    performance_eligible: bool = False
    native_module: object | None = None
    batch_size: int = 1
    workspace_measurement: AmpcorNccWorkspaceMeasurement | None = None
    quarantined: bool = False
    candidate_generation: str = ""

    def __post_init__(self) -> None:
        """Normalize the device identity before registry publication."""
        object.__setattr__(self, "device", canonical_torch_device(self.device))
        if len(self.search_shape) != 2 or any(value < 1 for value in self.search_shape):
            message = "NCC search_shape must contain positive dimensions"
            logger.error(message)
            raise ValueError(message)
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            message = "NCC batch_size must be a positive integer"
            logger.error(message)
            raise TypeError(message)
        if self.batch_size < 1:
            message = "NCC batch_size must be a positive integer"
            logger.error(message)
            raise ValueError(message)
        if self.workspace_measurement is not None:
            measurement = self.workspace_measurement
            if (
                measurement.operation != _NCC_OPERATION
                or measurement.search_shape != self.search_shape
                or measurement.batch_size != self.batch_size
                or measurement.runtime_profile != self.runtime_profile
                or measurement.source_digest != self.source_digest
                or measurement.abi_version != self.abi_version
                or measurement.candidate_generation != self.candidate_generation
            ):
                message = "NCC workspace measurement does not match candidate"
                logger.error(message)
                raise ValueError(message)

    def quarantine(self, reason: str = "native execution failure") -> None:
        """Quarantine this candidate after a native failure."""
        if not self.quarantined:
            logger.error("Quarantining native NCC candidate: %s", reason)
            self.quarantined = True

    def validate_inputs(self, correlation: object, energy: object) -> None:
        """Validate the native ABI before entering the prepared callable."""
        import torch

        if not self.prepared or not self.correctness_qualified or self.quarantined:
            message = "NCC candidate is not eligible for dispatch"
            logger.error(message)
            raise AmpcorNccPreDispatchError(message)
        if not torch.is_tensor(correlation) or not torch.is_tensor(energy):
            message = "NCC candidate requires Torch tensors"
            logger.error(message)
            raise AmpcorNccPreDispatchError(message)
        if correlation.dim() != 3 or energy.dim() != 3:
            message = "NCC candidate requires rank-3 input"
            logger.error(message)
            raise AmpcorNccPreDispatchError(message)
        if (
            str(correlation.device) != self.device
            or correlation.device != energy.device
        ):
            message = "NCC candidate device does not match input"
            logger.error(message)
            raise AmpcorNccPreDispatchError(message)
        actual_batch = int(correlation.shape[0])
        if actual_batch < 1 or actual_batch > self.batch_size:
            message = "NCC candidate batch_size exceeds prepared batch capacity"
            logger.error(message)
            raise AmpcorNccPreDispatchError(message)
        expected = (actual_batch, *self.search_shape)
        if tuple(correlation.shape) != expected or tuple(energy.shape) != expected:
            message = "NCC candidate input shape does not match"
            logger.error(message)
            raise AmpcorNccPreDispatchError(message)
        if correlation.dtype is not torch.float64 or energy.dtype is not torch.float64:
            message = "NCC candidate requires float64 input"
            logger.error(message)
            raise AmpcorNccPreDispatchError(message)
        if not correlation.is_contiguous() or not energy.is_contiguous():
            message = "NCC candidate requires contiguous input"
            logger.error(message)
            raise AmpcorNccPreDispatchError(message)

    def _validate_outputs(
        self,
        result: object,
        *,
        batch_size: int,
        snr_threshold: float,
        max_abs_residual: float,
        torch_module: object,
    ) -> tuple[object, object, object, object, object]:
        """Validate native outputs before publishing them to public code."""
        if not isinstance(result, tuple) or len(result) != 5:
            message = "NCC candidate returned invalid outputs"
            logger.error(message)
            raise AmpcorNccExecutionError(message)
        for index, value in enumerate(result):
            if not torch_module.is_tensor(value) or value.shape != (batch_size,):
                message = "NCC candidate returned invalid shape"
                logger.error(message)
                raise AmpcorNccExecutionError(message)
            expected_dtype = torch_module.bool if index >= 3 else torch_module.float64
            if value.dtype is not expected_dtype or value.device != torch_module.device(
                self.device
            ):
                message = "NCC candidate returned invalid dtype/device"
                logger.error(message)
                raise AmpcorNccExecutionError(message)
        d_rg, d_az, snr, surface_edge, valid = result
        max_shift = max(self.search_shape) // 2 + 1
        shifts_finite = torch_module.isfinite(d_rg) & torch_module.isfinite(d_az)
        shifts_in_domain = (d_rg.abs() <= max_shift) & (d_az.abs() <= max_shift)
        if not bool(torch_module.all(shifts_finite & shifts_in_domain).item()):
            message = "NCC candidate returned shifts outside its search domain"
            logger.error(message)
            raise AmpcorNccExecutionError(message)
        expected_valid = (
            torch_module.isfinite(snr)
            & shifts_finite
            & (snr >= snr_threshold)
            & (d_rg.abs() <= max_abs_residual)
            & (d_az.abs() <= max_abs_residual)
        )
        if not bool(torch_module.equal(valid, expected_valid)):
            message = "NCC candidate valid output violates cull invariants"
            logger.error(message)
            raise AmpcorNccExecutionError(message)
        return d_rg, d_az, snr, surface_edge, valid

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

        self.validate_inputs(correlation, energy)
        try:
            result = self.executor(
                correlation,
                energy,
                bool(subpixel),
                float(snr_threshold),
                float(max_abs_residual),
            )
        except Exception as error:
            message = "NCC candidate execution failed"
            logger.exception(message)
            self.quarantine()
            raise AmpcorNccExecutionError(message) from error
        try:
            return self._validate_outputs(
                result,
                batch_size=int(correlation.shape[0]),
                snr_threshold=snr_threshold,
                max_abs_residual=max_abs_residual,
                torch_module=torch,
            )
        except AmpcorNccExecutionError:
            self.quarantine()
            raise


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
        message = "native Ampcor shapes are invalid"
        logger.error(message)
        raise ValueError(message)
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
        """Atomically publish an eligible experimental NCC candidate.

        The caller-supplied ``performance_eligible`` flag is intentionally
        ignored.  Invalid or unqualified candidates are left unpublished so a
        correctness-only implementation can remain available to the caller.
        """
        if (
            not candidate.prepared
            or not candidate.correctness_qualified
            or candidate.quarantined
            or candidate.workspace_measurement is None
            or candidate.workspace_measurement.admitted_bytes <= 0
        ):
            return
        if (
            candidate.search_shape,
            candidate.batch_size,
        ) not in _NCC_PERFORMANCE_COMBINATIONS:
            return
        if not _is_qualified_ncc_profile(
            candidate.profile,
            search_shape=candidate.search_shape,
            batch_size=candidate.batch_size,
            source_digest=candidate.source_digest,
            abi_version=candidate.abi_version,
        ):
            return
        if candidate.profile is None:
            return
        self._ncc_candidates[
            (
                candidate.device,
                candidate.search_shape,
                candidate.batch_size,
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
        batch_size: int = 1,
        runtime_profile: str = "",
        source_digest: str = "",
        abi_version: str = _NCC_NATIVE_ABI,
    ) -> AmpcorNccCandidate | None:
        """Return an independently qualified NCC candidate, if available."""
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or (search_shape, batch_size) not in _NCC_PERFORMANCE_COMBINATIONS
        ):
            return None
        resolved = canonical_torch_device(device)
        candidates = tuple(
            candidate
            for key, candidate in self._ncc_candidates.items()
            if (key[0] == resolved and key[1] == search_shape and key[2] == batch_size)
        )
        current_source_digest = _current_ncc_source_digest() if candidates else ""
        if not current_source_digest:
            return None
        for candidate in candidates:
            measurement = candidate.workspace_measurement
            if candidate.quarantined or measurement is None:
                continue
            if candidate.source_digest != current_source_digest:
                candidate.quarantine("native NCC source digest is stale")
                continue
            if (
                measurement.operation != _NCC_OPERATION
                or measurement.search_shape != candidate.search_shape
                or measurement.batch_size != candidate.batch_size
                or measurement.runtime_profile != candidate.runtime_profile
                or measurement.source_digest != candidate.source_digest
                or measurement.abi_version != candidate.abi_version
                or measurement.candidate_generation != candidate.candidate_generation
                or measurement.admitted_bytes <= 0
            ):
                candidate.quarantine("native NCC workspace identity is invalid")
                continue
            if not _is_qualified_ncc_profile(
                candidate.profile,
                search_shape=candidate.search_shape,
                batch_size=candidate.batch_size,
                source_digest=candidate.source_digest,
                abi_version=candidate.abi_version,
            ):
                continue
            if source_digest not in ("", candidate.source_digest):
                continue
            if abi_version not in ("", candidate.abi_version):
                continue
            if runtime_profile not in ("", candidate.runtime_profile):
                continue
            current_profile = _current_ncc_runtime_profile(
                resolved,
                source_digest=current_source_digest,
                abi_version=candidate.abi_version,
            )
            if current_profile != candidate.profile:
                continue
            return candidate
        return None

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
        executor=prepared.dispatch,
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
    batch_size: int,
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
        message = "search_shape must contain positive dimensions"
        logger.error(message)
        raise ValueError(message)
    if search_shape[0] < 3 or search_shape[1] < 3:
        message = "NCC postprocess requires a 3x3 or larger surface"
        logger.error(message)
        raise ValueError(message)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        message = "batch_size must be a positive integer"
        logger.error(message)
        raise TypeError(message)
    if batch_size < 1:
        message = "batch_size must be a positive integer"
        logger.error(message)
        raise ValueError(message)
    resolved_device = canonical_torch_device(device)
    if not resolved_device.startswith("cuda:"):
        message = "NCC postprocess native candidate requires CUDA"
        logger.error(message)
        raise ValueError(message)
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
        message = plan.unsupported_reason or "NCC native backend unsupported"
        logger.error(message)
        raise RuntimeError(message)
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
        message = prepared.reason or "NCC native preparation failed"
        logger.error(message)
        raise RuntimeError(message)
    module = module_holder["module"]
    source_abi = getattr(module, "native_source_abi", None)
    if source_abi is None or source_abi() != _NCC_NATIVE_ABI:
        message = "NCC native source ABI mismatch"
        logger.error(message)
        raise RuntimeError(message)
    entry = getattr(module, "ampcor_ncc_postprocess_cuda", None)
    if entry is None:
        message = "NCC native module has no postprocess entry point"
        logger.error(message)
        raise RuntimeError(message)

    height, width = search_shape
    source_digest = sha256(
        b"".join(path.read_bytes() for path in plan.sources)
    ).hexdigest()
    profile = _current_ncc_runtime_profile(
        resolved_device,
        source_digest=source_digest,
        abi_version=_NCC_NATIVE_ABI,
    )
    runtime_profile = profile.canonical() if profile is not None else ""
    candidate_generation = sha256(
        f"{plan.extension_name}:{source_digest}:{_NCC_NATIVE_ABI}".encode()
    ).hexdigest()
    sample_corr = _centered_sample((batch_size, height, width), resolved_device)
    sample_energy = torch.ones_like(sample_corr)
    torch.cuda.synchronize(torch.device(resolved_device))
    torch.cuda.reset_peak_memory_stats(torch.device(resolved_device))
    allocation_before = int(torch.cuda.memory_allocated(torch.device(resolved_device)))
    native_result = entry(
        sample_corr,
        sample_energy,
        height // 2,
        width // 2,
        True,
        0.0,
        1e9,
    )
    torch.cuda.synchronize(torch.device(resolved_device))
    peak_allocated = max(
        0,
        int(torch.cuda.max_memory_allocated(torch.device(resolved_device)))
        - allocation_before,
    )
    output_bytes = sum(
        int(value.numel()) * int(value.element_size()) for value in native_result
    )
    properties = torch.cuda.get_device_properties(torch.device(resolved_device))
    max_grid = int(getattr(properties, "max_grid_size", (batch_size,))[0])
    workspace_measurement = AmpcorNccWorkspaceMeasurement(
        operation=_NCC_OPERATION,
        search_shape=search_shape,
        batch_size=batch_size,
        runtime_profile=runtime_profile,
        source_digest=source_digest,
        abi_version=_NCC_NATIVE_ABI,
        candidate_generation=candidate_generation,
        peak_allocated_bytes=peak_allocated,
        output_bytes=output_bytes,
        global_scratch_bytes=0,
        shared_memory_bytes=3 * 256 * 8,
        launch_blocks=min(batch_size, max_grid),
        threads_per_block=256,
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
        batch_size=batch_size,
        runtime_profile=runtime_profile,
        source_digest=source_digest,
        profile=profile,
        performance_eligible=(search_shape, batch_size) in _NCC_PERFORMANCE_COMBINATIONS
        and _is_qualified_ncc_profile(
            profile,
            search_shape=search_shape,
            batch_size=batch_size,
            source_digest=source_digest,
            abi_version=_NCC_NATIVE_ABI,
        ),
        native_module=module,
        workspace_measurement=workspace_measurement,
        candidate_generation=candidate_generation,
    )


__all__ = [
    "AmpcorBackend",
    "AmpcorBackendRegistry",
    "AmpcorCandidateError",
    "AmpcorEnergyCandidate",
    "AmpcorNccCandidate",
    "AmpcorNccCandidateError",
    "AmpcorNccExecutionError",
    "AmpcorNccPreDispatchError",
    "AmpcorNccRuntimeProfile",
    "AmpcorNccWorkspaceMeasurement",
    "ampcor_ncc_postprocess_reference",
    "canonical_torch_device",
    "eager_ampcor_candidate",
    "native_workspace_bytes",
    "prepare_ampcor_compile",
    "prepare_ampcor_native",
    "prepare_ampcor_ncc_native",
    "torch_integral_energy",
]
