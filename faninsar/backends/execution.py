"""Typed preflight contracts for locality-aware array execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from faninsar.logging import setup_logger

logger = setup_logger(__name__)


class Locality(StrEnum):
    """Data dependency class used to select a scheduling strategy."""

    ELEMENTWISE = "elementwise"
    FINITE_HALO = "finite-halo"
    TILE_GLOBAL = "tile-global"
    GLOBALLY_COUPLED = "globally-coupled"
    REDUCTION_NETWORK = "reduction-network"


class Backend(StrEnum):
    """Numerical implementation selected for an operation."""

    NUMPY = "numpy"
    SCIPY = "scipy"
    TORCH_CUDA = "torch-cuda"
    TORCH_MPS = "torch-mps"


class DTypePolicy(StrEnum):
    """Deterministic dtype policy for scientific and accelerated kernels."""

    SCIENTIFIC = "scientific"
    ACCELERATED = "accelerated"


MPS_SUPPORTED_OPERATIONS: Final[frozenset[str]] = frozenset(
    {"complex_multiply", "phase_rotation", "window_statistics"}
)


class ExecutionValidationError(RuntimeError):
    """Raised when an execution plan cannot safely be submitted."""


@dataclass(frozen=True, slots=True)
class ChunkLayout:
    """Full array shape and explicit per-axis Dask chunk lengths."""

    shape: tuple[int, ...]
    chunks: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """Reject malformed chunk layouts at construction time."""
        valid_rank = len(self.shape) == len(self.chunks) and bool(self.shape)
        valid_sizes = all(size > 0 for size in self.shape)
        exact_coverage = valid_rank and all(
            sum(axis_chunks) == axis_size and all(chunk > 0 for chunk in axis_chunks)
            for axis_size, axis_chunks in zip(self.shape, self.chunks, strict=True)
        )
        if valid_rank and valid_sizes and exact_coverage:
            return
        message = "Chunk layout must have positive dimensions and exactly cover shape"
        logger.error(message, stacklevel=2)
        raise ExecutionValidationError(message)

    @property
    def maximum_chunk_shape(self) -> tuple[int, ...]:
        """Return the largest chunk extent on each axis."""
        return tuple(max(axis_chunks) for axis_chunks in self.chunks)


@dataclass(frozen=True, slots=True)
class MemoryBudget:
    """Per-worker memory and the usable fraction reserved for tasks."""

    worker_bytes: int
    safety_fraction: float = 0.7

    def __post_init__(self) -> None:
        """Reject budgets that remove or overcommit safety headroom."""
        if self.worker_bytes > 0 and 0.0 < self.safety_fraction <= 1.0:
            return
        message = "Memory budget needs positive bytes and a fraction in (0, 1]"
        logger.error(message, stacklevel=2)
        raise ExecutionValidationError(message)

    @property
    def usable_bytes(self) -> int:
        """Return bytes available after reserving scheduler safety headroom."""
        return int(self.worker_bytes * self.safety_fraction)


@dataclass(frozen=True, slots=True)
class WorkerProfile:
    """Scheduler-visible worker name and resource labels."""

    name: str
    resources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """Worker capabilities and common per-worker memory policy."""

    workers: tuple[WorkerProfile, ...]
    memory: MemoryBudget

    @classmethod
    def local_cpu(cls) -> ExecutionProfile:
        """Build the default single-worker CPU profile."""
        return cls(
            workers=(WorkerProfile(name="local-cpu", resources=()),),
            memory=MemoryBudget(worker_bytes=2 * 1024**3),
        )


@dataclass(frozen=True, slots=True)
class OperationPlan:
    """Validated scheduling inputs for one numerical operation."""

    name: str
    locality: Locality
    backend: Backend
    dtype_policy: DTypePolicy
    layout: ChunkLayout
    itemsize: int
    temporary_arrays: int
    halo: tuple[int, ...] = ()
    transform_axes: tuple[int, ...] = ()

    @classmethod
    def cuda_elementwise(
        cls,
        *,
        name: str,
        shape: tuple[int, ...],
        chunks: tuple[tuple[int, ...], ...],
    ) -> OperationPlan:
        """Build a deterministic complex64 CUDA elementwise plan."""
        return cls(
            name=name,
            locality=Locality.ELEMENTWISE,
            backend=Backend.TORCH_CUDA,
            dtype_policy=DTypePolicy.ACCELERATED,
            layout=ChunkLayout(shape=shape, chunks=chunks),
            itemsize=8,
            temporary_arrays=2,
        )


@dataclass(frozen=True, slots=True)
class ExecutionAssignment:
    """Result of a successful preflight worker assignment."""

    worker_name: str
    estimated_peak_bytes: int


def plan_chunks(
    *, shape: tuple[int, ...], target_chunk_bytes: int, itemsize: int
) -> ChunkLayout:
    """Plan balanced chunks under a byte target.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array dimensions.
    target_chunk_bytes : int
        Maximum bytes in one input chunk.
    itemsize : int
        Bytes per array element.

    Returns
    -------
    ChunkLayout
        Deterministic balanced chunk layout.

    """
    if not shape or target_chunk_bytes <= 0 or itemsize <= 0:
        message = "Shape, target bytes, and item size must be positive"
        logger.error(message, stacklevel=2)
        raise ExecutionValidationError(message)
    elements = max(1, target_chunk_bytes // itemsize)
    edge = max(1, math.floor(math.exp(math.log(elements) / len(shape))))
    chunks = tuple(_split_axis(axis_size, min(axis_size, edge)) for axis_size in shape)
    return ChunkLayout(shape=shape, chunks=chunks)


def _split_axis(axis_size: int, chunk_size: int) -> tuple[int, ...]:
    count, remainder = divmod(axis_size, chunk_size)
    chunks = (chunk_size,) * count
    return chunks if remainder == 0 else (*chunks, remainder)


def estimate_peak_bytes(
    layout: ChunkLayout,
    *,
    itemsize: int,
    temporary_arrays: int,
    halo: tuple[int, ...] = (),
) -> int:
    """Estimate a conservative per-task live-array memory peak."""
    valid_memory_inputs = (
        itemsize > 0 and temporary_arrays >= 0 and all(depth >= 0 for depth in halo)
    )
    if not valid_memory_inputs:
        message = (
            "Item size must be positive; temporary arrays and halo must be non-negative"
        )
        logger.error(message, stacklevel=2)
        raise ExecutionValidationError(message)
    chunk_shape = layout.maximum_chunk_shape
    if halo:
        if len(halo) != len(chunk_shape):
            message = "Halo rank must match chunk rank"
            logger.error(message, stacklevel=2)
            raise ExecutionValidationError(message)
        chunk_shape = tuple(
            size + 2 * depth for size, depth in zip(chunk_shape, halo, strict=True)
        )
    live_elements = math.prod(chunk_shape) * (1 + temporary_arrays)
    return live_elements * itemsize


def validate_execution(
    plan: OperationPlan, profile: ExecutionProfile
) -> ExecutionAssignment:
    """Reject impossible resource, chunking, backend, and memory plans."""
    for axis in plan.transform_axes:
        axis_out_of_range = axis < 0 or axis >= len(plan.layout.chunks)
        axis_is_fragmented = (
            not axis_out_of_range and len(plan.layout.chunks[axis]) != 1
        )
        if axis_out_of_range or axis_is_fragmented:
            message = f"FFT transform axis {axis} must contain exactly one chunk"
            logger.error(message, stacklevel=2)
            raise ExecutionValidationError(message)

    required_resource = _required_resource(plan)
    matching_workers = tuple(
        worker
        for worker in profile.workers
        if required_resource is None or required_resource in worker.resources
    )
    if not matching_workers:
        message = f"No worker advertises required {required_resource or 'CPU'} resource"
        logger.error(message, stacklevel=2)
        raise ExecutionValidationError(message)

    peak_bytes = estimate_peak_bytes(
        plan.layout,
        itemsize=plan.itemsize,
        temporary_arrays=plan.temporary_arrays,
        halo=plan.halo,
    )
    if peak_bytes > profile.memory.usable_bytes:
        message = (
            f"Estimated peak {peak_bytes} exceeds memory budget "
            f"{profile.memory.usable_bytes}"
        )
        logger.error(message, stacklevel=2)
        raise ExecutionValidationError(message)
    return ExecutionAssignment(
        worker_name=matching_workers[0].name,
        estimated_peak_bytes=peak_bytes,
    )


def _required_resource(plan: OperationPlan) -> str | None:
    match plan.backend:
        case Backend.NUMPY | Backend.SCIPY:
            return None
        case Backend.TORCH_CUDA:
            return "GPU"
        case Backend.TORCH_MPS:
            if plan.name in MPS_SUPPORTED_OPERATIONS:
                return "MPS"
            message = f"Operation {plan.name!r} is outside the bounded MPS subset"
            logger.error(message, stacklevel=2)
            raise ExecutionValidationError(message)
