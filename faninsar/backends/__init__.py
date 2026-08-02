"""Backends for lazy loading and parallel computation.

Greenfield target: collapse into ``faninsar.compute`` (Dask/Torch),
``faninsar.ports`` (protocols), and ``faninsar.io`` (lazy readers).
This package remains the implementation home until callers migrate.
"""

from __future__ import annotations

from faninsar.io.lazy_rasterio import LazyMultiFileReader, LazyRasterioReader

from .dask_exec import compute_with_profile, map_elementwise, map_finite_halo
from .device_matrix import (
    DeviceKind,
    KernelCapability,
    capability_matrix,
    complex_multiply,
    probe_devices,
    select_device_for_kernel,
)
from .execution import (
    Backend,
    ChunkLayout,
    DTypePolicy,
    ExecutionAssignment,
    ExecutionProfile,
    ExecutionValidationError,
    Locality,
    MemoryBudget,
    OperationPlan,
    WorkerProfile,
    estimate_peak_bytes,
    plan_chunks,
    validate_execution,
)

__all__ = [
    "Backend",
    "ChunkLayout",
    "DTypePolicy",
    "DeviceKind",
    "ExecutionAssignment",
    "ExecutionProfile",
    "ExecutionValidationError",
    "KernelCapability",
    "LazyMultiFileReader",
    "LazyRasterioReader",
    "Locality",
    "MemoryBudget",
    "OperationPlan",
    "WorkerProfile",
    "capability_matrix",
    "complex_multiply",
    "compute_with_profile",
    "estimate_peak_bytes",
    "map_elementwise",
    "map_finite_halo",
    "plan_chunks",
    "probe_devices",
    "select_device_for_kernel",
    "validate_execution",
]
