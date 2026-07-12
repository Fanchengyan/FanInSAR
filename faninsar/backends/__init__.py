"""Backends for lazy loading and parallel computation."""

from __future__ import annotations

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
from .lazy_rasterio import LazyMultiFileReader, LazyRasterioReader

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
