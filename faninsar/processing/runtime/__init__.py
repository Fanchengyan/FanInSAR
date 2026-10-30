"""Execution runtime for device admission, Dask scheduling, and compilation."""

from __future__ import annotations

from .backends import ArrayReader, ArrayWriter
from .device import GpuMemoryReclaim, cuda_available, gpu_available, mps_available
from .memory import (
    MemoryWatchdog,
    available_memory_bytes,
    close_memmap,
    live_rss_bytes,
    release_memmap_pages,
)
from .protocols import ComputeBackend
from .release import ReleaseGateResult, run_release_gates
from .resources import (
    ResourceAdmissionError,
    ResourceAdmissionLedger,
    ResourceBudget,
    ResourceEstimate,
    ResourceReservation,
    ResourceUsage,
)

__all__ = [
    "ArrayReader",
    "ArrayWriter",
    "ComputeBackend",
    "GpuMemoryReclaim",
    "MemoryWatchdog",
    "ReleaseGateResult",
    "ResourceAdmissionError",
    "ResourceAdmissionLedger",
    "ResourceBudget",
    "ResourceEstimate",
    "ResourceReservation",
    "ResourceUsage",
    "available_memory_bytes",
    "close_memmap",
    "cuda_available",
    "gpu_available",
    "live_rss_bytes",
    "mps_available",
    "release_memmap_pages",
    "run_release_gates",
]
