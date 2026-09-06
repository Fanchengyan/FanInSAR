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
    ProcessTreeAdmission,
    ProcessTreeMemoryWatchdog,
    ProcessTreeSampler,
    ProcessTreeSamplingError,
    ProcessTreeSnapshot,
    ResourceAdmissionError,
    ResourceAdmissionLedger,
    ResourceBudget,
    ResourceEstimate,
    ResourceReservation,
    ResourceUsage,
    bootstrap_worker_runtime,
)
from .source_snapshots import (
    ImmutableSourceSnapshot,
    SourceSnapshotEntry,
    snapshot_local_source,
)

__all__ = [
    "ArrayReader",
    "ArrayWriter",
    "ComputeBackend",
    "GpuMemoryReclaim",
    "ImmutableSourceSnapshot",
    "MemoryWatchdog",
    "ProcessTreeAdmission",
    "ProcessTreeMemoryWatchdog",
    "ProcessTreeSampler",
    "ProcessTreeSamplingError",
    "ProcessTreeSnapshot",
    "ReleaseGateResult",
    "ResourceAdmissionError",
    "ResourceAdmissionLedger",
    "ResourceBudget",
    "ResourceEstimate",
    "ResourceReservation",
    "ResourceUsage",
    "SourceSnapshotEntry",
    "available_memory_bytes",
    "bootstrap_worker_runtime",
    "close_memmap",
    "cuda_available",
    "gpu_available",
    "live_rss_bytes",
    "mps_available",
    "release_memmap_pages",
    "run_release_gates",
    "snapshot_local_source",
]
