"""Tests for locality-aware execution planning."""

from __future__ import annotations

import pytest

from faninsar.backends.execution import (
    Backend,
    ChunkLayout,
    DTypePolicy,
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


def test_cpu_profile_accepts_finite_halo_operation() -> None:
    """Accept a bounded CPU overlap operation."""
    plan = OperationPlan(
        name="coherence",
        locality=Locality.FINITE_HALO,
        backend=Backend.NUMPY,
        dtype_policy=DTypePolicy.SCIENTIFIC,
        layout=ChunkLayout(shape=(1024, 1024), chunks=((512, 512), (512, 512))),
        itemsize=8,
        temporary_arrays=4,
        halo=(5, 5),
    )
    profile = ExecutionProfile(
        workers=(WorkerProfile(name="cpu-0", resources=()),),
        memory=MemoryBudget(worker_bytes=512 * 1024**2, safety_fraction=0.8),
    )

    assert validate_execution(plan, profile).worker_name == "cpu-0"


def test_fft_rejects_multiple_chunks_on_transform_axis() -> None:
    """Reject a fragmented FFT transform axis."""
    plan = OperationPlan(
        name="azimuth_fft",
        locality=Locality.TILE_GLOBAL,
        backend=Backend.NUMPY,
        dtype_policy=DTypePolicy.SCIENTIFIC,
        layout=ChunkLayout(shape=(1024, 2048), chunks=((512, 512), (2048,))),
        itemsize=8,
        temporary_arrays=2,
        transform_axes=(0,),
    )

    with pytest.raises(ExecutionValidationError, match="transform axis 0"):
        _ = validate_execution(plan, ExecutionProfile.local_cpu())


def test_gpu_resource_rejects_profile_without_matching_worker() -> None:
    """Reject GPU work without a GPU worker."""
    plan = OperationPlan.cuda_elementwise(
        name="complex_multiply",
        shape=(256, 256),
        chunks=((256,), (256,)),
    )

    with pytest.raises(ExecutionValidationError, match="GPU resource"):
        _ = validate_execution(plan, ExecutionProfile.local_cpu())


def test_cuda_plan_accepts_localcluster_gpu_mock_profile() -> None:
    """Route GPU work to a resource-labelled mock worker."""
    profile = ExecutionProfile(
        workers=(WorkerProfile(name="gpu-mock-0", resources=("GPU",)),),
        memory=MemoryBudget(worker_bytes=1024**3, safety_fraction=0.75),
    )
    plan = OperationPlan.cuda_elementwise(
        name="complex_multiply",
        shape=(256, 256),
        chunks=((256,), (256,)),
    )

    assert validate_execution(plan, profile).worker_name == "gpu-mock-0"


def test_oversized_tile_rejected_by_peak_memory_estimate() -> None:
    """Reject a tile larger than the safe worker budget."""
    plan = OperationPlan(
        name="large_tile",
        locality=Locality.TILE_GLOBAL,
        backend=Backend.NUMPY,
        dtype_policy=DTypePolicy.SCIENTIFIC,
        layout=ChunkLayout(shape=(4096, 4096), chunks=((4096,), (4096,))),
        itemsize=8,
        temporary_arrays=6,
    )
    profile = ExecutionProfile(
        workers=(WorkerProfile(name="cpu-0", resources=()),),
        memory=MemoryBudget(worker_bytes=128 * 1024**2, safety_fraction=0.7),
    )

    with pytest.raises(ExecutionValidationError, match="memory budget"):
        _ = validate_execution(plan, profile)


def test_chunk_planner_and_estimator_are_deterministic() -> None:
    """Create stable bounded chunks and a stable peak estimate."""
    layout = plan_chunks(shape=(1000, 600), target_chunk_bytes=256_000, itemsize=8)

    assert layout.shape == (1000, 600)
    assert all(
        sum(axis_chunks) == axis
        for axis, axis_chunks in zip(layout.shape, layout.chunks, strict=True)
    )
    assert estimate_peak_bytes(layout, itemsize=8, temporary_arrays=2) <= 768_000


def test_mps_rejects_operation_outside_probed_subset() -> None:
    """Reject an unproven MPS operation instead of claiming CUDA parity."""
    plan = OperationPlan(
        name="irls_sparse_solve",
        locality=Locality.GLOBALLY_COUPLED,
        backend=Backend.TORCH_MPS,
        dtype_policy=DTypePolicy.ACCELERATED,
        layout=ChunkLayout(shape=(128, 128), chunks=((128,), (128,))),
        itemsize=8,
        temporary_arrays=3,
    )
    profile = ExecutionProfile(
        workers=(WorkerProfile(name="mps-0", resources=("MPS",)),),
        memory=MemoryBudget(worker_bytes=1024**3),
    )

    with pytest.raises(ExecutionValidationError, match="bounded MPS subset"):
        _ = validate_execution(plan, profile)


@pytest.mark.parametrize(
    ("worker_bytes", "safety_fraction"),
    [(0, 0.7), (1024, 0.0), (1024, 1.1)],
)
def test_memory_budget_rejects_malformed_values(
    worker_bytes: int, safety_fraction: float
) -> None:
    """Reject non-positive or unsafe worker memory policies."""
    with pytest.raises(ExecutionValidationError, match="Memory budget"):
        _ = MemoryBudget(
            worker_bytes=worker_bytes,
            safety_fraction=safety_fraction,
        )


def test_chunk_layout_rejects_incomplete_axis_coverage() -> None:
    """Reject chunks that do not exactly cover their array axis."""
    with pytest.raises(ExecutionValidationError, match="exactly cover shape"):
        _ = ChunkLayout(shape=(64, 64), chunks=((32,), (64,)))


def test_peak_estimator_rejects_mismatched_halo_rank() -> None:
    """Reject halo depths that cannot map to every array axis."""
    layout = ChunkLayout(shape=(64, 64), chunks=((64,), (64,)))

    with pytest.raises(ExecutionValidationError, match="Halo rank"):
        _ = estimate_peak_bytes(
            layout,
            itemsize=8,
            temporary_arrays=2,
            halo=(2,),
        )


@pytest.mark.parametrize(
    ("itemsize", "temporary_arrays", "halo"),
    [(-8, 2, (0, 0)), (8, -1, (0, 0)), (8, 2, (-1, 0))],
)
def test_peak_estimator_rejects_negative_memory_inputs(
    itemsize: int,
    temporary_arrays: int,
    halo: tuple[int, int],
) -> None:
    """Reject values that could create a negative peak estimate."""
    layout = ChunkLayout(shape=(64, 64), chunks=((64,), (64,)))

    with pytest.raises(ExecutionValidationError, match="non-negative"):
        _ = estimate_peak_bytes(
            layout,
            itemsize=itemsize,
            temporary_arrays=temporary_arrays,
            halo=halo,
        )
