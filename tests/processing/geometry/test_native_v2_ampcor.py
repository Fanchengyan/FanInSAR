"""Focused tests for the P28 Ampcor native CPU slice."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from faninsar.processing.geometry.native_v2 import (
    GeometryOperation,
    NativeBackend,
    NativeBuilder,
    NativeBuildRequest,
    NativeOperation,
)

SOURCE_ROOT = (
    Path(__file__).parents[3] / "faninsar" / "processing" / "geometry" / "native_v2"
)


def test_ampcor_plan_has_exact_identity_sources_and_openmp_flags() -> None:
    """Ampcor has a dedicated CPU plan independent of geometry symbols."""
    plan = NativeBuilder().plan(
        NativeBuildRequest(
            NativeOperation.AMPCOR_PREFIX_ENERGY,
            NativeBackend.CPU,
            source_root=SOURCE_ROOT,
            platform="linux",
            compiler="g++",
        )
    )

    assert plan.operation is NativeOperation.AMPCOR_PREFIX_ENERGY
    assert plan.extension_name == "faninsar_ampcor_prefix_energy_v1_cpu"
    assert plan.geometry_symbol == plan.extension_name
    assert plan.sources == (
        SOURCE_ROOT / "ampcor_bindings.cpp",
        SOURCE_ROOT / "ampcor_prefix_energy.cpp",
    )
    assert plan.compile_flags == ("-fopenmp", "-DFANINSAR_OPENMP_RUNTIME_LIBGOMP")
    assert plan.link_flags == ("-fopenmp",)
    assert plan.supported


def test_ampcor_cuda_plan_uses_a_dedicated_binding_and_kernel() -> None:
    """CUDA Ampcor is a supported one-operation extension plan."""
    plan = NativeBuilder().plan(
        NativeBuildRequest(
            NativeOperation.AMPCOR_PREFIX_ENERGY,
            NativeBackend.CUDA,
            source_root=SOURCE_ROOT,
        )
    )

    assert plan.extension_name == "faninsar_ampcor_prefix_energy_v1_cuda"
    assert plan.geometry_symbol == plan.extension_name
    assert plan.sources == (
        SOURCE_ROOT / "ampcor_cuda_bindings.cpp",
        SOURCE_ROOT / "ampcor_prefix_energy_cuda.cu",
    )
    assert plan.compile_flags == (
        "-O3",
        "-DFANINSAR_NATIVE_AMPCOR_CUDA=1",
    )
    assert plan.link_flags == ()
    assert plan.supported
    assert plan.unsupported_reason == ""


def test_ampcor_ncc_cuda_plan_is_experimental_and_separate() -> None:
    """NCC postprocess uses its own CUDA binding and ABI operation."""
    plan = NativeBuilder().plan(
        NativeBuildRequest(
            NativeOperation.AMPCOR_NCC_POSTPROCESS,
            NativeBackend.CUDA,
            source_root=SOURCE_ROOT,
        )
    )
    assert plan.extension_name == "faninsar_ampcor_ncc_postprocess_v1_cuda"
    assert plan.sources == (
        SOURCE_ROOT / "ampcor_ncc_cuda_bindings.cpp",
        SOURCE_ROOT / "ampcor_ncc_postprocess_cuda.cu",
    )
    assert plan.supported


def test_ampcor_ncc_cuda_source_preserves_peak_and_cull_contract() -> None:
    """The prototype source carries the strict numerical boundary markers."""
    binding = (SOURCE_ROOT / "ampcor_ncc_cuda_bindings.cpp").read_text()
    source = (SOURCE_ROOT / "ampcor_ncc_postprocess_cuda.cu").read_text()
    assert '"faninsar.ampcor_ncc_postprocess.v1"' in binding
    assert '"ampcor_ncc_postprocess_cuda"' in binding
    assert "torch::kFloat64" in source
    assert "is_contiguous()" in source
    assert "CUDAGuard" in source
    assert "getCurrentCUDAStream" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in source
    assert "checked_twice_plus_one" in source
    assert "blockIdx.x" in source
    assert "__shared__" in source
    assert "llabs(row - peak_az) <= 1" in source
    assert "fabs(az_denom) < 1e-12" in source
    assert "snr >= snr_threshold" in source


def test_geometry_operation_remains_a_compatible_native_operation_alias() -> None:
    """Existing geometry callers retain their enum members and values."""
    assert GeometryOperation is NativeOperation
    assert GeometryOperation.GEO2RDR.value == "geo2rdr"
    assert GeometryOperation.RDR2GEO.value == "rdr2geo"


def test_ampcor_sources_define_a_validated_static_openmp_boundary() -> None:
    """The Ampcor translation units expose only the validated local-energy ABI."""
    binding = (SOURCE_ROOT / "ampcor_bindings.cpp").read_text()
    source = (SOURCE_ROOT / "ampcor_prefix_energy.cpp").read_text()

    assert '"ampcor_prefix_energy_cpu"' in binding
    assert "geo2rdr" not in binding
    assert "rdr2geo" not in binding
    assert "torch::kFloat64" in source
    assert "is_cuda()" in source
    assert "dim() == 3" in source
    assert "is_contiguous()" in source
    assert "std::numeric_limits<int64_t>::max()" in source
    assert "#pragma omp parallel for schedule(static)" in source
    assert "clamp" not in source
    assert "normalize" not in source.lower()


def test_ampcor_cuda_sources_define_a_validated_single_operation_boundary() -> None:
    """The CUDA extension has one binding and validates its device contract."""
    binding = (SOURCE_ROOT / "ampcor_cuda_bindings.cpp").read_text()
    source = (SOURCE_ROOT / "ampcor_prefix_energy_cuda.cu").read_text()

    assert binding.count("PYBIND11_MODULE") == 1
    assert '"ampcor_prefix_energy_cuda"' in binding
    assert "ampcor_prefix_energy_cpu" not in binding
    assert "geo2rdr" not in binding
    assert "rdr2geo" not in binding
    assert "is_cuda()" in source
    assert "current_device" in source
    assert "torch::kFloat64" in source
    assert "dim() == 3" in source
    assert "is_contiguous()" in source
    assert "std::numeric_limits<int64_t>::max()" in source
    assert "__global__" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in source
    assert "ampcor_prefix_energy_cpu" not in source


@pytest.mark.skipif(
    os.environ.get("FANINSAR_TEST_NATIVE_V2_BUILD") != "1",
    reason="native extension build is explicitly enabled",
)
def test_ampcor_native_cpu_fixture_returns_float64_local_energy(tmp_path: Path) -> None:
    """An opt-in real build computes each secondary-search window energy."""
    torch = pytest.importorskip("torch")
    extension = pytest.importorskip("torch.utils.cpp_extension")
    plan = NativeBuilder().plan(
        NativeBuildRequest(
            NativeOperation.AMPCOR_PREFIX_ENERGY,
            NativeBackend.CPU,
            source_root=SOURCE_ROOT,
        )
    )
    if not plan.supported:
        pytest.skip(plan.unsupported_reason)
    module = extension.load(
        name=plan.extension_name,
        sources=[str(source) for source in plan.sources],
        extra_cflags=list(plan.compile_flags),
        extra_ldflags=list(plan.link_flags),
        build_directory=str(tmp_path),
        with_cuda=False,
        verbose=False,
    )

    secondary = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-1.0, 0.5, 2.0], [3.0, -4.0, 5.0]],
        ],
        dtype=torch.float64,
    )
    result = module.ampcor_prefix_energy_cpu(secondary, 2, 2)

    assert result.dtype == torch.float64
    assert result.device.type == "cpu"
    assert tuple(result.shape) == (2, 1, 2)
    torch.testing.assert_close(
        result,
        torch.tensor([[[46.0, 74.0]], [[26.25, 45.25]]], dtype=torch.float64),
    )


@pytest.mark.skipif(
    os.environ.get("FANINSAR_TEST_NATIVE_V2_BUILD") != "1",
    reason="native extension build is explicitly enabled",
)
def test_ampcor_native_cpu_fixture_rejects_invalid_contract(tmp_path: Path) -> None:
    """The real extension rejects dtype, rank, layout, dimensions, and overflow."""
    torch = pytest.importorskip("torch")
    extension = pytest.importorskip("torch.utils.cpp_extension")
    plan = NativeBuilder().plan(
        NativeBuildRequest(
            NativeOperation.AMPCOR_PREFIX_ENERGY,
            NativeBackend.CPU,
            source_root=SOURCE_ROOT,
        )
    )
    if not plan.supported:
        pytest.skip(plan.unsupported_reason)
    module = extension.load(
        name=f"{plan.extension_name}_invalid",
        sources=[str(source) for source in plan.sources],
        extra_cflags=list(plan.compile_flags),
        extra_ldflags=list(plan.link_flags),
        build_directory=str(tmp_path),
        with_cuda=False,
        verbose=False,
    )

    valid = torch.ones((1, 3, 3), dtype=torch.float64)
    with pytest.raises(RuntimeError, match="float64"):
        module.ampcor_prefix_energy_cpu(valid.float(), 2, 2)
    with pytest.raises(RuntimeError, match="three-dimensional"):
        module.ampcor_prefix_energy_cpu(valid[0], 2, 2)
    with pytest.raises(RuntimeError, match="contiguous"):
        module.ampcor_prefix_energy_cpu(valid[:, :, ::2], 2, 2)
    with pytest.raises(RuntimeError, match="positive"):
        module.ampcor_prefix_energy_cpu(valid, 0, 2)
    with pytest.raises(RuntimeError, match="fit"):
        module.ampcor_prefix_energy_cpu(valid, 4, 2)


@pytest.mark.skipif(
    os.environ.get("FANINSAR_TEST_NATIVE_V2_CUDA_BUILD") != "1",
    reason="native CUDA extension build is explicitly enabled",
)
def test_ampcor_native_cuda_fixture_matches_float64_local_energy(
    tmp_path: Path,
) -> None:
    """An opt-in real CUDA build computes local energy with float64 parity."""
    torch = pytest.importorskip("torch")
    extension = pytest.importorskip("torch.utils.cpp_extension")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    plan = NativeBuilder().plan(
        NativeBuildRequest(
            NativeOperation.AMPCOR_PREFIX_ENERGY,
            NativeBackend.CUDA,
            source_root=SOURCE_ROOT,
        )
    )
    module = extension.load(
        name=plan.extension_name,
        sources=[str(source) for source in plan.sources],
        extra_cflags=list(plan.compile_flags),
        extra_cuda_cflags=list(plan.compile_flags),
        build_directory=str(tmp_path),
        with_cuda=True,
        verbose=False,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    secondary = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[-1.0, 0.5, 2.0], [3.0, -4.0, 5.0], [6.0, 2.0, -3.0]],
        ],
        dtype=torch.float64,
        device=device,
    )
    result = module.ampcor_prefix_energy_cuda(secondary, 2, 2)
    expected = secondary.square().unfold(1, 2, 1).unfold(2, 2, 1).sum((-1, -2))

    assert result.dtype == torch.float64
    assert result.device == device
    assert tuple(result.shape) == (2, 2, 2)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(
    os.environ.get("FANINSAR_TEST_NATIVE_V2_CUDA_BUILD") != "1",
    reason="native CUDA extension build is explicitly enabled",
)
def test_ampcor_native_cuda_fixture_rejects_invalid_contract(tmp_path: Path) -> None:
    """The CUDA extension rejects CPU, dtype, rank, layout, and dimensions."""
    torch = pytest.importorskip("torch")
    extension = pytest.importorskip("torch.utils.cpp_extension")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    plan = NativeBuilder().plan(
        NativeBuildRequest(
            NativeOperation.AMPCOR_PREFIX_ENERGY,
            NativeBackend.CUDA,
            source_root=SOURCE_ROOT,
        )
    )
    module = extension.load(
        name=f"{plan.extension_name}_invalid",
        sources=[str(source) for source in plan.sources],
        extra_cflags=list(plan.compile_flags),
        extra_cuda_cflags=list(plan.compile_flags),
        build_directory=str(tmp_path),
        with_cuda=True,
        verbose=False,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    valid = torch.ones((1, 3, 3), dtype=torch.float64, device=device)
    with pytest.raises(RuntimeError, match="CUDA"):
        module.ampcor_prefix_energy_cuda(valid.cpu(), 2, 2)
    with pytest.raises(RuntimeError, match="float64"):
        module.ampcor_prefix_energy_cuda(valid.float(), 2, 2)
    with pytest.raises(RuntimeError, match="three-dimensional"):
        module.ampcor_prefix_energy_cuda(valid[0], 2, 2)
    with pytest.raises(RuntimeError, match="contiguous"):
        module.ampcor_prefix_energy_cuda(valid[:, :, ::2], 2, 2)
    with pytest.raises(RuntimeError, match="positive"):
        module.ampcor_prefix_energy_cuda(valid, 0, 2)
    with pytest.raises(RuntimeError, match="fit"):
        module.ampcor_prefix_energy_cuda(valid, 4, 2)
