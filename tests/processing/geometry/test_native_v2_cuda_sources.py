"""Source and optional direct-array tests for the native-v2 CUDA kernels."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CUDA_SOURCE_ROOT = (
    Path(__file__).parents[3]
    / "faninsar"
    / "processing"
    / "geometry"
    / "native_v2"
    / "cuda"
)
RESULT_FLOAT_FIELDS = (0, 1, 2, 3, 4, 7, 8, 9, 12, 13)


def test_cuda_sources_expose_operation_entry_points_and_diagnostics() -> None:
    """Both operation kernels expose complete private CUDA entry points."""
    binding = (CUDA_SOURCE_ROOT.parent / "bindings.cpp").read_text()
    common = (CUDA_SOURCE_ROOT / "geometry_cuda.cuh").read_text()
    geo2rdr = (CUDA_SOURCE_ROOT / "geo2rdr_cuda.cu").read_text()
    rdr2geo = (CUDA_SOURCE_ROOT / "rdr2geo_tcn_cuda.cu").read_text()

    assert "geo2rdr_cuda_v2" in geo2rdr
    assert "rdr2geo_tcn_cuda_v2" in rdr2geo
    assert "rdr2geo_cuda_v2" in rdr2geo
    assert "_with_visit_counts" in geo2rdr
    assert "_with_visit_counts" in rdr2geo
    for source in (geo2rdr, rdr2geo):
        assert "max_iter" in source
        assert "extra_iter" in source
        assert "iterations" in source
        assert "converged" in source
        assert "decision_residual" in source
        assert "final_residual" in source
        assert "max_iter_exhausted" in source
        assert "boundary_rechecked" in source
        assert "visit_counts" in source
        assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in source
    assert "wavelength_m" in geo2rdr
    assert "doppler_tolerance_hz" in geo2rdr
    assert "orbit positions and velocities must be finite" in geo2rdr
    assert "orbit positions and velocities must be finite" in rdr2geo
    assert "iteration >= primary_iter" in rdr2geo
    assert "old_llh[1] / kDegreesToRadians" in rdr2geo
    assert "old_llh[0] / kDegreesToRadians" in rdr2geo
    assert "height = new_height" in rdr2geo
    assert "pop_back" in geo2rdr
    assert "pop_back" in rdr2geo
    assert "prepare_rdr2geo_contexts_kernel" in rdr2geo
    assert "kContextStride" in rdr2geo
    assert "row_width" in rdr2geo
    assert "row_count" in rdr2geo
    assert "atomicAdd" in rdr2geo
    assert "work_index" in rdr2geo
    assert "kBlocksPerSm" in rdr2geo
    assert "FANINSAR_NATIVE_V2_BLOCKS_PER_SM" in rdr2geo
    assert "multiProcessorCount" in rdr2geo
    assert "point_blocks" in rdr2geo
    assert "worker_blocks" in rdr2geo
    assert "prepare_rdr2geo_contexts_kernel<<<input_blocks" in rdr2geo
    assert "rdr2geo_tcn_kernel<<<worker_blocks" in rdr2geo
    assert "rdr2geo_tcn_cuda_v2_with_visit_counts_row_width" in rdr2geo
    assert "rdr2geo_tcn_cuda_v2_with_visit_counts_row_width" in common
    assert "row_width" in binding
    assert "azimuth.narrow" in binding
    assert "row_width = 1" in binding
    assert "sample_dem" in common
    assert "bool* valid" in common
    assert "spline_six" in common
    assert "spline_six_weights" in common
    assert "recurrence" not in common
    # CUDA and Torch/CPU TCN paths must use the same closed-form ECEF -> LLH
    # conversion; an iterative latitude update introduces millimetre-scale
    # coordinate drift that is visible to the public parity contract.
    assert "const double e4 = kWgs84E2 * kWgs84E2" in common
    assert "const double cubic = e4 * lateral * polar" in common
    assert "latitude_estimate" not in common
    assert "is_contiguous()" in common
    assert "geo2rdr_cuda_public" in binding
    assert "rdr2geo_cuda_public" in binding
    assert "geo2rdr_cuda_visit_counts" in binding
    assert "rdr2geo_cuda_visit_counts" in binding
    assert '"native CUDA geo2rdr public ABI must return 14 fields"' in binding


def test_cuda_rdr2geo_exhaustion_uses_common_final_publication() -> None:
    """Budget exhaustion publishes the final state without marking it solved."""
    source = (CUDA_SOURCE_ROOT / "rdr2geo_tcn_cuda.cu").read_text()

    assert (
        "const bool publish_converged = final_converged && !budget_exhausted;" in source
    )
    assert (
        "iterations[point] = budget_exhausted ? "
        "static_cast<int32_t>(budget) : attempts;" in source
    )
    assert "max_iter_exhausted[point] = budget_exhausted;" in source
    assert "converged[point] = publish_converged;" in source


@pytest.mark.skipif(
    os.environ.get("FANINSAR_TEST_NATIVE_V2_CUDA") != "1",
    reason="native-v2 CUDA build test is explicitly enabled",
)
def test_cuda_direct_array_fixture_matches_result_contract(tmp_path: Path) -> None:
    """Compile both kernels and exercise valid, invalid, boundary, and DEM-OOB lanes."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    extension = pytest.importorskip("torch.utils.cpp_extension")
    binding = r"""
#include "geometry_cuda.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("geo2rdr", &faninsar::geometry::cuda_v2::geo2rdr_cuda_v2);
  module.def("rdr2geo", &faninsar::geometry::cuda_v2::rdr2geo_tcn_cuda_v2);
  module.def("geo2rdr_visit_counts",
             &faninsar::geometry::cuda_v2::geo2rdr_cuda_v2_visit_counts);
  module.def("rdr2geo_visit_counts",
             &faninsar::geometry::cuda_v2::rdr2geo_tcn_cuda_v2_visit_counts);
}
"""
    (tmp_path / "binding.cpp").write_text(binding)
    module = extension.load(
        name="faninsar_native_v2_cuda_fixture",
        sources=[
            str(tmp_path / "binding.cpp"),
            str(CUDA_SOURCE_ROOT / "geo2rdr_cuda.cu"),
            str(CUDA_SOURCE_ROOT / "rdr2geo_tcn_cuda.cu"),
        ],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        extra_include_paths=[str(CUDA_SOURCE_ROOT)],
        build_directory=str(tmp_path),
        with_cuda=True,
        verbose=False,
    )
    dtype = torch.float64
    device = torch.device("cuda")
    times = torch.tensor([-10.0, 10.0], dtype=dtype, device=device)
    positions = torch.tensor(
        [[7_000_000.0, -10_000.0, 0.0], [7_000_000.0, 10_000.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    velocities = torch.tensor(
        [[0.0, 1_000.0, 0.0], [0.0, 1_000.0, 0.0]], dtype=dtype, device=device
    )
    geo = module.geo2rdr(
        torch.tensor([0.0, float("nan")], dtype=dtype, device=device),
        torch.tensor([0.0, 0.0], dtype=dtype, device=device),
        torch.tensor([0.0, 0.0], dtype=dtype, device=device),
        times,
        positions,
        velocities,
        0.0,
        1.0,
        600_000.0,
        10.0,
        0.056,
        20,
        5,
        1.0e-6,
        1.0e-3,
        1.0e-4,
    )
    assert len(geo) == 14
    assert bool(geo[5][0])
    assert not bool(geo[5][1])
    assert int(geo[6][1]) == -1
    assert all(torch.isnan(geo[index][1]) for index in RESULT_FLOAT_FIELDS)
    visits = module.geo2rdr_visit_counts(
        torch.tensor([0.0, float("nan")], dtype=dtype, device=device),
        torch.tensor([0.0, 0.0], dtype=dtype, device=device),
        torch.tensor([0.0, 0.0], dtype=dtype, device=device),
        times,
        positions,
        velocities,
        0.0,
        1.0,
        600_000.0,
        10.0,
        0.056,
        20,
        5,
        1.0e-6,
        1.0e-3,
        1.0e-4,
    )
    assert visits.tolist() == [1, 1]

    radar = module.rdr2geo(
        torch.tensor([0.0, 20.0, float("nan")], dtype=dtype, device=device),
        torch.tensor([2186.3, 2186.3, 2186.3], dtype=dtype, device=device),
        torch.zeros(3, dtype=dtype, device=device),
        times,
        positions,
        velocities,
        0.0,
        1.0,
        600_000.0,
        10.0,
        torch.empty((), dtype=dtype, device=device),
        0.0,
        0.0,
        1.0,
        1.0,
        100.0,
        -1_000.0,
        1_000.0,
        0.056,
        0.1,
        1.0e-4,
        20,
        5,
        True,
    )
    assert len(radar) == 14
    assert bool(radar[5][0])
    assert int(radar[6][0]) > 1
    assert not bool(radar[5][1])
    assert int(radar[6][1]) == -1
    assert all(torch.isnan(radar[index][1]) for index in RESULT_FLOAT_FIELDS)

    variable_dem = 100.0 + 0.01 * torch.arange(64, dtype=dtype, device=device).reshape(
        8, 8
    )
    exhausted = module.rdr2geo(
        torch.tensor([0.0], dtype=dtype, device=device),
        torch.tensor([2186.3], dtype=dtype, device=device),
        torch.zeros(1, dtype=dtype, device=device),
        times,
        positions,
        velocities,
        0.0,
        1.0,
        600_000.0,
        10.0,
        variable_dem,
        -0.6,
        -0.6,
        0.2,
        0.2,
        100.0,
        -1_000.0,
        1_000.0,
        0.056,
        1.0e-12,
        0.1,
        1,
        0,
        True,
    )
    assert not bool(exhausted[5][0])
    assert int(exhausted[6][0]) == 1
    assert bool(exhausted[10][0])
    for index in (0, 1, 2, 7, 8, 12, 13):
        assert bool(torch.isfinite(exhausted[index][0]))

    dem = torch.full((8, 8), 100.0, dtype=dtype, device=device)
    out_of_bounds = module.rdr2geo(
        torch.tensor([0.0], dtype=dtype, device=device),
        torch.tensor([2186.3], dtype=dtype, device=device),
        torch.zeros(1, dtype=dtype, device=device),
        times,
        positions,
        velocities,
        0.0,
        1.0,
        600_000.0,
        10.0,
        dem,
        100.0,
        100.0,
        0.2,
        0.2,
        100.0,
        -1_000.0,
        1_000.0,
        0.056,
        0.1,
        1.0e-4,
        20,
        5,
        True,
    )
    assert not bool(out_of_bounds[5][0])
    assert torch.isnan(out_of_bounds[0][0])
    assert int(out_of_bounds[6][0]) == 0
