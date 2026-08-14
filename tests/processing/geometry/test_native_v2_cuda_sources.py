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
    assert "sample_dem" in common
    assert "bool* valid" in common
    assert "spline_six" in common
    assert "is_contiguous()" in common
    assert "geo2rdr_cuda_public" in binding
    assert "rdr2geo_cuda_public" in binding
    assert "result.pop_back()" in binding
    assert "geo2rdr_cuda_diagnostic" in binding
    assert "rdr2geo_cuda_diagnostic" in binding


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
    assert len(geo) == 15
    assert bool(geo[5][0])
    assert not bool(geo[5][1])
    assert int(geo[6][1]) == -1
    assert int(geo[14][0]) == 1
    assert all(torch.isnan(geo[index][1]) for index in RESULT_FLOAT_FIELDS)

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
    assert len(radar) == 15
    assert bool(radar[5][0])
    assert int(radar[6][0]) > 1
    assert not bool(radar[5][1])
    assert int(radar[6][1]) == -1
    assert int(radar[14][0]) == 1
    assert int(radar[14][1]) == 1
    assert all(torch.isnan(radar[index][1]) for index in RESULT_FLOAT_FIELDS)

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
