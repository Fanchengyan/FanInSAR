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


def test_cuda_sources_expose_operation_entry_points_and_diagnostics() -> None:
    """Both operation kernels expose private, diagnostic-rich CUDA entry points."""
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
        assert "range_residual" in source
        assert "doppler_residual" in source
        assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in source
    assert "sample_dem" in common
    assert "spline_six" in common
    assert "is_contiguous()" in common


@pytest.mark.skipif(
    os.environ.get("FANINSAR_TEST_NATIVE_V2_CUDA") != "1",
    reason="native-v2 CUDA build test is explicitly enabled",
)
def test_cuda_direct_array_fixture_matches_constructed_geometry(tmp_path: Path) -> None:
    """Compile both kernels and solve a one-point ellipsoid fixture on CUDA."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    extension = pytest.importorskip("torch.utils.cpp_extension")
    binding = r'''
#include "geometry_cuda.cuh"
#include <vector>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("geo2rdr", &faninsar::geometry::cuda_v2::geo2rdr_cuda_v2);
  module.def("rdr2geo", &faninsar::geometry::cuda_v2::rdr2geo_tcn_cuda_v2);
}
'''
    module = extension.load_inline(
        name="faninsar_native_v2_cuda_fixture",
        cpp_sources=binding,
        cuda_sources=[
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
    latitude = torch.tensor([0.0], dtype=dtype, device=device)
    longitude = torch.tensor([0.0], dtype=dtype, device=device)
    height = torch.tensor([0.0], dtype=dtype, device=device)
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
        latitude,
        longitude,
        height,
        times,
        positions,
        velocities,
        0.0,
        1.0,
        600_000.0,
        10.0,
        20,
        5,
        1.0e-6,
        1.0e-4,
    )
    assert bool(geo[2].item())
    assert abs(float(geo[0].item())) < 1.0e-8
    assert abs(float(geo[4].item())) < 1.0e-4

    target_range = torch.tensor([621_863.0], dtype=dtype, device=device)
    seed = torch.tensor([0.0], dtype=dtype, device=device)
    satellite = torch.tensor([[7_000_000.0, 0.0, 0.0]], dtype=dtype, device=device)
    velocity = torch.tensor([[0.0, 1_000.0, 0.0]], dtype=dtype, device=device)
    empty_dem = torch.empty((), dtype=dtype, device=device)
    geo_from_radar = module.rdr2geo(
        target_range,
        seed,
        satellite,
        velocity,
        empty_dem,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.056,
        0.1,
        1.0e-4,
        20,
        5,
        True,
    )
    assert bool(geo_from_radar[3].item())
    assert abs(float(geo_from_radar[0].item())) < 1.0e-6
    assert abs(float(geo_from_radar[1].item())) < 1.0e-6
    assert abs(float(geo_from_radar[4].item())) < 0.1
    assert abs(float(geo_from_radar[5].item())) < 1.0e-4
