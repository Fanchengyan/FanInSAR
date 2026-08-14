"""Contract and CPU-boundary tests for the native-v2 vertical slice."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from faninsar.processing.geometry.native_v2 import (
    NATIVE_RESULT_FIELDS,
    result_from_native_outputs,
)

SOURCE_ROOT = (
    Path(__file__).parents[3] / "faninsar" / "processing" / "geometry" / "native_v2"
)


@pytest.fixture(scope="module")
def serial_native_extension(tmp_path_factory: pytest.TempPathFactory) -> object:
    """Build the CPU sources without OpenMP for a diagnostic ABI check."""
    pytest.importorskip("torch")
    extension = pytest.importorskip("torch.utils.cpp_extension")
    build_dir = tmp_path_factory.mktemp("native-v2-serial")
    return extension.load(
        name="faninsar_native_v2_serial_fixture",
        sources=[
            str(SOURCE_ROOT / name)
            for name in (
                "bindings.cpp",
                "native_v2_abi.cpp",
                "geo2rdr.cpp",
                "rdr2geo.cpp",
            )
        ],
        extra_cflags=["-O0"],
        build_directory=str(build_dir),
        with_cuda=False,
        verbose=False,
    )


def test_cpu_sources_export_both_operation_symbols_and_openmp_loop() -> None:
    """Both operation translation units contain the exact OpenMP seam."""
    binding = (SOURCE_ROOT / "bindings.cpp").read_text()
    for operation in ("geo2rdr", "rdr2geo"):
        source = (SOURCE_ROOT / f"{operation}.cpp").read_text()
        assert f"{operation}_cpu" in source
        assert "#pragma omp parallel for" in source
        assert "record_visit(point)" in source
        assert f'"{operation}_cpu"' in binding


def test_native_result_binding_centralizes_fourteen_field_validation() -> None:
    """Native outputs are converted to the exact public v2 result contract."""
    values: list[np.ndarray] = [
        np.zeros(2, dtype=np.float64) for _ in NATIVE_RESULT_FIELDS
    ]
    values[5] = np.ones(2, dtype=bool)
    values[6] = np.ones(2, dtype=np.int32)
    values[9] = np.ones(2, dtype=np.float64)
    values[10] = np.zeros(2, dtype=bool)
    values[11] = np.zeros(2, dtype=bool)
    result = result_from_native_outputs(values, operation="geo2rdr")

    assert result.fields == NATIVE_RESULT_FIELDS
    assert result.iterations.dtype == np.dtype(np.int32)
    assert result.converged.dtype == np.dtype(bool)


def test_native_result_binding_rejects_non_fourteen_field_abi() -> None:
    """The Python seam fails closed if an extension returns the wrong ABI."""
    with pytest.raises(ValueError, match="exactly fourteen"):
        result_from_native_outputs([np.zeros(1)] * 13, operation="rdr2geo")
    with pytest.raises(ValueError, match="exactly fourteen"):
        result_from_native_outputs([np.zeros(1)] * 15, operation="rdr2geo")


def test_cpu_contract_requires_strict_metrics_and_invalid_lane_rules() -> None:
    """The native source documents strict metrics and invalid-lane sentinels."""
    for name in ("geo2rdr.cpp", "rdr2geo.cpp"):
        source = (SOURCE_ROOT / name).read_text()
        assert "< 1.0" in source or "< range_tol_m" in source
        assert (
            "exhausted_values[point] = false" in source
            or "exhausted[point] = false" in source
        )
        assert "std::numeric_limits<double>::quiet_NaN" in source


def test_cpu_telemetry_contract_keeps_qualification_metadata_out_of_result() -> None:
    """Operation and coverage metadata stay in telemetry, not result fields."""
    binding = (SOURCE_ROOT / "bindings.cpp").read_text()
    abi = (SOURCE_ROOT / "native_v2_abi.h").read_text()
    assert "operation_symbol" in binding
    assert "processed_point_count" in abi
    assert "observed_affinity" in abi
    assert "operation_symbol" not in NATIVE_RESULT_FIELDS


def test_cpu_validation_and_runtime_identity_fail_closed() -> None:
    """Native validation rejects nonfinite geometry and unknown runtimes."""
    abi_source = (SOURCE_ROOT / "native_v2_abi.cpp").read_text()
    rdr_source = (SOURCE_ROOT / "rdr2geo.cpp").read_text()
    assert "orbit position/velocity values must be finite" in abi_source
    assert "dem_samples must contain only finite values" in rdr_source
    assert "std::getenv" not in abi_source
    assert 'runtime_name = "unknown"' in abi_source


def test_rdr2geo_cpu_is_closed_form_tcn_not_finite_difference_newton() -> None:
    """The CPU core retains the accepted closed-form TCN construction."""
    source = (SOURCE_ROOT / "rdr2geo.cpp").read_text()
    assert "normal_dot_velocity" in source
    assert "velocity_dot_along" in source
    assert "cos_theta" in source
    assert "alpha" in source
    assert "beta" in source
    assert "determinant" not in source
    assert "d_range_dlat" not in source


@pytest.mark.skipif(
    os.environ.get("FANINSAR_TEST_NATIVE_V2_BUILD") != "1",
    reason="native extension build is explicitly enabled",
)
def test_serial_native_fixture_covers_invalid_lane_and_dem_path(tmp_path: Path) -> None:
    """An opt-in fixture covers invalid lanes, telemetry, and DEM fixed point."""
    torch = pytest.importorskip("torch")
    cpp_extension = pytest.importorskip("torch.utils.cpp_extension")
    module = cpp_extension.load(
        name="faninsar_native_v2_cpu_fixture",
        sources=[
            str(SOURCE_ROOT / name)
            for name in (
                "bindings.cpp",
                "native_v2_abi.cpp",
                "geo2rdr.cpp",
                "rdr2geo.cpp",
            )
        ],
        build_directory=str(tmp_path),
        extra_cflags=["-O0"],
        verbose=False,
    )
    dtype = torch.float64
    times = torch.tensor([-10.0, 10.0], dtype=dtype)
    positions = torch.tensor(
        [[7_000_000.0, -10_000.0, 0.0], [7_000_000.0, 10_000.0, 0.0]],
        dtype=dtype,
    )
    velocities = torch.tensor([[0.0, 1_000.0, 0.0], [0.0, 1_000.0, 0.0]], dtype=dtype)
    geo = module.geo2rdr_cpu(
        torch.tensor([0.0, float("nan")], dtype=dtype),
        torch.tensor([0.0, 0.0], dtype=dtype),
        torch.tensor([0.0, 0.0], dtype=dtype),
        times,
        positions,
        velocities,
        0.0,
        1.0,
        600_000.0,
        10.0,
        0.0555,
        20,
        0,
        1.0e-6,
        0.01,
        0.1,
        True,
    )
    assert bool(geo[5][0])
    assert int(geo[6][0]) > 0
    assert not bool(geo[5][1])
    assert int(geo[6][1]) == -1
    assert not bool(geo[10][1])
    assert float(geo[3][0]) == pytest.approx(2186.3, abs=1.0e-8)
    assert float(geo[8][0]) < 1.0
    telemetry = module.native_v2_telemetry()
    assert telemetry["operation_symbol"] == "geo2rdr_cpu"
    assert telemetry["processed_point_count"] == 2
    assert telemetry["visit_counts"] == [1, 1]

    dem = torch.full((6, 6), 0.0, dtype=dtype)
    rdr = module.rdr2geo_cpu_dem(
        torch.tensor([0.0], dtype=dtype),
        torch.tensor([2186.3], dtype=dtype),
        torch.tensor([0.0], dtype=dtype),
        times,
        positions,
        velocities,
        0.0,
        1.0,
        600_000.0,
        10.0,
        0.0555,
        20,
        0,
        0.01,
        0.1,
        True,
        dem,
        -0.2,
        -0.2,
        0.1,
        0.1,
        4,
        0.001,
    )
    assert bool(rdr[5][0])
    assert int(rdr[6][0]) >= 1
    assert float(rdr[2][0]) == pytest.approx(0.0)
    assert float(rdr[8][0]) < 1.0
    assert float(rdr[7][0]) == pytest.approx(float(rdr[12][0]))
    assert float(rdr[8][0]) == pytest.approx(float(rdr[12][0]))
    geo_source = (SOURCE_ROOT / "geo2rdr.cpp").read_text()
    assert "residual_range[point] = 0.0" not in geo_source

    exhausted = module.rdr2geo_cpu(
        torch.tensor([0.0], dtype=dtype),
        torch.tensor([2196.3], dtype=dtype),
        torch.tensor([0.0], dtype=dtype),
        times,
        positions,
        velocities,
        0.0,
        1.0,
        600_000.0,
        10.0,
        0.0555,
        1,
        0,
        0.01,
        0.1,
        True,
    )
    assert int(exhausted[6][0]) == 1
    assert bool(exhausted[10][0])
    assert not bool(exhausted[5][0])


def test_serial_cpu_extension_returns_validated_fourteen_field_result(
    serial_native_extension: object,
) -> None:
    """A serial diagnostic build publishes the shared v2 result contract."""
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    extension = serial_native_extension
    outputs = extension.geo2rdr_cpu(
        torch.tensor([0.0], dtype=dtype),
        torch.tensor([0.0], dtype=dtype),
        torch.tensor([0.0], dtype=dtype),
        torch.tensor([-10.0, 10.0], dtype=dtype),
        torch.tensor(
            [[7_000_000.0, -10_000.0, 0.0], [7_000_000.0, 10_000.0, 0.0]],
            dtype=dtype,
        ),
        torch.tensor([[0.0, 1_000.0, 0.0], [0.0, 1_000.0, 0.0]], dtype=dtype),
        0.0,
        1.0,
        600_000.0,
        10.0,
        0.056,
        20,
        5,
        1.0e-6,
        1.0e-4,
        1.0e-4,
        True,
    )
    result = result_from_native_outputs(outputs, operation="geo2rdr")

    assert len(outputs) == len(NATIVE_RESULT_FIELDS) == 14
    assert result.fields == NATIVE_RESULT_FIELDS
    assert result.converged.tolist() == [True]
    assert result.iterations.tolist() == [1]
    telemetry = extension.native_v2_telemetry()
    assert telemetry["openmp_defined"] is False
    assert telemetry["runtime_name"] == "unknown"
    assert telemetry["visit_counts"] == [1]
