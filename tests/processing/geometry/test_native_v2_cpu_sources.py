"""Contract and CPU-boundary tests for the native-v2 vertical slice."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from faninsar.processing.geometry import ecef_to_llh
from faninsar.processing.geometry.native_v2 import (
    NATIVE_RESULT_FIELDS,
    result_from_native_outputs,
)
from faninsar.processing.geometry.torch_kernels import _rdr2geo_once

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


class _FakeCpuTransfer:
    """Expose one independent host transfer from a fake CUDA tensor."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def numpy(self) -> np.ndarray:
        """Return the host allocation represented by this transfer."""
        return self.values


class _FakeCudaTensor:
    """Model CUDA ``.cpu()`` transfers with fresh host allocations."""

    device = SimpleNamespace(type="cuda")

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.transfers: list[np.ndarray] = []

    def detach(self) -> _FakeCudaTensor:
        """Return the detached tensor view."""
        return self

    def cpu(self) -> _FakeCpuTransfer:
        """Return a fresh host copy, matching a CUDA-to-CPU transfer."""
        transfer = np.array(self.values, copy=True)
        self.transfers.append(transfer)
        return _FakeCpuTransfer(transfer)


class _FakeCpuTensor:
    """Model a CPU tensor whose host buffer may be reused by the caller."""

    device = SimpleNamespace(type="cpu")

    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def detach(self) -> _FakeCpuTensor:
        """Return the detached tensor view."""
        return self

    def cpu(self) -> _FakeCpuTransfer:
        """Return a view of the potentially reused CPU buffer."""
        return _FakeCpuTransfer(self.values)


def _valid_native_arrays(value: float) -> list[np.ndarray]:
    """Create valid two-lane arrays for the fourteen-field result ABI."""
    arrays = [np.full(2, value, dtype=np.float64) for _ in NATIVE_RESULT_FIELDS]
    arrays[5] = np.ones(2, dtype=bool)
    arrays[6] = np.ones(2, dtype=np.int32)
    arrays[9] = np.ones(2, dtype=np.float64)
    arrays[10] = np.zeros(2, dtype=bool)
    arrays[11] = np.zeros(2, dtype=bool)
    return arrays


def test_cuda_native_result_is_snapshot_without_second_host_copy() -> None:
    """A later CUDA transfer cannot overwrite an earlier public result."""
    tensors = [_FakeCudaTensor(values) for values in _valid_native_arrays(1.0)]

    first = result_from_native_outputs(tensors, operation="geo2rdr")
    second_values = _valid_native_arrays(2.0)
    for tensor, values in zip(tensors, second_values, strict=True):
        tensor.values[...] = values
    second = result_from_native_outputs(tensors, operation="geo2rdr")

    assert np.shares_memory(first.latitude_deg, tensors[0].transfers[0])
    assert not np.shares_memory(first.latitude_deg, second.latitude_deg)
    np.testing.assert_array_equal(first.latitude_deg, 1.0)
    np.testing.assert_array_equal(second.latitude_deg, 2.0)


def test_cpu_native_result_keeps_copy_for_reused_tensor_buffers() -> None:
    """CPU tensor buffers remain isolated from later native calls."""
    tensors = [_FakeCpuTensor(values) for values in _valid_native_arrays(1.0)]

    first = result_from_native_outputs(tensors, operation="geo2rdr")
    for tensor, values in zip(tensors, _valid_native_arrays(2.0), strict=True):
        tensor.values[...] = values
    second = result_from_native_outputs(tensors, operation="geo2rdr")

    np.testing.assert_array_equal(first.latitude_deg, 1.0)
    np.testing.assert_array_equal(second.latitude_deg, 2.0)


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


def test_native_dem_spline_uses_validated_closed_form_weights() -> None:
    """Lock the validated spline kernel while allowing iteration diagnostics to vary.

    The closed-form weights preserve the convergence mask and physical outputs
    within the qualification tolerances.  Floating-point evaluation order can
    legitimately change per-point iteration counters and residual diagnostics.
    """
    source = (SOURCE_ROOT / "rdr2geo.cpp").read_text()
    spline = source.split("double natural_spline_six", maxsplit=1)[1].split(
        "double sample_dem_six", maxsplit=1
    )[0]

    assert "constexpr double second_one[6]" in spline
    assert "constexpr double second_two[6]" in spline
    for coefficient in (
        "1.6076555023923444",
        "-3.6459330143540667",
        "2.5837320574162677",
        "-0.6889952153110048",
        "0.1722488038277512",
        "-0.0287081339712919",
        "-0.4306220095693780",
        "-4.3349282296650715",
        "2.7559808612440193",
        "0.1148325358851674",
    ):
        assert coefficient in spline
    assert "fraction_cubed" in spline
    assert "double weights[6]" in spline
    assert "result += values[index] * weights[index]" in spline
    assert "recurrence" not in spline


def test_cpu_publishes_degree_coordinates_and_final_attempt_residuals() -> None:
    """The native kernels publish degree coordinates and final-state metrics."""
    rdr = (SOURCE_ROOT / "rdr2geo.cpp").read_text()
    geo = (SOURCE_ROOT / "geo2rdr.cpp").read_text()
    assert "radians_to_degrees" in rdr
    assert "dem_latitude_start_deg" in rdr
    assert "final_range_residual" in rdr
    assert "final_doppler" in rdr
    assert "final_metric" in geo
    assert "attempts_evaluated >= budget" in geo
    assert "row_base > rows - 5" in rdr
    assert "column_base > columns - 5" in rdr
    assert "latitude_spacing == 0.0" in rdr
    assert "previous_fixed_height" in rdr
    assert "slope > -0.95 && slope < 0.95" in rdr
    assert "candidate >= dem_min" in rdr
    assert "candidate <= dem_max" in rdr


def test_geo2rdr_writes_residuals_through_raw_output_pointers() -> None:
    """Keep the per-point residual stores outside the Tensor API hot path."""
    source = (SOURCE_ROOT / "geo2rdr.cpp").read_text()

    assert "doppler_residuals[point] = last_doppler;" in source
    assert "range_residuals[point] = last_range_residual;" in source
    assert "residual_doppler[point] = last_doppler;" not in source
    assert "residual_range[point] = last_range_residual;" not in source


def test_native_dem_matches_torch_post_budget_ecef_transition() -> None:
    """The DEM loop carries Torch's post-budget ECEF damping state."""
    source = (SOURCE_ROOT / "rdr2geo.cpp").read_text()

    assert "iteration >= primary_budget" in source
    assert "old_latitude_rad" in source
    assert "old_longitude_rad" in source
    assert "0.5 * (old_xyz[0] + dem_xyz[0])" in source
    assert "average_llh = ecef_to_llh_tcn(average_xyz)" in source
    assert "next_old_height = average_llh[2]" in source
    assert "old_height = next_old_height" in source
    assert "aitken_enabled = false" in source
    assert "restart_after_damping" in source
    assert "previous_fixed_height = kNan" in source


def test_native_orbit_lookup_and_telemetry_visit_are_hot_loop_safe() -> None:
    """Orbit lookup is logarithmic and per-point telemetry has no lock."""
    abi = (SOURCE_ROOT / "native_v2_abi.cpp").read_text()
    record_visit = abi.split("void record_visit", maxsplit=1)[1].split(
        "TelemetrySnapshot telemetry_snapshot", maxsplit=1
    )[0]

    assert "std::upper_bound" in abi
    assert "std::lock_guard<std::mutex>" not in record_visit


def test_geo2rdr_cpu_packs_uniform_scene_orbit_and_hoists_seed() -> None:
    """The CPU fast path packs uniform Hermite knots and reuses its seed."""
    source = (SOURCE_ROOT / "geo2rdr.cpp").read_text()

    assert "pack_uniform_orbit" in source
    assert "interpolate_scene_orbit" in source
    assert "std::floor" in source
    assert "time_s < times[segment]" in source
    assert "time_s >= times[segment + 1]" in source
    assert "const double local_time = time_s - times[segment];" in source
    assert "const OrbitState reference_seed" in source
    assert "const double reference_speed_squared" in source
    assert "const OrbitState seed = reference_seed" in source
    assert "const double speed_squared = reference_speed_squared" in source


def test_geo2rdr_uniform_packed_orbit_matches_nonuniform_fallback(
    serial_native_extension: object,
) -> None:
    """Uniform packing preserves a nonlinear Hermite trajectory at boundaries."""
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    extension = serial_native_extension
    time_origin = 1.0e9
    uniform_times = time_origin + np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
    irregular_times = time_origin + np.array([-20.0, -9.0, 0.0, 11.0, 20.0])
    internal_knots = uniform_times[1:-1]
    knot_probes = np.column_stack(
        (
            np.nextafter(internal_knots, -np.inf),
            internal_knots,
            np.nextafter(internal_knots, np.inf),
        )
    ).reshape(-1)
    probe_times = np.concatenate(([uniform_times[0]], knot_probes, [uniform_times[-1]]))

    def trajectory(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return one cubic ECEF trajectory and its analytic velocity."""
        relative_time = times - time_origin
        position = np.column_stack(
            (
                7_000_000.0 + 0.01 * relative_time**3,
                7_500.0 * relative_time + 0.1 * relative_time**3,
                0.02 * relative_time**3,
            )
        )
        velocity = np.column_stack(
            (
                0.03 * relative_time**2,
                7_500.0 + 0.3 * relative_time**2,
                0.06 * relative_time**2,
            )
        )
        return position, velocity

    uniform_positions, uniform_velocities = trajectory(uniform_times)
    irregular_positions, irregular_velocities = trajectory(irregular_times)
    target_positions, target_velocities = trajectory(probe_times)
    targets = []
    for position, velocity in zip(target_positions, target_velocities, strict=True):
        radial = position / np.linalg.norm(position)
        look = np.cross(velocity, radial)
        look /= np.linalg.norm(look)
        targets.append(position + 650_000.0 * look)
    targets_array = np.asarray(targets)
    latitude, longitude, height = ecef_to_llh(
        targets_array[:, 0], targets_array[:, 1], targets_array[:, 2]
    )
    latitude = torch.as_tensor(np.asarray(latitude), dtype=dtype)
    longitude = torch.as_tensor(np.asarray(longitude), dtype=dtype)
    height = torch.as_tensor(np.asarray(height), dtype=dtype)

    def solve(
        times: np.ndarray, positions: np.ndarray, velocities: np.ndarray
    ) -> list[object]:
        return extension.geo2rdr_cpu(
            latitude,
            longitude,
            height,
            torch.as_tensor(times, dtype=dtype),
            torch.as_tensor(positions, dtype=dtype),
            torch.as_tensor(velocities, dtype=dtype),
            time_origin,
            1.0,
            600_000.0,
            10.0,
            0.056,
            40,
            5,
            1.0e-6,
            1.0e-4,
            1.0e-4,
            True,
        )

    packed = solve(uniform_times, uniform_positions, uniform_velocities)
    fallback = solve(irregular_times, irregular_positions, irregular_velocities)
    assert packed[5].tolist() == fallback[5].tolist()
    assert packed[6].tolist() == fallback[6].tolist()
    assert bool(packed[5][1:-1].all())
    for index in (3, 4, 7, 8, 12, 13):
        np.testing.assert_allclose(
            packed[index].detach().cpu().numpy(),
            fallback[index].detach().cpu().numpy(),
            rtol=0.0,
            atol=1.0e-5,
            equal_nan=True,
        )


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

    # Use a nonlinear, physically valid orbit so one Newton update is
    # insufficient under ordinary tolerances.  The target is constructed at
    # an in-range time with a perpendicular look vector, making the exhausted
    # lane stable for both packed uniform and generic Hermite interpolation.
    finite_miss_times = 1.0e9 + torch.tensor(
        [-20.0, -10.0, 0.0, 10.0, 20.0], dtype=dtype
    )
    finite_miss_relative_time = finite_miss_times - 1.0e9
    finite_miss_positions = torch.stack(
        (
            7_000_000.0 + 0.01 * finite_miss_relative_time**3,
            7_500.0 * finite_miss_relative_time + 0.1 * finite_miss_relative_time**3,
            0.02 * finite_miss_relative_time**3,
        ),
        dim=1,
    )
    finite_miss_velocities = torch.stack(
        (
            0.03 * finite_miss_relative_time**2,
            7_500.0 + 0.3 * finite_miss_relative_time**2,
            0.06 * finite_miss_relative_time**2,
        ),
        dim=1,
    )
    finite_miss_position = torch.stack(
        (
            torch.tensor(7_000_000.0, dtype=dtype) + 0.01 * 15.0**3,
            torch.tensor(7_500.0 * 15.0 + 0.1 * 15.0**3, dtype=dtype),
            torch.tensor(0.02 * 15.0**3, dtype=dtype),
        )
    )
    finite_miss_velocity = torch.stack(
        (
            torch.tensor(0.03 * 15.0**2, dtype=dtype),
            torch.tensor(7_500.0 + 0.3 * 15.0**2, dtype=dtype),
            torch.tensor(0.06 * 15.0**2, dtype=dtype),
        )
    )
    finite_miss_radial = finite_miss_position / torch.linalg.vector_norm(
        finite_miss_position
    )
    finite_miss_look = torch.linalg.cross(
        finite_miss_velocity, finite_miss_radial, dim=0
    )
    finite_miss_look /= torch.linalg.vector_norm(finite_miss_look)
    finite_miss_target = finite_miss_position + 650_000.0 * finite_miss_look
    finite_miss_latitude, finite_miss_longitude, finite_miss_height = ecef_to_llh(
        *(finite_miss_target.detach().cpu().numpy())
    )
    finite_miss = module.geo2rdr_cpu(
        torch.tensor([float(finite_miss_latitude)], dtype=dtype),
        torch.tensor([float(finite_miss_longitude)], dtype=dtype),
        torch.tensor([float(finite_miss_height)], dtype=dtype),
        finite_miss_times,
        finite_miss_positions,
        finite_miss_velocities,
        0.0,
        1.0,
        600_000.0,
        10.0,
        0.056,
        1,
        0,
        1.0e-6,
        1.0e-4,
        1.0e-4,
        True,
    )
    assert not bool(finite_miss[5][0])
    assert int(finite_miss[6][0]) == 1
    assert bool(finite_miss[10][0])
    assert torch.isfinite(finite_miss[7][0])

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
        -0.100001,
        -0.100001,
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
    dem_telemetry = module.native_v2_telemetry()
    assert dem_telemetry["operation_symbol"] == "rdr2geo_cpu_dem"
    assert dem_telemetry["processed_point_count"] == 1
    assert dem_telemetry["visit_counts"] == [1]
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
    assert module.native_v2_telemetry()["operation_symbol"] == "rdr2geo_cpu"


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


def test_rdr2geo_cpu_dem_accepts_full_2d_windows_and_negative_spacing(
    serial_native_extension: object,
) -> None:
    """The native DEM sampler bounds a full 2-D stencil in either direction."""
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    extension = serial_native_extension
    outputs = extension.rdr2geo_cpu_dem(
        torch.tensor([0.0], dtype=dtype),
        torch.tensor([2186.3], dtype=dtype),
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
        0.0555,
        20,
        0,
        0.01,
        0.1,
        True,
        torch.arange(100.0, dtype=dtype).reshape(10, 10) + 100.0,
        0.3,
        0.3,
        -0.1,
        -0.1,
        2,
        0.001,
    )

    assert bool(outputs[5][0])
    assert torch.isfinite(outputs[2][0])


def test_rdr2geo_cpu_dem_invalid_context_uses_invalid_lane_sentinels(
    serial_native_extension: object,
) -> None:
    """The fused DEM path matches legacy NaN sentinels for invalid points."""
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    extension = serial_native_extension
    outputs = extension.rdr2geo_cpu_dem(
        torch.tensor([0.0, float("nan")], dtype=dtype),
        torch.tensor([2186.3, 2186.3], dtype=dtype),
        torch.tensor([0.0, 25.0], dtype=dtype),
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
        0.0555,
        20,
        0,
        0.01,
        0.1,
        True,
        torch.full((10, 10), 25.0, dtype=dtype),
        0.3,
        0.3,
        -0.1,
        -0.1,
        3,
        0.001,
    )

    assert not bool(outputs[5][1])
    assert torch.isnan(outputs[0][1])
    assert torch.isnan(outputs[1][1])
    assert torch.isnan(outputs[2][1])
    assert torch.isnan(outputs[3][1])
    assert torch.isnan(outputs[4][1])
    assert torch.isnan(outputs[8][1])
    assert torch.isnan(outputs[12][1])
    assert torch.isnan(outputs[13][1])


def test_rdr2geo_cpu_dem_matches_final_constant_height_solve_for_2d_dem(
    serial_native_extension: object,
) -> None:
    """Fused DEM passes preserve the final solve for a real two-dimensional DEM."""
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    extension = serial_native_extension
    azimuth = torch.tensor([0.0, 0.1], dtype=dtype)
    range_index = torch.tensor([2186.3, 2186.4], dtype=dtype)
    seed = torch.zeros(2, dtype=dtype)
    times = torch.tensor([-10.0, 10.0], dtype=dtype)
    positions = torch.tensor(
        [[7_000_000.0, -10_000.0, 0.0], [7_000_000.0, 10_000.0, 0.0]],
        dtype=dtype,
    )
    velocities = torch.tensor([[0.0, 1_000.0, 0.0], [0.0, 1_000.0, 0.0]], dtype=dtype)
    common = (
        azimuth,
        range_index,
        times,
        positions,
        velocities,
    )
    dem = torch.full((10, 10), 25.0, dtype=dtype)
    fused = extension.rdr2geo_cpu_dem(
        common[0],
        common[1],
        seed,
        common[2],
        common[3],
        common[4],
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
        0.3,
        0.3,
        -0.1,
        -0.1,
        3,
        0.001,
    )
    assert bool(torch.all(fused[5]))
    assert torch.all(fused[6] > 0)
    assert torch.all(fused[6] <= 20)
    assert torch.max(torch.abs(fused[12])) < 0.01
    assert torch.max(torch.abs(fused[13])) < 1.0e-12
    assert torch.allclose(fused[2], torch.full_like(seed, 25.0), atol=1.0e-6)


def test_rdr2geo_cpu_dem_closes_variable_2d_dem_fixed_point(
    serial_native_extension: object,
) -> None:
    """The fused path closes a variable two-dimensional DEM fixed point."""
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    extension = serial_native_extension
    azimuth = torch.tensor([0.0, 0.1], dtype=dtype)
    range_index = torch.tensor([2186.3, 2186.4], dtype=dtype)
    seed = torch.zeros(2, dtype=dtype)
    times = torch.tensor([-10.0, 10.0], dtype=dtype)
    positions = torch.tensor(
        [[7_000_000.0, -10_000.0, 0.0], [7_000_000.0, 10_000.0, 0.0]],
        dtype=dtype,
    )
    velocities = torch.tensor([[0.0, 1_000.0, 0.0], [0.0, 1_000.0, 0.0]], dtype=dtype)
    dem_start = 0.3
    dem_spacing = -0.1
    latitude = dem_start + dem_spacing * torch.arange(20, dtype=dtype)
    longitude = dem_start + dem_spacing * torch.arange(20, dtype=dtype)
    dem = 1_000.0 + 100.0 * latitude[:, None] + 50.0 * longitude[None, :]

    args = (azimuth, range_index, times, positions, velocities)
    fused = extension.rdr2geo_cpu_dem(
        *args[:2],
        seed,
        *args[2:],
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
        dem_start,
        dem_start,
        dem_spacing,
        dem_spacing,
        4,
        0.001,
    )

    assert bool(torch.all(fused[5]))
    assert torch.all(fused[6] > 0)
    assert torch.all(fused[6] <= 20)
    dem_values = torch.stack(
        [
            1_000.0 + 100.0 * fused[0],
            50.0 * fused[1],
        ]
    ).sum(dim=0)
    assert torch.max(torch.abs(fused[2] - dem_values)) <= 1.0e-3
    assert torch.max(torch.abs(fused[12])) < 0.01
    assert torch.max(torch.abs(fused[13])) < 1.0e-12


def test_rdr2geo_cpu_commits_the_input_height_on_convergence(
    serial_native_extension: object,
) -> None:
    """Near-threshold native CPU lanes retain the height used for their solve."""
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    extension = serial_native_extension
    heights = torch.full((2,), 10.0, dtype=dtype)
    outputs = extension.rdr2geo_cpu(
        torch.zeros(2, dtype=dtype),
        torch.tensor([2186.3, 2186.4], dtype=dtype),
        heights,
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
        0.0555,
        20,
        0,
        0.01,
        0.1,
        True,
    )

    assert outputs[5].tolist() == [True, True]
    assert outputs[6].tolist() == [1, 1]
    assert torch.allclose(outputs[2], heights)
    assert torch.all(torch.abs(outputs[12]) < 0.01)


def test_rdr2geo_cpu_rebuilds_final_tcn_state_like_torch(
    serial_native_extension: object,
) -> None:
    """Native CPU recomputes committed-height coordinates before final metrics."""
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    count = 64
    phase = torch.linspace(-1.0, 1.0, count, dtype=dtype)
    azimuth = 2.0 * torch.sin(phase)
    range_index = 2186.3 + 2.0 * torch.cos(phase)
    height = torch.zeros(count, dtype=dtype)
    times = torch.arange(9, dtype=dtype) * 10.0
    positions = torch.stack(
        (
            torch.full_like(times, 7_000_000.0),
            -10_000.0 + 2_500.0 * torch.arange(9, dtype=dtype),
            100.0 * torch.arange(9, dtype=dtype),
        ),
        dim=-1,
    )
    velocities = torch.stack(
        (
            torch.zeros_like(times),
            torch.full_like(times, 1_000.0),
            torch.full_like(times, 7.5),
        ),
        dim=-1,
    )
    native = serial_native_extension.rdr2geo_cpu(
        azimuth,
        range_index,
        height,
        times,
        positions,
        velocities,
        40.0,
        0.002,
        600_000.0,
        10.0,
        0.0555,
        4,
        0,
        0.01,
        0.1,
        True,
    )
    torch_result = _rdr2geo_once(
        azimuth,
        range_index,
        height,
        times,
        positions,
        velocities,
        sensing_offset_s=40.0,
        azimuth_interval_s=0.002,
        starting_range_m=600_000.0,
        range_spacing_m=10.0,
        wavelength_m=0.0555,
        look_sign=1.0,
        max_iter=4,
        range_tol_m=0.01,
        doppler_tol_hz=0.1,
        dynamic_iterations=True,
    )
    common = native[5] & torch_result["converged"]

    assert bool(common.any())
    assert torch.allclose(native[0][common], torch_result["latitude_deg"][common])
    assert torch.allclose(native[1][common], torch_result["longitude_deg"][common])
    assert torch.allclose(native[12][common], torch_result["residual_range_m"][common])
