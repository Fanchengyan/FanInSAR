"""Tests for the prepared fixed-shape Torch compile fast path."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.geometry import torch_backends_v2, torch_kernels

from .test_public_geometry_v2 import _model


def test_cpu_geo2rdr_compile_creates_two_unwarmed_graphs_for_real_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU compile creates two wrappers and first invokes them with real data."""
    import torch

    compile_functions: list[object] = []
    invocations: list[tuple[object, ...]] = []

    def fake_compile(function: object, **kwargs: object) -> object:
        del kwargs
        compile_functions.append(function)

        def compiled(*args: object, **call_kwargs: object) -> object:
            del call_kwargs
            invocations.append(args)
            return function(*args)  # type: ignore[operator]

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    prepared = torch_backends_v2.prepare_torch_geometry(
        "geo2rdr",
        _model(),
        shape=(1,),
        compile_kernel=True,
        max_iter=3,
    )

    assert len(compile_functions) == 2
    assert invocations == []
    latitude = np.array([-60.0], dtype=np.float64)
    longitude = np.array([-175.5], dtype=np.float64)
    height = np.array([123.0], dtype=np.float64)
    prepared.execute(latitude, longitude, height)

    assert len(invocations) >= 2
    np.testing.assert_array_equal(invocations[0][0], torch.as_tensor(latitude))


def test_cpu_geo2rdr_compile_wrappers_are_immutable_across_concurrent_first_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent first executions reuse the two prepared wrappers."""
    from concurrent.futures import ThreadPoolExecutor

    import torch

    compile_count = 0

    def fake_compile(function: object, **kwargs: object) -> object:
        nonlocal compile_count
        del kwargs
        compile_count += 1
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    prepared = torch_backends_v2.prepare_torch_geometry(
        "geo2rdr", _model(), shape=(1,), compile_kernel=True, max_iter=3
    )
    assert compile_count == 2

    values = (
        np.array([-60.0], dtype=np.float64),
        np.array([-175.5], dtype=np.float64),
        np.array([123.0], dtype=np.float64),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: prepared.execute(*values), range(2)))

    assert compile_count == 2
    assert all(result.iterations.tolist() == [3] for result in results)


def test_cpu_geo2rdr_compile_reuses_each_prepared_graph_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate real shapes compile once and reuse their two graph callables."""
    import torch

    compile_count = 0

    def fake_compile(function: object, **kwargs: object) -> object:
        nonlocal compile_count
        del kwargs
        compile_count += 1
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = _model()
    first = torch_backends_v2.prepare_torch_geometry(
        "geo2rdr", model, shape=(1,), compile_kernel=True, max_iter=3
    )
    second = torch_backends_v2.prepare_torch_geometry(
        "geo2rdr", model, shape=(2,), compile_kernel=True, max_iter=3
    )
    assert compile_count == 4

    first.execute(
        np.array([-60.0], dtype=np.float64),
        np.array([-175.5], dtype=np.float64),
        np.array([10.0], dtype=np.float64),
    )
    assert compile_count == 4
    first.execute(
        np.array([-59.0], dtype=np.float64),
        np.array([-174.5], dtype=np.float64),
        np.array([1000.0], dtype=np.float64),
    )
    assert compile_count == 4

    second.execute(
        np.array([-60.0, -59.0], dtype=np.float64),
        np.array([-175.5, -174.5], dtype=np.float64),
        np.array([10.0, 1000.0], dtype=np.float64),
    )
    assert compile_count == 4
    second.execute(
        np.array([-58.0, -57.0], dtype=np.float64),
        np.array([-173.5, -172.5], dtype=np.float64),
        np.array([2000.0, 3000.0], dtype=np.float64),
    )
    assert compile_count == 4


def test_compile_configuration_is_fullgraph_and_shape_static(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution requests two fixed-shape fullgraph compile callables."""
    import torch

    calls: list[dict[str, object]] = []

    def fake_compile(function: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    prepared = torch_backends_v2.prepare_torch_geometry(
        "geo2rdr",
        _model(),
        shape=(1,),
        compile_kernel=True,
        max_iter=2,
    )
    assert len(calls) == 2

    prepared.execute(
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    )
    assert calls == [
        {
            "mode": "reduce-overhead",
            "fullgraph": True,
            "dynamic": False,
        },
        {
            "mode": "reduce-overhead",
            "fullgraph": True,
            "dynamic": False,
        },
    ]


def test_non_geo2rdr_compile_branch_keeps_single_warmed_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The non-CPU-GEO2RDR compile path retains its existing warm-up contract."""
    import torch

    calls: list[dict[str, object]] = []
    invocations: list[tuple[object, ...]] = []

    def fake_compile(function: object, **kwargs: object) -> object:
        calls.append(kwargs)

        def compiled(*args: object) -> object:
            invocations.append(args)
            return function(*args)  # type: ignore[operator]

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    torch_backends_v2.prepare_torch_geometry(
        "rdr2geo", _model(), shape=(1,), compile_kernel=True
    )

    assert calls == [
        {
            "mode": "reduce-overhead",
            "fullgraph": True,
            "dynamic": False,
        }
    ]
    assert len(invocations) == 1
    assert all(
        torch.equal(value, torch.zeros(1, dtype=torch.float64))
        for value in invocations[0]
    )


def test_cpu_geo2rdr_compile_continues_state_without_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mixed batch advances exactly four persistent compiled states."""
    import torch

    attempts_before_step: list[np.ndarray] = []
    original_step = torch_backends_v2._geo2rdr_step

    def fake_compile(function: object, **kwargs: object) -> object:
        del kwargs
        return function

    def counted_step(
        state: tuple[object, ...], *args: object, **kwargs: object
    ) -> tuple[object, ...]:
        attempts_before_step.append(state[8].detach().cpu().numpy().copy())
        return original_step(state, *args, **kwargs)

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch_backends_v2, "_geo2rdr_step", counted_step)
    values = (
        np.array([-60.0, -60.0], dtype=np.float64),
        np.array([-180.0, -175.5], dtype=np.float64),
        np.zeros(2, dtype=np.float64),
    )
    settings = {
        "shape": (2,),
        "max_iter": 25,
        "extra_iter": 15,
        "range_tol_m": 1.0e-4,
        "doppler_tol_hz": 0.1,
    }
    eager = torch_backends_v2.prepare_torch_geometry(
        "geo2rdr", _model(), compile_kernel=False, **settings
    ).execute(*values)
    prepared = torch_backends_v2.prepare_torch_geometry(
        "geo2rdr", _model(), compile_kernel=True, **settings
    )

    def fail_eager(*unused: object) -> dict[str, object]:
        del unused
        message = "compiled execution must not call the Eager kernel"
        raise AssertionError(message)

    object.__setattr__(prepared, "_kernel", fail_eager)
    attempts_before_step.clear()
    compiled = prepared.execute(*values)

    assert [row.tolist() for row in attempts_before_step] == [
        [0, 0],
        [1, 1],
        [1, 2],
        [1, 3],
    ]
    assert compiled.iterations.tolist() == [1, 4]
    assert compiled.iterations.tolist() == eager.iterations.tolist()
    assert compiled.converged.tolist() == eager.converged.tolist()
    np.testing.assert_allclose(compiled.range_index, eager.range_index)
    np.testing.assert_allclose(compiled.azimuth_index, eager.azimuth_index)
    np.testing.assert_allclose(compiled.residual_range_m, eager.residual_range_m)
    np.testing.assert_allclose(compiled.residual_doppler_hz, eager.residual_doppler_hz)


def test_cpu_geo2rdr_compile_does_not_fallback_for_invalid_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid lanes do not trigger the compiled numerical fallback."""
    import torch

    calls = 0
    original_step = torch_backends_v2._geo2rdr_step

    def fake_compile(function: object, **kwargs: object) -> object:
        del kwargs
        return function

    def counted_step(
        state: tuple[object, ...], *args: object, **kwargs: object
    ) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return original_step(state, *args, **kwargs)

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch_backends_v2, "_geo2rdr_step", counted_step)
    prepared = torch_backends_v2.prepare_torch_geometry(
        "geo2rdr", _model(), shape=(1,), compile_kernel=True, max_iter=40
    )
    calls = 0
    result = prepared.execute(
        np.array([np.nan], dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    )

    assert calls == 1
    assert result.converged.tolist() == [False]
    assert result.iterations.tolist() == [-1]


def test_cpu_geo2rdr_finite_lane_failed_derivative_is_not_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finite lane rejected by Newton is failed, not max-iteration exhausted."""
    import torch

    model = _model()
    orbit = torch_kernels.prepared_orbit_tensors(model, "cpu")
    zero_velocity_orbit = (
        orbit[0],
        orbit[1],
        torch.zeros_like(orbit[2]),
        orbit[3],
    )
    monkeypatch.setattr(
        torch_backends_v2,
        "prepared_orbit_tensors",
        lambda *_args, **_kwargs: zero_velocity_orbit,
    )
    prepared = torch_backends_v2.prepare_torch_geometry(
        "geo2rdr", model, shape=(1,), max_iter=5
    )

    result = prepared.execute(
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
    )

    assert result.iterations.tolist() == [-1]
    assert result.converged.tolist() == [False]
    assert result.max_iter_exhausted.tolist() == [False]


@pytest.mark.parametrize(
    ("budget", "expected_converged"),
    [
        (1, False),
        (2, False),
        (4, True),
    ],
)
def test_cpu_geo2rdr_compile_honors_zero_or_one_remaining_iteration(
    monkeypatch: pytest.MonkeyPatch,
    budget: int,
    expected_converged: bool,
) -> None:
    """The state loop executes exactly the caller's available budget."""
    import torch

    calls = 0
    original_step = torch_backends_v2._geo2rdr_step

    def fake_compile(function: object, **kwargs: object) -> object:
        del kwargs
        return function

    def counted_step(
        state: tuple[object, ...], *args: object, **kwargs: object
    ) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return original_step(state, *args, **kwargs)

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch_backends_v2, "_geo2rdr_step", counted_step)
    prepared = torch_backends_v2.prepare_torch_geometry(
        "geo2rdr", _model(), shape=(1,), compile_kernel=True, max_iter=budget
    )
    calls = 0
    result = prepared.execute(
        np.array([-60.0], dtype=np.float64),
        np.array([-175.5], dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    )

    assert calls == budget
    assert result.converged.tolist() == [expected_converged]
    assert result.iterations.tolist() == [budget]


def test_cpu_geo2rdr_compile_preserves_full_budget_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unresolved valid lane remains exhausted after compiled fallback."""
    import torch

    calls = 0
    original_step = torch_backends_v2._geo2rdr_step

    def fake_compile(function: object, **kwargs: object) -> object:
        del kwargs
        return function

    def counted_step(
        state: tuple[object, ...], *args: object, **kwargs: object
    ) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return original_step(state, *args, **kwargs)

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch_backends_v2, "_geo2rdr_step", counted_step)
    prepared = torch_backends_v2.prepare_torch_geometry(
        "geo2rdr",
        _model(),
        shape=(1,),
        compile_kernel=True,
        max_iter=3,
    )
    calls = 0
    result = prepared.execute(
        np.array([-60.0], dtype=np.float64),
        np.array([-175.5], dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    )

    assert calls == 3
    assert result.converged.tolist() == [False]
    assert result.iterations.tolist() == [3]


def test_rdr2geo_runs_one_tcn_loop_for_all_dem_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raster DEM sampling stays inside one prepared TCN solver invocation."""
    import torch

    model = _model()
    orbit = torch_kernels.prepared_orbit_tensors(model, "cpu")
    state_ids: list[int] = []
    original_once = torch_kernels._rdr2geo_once

    def wrapped_once(*args: object, **kwargs: object) -> dict[str, object]:
        state_ids.append(id(kwargs["geometry_state"]))
        return original_once(*args, **kwargs)

    monkeypatch.setattr(torch_kernels, "_rdr2geo_once", wrapped_once)
    torch_kernels.rdr2geo_kernel(
        torch.zeros(2, dtype=torch.float64),
        torch.full((2,), 1000.0, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        *orbit[:3],
        sensing_offset_s=orbit[3],
        azimuth_interval_s=model.azimuth_time_interval_s,
        starting_range_m=model.starting_slant_range_m,
        range_spacing_m=model.range_spacing_m,
        wavelength_m=model.wavelength_m,
        look_sign=1.0,
        max_iter=1,
        range_tol_m=1.0,
        doppler_tol_hz=0.1,
        dynamic_iterations=False,
        dem_samples=torch.zeros((8, 8), dtype=torch.float64),
        dem_iterations=3,
    )

    assert len(state_ids) == 1


def test_rdr2geo_dem_sampling_respects_max_iter_not_dem_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raster sampling is bounded by the TCN loop rather than DEM budget."""
    import torch

    model = _model()
    orbit = torch_kernels.prepared_orbit_tensors(model, "cpu")
    sample_calls = 0
    original_sample = torch_kernels._sample_dem_six

    def counted_sample(*args: object, **kwargs: object) -> object:
        nonlocal sample_calls
        sample_calls += 1
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(torch_kernels, "_sample_dem_six", counted_sample)
    result = torch_kernels.rdr2geo_kernel(
        torch.zeros(2, dtype=torch.float64),
        torch.full((2,), 1000.0, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        *orbit[:3],
        sensing_offset_s=orbit[3],
        azimuth_interval_s=model.azimuth_time_interval_s,
        starting_range_m=model.starting_slant_range_m,
        range_spacing_m=model.range_spacing_m,
        wavelength_m=model.wavelength_m,
        look_sign=1.0,
        max_iter=3,
        range_tol_m=1.0,
        doppler_tol_hz=0.1,
        dynamic_iterations=False,
        dem_samples=torch.zeros((8, 8), dtype=torch.float64),
        dem_iterations=50,
    )

    assert 0 < sample_calls <= 4
    assert bool(torch.all(result["iterations"] <= 3))


@pytest.mark.parametrize("regular_rows", [True, False])
def test_rdr2geo_variable_dem_compiled_geometry_state_matches_eager(
    regular_rows: bool,
) -> None:
    """Capture both geometry-state branches with a nonconstant DEM surface."""
    import torch

    model = _model()
    orbit = torch_kernels.prepared_orbit_tensors(model, "cpu")
    shape = (2, 4)
    if regular_rows:
        azimuth = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [0.001, 0.001, 0.001, 0.001]],
            dtype=torch.float64,
        )
    else:
        azimuth = torch.tensor(
            [[0.0, 0.001, 0.002, 0.003], [0.004, 0.005, 0.006, 0.007]],
            dtype=torch.float64,
        )
    range_index = torch.arange(4, dtype=torch.float64).expand(shape)
    height_seed = torch.zeros(shape, dtype=torch.float64)
    rows = torch.arange(20, dtype=torch.float64).unsqueeze(-1)
    columns = torch.arange(20, dtype=torch.float64).unsqueeze(0)
    dem_samples = 25.0 + 0.75 * rows + 0.25 * columns + torch.sin(rows + columns)
    kwargs = {
        "sensing_offset_s": orbit[3],
        "azimuth_interval_s": model.azimuth_time_interval_s,
        "starting_range_m": model.starting_slant_range_m,
        "range_spacing_m": model.range_spacing_m,
        "wavelength_m": model.wavelength_m,
        "look_sign": 1.0,
        "max_iter": 4,
        "range_tol_m": 1.0,
        "doppler_tol_hz": 0.1,
        "dynamic_iterations": False,
        "dem_samples": dem_samples,
        "dem_latitude_start_deg": -10.0,
        "dem_longitude_start_deg": -10.0,
        "dem_latitude_spacing_deg": 1.0,
        "dem_longitude_spacing_deg": 1.0,
        "dem_iterations": 50,
        "dem_height_tol_m": 1.0e-3,
    }

    def eager_kernel(a: object, r: object, h: object) -> dict[str, object]:
        """Run the direct eager kernel with captured device inputs."""
        return torch_kernels.rdr2geo_kernel(a, r, h, *orbit[:3], **kwargs)

    eager = eager_kernel(azimuth, range_index, height_seed)
    compiled = torch.compile(
        eager_kernel,
        backend="aot_eager",
        fullgraph=True,
        dynamic=False,
    )
    result = compiled(azimuth, range_index, height_seed)

    assert bool(torch.all(torch.isfinite(dem_samples)))
    for name in (
        "latitude_deg",
        "longitude_deg",
        "height_m",
        "range_index",
        "azimuth_index",
        "residual_range_m",
        "residual_doppler_hz",
    ):
        torch.testing.assert_close(result[name], eager[name], atol=1.0e-8, rtol=1.0e-8)
    assert torch.equal(result["converged"], eager["converged"])
    assert torch.equal(result["invalid"], eager["invalid"])


def test_rdr2geo_publishes_device_ecef_from_authoritative_tcn_point() -> None:
    """Rdr2Geo exposes ECEF components without a host LLH round trip."""
    import torch

    model = _model()
    orbit = torch_kernels.prepared_orbit_tensors(model, "cpu")
    result = torch_kernels.rdr2geo_kernel(
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        *orbit[:3],
        sensing_offset_s=orbit[3],
        azimuth_interval_s=model.azimuth_time_interval_s,
        starting_range_m=model.starting_slant_range_m,
        range_spacing_m=model.range_spacing_m,
        wavelength_m=model.wavelength_m,
        look_sign=1.0,
        max_iter=4,
        range_tol_m=1.0,
        doppler_tol_hz=0.1,
        dynamic_iterations=False,
        dem_height_m=0.0,
    )

    ecef = torch.stack(
        tuple(result[name] for name in ("ecef_x_m", "ecef_y_m", "ecef_z_m")),
        dim=-1,
    )
    ecef_latitude, ecef_longitude, _ = torch_kernels._ecef_to_llh(ecef)
    torch.testing.assert_close(
        ecef_latitude,
        result["latitude_deg"],
        atol=1.0e-12,
        rtol=1.0e-12,
        equal_nan=True,
    )
    torch.testing.assert_close(
        ecef_longitude,
        result["longitude_deg"],
        atol=1.0e-12,
        rtol=1.0e-12,
        equal_nan=True,
    )
    for name in ("ecef_x_m", "ecef_y_m", "ecef_z_m"):
        assert name in result
        assert result[name].dtype == torch.float64
    invalid = result["invalid"]
    for name in ("ecef_x_m", "ecef_y_m", "ecef_z_m"):
        assert bool(torch.all(torch.isnan(result[name][invalid])))


def test_prepared_rdr2geo_ecef_transfer_skips_non_coordinate_fields() -> None:
    """Prepared adapters expose an ECEF-only host transfer for Rdr2Geo."""
    import torch

    model = _model()
    prepared = torch_backends_v2.prepare_torch_geometry(
        "rdr2geo", model, shape=(2,), max_iter=4
    )
    inputs = (
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
    )
    ecef = prepared.execute_ecef(*inputs)
    raw = prepared._kernel(*inputs)
    for index, values in enumerate(ecef):
        expected = raw[("ecef_x_m", "ecef_y_m", "ecef_z_m")[index]]
        np.testing.assert_allclose(values, expected.numpy(), equal_nan=True)


@pytest.mark.parametrize("dynamic_iterations", [False, True])
def test_rdr2geo_promotes_final_boundary_residual_to_converged(
    dynamic_iterations: bool,
) -> None:
    """A final TCN residual can converge on the last allowed attempt."""
    import torch

    model = _model()
    orbit = torch_kernels.prepared_orbit_tensors(model, "cpu")
    result = torch_kernels.rdr2geo_kernel(
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        *orbit[:3],
        sensing_offset_s=orbit[3],
        azimuth_interval_s=model.azimuth_time_interval_s,
        starting_range_m=model.starting_slant_range_m,
        range_spacing_m=model.range_spacing_m,
        wavelength_m=model.wavelength_m,
        look_sign=1.0,
        max_iter=1,
        range_tol_m=1.0e-8,
        doppler_tol_hz=0.1,
        dynamic_iterations=dynamic_iterations,
        dem_samples=torch.zeros((20, 20), dtype=torch.float64),
        dem_latitude_start_deg=-10.0,
        dem_longitude_start_deg=-10.0,
        dem_latitude_spacing_deg=1.0,
        dem_longitude_spacing_deg=1.0,
        dem_height_tol_m=1.0,
    )

    assert bool(result["invalid"].item()) is False
    assert bool(result["converged"].item()) is True
    assert int(result["iterations"].item()) == 1
    assert abs(float(result["residual_range_m"].item())) < 1.0e-8


@pytest.mark.parametrize("dynamic_iterations", [False, True])
def test_rdr2geo_final_dem_recheck_failure_keeps_finite_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    dynamic_iterations: bool,
) -> None:
    """Eager and fixed-shape paths report DEM failure without erasing TCN output."""
    import torch

    model = _model()
    orbit = torch_kernels.prepared_orbit_tensors(model, "cpu")
    original_sample = torch_kernels._sample_dem_six
    calls = 0

    def sample_once_then_fail(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        sampled = original_sample(*args, **kwargs)
        if calls == 1:
            return sampled
        return torch.full_like(sampled, torch.nan)

    monkeypatch.setattr(torch_kernels, "_sample_dem_six", sample_once_then_fail)
    result = torch_kernels.rdr2geo_kernel(
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        *orbit[:3],
        sensing_offset_s=orbit[3],
        azimuth_interval_s=model.azimuth_time_interval_s,
        starting_range_m=model.starting_slant_range_m,
        range_spacing_m=model.range_spacing_m,
        wavelength_m=model.wavelength_m,
        look_sign=1.0,
        max_iter=1,
        range_tol_m=1.0,
        doppler_tol_hz=0.1,
        dynamic_iterations=dynamic_iterations,
        dem_samples=torch.zeros((20, 20), dtype=torch.float64),
        dem_latitude_start_deg=-10.0,
        dem_longitude_start_deg=-10.0,
        dem_latitude_spacing_deg=1.0,
        dem_longitude_spacing_deg=1.0,
        dem_height_tol_m=1.0,
    )

    assert calls == 2
    assert bool(result["invalid"].item()) is False
    assert bool(result["converged"].item()) is False
    assert int(result["iterations"].item()) == 1
    for name in ("latitude_deg", "longitude_deg", "height_m"):
        assert bool(torch.isfinite(result[name]).item())


def test_rdr2geo_eager_raster_uses_one_tcn_dem_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eager raster execution does not repeat the TCN for DEM iterations."""
    import torch

    model = _model()
    orbit = torch_kernels.prepared_orbit_tensors(model, "cpu")
    calls = 0
    original_once = torch_kernels._rdr2geo_once

    def wrapped_once(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return original_once(*args, **kwargs)

    monkeypatch.setattr(torch_kernels, "_rdr2geo_once", wrapped_once)
    torch_kernels.rdr2geo_kernel(
        torch.zeros(2, dtype=torch.float64),
        torch.full((2,), 1000.0, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        *orbit[:3],
        sensing_offset_s=orbit[3],
        azimuth_interval_s=model.azimuth_time_interval_s,
        starting_range_m=model.starting_slant_range_m,
        range_spacing_m=model.range_spacing_m,
        wavelength_m=model.wavelength_m,
        look_sign=1.0,
        max_iter=3,
        range_tol_m=1.0,
        doppler_tol_hz=0.1,
        dynamic_iterations=True,
        dem_samples=torch.zeros((8, 8), dtype=torch.float64),
        dem_iterations=50,
    )

    assert calls == 1


def test_rdr2geo_eager_gathers_tail_lanes_between_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eager raster iterations do not keep solved lanes in the work batch."""
    import torch

    sample_shapes: list[tuple[int, ...]] = []

    def fake_ecef_to_llh(ecef: object) -> tuple[object, object, object]:
        value = torch.as_tensor(ecef)[..., 0]
        return torch.zeros_like(value), torch.zeros_like(value), torch.zeros_like(value)

    def fake_llh_to_ecef(latitude: object, longitude: object, height: object) -> object:
        del latitude, longitude
        value = torch.as_tensor(height)
        return torch.stack(
            (value, torch.zeros_like(value), torch.zeros_like(value)), -1
        )

    def fake_sample(*args: object) -> object:
        latitude = torch.as_tensor(args[1])
        sample_shapes.append(tuple(latitude.shape))
        if latitude.numel() == 2:
            return torch.tensor([0.0, 10.0], dtype=latitude.dtype)
        return torch.zeros_like(latitude)

    monkeypatch.setattr(torch_kernels, "_ecef_to_llh", fake_ecef_to_llh)
    monkeypatch.setattr(torch_kernels, "_llh_to_ecef", fake_llh_to_ecef)
    monkeypatch.setattr(torch_kernels, "_sample_dem_six", fake_sample)

    state = (
        torch.full((2,), 100.0),
        torch.tensor([[0.0, 0.0, 100.0]]).expand(2, -1).clone(),
        torch.tensor([[0.0, 1.0, 0.0]]).expand(2, -1).clone(),
        torch.full((2,), 100.0),
        torch.tensor([[0.0, 0.0, -1.0]]).expand(2, -1).clone(),
        torch.tensor([[1.0, 0.0, 0.0]]).expand(2, -1).clone(),
        torch.tensor([[0.0, 1.0, 0.0]]).expand(2, -1).clone(),
        torch.zeros(2),
        torch.ones(2),
        torch.zeros(2),
        torch.zeros(2),
        torch.ones(2, dtype=torch.bool),
    )
    result = torch_kernels._rdr2geo_once(
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2),
        None,
        None,
        None,
        sensing_offset_s=0.0,
        azimuth_interval_s=1.0,
        starting_range_m=100.0,
        range_spacing_m=1.0,
        wavelength_m=1.0,
        look_sign=1.0,
        max_iter=4,
        range_tol_m=1.0e-6,
        doppler_tol_hz=1.0,
        dynamic_iterations=True,
        dem_samples=torch.zeros((8, 8)),
        geometry_state=state,
    )

    assert sample_shapes == [(2,), (1,), (1,), (2,)]
    assert result["iterations"].tolist() == [1, 3]
    assert result["converged"].tolist() == [True, False]


def test_rdr2geo_exhausted_lane_publishes_one_final_tcn_triple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exhausted valid lanes publish latitude, longitude, and height together."""
    import torch

    llh_values = iter(
        ((1.0, 2.0, 3.0), (3.0, 4.0, 5.0), (6.0, 7.0, 8.0), (9.0, 10.0, 11.0))
    )

    def fake_ecef_to_llh(ecef: object) -> tuple[object, object, object]:
        shape = torch.as_tensor(ecef)[..., 0]
        latitude, longitude, height = next(llh_values)
        return (
            torch.full_like(shape, latitude),
            torch.full_like(shape, longitude),
            torch.full_like(shape, height),
        )

    def fake_llh_to_ecef(latitude: object, longitude: object, height: object) -> object:
        value = torch.as_tensor(latitude)
        del longitude, height
        return torch.stack(
            (
                torch.zeros_like(value),
                torch.zeros_like(value),
                torch.full_like(value, 10.0),
            ),
            dim=-1,
        )

    def fake_sample(*args: object) -> object:
        latitude = torch.as_tensor(args[1])
        return torch.full_like(latitude, 5.0)

    monkeypatch.setattr(torch_kernels, "_ecef_to_llh", fake_ecef_to_llh)
    monkeypatch.setattr(torch_kernels, "_llh_to_ecef", fake_llh_to_ecef)
    monkeypatch.setattr(torch_kernels, "_sample_dem_six", fake_sample)

    state = (
        torch.tensor([100.0]),
        torch.tensor([[0.0, 0.0, 100.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([100.0]),
        torch.tensor([[0.0, 0.0, -1.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([True]),
    )
    result = torch_kernels._rdr2geo_once(
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        None,
        None,
        None,
        sensing_offset_s=0.0,
        azimuth_interval_s=1.0,
        starting_range_m=100.0,
        range_spacing_m=1.0,
        wavelength_m=1.0,
        look_sign=1.0,
        max_iter=1,
        extra_iter=1,
        range_tol_m=1.0e-12,
        doppler_tol_hz=1.0,
        dynamic_iterations=False,
        dem_samples=torch.zeros((8, 8)),
        geometry_state=state,
    )

    assert not bool(result["converged"][0])
    assert int(result["iterations"][0]) == 2
    torch.testing.assert_close(result["latitude_deg"], torch.tensor([9.0]))
    torch.testing.assert_close(result["longitude_deg"], torch.tensor([10.0]))
    torch.testing.assert_close(result["height_m"], torch.tensor([11.0]))


def test_rdr2geo_extra_iterations_apply_canonical_ecef_damping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extra iterations revisit the previous LLH through ECEF averaging."""
    import torch

    llh_to_ecef_heights: list[torch.Tensor] = []
    dem_heights = iter((10.0, 20.0, 30.0, 30.0))

    def fake_llh_to_ecef(latitude: object, longitude: object, height: object) -> object:
        del latitude, longitude
        value = torch.as_tensor(height)
        llh_to_ecef_heights.append(value.detach().clone())
        return torch.stack(
            (value + 100.0, torch.zeros_like(value), torch.zeros_like(value)),
            dim=-1,
        )

    def fake_ecef_to_llh(ecef: object) -> tuple[object, object, object]:
        value = torch.as_tensor(ecef)[..., 0]
        return torch.zeros_like(value), torch.zeros_like(value), torch.zeros_like(value)

    def fake_sample(*args: object) -> object:
        latitude = torch.as_tensor(args[1])
        return torch.full_like(latitude, next(dem_heights))

    monkeypatch.setattr(torch_kernels, "_llh_to_ecef", fake_llh_to_ecef)
    monkeypatch.setattr(torch_kernels, "_ecef_to_llh", fake_ecef_to_llh)
    monkeypatch.setattr(torch_kernels, "_sample_dem_six", fake_sample)

    state = (
        torch.tensor([100.0]),
        torch.tensor([[0.0, 0.0, 100.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([100.0]),
        torch.tensor([[0.0, 0.0, -1.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([True]),
    )
    result = torch_kernels._rdr2geo_once(
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        None,
        None,
        None,
        sensing_offset_s=0.0,
        azimuth_interval_s=1.0,
        starting_range_m=100.0,
        range_spacing_m=1.0,
        wavelength_m=1.0,
        look_sign=1.0,
        max_iter=1,
        extra_iter=2,
        range_tol_m=1.0e-12,
        doppler_tol_hz=1.0,
        dynamic_iterations=False,
        dem_samples=torch.zeros((8, 8)),
        geometry_state=state,
    )

    assert len(llh_to_ecef_heights) == 5
    torch.testing.assert_close(llh_to_ecef_heights[2], torch.tensor([10.0]))
    torch.testing.assert_close(result["height_m"], torch.zeros(1))


def test_rdr2geo_first_extra_iteration_applies_damping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first attempt after the primary budget uses ECEF damping."""
    import torch

    ecef_x_values: list[torch.Tensor] = []
    dem_heights = iter((10.0, 20.0, 20.0))

    def fake_llh_to_ecef(latitude: object, longitude: object, height: object) -> object:
        del latitude, longitude
        value = torch.as_tensor(height)
        return torch.stack(
            (value + 100.0, torch.zeros_like(value), torch.zeros_like(value)),
            dim=-1,
        )

    def fake_ecef_to_llh(ecef: object) -> tuple[object, object, object]:
        value = torch.as_tensor(ecef)[..., 0]
        ecef_x_values.append(value.detach().clone())
        return torch.zeros_like(value), torch.zeros_like(value), torch.zeros_like(value)

    def fake_sample(*args: object) -> object:
        latitude = torch.as_tensor(args[1])
        return torch.full_like(latitude, next(dem_heights))

    monkeypatch.setattr(torch_kernels, "_llh_to_ecef", fake_llh_to_ecef)
    monkeypatch.setattr(torch_kernels, "_ecef_to_llh", fake_ecef_to_llh)
    monkeypatch.setattr(torch_kernels, "_sample_dem_six", fake_sample)

    state = (
        torch.tensor([100.0]),
        torch.tensor([[0.0, 0.0, 100.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([100.0]),
        torch.tensor([[0.0, 0.0, -1.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([True]),
    )
    torch_kernels._rdr2geo_once(
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        None,
        None,
        None,
        sensing_offset_s=0.0,
        azimuth_interval_s=1.0,
        starting_range_m=100.0,
        range_spacing_m=1.0,
        wavelength_m=1.0,
        look_sign=1.0,
        max_iter=1,
        extra_iter=1,
        range_tol_m=1.0e-12,
        doppler_tol_hz=1.0,
        dynamic_iterations=False,
        dem_samples=torch.zeros((8, 8)),
        geometry_state=state,
    )

    assert any(torch.allclose(value, torch.tensor([115.0])) for value in ecef_x_values)


def test_prepared_rdr2geo_passes_primary_and_extra_budgets_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prepared adapter preserves the split primary and extra budgets."""
    calls: list[tuple[int, int]] = []
    original_kernel = torch_backends_v2.rdr2geo_kernel

    def wrapped_kernel(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((int(kwargs["max_iter"]), int(kwargs["extra_iter"])))
        return original_kernel(*args, **kwargs)

    monkeypatch.setattr(torch_backends_v2, "rdr2geo_kernel", wrapped_kernel)
    prepared = torch_backends_v2.prepare_torch_geometry(
        "rdr2geo",
        _model(),
        shape=(1,),
        max_iter=2,
        extra_iter=3,
    )
    prepared.execute(
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    )

    assert calls == [(2, 3)]


def test_rdr2geo_invalid_dem_sample_publishes_invalid_sentinels() -> None:
    """Raster samples outside the six-point stencil invalidate the lane."""
    import torch

    model = _model()
    orbit = torch_kernels.prepared_orbit_tensors(model, "cpu")
    result = torch_kernels.rdr2geo_kernel(
        torch.zeros(1, dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        *orbit[:3],
        sensing_offset_s=orbit[3],
        azimuth_interval_s=model.azimuth_time_interval_s,
        starting_range_m=model.starting_slant_range_m,
        range_spacing_m=model.range_spacing_m,
        wavelength_m=model.wavelength_m,
        look_sign=1.0,
        max_iter=2,
        range_tol_m=1.0,
        doppler_tol_hz=0.1,
        dynamic_iterations=False,
        dem_samples=torch.zeros((8, 8), dtype=torch.float64),
        dem_latitude_start_deg=90.0,
        dem_longitude_start_deg=90.0,
    )

    assert bool(result["invalid"].item())
    assert not bool(result["converged"].item())
    assert int(result["iterations"].item()) == -1
    for name in (
        "latitude_deg",
        "longitude_deg",
        "height_m",
        "range_index",
        "azimuth_index",
        "residual_range_m",
        "residual_doppler_hz",
    ):
        assert bool(torch.isnan(result[name]).item())


def test_rdr2geo_failed_lane_publishes_invalid_sentinels() -> None:
    """A finite lane rejected by the TCN bound is not published as valid."""
    import torch

    model = _model()
    prepared = torch_backends_v2.prepare_torch_geometry(
        "rdr2geo",
        model,
        shape=(1,),
        max_iter=2,
        dem=None,
    )
    result = prepared.execute(
        torch.zeros(1, dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
        torch.full((1,), -1.0e9, dtype=torch.float64),
    ).transform

    assert not bool(result.converged.item())
    assert not bool(result.max_iter_exhausted.item())
    assert int(result.iterations.item()) == -1
    for name in (
        "latitude_deg",
        "longitude_deg",
        "height_m",
        "range_index",
        "azimuth_index",
        "decision_residual",
        "final_residual",
        "tolerance",
        "residual_range_m",
        "residual_doppler_hz",
    ):
        assert bool(torch.isnan(torch.as_tensor(getattr(result, name))).item())


def test_rdr2geo_regular_rows_reuse_one_orbit_context_per_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A regular 2-D radar grid evaluates orbit state once per row."""
    import torch

    model = _model()
    orbit = torch_kernels.prepared_orbit_tensors(model, "cpu")
    azimuth = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    range_index = torch.full_like(azimuth, 1000.0)
    calls: list[tuple[int, ...]] = []
    original_state = torch_kernels._orbit_state

    def counted_state(times: object, *args: object) -> tuple[object, ...]:
        calls.append(tuple(torch.as_tensor(times).shape))
        return original_state(times, *args)

    monkeypatch.setattr(torch_kernels, "_orbit_state", counted_state)
    state = torch_kernels._rdr2geo_geometry_state(
        azimuth,
        range_index,
        *orbit[:3],
        sensing_offset_s=orbit[3],
        azimuth_interval_s=model.azimuth_time_interval_s,
        starting_range_m=model.starting_slant_range_m,
        range_spacing_m=model.range_spacing_m,
    )

    assert calls == [(2, 1)]
    assert state[1].shape == (2, 3, 3)


def test_rdr2geo_irregular_rows_fall_back_to_point_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An irregular 2-D grid keeps point-wise orbit evaluation semantics."""
    import torch

    model = _model()
    orbit = torch_kernels.prepared_orbit_tensors(model, "cpu")
    azimuth = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    range_index = torch.full_like(azimuth, 1000.0)
    calls: list[tuple[int, ...]] = []
    original_state = torch_kernels._orbit_state

    def counted_state(times: object, *args: object) -> tuple[object, ...]:
        calls.append(tuple(torch.as_tensor(times).shape))
        return original_state(times, *args)

    monkeypatch.setattr(torch_kernels, "_orbit_state", counted_state)
    state = torch_kernels._rdr2geo_geometry_state(
        azimuth,
        range_index,
        *orbit[:3],
        sensing_offset_s=orbit[3],
        azimuth_interval_s=model.azimuth_time_interval_s,
        starting_range_m=model.starting_slant_range_m,
        range_spacing_m=model.range_spacing_m,
    )

    assert calls == [(2, 3)]
    assert state[1].shape == (2, 3, 3)


def test_rdr2geo_compiling_uses_point_context_without_cond(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compiled geometry state matches point context without graph control flow."""
    import torch

    model = _model()
    orbit = torch_kernels.prepared_orbit_tensors(model, "cpu")
    azimuth = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    range_index = torch.full_like(azimuth, 1000.0)
    kwargs = {
        "sensing_offset_s": orbit[3],
        "azimuth_interval_s": model.azimuth_time_interval_s,
        "starting_range_m": model.starting_slant_range_m,
        "range_spacing_m": model.range_spacing_m,
    }
    expected = torch_kernels._rdr2geo_geometry_state_for_context(
        azimuth,
        range_index,
        *orbit[:3],
        context_azimuth=azimuth,
        **kwargs,
    )

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    def fail_cond(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError

    monkeypatch.setattr(torch, "cond", fail_cond)
    actual = torch_kernels._rdr2geo_geometry_state(
        azimuth,
        range_index,
        *orbit[:3],
        **kwargs,
    )

    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert actual_value.shape == expected_value.shape
        torch.testing.assert_close(actual_value, expected_value)


def test_rdr2geo_one_dimensional_input_keeps_point_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One-dimensional inputs retain the existing point-wise path."""
    import torch

    model = _model()
    orbit = torch_kernels.prepared_orbit_tensors(model, "cpu")
    azimuth = torch.zeros(3)
    range_index = torch.full_like(azimuth, 1000.0)
    calls: list[tuple[int, ...]] = []
    original_state = torch_kernels._orbit_state

    def counted_state(times: object, *args: object) -> tuple[object, ...]:
        calls.append(tuple(torch.as_tensor(times).shape))
        return original_state(times, *args)

    monkeypatch.setattr(torch_kernels, "_orbit_state", counted_state)
    torch_kernels._rdr2geo_geometry_state(
        azimuth,
        range_index,
        *orbit[:3],
        sensing_offset_s=orbit[3],
        azimuth_interval_s=model.azimuth_time_interval_s,
        starting_range_m=model.starting_slant_range_m,
        range_spacing_m=model.range_spacing_m,
    )

    assert calls == [(3,)]
