"""Tests for the prepared fixed-shape Torch compile fast path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from faninsar.processing.geometry import torch_backends_v2, torch_kernels

from .test_public_geometry_v2 import _model

if TYPE_CHECKING:
    import pytest


def test_compile_configuration_is_fullgraph_and_shape_static(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparation requests the old fixed-shape fullgraph compile contract."""
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

    assert calls == [
        {
            "mode": "reduce-overhead",
            "fullgraph": True,
            "dynamic": False,
        }
    ]
    prepared.execute(
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    )
    assert len(calls) == 1


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

    assert 0 < sample_calls <= 3
    assert bool(torch.all(result["iterations"] <= 3))


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
    dem_heights = iter((10.0, 20.0, 30.0))

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
    dem_heights = iter((10.0, 20.0))

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
