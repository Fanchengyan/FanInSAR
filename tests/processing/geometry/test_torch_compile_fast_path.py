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


def test_rdr2geo_reuses_tcn_state_across_dem_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEM iterations receive one shared device-resident TCN state."""
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

    assert len(state_ids) == 4
    assert len(set(state_ids)) == 1


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
