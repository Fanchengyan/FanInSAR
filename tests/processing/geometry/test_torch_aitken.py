"""Regression tests for the guarded Torch DEM Aitken step."""

from __future__ import annotations

import pytest

from faninsar.processing.geometry import torch_kernels


def _controlled_geometry_state() -> tuple[object, ...]:
    """Return a minimal valid TCN state for fixed-point loop tests."""
    import torch

    return (
        torch.tensor([100.0, 100.0]),
        torch.tensor([[0.0, 0.0, 100.0]]).expand(2, -1).clone(),
        torch.tensor([[0.0, 1.0, 0.0]]).expand(2, -1).clone(),
        torch.tensor([100.0, 100.0]),
        torch.tensor([[0.0, 0.0, -1.0]]).expand(2, -1).clone(),
        torch.tensor([[1.0, 0.0, 0.0]]).expand(2, -1).clone(),
        torch.tensor([[0.0, 1.0, 0.0]]).expand(2, -1).clone(),
        torch.zeros(2),
        torch.ones(2),
        torch.zeros(2),
        torch.zeros(2),
        torch.ones(2, dtype=torch.bool),
    )


def test_guarded_aitken_accepts_only_a_safe_dem_update() -> None:
    """A contracting local slope may accelerate inside the current interval."""
    import torch

    from faninsar.processing.geometry.torch_kernels import _guarded_aitken_height

    candidate, enabled = _guarded_aitken_height(
        previous_previous=torch.tensor([100.0]),
        previous=torch.tensor([90.0]),
        current=torch.tensor([91.0]),
        dem_min=torch.tensor(0.0),
        dem_max=torch.tensor(200.0),
    )

    torch.testing.assert_close(candidate, torch.tensor([90.9090909091]))
    assert enabled.tolist() == [True]


def test_guarded_aitken_accepts_a_safe_positive_contraction() -> None:
    """A small positive contraction may be accelerated within the 1 mm scale."""
    import torch

    candidate, enabled = torch_kernels._guarded_aitken_height(
        previous_previous=torch.tensor([80.0]),
        previous=torch.tensor([90.0]),
        current=torch.tensor([90.0005]),
        dem_min=torch.tensor(0.0),
        dem_max=torch.tensor(200.0),
    )

    torch.testing.assert_close(candidate, torch.tensor([90.0005000250]))
    assert enabled.tolist() == [True]


def test_guarded_aitken_accepts_the_widened_negative_slope_guard() -> None:
    """A slope of -0.9 is accepted inside the qualified (-0.95, 0.95) guard."""
    import torch

    candidate, enabled = torch_kernels._guarded_aitken_height(
        previous_previous=torch.tensor([100.0]),
        previous=torch.tensor([90.0]),
        current=torch.tensor([99.0]),
        dem_min=torch.tensor(0.0),
        dem_max=torch.tensor(200.0),
    )

    torch.testing.assert_close(candidate, torch.tensor([94.7368421053]))
    assert enabled.tolist() == [True]


@pytest.mark.parametrize(
    ("previous_previous", "previous", "current", "dem_min", "dem_max"),
    [
        # A positive proposal outside the DEM range.
        (100.0, 90.0, 91.0, 0.0, 90.5),
        # Local slope is outside the qualified (-0.95, 0.95) interval.
        (100.0, 90.0, 99.9, 0.0, 200.0),
        # A nearly-zero Aitken denominator.
        (100.0, 90.0, 80.0000001, 0.0, 200.0),
        # Candidate is outside the DEM's materialized height range.
        (100.0, 90.0, 95.0, 94.0, 200.0),
        # Non-finite denominator/input.
        (100.0, 90.0, float("nan"), 0.0, 200.0),
    ],
)
def test_guarded_aitken_rejects_unsafe_updates(
    previous_previous: float,
    previous: float,
    current: float,
    dem_min: float,
    dem_max: float,
) -> None:
    """Unsafe slope, denominator, range, or finite checks disable acceleration."""
    import torch

    from faninsar.processing.geometry.torch_kernels import _guarded_aitken_height

    candidate, enabled = _guarded_aitken_height(
        previous_previous=torch.tensor([previous_previous]),
        previous=torch.tensor([previous]),
        current=torch.tensor([current]),
        dem_min=torch.tensor(dem_min),
        dem_max=torch.tensor(dem_max),
    )

    assert enabled.tolist() == [False]
    torch.testing.assert_close(candidate, torch.tensor([current]), equal_nan=True)


@pytest.mark.parametrize("dynamic_iterations", [False, True])
def test_guarded_aitken_controlled_loop_matches_eager_and_fixed_shape(
    monkeypatch: pytest.MonkeyPatch, dynamic_iterations: bool
) -> None:
    """A raster loop accepts a proposal without changing path semantics."""
    import torch

    samples = iter((100.0, 90.0, 91.0))
    accepted: list[bool] = []
    original_helper = torch_kernels._guarded_aitken_height

    def wrapped_helper(*args: object) -> tuple[object, object]:
        candidate, enabled = original_helper(*args)
        accepted.extend(torch.as_tensor(enabled).tolist())
        return candidate, enabled

    def fake_sample(*args: object) -> object:
        latitude = torch.as_tensor(args[1])
        try:
            value = next(samples)
        except StopIteration:
            value = 0.0
        return torch.full_like(latitude, value)

    def fake_ecef_to_llh(ecef: object) -> tuple[object, object, object]:
        shape = torch.as_tensor(ecef)[..., 0]
        return torch.zeros_like(shape), torch.zeros_like(shape), torch.zeros_like(shape)

    def fake_llh_to_ecef(latitude: object, longitude: object, height: object) -> object:
        del latitude, longitude
        value = torch.as_tensor(height)
        return torch.stack(
            (value, torch.zeros_like(value), torch.zeros_like(value)), -1
        )

    monkeypatch.setattr(torch_kernels, "_guarded_aitken_height", wrapped_helper)
    monkeypatch.setattr(torch_kernels, "_sample_dem_six", fake_sample)
    monkeypatch.setattr(torch_kernels, "_ecef_to_llh", fake_ecef_to_llh)
    monkeypatch.setattr(torch_kernels, "_llh_to_ecef", fake_llh_to_ecef)
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
        max_iter=3,
        range_tol_m=1.0e-12,
        doppler_tol_hz=1.0,
        dynamic_iterations=dynamic_iterations,
        dem_samples=torch.arange(64.0).reshape(8, 8) * 4.0,
        geometry_state=_controlled_geometry_state(),
    )

    assert any(accepted)
    assert result["height_m"].shape == (2,)


@pytest.mark.parametrize("dynamic_iterations", [False, True])
def test_dem_damping_restarts_aitken_history(
    monkeypatch: pytest.MonkeyPatch, dynamic_iterations: bool
) -> None:
    """A post-budget damping step rebuilds history before Aitken resumes."""
    import torch

    helper_history: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    dem_heights = iter((10.0, 20.0, 30.0, 40.0, 50.0))

    def fake_aitken(
        previous_previous: object,
        previous: object,
        current: object,
        dem_min: object,
        dem_max: object,
    ) -> tuple[object, object]:
        del dem_min, dem_max
        helper_history.append(
            tuple(
                torch.as_tensor(value).detach().clone()
                for value in (previous_previous, previous, current)
            )
        )
        current_tensor = torch.as_tensor(current)
        return current_tensor, torch.ones_like(current_tensor, dtype=torch.bool)

    def fake_sample(*args: object) -> object:
        latitude = torch.as_tensor(args[1])
        try:
            value = next(dem_heights)
        except StopIteration:
            value = 0.0
        return torch.full_like(latitude, value)

    def fake_ecef_to_llh(ecef: object) -> tuple[object, object, object]:
        value = torch.as_tensor(ecef)[..., 0]
        return torch.zeros_like(value), torch.zeros_like(value), value

    def fake_llh_to_ecef(latitude: object, longitude: object, height: object) -> object:
        del latitude, longitude
        value = torch.as_tensor(height)
        return torch.stack(
            (value, torch.zeros_like(value), torch.zeros_like(value)), -1
        )

    monkeypatch.setattr(torch_kernels, "_guarded_aitken_height", fake_aitken)
    monkeypatch.setattr(torch_kernels, "_sample_dem_six", fake_sample)
    monkeypatch.setattr(torch_kernels, "_ecef_to_llh", fake_ecef_to_llh)
    monkeypatch.setattr(torch_kernels, "_llh_to_ecef", fake_llh_to_ecef)
    state = tuple(value[:1] for value in _controlled_geometry_state())

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
        max_iter=2,
        extra_iter=3,
        range_tol_m=1.0e-12,
        doppler_tol_hz=1.0,
        dynamic_iterations=dynamic_iterations,
        dem_samples=torch.zeros((8, 8)),
        geometry_state=state,
    )

    assert len(helper_history) == 3
    torch.testing.assert_close(
        torch.stack(helper_history[0]), torch.tensor([[10.0], [20.0], [30.0]])
    )
    assert torch.isnan(helper_history[1][0]).all()
    torch.testing.assert_close(
        torch.stack(helper_history[1][1:]), torch.tensor([[25.0], [40.0]])
    )
    torch.testing.assert_close(
        torch.stack(helper_history[2]), torch.tensor([[25.0], [32.5], [50.0]])
    )


def test_constant_dem_does_not_call_aitken_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constant-height execution retains the no-raster fast path."""
    import torch

    monkeypatch.setattr(
        torch_kernels,
        "_guarded_aitken_height",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected Aitken")),
    )
    state = _controlled_geometry_state()
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
        max_iter=1,
        range_tol_m=1.0,
        doppler_tol_hz=1.0,
        dynamic_iterations=False,
        dem_height_m=0.0,
        geometry_state=state,
    )
    assert result["height_m"].shape == (2,)


def test_cuda_raster_eager_uses_fixed_shape_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA raster Eager avoids host-synchronizing active-lane compaction."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    state = tuple(
        value.to("cuda") if isinstance(value, torch.Tensor) else value
        for value in _controlled_geometry_state()
    )

    def unexpected_active_path(*_args: object, **_kwargs: object) -> object:
        message = "CUDA Eager must use the fixed-shape raster loop"
        raise AssertionError(message)

    monkeypatch.setattr(
        torch_kernels, "_rdr2geo_once_eager_active", unexpected_active_path
    )
    result = torch_kernels._rdr2geo_once(
        torch.zeros(2, device="cuda"),
        torch.zeros(2, device="cuda"),
        torch.zeros(2, device="cuda"),
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
        range_tol_m=1.0,
        doppler_tol_hz=1.0,
        dynamic_iterations=True,
        dem_samples=torch.arange(64.0, device="cuda").reshape(8, 8),
        geometry_state=state,
    )
    assert result["height_m"].device.type == "cuda"


def test_dem_sampler_flat_gather_matches_explicit_spline_window() -> None:
    """The flat gather preserves the exact six-point spline values."""
    import torch

    dem = torch.arange(16 * 17, dtype=torch.float64).reshape(16, 17)
    latitude = torch.tensor([2.25, 7.5, 10.75], dtype=torch.float64)
    longitude = torch.tensor([3.5, 8.125, 11.875], dtype=torch.float64)
    actual = torch_kernels._sample_dem_six(
        dem,
        latitude,
        longitude,
        0.0,
        0.0,
        1.0,
        1.0,
    )

    expected_values: list[torch.Tensor] = []
    for row_value, column_value in zip(latitude, longitude, strict=True):
        row_base = int(torch.floor(row_value).item())
        column_base = int(torch.floor(column_value).item())
        row_fraction = row_value - row_base
        column_fraction = column_value - column_base
        row_values = [
            torch_kernels._natural_spline_six(
                dem[row_base + offset, column_base - 1 : column_base + 5],
                column_fraction,
            )
            for offset in range(-1, 5)
        ]
        expected_values.append(
            torch_kernels._natural_spline_six(torch.stack(row_values), row_fraction)
        )
    expected = torch.stack(expected_values)

    torch.testing.assert_close(actual, expected)


def test_dem_sampler_accepts_only_the_interior_six_point_base() -> None:
    """A 6x6 tile has one valid base for the ``-1..4`` stencil."""
    import torch

    base = 1
    dem = torch.arange(36, dtype=torch.float64).reshape(6, 6)
    latitude = torch.tensor([base + 0.25], dtype=torch.float64)
    longitude = torch.tensor([base + 0.5], dtype=torch.float64)
    actual = torch_kernels._sample_dem_six(
        dem,
        latitude,
        longitude,
        0.0,
        0.0,
        1.0,
        1.0,
    )

    offsets = torch.arange(-1, 5)
    rows = base + offsets
    columns = base + offsets
    window = dem[rows[:, None], columns[None, :]]
    along_columns = torch_kernels._natural_spline_six(window, longitude - base)
    expected = torch_kernels._natural_spline_six(along_columns, latitude - base)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", [(3, 6), (6, 3), (5, 5)])
def test_dem_sampler_rejects_tiles_smaller_than_six_points(
    shape: tuple[int, int],
) -> None:
    """A six-point stencil returns invalid for an undersized DEM tile."""
    import torch

    dem = torch.arange(shape[0] * shape[1], dtype=torch.float64).reshape(shape)
    actual = torch_kernels._sample_dem_six(
        dem,
        torch.tensor([1.25], dtype=torch.float64),
        torch.tensor([1.25], dtype=torch.float64),
        0.0,
        0.0,
        1.0,
        1.0,
    )

    assert torch.isnan(actual).all()


@pytest.mark.parametrize("base", [0, 2])
def test_dem_sampler_rejects_incomplete_six_point_windows_in_a_six_by_six_tile(
    base: int,
) -> None:
    """A 6x6 tile rejects bases whose six-point window exceeds its bounds."""
    import torch

    dem = torch.arange(36, dtype=torch.float64).reshape(6, 6)
    actual = torch_kernels._sample_dem_six(
        dem,
        torch.tensor([base + 0.25], dtype=torch.float64),
        torch.tensor([base + 0.5], dtype=torch.float64),
        0.0,
        0.0,
        1.0,
        1.0,
    )

    assert torch.isnan(actual).all()


def test_dem_sampler_accepts_upper_valid_base_in_an_eight_by_eight_tile() -> None:
    """An 8x8 tile accepts base 3, whose window ends at the final row/column."""
    import torch

    rows = torch.arange(8, dtype=torch.float64).unsqueeze(1)
    columns = torch.arange(8, dtype=torch.float64).unsqueeze(0)
    dem = 10.0 + 2.0 * rows + 3.0 * columns
    latitude = torch.tensor([3.25], dtype=torch.float64)
    longitude = torch.tensor([3.5], dtype=torch.float64)
    actual = torch_kernels._sample_dem_six(
        dem,
        latitude,
        longitude,
        0.0,
        0.0,
        1.0,
        1.0,
    )

    expected = 10.0 + 2.0 * latitude + 3.0 * longitude
    torch.testing.assert_close(actual, expected)
