"""P0087 geometry delegation preserves the qualified six-point Torch graph."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch
from scipy.interpolate import CubicSpline

from faninsar.processing.geometry import torch_kernels


def _core_spline():
    return importlib.import_module("faninsar.core.algorithms.sampling._spline_six")


def _legacy_weights(fraction):
    """Freeze the pre-extraction geometry expression, including coefficient order."""
    second_one = (
        1.6076555023923444,
        -3.6459330143540667,
        2.5837320574162677,
        -0.6889952153110048,
        0.1722488038277512,
        -0.0287081339712919,
    )
    second_two = (
        -0.4306220095693780,
        2.5837320574162677,
        -4.3349282296650715,
        2.7559808612440193,
        -0.6889952153110048,
        0.1148325358851674,
    )
    fraction2 = fraction * fraction
    fraction3 = fraction2 * fraction
    weights = [
        fraction * (-one / 3.0 - two / 6.0)
        + fraction2 * (one / 2.0)
        + fraction3 * (two - one) / 6.0
        for one, two in zip(second_one, second_two, strict=True)
    ]
    weights[1] = weights[1] + 1.0 - fraction
    weights[2] = weights[2] + fraction
    return torch.stack(weights, dim=-1)


def _legacy_sample(source, row, column):
    """Freeze the pre-extraction index-space window/reduction/NaN expression."""
    row_base = torch.floor(row).to(torch.int64)
    column_base = torch.floor(column).to(torch.int64)
    rows, columns = source.shape[-2:]
    if rows < 6 or columns < 6:
        return torch.full_like(row, torch.nan)
    valid = (
        torch.isfinite(row)
        & torch.isfinite(column)
        & (row_base >= 1)
        & (row_base <= rows - 5)
        & (column_base >= 1)
        & (column_base <= columns - 5)
    )
    safe_row = row_base.clamp(1, rows - 5)
    safe_column = column_base.clamp(1, columns - 5)
    column_fraction = column - safe_column
    row_fraction = row - safe_row
    column_weights = _legacy_weights(column_fraction)
    windows = source.unfold(0, 6, 1).unfold(1, 6, 1)
    window = windows[safe_row - 1, safe_column - 1]
    row_values = torch.sum(window * column_weights.unsqueeze(-2), dim=-1)
    sampled = torch.sum(row_values * _legacy_weights(row_fraction), dim=-1)
    return torch.where(
        valid & torch.isfinite(sampled), sampled, torch.full_like(sampled, torch.nan)
    )


def test_geometry_spline_adapters_delegate_to_core(monkeypatch):
    core = _core_spline()
    marker = torch.tensor([87.0], dtype=torch.float64)
    fraction = torch.tensor([0.25], dtype=torch.float64)
    values = torch.ones((1, 6), dtype=torch.float64)
    monkeypatch.setattr(core, "_spline_six_weights", lambda *args: marker)
    assert torch_kernels._spline_six_weights(fraction) is marker
    monkeypatch.setattr(core, "_natural_spline_six_weighted", lambda *args: marker)
    assert torch_kernels._natural_spline_six(values, fraction) is marker
    captured = []

    def sample(source, rows, cols):
        captured.append((source, rows, cols))
        return marker

    monkeypatch.setattr(core, "_sample_six", sample)
    source = torch.ones((8, 9), dtype=torch.float32)
    result = torch_kernels._sample_dem_six(
        source, torch.tensor([9.5]), torch.tensor([21.5]), 10.0, 20.0, -0.25, 0.5
    )
    assert result is marker
    assert captured[0][0] is source
    torch.testing.assert_close(captured[0][1], torch.tensor([2.0]))
    torch.testing.assert_close(captured[0][2], torch.tensor([3.0]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_weights_are_float64_and_conform_to_independent_natural_spline(dtype):
    core = _core_spline()
    fraction = torch.linspace(0, 1, 101, dtype=dtype)
    weights = core._spline_six_weights(fraction)
    expected = CubicSpline(np.arange(6), np.eye(6), axis=0, bc_type="natural")(
        1.0 + fraction.double().numpy()
    )
    assert weights.dtype == torch.float64
    assert weights.device == fraction.device
    np.testing.assert_allclose(weights.numpy(), expected, atol=9e-16, rtol=0)
    torch.testing.assert_close(
        weights, _legacy_weights(fraction.double()), atol=0, rtol=0
    )


def test_weighted_and_recurrence_forms_share_the_same_mathematics():
    from faninsar.core.algorithms.sampling._kernels import _natural_spline_six

    generator = torch.Generator().manual_seed(87)
    values = torch.randn((257, 6), generator=generator, dtype=torch.float64) * 3000
    fraction = torch.rand(257, generator=generator, dtype=torch.float64)
    actual = _core_spline()._natural_spline_six_weighted(values, fraction)
    recurrence = _natural_spline_six(values, fraction)
    torch.testing.assert_close(actual, recurrence, atol=5e-12, rtol=1e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("transpose", [False, True])
def test_random_terrain_preserves_exact_legacy_values_and_nan_mask(dtype, transpose):
    generator = torch.Generator().manual_seed(87)
    source = torch.randn((19, 23), generator=generator, dtype=dtype) * 1000
    source[7, 9] = torch.nan
    source[13, 17] = torch.inf
    if transpose:
        source = source.T
    rows = torch.rand((13, 17), generator=generator, dtype=torch.float64) * 25 - 3
    cols = torch.rand((13, 17), generator=generator, dtype=torch.float64) * 25 - 3
    rows[0, :4] = torch.tensor([torch.nan, torch.inf, -torch.inf, 1e300])
    cols[1, :4] = torch.tensor([torch.nan, torch.inf, -torch.inf, -1e300])
    expected = _legacy_sample(source, rows, cols)
    actual = _core_spline()._sample_six(source, rows, cols)
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    np.testing.assert_array_equal(torch.isnan(actual), torch.isnan(expected))
    adapted = torch_kernels._sample_dem_six(
        source, 50.0 - rows / 16.0, 10.0 + cols / 16.0, 50.0, 10.0, -1 / 16, 1 / 16
    )
    # Compare after the unchanged physical-coordinate conversion, not against
    # pre-conversion coordinates whose final bits may differ after subtraction.
    expected_adapted = _legacy_sample(
        source,
        ((50.0 - rows / 16.0) - 50.0) / (-1 / 16),
        ((10.0 + cols / 16.0) - 10.0) / (1 / 16),
    )
    np.testing.assert_array_equal(adapted.numpy(), expected_adapted.numpy())


@pytest.mark.parametrize("bad", [torch.nan, torch.inf, -torch.inf])
@pytest.mark.parametrize("row_offset", range(6))
def test_integer_queries_require_every_stencil_sample(bad, row_offset):
    core = _core_spline()
    row = col = torch.tensor([2.0], dtype=torch.float64)
    for col_offset in range(6):
        source = torch.ones((9, 10), dtype=torch.float64)
        source[1 + row_offset, 1 + col_offset] = bad
        assert torch.isnan(core._sample_six(source, row, col)).all()
        assert torch.isnan(_legacy_sample(source, row, col)).all()


@pytest.mark.parametrize("shape", [(0, 0), (1, 8), (5, 9), (9, 5)])
def test_undersized_geometry_tiles_remain_nan(shape):
    source = torch.zeros(shape, dtype=torch.float32)
    coordinates = torch.tensor([0.0, 2.0, torch.nan], dtype=torch.float64)
    result = _core_spline()._sample_six(source, coordinates, coordinates)
    assert result.dtype == torch.float64
    assert torch.isnan(result).all()
    adapted = torch_kernels._sample_dem_six(
        source, coordinates, coordinates, 0.0, 0.0, 1.0, 1.0
    )
    torch.testing.assert_close(result, adapted, equal_nan=True)


def test_geometry_sampler_remains_fullgraph_without_host_scalar_or_transfer():
    graph_targets = []

    def capture(graph, inputs):
        graph_targets.extend(str(node.target) for node in graph.graph.nodes)
        return graph.forward

    generator = torch.Generator().manual_seed(88)
    source = torch.randn((11, 13), generator=generator, dtype=torch.float64)
    rows = torch.tensor([1.0, 2.125, 5.75, 7.0, torch.nan], dtype=torch.float64)
    cols = torch.tensor([1.0, 4.25, 8.75, 4.0, 2.0], dtype=torch.float64)
    compiled = torch.compile(
        torch_kernels._sample_dem_six, backend=capture, fullgraph=True, dynamic=False
    )
    expected = _legacy_sample(source, rows, cols)
    actual = compiled(source, rows, cols, 0.0, 0.0, 1.0, 1.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    assert graph_targets
    assert not {"item", "cpu", "numpy", "_local_scalar_dense"}.intersection(
        graph_targets
    )
