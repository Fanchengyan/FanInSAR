"""Public polynomial consumers against frozen pre-P0087 arithmetic and NumPy."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from faninsar.core.algorithms import fit_surface2d
from faninsar.core.algorithms.fitting import _compat
from faninsar.processing.alg import detrend_surface
from faninsar.processing.atmosphere import estimation
from faninsar.processing.atmosphere.config import IonosphereEstimationConfig
from faninsar.processing.runtime import torch_kernels


@pytest.fixture(autouse=True)
def _one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def frozen_detrend(
    data, mask=None, *, x_degree=1, y_degree=1, max_num_sample=None, dtype=None
):
    """Freeze the original moment/solve/evaluate order, independent of core."""
    basis = [(0, 0)]
    basis.extend((i, 0) for i in range(1, y_degree + 1))
    basis.extend((0, j) for j in range(1, x_degree + 1))
    if x_degree >= 2 and y_degree >= 2:
        basis.append((1, 1))
    n_param = len(basis)
    target = data
    out_dtype = dtype if dtype is not None else data.dtype
    lead_shape = target.shape[:-2]
    height, width = target.shape[-2:]
    n_slice = lead_shape.numel() if lead_shape else 1
    slices = target.reshape(n_slice, height, width)
    valid = torch.isfinite(slices)
    if mask is not None:
        valid = valid & mask.to(device=target.device).to(torch.bool).unsqueeze(0)
    if max_num_sample is not None:
        n_valid_max = int(valid.sum(dim=(1, 2)).max().item())
        if n_valid_max > max_num_sample:
            step = math.ceil(math.sqrt(n_valid_max / max_num_sample))
            keep = torch.zeros(height, width, dtype=torch.bool, device=target.device)
            keep[step // 2 :: step, step // 2 :: step] = True
            valid = valid & keep.unsqueeze(0)
    acc = torch.float64
    y_lin = torch.arange(height, dtype=acc, device=target.device)
    x_lin = torch.arange(width, dtype=acc, device=target.device)
    max_pow = 2 * max(x_degree, y_degree, 1)
    y_pow = [torch.ones(height, dtype=acc, device=target.device)]
    x_pow = [torch.ones(width, dtype=acc, device=target.device)]
    for _ in range(max_pow):
        y_pow.append(y_pow[-1] * y_lin)
        x_pow.append(x_pow[-1] * x_lin)
    values = slices.to(acc)
    valid_f = valid.to(acc)
    masked = torch.where(
        valid, values, torch.zeros((), dtype=acc, device=target.device)
    )
    moments = torch.empty(n_param, n_slice, dtype=acc, device=target.device)
    for idx, (i, j) in enumerate(basis):
        row = torch.einsum("h,ehw->ew", y_pow[i], masked)
        moments[idx] = torch.einsum("w,ew->e", x_pow[j], row)
    rows_valid = {}
    normal = torch.empty(n_param, n_param, n_slice, dtype=acc, device=target.device)
    for ai, (i1, j1) in enumerate(basis):
        for bi, (i2, j2) in enumerate(basis):
            power = i1 + i2
            if power not in rows_valid:
                rows_valid[power] = torch.einsum("h,ehw->ew", y_pow[power], valid_f)
            normal[ai, bi] = torch.einsum("w,ew->e", x_pow[j1 + j2], rows_valid[power])
    scale = torch.sqrt(
        torch.diagonal(normal, dim1=0, dim2=1).clamp_min(torch.finfo(acc).tiny)
    )
    normal_scaled = normal / (scale.T[:, None, :] * scale.T[None, :, :])
    moments_scaled = moments / scale.T
    coef_scaled = torch.linalg.solve(
        normal_scaled.permute(2, 0, 1), moments_scaled.T.unsqueeze(-1)
    ).squeeze(-1)
    coef = coef_scaled / scale
    y_eval = [tensor.to(out_dtype) for tensor in y_pow]
    x_eval = [tensor.to(out_dtype) for tensor in x_pow]
    coef_eval = coef.to(out_dtype)
    surface = torch.zeros(n_slice, height, width, dtype=out_dtype, device=target.device)
    for idx, (i, j) in enumerate(basis):
        surface += (
            coef_eval[:, idx, None, None]
            * y_eval[i][None, :, None]
            * x_eval[j][None, None, :]
        )
    detrended = target.to(out_dtype) - surface
    out_shape = (*lead_shape, height, width)
    return detrended.reshape(out_shape), surface.reshape(out_shape)


def frozen_weighted(y, x_norm, weight, *, degree=2):
    """Frozen supplied-coordinate total-degree least-squares implementation."""
    orig_shape = y.shape
    xn = x_norm[..., 0].reshape(-1).to(torch.float64)
    yn = x_norm[..., 1].reshape(-1).to(torch.float64)
    flat_y = y.reshape(-1).to(torch.float64)
    w = weight.reshape(-1).to(torch.float64)
    columns = []
    for total in range(degree + 1):
        for py in range(total + 1):
            px = total - py
            columns.append((yn**py) * (xn**px))
    design = torch.stack(columns, dim=-1)
    sqrt_w = torch.sqrt(w.clamp(min=0.0))
    theta = torch.linalg.lstsq(design * sqrt_w.unsqueeze(-1), flat_y * sqrt_w).solution
    surface = design @ theta
    return surface.reshape(orig_shape).to(y.dtype)


@pytest.mark.parametrize("degrees", [(0, 0), (2, 0), (2, 1), (2, 2), (3, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sample", [None, 120])
def test_public_detrend_is_bitwise_frozen_and_delegates(
    monkeypatch, degrees, dtype, sample
):
    generator = torch.Generator().manual_seed(87)
    source = torch.randn(2, 19, 23, generator=generator, dtype=dtype).transpose(-1, -2)
    source[0, 0, 0] = torch.nan
    source[1, 2, 3] = torch.inf
    mask = torch.ones(source.shape[-2:], dtype=torch.bool)
    mask[2:5, 4:8] = False
    before = source.clone()
    calls = []
    original = _compat._separable_polynomial_fit

    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(_compat, "_separable_polynomial_fit", observed)
    kwargs = dict(x_degree=degrees[0], y_degree=degrees[1], max_num_sample=sample)
    expected = frozen_detrend(source, mask, **kwargs)
    actual = detrend_surface(source, mask, **kwargs)
    assert calls
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0, equal_nan=True)
        assert value.dtype == dtype
    torch.testing.assert_close(source, before, rtol=0, atol=0, equal_nan=True)


def test_public_detrend_retains_rank_failure_and_sampling_policy():
    with pytest.raises(ValueError, match="singular"):
        detrend_surface(torch.ones(1, 9, dtype=torch.float64))
    with pytest.raises(ValueError, match="At least 3 valid pixels"):
        detrend_surface(torch.ones(9, 9), max_num_sample=1)


def test_public_svd_total_degree_is_independent_of_legacy_selected_powers():
    yy, xx = np.mgrid[:9, :11].astype(np.float64)
    source = 2 + 0.2 * xx + 0.4 * yy + 0.05 * xx * yy
    result = fit_surface2d(source, degree=2)
    actual = result.evaluate(yy, xx).numpy()
    independent_design = np.column_stack(
        [
            np.ones(xx.size),
            xx.ravel(),
            yy.ravel(),
            xx.ravel() ** 2,
            (xx * yy).ravel(),
            yy.ravel() ** 2,
        ]
    )
    expected = (
        independent_design
        @ np.linalg.lstsq(independent_design, source.ravel(), rcond=None)[0]
    )
    np.testing.assert_allclose(actual.ravel(), expected, atol=1e-10, rtol=1e-12)
    residual, _ = detrend_surface(torch.from_numpy(source), x_degree=2, y_degree=1)
    assert float(residual.abs().max()) > 0.1
    crossed, _ = detrend_surface(torch.from_numpy(source), x_degree=2, y_degree=2)
    assert float(crossed.abs().max()) < 1e-10


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("public_name", ["align_absolute_jumps", "solve_guided_split"])
def test_public_phase_cycles_are_frozen_at_rounding_thresholds(
    monkeypatch, dtype, public_name
):
    rows, cols = 17, 21
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, rows), torch.linspace(-1, 1, cols), indexing="ij"
    )
    low = (30 + xx + 0.7 * yy).to(dtype)
    # Low-weight isolated deviations bracket +/- half cycles while the reference
    # surface is fixed by the remaining well-conditioned grid.
    diff = (0.2 + 0.3 * xx - 0.1 * yy + 0.07 * xx * yy).to(dtype)
    for index, delta in enumerate([-1e-4, 1e-4, -1e-4, 1e-4]):
        diff[3 + index, 4] += (1 if index < 2 else -1) * math.pi + delta
    high = low - diff
    coherence = torch.full_like(low, 0.9)
    coherence[3:7, 4] = 0.01
    config = IonosphereEstimationConfig(
        f0=1.25e9,
        freq_low=1.23e9,
        freq_high=1.27e9,
        alignment_strategy="alosstack_pixel_rounded",
    )
    calls = []
    original = _compat._weighted_total_degree_surface

    def observed(y, coords, weight, degree):
        assert coords.dtype == torch.float32
        calls.append(True)
        return original(y, coords, weight, degree)

    monkeypatch.setattr(_compat, "_weighted_total_degree_surface", observed)
    public = getattr(estimation, public_name)
    args = (
        (low, high, coherence, config)
        if public_name == "solve_guided_split"
        else (low, high, config)
    )
    actual = public(*args)
    assert calls
    monkeypatch.setattr(estimation, "_weighted_polyfit2d", frozen_weighted)
    expected = public(*args)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("container", [tuple, list])
def test_runtime_explicit_power_is_exact_for_legacy_coefficients(
    monkeypatch, dtype, container
):
    coordinates = torch.linspace(-1.7, 2.3, 257, dtype=dtype)
    coefficients = container([0.1, -3.0, 4.25, -0.003])
    expected = torch.zeros_like(coordinates)
    power = torch.ones_like(coordinates)
    for coefficient in coefficients:
        expected = expected + float(coefficient) * power
        power = power * coordinates
    calls = []
    original = _compat._explicit_power_polynomial

    def observed(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(_compat, "_explicit_power_polynomial", observed)
    actual = torch_kernels._poly_eval_torch(coefficients, coordinates)
    assert calls
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_compatibility_leaf_import_does_not_require_upper_layers():
    from tests.core.algorithms.test_imports import _run

    _run("""
import builtins
import pathlib
import sys
import types
root = pathlib.Path(sys.argv[1])
for name, path in [("faninsar", root / "faninsar"),
                   ("faninsar.core", root / "faninsar" / "core")]:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert not any(name == "faninsar." + part or name.startswith("faninsar." + part + ".")
                   for part in ("processing", "runtime", "timeseries", "io", "missions")), name
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from faninsar.core.algorithms.fitting import _compat
assert "torch" not in sys.modules
import torch
assert _compat._explicit_power_polynomial((2., 3.), torch.tensor([1.])).item() == 5.
""")
