"""Tests for the polynomial trend-surface removal (MintPy deramp equivalent)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import xarray as xr
from pyproj import Transformer

import faninsar  # registers the ``fis`` accessor
from faninsar.data.query import Points
from faninsar.processing.alg import detrend_surface


def _grid(height: int = 12, width: int = 15) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    return yy, xx


class TestDetrendSurfaceCore:
    """Torch kernel: exact fits, masks, validation, and dtypes."""

    def test_exact_plane_is_removed(self) -> None:
        """A (1, 1) plane surface is removed exactly."""
        yy, xx = _grid()
        data = torch.as_tensor(3.0 * xx - 2.0 * yy + 5.0)
        detrended, surface = detrend_surface(data)
        assert detrended.shape == data.shape
        assert detrended.dtype == data.dtype
        assert detrended.abs().max().item() == pytest.approx(0.0, abs=1e-10)
        assert surface.abs().max().item() > 1.0

    def test_x_only_parabola_with_zero_y_degree(self) -> None:
        """A pure-x parabola is removed with y_degree=0."""
        _, xx = _grid()
        data = torch.as_tensor(0.5 * xx**2 + 2.0)
        detrended, _ = detrend_surface(data, x_degree=2, y_degree=0)
        assert detrended.abs().max().item() < 1e-9

    def test_asymmetric_two_one_surface(self) -> None:
        """Independent degrees fit asymmetric surfaces; x*y needs (2, 2)."""
        yy, xx = _grid()
        data = torch.as_tensor(0.01 * xx**2 + 0.02 * yy + 7.0)
        detrended, _ = detrend_surface(data, x_degree=2, y_degree=1)
        assert detrended.abs().max().item() < 1e-8
        # the default plane (1, 1) cannot represent the x**2 curvature
        detrended_plane, _ = detrend_surface(data)
        assert detrended_plane.abs().max().item() > 0.1
        # the x * y cross term needs (2, 2), not (2, 1)
        crossed = torch.as_tensor(0.005 * xx * yy)
        detrended_two_one, _ = detrend_surface(crossed, x_degree=2, y_degree=1)
        assert detrended_two_one.abs().max().item() > 0.1
        detrended_two_two, _ = detrend_surface(crossed, x_degree=2, y_degree=2)
        assert detrended_two_two.abs().max().item() < 1e-8

    def test_zero_degrees_remove_mean(self) -> None:
        """(0, 0) reduces to per-slice mean removal."""
        rng = np.random.default_rng(0)
        data = torch.as_tensor(rng.normal(size=(10, 8)) + 4.0)
        detrended, _ = detrend_surface(data, x_degree=0, y_degree=0)
        assert detrended.mean().item() == pytest.approx(0.0, abs=1e-12)

    def test_mask_excludes_pixels_from_fit(self) -> None:
        """Masked-out pixels do not contribute to the fit."""
        yy, xx = _grid()
        data = 0.1 * xx + 0.2 * yy
        data[:4, :4] += 100.0
        mask = np.ones(data.shape, dtype=bool)
        mask[:4, :4] = False
        detrended, _ = detrend_surface(torch.as_tensor(data), torch.as_tensor(mask))
        assert detrended[4:, 4:].abs().max().item() < 1e-9

    def test_masked_pixels_are_still_corrected(self) -> None:
        """Masked-out pixels still have the fitted surface subtracted."""
        yy, xx = _grid()
        data = torch.as_tensor(0.1 * xx + 0.2 * yy)
        # fit on the left half only; the right half is corrected anyway
        mask = torch.zeros(data.shape, dtype=torch.bool)
        mask[:, : data.shape[1] // 2] = True
        detrended, _ = detrend_surface(data, mask)
        assert detrended.abs().max().item() < 1e-9

    def test_slices_are_detrended_independently(self) -> None:
        """Leading dimensions are fitted independently per slice."""
        yy, xx = _grid()
        first = 0.5 * xx + 1.0
        second = -0.25 * yy + 2.0
        data = torch.as_tensor(np.stack([first, second]))
        detrended, _ = detrend_surface(data)
        assert detrended.abs().max().item() < 1e-10

    def test_nan_pixels_are_ignored_and_preserved(self) -> None:
        """NaN pixels are excluded from the fit and stay NaN."""
        yy, xx = _grid()
        data = 0.3 * xx + 0.1 * yy
        data[0, 0] = np.nan
        detrended, surface = detrend_surface(torch.as_tensor(data))
        assert torch.isnan(detrended[0, 0])
        assert torch.isfinite(surface).all()
        finite = detrended[1:, 1:]
        assert finite.abs().max().item() < 1e-9

    def test_too_few_valid_pixels_raises(self) -> None:
        """Slices with fewer valid pixels than parameters raise."""
        with pytest.raises(ValueError, match="valid pixels"):
            detrend_surface(torch.ones(2, 3), x_degree=3, y_degree=3)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"x_degree": 4},
            {"y_degree": -1},
            {"x_degree": 1.5},
            {"x_degree": True},
        ],
    )
    def test_invalid_degree_raises(self, kwargs: dict) -> None:
        """Non-integer or out-of-range degrees raise."""
        with pytest.raises(ValueError, match="degree"):
            detrend_surface(torch.ones(4, 4), **kwargs)

    def test_integer_data_raises(self) -> None:
        """Non-floating input dtype raises."""
        with pytest.raises(ValueError, match="floating"):
            detrend_surface(torch.ones(4, 4, dtype=torch.int64))

    def test_bad_mask_shape_raises(self) -> None:
        """A mask that does not match the spatial shape raises."""
        with pytest.raises(ValueError, match="mask"):
            detrend_surface(torch.ones(4, 4), torch.ones(3, 3))

    def test_dtype_follows_input_and_override(self) -> None:
        """Output dtype follows input unless overridden."""
        data = torch.ones(5, 5, dtype=torch.float32)
        detrended, surface = detrend_surface(data)
        assert detrended.dtype == torch.float32
        assert surface.dtype == torch.float32
        detrended64, surface64 = detrend_surface(data, dtype=torch.float64)
        assert detrended64.dtype == torch.float64
        assert surface64.dtype == torch.float64

    def test_max_num_sample_stride_keeps_exact_plane(self) -> None:
        """Stride subsampling still recovers an exact plane."""
        yy, xx = _grid(40, 40)
        data = torch.as_tensor(0.05 * xx + 0.02 * yy)
        detrended, _ = detrend_surface(data, max_num_sample=100)
        assert detrended.abs().max().item() < 1e-8


def _pair_da() -> xr.DataArray:
    x = np.arange(6.0)
    y = np.arange(4.0)
    data = 2.0 * x[None, :] - 1.0 * y[:, None]
    return xr.DataArray(
        np.stack([data, data * 0.5]),
        coords={"pair": ["a", "b"], "y": y, "x": x},
        dims=("pair", "y", "x"),
        attrs={"unit": "rad"},
        name="unw",
    )


class TestDetrendSurfaceAccessor:
    """``fis`` DataArray accessor: metadata, masks, and reference points."""

    def test_removes_plane_and_preserves_metadata(self) -> None:
        """Detrending preserves dims, attrs, name, and shape."""
        da = _pair_da()
        out = da.fis.detrend_surface()
        assert out.dims == da.dims
        assert out.attrs == da.attrs
        assert out.name == da.name
        assert out.shape == da.shape
        assert np.abs(out.values).max() < 1e-10

    def test_return_surface_sum_invariant(self) -> None:
        """Detrended + surface reconstructs the input array."""
        da = _pair_da()
        detrended, surface = da.fis.detrend_surface(return_surface=True)
        assert detrended.dims == da.dims
        assert surface.dims == da.dims
        np.testing.assert_allclose(
            detrended.values + surface.values, da.values, rtol=0, atol=1e-10
        )

    def test_transposed_dims_are_handled(self) -> None:
        """Spatial dims are resolved regardless of dimension order."""
        da = _pair_da().transpose("x", "y", "pair")
        out = da.fis.detrend_surface()
        assert out.dims == da.dims
        assert np.abs(out.values.transpose(2, 1, 0)).max() < 1e-10

    def test_ref_points_reference_to_zero(self) -> None:
        """Reference points tie the detrended output to zero."""
        x = np.arange(6.0)
        y = np.arange(4.0)
        plane = 2.0 * x[None, :] - 1.0 * y[:, None]
        bump = 5.0 * np.exp(-((y[:, None] - 1) ** 2 + (x[None, :] - 1) ** 2))
        da = xr.DataArray(
            plane + bump, coords={"y": y, "x": x}, dims=("y", "x")
        ).fis.set_spatial_ref(crs="EPSG:4326")
        pts = Points([[1.0, 1.0]], crs="EPSG:4326")
        out = da.fis.detrend_surface(ref_points=pts)
        assert out.sel(x=1.0, y=1.0).item() == pytest.approx(0.0, abs=1e-12)
        detrended, surface = da.fis.detrend_surface(ref_points=pts, return_surface=True)
        np.testing.assert_allclose(
            detrended.values + surface.values, da.values, rtol=0, atol=1e-10
        )

    def test_ref_points_crs_is_converted(self) -> None:
        """Reference points in another CRS are transformed first."""
        x = np.arange(500000.0, 500600.0, 100.0)
        y = np.arange(4447000.0, 4447400.0, 100.0)
        plane = 0.001 * (x[None, :] - 500000.0) + 0.002 * (y[:, None] - 4447000.0)
        da = xr.DataArray(
            plane, coords={"y": y, "x": x}, dims=("y", "x")
        ).fis.set_spatial_ref(crs="EPSG:32650")
        to_geographic = Transformer.from_crs("EPSG:32650", "EPSG:4326", always_xy=True)
        lon, lat = to_geographic.transform(500200.0, 4447200.0)
        pts = Points([[lon, lat]], crs="EPSG:4326")
        out = da.fis.detrend_surface(ref_points=pts)
        assert out.sel(x=500200.0, y=4447200.0).item() == pytest.approx(0.0, abs=1e-10)

    def test_ref_points_all_nan_slice_raises(self) -> None:
        """All-NaN reference samples on a slice raise."""
        da = _pair_da()
        values = da.values.copy()
        values[1, 1, 1] = np.nan
        da = da.copy(data=values)
        pts = Points([[1.0, 1.0]], crs=None)
        with pytest.raises(ValueError, match="reference points"):
            da.fis.detrend_surface(ref_points=pts)

    def test_ref_points_with_crs_on_crsless_array_raises(self) -> None:
        """CRS-carrying points on a CRS-less array raise."""
        da = _pair_da()
        pts = Points([[1.0, 1.0]], crs="EPSG:4326")
        with pytest.raises(ValueError, match="CRS"):
            da.fis.detrend_surface(ref_points=pts)

    def test_ref_points_wrong_type_raises(self) -> None:
        """Non-Points ref_points raise TypeError."""
        da = _pair_da()
        with pytest.raises(TypeError, match="ref_points"):
            da.fis.detrend_surface(ref_points=[1.0, 1.0])

    def test_mask_restricts_fit(self) -> None:
        """A DataArray mask restricts the pixels used in the fit."""
        x = np.arange(6.0)
        y = np.arange(4.0)
        data = 2.0 * x[None, :] - 1.0 * y[:, None]
        data[:2, :2] += 50.0
        da = xr.DataArray(data, coords={"y": y, "x": x}, dims=("y", "x"))
        mask = xr.DataArray(
            np.ones((4, 6), dtype=bool), coords={"y": y, "x": x}, dims=("y", "x")
        )
        mask[:2, :2] = False
        out = da.fis.detrend_surface(mask=mask)
        assert np.abs(out.values[2:, 2:]).max() < 1e-9

    def test_mask_coord_mismatch_raises(self) -> None:
        """A mask on different coordinates raises."""
        da = _pair_da()
        mask = xr.DataArray(
            np.ones((4, 6), dtype=bool),
            coords={"y": np.arange(4.0) + 0.5, "x": np.arange(6.0)},
            dims=("y", "x"),
        )
        with pytest.raises((ValueError, KeyError)):
            da.fis.detrend_surface(mask=mask)


# (y power, x power) columns per MintPy ramp type, MintPy column order.
_MINTPY_BASIS = {
    "linear": [(1, 0), (0, 1), (0, 0)],
    "linear_range": [(0, 1), (0, 0)],
    "linear_azimuth": [(1, 0), (0, 0)],
    "quadratic": [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1), (0, 0)],
    "quadratic_range": [(0, 2), (0, 1), (0, 0)],
    "quadratic_azimuth": [(2, 0), (1, 0), (0, 0)],
}


def _mintpy_deramp(data: np.ndarray, ramp_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Float64 port of ``mintpy.objects.ramp.deramp``.

    The design-matrix column orders follow the MintPy source verbatim
    (linear: ``yy, xx, 1``; quadratic: ``yy**2, xx**2, yy*xx, yy, xx, 1``;
    and the range/azimuth subsets). The ``ignore_zero_value`` branch is
    intentionally not ported because FanInSAR marks invalid pixels with
    NaN, which both implementations drop here.
    """
    length, width = data.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(length))
    xx = xx.reshape(-1, 1).astype(np.float64)
    yy = yy.reshape(-1, 1).astype(np.float64)
    ones = np.ones_like(xx)
    if ramp_type == "linear":
        g = np.hstack((yy, xx, ones))
    elif ramp_type == "quadratic":
        g = np.hstack((yy**2, xx**2, yy * xx, yy, xx, ones))
    elif ramp_type == "linear_range":
        g = np.hstack((xx, ones))
    elif ramp_type == "linear_azimuth":
        g = np.hstack((yy, ones))
    elif ramp_type == "quadratic_range":
        g = np.hstack((xx**2, xx, ones))
    elif ramp_type == "quadratic_azimuth":
        g = np.hstack((yy**2, yy, ones))
    else:
        msg = f"unknown ramp type {ramp_type!r}"
        raise ValueError(msg)

    values = data.reshape(-1, 1)
    mask = ~np.isnan(values[:, 0])
    coef = np.linalg.pinv(g[mask], rcond=1e-15) @ values[mask]
    ramp = (g @ coef).reshape(data.shape)
    return data - ramp, ramp


def _synthetic_mintpy_case(ramp_type: str, rng: np.random.Generator) -> np.ndarray:
    height, width = 37, 53
    yy, xx = _grid(height, width)
    data = np.zeros((height, width), dtype=np.float64)
    for i, j in _MINTPY_BASIS[ramp_type]:
        scale = 10.0 / max(float(height) ** i * float(width) ** j, 1.0)
        data += rng.normal() * scale * yy**i * xx**j
    data += rng.normal(0.0, 0.5, size=data.shape)
    data[:4, :5] = np.nan
    data[rng.random(data.shape) < 0.05] = np.nan
    return data


class TestMintPyEquivalence:
    """Float64 equivalence with the MintPy deramp reference."""

    @pytest.mark.parametrize(
        ("ramp_type", "x_degree", "y_degree"),
        [
            ("linear", 1, 1),
            ("linear_range", 1, 0),
            ("linear_azimuth", 0, 1),
            ("quadratic", 2, 2),
            ("quadratic_range", 2, 0),
            ("quadratic_azimuth", 0, 2),
        ],
    )
    def test_matches_mintpy_deramp(
        self, ramp_type: str, x_degree: int, y_degree: int
    ) -> None:
        """Detrended output and surface match MintPy deramp."""
        rng = np.random.default_rng(42)
        data = _synthetic_mintpy_case(ramp_type, rng)

        expected, expected_ramp = _mintpy_deramp(data, ramp_type)
        detrended, surface = detrend_surface(
            torch.as_tensor(data), x_degree=x_degree, y_degree=y_degree
        )

        np.testing.assert_allclose(
            detrended.numpy(), expected, rtol=0, atol=1e-8, equal_nan=True
        )
        np.testing.assert_allclose(surface.numpy(), expected_ramp, rtol=0, atol=1e-8)
