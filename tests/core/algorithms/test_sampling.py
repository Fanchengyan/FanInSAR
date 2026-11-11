"""P0087 sampling contracts and independent numerical qualification."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_REFERENCE_PATH = Path(__file__).parent / "reference" / "sampling.py"
_SPEC = importlib.util.spec_from_file_location("sampling_reference", _REFERENCE_PATH)
_REFERENCE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REFERENCE)
interpolate_reference = _REFERENCE.interpolate_reference


def _interpolate(*args, **kwargs):
    """Resolve the public export after test collection for clear red evidence."""
    from faninsar.core.algorithms import interpolate2d

    return interpolate2d(*args, **kwargs)


def test_bilinear_plane_broadcasts_queries_across_images():
    row_grid, col_grid = np.indices((7, 8), dtype=np.float64)
    plane = 2 * row_grid - 3 * col_grid + 10
    source = np.stack((plane, plane + 100)).astype(np.float32)
    rows = np.array([[0.0], [1.25], [6.0]])
    cols = np.array([[0.0, 3.5, 7.0]])
    expected, expected_valid = interpolate_reference(source, rows, cols)

    result, valid = _interpolate(source, rows, cols, return_valid=True)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float64
    assert result.device.type == "cpu"
    assert result.shape == (2, 3, 3)
    np.testing.assert_allclose(result.numpy(), expected, atol=1e-10, rtol=1e-12)
    np.testing.assert_array_equal(valid.numpy(), expected_valid)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_biquintic_matches_independent_scipy_and_unchanged_numpy(dtype):
    from faninsar.processing.geometry.api import _sample_biquintic

    rng = np.random.default_rng(87)
    source = rng.uniform(-1000, 10000, size=(15, 17)).astype(dtype)
    rows = rng.uniform(1, 11, size=(5, 7))
    cols = rng.uniform(1, 13, size=(5, 7))
    scipy_expected, expected_valid = interpolate_reference(
        source, rows, cols, method="biquintic"
    )
    numpy_expected = _sample_biquintic(source, rows, cols)

    result, valid = _interpolate(
        source, rows, cols, method="biquintic", return_valid=True
    )

    np.testing.assert_array_equal(valid.numpy(), expected_valid)
    np.testing.assert_allclose(result.numpy(), scipy_expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose(result.numpy(), numpy_expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize("method", ["nearest", "bilinear", "biquintic"])
@pytest.mark.parametrize("boundary", ["nan", "clamp"])
def test_boundaries_nonfinite_and_huge_coordinates(method, boundary):
    source = np.arange(100, dtype=np.float64).reshape(10, 10)
    rows = np.array(
        [
            -1e300,
            -0.51,
            -0.5,
            -0.49,
            0,
            1,
            5.9,
            6,
            9,
            9.5,
            1e300,
            np.nan,
            np.inf,
            -np.inf,
        ]
    )
    cols = np.full(rows.shape, 2.0)
    if method == "biquintic" and boundary == "clamp":
        with pytest.raises(ValueError, match="clamp"):
            _interpolate(source, rows, cols, method=method, boundary=boundary)
        return
    expected, expected_valid = interpolate_reference(
        source, rows, cols, method=method, boundary=boundary
    )
    actual, actual_valid = _interpolate(
        source, rows, cols, method=method, boundary=boundary, return_valid=True
    )
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-10, rtol=1e-12)
    np.testing.assert_array_equal(actual_valid.numpy(), expected_valid)


@pytest.mark.parametrize("method", ["nearest", "bilinear", "biquintic"])
def test_masks_nonfinite_and_extra_valid_never_resurrect_data(method):
    data = np.arange(2 * 12 * 13, dtype=np.float64).reshape(2, 12, 13)
    data[0, 3, 3] = np.nan
    data[1, 4, 4] = np.inf
    native_mask = np.zeros(data.shape, dtype=bool)
    native_mask[0, 5, 5] = True
    source = np.ma.array(data, mask=native_mask)
    valid = np.ones(data.shape, dtype=bool)
    valid[1, 6, 6] = False
    rows = np.array([3.0, 4.0, 5.0, 6.0, 1.2])
    cols = rows.copy()
    before_data, before_mask, before_valid = (
        data.copy(),
        source.mask.copy(),
        valid.copy(),
    )
    expected, expected_valid = interpolate_reference(
        source, rows, cols, method=method, valid=valid
    )
    actual, actual_valid = _interpolate(
        source, rows, cols, method=method, valid=valid, return_valid=True
    )
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-10, rtol=1e-12)
    np.testing.assert_array_equal(actual_valid.numpy(), expected_valid)
    np.testing.assert_array_equal(data, before_data)
    np.testing.assert_array_equal(source.mask, before_mask)
    np.testing.assert_array_equal(valid, before_valid)


def test_bilinear_zero_weight_taps_do_not_contaminate_cardinal_values():
    source = np.array([[4.0, np.nan], [np.inf, 16.0]])
    rows = np.array([0.0, 1.0, 0.5, np.nextafter(0.0, 1.0)])
    cols = np.array([0.0, 1.0, 0.5, 0.0])
    actual, valid = _interpolate(source, rows, cols, return_valid=True)
    np.testing.assert_allclose(actual.numpy(), [4, 16, np.nan, np.nan])
    np.testing.assert_array_equal(valid.numpy(), [True, True, False, False])


def test_biquintic_full_asymmetric_support_is_required_even_at_integers():
    source = np.ones((9, 9), dtype=np.float64)
    source[6, 6] = np.nan  # floor(2)-1..+4 includes this far corner.
    rows = np.array([1.0, 2.0, 4.999999999, 5.0, 0.999999999])
    actual, valid = _interpolate(
        source, rows, rows, method="biquintic", return_valid=True
    )
    np.testing.assert_allclose(actual.numpy(), [1, np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(valid.numpy(), [True, False, False, False, False])


@pytest.mark.parametrize("method", ["nearest", "bilinear"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (3, 1)])
def test_singleton_axes_preserve_strict_nonzero_support(method, shape):
    source = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    rows = np.array([0.0, 0.25, -0.25, 50.0])
    cols = np.array([0.0, 0.0, 0.0, 50.0])
    for boundary in ("nan", "clamp"):
        expected, expected_valid = interpolate_reference(
            source, rows, cols, method=method, boundary=boundary
        )
        actual, valid = _interpolate(
            source, rows, cols, method=method, boundary=boundary, return_valid=True
        )
        np.testing.assert_allclose(actual.numpy(), expected)
        np.testing.assert_array_equal(valid.numpy(), expected_valid)


@pytest.mark.parametrize("method", ["nearest", "bilinear", "biquintic"])
def test_empty_queries_and_zero_image_batch(method):
    source = torch.ones((2, 3, 8, 9), dtype=torch.float32)
    result, valid = _interpolate(
        source, np.empty((0, 1)), np.empty((1, 5)), method=method, return_valid=True
    )
    assert result.shape == valid.shape == (2, 3, 0, 5)
    assert result.dtype == torch.float64 and valid.dtype == torch.bool
    empty_batch = _interpolate(source[:0], np.array(2.0), np.array(3.0), method=method)
    assert empty_batch.shape == (0, 3)


@pytest.mark.parametrize("method", ["nearest", "bilinear", "biquintic"])
def test_noncontiguous_storage_batched_masks_and_masked_coordinates(method):
    source = (
        torch.arange(2 * 13 * 12, dtype=torch.float32)
        .reshape(2, 13, 12)
        .transpose(-1, -2)
    )
    valid = torch.ones((12, 13), dtype=torch.bool).T.contiguous().T
    valid[4, 4] = False
    rows = np.ma.array([[1.2], [4.0], [5.5]], mask=[[False], [False], [True]])
    cols = np.array([[1.0, 4.0, 6.0]])[:, ::-1]
    expected, expected_valid = interpolate_reference(
        source.numpy(), rows, cols, method=method, valid=valid.numpy()
    )
    result, result_valid = _interpolate(
        source, rows, cols, method=method, valid=valid, return_valid=True
    )
    np.testing.assert_allclose(result.numpy(), expected, atol=1e-10, rtol=1e-12)
    np.testing.assert_array_equal(result_valid.numpy(), expected_valid)


@pytest.mark.parametrize("shape", [(0, 8), (8, 0)])
@pytest.mark.parametrize("method", ["nearest", "bilinear", "biquintic"])
def test_empty_source_spatial_axes_fail(shape, method):
    with pytest.raises(ValueError, match="source|support"):
        _interpolate(np.empty(shape), np.array(0.0), np.array(0.0), method=method)


@pytest.mark.parametrize("dtype", [np.int64, np.complex128, np.bool_, np.float16])
def test_unsupported_source_dtypes_fail(dtype):
    with pytest.raises((TypeError, ValueError), match="float|dtype|real"):
        _interpolate(np.ones((8, 8), dtype=dtype), np.array(1.0), np.array(1.0))


@pytest.mark.parametrize("argument", ["source", "rows", "cols"])
def test_requires_grad_fails(argument):
    arguments = {
        "source": torch.ones((8, 8), dtype=torch.float64),
        "rows": torch.tensor([2.0]),
        "cols": torch.tensor([2.0]),
    }
    arguments[argument].requires_grad_(True)
    with pytest.raises((TypeError, ValueError), match="grad"):
        _interpolate(**arguments)


def test_shapes_masks_methods_and_source_support_are_validated():
    source, coordinate = np.ones((8, 8)), np.array(2.0)
    for options in ({"method": "bicubic"}, {"boundary": "reflect"}):
        with pytest.raises(ValueError):
            _interpolate(source, coordinate, coordinate, **options)
    with pytest.raises(ValueError, match="6|support"):
        _interpolate(source[:5], coordinate, coordinate, method="biquintic")
    with pytest.raises(ValueError, match="shape|broadcast"):
        _interpolate(source, np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="valid|shape"):
        _interpolate(
            np.ones((2, 8, 8)),
            coordinate,
            coordinate,
            valid=np.ones((1, 8, 8), dtype=bool),
        )
    with pytest.raises((TypeError, ValueError), match="[Bb]ool"):
        _interpolate(source, coordinate, coordinate, valid=source)
    with pytest.raises((TypeError, ValueError)):
        _interpolate([[1.0]], coordinate, coordinate)
    with pytest.raises(ValueError, match="source|dimension"):
        _interpolate(np.ones(8), coordinate, coordinate)


@pytest.mark.parametrize("method", ["nearest", "bilinear", "biquintic"])
@pytest.mark.parametrize("native_mask", [False, True])
def test_scalar_native_masks_and_masked_extra_valid(method, native_mask):
    source = np.ma.array(np.ones((2, 10, 10)), mask=native_mask)
    valid = np.ma.array(np.ones((10, 10), dtype=bool), mask=False)
    valid.mask[2, 2] = True
    coordinates = np.array([1.0, 2.0])
    expected, expected_valid = interpolate_reference(
        source, coordinates, coordinates, method=method, valid=valid
    )
    actual, actual_valid = _interpolate(
        source, coordinates, coordinates, method=method, valid=valid, return_valid=True
    )
    np.testing.assert_allclose(actual.numpy(), expected)
    np.testing.assert_array_equal(actual_valid.numpy(), expected_valid)
    masked_query = np.ma.array(np.array(2.0), mask=True)
    scalar, scalar_valid = _interpolate(
        np.ones((10, 10)), masked_query, np.array(2.0), method=method, return_valid=True
    )
    assert scalar.shape == scalar_valid.shape == ()
    assert torch.isnan(scalar) and not scalar_valid


@pytest.mark.parametrize("method", ["nearest", "bilinear", "biquintic"])
def test_chunking_is_identical_and_gathers_are_bounded(monkeypatch, method):
    import faninsar.core.algorithms.sampling._interpolate as implementation

    rng = np.random.default_rng(871)
    source = rng.normal(size=(2, 12, 13))
    source[0, 4, 5] = np.nan
    rows = np.linspace(1.0, 7.9, 83)[:, None]
    cols = np.linspace(1.0, 8.9, 61)[None, :]
    baseline, baseline_valid = _interpolate(
        source, rows, cols, method=method, return_valid=True
    )
    gather_sizes = []
    original_sample = implementation._sample_chunk

    def bounded_sample(*args, **kwargs):
        gather_sizes.append(args[4].numel())
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(implementation, "_QUERY_BATCH_SIZE", 23)
    monkeypatch.setattr(implementation, "_sample_chunk", bounded_sample)
    actual, actual_valid = _interpolate(
        source, rows, cols, method=method, return_valid=True
    )
    assert len(gather_sizes) > 2 and max(gather_sizes) <= 23
    torch.testing.assert_close(actual, baseline, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(actual_valid, baseline_valid)


def test_nearest_adjacent_half_pixel_boundary_does_not_index_out_of_bounds():
    rows = np.array([np.nextafter(0.5, 0.0), 0.5, np.nextafter(-0.5, -1.0), -0.5])
    cols = np.zeros(rows.shape)
    expected, expected_valid = interpolate_reference(
        np.array([[7.0]]), rows, cols, method="nearest"
    )
    actual, valid = _interpolate(
        np.array([[7.0]]), rows, cols, method="nearest", return_valid=True
    )
    np.testing.assert_allclose(actual.numpy(), expected)
    np.testing.assert_array_equal(valid.numpy(), expected_valid)


def test_bilinear_underflow_does_not_discard_nonzero_contributing_support():
    source = np.array([[1.0, 2.0], [3.0, np.nan]])
    tiny = np.nextafter(0.0, 1.0)
    result, valid = _interpolate(
        source, np.array(tiny), np.array(tiny), return_valid=True
    )
    assert torch.isnan(result) and not valid


def test_sparse_tensor_and_non_cpu_cuda_coordinates_are_rejected():
    sparse = torch.ones((8, 8)).to_sparse()
    with pytest.raises(TypeError, match="dense|strided"):
        _interpolate(sparse, np.array(2.0), np.array(2.0))
    with pytest.raises(ValueError, match="CPU|CUDA|device"):
        _interpolate(np.ones((8, 8)), torch.empty(1, device="meta"), np.array(2.0))


def test_impractical_shapes_fail_before_numpy_transfer():
    huge_source = np.broadcast_to(np.ones((1, 1)), (2**20, 2**20))
    with pytest.raises(ValueError, match="size|bytes|limit|large"):
        _interpolate(huge_source, np.array(1.0), np.array(1.0))
    huge_query = np.broadcast_to(np.array(1.0), (2**32,))
    with pytest.raises(ValueError, match="size|bytes|limit|large"):
        _interpolate(np.ones((8, 8)), huge_query, np.array(1.0))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_tensor_rejected_before_float64_work():
    with pytest.raises((TypeError, ValueError), match="cpu|CPU|CUDA|mps|MPS"):
        _interpolate(torch.ones((8, 8), device="mps"), np.array(2.0), np.array(2.0))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA qualification not run: unavailable"
)
@pytest.mark.parametrize("method", ["nearest", "bilinear", "biquintic"])
def test_cuda_same_device_numpy_companions_and_cpu_tensor_mixing(method):
    source = torch.arange(144, dtype=torch.float32, device="cuda").reshape(12, 12)
    rows, cols = np.array([1.2, 4.5]), np.array([2.3, 5.5])
    expected, expected_valid = interpolate_reference(
        source.cpu().numpy(), rows, cols, method=method
    )
    actual, valid = _interpolate(source, rows, cols, method=method, return_valid=True)
    assert actual.device == valid.device == source.device
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-9, rtol=0)
    np.testing.assert_array_equal(valid.cpu().numpy(), expected_valid)
    with pytest.raises(ValueError, match="device"):
        _interpolate(source, torch.from_numpy(rows), cols, method=method)
