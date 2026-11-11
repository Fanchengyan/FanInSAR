"""Independent reference and public block-reduction contract tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from .reference.reduction import block_reference


def test_independent_reference_has_hand_computed_partial_blocks():
    source = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
    values, valid = block_reference(source, (2, 2), invalid="omit", remainder="partial")
    np.testing.assert_allclose(values, [[7 / 3, 4.5], [7.5, 9.0]])
    assert valid.all()


def test_mean_partial_omits_missing_members_independently_per_batch():
    from faninsar.core.algorithms.reduction import block_reduce2d

    source = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]],
            [[-1.0, 2.0, np.inf], [8.0, 5.0, 6.0], [0.0, 3.0, 4.0]],
        ]
    )
    actual, actual_valid = block_reduce2d(
        source, (2, 2), invalid="omit", remainder="partial", return_valid=True
    )
    expected, expected_valid = block_reference(
        source, (2, 2), invalid="omit", remainder="partial"
    )
    assert isinstance(actual, torch.Tensor)
    assert actual.dtype == torch.float64
    assert actual.device.type == "cpu"
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), atol=1e-10, rtol=1e-12
    )
    np.testing.assert_array_equal(actual_valid, expected_valid)


@pytest.mark.parametrize("method", ["mean", "sum", "min", "max"])
@pytest.mark.parametrize("invalid", ["propagate", "omit"])
@pytest.mark.parametrize("remainder", ["drop", "partial"])
def test_all_reductions_preserve_masks_and_tail_admission(method, invalid, remainder):
    from faninsar.core.algorithms import block_reduce2d

    data = np.arange(2 * 5 * 7, dtype=np.float64).reshape(2, 5, 7)
    data[0, 0, 0] = np.nan
    data[0, 1, 3] = np.inf
    source = np.ma.array(data, mask=False)
    source.mask[1, :2, :3] = True
    source.mask[0, 4, 6] = True
    valid = np.ones((5, 7), dtype=bool)
    valid[2, 2] = False
    before = source.copy()
    expected, supported = block_reference(
        source, (2, 3), method=method, valid=valid, invalid=invalid, remainder=remainder
    )
    actual, actual_valid = block_reduce2d(
        source,
        (2, 3),
        method=method,
        valid=valid,
        invalid=invalid,
        remainder=remainder,
        return_valid=True,
    )
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), equal_nan=True, atol=1e-10, rtol=1e-12
    )
    np.testing.assert_array_equal(actual_valid, supported)
    np.testing.assert_array_equal(source.data, before.data)
    np.testing.assert_array_equal(source.mask, before.mask)


@pytest.mark.parametrize("method", ["mean", "sum", "min", "max"])
def test_all_invalid_blocks_are_nan_even_for_sum(method):
    from faninsar.core.algorithms import block_reduce2d

    for source in (
        np.full((2, 2), np.nan),
        np.ma.masked_all((2, 2)),
        np.ma.array(np.ones((2, 2)), mask=True),
    ):
        actual, valid = block_reduce2d(
            source, (2, 2), method=method, invalid="omit", return_valid=True
        )
        assert torch.isnan(actual).all()
        assert not valid.any()


@pytest.mark.parametrize("shape", [(0, 3, 4), (2, 0, 4), (2, 3, 0), (2, 1, 1)])
def test_empty_batch_and_dropped_spatial_outputs_are_well_shaped(shape):
    from faninsar.core.algorithms import block_reduce2d

    source = np.empty(shape, dtype=np.float32)
    actual, valid = block_reduce2d(source, (2, 3), return_valid=True)
    assert actual.shape == (*shape[:-2], shape[-2] // 2, shape[-1] // 3)
    assert valid.shape == actual.shape
    assert actual.dtype == torch.float64
    assert valid.dtype == torch.bool


def test_noncontiguous_leading_batches_and_masked_valid_are_preserved():
    from faninsar.core.algorithms import block_reduce2d

    source = torch.arange(96, dtype=torch.float32).reshape(2, 2, 4, 6).transpose(-1, -2)
    valid = np.ma.array(np.ones(source.shape, dtype=bool), mask=False)
    valid.mask[0, 1, :2, :2] = True
    before = source.clone()
    expected, support = block_reference(
        source.numpy(), (2, 2), valid=valid, invalid="omit", remainder="error"
    )
    actual, actual_valid = block_reduce2d(
        source,
        (2, 2),
        valid=valid,
        invalid="omit",
        remainder="error",
        return_valid=True,
    )
    torch.testing.assert_close(actual, torch.from_numpy(expected), equal_nan=True)
    np.testing.assert_array_equal(actual_valid, support)
    torch.testing.assert_close(source, before)


@pytest.mark.parametrize("method", ["mean", "sum", "min", "max"])
def test_vectorized_reduction_crosses_bounded_tiles_and_retains_partial_tails(method):
    from faninsar.core.algorithms import block_reduce2d

    source = np.ma.array(
        np.arange(513 * 517, dtype=np.float64).reshape(513, 517) / 100,
        mask=False,
    )
    source.mask[500:505, 490:497] = True
    source.data[400, 500] = np.nan
    valid = np.ones(source.shape, dtype=bool)
    valid[510:, 510:] = False
    expected, support = block_reference(
        source, (5, 7), method=method, valid=valid, invalid="omit", remainder="partial"
    )
    actual, admission = block_reduce2d(
        source,
        (5, 7),
        method=method,
        valid=valid,
        invalid="omit",
        remainder="partial",
        return_valid=True,
    )
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), equal_nan=True, atol=1e-10, rtol=1e-12
    )
    np.testing.assert_array_equal(admission, support)
    values_only = block_reduce2d(
        source, (5, 7), method=method, valid=valid, invalid="omit", remainder="partial"
    )
    torch.testing.assert_close(values_only, actual, equal_nan=True)


@pytest.mark.parametrize("block", [(0, 2), (-1, 2), (True, 2), (1.5, 2), (2,), 2])
def test_invalid_block_shape_rejected(block):
    from faninsar.core.algorithms import block_reduce2d

    with pytest.raises((ValueError, TypeError)):
        block_reduce2d(np.ones((3, 4)), block)


@pytest.mark.parametrize(
    "options",
    [
        dict(method="median"),
        dict(invalid="ignore"),
        dict(remainder="pad"),
        dict(remainder="error"),
        dict(valid=np.ones((3, 4))),
        dict(valid=np.ones((1, 4), dtype=bool)),
    ],
)
def test_invalid_options_and_ambiguous_masks_rejected(options):
    from faninsar.core.algorithms import block_reduce2d

    with pytest.raises((ValueError, TypeError)):
        block_reduce2d(np.ones((3, 4)), (2, 2), **options)


@pytest.mark.parametrize(
    "source",
    [
        np.ones((2, 2), dtype=int),
        np.ones((2, 2), dtype=complex),
        np.ones((2, 2), dtype=bool),
        np.ones(3),
        [[1.0, 2.0]],
        torch.ones((2, 2), requires_grad=True),
        torch.ones((2, 2), dtype=torch.float16),
        torch.eye(2).to_sparse(),
        torch.empty((2, 2), device="meta"),
    ],
)
def test_unsupported_source_types_fail_explicitly(source):
    from faninsar.core.algorithms import block_reduce2d

    with pytest.raises((ValueError, TypeError)):
        block_reduce2d(source, (1, 1))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware unavailable; not qualified"
)
@pytest.mark.parametrize("method", ["mean", "sum", "min", "max"])
def test_cuda_reduction_matches_cpu_oracle_without_device_fallback(method):
    from faninsar.core.algorithms import block_reduce2d

    source = np.arange(30, dtype=np.float64).reshape(5, 6)
    source[1, 1] = np.nan
    expected, supported = block_reference(
        source, (2, 3), method=method, invalid="omit", remainder="partial"
    )
    actual, valid = block_reduce2d(
        torch.from_numpy(source).cuda(),
        (2, 3),
        method=method,
        invalid="omit",
        remainder="partial",
        return_valid=True,
    )
    assert actual.device.type == "cuda"
    assert valid.device == actual.device
    torch.testing.assert_close(
        actual.cpu(), torch.from_numpy(expected), equal_nan=True, atol=1e-10, rtol=1e-12
    )
    np.testing.assert_array_equal(valid.cpu(), supported)
    with pytest.raises(ValueError, match="device"):
        block_reduce2d(
            torch.from_numpy(source).cuda(),
            (2, 3),
            valid=torch.ones(source.shape, dtype=torch.bool),
        )
