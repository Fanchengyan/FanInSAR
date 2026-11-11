"""Input admission and allocation ordering for the shared P0087 adapter."""

from __future__ import annotations

import numpy as np
import pytest
import torch


def test_masked_float32_metadata_and_tensor_adapter_preserve_input():
    """Native invalidity survives preflight without promoting resident storage."""
    from faninsar.core.algorithms._validation import (
        _as_tensor,
        _checked_size,
        _execution_device,
        _inspect_array,
    )

    source = np.ma.array(
        [[1.0, 17.0], [3.0, np.inf]],
        dtype=np.float32,
        mask=[[False, True], [False, False]],
    )
    original = source.copy()
    info = _inspect_array(source, name="source")
    assert info.shape == (2, 2)
    assert info.dtype == np.dtype("float32")
    assert np.shares_memory(info.data, source.data)
    assert np.shares_memory(info.invalid, source.mask)
    assert _checked_size(info.shape, itemsize=8, name="output", limit_bytes=32) == 32
    device = _execution_device(source)
    assert device == torch.device("cpu")
    values, invalid = _as_tensor(source, device=device, name="source")
    assert values.dtype == torch.float32
    assert values.device == device
    assert values.tolist() == [[1.0, 17.0], [3.0, float("inf")]]
    assert invalid.dtype == torch.bool
    assert invalid.tolist() == [[False, True], [False, False]]
    np.testing.assert_array_equal(source.data, original.data)
    np.testing.assert_array_equal(source.mask, original.mask)


@pytest.mark.parametrize("layout", ["reversed", "transposed", "readonly", "endian"])
def test_numpy_storage_layout_adaptation_preserves_float32_and_mask(layout):
    """Container-only copies happen after preflight, without lost native masks."""
    from faninsar.core.algorithms._validation import _as_tensor, _inspect_array

    data = np.ma.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float32,
        mask=[[False, True, False], [True, False, False]],
    )
    if layout == "reversed":
        data = data[:, ::-1]
    elif layout == "transposed":
        data = data.T
    elif layout == "readonly":
        data.flags.writeable = False
        data._mask.flags.writeable = False
    else:
        data = data.astype(np.dtype(">f4" if np.little_endian else "<f4"))
    info = _inspect_array(data, name="source")
    values, invalid = _as_tensor(data, device=torch.device("cpu"), name="source")
    assert values.dtype == torch.float32
    assert invalid.dtype == torch.bool and invalid.device == values.device
    assert values.tolist() == data.data.tolist()
    assert invalid.tolist() == data.mask.tolist()
    assert info.shape == data.shape


def test_native_scalar_and_broadcast_masks_remain_small_storage():
    """Large image masks do not require materializing their broadcast footprint."""
    from faninsar.core.algorithms._validation import _as_tensor

    source = np.ma.array(np.ones((100, 2), dtype=np.float32), mask=np.ma.nomask)
    _, invalid = _as_tensor(source, device=torch.device("cpu"), name="source")
    assert invalid is False
    source._mask = np.bool_(True)
    _, invalid = _as_tensor(source, device=torch.device("cpu"), name="source")
    assert invalid is True
    source._mask = np.broadcast_to(np.array([[False, True]]), source.shape)
    _, invalid = _as_tensor(source, device=torch.device("cpu"), name="source")
    assert invalid.numel() == 2
    assert invalid.tolist() == [[False, True]]
    assert invalid.device == torch.device("cpu")


@pytest.mark.parametrize(
    "shape,itemsize,limit",
    [
        ((True, 2), 8, 16),
        ((2,), True, 16),
        ((2,), 8, True),
    ],
)
def test_allocation_metadata_rejects_boolean_dimensions_and_sizes(
    shape, itemsize, limit
):
    from faninsar.core.algorithms._validation import _checked_size

    with pytest.raises(ValueError, match="integer"):
        _checked_size(shape, itemsize=itemsize, name="output", limit_bytes=limit)


@pytest.mark.parametrize(
    "source",
    [
        np.array([1], dtype=np.int64),
        np.array([True]),
        np.array([1j]),
        np.array([1.0], dtype=np.float16),
        [1.0, 2.0],
        torch.ones(2, dtype=torch.int32),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.complex64),
        torch.ones(2, dtype=torch.float16),
    ],
)
def test_numerical_inputs_reject_unadmitted_container_or_dtype(source):
    from faninsar.core.algorithms._validation import _inspect_array

    with pytest.raises(TypeError, match="source"):
        _inspect_array(source, name="source")


def test_gradient_and_sparse_inputs_fail_before_execution():
    from faninsar.core.algorithms._validation import _inspect_array

    with pytest.raises(ValueError, match="gradients"):
        _inspect_array(torch.ones(2, requires_grad=True), name="source")
    with pytest.raises(TypeError, match="dense"):
        _inspect_array(torch.eye(2).to_sparse(), name="source")


def test_boolean_masks_require_explicit_admission_flag():
    from faninsar.core.algorithms._validation import _as_tensor, _inspect_array

    for value in (np.array([True, False]), torch.tensor([True, False])):
        info = _inspect_array(value, name="valid", allow_bool=True)
        assert info.shape == (2,)
        mask, native = _as_tensor(
            value, name="valid", device=torch.device("cpu"), allow_bool=True
        )
        assert mask.dtype == torch.bool
        assert native is None
    assert _inspect_array(
        np.ones(2, dtype=np.float32), name="valid", allow_bool=True
    ).dtype == np.dtype("float32")


def test_tensor_storage_is_not_copied_or_promoted():
    from faninsar.core.algorithms._validation import _as_tensor, _inspect_array

    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).T
    info = _inspect_array(source, name="source")
    assert info.data is source
    values, invalid = _as_tensor(source, device=source.device, name="source")
    assert values is source and invalid is None
    assert values.dtype == torch.float32


def test_metadata_preflight_does_not_check_finiteness_or_convert(monkeypatch):
    from faninsar.core.algorithms._validation import _checked_size, _inspect_array

    def unexpected(*args, **kwargs):
        raise AssertionError("Numerical work happened during metadata inspection")

    source = np.array([np.nan, np.inf], dtype=np.float32)
    monkeypatch.setattr(np, "isfinite", unexpected)
    monkeypatch.setattr(torch, "isfinite", unexpected)
    monkeypatch.setattr(torch, "as_tensor", unexpected)
    info = _inspect_array(source, name="source")
    assert info.shape == (2,) and info.invalid is None
    with pytest.raises(ValueError, match="output"):
        _checked_size((2**40, 2**40), itemsize=8, name="output", limit_bytes=2**30)
    with pytest.raises(ValueError, match="output"):
        _checked_size((-1,), itemsize=8, name="output", limit_bytes=16)
    assert _checked_size((0, 100), itemsize=8, name="empty", limit_bytes=0) == 0
    assert _checked_size((), itemsize=8, name="scalar", limit_bytes=8) == 8


def test_non_cpu_cuda_placement_fails_without_computation():
    from faninsar.core.algorithms._validation import _as_tensor, _execution_device

    source = torch.empty(2, device="meta")
    with pytest.raises(ValueError, match="CPU or CUDA"):
        _execution_device(np.ones(2, dtype=np.float32), source)
    with pytest.raises(ValueError, match="concrete CPU/CUDA"):
        _as_tensor(np.ones(2, dtype=np.float32), name="source", device="mps")
    with pytest.raises(ValueError, match="explicit"):
        _as_tensor(source, name="source", device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_all_tensor_arguments_including_masks_and_weights_fix_placement():
    from faninsar.core.algorithms._validation import _as_tensor, _execution_device

    source = torch.ones(2, dtype=torch.float32, device="cuda:0")
    valid = torch.ones(2, dtype=torch.bool)
    weights = torch.ones(2)
    assert _execution_device(np.ones(2, dtype=np.float32), source) == source.device
    with pytest.raises(ValueError, match="one device"):
        _execution_device(source, valid)
    with pytest.raises(ValueError, match="one device"):
        _execution_device(source, weights)
    value, invalid = _as_tensor(
        np.ma.array([1.0, 2.0], mask=[False, True]), name="source", device=source.device
    )
    assert value.device == invalid.device == source.device
