"""ComputeBackend implementations."""

from __future__ import annotations

import numpy as np

from faninsar.compute.dask_torch import DaskTorchBackend
from faninsar.compute.numpy_backend import NumpyBackend
from faninsar.ports.compute import ComputeBackend


def test_numpy_backend_protocol() -> None:
    backend = NumpyBackend()
    assert isinstance(backend, ComputeBackend)
    x = np.arange(4.0)
    y = backend.to_device_array(x)
    out = backend.map_blocks(lambda a: a * 2, y)
    (result,) = backend.compute(out)
    np.testing.assert_array_equal(result, x * 2)


def test_dask_torch_backend_protocol() -> None:
    backend = DaskTorchBackend()
    assert isinstance(backend, ComputeBackend)
    x = np.ones((8, 8))
    y = backend.to_device_array(x)
    out = backend.map_blocks(lambda a: a + 1, y)
    (result,) = backend.compute(out)
    np.testing.assert_array_equal(result, x + 1)
