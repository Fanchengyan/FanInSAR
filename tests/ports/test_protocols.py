"""Structural Protocol checks for the three ports."""

from __future__ import annotations

from typing import Any

import numpy as np

from faninsar.ports.compute import ComputeBackend
from faninsar.ports.io import IOBackend, Store
from faninsar.ports.sensor import SensorAdapter


class _DummyCompute:
    name = "dummy"

    def to_device_array(self, x: np.ndarray) -> np.ndarray:
        return x

    def map_blocks(self, fn, *args, chunks=None, dtype=None, resources=None):
        return fn(*args)

    def compute(self, *arrays, sync=True):
        return tuple(np.asarray(a) for a in arrays)


class _DummySensor:
    name = "dummy"

    def open_product(self, uri: str, **kwargs: Any) -> str:
        return uri

    def to_slc_product(self, handle: Any, **kwargs: Any) -> Any:
        return handle

    def read_slc_window(self, handle: Any, window: Any, **kwargs: Any) -> np.ndarray:
        return np.zeros((1, 1), dtype=np.complex64)


class _DummyStore:
    uri = "memory://"

    def exists(self, key: str = "") -> bool:
        return True


class _DummyIO:
    def open_store(self, uri: str, **kwargs: Any) -> Store:
        return _DummyStore()

    def write(self, product: Any, uri: str, *, format: str = "cog") -> None:
        return None

    def read(self, uri: str, **kwargs: Any) -> Any:
        return None


def test_compute_backend_protocol() -> None:
    assert isinstance(_DummyCompute(), ComputeBackend)


def test_sensor_adapter_protocol() -> None:
    assert isinstance(_DummySensor(), SensorAdapter)


def test_io_backend_protocol() -> None:
    assert isinstance(_DummyIO(), IOBackend)


def test_missing_method_not_protocol() -> None:
    class Incomplete:
        name = "x"

        def to_device_array(self, x):
            return x

    assert not isinstance(Incomplete(), ComputeBackend)
