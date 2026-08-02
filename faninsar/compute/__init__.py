"""Compute backends and torch.compile infrastructure."""

from __future__ import annotations

from faninsar.compute.dask_torch import DaskTorchBackend
from faninsar.compute.numpy_backend import NumpyBackend

__all__ = ["DaskTorchBackend", "NumpyBackend"]
