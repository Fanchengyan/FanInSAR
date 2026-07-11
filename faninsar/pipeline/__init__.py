"""Orchestration pipelines that consume Frame products.

Currently provides :class:`InversionPipeline`, which runs NSBAS time-series
inversion on a :class:`~faninsar.datasets.frame.Frame` via Dask spatial
tiling and (optionally) GPU acceleration.
"""

from __future__ import annotations

from faninsar.pipeline.inversion import InversionPipeline

__all__ = ["InversionPipeline"]
