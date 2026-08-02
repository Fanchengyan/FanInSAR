"""Time-series orchestration pipelines.

InversionPipeline previously lived under ``faninsar.pipeline``.  The greenfield
home is ``timeseries`` (Frame → InterferogramStack → NSBAS).  The top-level
``faninsar.pipeline`` package remains a thin re-export for existing tests.
"""

from __future__ import annotations

# Implementation still in faninsar.pipeline.inversion until Frame loader is
# fully rehomed out of datasets/frame; re-export here as the public timeseries path.
from faninsar.pipeline.inversion import InversionPipeline

__all__ = ["InversionPipeline"]
