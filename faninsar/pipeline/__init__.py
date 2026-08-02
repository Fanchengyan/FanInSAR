"""Thin re-export of time-series orchestration pipelines.

.. deprecated::
    Prefer :mod:`faninsar.timeseries.pipeline`.  This package remains for
    import stability during the greenfield dual-namespace collapse.
"""

from __future__ import annotations

from faninsar.pipeline.inversion import InversionPipeline

__all__ = ["InversionPipeline"]
