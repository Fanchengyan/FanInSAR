"""Fetch facade: complete-object download and async byte-range transfer."""

from __future__ import annotations

from .complete import download, probe
from .ranges import RangeFetcher, RangeSpan, fetch_ranges_async

__all__ = [
    "RangeFetcher",
    "RangeSpan",
    "download",
    "fetch_ranges_async",
    "probe",
]
