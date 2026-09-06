"""Sentinel-1 measurement I/O package.

Submodules:

- :mod:`faninsar.missions.s1.io.read`   — rasterio-window full-burst reads
- :mod:`faninsar.missions.s1.io.extract` — byte-offset fast burst extraction
- :mod:`faninsar.missions.s1.io.remote`  — remote (ASF) burst extraction via ``requests``
- :mod:`faninsar.missions.s1.io.export`  — SAFE subset directory export

Public names are re-exported here so existing imports
(``from faninsar.missions.s1.io import read_full_burst``) keep working
after the ``io.py`` -> ``io/`` package upgrade.
"""

from __future__ import annotations

from faninsar.missions.s1.io.export import (
    export_burst_safe,
    export_swath_bursts,
)
from faninsar.missions.s1.io.extract import (
    estimate_burst_bytes,
    extract_burst,
    extract_bursts,
)
from faninsar.missions.s1.io.read import (
    BurstArray,
    read_burst_window,
    read_full_burst,
    read_swath_bursts,
    stitch_bursts,
)
from faninsar.missions.s1.io.remote import (
    RemoteSafe,
    extract_remote_burst,
)

__all__ = [
    "BurstArray",
    "RemoteSafe",
    "estimate_burst_bytes",
    "export_burst_safe",
    "export_swath_bursts",
    "extract_burst",
    "extract_bursts",
    "extract_remote_burst",
    "read_burst_window",
    "read_full_burst",
    "read_swath_bursts",
    "stitch_bursts",
]
