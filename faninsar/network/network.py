"""Canonical Network reader/open seam backed by the existing data loader."""

from __future__ import annotations

# ``faninsar.datasets.network.Network`` remains the sole concrete data-backed
# implementation during the compatibility wave.  Re-exporting the exact class
# avoids a second Network model while exposing the new owning package path.
from faninsar.datasets.network import Network

__all__ = ["Network"]
