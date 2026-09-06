"""URI-based I/O, STAC at product boundaries, dataset STAC emitters."""

from __future__ import annotations

from faninsar.io.protocols import Catalog, Format, IOBackend, Store
from faninsar.io.readers import open_stac, open_zarr

__all__ = ["Catalog", "Format", "IOBackend", "Store", "open_stac", "open_zarr"]
