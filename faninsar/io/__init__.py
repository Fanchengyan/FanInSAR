"""URI-based I/O, STAC at product boundaries, dataset STAC emitters."""

from __future__ import annotations

from faninsar.io.cog import (
    CogExportItem,
    CogExportReport,
    ExportRequest,
    export_cog,
    export_identity,
    export_label,
)
from faninsar.io.protocols import Catalog, Format, IOBackend, Store
from faninsar.io.raster import (
    RasterGridSpec,
    read_geogrid,
    validate_cog,
    write_cog,
)
from faninsar.io.readers import open_stac, open_zarr

__all__ = [
    "Catalog",
    "CogExportItem",
    "CogExportReport",
    "ExportRequest",
    "Format",
    "IOBackend",
    "RasterGridSpec",
    "Store",
    "export_cog",
    "export_identity",
    "export_label",
    "open_stac",
    "open_zarr",
    "read_geogrid",
    "validate_cog",
    "write_cog",
]
