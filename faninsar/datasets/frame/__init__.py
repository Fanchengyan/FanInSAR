"""FanInSAR frame-level InSAR products.

This subpackage provides standardized frame-level geometry and interferogram
assets built on top of existing FanInSAR geospatial primitives.

Scope
-----

**Frame** targets **geocoded raster InSAR** products — the per-pair
geocoded rasters (unwrapped phase, coherence, etc.) produced by processors
such as ISCE2, GAMMA, GMTSAR, HyP3, ARIA, and MintPy.

The central abstraction is the :class:`Frame`, a layered data container:

- **COG for archive** — per-pair Cloud-Optimized GeoTIFFs (the default
  on-disk output, STAC-compatible, efficient HTTP range reads via
  ``/vsicurl/``).
- **Zarr for compute** — persistent ``(pair, y, x)`` interferogram cubes and
  ``(time, y, x)`` displacement time series, written via
  :meth:`FrameInterferogramCollection.to_zarr_stack` and
  :meth:`Frame.to_zarr` (direct S3/GCS writes via ``fsspec`` FSStore, no
  GDAL required).
- **STAC for discovery** — :meth:`Frame.to_stac` emits a catalog with COG
  assets, Zarr assets (``stac-extensions/zarr`` v1.1.0), and InSAR fields
  (``stac-extensions/insar`` v1.0.0).
- **Multi-engine reading** — COG (rioxarray), Zarr (:func:`xarray.open_zarr`),
  and HDF5 (:func:`xarray.open_dataset` with ``h5netcdf``) via a unified
  xarray interface ("read anything, write standard").

Out of scope
------------

The following are explicitly **not** supported by Frame:

- **StaMPS point-cloud PS** — PS data is a point cloud, not a raster; a
  categorically different data model. A separate ``FramePointSeries``
  abstraction would be required.
- **Non-geocoded products (raw SLC)** — Frame assumes geocoded rasters.
- **``FrameGeometry.from_hyp3`` azimuth/heading** — raises
  ``NotImplementedError``; requires a non-trivial ``lv_phi`` → look-azimuth
  convention converter (see :class:`FrameGeometry`).

``Frame.from_mintpy`` (reverse of :meth:`Frame.to_mintpy`) **is** supported:
it reads MintPy ``ifgramStack.h5`` / ``geometryGeo.h5`` via xarray's
h5netcdf engine and produces the standard Frame layout with Zarr cubes —
no per-pair COG materialization is needed.

Orchestration (NSBAS time-series inversion) lives in
:mod:`faninsar.pipeline`, not on the Frame itself. InSAR-specific plotting
lives in :mod:`faninsar.plots.frame`.
"""

from __future__ import annotations

from .exceptions import (
    COGValidationError,
    FrameGeometryError,
    FrameInterferogramError,
    FrameProductError,
    GridMismatchError,
    MetadataError,
    MissingGeometryAssetError,
    MissingInterferogramAssetError,
    PairNotFoundError,
)
from .frame import Frame
from .geometry import FrameGeometry
from .interferogram import FrameInterferogramCollection
from .metadata import (
    GEOMETRY_ASSETS,
    INTERFEROGRAM_ASSETS,
    GeometryAssetName,
    InterferogramAssetName,
)
from .mintpy import build_mintpy
from .publish import build_upload_plan, iter_frame_assets, publish_to_huggingface
from .remote import RemoteFrame
from .timeseries import FrameTimeSeries

__all__ = [
    "GEOMETRY_ASSETS",
    "INTERFEROGRAM_ASSETS",
    "COGValidationError",
    "Frame",
    "FrameGeometry",
    "FrameGeometryError",
    "FrameInterferogramCollection",
    "FrameInterferogramError",
    "FrameProductError",
    "FrameTimeSeries",
    "GeometryAssetName",
    "GridMismatchError",
    "InterferogramAssetName",
    "MetadataError",
    "MissingGeometryAssetError",
    "MissingInterferogramAssetError",
    "PairNotFoundError",
    "RemoteFrame",
    "build_mintpy",
    "build_upload_plan",
    "iter_frame_assets",
    "publish_to_huggingface",
]
