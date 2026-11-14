"""Windowed, lazily warped radar-domain views over internal Zarr stores.

A :class:`RadarGeolocationSource` describes one persisted scalar product and
the dense longitude/latitude geolocation arrays that place its pixels on the
WGS84 ellipsoid. :func:`open_lazy_radar_view` turns that description into a
:class:`LazyRadarView` whose data is warped by GDAL only when a window is
read. Initialization reads metadata and geolocation arrays, never the
scientific value chunks.

The implementation deliberately follows the proven in-memory workflow: a
VRT with ``GEOLOCATION`` metadata, an automatically gridded
:class:`rasterio.vrt.WarpedVRT`, a serialized VRT copied into ``/vsimem``,
and a reopened dataset whose warp block size is rewritten. No external GDAL
command, ``ctypes`` call, or temporary-file-on-disk step is required.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np
import rasterio
import rasterio.shutil
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rasterio.io import DatasetReader

__all__ = [
    "LazyGeoViewError",
    "LazyRadarView",
    "RadarGeolocationSource",
    "open_lazy_radar_view",
]

logger = setup_logger(__name__)

_CRS: Final[str] = "EPSG:4326"
_GDAL_CACHE_BYTES: Final[int] = 64 * 1024 * 1024
_WARP_MEMORY_MB: Final[int] = 64
_MAX_EXTENT_DEGREES: Final[float] = 180.0
_MAX_ABSOLUTE_LATITUDE: Final[float] = 89.0
_GEOMETRY_EPSILON: Final[float] = 1e-9

# GDAL's geolocation SRS metadata must be a full WKT string. An ``EPSG:4326``
# shorthand opens, but serializing the automatically gridded warp VRT fails
# with an unhelpful "missing [" error.
_SRS_WKT: Final[str] = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
    '298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],'
    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AXIS["Latitude",NORTH],AXIS["Longitude",EAST],'
    'AUTHORITY["EPSG","4326"]]'
)

_GDAL_DATA_TYPES: Final[dict[str, str]] = {
    "float32": "Float32",
    "float64": "Float64",
}

#: Scientific assets that cannot be served by the scalar bilinear backend.
_UNSUPPORTED_KINDS: Final[frozenset[str]] = frozenset(
    {
        "class",
        "classification",
        "classes",
        "complex",
        "integer",
        "label",
        "labels",
        "mask",
        "slc",
        "wrapped",
        "wrapped-phase",
        "wrapped_phase",
        "wrapped_phase_angle",
        "wrapped-phase-angle",
    }
)

_PLACEHOLDER_VRT: Final[bytes] = (
    b'<VRTDataset rasterXSize="1" rasterYSize="1">'
    b'<VRTRasterBand dataType="Byte" band="1"/>'
    b"</VRTDataset>"
)


class LazyGeoViewError(ValueError):
    """Rejected lazy radar view request with a stable machine reason.

    Parameters
    ----------
    reason : str
        Short identifier such as ``"geolocation_too_large"``.
    detail : str
        Human-readable diagnostic explaining the rejection.

    Attributes
    ----------
    reason : str
        Short machine-readable rejection reason.
    detail : str
        Human-readable rejection detail.

    """

    def __init__(self, reason: str, detail: str) -> None:
        """Store the rejection reason and detail."""
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True, slots=True)
class RadarGeolocationSource:
    """Persisted scalar product with dense WGS84 geolocation arrays.

    The value store has the product shape. The longitude and latitude stores
    hold the dense full-reference geolocation map: native pixel centres are
    indexed by integer coordinates ``0 .. geolocation_shape - 1``. Product
    pixel ``(row, column)`` maps to full-map coordinate
    ``first_center + (row, column) * step``.

    Parameters
    ----------
    values_uri : str
        Internal GDAL dataset path for the scalar value array. The HTTP
        adapter supplies a validated native Zarr source, for example
        ``ZARR:"/vsicurl/http://127.0.0.1:8000/assets/<id>/values/"``.
    longitude_uri : str
        Internal GDAL dataset path for the full-reference longitude array.
    latitude_uri : str
        Internal GDAL dataset path for the full-reference latitude array.
    shape : tuple[int, int]
        Product shape as ``(height, width)``.
    geolocation_shape : tuple[int, int]
        Full geolocation map shape as ``(height, width)``.
    data_type : str
        Scalar data type; ``"float32"`` and ``"float64"`` are supported.
    first_center : tuple[float, float], default=(0, 0)
        Full-map centre coordinate of product pixel ``(0, 0)`` expressed as
        ``(row, column)``.
    step : tuple[float, float], default=(1, 1)
        Product spacing in full-map pixels as ``(row, column)``.
    kind : str, default="coherence"
        Scientific kind used to reject label, complex, and wrapped-phase
        assets that the scalar backend cannot serve.
    valid_uri : str | None, default=None
        Optional internal GDAL dataset path for a validity array on the
        product grid. Non-zero values are valid; zero values are masked.
    source_fingerprint : str, default=""
        Adapter-provided identity used by callers for caching.

    """

    values_uri: str
    longitude_uri: str
    latitude_uri: str
    shape: tuple[int, int]
    geolocation_shape: tuple[int, int]
    data_type: str
    first_center: tuple[float, float] = (0, 0)
    step: tuple[float, float] = (1, 1)
    kind: str = "coherence"
    valid_uri: str | None = None
    source_fingerprint: str = ""


class LazyRadarView:
    """Lazily warped WGS84 view over one scalar radar product.

    Instances are returned by :func:`open_lazy_radar_view`. All internal
    GDAL owners (memory files, source dataset, automatic warp, serialized
    copy, and reopened dataset) remain alive until :meth:`close`. Reads and
    close are serialized by one lock.

    Parameters
    ----------
    source : RadarGeolocationSource
        The supplied product description.
    dataset : DatasetReader
        Reopened serialized warp dataset.
    owners : tuple[object, ...]
        GDAL objects and memory files that must stay alive.

    Attributes
    ----------
    shape : tuple[int, int]
        Warped view shape as ``(height, width)``.
    width, height : int
        Warped view dimensions.
    transform : tuple[float, ...]
        Six-coefficient WGS84 affine transform
        ``(a, b, c, d, e, f)``.
    crs : str
        Always ``"EPSG:4326"``.
    data_type : str
        Preserved scalar type, ``"float32"`` or ``"float64"``.
    fill_value : float
        Invalid pixels are filled with ``nan``.
    chunk_shape : tuple[int, int]
        Internal read block shape as ``(height, width)``.
    source_fingerprint : str
        Adapter-provided source identity for cache invalidation.

    """

    def __init__(
        self,
        source: RadarGeolocationSource,
        dataset: DatasetReader,
        owners: tuple[object, ...],
        *,
        data_type: str,
        masked: bool,
    ) -> None:
        """Initialize the view from an open warp dataset and its owners."""
        self.source = source
        self.source_fingerprint = source.source_fingerprint
        self.shape = (int(dataset.height), int(dataset.width))
        self.height = self.shape[0]
        self.width = self.shape[1]
        transform = dataset.transform
        self.transform = (
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        )
        self.crs = _CRS
        self.data_type = data_type
        self.fill_value = float("nan")
        self.chunk_shape = (
            min(int(dataset.block_shapes[0][0]), self.height),
            min(int(dataset.block_shapes[0][1]), self.width),
        )
        self._dataset = dataset
        self._owners = owners
        self._masked = masked
        self._lock = threading.Lock()
        self._closed = False

    def read_window(
        self,
        window: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Read one half-open pixel window and fill invalid pixels with NaN.

        Parameters
        ----------
        window : tuple[int, int, int, int]
            Window as ``(row_start, row_stop, column_start, column_stop)``,
            with the stop values exclusive.

        Returns
        -------
        numpy.ndarray
            Two-dimensional array with shape
            ``(row_stop - row_start, column_stop - column_start)`` and the
            product's preserved floating-point data type. Masked or
            unobserved pixels are ``nan``. A valid coherence value of zero
            remains zero and is never treated as nodata.

        Raises
        ------
        LazyGeoViewError
            If the view is closed, the window is malformed, or the window
            lies outside the view.

        """
        row_start, row_stop, col_start, col_stop = _validate_window(window, self.shape)
        raster_window = Window(
            col_start,
            row_start,
            col_stop - col_start,
            row_stop - row_start,
        )
        with self._lock:
            if self._closed:
                _reject("view_closed", "the lazy radar view is already closed")
            values = self._dataset.read(1, window=raster_window)
            if self._masked:
                alpha = self._dataset.read(2, window=raster_window)
                invalid = alpha < 1.0
                if bool(invalid.any()):
                    values = np.array(values, copy=True)
                    values[invalid] = self.fill_value
            return values

    def close(self) -> None:
        """Release every GDAL owner. Repeated calls are no-ops."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for owner in reversed(self._owners):
                _close_quietly(owner)

    def __enter__(self) -> Self:
        """Return this view for context-manager use."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close this view when leaving a context manager."""
        self.close()


def open_lazy_radar_view(
    source: RadarGeolocationSource,
    *,
    block_size: int = 256,
    max_geolocation_pixels: int = 16_000_000,
) -> LazyRadarView:
    """Open a windowed WGS84 view over a persisted radar product.

    The product value store is not read. Initialization reads the dense
    longitude and latitude geolocation arrays, lets GDAL select the target
    grid from those arrays, serializes that warp into memory, and rewrites
    the block size so later windows read bounded regions.

    Parameters
    ----------
    source : RadarGeolocationSource
        Internal GDAL Zarr sources and the product-to-geolocation mapping.
    block_size : int, default=256
        Warp block size used for windowed reads. The reported
        ``chunk_shape`` is clamped to the view dimensions.
    max_geolocation_pixels : int, default=16_000_000
        Upper bound on ``geolocation_shape`` area. The cap is checked
        before any GDAL dataset is opened.

    Returns
    -------
    LazyRadarView
        Open view. Call :meth:`LazyRadarView.close` when finished.

    Raises
    ------
    LazyGeoViewError
        If the source contract is invalid, the asset kind is unsupported,
        the geolocation map exceeds the cap, the mapping points outside
        the full map, or the automatic warp produces a dateline or polar
        footprint that cannot be represented safely.

    """
    if not isinstance(source, RadarGeolocationSource):
        _reject(
            "invalid_source",
            "source must be a RadarGeolocationSource instance",
        )
    if (
        not isinstance(block_size, (int, np.integer))
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        _reject("invalid_source", "block_size must be a positive integer")
    block_size = int(block_size)
    if (
        not isinstance(max_geolocation_pixels, (int, np.integer))
        or isinstance(max_geolocation_pixels, bool)
        or max_geolocation_pixels <= 0
    ):
        _reject(
            "invalid_source",
            "max_geolocation_pixels must be a positive integer",
        )
    max_geolocation_pixels = int(max_geolocation_pixels)

    height, width = _require_shape(source.shape, "shape")
    geo_height, geo_width = _require_shape(
        source.geolocation_shape, "geolocation_shape"
    )
    geolocation_pixels = geo_height * geo_width
    if geolocation_pixels > max_geolocation_pixels:
        _reject(
            "geolocation_too_large",
            f"geolocation_shape {source.geolocation_shape} has "
            f"{geolocation_pixels} pixels, above the "
            f"{max_geolocation_pixels}-pixel cap",
        )

    data_type = _require_data_type(source.data_type)
    _require_supported_kind(source.kind, data_type)
    first_row, first_col = _require_mapping(source.first_center, "first_center")
    step_row, step_col = _require_mapping(source.step, "step")
    if step_row <= 0 or step_col <= 0:
        _reject("invalid_geolocation", "step components must be positive")
    _validate_mapping_window(
        height=height,
        width=width,
        geolocation_height=geo_height,
        geolocation_width=geo_width,
        first_row=first_row,
        first_col=first_col,
        step_row=step_row,
        step_col=step_col,
    )

    values_uri = _gdal_source_uri(source.values_uri, "values_uri")
    longitude_uri = _gdal_source_uri(source.longitude_uri, "longitude_uri")
    latitude_uri = _gdal_source_uri(source.latitude_uri, "latitude_uri")
    valid_uri = (
        _gdal_source_uri(source.valid_uri, "valid_uri")
        if source.valid_uri is not None
        else None
    )

    owners: list[object] = []
    environment = rasterio.Env(GDAL_CACHEMAX=_GDAL_CACHE_BYTES)
    environment.__enter__()
    owners.append(_EnvironmentOwner(environment))
    try:
        geoposition = _build_geoposition(
            height=height,
            width=width,
            geo_height=geo_height,
            geo_width=geo_width,
            first_row=first_row,
            first_col=first_col,
            step_row=step_row,
            step_col=step_col,
            longitude_uri=longitude_uri,
            latitude_uri=latitude_uri,
            owners=owners,
        )
        source_mem = MemoryFile(
            _source_vrt_xml(
                values_uri=values_uri,
                geoposition=geoposition,
                height=height,
                width=width,
                block_size=block_size,
                gdal_data_type=_GDAL_DATA_TYPES[data_type],
                valid_uri=valid_uri,
            ).encode(),
            ext=".vrt",
        )
        owners.append(source_mem)
        source_dataset = source_mem.open()
        owners.append(source_dataset)
        if valid_uri is not None:
            _require_validity_band(source_dataset)

        auto = WarpedVRT(
            source_dataset,
            crs=_CRS,
            resampling=Resampling.bilinear,
            tolerance=0.0,
            dst_alpha=2 if valid_uri is not None else None,
            warp_mem_limit=_WARP_MEMORY_MB,
            warp_extras={
                "SRC_METHOD": "GEOLOC_ARRAY",
                "GEOLOC_BACKMAP_OVERSAMPLE_FACTOR": "1.0",
            },
        )
        owners.append(auto)
        _validate_auto_warp(auto)
        if valid_uri is not None and auto.count < 2:
            _reject(
                "mask_unverified",
                "valid_uri was supplied but the GDAL warp did not expose "
                "a validity alpha band",
            )

        serialized_mem = MemoryFile(_PLACEHOLDER_VRT, ext=".vrt")
        owners.append(serialized_mem)
        rasterio.shutil.copy(auto, serialized_mem.name, driver="VRT")
        serialized_xml = _rewrite_serialized_vrt(
            bytes(serialized_mem.getbuffer()),
            source_dataset_path=source_mem.name,
            block_size=block_size,
        )
        reopened_mem = MemoryFile(serialized_xml, ext=".vrt")
        owners.append(reopened_mem)
        reopened = reopened_mem.open()
        owners.append(reopened)
        if valid_uri is not None and reopened.count < 2:
            _reject(
                "mask_unverified",
                "the serialized warp lost the validity alpha band",
            )
        _validate_reopened_warp(reopened)
        return LazyRadarView(
            source,
            reopened,
            tuple(owners),
            data_type=data_type,
            masked=valid_uri is not None,
        )
    except LazyGeoViewError:
        _close_owners(owners)
        raise
    except Exception as exc:
        _close_owners(owners)
        reason = "source_error"
        message = f"failed to open lazy radar view: {exc}"
        raise LazyGeoViewError(reason, message) from exc


class _EnvironmentOwner:
    """Context-manager owner that restores the GDAL environment on close."""

    def __init__(self, environment: rasterio.Env) -> None:
        self._environment = environment

    def close(self) -> None:
        self._environment.__exit__(None, None, None)


@dataclass(frozen=True, slots=True)
class _Geoposition:
    """Geolocation dataset names for the source VRT."""

    longitude_dataset: str
    latitude_dataset: str


def _reject(reason: str, detail: str) -> None:
    """Log and raise a :class:`LazyGeoViewError`."""
    logger.error("lazy radar view rejected (%s): %s", reason, detail)
    raise LazyGeoViewError(reason, detail)


def _close_quietly(owner: object) -> None:
    """Close one GDAL owner, ignoring cleanup failures."""
    close = getattr(owner, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:
        logger.debug("ignored lazy-view owner cleanup failure", exc_info=True)


def _close_owners(owners: Sequence[object]) -> None:
    """Close owners in reverse acquisition order."""
    for owner in reversed(owners):
        _close_quietly(owner)


def _require_shape(
    value: tuple[int, int],
    name: str,
) -> tuple[int, int]:
    """Validate a positive two-dimensional shape."""
    try:
        first, second = value
    except (TypeError, ValueError):
        _reject("invalid_source", f"{name} must be a two-integer tuple")
    if (
        not isinstance(first, (int, np.integer))
        or isinstance(first, bool)
        or not isinstance(second, (int, np.integer))
        or isinstance(second, bool)
        or first <= 0
        or second <= 0
    ):
        _reject("invalid_source", f"{name} must contain positive integers")
    return int(first), int(second)


def _require_mapping(
    value: tuple[float, float],
    name: str,
) -> tuple[float, float]:
    """Validate a finite two-component mapping value."""
    try:
        first, second = value
    except (TypeError, ValueError):
        _reject(
            "invalid_geolocation",
            f"{name} must be a two-number tuple",
        )
    first_value = float(first)
    second_value = float(second)
    if not math.isfinite(first_value) or not math.isfinite(second_value):
        _reject("invalid_geolocation", f"{name} components must be finite")
    return first_value, second_value


def _require_data_type(data_type: str) -> str:
    """Validate and normalize the supported scalar data type."""
    normalized = str(data_type).strip().lower()
    if normalized not in _GDAL_DATA_TYPES:
        _reject(
            "unsupported_scientific_asset",
            f"data_type {data_type!r} is not a continuous scalar float32/float64 asset",
        )
    return normalized


def _require_supported_kind(kind: str, data_type: str) -> None:
    """Reject label, complex, and wrapped-phase assets early."""
    normalized = str(kind).strip().lower()
    if normalized in {
        "wrapped",
        "wrapped-phase",
        "wrapped_phase",
        "wrapped_phase_angle",
        "wrapped-phase-angle",
    }:
        _reject(
            "unsupported_scientific_asset",
            "wrapped phase requires a complex unit phasor; the scalar "
            "lazy view does not combine phase angle and unit magnitude",
        )
    if normalized in _UNSUPPORTED_KINDS:
        _reject(
            "unsupported_scientific_asset",
            f"kind {kind!r} is not a continuous scalar asset",
        )
    if data_type not in _GDAL_DATA_TYPES:
        _reject(
            "unsupported_scientific_asset",
            f"data_type {data_type!r} is not a continuous scalar asset",
        )


def _validate_mapping_window(
    *,
    height: int,
    width: int,
    geolocation_height: int,
    geolocation_width: int,
    first_row: float,
    first_col: float,
    step_row: float,
    step_col: float,
) -> None:
    """Reject mappings whose centres or sampled window leave the full map."""
    last_row = first_row + (height - 1) * step_row
    last_col = first_col + (width - 1) * step_col
    if (
        first_row < 0
        or first_col < 0
        or last_row > geolocation_height - 1
        or last_col > geolocation_width - 1
    ):
        _reject(
            "invalid_geolocation",
            "product centres fall outside the full geolocation map",
        )
    source_off_row = first_row - step_row / 2.0 + 0.5
    source_off_col = first_col - step_col / 2.0 + 0.5
    source_size_row = height * step_row
    source_size_col = width * step_col
    if (
        source_off_row < -_GEOMETRY_EPSILON
        or source_off_col < -_GEOMETRY_EPSILON
        or source_off_row + source_size_row > geolocation_height + _GEOMETRY_EPSILON
        or source_off_col + source_size_col > geolocation_width + _GEOMETRY_EPSILON
    ):
        _reject(
            "invalid_geolocation",
            "the product's full-map sampling window leaves the geolocation array",
        )


def _gdal_source_uri(uri: str, name: str) -> str:
    """Return an internal GDAL source filename for one supplied URI."""
    text = str(uri).strip()
    if not text:
        _reject("invalid_source", f"{name} must be a non-empty URI")
    if text.startswith(("http://", "https://")):
        return f'ZARR:"/vsicurl/{text}"'
    if re.match(r"^[A-Za-z][A-Za-z0-9_+.-]*:", text):
        return text
    return f'ZARR:"{text}"'


def _build_geoposition(
    *,
    height: int,
    width: int,
    geo_height: int,
    geo_width: int,
    first_row: float,
    first_col: float,
    step_row: float,
    step_col: float,
    longitude_uri: str,
    latitude_uri: str,
    owners: list[object],
) -> _Geoposition:
    """Return direct or cropped geolocation dataset names.

    A unit-step mapping rooted at full-map ``(0, 0)`` can reference the full
    arrays directly. Every other mapping needs two geolocation-only VRTs
    cropped with bilinear resampling so the product centres are sampled from
    the full map without stretching the whole array into the product shape.
    """
    direct = (
        first_row == 0.0
        and first_col == 0.0
        and step_row == 1.0
        and step_col == 1.0
        and (height, width) == (geo_height, geo_width)
    )
    if direct:
        return _Geoposition(
            longitude_dataset=longitude_uri,
            latitude_dataset=latitude_uri,
        )

    source_off_col = first_col - step_col / 2.0 + 0.5
    source_off_row = first_row - step_row / 2.0 + 0.5
    source_size_col = width * step_col
    source_size_row = height * step_row
    longitude_mem = MemoryFile(
        _geolocation_vrt_xml(
            uri=longitude_uri,
            destination_height=height,
            destination_width=width,
            source_height=geo_height,
            source_width=geo_width,
            source_off_row=source_off_row,
            source_off_col=source_off_col,
            source_size_row=source_size_row,
            source_size_col=source_size_col,
        ).encode(),
        ext=".vrt",
    )
    owners.append(longitude_mem)
    latitude_mem = MemoryFile(
        _geolocation_vrt_xml(
            uri=latitude_uri,
            destination_height=height,
            destination_width=width,
            source_height=geo_height,
            source_width=geo_width,
            source_off_row=source_off_row,
            source_off_col=source_off_col,
            source_size_row=source_size_row,
            source_size_col=source_size_col,
        ).encode(),
        ext=".vrt",
    )
    owners.append(latitude_mem)
    return _Geoposition(
        longitude_dataset=longitude_mem.name,
        latitude_dataset=latitude_mem.name,
    )


def _numeric(value: float) -> str:
    """Format one mapping value for VRT XML without losing precision."""
    return repr(float(value))


def _geolocation_vrt_xml(
    *,
    uri: str,
    destination_height: int,
    destination_width: int,
    source_height: int,
    source_width: int,
    source_off_row: float,
    source_off_col: float,
    source_size_row: float,
    source_size_col: float,
) -> str:
    """Build a geolocation-only VRT that samples product pixel centres."""
    return f"""<VRTDataset rasterXSize="{destination_width}" \
rasterYSize="{destination_height}">
  <VRTRasterBand dataType="Float64" band="1">
    <SimpleSource resampling="bilinear">
      <SourceFilename relativeToVRT="0">{escape(uri)}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties RasterXSize="{source_width}" \
RasterYSize="{source_height}" DataType="Float64" \
BlockXSize="{min(256, source_width)}" \
BlockYSize="{min(256, source_height)}"/>
      <SrcRect xOff="{_numeric(source_off_col)}" \
yOff="{_numeric(source_off_row)}" \
xSize="{_numeric(source_size_col)}" \
ySize="{_numeric(source_size_row)}"/>
      <DstRect xOff="0" yOff="0" \
xSize="{destination_width}" ySize="{destination_height}"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
"""


def _source_vrt_xml(
    *,
    values_uri: str,
    geoposition: _Geoposition,
    height: int,
    width: int,
    block_size: int,
    gdal_data_type: str,
    valid_uri: str | None,
) -> str:
    """Build the geolocated source VRT for one product."""
    validity_band = ""
    if valid_uri is not None:
        # The validity array is an explicit second band. Warping it with
        # DstAlphaBand keeps the data band independent: a dataset-level
        # MaskBand combined with add_alpha altered bilinear data on GDAL
        # 3.12, while this form preserved the proven value field.
        validity_band = f"""
  <VRTRasterBand dataType="Byte" band="2">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{escape(valid_uri)}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties RasterXSize="{width}" RasterYSize="{height}" \
DataType="Byte" BlockXSize="{block_size}" BlockYSize="{block_size}"/>
      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>
      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>
    </SimpleSource>
  </VRTRasterBand>"""
    return f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <Metadata domain="GEOLOCATION">
    <MDI key="X_DATASET">{escape(geoposition.longitude_dataset)}</MDI>
    <MDI key="Y_DATASET">{escape(geoposition.latitude_dataset)}</MDI>
    <MDI key="X_BAND">1</MDI>
    <MDI key="Y_BAND">1</MDI>
    <MDI key="PIXEL_OFFSET">0</MDI>
    <MDI key="LINE_OFFSET">0</MDI>
    <MDI key="PIXEL_STEP">1</MDI>
    <MDI key="LINE_STEP">1</MDI>
    <MDI key="SRS">{escape(_SRS_WKT)}</MDI>
    <MDI key="GEOREFERENCING_CONVENTION">PIXEL_CENTER</MDI>
  </Metadata>
  <VRTRasterBand dataType="{gdal_data_type}" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{escape(values_uri)}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties RasterXSize="{width}" RasterYSize="{height}" \
DataType="{gdal_data_type}" BlockXSize="{block_size}" \
BlockYSize="{block_size}"/>
      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>
      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>
    </SimpleSource>
  </VRTRasterBand>{validity_band}
</VRTDataset>
"""


def _require_validity_band(source_dataset: DatasetReader) -> None:
    """Reject a supplied validity source that GDAL did not expose as a band."""
    if source_dataset.count < 2:
        _reject(
            "mask_unverified",
            "valid_uri was supplied but GDAL did not expose the validity "
            "array as the second source band",
        )


def _validate_auto_warp(auto: WarpedVRT) -> None:
    """Validate the automatically selected warp grid."""
    if auto.width <= 0 or auto.height <= 0:
        _reject("invalid_geographic_footprint", "warp grid is empty")
    bounds = auto.bounds
    values = (bounds.left, bounds.bottom, bounds.right, bounds.top)
    if not all(math.isfinite(value) for value in values):
        _reject(
            "invalid_geographic_footprint",
            "automatic warp bounds are not finite",
        )
    left, bottom, right, top = values
    if right <= left or top <= bottom:
        _reject(
            "invalid_geographic_footprint",
            "automatic warp bounds are empty or inverted",
        )
    if right - left > _MAX_EXTENT_DEGREES + _GEOMETRY_EPSILON:
        _reject(
            "antimeridian_or_polar",
            "automatic warp longitude extent exceeds 180 degrees",
        )
    if top - bottom > _MAX_EXTENT_DEGREES + _GEOMETRY_EPSILON:
        _reject(
            "antimeridian_or_polar",
            "automatic warp latitude extent exceeds 180 degrees",
        )
    if (
        left < -_MAX_EXTENT_DEGREES - _GEOMETRY_EPSILON
        or right > _MAX_EXTENT_DEGREES + _GEOMETRY_EPSILON
    ):
        _reject(
            "antimeridian_or_polar",
            "automatic warp bounds cross the antimeridian",
        )
    if abs(bottom) >= _MAX_ABSOLUTE_LATITUDE or abs(top) >= _MAX_ABSOLUTE_LATITUDE:
        _reject(
            "antimeridian_or_polar",
            "automatic warp reaches the polar cap",
        )


def _validate_reopened_warp(dataset: DatasetReader) -> None:
    """Validate the reopened serialized warp dataset."""
    if dataset.width <= 0 or dataset.height <= 0:
        _reject("invalid_geographic_footprint", "reopened warp grid is empty")
    transform = dataset.transform
    if not all(
        math.isfinite(value)
        for value in (
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        )
    ):
        _reject(
            "invalid_geographic_footprint",
            "reopened warp transform is not finite",
        )


def _rewrite_serialized_vrt(
    payload: bytes,
    *,
    source_dataset_path: str,
    block_size: int,
) -> bytes:
    """Rewrite a serialized warp VRT for bounded window reads."""
    root = ET.fromstring(payload)
    block_x = min(block_size, int(root.attrib["rasterXSize"]))
    block_y = min(block_size, int(root.attrib["rasterYSize"]))
    for tag, size in (("BlockXSize", block_x), ("BlockYSize", block_y)):
        node = root.find(tag)
        if node is None:
            node = ET.SubElement(root, tag)
        node.text = str(size)
    for warp_options in root.iter("GDALWarpOptions"):
        for option in list(warp_options.findall("Option")):
            if option.get("name") == "WARP_EXTRAS":
                warp_options.remove(option)
    for source_dataset in root.iter("SourceDataset"):
        source_dataset.set("relativeToVRT", "0")
        source_dataset.text = source_dataset_path
    return ET.tostring(root, encoding="unicode").encode()


def _validate_window(
    window: tuple[int, int, int, int] | Sequence[int],
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Validate and unpack a half-open window."""
    try:
        row_start, row_stop, col_start, col_stop = window
    except (TypeError, ValueError):
        _reject(
            "invalid_window",
            "window must be a four-integer tuple "
            "(row_start, row_stop, col_start, col_stop)",
        )
    if any(
        not isinstance(value, (int, np.integer)) or isinstance(value, bool)
        for value in (row_start, row_stop, col_start, col_stop)
    ):
        _reject("invalid_window", "window bounds must be integers")
    row_start = int(row_start)
    row_stop = int(row_stop)
    col_start = int(col_start)
    col_stop = int(col_stop)
    height, width = shape
    if (
        row_start < 0
        or col_start < 0
        or row_stop <= row_start
        or col_stop <= col_start
        or row_stop > height
        or col_stop > width
    ):
        _reject(
            "window_out_of_bounds",
            f"window {window!r} is not a non-empty half-open subset of shape {shape}",
        )
    return row_start, row_stop, col_start, col_stop
