"""HTTP-only lazy radar geolocation previews for the STAC service (P0089).

Radar-coordinate Items keep their canonical raw assets and null geometry. When
a committed Reference scene already declares ``latitude``/``longitude`` arrays
and a scalar IFG, unwrap, or scene unit asset is bound to that Reference
through recorded ``faninsar:inputs`` write ids, this module exposes the
scientific asset under its existing key through a derived, lazily resampled
EPSG:4326 view.

Discovery is metadata-only. A view handle is opened only for one asset after a
metadata or chunk request; startup and catalogue rendering never initialise a
warp. The service supplies pinned canonical-receipt, Zarr-metadata, and
receipt-validity callbacks, so this module never opens a filesystem path or a
user-supplied URI itself.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, unquote

__all__ = [
    "PREVIEW_ASSETS",
    "RadarDiagnostic",
    "RadarProvider",
    "RadarResolution",
    "RadarUnavailableError",
    "manifest_html",
    "parse_view_tail",
    "view_path",
]

#: Scientific assets that may receive a lazy geolocation preview.
PREVIEW_ASSETS: tuple[str, ...] = ("coherence", "unwrapped_phase")

#: Native payload facts that do not describe the derived preview.
_NATIVE_ASSET_FIELDS = frozenset({"file:checksum", "file:size"})

#: Projection keys describe the raw radar grid, not the EPSG:4326 preview.
_PROJECTION_FIELDS = frozenset({"proj:code", "proj:shape", "proj:transform"})

#: The only generated metadata documents, in fixed manifest order.
METADATA_PATHS: tuple[str, ...] = (
    "zarr.json",
    "values/zarr.json",
    "x/zarr.json",
    "y/zarr.json",
)

_MANIFEST_CHILDREN = ("zarr.json", "values/", "x/", "y/")
_VIEW_SUFFIX = ".zarr"
_SUPPORTED_DATA_TYPES = frozenset({"float32", "float64"})
_VALID_MASK_DATA_TYPES = frozenset({"bool", "uint8", "uint16", "uint32", "uint64"})
_MAX_COORDINATE_SAMPLES = 1024
_MAX_POSITIVE_VIEWS = 8
_CHUNK_CACHE_BYTES = 16 * 1024 * 1024
_BLOCK_SIZE = 256
_MAX_GEOLOCATION_PIXELS = 16_000_000
_EPSG_4326_URL = "http://www.opengis.net/def/crs/EPSG/0/4326"


def _unavailable(reason: str, status: int = 404, detail: str = "") -> None:
    """Raise one bounded preview diagnostic from a non-literal call site."""
    raise RadarUnavailableError(reason, status, detail)


class RadarUnavailableError(RuntimeError):
    """Report one asset that cannot be represented as a lazy radar preview.

    Parameters
    ----------
    reason : str
        Stable machine-readable diagnostic.
    status : int
        Suggested HTTP status: ``404`` for unavailable, ``503`` for a bounded
        backend limit, and ``400`` for a rejected request or source.
    detail : str, optional
        Developer-facing detail. It is never published on the Item.

    """

    def __init__(self, reason: str, status: int = 404, detail: str = "") -> None:
        self.reason = reason
        self.status = status
        self.detail = detail or reason
        super().__init__(f"{reason}: {self.detail}")


@dataclass(frozen=True, slots=True)
class RadarDiagnostic:
    """One metadata-only rejection of a radar preview candidate."""

    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class _Discovery:
    """Metadata needed to name one lazy radar source without opening it."""

    source_key: str
    kind: str
    shape: tuple[int, int]
    geolocation_shape: tuple[int, int]
    data_type: str
    first_center: tuple[float, float]
    step: tuple[float, float]
    values_uri: str
    longitude_uri: str
    latitude_uri: str
    valid_uri: str | None
    fingerprint: str
    modified: float


@dataclass(slots=True)
class _CachedView:
    """One open backend handle with its own read lock and chunk LRU."""

    fingerprint: str
    view: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    chunks: OrderedDict[str, bytes] = field(default_factory=OrderedDict)
    chunk_bytes: int = 0
    closed: bool = False

    def close(self) -> None:
        """Close the backend handle once, under the per-handle lock."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            close = getattr(self.view, "close", None)
            if callable(close):
                close()
            self.chunks.clear()
            self.chunk_bytes = 0


@dataclass(frozen=True, slots=True)
class RadarResolution:
    """One current, resolvable radar preview ready for HTTP presentation."""

    source_key: str
    fingerprint: str
    modified: float
    shape: tuple[int, int]
    transform: tuple[float, float, float, float, float, float]
    data_type: str
    fill_value: Any
    chunk_shape: tuple[int, int]
    _view: _CachedView
    _metadata: Mapping[str, Mapping[str, Any]]

    def metadata(self, relative: str) -> dict[str, Any] | None:
        """Return one generated Zarr metadata document, or ``None``."""
        document = self._metadata.get(relative)
        return copy.deepcopy(dict(document)) if document is not None else None

    def coordinate_chunk(self, relative: str) -> bytes | None:
        """Return one generated pixel-centre coordinate chunk."""
        path = relative.strip("/")
        parts = path.split("/")
        if len(parts) != 3 or parts[1] != "c" or parts[0] not in {"x", "y"}:
            return None
        index = _canonical_index(parts[2])
        if index is None:
            return None
        axis = parts[0]
        length = self.shape[1] if axis == "x" else self.shape[0]
        chunk = min(_MAX_COORDINATE_SAMPLES, length)
        if index >= (length + chunk - 1) // chunk:
            return None
        start = index * chunk
        valid = min(chunk, length - start)
        values = [0.0] * chunk
        for offset in range(valid):
            position = start + offset + 0.5
            if axis == "x":
                values[offset] = self.transform[2] + self.transform[0] * position
            else:
                values[offset] = self.transform[5] + self.transform[4] * position
        import numpy as np

        return np.asarray(values, dtype="<f8").tobytes()

    def chunk_index(self, relative: str) -> tuple[int, int] | None:
        """Return validated ``values`` chunk indices, or ``None``."""
        path = relative.strip("/")
        parts = path.split("/")
        if len(parts) != 4 or parts[0] != "values" or parts[1] != "c":
            return None
        row = _canonical_index(parts[2])
        column = _canonical_index(parts[3])
        if row is None or column is None:
            return None
        rows, cols = self.chunk_shape
        if row >= (self.shape[0] + rows - 1) // rows:
            return None
        if column >= (self.shape[1] + cols - 1) // cols:
            return None
        return row, column

    def chunk_length(self) -> int:
        """Return the always-padded full chunk byte length."""
        import numpy as np

        rows, cols = self.chunk_shape
        return rows * cols * np.dtype(self.data_type).itemsize


class RadarProvider:
    """Discover, cache, and serve lazy radar geolocation previews.

    Parameters
    ----------
    base_url : str
        Loopback service base URL without a trailing slash.
    open_metadata : callable
        ``open_metadata(record, asset_key)`` returns the parsed native
        ``zarr.json`` mapping and its modification time, or ``(None, 0.0)``.
        The service implementation performs the pinned safe open.
    receipt_valid : callable
        ``receipt_valid(record)`` returns whether the canonical product
        receipt still names the record write id.
    receipt_fingerprint : callable
        ``receipt_fingerprint(record)`` returns a bounded current receipt
        digest, or an empty string when the receipt cannot be read.
    open_view : callable, optional
        Backend factory used by tests. Defaults to the scientific lazy-view
        module, imported only when a handle is actually requested.

    """

    def __init__(
        self,
        *,
        base_url: str,
        open_metadata: Callable[[Any, str], tuple[Mapping[str, Any] | None, float]],
        receipt_valid: Callable[[Any], bool],
        receipt_fingerprint: Callable[[Any], str],
        open_view: Callable[[_Discovery], Any] | None = None,
        block_size: int = _BLOCK_SIZE,
        max_geolocation_pixels: int = _MAX_GEOLOCATION_PIXELS,
        max_views: int = _MAX_POSITIVE_VIEWS,
        chunk_cache_bytes: int = _CHUNK_CACHE_BYTES,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._open_metadata = open_metadata
        self._receipt_valid = receipt_valid
        self._receipt_fingerprint = receipt_fingerprint
        self._open_view = open_view or (
            lambda discovery: _open_backend_view(
                discovery, block_size, max_geolocation_pixels
            )
        )
        self._block_size = block_size
        self._max_geolocation_pixels = max_geolocation_pixels
        self._max_views = max(1, max_views)
        self._chunk_cache_bytes = max(0, chunk_cache_bytes)
        self._lock = threading.Lock()
        self._open_lock = threading.Lock()
        self._views: OrderedDict[tuple[str, str], _CachedView] = OrderedDict()
        self._index: Any | None = None
        self._by_write_id: dict[str, Any] = {}
        self._closed = False

    # -- Item presentation -------------------------------------------------

    def augment_item(self, record: Any, item: dict[str, Any], index: Any) -> None:
        """Add lazy radar previews or stable diagnostics to one HTTP Item.

        Only metadata and source provenance are resolved here. No backend
        geolocation warp is opened, so rendering global search results cannot
        initialise maps for unrelated assets.
        """
        assets = item.get("assets")
        if not isinstance(assets, dict) or not _radar(record):
            return
        kind = str(record.kind)
        if kind in {"ifg", "unwrap", "unwrap_ion"}:
            candidates = [key for key in PREVIEW_ASSETS if key in assets]
        elif kind == "scene":
            candidates = sorted(
                key
                for key in assets
                if _scene_asset_placement(_assets(record).get(key)) is not None
            )
        else:
            return
        for source_key in candidates:
            source_asset = assets.get(source_key)
            if not isinstance(source_asset, dict):
                continue
            discovery, diagnostic = self._classify(index, record, source_key)
            if discovery is None:
                if diagnostic is not None and not (
                    kind == "scene"
                    and diagnostic.reason
                    in {"unsupported-asset", "unsupported-data-type"}
                ):
                    document = dict(source_asset)
                    document["faninsar:view_unavailable"] = diagnostic.reason
                    assets[source_key] = document
                continue
            assets[source_key] = _preview_asset_for(
                source_asset,
                source_key,
                self._base_url,
                str(record.public_id),
            )

    # -- HTTP resolution ---------------------------------------------------

    def resolve(
        self, index: Any, record: Any, source_key: str
    ) -> RadarResolution | None:
        """Return a current view handle, opening it on first use.

        The receipt and asset metadata fingerprints are recomputed for every
        call. A changed fingerprint closes the old handle before another one
        is installed, and a changed receipt returns ``None``.
        """
        if not _radar(record):
            return None
        discovery, _diagnostic = self._classify(index, record, source_key)
        if discovery is None:
            return None
        key = (str(record.public_id), source_key)
        cached = self._current(key, discovery.fingerprint)
        if cached is None:
            with self._open_lock:
                cached = self._current(key, discovery.fingerprint)
                if cached is None:
                    try:
                        view = self._open_view(discovery)
                    except RadarUnavailableError:
                        raise
                    except Exception as error:
                        raise _open_failure(error) from error
                    cached = _CachedView(fingerprint=discovery.fingerprint, view=view)
                    self._install(key, cached)
        try:
            shape, transform, data_type, fill_value, chunk_shape = _view_contract(
                cached.view
            )
        except RadarUnavailableError:
            raise
        except Exception as error:
            raise _open_failure(error) from error
        return RadarResolution(
            source_key=source_key,
            fingerprint=discovery.fingerprint,
            modified=discovery.modified,
            shape=shape,
            transform=transform,
            data_type=data_type,
            fill_value=fill_value,
            chunk_shape=chunk_shape,
            _view=cached,
            _metadata=_metadata_documents(shape, data_type, fill_value, chunk_shape),
        )

    def read_chunk(
        self, resolution: RadarResolution, row: int, column: int
    ) -> bytes | None:
        """Read one target chunk through the lazy backend and cache its bytes."""
        cached = resolution._view
        key = f"{row}/{column}"
        with cached.lock:
            if cached.closed:
                return None
            payload = cached.chunks.get(key)
            if payload is not None:
                cached.chunks.move_to_end(key)
                return payload
            rows, cols = resolution.chunk_shape
            shape_rows, shape_cols = resolution.shape
            r0 = row * rows
            c0 = column * cols
            r1 = min(shape_rows, r0 + rows)
            c1 = min(shape_cols, c0 + cols)
            if r0 >= r1 or c0 >= c1:
                return None
            import numpy as np

            dtype = np.dtype(resolution.data_type)
            window = cached.view.read_window((r0, r1, c0, c1))
            data = np.ma.filled(np.asarray(window), np.nan)
            if data.shape != (r1 - r0, c1 - c0):
                raise _unavailable(
                    "backend-shape-mismatch",  # noqa: EM101
                    503,
                    f"window returned {data.shape}, expected {(r1 - r0, c1 - c0)}",
                )
            padded = np.full((rows, cols), np.nan, dtype=dtype)
            padded[: r1 - r0, : c1 - c0] = data
            payload = padded.astype(dtype.newbyteorder("<"), copy=False).tobytes()
            self._remember_chunk(cached, key, payload)
            return payload

    def close(self) -> None:
        """Close every cached backend handle; safe to call repeatedly."""
        with self._lock:
            self._closed = True
            views = list(self._views.values())
            self._views.clear()
        for cached in views:
            cached.close()

    def _remember_chunk(self, cached: _CachedView, key: str, payload: bytes) -> None:
        """Store one chunk under the byte-bounded LRU policy."""
        if self._chunk_cache_bytes <= 0:
            return
        while cached.chunks and (
            cached.chunk_bytes + len(payload) > self._chunk_cache_bytes
        ):
            _, removed = cached.chunks.popitem(last=False)
            cached.chunk_bytes -= len(removed)
        if len(payload) > self._chunk_cache_bytes:
            return
        if key not in cached.chunks:
            cached.chunks[key] = payload
            cached.chunk_bytes += len(payload)

    def _current(self, key: tuple[str, str], fingerprint: str) -> _CachedView | None:
        """Return the cached handle when its fingerprint is still current."""
        with self._lock:
            if self._closed:
                return None
            cached = self._views.get(key)
            if cached is None:
                return None
            if cached.fingerprint == fingerprint and not cached.closed:
                self._views.move_to_end(key)
                return cached
        cached.close()
        with self._lock:
            if self._views.get(key) is cached:
                self._views.pop(key, None)
        return None

    def _install(self, key: tuple[str, str], cached: _CachedView) -> None:
        """Install one opened handle and close the least recently used one."""
        evicted: _CachedView | None = None
        with self._lock:
            if self._closed:
                evicted = cached
            else:
                self._views[key] = cached
                self._views.move_to_end(key)
                while len(self._views) > self._max_views:
                    _, evicted = self._views.popitem(last=False)
        if evicted is not None:
            evicted.close()

    # -- Discovery ---------------------------------------------------------

    def _classify(  # noqa: PLR0911 - one explicit diagnostic per stage
        self, index: Any, record: Any, source_key: str
    ) -> tuple[_Discovery | None, RadarDiagnostic | None]:
        """Resolve provenance and validate metadata without opening a view."""
        if not _radar(record):
            return None, RadarDiagnostic("unsupported-kind")
        record_kind = str(record.kind)
        by_write = self._by_write(index)
        if record_kind in {"ifg", "unwrap", "unwrap_ion"}:
            if source_key not in PREVIEW_ASSETS:
                return None, RadarDiagnostic("unsupported-asset")
            binding = _bind(index, record, by_write)
            if isinstance(binding, RadarDiagnostic):
                return None, binding
            target, ifg, reference = binding
            placement = _placement(ifg)
            preview_kind = (
                "unwrapped_phase" if source_key == "unwrapped_phase" else "coherence"
            )
        elif record_kind == "scene":
            scene_binding = _bind_scene(index, record, by_write)
            if isinstance(scene_binding, RadarDiagnostic):
                return None, scene_binding
            target, reference = scene_binding
            placement = _scene_asset_placement(_assets(target).get(source_key))
            if placement is None:
                return None, RadarDiagnostic("unsupported-asset")
            preview_kind = "coherence"
        else:
            return None, RadarDiagnostic("unsupported-kind")
        if source_key not in target.asset_paths or source_key not in _assets(target):
            return None, RadarDiagnostic("missing-source")
        if (
            "latitude" not in reference.asset_paths
            or "longitude" not in reference.asset_paths
        ):
            return None, RadarDiagnostic("missing-geolocation")
        receipt = self._receipt_fingerprint(reference)
        if not receipt or not self._receipt_valid(reference):
            return None, RadarDiagnostic("receipt-invalid")
        source_metadata, source_modified = self._open_metadata(target, source_key)
        if not source_metadata:
            return None, RadarDiagnostic("source-unavailable")
        source = _array_metadata(source_metadata)
        if source is None:
            return None, RadarDiagnostic("invalid-source")
        shape, data_type, _chunk_shape = source
        if record_kind == "scene":
            expected = _scene_asset_shape(_assets(target).get(source_key))
            if expected != shape:
                return None, RadarDiagnostic("invalid-source")
        if data_type not in _SUPPORTED_DATA_TYPES:
            return None, RadarDiagnostic("unsupported-data-type")
        reference_shape = _shape_property(reference)
        if reference_shape is None:
            return None, RadarDiagnostic("missing-geolocation")
        if shape[0] <= 0 or shape[1] <= 0:
            return None, RadarDiagnostic("invalid-source")
        geolocation = _geolocation_metadata(
            reference,
            reference_shape,
            self._open_metadata,
        )
        if isinstance(geolocation, RadarDiagnostic):
            return None, geolocation
        longitude_metadata, latitude_metadata = geolocation
        if isinstance(placement, RadarDiagnostic):
            return None, placement
        first_center, step = placement
        valid_uri = None
        valid_modified = 0.0
        valid_key: str | None = None
        if record_kind == "scene":
            valid_key = _scene_valid_key(target, source_key)
            if isinstance(valid_key, RadarDiagnostic):
                return None, valid_key
        elif source_key in PREVIEW_ASSETS and "valid_mask" in _assets(target):
            if "valid_mask" not in target.asset_paths:
                return None, RadarDiagnostic("missing-source")
            valid_key = "valid_mask"
        if valid_key is not None:
            valid_metadata, valid_modified = self._open_metadata(target, valid_key)
            if _valid_mask_metadata(valid_metadata, shape) is None:
                return None, RadarDiagnostic("invalid-source")
            valid_uri = _asset_uri(self._base_url, target.public_id, valid_key)
        if not self._receipt_valid(target):
            return None, RadarDiagnostic("receipt-invalid")
        target_receipt = self._receipt_fingerprint(target)
        if not target_receipt:
            return None, RadarDiagnostic("receipt-invalid")
        fingerprint = _fingerprint(
            reference=reference,
            target=target,
            reference_receipt=receipt,
            target_receipt=target_receipt,
            source_key=source_key,
            shape=shape,
            data_type=data_type,
            source_modified=source_modified,
            valid_modified=valid_modified,
            longitude_modified=longitude_metadata[1],
            latitude_modified=latitude_metadata[1],
            first_center=first_center,
            step=step,
        )
        discovery = _Discovery(
            source_key=source_key,
            kind=preview_kind,
            shape=shape,
            geolocation_shape=reference_shape,
            data_type=data_type,
            first_center=first_center,
            step=step,
            values_uri=_asset_uri(self._base_url, target.public_id, source_key),
            longitude_uri=_asset_uri(self._base_url, reference.public_id, "longitude"),
            latitude_uri=_asset_uri(self._base_url, reference.public_id, "latitude"),
            valid_uri=valid_uri,
            fingerprint=fingerprint,
            modified=max(source_modified, longitude_metadata[1], latitude_metadata[1]),
        )
        return discovery, None

    def _by_write(self, index: Any) -> dict[str, Any]:
        """Return a metadata-only write-id map for the current index object."""
        with self._lock:
            if index is self._index:
                return self._by_write_id
        mapping = {
            str(record.write_id): record for record in getattr(index, "records", ())
        }
        with self._lock:
            self._index = index
            self._by_write_id = mapping
        return mapping


def _bind(  # noqa: PLR0911 - one explicit diagnostic per stage
    index: Any, record: Any, by_write: Mapping[str, Any]
) -> tuple[Any, Any, Any] | RadarDiagnostic:
    """Follow recorded inputs to one unique radar Reference for an IFG chain."""
    data_record = record
    if str(record.kind) in {"unwrap", "unwrap_ion"}:
        ifg = _follow_ifg(record, by_write)
        if isinstance(ifg, RadarDiagnostic):
            return ifg
    else:
        ifg = record
    if str(ifg.kind) != "ifg" or not _radar(ifg):
        return RadarDiagnostic("unsupported-kind")
    inputs = _inputs(ifg)
    primary_id = inputs.get("primary")
    secondary_id = inputs.get("secondary")
    if not primary_id or not secondary_id:
        return RadarDiagnostic("missing-source")
    primary = by_write.get(primary_id)
    secondary = by_write.get(secondary_id)
    if (
        primary is None
        or secondary is None
        or str(primary.kind) != "scene"
        or str(secondary.kind) != "scene"
        or not _radar(primary)
        or not _radar(secondary)
    ):
        return RadarDiagnostic("missing-source")
    dates = [
        _text_property(scene, "faninsar:reference") for scene in (primary, secondary)
    ]
    if any(value is None for value in dates):
        return RadarDiagnostic("missing-reference")
    if len(set(dates)) != 1:
        return RadarDiagnostic("ambiguous-reference")
    reference_date = next(iter(dates))
    candidates = [
        item
        for item in getattr(index, "records", ())
        if str(item.kind) == "scene"
        and _radar(item)
        and _text_property(item, "faninsar:role") == "reference"
        and _text_property(item, "faninsar:date") == reference_date
    ]
    if len(candidates) != 1:
        return RadarDiagnostic("ambiguous-reference")
    reference = candidates[0]
    for scene in (primary, secondary):
        diagnostic = _check_reference_binding(scene, reference)
        if diagnostic is not None:
            return diagnostic
    if not _grid_shape_matches(ifg, reference):
        return RadarDiagnostic("grid-shape-mismatch")
    return data_record, ifg, reference


def _check_reference_binding(scene: Any, reference: Any) -> RadarDiagnostic | None:
    """Require an IFG parent scene to name the resolved Reference write id."""
    role = _text_property(scene, "faninsar:role")
    if role == "reference":
        if str(scene.write_id) == str(reference.write_id):
            return None
        return RadarDiagnostic("ambiguous-reference")
    if role == "secondary":
        declared = _inputs(scene).get("reference")
        if not declared:
            return RadarDiagnostic("missing-reference")
        if declared == str(reference.write_id):
            return None
        return RadarDiagnostic("ambiguous-reference")
    return RadarDiagnostic("missing-reference")


def _bind_scene(
    _index: Any, record: Any, by_write: Mapping[str, Any]
) -> tuple[Any, Any] | RadarDiagnostic:
    """Bind one scene SLC asset to its declared canonical Reference scene."""
    role = _text_property(record, "faninsar:role")
    if role == "reference":
        return record, record
    if role != "secondary":
        return RadarDiagnostic("unsupported-kind")
    declared = _inputs(record).get("reference")
    if not declared:
        return RadarDiagnostic("missing-reference")
    reference = by_write.get(declared)
    if (
        reference is None
        or str(reference.kind) != "scene"
        or not _radar(reference)
        or _text_property(reference, "faninsar:role") != "reference"
    ):
        return RadarDiagnostic("ambiguous-reference")
    reference_date = _text_property(record, "faninsar:reference")
    if (
        reference_date is not None
        and _text_property(reference, "faninsar:date") != reference_date
    ):
        return RadarDiagnostic("ambiguous-reference")
    return record, reference


def _scene_valid_key(record: Any, source_key: str) -> str | RadarDiagnostic | None:
    """Return the declared ``<source_key>.valid`` companion, if any."""
    key = f"{source_key}.valid"
    asset = _assets(record).get(key)
    if not isinstance(asset, Mapping):
        return None
    if asset.get("faninsar:role") != "valid_mask":
        return RadarDiagnostic("invalid-source")
    if key not in record.asset_paths:
        return RadarDiagnostic("missing-source")
    return key


def _valid_mask_metadata(
    metadata: Mapping[str, Any] | None,
    shape: tuple[int, int],
) -> tuple[tuple[int, int], str, tuple[int, int]] | None:
    """Validate one rank-2 boolean or unsigned validity companion."""
    parsed = _array_metadata(metadata) if metadata else None
    if parsed is None:
        return None
    valid_shape, data_type, _chunk_shape = parsed
    if valid_shape != shape or data_type not in _VALID_MASK_DATA_TYPES:
        return None
    return parsed


def _scene_asset_shape(asset: Any) -> tuple[int, int] | None:
    """Return one scene unit asset's declared ``(rows, cols)``."""
    if not isinstance(asset, Mapping):
        return None
    try:
        rows = int(asset["faninsar:rows"])
        cols = int(asset["faninsar:cols"])
    except (KeyError, TypeError, ValueError):
        return None
    return (rows, cols) if rows > 0 and cols > 0 else None


def _scene_asset_placement(
    asset: Any,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return an SLC unit's native first centre and unit step, if declared."""
    if not isinstance(asset, Mapping):
        return None
    shape = _scene_asset_shape(asset)
    if shape is None:
        return None
    try:
        row_origin = int(asset["faninsar:row_origin"])
        col_origin = int(asset["faninsar:col_origin"])
    except (KeyError, TypeError, ValueError):
        return None
    return (float(row_origin), float(col_origin)), (1.0, 1.0)


def _follow_ifg(record: Any, by_write: Mapping[str, Any]) -> Any:
    """Follow the recorded IFG input of an unwrap product without guessing."""
    seen = {str(record.write_id)}
    current = record
    while str(current.kind) in {"unwrap", "unwrap_ion"}:
        next_id = _inputs(current).get("ifg")
        if not next_id:
            return RadarDiagnostic("missing-source")
        if next_id in seen:
            return RadarDiagnostic("cyclic-inputs")
        seen.add(next_id)
        current = by_write.get(next_id)
        if current is None:
            return RadarDiagnostic("missing-source")
    return current


def _placement(
    ifg: Any,
) -> tuple[tuple[float, float], tuple[float, float]] | RadarDiagnostic:
    """Return native-pixel first-sample centres and spacing for one IFG."""
    properties = _properties(ifg)
    looks = properties.get("faninsar:looks")
    if not isinstance(looks, list | tuple) or len(looks) != 2:
        return RadarDiagnostic("invalid-placement")
    try:
        look_rows, look_cols = int(looks[0]), int(looks[1])
        row_origin = int(properties.get("faninsar:row_origin", 0))
        col_origin = int(properties.get("faninsar:col_origin", 0))
        origin = properties.get("faninsar:look_origin", [0, 0])
        if not isinstance(origin, list | tuple) or len(origin) != 2:
            return RadarDiagnostic("invalid-placement")
        look_row_origin, look_col_origin = int(origin[0]), int(origin[1])
    except (TypeError, ValueError):
        return RadarDiagnostic("invalid-placement")
    if look_rows <= 0 or look_cols <= 0:
        return RadarDiagnostic("invalid-placement")
    first_center = (
        float(row_origin + look_row_origin * look_rows + (look_rows - 1) / 2),
        float(col_origin + look_col_origin * look_cols + (look_cols - 1) / 2),
    )
    return first_center, (float(look_rows), float(look_cols))


def _grid_shape_matches(ifg: Any, reference: Any) -> bool:
    """Require the declared IFG parent grid to equal the Reference grid shape."""
    declared = _shape_property(ifg, "faninsar:grid_shape")
    reference_shape = _shape_property(reference)
    return (
        declared is not None
        and reference_shape is not None
        and declared == reference_shape
    )


def _geolocation_metadata(
    reference: Any,
    reference_shape: tuple[int, int],
    open_metadata: Callable[[Any, str], tuple[Mapping[str, Any] | None, float]],
) -> (
    tuple[tuple[Mapping[str, Any], float], tuple[Mapping[str, Any], float]]
    | RadarDiagnostic
):
    """Read and validate both declared Reference geolocation arrays."""
    result: list[tuple[Mapping[str, Any], float]] = []
    for asset_key in ("longitude", "latitude"):
        metadata, modified = open_metadata(reference, asset_key)
        parsed = _array_metadata(metadata) if metadata else None
        if parsed is None:
            return RadarDiagnostic("invalid-geolocation")
        shape, data_type, _chunk_shape = parsed
        if shape != reference_shape or data_type != "float64":
            return RadarDiagnostic("invalid-geolocation")
        result.append((metadata, modified))
    return result[0], result[1]


def _array_metadata(
    metadata: Mapping[str, Any],
) -> tuple[tuple[int, int], str, tuple[int, int]] | None:
    """Validate one native Zarr v3 rank-2 array metadata document."""
    if metadata.get("zarr_format") != 3 or metadata.get("node_type") != "array":
        return None
    raw_shape = metadata.get("shape")
    if not isinstance(raw_shape, list | tuple) or len(raw_shape) != 2:
        return None
    try:
        shape = (int(raw_shape[0]), int(raw_shape[1]))
    except (TypeError, ValueError):
        return None
    if shape[0] <= 0 or shape[1] <= 0:
        return None
    data_type = metadata.get("data_type")
    if not isinstance(data_type, str) or not data_type:
        return None
    chunk_grid = metadata.get("chunk_grid")
    configuration = (
        chunk_grid.get("configuration") if isinstance(chunk_grid, Mapping) else None
    )
    raw_chunk = (
        configuration.get("chunk_shape") if isinstance(configuration, Mapping) else None
    )
    if not isinstance(raw_chunk, list | tuple) or len(raw_chunk) != 2:
        chunk_shape = None
    else:
        try:
            chunk_shape = (int(raw_chunk[0]), int(raw_chunk[1]))
        except (TypeError, ValueError):
            chunk_shape = None
        if chunk_shape is not None and min(chunk_shape) <= 0:
            chunk_shape = None
    return shape, data_type, chunk_shape


def _view_contract(
    view: Any,
) -> tuple[
    tuple[int, int],
    tuple[float, float, float, float, float, float],
    str,
    Any,
    tuple[int, int],
]:
    """Validate the scientific backend object before serving any bytes."""
    raw_shape = getattr(view, "shape", None)
    raw_transform = getattr(view, "transform", None)
    data_type = getattr(view, "data_type", None)
    if (
        not isinstance(raw_shape, tuple | list)
        or len(raw_shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in raw_shape
        )
        or any(value <= 0 for value in raw_shape)
    ):
        raise _unavailable("backend-contract", 503, "invalid view shape")  # noqa: EM101
    shape = (int(raw_shape[0]), int(raw_shape[1]))
    if (
        not isinstance(raw_transform, tuple | list)
        or len(raw_transform) != 6
        or any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in raw_transform
        )
        or any(not math.isfinite(float(value)) for value in raw_transform)
    ):
        raise _unavailable("backend-contract", 503, "invalid view transform")  # noqa: EM101
    transform = tuple(float(value) for value in raw_transform)
    if data_type not in _SUPPORTED_DATA_TYPES:
        raise _unavailable("unsupported-data-type", 404)  # noqa: EM101
    raw_chunk = getattr(view, "chunk_shape", None)
    if (
        isinstance(raw_chunk, tuple | list)
        and len(raw_chunk) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in raw_chunk
        )
    ):
        chunk_shape = (int(raw_chunk[0]), int(raw_chunk[1]))
    else:
        chunk_shape = (_BLOCK_SIZE, _BLOCK_SIZE)
    crs = getattr(view, "crs", None)
    if not _is_epsg_4326(crs):
        raise _unavailable("backend-crs", 503, f"unexpected CRS {crs!r}")  # noqa: EM101
    fill_value = getattr(view, "fill_value", float("nan"))
    return shape, transform, data_type, fill_value, chunk_shape


def _is_epsg_4326(crs: Any) -> bool:
    """Return whether a backend CRS value denotes EPSG:4326."""
    if isinstance(crs, str):
        return crs.strip().upper() in {"EPSG:4326", "WGS84", "OGC:CRS84"}
    try:
        return int(crs.to_epsg()) == 4326
    except (AttributeError, TypeError, ValueError):
        return False


def _metadata_documents(
    shape: tuple[int, int],
    data_type: str,
    fill_value: Any,
    chunk_shape: tuple[int, int],
) -> dict[str, Mapping[str, Any]]:
    """Build the four honest Zarr v3 documents for one preview."""
    values = {
        "shape": [int(shape[0]), int(shape[1])],
        "data_type": data_type,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(chunk_shape)},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": _json_fill(fill_value),
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": {
            "_CRS": {"url": _EPSG_4326_URL},
            "faninsar:preview": True,
            "faninsar:resampling": "bilinear",
        },
        "dimension_names": ["y", "x"],
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }
    x = _coordinate_metadata("x", shape[1])
    y = _coordinate_metadata("y", shape[0])
    root = {
        "attributes": {},
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {
                "values": values,
                "x": x,
                "y": y,
            },
        },
    }
    return {
        "zarr.json": root,
        "values/zarr.json": values,
        "x/zarr.json": x,
        "y/zarr.json": y,
    }


def _coordinate_metadata(axis: str, length: int) -> dict[str, Any]:
    """Return one generated pixel-centre coordinate array document."""
    return {
        "shape": [int(length)],
        "data_type": "float64",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [min(_MAX_COORDINATE_SAMPLES, length)]},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0.0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": {
            "standard_name": "longitude" if axis == "x" else "latitude",
            "axis": axis.upper(),
            "units": "degrees_east" if axis == "x" else "degrees_north",
        },
        "dimension_names": [axis],
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }


def _json_fill(value: Any) -> Any:
    """Return a strict-JSON Zarr fill value for a float preview."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NaN"
    return "NaN" if math.isnan(number) else number


def _fingerprint(
    *,
    reference: Any,
    target: Any,
    reference_receipt: str,
    target_receipt: str,
    source_key: str,
    shape: tuple[int, int],
    data_type: str,
    source_modified: float,
    valid_modified: float,
    longitude_modified: float,
    latitude_modified: float,
    first_center: tuple[float, float],
    step: tuple[float, float],
) -> str:
    """Return a bounded identity for the current preview inputs."""
    payload = {
        "reference_write_id": str(reference.write_id),
        "reference_receipt": reference_receipt,
        "target_write_id": str(target.write_id),
        "target_receipt": target_receipt,
        "source_key": source_key,
        "shape": list(shape),
        "data_type": data_type,
        "source_modified": float(source_modified),
        "valid_modified": float(valid_modified),
        "longitude_modified": float(longitude_modified),
        "latitude_modified": float(latitude_modified),
        "first_center": list(first_center),
        "step": list(step),
        "resampling": "bilinear",
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _preview_asset_for(
    source_asset: Mapping[str, Any],
    source_key: str,
    base_url: str,
    public_id: str,
) -> dict[str, Any]:
    """Build one preview asset document for a known public Item id."""
    href_path = view_path(public_id, source_key)
    document = {
        key: value
        for key, value in source_asset.items()
        if key not in _NATIVE_ASSET_FIELDS and key not in _PROJECTION_FIELDS
    }
    document.update(
        {
            "href": f"{base_url.rstrip('/')}{href_path}",
            "type": "application/vnd+zarr",
            "title": _asset_title(source_asset, source_key),
            "roles": source_asset.get("roles", ["data"]),
            "faninsar:source_asset": source_key,
            "faninsar:gdal_uri": (
                f'ZARR:"/vsicurl/{base_url.rstrip("/")}{href_path}":/values'
            ),
            "faninsar:resampling": "bilinear",
            "faninsar:preview": True,
            "faninsar:view_status": "lazy-geolocation",
        }
    )
    native_href = source_asset.get("href")
    if isinstance(native_href, str) and native_href:
        document["faninsar:native_href"] = native_href
    return document


def _asset_title(source_asset: Mapping[str, Any], source_key: str) -> str:
    """Return an existing human title, or the scientific source key."""
    title = source_asset.get("title")
    if isinstance(title, str) and title.strip():
        return title
    return source_key


def _asset_uri(base_url: str, public_id: str, asset_key: str) -> str:
    """Return the registered raw-array URI passed to the scientific backend."""
    return (
        'ZARR:"/vsicurl/'
        f"{base_url.rstrip('/')}/assets/{quote(str(public_id), safe='')}/"
        f'{quote(str(asset_key), safe="")}/"'
    )


def view_path(public_id: str, source_key: str) -> str:
    """Return the absolute-path href of one radar preview store."""
    return (
        f"/radar/{quote(public_id, safe='')}/{quote(source_key, safe='')}"
        f"{_VIEW_SUFFIX}/"
    )


def parse_view_tail(asset_keys: Any, tail: str) -> tuple[str, str, bool] | None:
    """Split a decoded route tail into ``(source_key, relative, slashless)``."""
    decoded = unquote(tail or "")
    if "\x00" in decoded or "\\" in decoded or decoded.startswith("/"):
        return None
    for key in sorted(asset_keys, key=len, reverse=True):
        if not isinstance(key, str) or not key:
            continue
        stem = f"{key}{_VIEW_SUFFIX}"
        if decoded == stem:
            return key, "", True
        prefix = f"{stem}/"
        if decoded.startswith(prefix):
            return key, decoded[len(prefix) :], False
    return None


def manifest_html(public_id: str, source_key: str) -> bytes:
    """Return the fixed four-entry discovery manifest for one radar group."""
    title = html.escape(
        f"Index of /radar/{public_id}/{source_key}{_VIEW_SUFFIX}/",
        quote=True,
    )
    rows = "\n".join(
        f'<tr><td><a href="{child}">{child}</a></td></tr>'
        for child in _MANIFEST_CHILDREN
    )
    document = (
        "<!DOCTYPE HTML>\n"
        "<html>\n"
        " <head>\n"
        f"  <title>{title}</title>\n"
        " </head>\n"
        " <body>\n"
        f"<h1>{title}</h1>\n"
        "<table>\n"
        f"{rows}\n"
        "</table>\n"
        " </body>\n"
        "</html>\n"
    )
    return document.encode("utf-8")


def _open_backend_view(
    discovery: _Discovery,
    block_size: int = _BLOCK_SIZE,
    max_geolocation_pixels: int = _MAX_GEOLOCATION_PIXELS,
) -> Any:
    """Open the scientific lazy radar view lazily."""
    try:
        from faninsar.processing.geocoding import lazyview
    except ImportError as error:
        msg = "backend-unavailable"
        raise RadarUnavailableError(
            msg, 503, "lazy radar backend is unavailable"
        ) from error
    source = lazyview.RadarGeolocationSource(
        values_uri=discovery.values_uri,
        longitude_uri=discovery.longitude_uri,
        latitude_uri=discovery.latitude_uri,
        shape=discovery.shape,
        geolocation_shape=discovery.geolocation_shape,
        data_type=discovery.data_type,
        first_center=discovery.first_center,
        step=discovery.step,
        kind=discovery.kind,
        valid_uri=discovery.valid_uri,
        source_fingerprint=discovery.fingerprint,
    )
    return lazyview.open_lazy_radar_view(
        source,
        block_size=block_size,
        max_geolocation_pixels=max_geolocation_pixels,
    )


def _open_failure(error: BaseException) -> RadarUnavailableError:
    """Map one backend failure onto an explicit HTTP status."""
    reason = str(getattr(error, "reason", ""))
    name = type(error).__name__.lower()
    message = str(error).lower()
    if reason == "geolocation_too_large" or (
        "limit" in name
        or "limit" in message
        or "too large" in message
        or "max_geolocation" in message
    ):
        return RadarUnavailableError("geolocation-limit", 503, str(error))
    if (
        "unsafe" in message
        or "symlink" in message
        or "outside" in message
        or "denied" in message
    ):
        return RadarUnavailableError("unsafe-source", 404, str(error))
    if reason in {"view_closed", "source_error"}:
        return RadarUnavailableError("source-unavailable", 503, str(error))
    if isinstance(error, ImportError):
        return RadarUnavailableError("backend-unavailable", 503, str(error))
    if isinstance(error, ValueError):
        return RadarUnavailableError("invalid-source", 400, str(error))
    return RadarUnavailableError("backend-failure", 500, str(error))


def _canonical_index(value: str) -> int | None:
    """Return a non-negative decimal chunk index without leading zeros."""
    if not value.isascii() or not value.isdigit():
        return None
    if len(value) > 1 and value.startswith("0"):
        return None
    return int(value)


def _properties(record: Any) -> Mapping[str, Any]:
    """Return a record Item properties mapping, or an empty mapping."""
    item = getattr(record, "item", {})
    properties = item.get("properties") if isinstance(item, Mapping) else None
    return properties if isinstance(properties, Mapping) else {}


def _assets(record: Any) -> Mapping[str, Any]:
    """Return a record Item assets mapping, or an empty mapping."""
    item = getattr(record, "item", {})
    assets = item.get("assets") if isinstance(item, Mapping) else None
    return assets if isinstance(assets, Mapping) else {}


def _inputs(record: Any) -> dict[str, str]:
    """Return recorded upstream write ids as a string mapping."""
    raw = _properties(record).get("faninsar:inputs", {})
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(name): str(value)
        for name, value in raw.items()
        if isinstance(name, str) and isinstance(value, str)
    }


def _text_property(record: Any, name: str) -> str | None:
    """Return one non-empty string property, or ``None``."""
    value = _properties(record).get(name)
    return value if isinstance(value, str) and value else None


def _shape_property(
    record: Any, name: str = "faninsar:shape"
) -> tuple[int, int] | None:
    """Return one validated two-element integer shape property."""
    value = _properties(record).get(name)
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    try:
        shape = (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None
    return shape if min(shape) > 0 else None


def _radar(record: Any) -> bool:
    """Return whether the record declares the radar coordinate domain."""
    return _text_property(record, "faninsar:domain") == "radar"
