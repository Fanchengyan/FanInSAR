"""HTTP-only georeferenced Zarr view helpers for the STAC service (P0089).

The service keeps canonical Item assets untouched and serves supported
two-dimensional geographic Zarr arrays under their original scientific asset
key. The HTTP copy of that asset points at the virtual georeferenced view while
the original native href is retained in ``faninsar:native_href``. This module
owns the view naming, the fixed discovery manifest, per-generation descriptor
caching, and Item augmentation. Filesystem safety, receipt revalidation, and
HTTP range serving stay in :mod:`faninsar.stack._stac_service` and are supplied
as callbacks so this module contains no path or descriptor handling of its own.
"""

from __future__ import annotations

import html
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote

__all__ = [
    "GEOREF_MEDIA_TYPE",
    "METADATA_PATHS",
    "GeorefEntry",
    "GeorefProvider",
    "manifest_html",
    "parse_view_tail",
    "view_path",
]

#: Media type used by the derived Zarr view asset.
GEOREF_MEDIA_TYPE = "application/vnd+zarr"

#: The only generated metadata documents, in fixed manifest order.
METADATA_PATHS = (
    "zarr.json",
    "values/zarr.json",
    "x/zarr.json",
    "y/zarr.json",
)

#: Relative children advertised by the synthetic discovery manifest.
_MANIFEST_CHILDREN = ("zarr.json", "values/", "x/", "y/")

#: Native payload facts that do not describe the virtual group.
_NATIVE_ASSET_FIELDS = frozenset({"file:checksum", "file:size"})

#: Suffix of the virtual store directory below ``/zarr/{public_id}/``.
_VIEW_SUFFIX = ".zarr"


@dataclass(frozen=True, slots=True)
class GeorefEntry:
    """One cached view descriptor or its unsupported diagnostic."""

    view: Any | None
    modified: float = 0.0
    reason: str | None = None
    detail: str | None = None


class GeorefProvider:
    """Build, cache, and describe HTTP-only georeferenced Zarr views.

    Parameters
    ----------
    open_metadata : callable
        ``open_metadata(record, asset_key)`` returns the parsed native
        ``zarr.json`` mapping and its modification time, or ``(None, 0.0)``.
        The service implementation performs the pinned safe open and checks
        the current receipt write id before and after the read.
    receipt_valid : callable
        ``receipt_valid(record)`` returns whether the product receipt still
        matches the record's write id.
    build_view : callable
        ``build_view(record, asset_key, metadata)`` returns the metadata
        worker's validated descriptor or raises its reason/detail error.

    """

    def __init__(
        self,
        *,
        open_metadata: Callable[[Any, str], tuple[Mapping[str, Any] | None, float]],
        receipt_valid: Callable[[Any], bool],
        build_view: Callable[[Any, str, Mapping[str, Any]], Any | None],
    ) -> None:
        self._open_metadata = open_metadata
        self._receipt_valid = receipt_valid
        self._build_view = build_view
        self._lock = threading.Lock()
        self._generation: str | None = None
        self._entries: dict[tuple[str, str], GeorefEntry] = {}

    def augment_item(
        self,
        record: Any,
        item: dict[str, Any],
        base_url: str,
        generation: str,
    ) -> None:
        """Point supported source-key assets at their georeferenced view.

        Only the mutable HTTP Item copy changes. Unsupported assets, including
        radar-coordinate products, keep their original raw asset document.
        """
        assets = item.get("assets")
        if not isinstance(assets, dict) or not _geographic(record):
            return
        for source_key in sorted(record.asset_paths):
            source_asset = assets.get(source_key)
            if not isinstance(source_asset, dict):
                continue
            entry = self._entry(record, source_key, generation)
            if entry is None or entry.view is None:
                continue
            assets[source_key] = _asset_document(
                record, source_key, source_asset, entry.view, base_url
            )

    def resolve(
        self,
        record: Any,
        source_key: str,
        generation: str,
    ) -> tuple[Any, float] | None:
        """Return a view after revalidating the current source receipt.

        The descriptor itself is cached per generation, public id, and source
        asset key. The native metadata is still read once per request through
        the pinned safe-open callback, and the receipt is checked again before
        the caller generates response bytes.
        """
        if not _geographic(record) or source_key not in record.asset_paths:
            return None
        metadata, modified = self._open_metadata(record, source_key)
        if metadata is None:
            return None
        entry = self._cached(record, source_key, generation)
        if entry is None:
            entry = self._build(record, source_key, metadata, modified)
            self._store(record, source_key, generation, entry)
        if entry.view is None or not self._receipt_valid(record):
            return None
        return entry.view, modified

    def _entry(
        self, record: Any, source_key: str, generation: str
    ) -> GeorefEntry | None:
        """Return a cached entry, building and caching one on a miss."""
        entry = self._cached(record, source_key, generation)
        if entry is not None:
            return entry
        metadata, modified = self._open_metadata(record, source_key)
        if metadata is None:
            entry = GeorefEntry(
                view=None,
                reason="unsupported",
                detail="native zarr.json is missing or unsafe",
            )
        else:
            entry = self._build(record, source_key, metadata, modified)
        self._store(record, source_key, generation, entry)
        return entry

    def _build(
        self,
        record: Any,
        source_key: str,
        metadata: Mapping[str, Any],
        modified: float,
    ) -> GeorefEntry:
        """Call the metadata worker and validate its descriptor interface."""
        try:
            view = self._build_view(record, source_key, metadata)
        except Exception as error:
            return GeorefEntry(
                view=None,
                reason=str(getattr(error, "reason", type(error).__name__)),
                detail=str(getattr(error, "detail", error)),
            )
        if view is None or not _descriptor_ready(view):
            return GeorefEntry(
                view=None,
                reason="unsupported",
                detail="georeferenced view descriptor is not available",
            )
        return GeorefEntry(view=view, modified=modified)

    def _cached(
        self, record: Any, source_key: str, generation: str
    ) -> GeorefEntry | None:
        """Return a cached entry for the current generation, if any."""
        with self._lock:
            self._sync_generation(generation)
            return self._entries.get((record.public_id, source_key))

    def _store(
        self,
        record: Any,
        source_key: str,
        generation: str,
        entry: GeorefEntry,
    ) -> None:
        """Cache one entry for the current generation."""
        with self._lock:
            self._sync_generation(generation)
            self._entries[(record.public_id, source_key)] = entry

    def _sync_generation(self, generation: str) -> None:
        """Drop entries when the served index generation changes."""
        if generation != self._generation:
            self._entries.clear()
            self._generation = generation


def view_path(public_id: str, source_key: str) -> str:
    """Return the absolute-path href of one virtual Zarr store."""
    return (
        f"/zarr/{quote(public_id, safe='')}/{quote(source_key, safe='')}{_VIEW_SUFFIX}/"
    )


def parse_view_tail(
    asset_keys: Iterable[str], tail: str
) -> tuple[str, str, bool] | None:
    """Split a decoded route tail into ``(source_key, relative, slashless)``.

    The source key is matched against the record's registered asset keys so a
    key containing periods is accepted without ambiguous suffix parsing.
    ``slashless`` is true for the bare virtual store root, which the caller
    redirects to its trailing-slash form.
    """
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
    """Return the fixed four-entry discovery manifest for one virtual group.

    The document contains no host name, absolute filesystem path, timestamp,
    or filesystem-derived entry. GDAL's Zarr driver discovers the fixed child
    set from these relative links; generic directory listing stays disabled.
    """
    title = html.escape(
        f"Index of /zarr/{public_id}/{source_key}{_VIEW_SUFFIX}/",
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


def _asset_document(
    record: Any,
    source_key: str,
    source_asset: Mapping[str, Any],
    view: Any,
    base_url: str,
) -> dict[str, Any]:
    """Build the view asset under its original scientific asset key.

    The descriptor's internal derived asset key is not used here: presentation
    keeps the source key. Source metadata such as the title and roles is
    preserved, but native payload facts such as ``file:checksum`` and
    ``file:size`` are omitted because they describe the raw store, not the
    virtual group. The original native href moves to ``faninsar:native_href``.
    """
    href_path = view_path(record.public_id, source_key)
    transform = [float(value) for value in view.transform]
    if len(transform) == 6:
        transform.extend((0.0, 0.0, 1.0))
    document = {
        key: value
        for key, value in source_asset.items()
        if key not in _NATIVE_ASSET_FIELDS
    }
    document.update(
        {
            "href": f"{base_url.rstrip('/')}{href_path}",
            "type": GEOREF_MEDIA_TYPE,
            "title": _asset_title(source_asset, source_key),
            "roles": document.get("roles", ["data"]),
            "proj:code": str(view.crs),
            "proj:shape": [int(value) for value in view.shape],
            "proj:transform": transform,
            "faninsar:source_asset": source_key,
            "faninsar:gdal_uri": (
                f'ZARR:"/vsicurl/{base_url.rstrip("/")}{href_path}":/values'
            ),
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


def _geographic(record: Any) -> bool:
    """Return whether normalization published a geographic Item."""
    item = getattr(record, "item", {})
    if not isinstance(item, Mapping) or item.get("geometry") is None:
        return False
    properties = item.get("properties")
    if not isinstance(properties, Mapping):
        return False
    return "proj:code" in properties and "proj:transform" in properties


def _descriptor_ready(view: Any) -> bool:
    """Return whether the metadata worker descriptor exposes the contract."""
    shape = getattr(view, "shape", None)
    transform = getattr(view, "transform", None)
    crs = getattr(view, "crs", None)
    data_type = getattr(view, "data_type", None)
    if not isinstance(crs, str) or not crs:
        return False
    if not isinstance(data_type, str) or not data_type:
        return False
    if (
        not isinstance(shape, tuple | list)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
        or any(value <= 0 for value in shape)
    ):
        return False
    if not isinstance(transform, tuple | list) or len(transform) != 6:
        return False
    if any(
        isinstance(value, bool) or not isinstance(value, int | float)
        for value in transform
    ):
        return False
    return all(
        callable(getattr(view, name, None))
        for name in ("metadata", "coordinate_chunk", "source_chunk")
    )
