"""Read-only Stack STAC API and range-capable asset service (PROPOSAL-0089).

The service adapts the disposable normalized index owned by ``faninsar.io.stac``
to loopback HTTP. It never writes scientific products: derived portable
catalogs are written only after the listening socket is bound, and only when
the caller asks for them. ``aiohttp`` and the ``io.stac`` modules are imported
lazily so core imports and Stack science never load the serving runtime.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import socket
import stat
import threading
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from email.utils import formatdate, parsedate_to_datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Self
from urllib.parse import quote, unquote, urlencode

from faninsar.logging import setup_logger

__all__ = [
    "StacRefreshError",
    "StacRefreshReport",
    "StackStacService",
    "start_stac_service",
]

logger = setup_logger(__name__)

CONFORMANCE: tuple[str, ...] = (
    "https://api.stacspec.org/v1.0.0/core",
    "https://api.stacspec.org/v1.0.0/item-search",
    "https://api.stacspec.org/v1.0.0/collections",
)

_CORS_EXPOSE_NAMES = (
    "Accept-Ranges",
    "Content-Range",
    "Content-Length",
    "Content-Type",
    "ETag",
    "Last-Modified",
)
_CORS_EXPOSE_HEADERS = ", ".join(_CORS_EXPOSE_NAMES)
_CORS_HEADERS: tuple[tuple[str, str], ...] = (
    ("Access-Control-Allow-Origin", "*"),
    ("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS"),
    (
        "Access-Control-Allow-Headers",
        "Accept, Content-Type, Range, If-Range, If-None-Match, If-Modified-Since",
    ),
    ("Access-Control-Expose-Headers", _CORS_EXPOSE_HEADERS),
    ("Access-Control-Max-Age", "600"),
)

_JSON_MEDIA_TYPE = "application/json"
_GEOJSON_MEDIA_TYPE = "application/geo+json"
_OPENAPI_MEDIA_TYPE = "application/vnd.oai.openapi+json;version=3.0"
_MAX_ITEM_BYTES = 4 * 1024 * 1024
_STREAM_CHUNK = 1024 * 1024
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_SEARCH_PARAMETERS = frozenset(
    {"ids", "collections", "bbox", "intersects", "datetime", "limit", "token"}
)
_MAX_LIMIT = 1000
_DEFAULT_LIMIT = 10
_READY_TIMEOUT = 30.0
_STOP_TIMEOUT = 10.0

_WEB_KEY = "faninsar:stac:web"
_IO_KEY = "faninsar:stac:io"
_STATE_KEY = "faninsar:stac:state"
_WORK_DIR_KEY = "faninsar:stac:work_dir"
_BASE_URL_KEY = "faninsar:stac:base_url"


class StacRefreshError(RuntimeError):
    """Raised when a requested STAC index refresh cannot be built.

    Parameters
    ----------
    generation : str
        The retained generation that continues to be served.
    error : BaseException
        The underlying build failure.

    """

    def __init__(self, generation: str, error: BaseException) -> None:
        msg = (
            f"STAC index refresh failed; generation {generation!r} is still "
            f"served: {error}"
        )
        super().__init__(msg)
        self.generation = generation
        self.error = error


class _ApiError(Exception):
    """One JSON HTTP error response."""

    def __init__(self, status: int, code: str, description: str) -> None:
        super().__init__(description)
        self.status = status
        self.code = code
        self.description = description


class _UnsafeAssetPathError(Exception):
    """An asset path is absolute, traversing, symlinked, or otherwise unsafe."""


class _IdentityMismatchError(Exception):
    """The validated path and the opened descriptor describe different files."""


class _InvalidRangeError(Exception):
    """A Range header is syntactically malformed."""


class _UnsatisfiableRangeError(Exception):
    """A Range header cannot be satisfied for the current representation."""


@dataclass(frozen=True, slots=True)
class StacRefreshReport:
    """Outcome of one service startup or explicit refresh.

    Attributes
    ----------
    generation : str
        Generation digest of the index now served.
    item_count : int
        Number of admitted public Items.
    collection_count : int
        Number of real extent-backed Collections.
    families : mapping of str to int
        Admitted Item count per family.
    exclusions : tuple
        Index exclusions that were not published.
    persistence : CatalogPersistenceReport
        Portable-catalog outcome for this generation.

    """

    generation: str
    item_count: int
    collection_count: int
    families: Mapping[str, int]
    exclusions: tuple[Any, ...]
    persistence: Any

    def as_dict(self) -> dict[str, Any]:
        """Return a strict-JSON mapping, raising on non-finite values."""
        payload = {
            "generation": self.generation,
            "item_count": self.item_count,
            "collection_count": self.collection_count,
            "families": dict(self.families),
            "exclusions": [_jsonable(exclusion) for exclusion in self.exclusions],
            "persistence": _jsonable(self.persistence),
        }
        _strict_dumps(payload)
        return payload


@dataclass(frozen=True, slots=True)
class _StackIO:
    """Imported ``faninsar.io.stac`` contract used by the HTTP adapter."""

    build_index: Any
    write_portable_catalog: Any
    catalog_report: Any
    link_targets: Any
    render_item: Any
    render_root_catalog: Any
    render_family_catalog: Any
    render_collection: Any
    search_request: Any
    search_error: Any


def _load_web() -> Any:
    """Import ``aiohttp.web`` lazily with a descriptive failure message."""
    try:
        from aiohttp import web
    except ImportError as error:
        msg = (
            "the FanInSAR STAC service requires the 'aiohttp' package, which "
            "is a required FanInSAR dependency; reinstall FanInSAR in its "
            "supported environment"
        )
        raise ImportError(msg) from error
    return web


def _load_io() -> _StackIO:
    """Import the ``faninsar.io.stac`` worker contract lazily."""
    try:
        index_module = import_module("faninsar.io.stac.index")
        catalog_module = import_module("faninsar.io.stac.catalog")
        render_module = import_module("faninsar.io.stac.render")
        search_module = import_module("faninsar.io.stac.search")
        import_module("faninsar.io.stac.record")
        return _StackIO(
            build_index=index_module.build_index,
            write_portable_catalog=catalog_module.write_portable_catalog,
            catalog_report=catalog_module.CatalogPersistenceReport,
            link_targets=render_module.LinkTargets,
            render_item=render_module.render_item,
            render_root_catalog=render_module.render_root_catalog,
            render_family_catalog=render_module.render_family_catalog,
            render_collection=render_module.render_collection,
            search_request=search_module.SearchRequest,
            search_error=search_module.StacSearchError,
        )
    except (ImportError, AttributeError) as error:
        msg = (
            "the FanInSAR STAC service requires the faninsar.io.stac package "
            "(build_index, search, renderers, and portable catalog support); "
            f"the installed FanInSAR tree cannot provide it: {error}"
        )
        raise ImportError(msg) from error


def _reject_json_constant(value: str) -> None:
    """Reject NaN and infinity while parsing strict JSON."""
    msg = f"non-finite JSON constant {value!r} is not allowed"
    raise ValueError(msg)


def _strict_loads(data: bytes) -> Any:
    """Parse strict UTF-8 JSON, rejecting non-finite constants."""
    return json.loads(data.decode("utf-8"), parse_constant=_reject_json_constant)


def _strict_dumps(payload: Any) -> str:
    """Serialize compact strict JSON, rejecting non-finite numbers."""
    return json.dumps(payload, allow_nan=False, separators=(",", ":"))


def _jsonable(value: Any) -> Any:
    """Convert dataclasses, paths, and mappings into strict JSON values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _is_loopback(host: str) -> bool:
    """Return whether ``host`` names the local machine."""
    return host in _LOOPBACK_HOSTS


def _base_url(host: str, port: int) -> str:
    """Return the reachable base URL for a bound socket, without a trailing slash."""
    display = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{display}:{port}"


def _validate_port(port: int) -> None:
    """Reject ports outside the TCP range."""
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        msg = f"port must be an integer between 0 and 65535, not {port!r}"
        raise ValueError(msg)


def _bind_socket(host: str, port: int) -> socket.socket:
    """Reserve and bind one listening socket before any derived write."""
    _validate_port(port)
    if not host:
        msg = "host must be a non-empty string"
        raise ValueError(msg)
    try:
        candidates = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        msg = f"cannot resolve STAC service host {host!r}: {error}"
        raise OSError(msg) from error
    last_error: OSError | None = None
    for family, socktype, protocol, _name, address in candidates:
        sock = socket.socket(family, socktype, protocol)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(address)
            sock.setblocking(False)
        except OSError as error:
            last_error = error
            sock.close()
            continue
        return sock
    if last_error is None:
        msg = f"cannot bind STAC service host {host!r}"
        raise OSError(msg)
    raise last_error


def _safe_asset_parts(relative: str) -> tuple[str, ...]:
    """Validate one percent-decoded relative asset tail without traversal.

    Parameters
    ----------
    relative : str
        The route tail, possibly still percent-encoded.

    Returns
    -------
    tuple of str
        Normalized components; an empty tuple denotes the asset root.

    Raises
    ------
    _UnsafeAssetPathError
        If the tail is absolute, contains ``..`` or null bytes, or has empty
        or ``.`` components.

    """
    decoded = unquote(relative)
    if "\x00" in decoded or "\\" in decoded:
        raise _UnsafeAssetPathError(relative)
    if decoded.startswith("/"):
        raise _UnsafeAssetPathError(relative)
    if not decoded:
        return ()
    parts = tuple(decoded.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise _UnsafeAssetPathError(relative)
    return parts


def _open_at(
    parent_fd: int, name: str, *, directory: bool | None
) -> tuple[int, os.stat_result]:
    """Open one child without following symlinks and verify its identity."""
    if name in {"", ".", ".."} or "\x00" in name:
        raise _UnsafeAssetPathError(name)
    flags = os.O_RDONLY | _NOFOLLOW | _NONBLOCK | _CLOEXEC
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode):
        raise _UnsafeAssetPathError(name)
    is_directory = stat.S_ISDIR(before.st_mode)
    if directory is True and not is_directory:
        raise _UnsafeAssetPathError(name)
    if directory is False and not stat.S_ISREG(before.st_mode):
        raise _UnsafeAssetPathError(name)
    if directory is True or (directory is None and is_directory):
        flags |= os.O_DIRECTORY
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    after = os.fstat(descriptor)
    if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
        os.close(descriptor)
        raise _IdentityMismatchError(name)
    return descriptor, after


def _open_parts(
    root_fd: int, parts: tuple[str, ...], *, final_directory: bool | None
) -> tuple[int, os.stat_result]:
    """Open a non-empty relative component chain below one pinned descriptor."""
    current = root_fd
    owned = False
    try:
        for part in parts[:-1]:
            descriptor, _ = _open_at(current, part, directory=True)
            if owned:
                os.close(current)
            current = descriptor
            owned = True
        return _open_at(current, parts[-1], directory=final_directory)
    finally:
        if owned:
            os.close(current)


def _open_directory(work_dir: Path, parts: tuple[str, ...]) -> int:
    """Open a directory below ``work_dir`` without following any symlink."""
    flags = os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW | _CLOEXEC
    if not parts:
        return os.open(work_dir, flags)
    root_fd = os.open(work_dir, flags)
    try:
        descriptor, _ = _open_parts(root_fd, parts, final_directory=True)
    finally:
        os.close(root_fd)
    return descriptor


def _read_write_id(product_fd: int) -> str | None:
    """Read a bounded source ``item.json`` receipt from a pinned product directory."""
    try:
        descriptor, metadata = _open_at(product_fd, "item.json", directory=False)
    except (OSError, _UnsafeAssetPathError, _IdentityMismatchError):
        return None
    try:
        if metadata.st_size > _MAX_ITEM_BYTES:
            return None
        chunks: list[bytes] = []
        remaining = _MAX_ITEM_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if len(data) > _MAX_ITEM_BYTES:
        return None
    try:
        payload = _strict_loads(data)
    except (UnicodeDecodeError, ValueError):
        return None
    properties = payload.get("properties") if isinstance(payload, dict) else None
    write_id = (
        properties.get("faninsar:write_id") if isinstance(properties, dict) else None
    )
    return write_id if isinstance(write_id, str) and write_id else None


def _product_directory(work_dir: Path, record: Any) -> Path:
    """Return the lexical product directory rooted at ``work_dir``."""
    product_dir = Path(record.source_item_path).parent
    return product_dir if product_dir.is_absolute() else work_dir / product_dir


def _product_parts(work_dir: Path, record: Any) -> tuple[str, ...]:
    """Return the product directory components relative to ``work_dir``."""
    product_dir = _product_directory(work_dir, record)
    try:
        relative = product_dir.relative_to(work_dir)
    except ValueError:
        try:
            relative = product_dir.resolve().relative_to(work_dir)
        except (OSError, ValueError) as error:
            raise _UnsafeAssetPathError(str(product_dir)) from error
    parts = tuple(relative.parts)
    if any(part in {"", ".", ".."} for part in parts):
        raise _UnsafeAssetPathError(str(product_dir))
    return parts


def _open_asset(
    work_dir: Path, product_fd: int, record: Any, asset_key: str, relative: str
) -> tuple[int, os.stat_result]:
    """Open one registered asset beneath a pinned product descriptor."""
    asset_root = Path(record.asset_paths[asset_key])
    product_dir = _product_directory(work_dir, record)
    if not asset_root.is_absolute():
        asset_root = product_dir / asset_root
    try:
        root_relative = asset_root.relative_to(product_dir)
    except ValueError as error:
        raise _UnsafeAssetPathError(str(asset_root)) from error
    root_parts = tuple(root_relative.parts)
    if not root_parts or any(part in {"", ".", ".."} for part in root_parts):
        raise _UnsafeAssetPathError(str(asset_root))
    if root_parts[-1] == "item.json":
        raise _UnsafeAssetPathError(str(asset_root))
    tail = _safe_asset_parts(relative)
    descriptor, metadata = _open_parts(product_fd, root_parts, final_directory=None)
    if stat.S_ISDIR(metadata.st_mode):
        if not tail:
            os.close(descriptor)
            raise _UnsafeAssetPathError(str(asset_root))
        try:
            return _open_parts(descriptor, tail, final_directory=False)
        finally:
            os.close(descriptor)
    if stat.S_ISREG(metadata.st_mode):
        if tail:
            os.close(descriptor)
            raise _UnsafeAssetPathError(str(asset_root))
        return descriptor, metadata
    os.close(descriptor)
    raise _UnsafeAssetPathError(str(asset_root))


def _parse_byte_range(value: str, size: int) -> tuple[int, int]:
    """Parse one ``bytes=`` Range value into an inclusive interval.

    Parameters
    ----------
    value : str
        Raw ``Range`` header value.
    size : int
        Full representation length in bytes.

    Returns
    -------
    tuple of int
        Inclusive ``(start, end)`` offsets.

    Raises
    ------
    _InvalidRangeError
        If the header is malformed.
    _UnsatisfiableRangeError
        If the requested interval is unsatisfiable or multi-range.

    """
    unit, separator, spec = value.partition("=")
    if not separator or unit.strip().lower() != "bytes":
        raise _UnsatisfiableRangeError(value)
    spec = spec.strip()
    if "," in spec:
        raise _UnsatisfiableRangeError(value)
    start_text, dash, end_text = spec.partition("-")
    if not dash:
        raise _InvalidRangeError(value)
    if not start_text and not end_text:
        raise _InvalidRangeError(value)
    if not start_text:
        if not end_text.isdigit():
            raise _InvalidRangeError(value)
        suffix = int(end_text)
        if suffix <= 0 or size <= 0:
            raise _UnsatisfiableRangeError(value)
        start = max(0, size - suffix)
        return start, size - 1
    if not start_text.isdigit() or (end_text and not end_text.isdigit()):
        raise _InvalidRangeError(value)
    start = int(start_text)
    end = int(end_text) if end_text else size - 1
    if start > end or start >= size:
        raise _UnsatisfiableRangeError(value)
    return start, min(end, size - 1)


def _asset_etag(write_id: str, asset_key: str, relative: str) -> str:
    """Return a strong validator scoped to the write id and asset path."""
    material = f"{write_id}\x00{asset_key}\x00{relative}".encode()
    return f'"{hashlib.sha256(material).hexdigest()}"'


def _weak_tag(value: str) -> str:
    """Return an entity tag without its weak prefix."""
    return value.removeprefix("W/")


def _if_none_match_matches(header: str, etag: str) -> bool:
    """Return whether an ``If-None-Match`` value matches the current tag."""
    for candidate in header.split(","):
        tag = candidate.strip()
        if tag == "*" or _weak_tag(tag) == _weak_tag(etag):
            return True
    return False


def _if_range_matches(header: str, etag: str, modified: float) -> bool:
    """Return whether an ``If-Range`` value permits a partial response."""
    value = header.strip()
    if not value:
        return False
    if value.startswith('"'):
        return value == etag
    if value.startswith("W/"):
        return False
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(modified) <= int(parsed.timestamp())


def _format_http_date(timestamp: float) -> str:
    """Format one UNIX timestamp as an HTTP date."""
    return formatdate(timestamp, usegmt=True)


def _conditional_status(request: Any, etag: str, modified: float) -> int | None:
    """Return 304 when preconditions allow a not-modified response."""
    if_none_match = request.headers.get("If-None-Match")
    if if_none_match is not None:
        return 304 if _if_none_match_matches(if_none_match, etag) else None
    if_modified_since = request.headers.get("If-Modified-Since")
    if if_modified_since:
        try:
            parsed = parsedate_to_datetime(if_modified_since)
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        if int(modified) <= int(parsed.timestamp()):
            return 304
    return None


def _asset_media_type(record: Any, asset_key: str, relative: str) -> str:
    """Derive a MIME type from the served suffix or canonical asset type."""
    name = relative or Path(str(record.asset_paths[asset_key])).name
    suffix = Path(name).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    if suffix == ".json":
        return _JSON_MEDIA_TYPE
    if not relative:
        assets = getattr(record, "item", {}).get("assets", {})
        declared = assets.get(asset_key, {}) if isinstance(assets, Mapping) else {}
        if isinstance(declared, Mapping):
            media_type = declared.get("type")
            if isinstance(media_type, str) and "/" in media_type:
                return media_type
    return "application/octet-stream"


def _pread(descriptor: int, size: int, offset: int) -> bytes:
    """Read at an offset without moving the shared file position."""
    if hasattr(os, "pread"):
        return os.pread(descriptor, size, offset)
    os.lseek(descriptor, offset, os.SEEK_SET)
    return os.read(descriptor, size)


def _status_code(status: int) -> str:
    """Return a stable error code for an HTTP status."""
    return {
        400: "BadRequest",
        404: "NotFound",
        405: "MethodNotAllowed",
        413: "PayloadTooLarge",
        415: "UnsupportedMediaType",
        416: "RangeNotSatisfiable",
        500: "InternalServerError",
    }.get(status, "HTTPError")


def _json_response(
    web: Any,
    payload: Any,
    *,
    status: int = 200,
    media_type: str = _JSON_MEDIA_TYPE,
) -> Any:
    """Build one strict-JSON aiohttp response."""
    return web.json_response(
        payload,
        status=status,
        content_type=media_type,
        dumps=_strict_dumps,
    )


def _apply_cors(_request: Any, response: Any) -> None:
    """Apply the read-only wildcard CORS policy to every response."""
    for name, value in _CORS_HEADERS:
        response.headers[name] = value


async def _normalise_http_error(request: Any, response: Any) -> None:
    """Render routing-level ``text/plain`` errors as JSON and apply CORS."""
    web = request.app[_WEB_KEY]
    content_type = response.content_type or ""
    if (
        response.status >= 400
        and isinstance(response, web.Response)
        and content_type.startswith("text/plain")
    ):
        response.content_type = _JSON_MEDIA_TYPE
        response.charset = "utf-8"
        response.text = _strict_dumps(
            {
                "code": _status_code(response.status),
                "description": response.reason or "request failed",
            }
        )
    _apply_cors(request, response)


async def _error_middleware(request: Any, handler: Any) -> Any:
    """Map service errors to JSON responses without leaking tracebacks."""
    web = request.app[_WEB_KEY]
    try:
        return await handler(request)
    except _ApiError as error:
        return _json_response(
            web,
            {"code": error.code, "description": error.description},
            status=error.status,
        )
    except web.HTTPException as error:
        response = _json_response(
            web,
            {"code": _status_code(error.status), "description": error.reason},
            status=error.status,
        )
        for name, value in error.headers.items():
            if name.lower() != "content-type":
                response.headers[name] = value
        return response
    except Exception:
        logger.exception(
            "unexpected STAC request failure for %s %s",
            request.method,
            request.path,
        )
        return _json_response(
            web,
            {
                "code": "InternalServerError",
                "description": "unexpected server failure",
            },
            status=500,
        )


def _http_targets(io: _StackIO, base_url: str, view_path: str) -> Any:
    """Return HTTP link targets for one rendered view."""
    return io.link_targets.for_http(base_url=base_url, view_path=view_path)


async def _landing(request: Any) -> Any:
    """Serve the API landing Catalog."""
    web = request.app[_WEB_KEY]
    io = request.app[_IO_KEY]
    state = request.app[_STATE_KEY]
    base_url = request.app[_BASE_URL_KEY]
    index = state.snapshot()
    root = io.render_root_catalog(index, _http_targets(io, base_url, "/"))
    links = [
        {"rel": "self", "href": f"{base_url}/", "type": _JSON_MEDIA_TYPE},
        {"rel": "root", "href": f"{base_url}/", "type": _JSON_MEDIA_TYPE},
        {
            "rel": "conformance",
            "href": f"{base_url}/conformance",
            "type": _JSON_MEDIA_TYPE,
        },
        {"rel": "data", "href": f"{base_url}/collections", "type": _JSON_MEDIA_TYPE},
        {
            "rel": "service-desc",
            "href": f"{base_url}/api",
            "type": _OPENAPI_MEDIA_TYPE,
        },
        {
            "rel": "search",
            "href": f"{base_url}/search",
            "type": _GEOJSON_MEDIA_TYPE,
            "method": "GET",
        },
        {
            "rel": "search",
            "href": f"{base_url}/search",
            "type": _GEOJSON_MEDIA_TYPE,
            "method": "POST",
        },
    ]
    links.extend(
        {
            "rel": "child",
            "href": f"{base_url}/catalogs/{quote(str(family), safe='')}.json",
            "type": _JSON_MEDIA_TYPE,
            "title": str(family),
        }
        for family in index.families()
    )
    for collection in index.collections:
        collection_id = quote(str(collection.collection_id), safe="")
        links.append(
            {
                "rel": "child",
                "href": f"{base_url}/collections/{collection_id}",
                "type": _JSON_MEDIA_TYPE,
                "title": str(collection.collection_id),
            }
        )
    document = {
        "type": "Catalog",
        "stac_version": "1.0.0",
        "id": root["id"],
        "description": root["description"],
        "conformsTo": list(CONFORMANCE),
        "links": links,
    }
    if isinstance(root.get("title"), str):
        document["title"] = root["title"]
    return _json_response(web, document)


async def _root_catalog(request: Any) -> Any:
    """Serve the root portable Catalog rendered with HTTP hrefs."""
    web = request.app[_WEB_KEY]
    io = request.app[_IO_KEY]
    index = request.app[_STATE_KEY].snapshot()
    base_url = request.app[_BASE_URL_KEY]
    document = io.render_root_catalog(
        index, _http_targets(io, base_url, "/catalog.json")
    )
    return _json_response(web, document)


async def _conformance(request: Any) -> Any:
    """Serve the advertised STAC API conformance classes."""
    web = request.app[_WEB_KEY]
    return _json_response(web, {"conformsTo": list(CONFORMANCE)})


async def _api(request: Any) -> Any:
    """Serve the OpenAPI 3.0 service description."""
    web = request.app[_WEB_KEY]
    base_url = request.app[_BASE_URL_KEY]
    return _json_response(
        web, _openapi_document(base_url), media_type=_OPENAPI_MEDIA_TYPE
    )


async def _family_catalog(request: Any) -> Any:
    """Serve one family Catalog view."""
    web = request.app[_WEB_KEY]
    io = request.app[_IO_KEY]
    index = request.app[_STATE_KEY].snapshot()
    base_url = request.app[_BASE_URL_KEY]
    family = request.match_info["family"]
    if family not in index.families():
        raise _ApiError(404, "NotFound", f"unknown family: {family}")
    view = f"/catalogs/{quote(family, safe='')}.json"
    document = io.render_family_catalog(
        index, family, _http_targets(io, base_url, view)
    )
    return _json_response(web, document)


async def _collections(request: Any) -> Any:
    """Serve the real extent-backed Collections."""
    web = request.app[_WEB_KEY]
    io = request.app[_IO_KEY]
    index = request.app[_STATE_KEY].snapshot()
    base_url = request.app[_BASE_URL_KEY]
    rendered = []
    for collection in index.collections:
        collection_id = quote(str(collection.collection_id), safe="")
        rendered.append(
            io.render_collection(
                collection,
                index,
                _http_targets(io, base_url, f"/collections/{collection_id}"),
            )
        )
    document = {
        "collections": rendered,
        "links": [
            {
                "rel": "self",
                "href": f"{base_url}/collections",
                "type": _JSON_MEDIA_TYPE,
            },
            {"rel": "root", "href": f"{base_url}/", "type": _JSON_MEDIA_TYPE},
        ],
    }
    return _json_response(web, document)


async def _collection(request: Any) -> Any:
    """Serve one real Collection."""
    web = request.app[_WEB_KEY]
    io = request.app[_IO_KEY]
    index = request.app[_STATE_KEY].snapshot()
    base_url = request.app[_BASE_URL_KEY]
    collection_id = request.match_info["collection_id"]
    collection = index.collection(collection_id)
    if collection is None:
        raise _ApiError(404, "NotFound", f"unknown collection: {collection_id}")
    view = f"/collections/{quote(collection_id, safe='')}"
    document = io.render_collection(
        collection, index, _http_targets(io, base_url, view)
    )
    return _json_response(web, document)


async def _item(request: Any) -> Any:
    """Serve one public Item."""
    web = request.app[_WEB_KEY]
    io = request.app[_IO_KEY]
    index = request.app[_STATE_KEY].snapshot()
    base_url = request.app[_BASE_URL_KEY]
    public_id = request.match_info["public_id"]
    record = index.record(public_id)
    if record is None:
        raise _ApiError(404, "NotFound", f"unknown item: {public_id}")
    view = f"/items/{quote(public_id, safe='')}"
    document = io.render_item(record, _http_targets(io, base_url, view))
    return _json_response(web, document, media_type=_GEOJSON_MEDIA_TYPE)


def _single_query(request: Any, name: str) -> str | None:
    """Return one query value, rejecting repeated singletons."""
    values = request.query.getall(name, [])
    if len(values) > 1:
        raise _ApiError(400, "InvalidParameter", f"{name} must occur once")
    return values[0] if values else None


def _split_query_list(request: Any, name: str) -> tuple[str, ...] | None:
    """Parse one comma-separated string list parameter."""
    raw = _single_query(request, name)
    if raw is None:
        return None
    parts = tuple(part.strip() for part in raw.split(","))
    if not parts or any(not part for part in parts):
        raise _ApiError(400, "InvalidParameter", f"{name} must not be empty")
    return parts


def _parse_bbox_values(values: tuple[str, ...] | None) -> tuple[float, ...] | None:
    """Parse four or six finite CRS84 numbers."""
    if values is None:
        return None
    if len(values) not in {4, 6}:
        msg = "bbox must contain four or six numbers"
        raise _ApiError(400, "InvalidParameter", msg)
    try:
        numbers = tuple(float(value) for value in values)
    except ValueError as error:
        msg = "bbox must contain only numbers"
        raise _ApiError(400, "InvalidParameter", msg) from error
    if any(not _is_finite(number) for number in numbers):
        raise _ApiError(400, "InvalidParameter", "bbox values must be finite")
    return numbers


def _is_finite(value: float) -> bool:
    """Return whether a float is neither NaN nor infinite."""
    return math.isfinite(value)


def _parse_limit(raw: str | None) -> int:
    """Parse and clamp the advertised limit parameter."""
    if raw is None:
        return _DEFAULT_LIMIT
    try:
        value = int(raw)
    except ValueError as error:
        msg = "limit must be an integer"
        raise _ApiError(400, "InvalidParameter", msg) from error
    if value < 1:
        raise _ApiError(400, "InvalidParameter", "limit must be at least 1")
    return min(value, _MAX_LIMIT)


def _parse_datetime(raw: str | None) -> str | None:
    """Validate one non-empty datetime parameter."""
    if raw is None:
        return None
    if not raw:
        raise _ApiError(400, "InvalidParameter", "datetime must not be empty")
    return raw


def _parse_intersects(raw: str | None) -> Mapping[str, Any] | None:
    """Decode one GeoJSON geometry object from a query string."""
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except ValueError as error:
        msg = "intersects must be a JSON-encoded GeoJSON geometry"
        raise _ApiError(400, "InvalidParameter", msg) from error
    if not isinstance(value, Mapping):
        raise _ApiError(400, "InvalidParameter", "intersects must be a GeoJSON object")
    return value


def _reject_unknown_query(request: Any) -> None:
    """Reject every unsupported Item Search query parameter."""
    unknown = sorted(set(request.query.keys()) - _SEARCH_PARAMETERS)
    if unknown:
        msg = f"unsupported search parameter(s): {', '.join(unknown)}"
        raise _ApiError(400, "InvalidParameter", msg)


def _search_request_from_get(
    io: _StackIO, request: Any
) -> tuple[Any, tuple[tuple[str, str], ...]]:
    """Parse a GET Item Search request into the io contract."""
    _reject_unknown_query(request)
    ids = _split_query_list(request, "ids")
    collections = _split_query_list(request, "collections")
    bbox = _parse_bbox_values(_split_query_list(request, "bbox"))
    intersects = _parse_intersects(_single_query(request, "intersects"))
    if bbox is not None and intersects is not None:
        raise _ApiError(
            400,
            "InvalidParameter",
            "bbox and intersects are mutually exclusive",
        )
    parsed = io.search_request(
        ids=ids,
        collections=collections,
        bbox=bbox,
        intersects=intersects,
        datetime=_parse_datetime(_single_query(request, "datetime")),
        limit=_parse_limit(_single_query(request, "limit")),
        token=_single_query(request, "token"),
    )
    pairs = tuple(
        (key, value) for key, value in request.query.items() if key != "token"
    )
    return parsed, pairs


def _body_sequence(value: Any, name: str) -> tuple[str, ...] | None:
    """Parse a JSON string list or comma-separated string."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = tuple(part.strip() for part in value.split(","))
    elif isinstance(value, list | tuple):
        if any(not isinstance(item, str) for item in value):
            raise _ApiError(400, "InvalidParameter", f"{name} must contain strings")
        parts = tuple(item.strip() for item in value)
    else:
        raise _ApiError(400, "InvalidParameter", f"{name} must be a string or list")
    if not parts or any(not part for part in parts):
        raise _ApiError(400, "InvalidParameter", f"{name} must not be empty")
    return parts


def _body_bbox(value: Any) -> tuple[float, ...] | None:
    """Parse a JSON bbox list."""
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        raise _ApiError(400, "InvalidParameter", "bbox must be a list")
    invalid = any(
        isinstance(item, bool) or not isinstance(item, int | float) for item in value
    )
    if invalid:
        raise _ApiError(400, "InvalidParameter", "bbox must contain only numbers")
    return _parse_bbox_values(tuple(str(item) for item in value))


def _body_limit(value: Any) -> int:
    """Parse a JSON limit value."""
    if value is None:
        return _DEFAULT_LIMIT
    if isinstance(value, bool) or not isinstance(value, int):
        raise _ApiError(400, "InvalidParameter", "limit must be an integer")
    if value < 1:
        raise _ApiError(400, "InvalidParameter", "limit must be at least 1")
    return min(value, _MAX_LIMIT)


def _body_token(value: Any) -> str | None:
    """Parse a JSON cursor token."""
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise _ApiError(400, "InvalidParameter", "token must be a non-empty string")
    return value


def _body_intersects(value: Any) -> Mapping[str, Any] | None:
    """Validate that a JSON intersects value is a geometry object."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise _ApiError(400, "InvalidParameter", "intersects must be a GeoJSON object")
    return value


def _search_request_from_body(
    io: _StackIO, body: Mapping[str, Any]
) -> tuple[Any, tuple[tuple[str, str], ...]]:
    """Parse a POST Item Search body into the io contract."""
    unknown = sorted(set(body) - _SEARCH_PARAMETERS)
    if unknown:
        msg = f"unsupported search parameter(s): {', '.join(unknown)}"
        raise _ApiError(400, "InvalidParameter", msg)
    bbox = _body_bbox(body.get("bbox"))
    intersects = _body_intersects(body.get("intersects"))
    if bbox is not None and intersects is not None:
        raise _ApiError(
            400,
            "InvalidParameter",
            "bbox and intersects are mutually exclusive",
        )
    ids = _body_sequence(body.get("ids"), "ids")
    collections = _body_sequence(body.get("collections"), "collections")
    timestamp = body.get("datetime")
    if timestamp is not None and not isinstance(timestamp, str):
        raise _ApiError(400, "InvalidParameter", "datetime must be a string")
    if timestamp == "":
        raise _ApiError(400, "InvalidParameter", "datetime must not be empty")
    parsed = io.search_request(
        ids=ids,
        collections=collections,
        bbox=bbox,
        intersects=intersects,
        datetime=timestamp or None,
        limit=_body_limit(body.get("limit")),
        token=_body_token(body.get("token")),
    )
    pairs: list[tuple[str, str]] = []
    if ids is not None:
        pairs.append(("ids", ",".join(ids)))
    if collections is not None:
        pairs.append(("collections", ",".join(collections)))
    if bbox is not None:
        pairs.append(("bbox", ",".join(_number_text(value) for value in bbox)))
    if intersects is not None:
        pairs.append(("intersects", _strict_dumps(intersects)))
    if timestamp:
        pairs.append(("datetime", timestamp))
    pairs.append(("limit", str(_body_limit(body.get("limit")))))
    return parsed, tuple(pairs)


def _number_text(value: float) -> str:
    """Render one bbox number without exponent notation when avoidable."""
    return f"{value:g}"


async def _search(request: Any) -> Any:
    """Serve GET or POST Item Search with generation-bound pagination."""
    web = request.app[_WEB_KEY]
    io = request.app[_IO_KEY]
    index = request.app[_STATE_KEY].snapshot()
    base_url = request.app[_BASE_URL_KEY]
    if request.method == "POST":
        content_type = request.headers.get("Content-Type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type not in {_JSON_MEDIA_TYPE, _GEOJSON_MEDIA_TYPE}:
            raise _ApiError(
                415,
                "UnsupportedMediaType",
                "POST /search requires a JSON content type",
            )
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError) as error:
            msg = "search body must be valid JSON"
            raise _ApiError(400, "InvalidParameter", msg) from error
        if not isinstance(body, Mapping):
            raise _ApiError(400, "InvalidParameter", "search body must be an object")
        parsed, pairs = _search_request_from_body(io, body)
    else:
        parsed, pairs = _search_request_from_get(io, request)
    try:
        page = index.search(parsed)
    except io.search_error as error:
        raise _ApiError(400, "InvalidParameter", str(error)) from error
    features = []
    for record in page.items:
        public_id = quote(str(record.public_id), safe="")
        features.append(
            io.render_item(
                record,
                _http_targets(io, base_url, f"/items/{public_id}"),
            )
        )
    links: list[dict[str, Any]] = []
    if request.method == "GET":
        query = request.rel_url.raw_query_string
        self_href = f"{base_url}/search" + (f"?{query}" if query else "")
    else:
        self_href = f"{base_url}/search"
    links.append({"rel": "self", "href": self_href, "type": _GEOJSON_MEDIA_TYPE})
    if page.next_token:
        next_href = f"{base_url}/search?" + urlencode(
            [*pairs, ("token", page.next_token)]
        )
        links.append({"rel": "next", "href": next_href, "type": _GEOJSON_MEDIA_TYPE})
    document = {
        "type": "FeatureCollection",
        "stac_version": "1.0.0",
        "features": features,
        "numberMatched": page.number_matched,
        "numberReturned": len(features),
        "links": links,
    }
    return _json_response(web, document, media_type=_GEOJSON_MEDIA_TYPE)


def _cache_headers(etag: str, modified: float) -> dict[str, str]:
    """Return validator headers shared by asset responses."""
    return {
        "ETag": etag,
        "Last-Modified": _format_http_date(modified),
        "Cache-Control": "no-cache",
        "Accept-Ranges": "bytes",
    }


async def _stream_asset(
    web: Any,
    request: Any,
    descriptor: int,
    *,
    status: int,
    start: int,
    end: int,
    headers: Mapping[str, str],
) -> Any:
    """Stream one pinned descriptor in bounded asynchronous chunks."""
    response = web.StreamResponse(status=status)
    for name, value in headers.items():
        response.headers[name] = value
    await response.prepare(request)
    if request.method == "HEAD":
        await response.write_eof()
        return response
    remaining = max(0, end - start + 1)
    offset = start
    loop = asyncio.get_running_loop()
    try:
        while remaining > 0:
            chunk = await loop.run_in_executor(
                None, _pread, descriptor, min(_STREAM_CHUNK, remaining), offset
            )
            if not chunk:
                break
            await response.write(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
    except (BrokenPipeError, ConnectionResetError):
        return response
    await response.write_eof()
    return response


async def _asset_response(
    request: Any,
    descriptor: int,
    metadata: os.stat_result,
    *,
    record: Any,
    asset_key: str,
    relative: str,
) -> Any:
    """Evaluate identity-checked validators and stream one open asset."""
    web = request.app[_WEB_KEY]
    if not stat.S_ISREG(metadata.st_mode):
        raise _ApiError(404, "NotFound", "asset is not available")
    size = metadata.st_size
    normalized = "/".join(_safe_asset_parts(relative))
    etag = _asset_etag(str(record.write_id), asset_key, normalized)
    headers = _cache_headers(etag, metadata.st_mtime)
    if _conditional_status(request, etag, metadata.st_mtime) == 304:
        return web.Response(status=304, headers=headers)
    headers["Content-Type"] = _asset_media_type(record, asset_key, relative)
    range_header = request.headers.get("Range")
    if (
        range_header
        and request.headers.get("If-Range")
        and not _if_range_matches(request.headers["If-Range"], etag, metadata.st_mtime)
    ):
        range_header = None
    if range_header:
        try:
            start, end = _parse_byte_range(range_header, size)
        except _UnsatisfiableRangeError:
            unsatisfiable = _cache_headers(etag, metadata.st_mtime)
            unsatisfiable["Content-Range"] = f"bytes */{size}"
            return web.Response(status=416, headers=unsatisfiable)
        except _InvalidRangeError as error:
            msg = f"malformed Range header: {error}"
            raise _ApiError(400, "InvalidRange", msg) from error
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(end - start + 1)
        return await _stream_asset(
            web,
            request,
            descriptor,
            status=206,
            start=start,
            end=end,
            headers=headers,
        )
    headers["Content-Length"] = str(size)
    return await _stream_asset(
        web,
        request,
        descriptor,
        status=200,
        start=0,
        end=size - 1,
        headers=headers,
    )


async def _asset(request: Any) -> Any:
    """Serve one registered file or Zarr asset after identity revalidation."""
    index = request.app[_STATE_KEY].snapshot()
    work_dir = request.app[_WORK_DIR_KEY]
    public_id = request.match_info["public_id"]
    record = index.record(public_id)
    if record is None:
        raise _ApiError(404, "NotFound", "asset is not available")
    asset_key = request.match_info["asset_key"]
    if asset_key not in record.asset_paths:
        raise _ApiError(404, "NotFound", "asset is not available")
    relative = request.match_info.get("relative", "")
    try:
        parts = _product_parts(work_dir, record)
        product_fd = _open_directory(work_dir, parts)
    except (OSError, _UnsafeAssetPathError, _IdentityMismatchError):
        raise _ApiError(404, "NotFound", "asset is not available") from None
    descriptor: int | None = None
    try:
        if _read_write_id(product_fd) != str(record.write_id):
            raise _ApiError(404, "NotFound", "asset is not available")
        try:
            descriptor, metadata = _open_asset(
                work_dir, product_fd, record, asset_key, relative
            )
        except (OSError, _UnsafeAssetPathError, _IdentityMismatchError):
            raise _ApiError(404, "NotFound", "asset is not available") from None
        if not stat.S_ISREG(metadata.st_mode):
            raise _ApiError(404, "NotFound", "asset is not available")
        if _read_write_id(product_fd) != str(record.write_id):
            raise _ApiError(404, "NotFound", "asset is not available")
        return await _asset_response(
            request,
            descriptor,
            metadata,
            record=record,
            asset_key=asset_key,
            relative=relative,
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(product_fd)


async def _options(request: Any) -> Any:
    """Answer a CORS preflight for any known route."""
    web = request.app[_WEB_KEY]
    return web.Response(status=204)


def _openapi_document(base_url: str) -> dict[str, Any]:
    """Build the OpenAPI 3.0 description for the implemented HTTP surface."""
    search_parameters = [
        {
            "name": "ids",
            "in": "query",
            "required": False,
            "description": "Comma-separated public Item ids.",
            "schema": {"type": "string"},
        },
        {
            "name": "collections",
            "in": "query",
            "required": False,
            "description": "Comma-separated real Collection ids.",
            "schema": {"type": "string"},
        },
        {
            "name": "bbox",
            "in": "query",
            "required": False,
            "description": (
                "Four or six CRS84 numbers. Mutually exclusive with intersects."
            ),
            "schema": {"type": "array", "items": {"type": "number"}},
            "style": "form",
            "explode": False,
        },
        {
            "name": "intersects",
            "in": "query",
            "required": False,
            "description": (
                "A JSON-encoded GeoJSON geometry. Mutually exclusive with bbox."
            ),
            "schema": {"type": "object"},
        },
        {
            "name": "datetime",
            "in": "query",
            "required": False,
            "description": "RFC 3339 instant or open/closed interval.",
            "schema": {"type": "string"},
        },
        {
            "name": "limit",
            "in": "query",
            "required": False,
            "description": "Page size from 1 to 1000; defaults to 10.",
            "schema": {"type": "integer", "minimum": 1, "maximum": _MAX_LIMIT},
        },
        {
            "name": "token",
            "in": "query",
            "required": False,
            "description": "Opaque cursor from a previous next link.",
            "schema": {"type": "string"},
        },
    ]
    json_object = {
        "description": "STAC JSON document",
        "content": {_JSON_MEDIA_TYPE: {"schema": {"type": "object"}}},
    }
    geojson_object = {
        "description": "STAC GeoJSON document",
        "content": {_GEOJSON_MEDIA_TYPE: {"schema": {"type": "object"}}},
    }
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "FanInSAR Stack STAC API",
            "version": "1.0.0",
            "description": (
                "Read-only STAC API over committed FanInSAR Stack products. "
                "bbox and intersects are mutually exclusive."
            ),
        },
        "servers": [{"url": base_url}],
        "paths": {
            "/": {"get": {"responses": {"200": json_object}}},
            "/catalog.json": {"get": {"responses": {"200": json_object}}},
            "/conformance": {"get": {"responses": {"200": json_object}}},
            "/api": {
                "get": {
                    "responses": {
                        "200": {
                            "description": "OpenAPI 3.0 service description",
                            "content": {
                                _OPENAPI_MEDIA_TYPE: {"schema": {"type": "object"}}
                            },
                        }
                    }
                }
            },
            "/catalogs/{family}.json": {
                "get": {
                    "parameters": [
                        {
                            "name": "family",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": json_object, "404": json_object},
                }
            },
            "/collections": {"get": {"responses": {"200": json_object}}},
            "/collections/{collectionId}": {
                "get": {
                    "parameters": [
                        {
                            "name": "collectionId",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": json_object, "404": json_object},
                }
            },
            "/items/{itemId}": {
                "get": {
                    "parameters": [
                        {
                            "name": "itemId",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": geojson_object, "404": json_object},
                }
            },
            "/search": {
                "get": {
                    "parameters": search_parameters,
                    "responses": {"200": geojson_object, "400": json_object},
                },
                "post": {
                    "requestBody": {
                        "required": False,
                        "description": (
                            "Item Search parameters. bbox and intersects are "
                            "mutually exclusive; supplying both returns 400."
                        ),
                        "content": {
                            _JSON_MEDIA_TYPE: {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "ids": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "collections": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "bbox": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                        },
                                        "intersects": {"type": "object"},
                                        "datetime": {"type": "string"},
                                        "limit": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": _MAX_LIMIT,
                                        },
                                        "token": {"type": "string"},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": geojson_object,
                        "400": json_object,
                        "415": json_object,
                    },
                },
            },
            "/assets/{publicId}/{assetKey}/{relative}": {
                "get": {
                    "parameters": _asset_parameters(),
                    "responses": {
                        "200": {"description": "Asset bytes"},
                        "206": {"description": "Partial asset bytes"},
                        "404": json_object,
                        "416": {"description": "Unsatisfiable range"},
                    },
                },
                "head": {
                    "parameters": _asset_parameters(),
                    "responses": {
                        "200": {"description": "Asset metadata"},
                        "206": {"description": "Partial asset metadata"},
                        "404": json_object,
                    },
                },
            },
        },
    }


def _asset_parameters() -> list[dict[str, Any]]:
    """Return OpenAPI path and header parameters for the asset route."""
    return [
        {
            "name": "publicId",
            "in": "path",
            "required": True,
            "schema": {"type": "string"},
        },
        {
            "name": "assetKey",
            "in": "path",
            "required": True,
            "schema": {"type": "string"},
        },
        {
            "name": "relative",
            "in": "path",
            "required": False,
            "schema": {"type": "string"},
        },
        {
            "name": "Range",
            "in": "header",
            "required": False,
            "schema": {"type": "string"},
        },
        {
            "name": "If-Range",
            "in": "header",
            "required": False,
            "schema": {"type": "string"},
        },
        {
            "name": "If-None-Match",
            "in": "header",
            "required": False,
            "schema": {"type": "string"},
        },
        {
            "name": "If-Modified-Since",
            "in": "header",
            "required": False,
            "schema": {"type": "string"},
        },
    ]


def _create_app(
    web: Any,
    io: _StackIO,
    state: _IndexState,
    work_dir: Path,
    base_url: str,
) -> Any:
    """Build the aiohttp application for one index holder."""
    app = web.Application(middlewares=[web.middleware(_error_middleware)])
    app[_WEB_KEY] = web
    app[_IO_KEY] = io
    app[_STATE_KEY] = state
    app[_WORK_DIR_KEY] = work_dir
    app[_BASE_URL_KEY] = base_url
    app.on_response_prepare.append(_normalise_http_error)
    app.router.add_get("/", _landing)
    app.router.add_get("/catalog.json", _root_catalog)
    app.router.add_get("/conformance", _conformance)
    app.router.add_get("/api", _api)
    app.router.add_get("/catalogs/{family}.json", _family_catalog)
    app.router.add_get("/collections", _collections)
    app.router.add_get("/collections/{collection_id}", _collection)
    app.router.add_get("/items/{public_id}", _item)
    app.router.add_get("/search", _search, allow_head=False)
    app.router.add_post("/search", _search)
    app.router.add_get("/assets/{public_id}/{asset_key}", _asset)
    app.router.add_get("/assets/{public_id}/{asset_key}/{relative:.*}", _asset)
    app.router.add_route("OPTIONS", "/", _options)
    app.router.add_route("OPTIONS", "/search", _options)
    app.router.add_route("OPTIONS", "/{tail:.*}", _options)
    return app


class _IndexState:
    """Hold the current immutable index snapshot for atomic swaps."""

    def __init__(self, index: Any) -> None:
        self._index = index
        self._lock = threading.Lock()

    def snapshot(self) -> Any:
        """Return the current immutable index without locking readers."""
        return self._index

    def swap(self, index: Any) -> None:
        """Replace the served index atomically."""
        with self._lock:
            self._index = index


class _ServerRuntime:
    """Own the service socket, event loop, SockSite, and daemon thread."""

    def __init__(self, web: Any, app: Any, sock: socket.socket) -> None:
        self._web = web
        self._app = app
        self._sock = sock
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._stopped = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="faninsar-stac-service",
            daemon=True,
        )

    def start(self) -> None:
        """Start the daemon thread and wait until the site is listening."""
        self._thread.start()
        if not self._ready.wait(_READY_TIMEOUT):
            self.stop()
            msg = "STAC service did not become ready within the startup timeout"
            raise RuntimeError(msg)
        if self._startup_error is not None:
            error = self._startup_error
            self._thread.join(_STOP_TIMEOUT)
            raise error

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        runner = None
        try:
            runner = self._web.AppRunner(self._app, access_log=None)
            loop.run_until_complete(runner.setup())
            site = self._web.SockSite(runner, self._sock)
            loop.run_until_complete(site.start())
        except BaseException as error:
            self._startup_error = error
            self._ready.set()
            with suppress(Exception):
                if runner is not None:
                    loop.run_until_complete(runner.cleanup())
            with suppress(OSError):
                self._sock.close()
            with suppress(Exception):
                loop.close()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            with suppress(Exception):
                loop.run_until_complete(runner.cleanup())
            with suppress(Exception):
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            with suppress(Exception):
                loop.close()

    def stop(self) -> None:
        """Stop the server once and join its thread."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        self._thread.join(_STOP_TIMEOUT)
        if self._thread.is_alive():
            logger.warning("STAC service thread did not stop within its timeout")

    def wait(self, timeout: float | None = None) -> None:
        """Block until the server thread stops or the timeout elapses."""
        self._thread.join(timeout)

    @property
    def stopped(self) -> bool:
        """Return whether ``stop`` has been requested."""
        return self._stopped


class StackStacService:
    """A running read-only Stack STAC service handle.

    Parameters
    ----------
    work_dir : Path
        Resolved Stack workspace root.
    host : str
        Host the listening socket is bound to.
    port : int
        Actual assigned port, including an ephemeral ``port=0`` assignment.
    base_url : str
        Reachable loopback base URL without a trailing slash.
    state : _IndexState
        Current immutable index holder.
    runtime : _ServerRuntime
        Socket, event loop, and daemon thread owner.
    io : _StackIO
        Normalization contract used by startup and refresh.
    report : StacRefreshReport
        Latest startup or refresh outcome.
    persist_catalog : bool
        Whether refresh writes derived portable catalogs.

    """

    def __init__(
        self,
        *,
        work_dir: Path,
        host: str,
        port: int,
        base_url: str,
        state: _IndexState,
        runtime: _ServerRuntime,
        io: _StackIO,
        report: StacRefreshReport,
        persist_catalog: bool,
    ) -> None:
        self._work_dir = work_dir
        self._host = host
        self._port = port
        self._base_url = base_url
        self._state = state
        self._runtime = runtime
        self._io = io
        self._report = report
        self._persist_catalog = persist_catalog
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        """Return the reachable base URL without a trailing slash."""
        return self._base_url

    @property
    def catalog_url(self) -> str:
        """Return the reachable portable Catalog URL."""
        return f"{self._base_url}/catalog.json"

    @property
    def api_url(self) -> str:
        """Return the reachable OpenAPI service description URL."""
        return f"{self._base_url}/api"

    @property
    def host(self) -> str:
        """Return the bound host."""
        return self._host

    @property
    def port(self) -> int:
        """Return the actual assigned port."""
        return self._port

    def refresh(self) -> StacRefreshReport:
        """Rebuild the index, persist when enabled, and swap atomically.

        Returns
        -------
        StacRefreshReport
            The new generation and its persistence outcome.

        Raises
        ------
        StacRefreshError
            If the rebuild fails; the previous generation is still served.

        """
        if self._runtime.stopped:
            raise StacRefreshError(
                self._report.generation,
                RuntimeError("STAC service is stopped"),
            )
        previous = self._state.snapshot()
        try:
            index = self._io.build_index(self._work_dir)
        except Exception as error:
            raise StacRefreshError(previous.generation, error) from error
        persistence = self._persist(index)
        self._state.swap(index)
        report = _build_report(index, persistence)
        with self._lock:
            self._report = report
        return report

    def _persist(self, index: Any) -> Any:
        """Persist one generated index, tolerating a reported write failure."""
        try:
            return _persist(self._io, index, self._work_dir, self._persist_catalog)
        except Exception as error:
            logger.warning("portable STAC catalog refresh was not persisted: %s", error)
            return self._io.catalog_report(
                requested=self._persist_catalog,
                persisted=False,
                generation=index.generation,
                catalog_path=None,
                error=str(error),
            )

    def report(self) -> dict[str, Any]:
        """Return the latest refresh outcome as a strict-JSON mapping."""
        with self._lock:
            return self._report.as_dict()

    def wait(self, timeout: float | None = None) -> None:
        """Block until the server stops or ``timeout`` elapses."""
        self._runtime.wait(timeout)

    def stop(self) -> None:
        """Stop the service and join the server thread; safe to call twice."""
        self._runtime.stop()

    def __enter__(self) -> Self:
        """Return this running service handle."""
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        """Stop the service when leaving a context manager."""
        self.stop()


def _build_report(index: Any, persistence: Any) -> StacRefreshReport:
    """Build one refresh report from an index and persistence outcome."""
    base = index.report()
    families = base.get("families") if isinstance(base, Mapping) else None
    if not isinstance(families, Mapping):
        families = {
            str(family): len(index.records_for_family(family))
            for family in index.families()
        }
    return StacRefreshReport(
        generation=str(index.generation),
        item_count=len(index.records),
        collection_count=len(index.collections),
        families={str(key): int(value) for key, value in sorted(families.items())},
        exclusions=tuple(index.exclusions),
        persistence=persistence,
    )


def _persist(
    io: _StackIO,
    index: Any,
    work_dir: Path,
    enabled: bool,
) -> Any:
    """Write derived portable catalogs when requested."""
    if not enabled:
        return io.catalog_report(
            requested=False,
            persisted=False,
            generation=index.generation,
            catalog_path=None,
            error=None,
        )
    report = io.write_portable_catalog(index, work_dir=work_dir)
    if not getattr(report, "persisted", False):
        logger.warning(
            "portable STAC catalog was not persisted: %s",
            getattr(report, "error", None),
        )
    return report


def start_stac_service(
    work_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    persist_catalog: bool = True,
) -> StackStacService:
    """Start the read-only Stack STAC service and return its handle.

    Startup order is import resolution, read-only index build, socket bind,
    application readiness, optional portable-catalog persistence, and only
    then URL printing, so a failed startup prints nothing and mutates no
    workspace file.

    Parameters
    ----------
    work_dir : str or path-like
        Stack workspace root.
    host : str, optional
        Bind host. Loopback is the default. A non-loopback host is honored
        only when passed explicitly and logs a warning.
    port : int, optional
        Bind port; ``0`` selects an ephemeral port.
    persist_catalog : bool, optional
        Write derived portable catalogs at startup and refresh.

    Returns
    -------
    StackStacService
        Running service handle with actual Catalog and API URLs.

    Raises
    ------
    ImportError
        If ``aiohttp`` or the ``faninsar.io.stac`` package is unavailable.
    OSError
        If the requested host and port cannot be bound.
    RuntimeError
        If the application does not become ready.

    """
    if not isinstance(persist_catalog, bool):
        msg = "persist_catalog must be a bool"
        raise TypeError(msg)
    work_root = Path(work_dir).expanduser().resolve()
    if not _is_loopback(host):
        logger.warning(
            "binding the STAC service to non-loopback host %s makes it "
            "reachable beyond this machine",
            host,
        )
    web = _load_web()
    io = _load_io()
    index = io.build_index(work_root)
    sock = _bind_socket(host, port)
    try:
        assigned = int(sock.getsockname()[1])
        base_url = _base_url(host, assigned)
        state = _IndexState(index)
        app = _create_app(web, io, state, work_root, base_url)
        runtime = _ServerRuntime(web, app, sock)
        try:
            runtime.start()
            persistence = _persist(io, index, work_root, persist_catalog)
            report = _build_report(index, persistence)
        except BaseException:
            runtime.stop()
            raise
    except BaseException:
        with suppress(OSError):
            sock.close()
        raise
    service = StackStacService(
        work_dir=work_root,
        host=host,
        port=assigned,
        base_url=base_url,
        state=state,
        runtime=runtime,
        io=io,
        report=report,
        persist_catalog=persist_catalog,
    )
    try:
        print(f"FanInSAR STAC Catalog: {base_url}/catalog.json", flush=True)  # noqa: T201
        print(f"FanInSAR STAC API: {base_url}/api", flush=True)  # noqa: T201
    except BaseException:
        runtime.stop()
        raise
    return service
