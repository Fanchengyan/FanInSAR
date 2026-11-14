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
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from email.utils import formatdate, parsedate_to_datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Self
from urllib.parse import quote, unquote, urlencode

from faninsar.logging import setup_logger
from faninsar.stack import _stac_georef as georef
from faninsar.stack import _stac_radar as radar

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
_MAX_ZARR_METADATA_BYTES = 4 * 1024 * 1024
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
_RESERVATION_TIMEOUT = _READY_TIMEOUT + _STOP_TIMEOUT + 5.0
_LIFECYCLE_STARTING = "starting"
_LIFECYCLE_RUNNING = "running"
_LIFECYCLE_STOPPING = "stopping"
_LIFECYCLE_FAILED = "failed"
_ZARR_METADATA_NAMES = ("zarr.json", ".zarray", ".zgroup")
_JSON_METADATA_SUFFIXES = frozenset({".zarray", ".zgroup", ".zattrs"})

_WEB_KEY = "faninsar:stac:web"
_IO_KEY = "faninsar:stac:io"
_STATE_KEY = "faninsar:stac:state"
_WORK_DIR_KEY = "faninsar:stac:work_dir"
_BASE_URL_KEY = "faninsar:stac:base_url"
_GEOREF_KEY = "faninsar:stac:georef"
_RADAR_KEY = "faninsar:stac:radar"

# Process-local strong references to endpoint reservations and healthy
# services, keyed by the literal requested host and actual bound port. One
# entry per endpoint lets a repeated ``stac_serve`` return the same handle
# without touching the index, while concurrent callers wait on the entry event.
_SERVICE_REGISTRY_LOCK = threading.RLock()
_SERVICE_REGISTRY: dict[tuple[str, int], _ServiceRegistration] = {}


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


@dataclass(slots=True)
class _ServiceRegistration:
    """One endpoint reservation, its canonical workspace, and its handle."""

    key: tuple[str, int]
    work_dir: Path
    persist_catalog: bool
    lifecycle: str = _LIFECYCLE_STARTING
    service: StackStacService | None = None
    error: BaseException | None = None
    changed: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True, slots=True)
class _StartDecision:
    """Outcome of reserving an endpoint or reusing its running handle."""

    service: StackStacService | None = None
    registration: _ServiceRegistration | None = None
    dead_service: StackStacService | None = None


def _normalized_host(host: str) -> str:
    """Return the literal bind host used as a registry key component."""
    return host.strip().lower()


def _service_key(host: str, port: int) -> tuple[str, int]:
    """Return the registry key for one literal host and actual port."""
    return (_normalized_host(host), int(port))


def _live_service(registration: _ServiceRegistration) -> StackStacService | None:
    """Return the healthy handle for one registration, or ``None``."""
    service = registration.service
    if registration.lifecycle != _LIFECYCLE_RUNNING or service is None:
        return None
    if service._stopping.is_set():
        return None
    if not service._runtime.is_alive():
        return None
    return service


def _busy_endpoint_error(
    work_root: Path, host: str, port: int, registration: _ServiceRegistration
) -> OSError:
    """Build the error for an endpoint owned by a different workspace."""
    msg = (
        f"cannot serve workspace {work_root} on "
        f"{_base_url(host, port)}: that endpoint is already served for "
        f"another workspace ({registration.work_dir}); stop that service "
        "before starting a different one"
    )
    return OSError(msg)


def _log_reuse(
    registration: _ServiceRegistration,
    service: StackStacService,
    work_root: Path,
    persist_catalog: bool,
) -> None:
    """Report a reused handle and any persistence-mode mismatch."""
    logger.info(
        "reusing the running STAC service for workspace %s at %s",
        work_root,
        service.base_url,
    )
    if registration.persist_catalog != persist_catalog:
        logger.info(
            "the reused STAC service keeps its original persist_catalog=%s; "
            "stop and restart it to change catalog persistence",
            registration.persist_catalog,
        )


def _reserve_start(
    work_root: Path,
    host: str,
    port: int,
    persist_catalog: bool,
) -> _StartDecision:
    """Reserve this endpoint, wait for a peer, or return a live handle.

    Only the short registry lookup and reservation happen under
    :data:`_SERVICE_REGISTRY_LOCK`; a caller that must wait for a peer does so
    on the registration event outside the lock. The starter performs the index
    build, bind, readiness wait, persistence, and URL printing without holding
    the global lock.
    """
    key = _service_key(host, port)
    deadline = time.monotonic() + _RESERVATION_TIMEOUT
    while True:
        wait_event: threading.Event | None = None
        with _SERVICE_REGISTRY_LOCK:
            registration = _SERVICE_REGISTRY.get(key)
            if registration is None:
                registration = _ServiceRegistration(
                    key=key,
                    work_dir=work_root,
                    persist_catalog=persist_catalog,
                )
                _SERVICE_REGISTRY[key] = registration
                return _StartDecision(registration=registration)
            service = _live_service(registration)
            if service is not None:
                if registration.work_dir != work_root:
                    raise _busy_endpoint_error(work_root, host, port, registration)
                _log_reuse(registration, service, work_root, persist_catalog)
                return _StartDecision(service=service)
            if registration.lifecycle in {
                _LIFECYCLE_STARTING,
                _LIFECYCLE_STOPPING,
            }:
                if registration.work_dir != work_root:
                    raise _busy_endpoint_error(work_root, host, port, registration)
                wait_event = registration.changed
            else:
                dead_service = (
                    registration.service
                    if registration.lifecycle == _LIFECYCLE_RUNNING
                    else None
                )
                registration.work_dir = work_root
                registration.persist_catalog = persist_catalog
                registration.lifecycle = _LIFECYCLE_STARTING
                registration.service = None
                registration.error = None
                registration.changed.clear()
                return _StartDecision(
                    registration=registration,
                    dead_service=dead_service,
                )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            msg = (
                f"timed out waiting for the STAC endpoint {_base_url(host, port)} "
                "to finish starting or stopping"
            )
            raise RuntimeError(msg)
        wait_event.wait(remaining)


def _abandon_start(registration: _ServiceRegistration, error: BaseException) -> None:
    """Release a failed reservation and wake peers waiting on it."""
    with _SERVICE_REGISTRY_LOCK:
        if _SERVICE_REGISTRY.get(registration.key) is registration:
            del _SERVICE_REGISTRY[registration.key]
        registration.lifecycle = _LIFECYCLE_FAILED
        registration.service = None
        registration.error = error
        registration.changed.set()


def _publish_running(
    registration: _ServiceRegistration, service: StackStacService
) -> bool:
    """Publish a ready service, or report that a live endpoint already owns it."""
    with _SERVICE_REGISTRY_LOCK:
        existing = _SERVICE_REGISTRY.get(registration.key)
        if (
            existing is not None
            and existing is not registration
            and _live_service(existing) is not None
        ):
            return False
        _SERVICE_REGISTRY[registration.key] = registration
        registration.lifecycle = _LIFECYCLE_RUNNING
        registration.service = service
        registration.error = None
        registration.changed.set()
        return True


def _begin_stop(service: StackStacService) -> _ServiceRegistration | None:
    """Mark this service's endpoint stopping under the registry lock."""
    with _SERVICE_REGISTRY_LOCK:
        for registration in _SERVICE_REGISTRY.values():
            if registration.service is service:
                if registration.lifecycle != _LIFECYCLE_STOPPING:
                    registration.lifecycle = _LIFECYCLE_STOPPING
                    registration.changed.clear()
                return registration
    return None


def _finish_stop(registration: _ServiceRegistration, terminated: bool) -> None:
    """Remove a terminated registration; keep ownership on timeout."""
    with _SERVICE_REGISTRY_LOCK:
        if _SERVICE_REGISTRY.get(registration.key) is not registration:
            return
        if not terminated:
            return
        del _SERVICE_REGISTRY[registration.key]
        registration.changed.set()


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


def _read_bounded(descriptor: int, limit: int) -> bytes | None:
    """Read at most ``limit`` bytes, returning ``None`` when the file is larger."""
    chunks: list[bytes] = []
    remaining = limit + 1
    while remaining > 0:
        chunk = os.read(descriptor, min(65536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    return data if len(data) <= limit else None


def _read_native_zarr_metadata(
    work_dir: Path, record: Any, asset_key: str
) -> tuple[dict[str, Any] | None, float]:
    """Read ``zarr.json`` through the pinned safe asset open, with receipt checks.

    The native metadata is the only source for descriptor construction. The
    asset root and the metadata file are opened without following symbolic
    links, the read is bounded, and the product write id is checked before and
    after the read.
    """
    try:
        product_fd = _open_directory(work_dir, _product_parts(work_dir, record))
    except (OSError, _UnsafeAssetPathError, _IdentityMismatchError):
        return None, 0.0
    try:
        return _read_metadata_from_product(product_fd, work_dir, record, asset_key)
    finally:
        os.close(product_fd)


def _read_metadata_from_product(
    product_fd: int, work_dir: Path, record: Any, asset_key: str
) -> tuple[dict[str, Any] | None, float]:
    """Read one bounded Zarr group document below an already pinned product."""
    if _read_write_id(product_fd) != str(record.write_id):
        return None, 0.0
    try:
        descriptor, metadata = _open_asset(
            work_dir, product_fd, record, asset_key, "zarr.json"
        )
    except (OSError, _UnsafeAssetPathError, _IdentityMismatchError):
        return None, 0.0
    try:
        parsed = _read_metadata_descriptor(descriptor, metadata)
    finally:
        os.close(descriptor)
    if parsed is None:
        return None, 0.0
    payload, modified = parsed
    if _read_write_id(product_fd) != str(record.write_id):
        return None, 0.0
    return payload, modified


def _read_metadata_descriptor(
    descriptor: int, metadata: os.stat_result
) -> tuple[dict[str, Any], float] | None:
    """Parse one bounded regular Zarr v3 group document from a descriptor."""
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size > _MAX_ZARR_METADATA_BYTES
    ):
        return None
    data = _read_bounded(descriptor, _MAX_ZARR_METADATA_BYTES)
    if data is None:
        return None
    try:
        payload = _strict_loads(data)
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("zarr_format") != 3:
        return None
    return payload, metadata.st_mtime


def _receipt_matches(work_dir: Path, record: Any) -> bool:
    """Return whether the product receipt still names the record write id."""
    try:
        product_fd = _open_directory(work_dir, _product_parts(work_dir, record))
    except (OSError, _UnsafeAssetPathError, _IdentityMismatchError):
        return False
    try:
        return _read_write_id(product_fd) == str(record.write_id)
    finally:
        os.close(product_fd)


def _receipt_fingerprint(work_dir: Path, record: Any) -> str:
    """Return a bounded digest of the current canonical product receipt.

    The scene write id is preserved when geometry assets are added, so the
    radar adapter cannot key caches on the write id or the index generation
    alone. Reading the bounded ``item.json`` bytes makes an added or replaced
    asset declaration change the preview identity immediately.
    """
    try:
        product_fd = _open_directory(work_dir, _product_parts(work_dir, record))
    except (OSError, _UnsafeAssetPathError, _IdentityMismatchError):
        return ""
    descriptor: int | None = None
    try:
        if _read_write_id(product_fd) != str(record.write_id):
            return ""
        descriptor, metadata = _open_at(product_fd, "item.json", directory=False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_ITEM_BYTES:
            return ""
        data = _read_bounded(descriptor, _MAX_ITEM_BYTES)
        if data is None or _read_write_id(product_fd) != str(record.write_id):
            return ""
        return hashlib.sha256(data).hexdigest()
    except (OSError, _UnsafeAssetPathError, _IdentityMismatchError):
        return ""
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(product_fd)


def _build_georeferenced_zarr_view(
    record: Any, asset_key: str, metadata: Mapping[str, Any]
) -> Any | None:
    """Import and call the metadata worker's descriptor builder lazily."""
    try:
        module = import_module("faninsar.io.stac.georef")
    except ImportError:
        return None
    builder = getattr(module, "build_georeferenced_zarr_view", None)
    if builder is None:
        return None
    return builder(record, asset_key, metadata)


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


def _asset_root_parts(work_dir: Path, record: Any, asset_key: str) -> tuple[str, ...]:
    """Return the registered asset root components below the product directory.

    The components are lexical and are later opened without following any
    symbolic link; an absolute root, ``..``, or ``item.json`` is rejected.
    """
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
    return root_parts


def _open_asset(
    work_dir: Path, product_fd: int, record: Any, asset_key: str, relative: str
) -> tuple[int, os.stat_result]:
    """Open one registered asset beneath a pinned product descriptor."""
    root_parts = _asset_root_parts(work_dir, record, asset_key)
    tail = _safe_asset_parts(relative)
    descriptor, metadata = _open_parts(product_fd, root_parts, final_directory=None)
    if stat.S_ISDIR(metadata.st_mode):
        if not tail:
            os.close(descriptor)
            raise _UnsafeAssetPathError(str(record.asset_paths[asset_key]))
        try:
            return _open_parts(descriptor, tail, final_directory=False)
        finally:
            os.close(descriptor)
    if stat.S_ISREG(metadata.st_mode):
        if tail:
            os.close(descriptor)
            raise _UnsafeAssetPathError(str(record.asset_paths[asset_key]))
        return descriptor, metadata
    os.close(descriptor)
    raise _UnsafeAssetPathError(str(record.asset_paths[asset_key]))


def _open_zarr_metadata(
    work_dir: Path, product_fd: int, record: Any, asset_key: str
) -> tuple[str, int] | None:
    """Open store-defining Zarr metadata for one directory asset root.

    Only ``zarr.json``, ``.zarray``, and ``.zgroup`` define a Zarr store root;
    ``.zattrs`` alone is child metadata and does not make a directory a store.
    Returns the basename and its already-open descriptor, or ``None`` when the
    asset root is a regular file or a directory without store metadata.
    Directories are opened without following any symbolic link, and the caller
    closes the descriptor.
    """
    root_parts = _asset_root_parts(work_dir, record, asset_key)
    root_fd, root_metadata = _open_parts(product_fd, root_parts, final_directory=None)
    try:
        if not stat.S_ISDIR(root_metadata.st_mode):
            return None
        for name in _ZARR_METADATA_NAMES:
            try:
                descriptor, metadata = _open_at(root_fd, name, directory=False)
            except (OSError, _UnsafeAssetPathError, _IdentityMismatchError):
                continue
            if stat.S_ISREG(metadata.st_mode):
                return name, descriptor
            os.close(descriptor)
        return None
    finally:
        os.close(root_fd)


def _zarr_redirect_location(request: Any, asset_key: str, metadata_name: str) -> str:
    """Build a safe relative Location that resolves to the metadata child URL."""
    quoted_key = quote(asset_key, safe="")
    quoted_name = quote(metadata_name, safe="")
    if request.path.endswith("/"):
        location = quoted_name
    else:
        location = f"{quoted_key}/{quoted_name}"
    if "\r" in location or "\n" in location:
        raise _ApiError(404, "NotFound", "asset is not available")
    return location


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
    if suffix == ".json" or Path(name).name in _JSON_METADATA_SUFFIXES:
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
        503: "ServiceUnavailable",
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


def _render_http_item(request: Any, record: Any, view_path: str) -> dict[str, Any]:
    """Render one Item and add supported derived Zarr view assets."""
    io = request.app[_IO_KEY]
    base_url = request.app[_BASE_URL_KEY]
    item = io.render_item(record, _http_targets(io, base_url, view_path))
    index = request.app[_STATE_KEY].snapshot()
    request.app[_GEOREF_KEY].augment_item(record, item, base_url, index.generation)
    request.app[_RADAR_KEY].augment_item(record, item, index)
    return item


async def _item(request: Any) -> Any:
    """Serve one public Item."""
    web = request.app[_WEB_KEY]
    index = request.app[_STATE_KEY].snapshot()
    public_id = request.match_info["public_id"]
    record = index.record(public_id)
    if record is None:
        raise _ApiError(404, "NotFound", f"unknown item: {public_id}")
    document = _render_http_item(request, record, f"/items/{quote(public_id, safe='')}")
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
        features.append(_render_http_item(request, record, f"/items/{public_id}"))
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


def _view_bytes_response(
    web: Any,
    request: Any,
    payload: bytes,
    *,
    etag: str,
    modified: float,
    media_type: str,
) -> Any:
    """Serve validated generated bytes with conditional and Range support."""
    size = len(payload)
    headers = _cache_headers(etag, modified)
    if _conditional_status(request, etag, modified) == 304:
        return web.Response(status=304, headers=headers)
    headers["Content-Type"] = media_type
    range_header = request.headers.get("Range")
    if (
        range_header
        and request.headers.get("If-Range")
        and not _if_range_matches(request.headers["If-Range"], etag, modified)
    ):
        range_header = None
    if range_header:
        try:
            start, end = _parse_byte_range(range_header, size)
        except _UnsatisfiableRangeError:
            unsatisfiable = _cache_headers(etag, modified)
            unsatisfiable["Content-Range"] = f"bytes */{size}"
            return web.Response(status=416, headers=unsatisfiable)
        except _InvalidRangeError as error:
            msg = f"malformed Range header: {error}"
            raise _ApiError(400, "InvalidRange", msg) from error
        body = payload[start : end + 1]
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(len(body))
        return web.Response(status=206, headers=headers, body=body)
    headers["Content-Length"] = str(size)
    if request.method == "HEAD":
        return web.Response(status=200, headers=headers)
    return web.Response(status=200, headers=headers, body=payload)


async def _serve_mapped_asset(
    request: Any, record: Any, asset_key: str, relative: str
) -> Any:
    """Serve one canonical source chunk through the pinned asset machinery."""
    work_dir = request.app[_WORK_DIR_KEY]
    try:
        product_fd = _open_directory(work_dir, _product_parts(work_dir, record))
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


async def _zarr_view(request: Any) -> Any:
    """Serve one HTTP-only georeferenced Zarr compatibility view."""
    web = request.app[_WEB_KEY]
    index = request.app[_STATE_KEY].snapshot()
    public_id = request.match_info["public_id"]
    record = index.record(public_id)
    if record is None:
        raise _ApiError(404, "NotFound", "asset is not available")
    parsed = georef.parse_view_tail(
        record.asset_paths, request.match_info.get("tail", "")
    )
    if parsed is None:
        raise _ApiError(404, "NotFound", "asset is not available")
    source_key, relative, slashless = parsed
    if slashless:
        response = web.Response(status=307)
        response.headers["Location"] = georef.view_path(public_id, source_key)
        response.headers["Cache-Control"] = "no-cache"
        return response
    try:
        parts = _safe_asset_parts(relative) if relative else ()
    except _UnsafeAssetPathError:
        raise _ApiError(404, "NotFound", "asset is not available") from None
    normalized = "/".join(parts)
    resolved = request.app[_GEOREF_KEY].resolve(record, source_key, index.generation)
    if resolved is None:
        raise _ApiError(404, "NotFound", "asset is not available")
    view, modified = resolved
    etag = _asset_etag(str(record.write_id), f"{source_key}_georef", normalized or "/")
    work_dir = request.app[_WORK_DIR_KEY]
    if not normalized:
        payload = georef.manifest_html(public_id, source_key)
        media_type = "text/html; charset=utf-8"
    elif normalized in georef.METADATA_PATHS:
        document = view.metadata(normalized)
        if not isinstance(document, Mapping):
            raise _ApiError(404, "NotFound", "asset is not available")
        payload = _strict_dumps(document).encode("utf-8")
        media_type = _JSON_MEDIA_TYPE
    elif normalized.startswith(("x/", "y/")):
        payload = view.coordinate_chunk(normalized)
        if not isinstance(payload, bytes):
            raise _ApiError(404, "NotFound", "asset is not available")
        media_type = "application/octet-stream"
    elif normalized.startswith("values/"):
        source_relative = view.source_chunk(normalized)
        if not isinstance(source_relative, str):
            raise _ApiError(404, "NotFound", "asset is not available")
        return await _serve_mapped_asset(request, record, source_key, source_relative)
    else:
        raise _ApiError(404, "NotFound", "asset is not available")
    if not _receipt_matches(work_dir, record):
        raise _ApiError(404, "NotFound", "asset is not available")
    return _view_bytes_response(
        web,
        request,
        payload,
        etag=etag,
        modified=modified,
        media_type=media_type,
    )


def _radar_head_response(
    web: Any,
    request: Any,
    size: int,
    *,
    etag: str,
    modified: float,
) -> Any:
    """Answer HEAD for a generated chunk without materialising its bytes."""
    headers = _cache_headers(etag, modified)
    range_header = request.headers.get("Range")
    if (
        range_header
        and request.headers.get("If-Range")
        and not _if_range_matches(request.headers["If-Range"], etag, modified)
    ):
        range_header = None
    if range_header:
        try:
            start, end = _parse_byte_range(range_header, size)
        except _UnsatisfiableRangeError:
            unsatisfiable = _cache_headers(etag, modified)
            unsatisfiable["Content-Range"] = f"bytes */{size}"
            return web.Response(status=416, headers=unsatisfiable)
        except _InvalidRangeError as error:
            msg = f"malformed Range header: {error}"
            raise _ApiError(400, "InvalidRange", msg) from error
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(end - start + 1)
        return web.Response(status=206, headers=headers)
    headers["Content-Length"] = str(size)
    return web.Response(status=200, headers=headers)


async def _radar_view(request: Any) -> Any:
    """Serve one lazily resampled radar geolocation preview."""
    web = request.app[_WEB_KEY]
    index = request.app[_STATE_KEY].snapshot()
    public_id = request.match_info["public_id"]
    record = index.record(public_id)
    if record is None:
        raise _ApiError(404, "NotFound", "asset is not available")
    parsed = radar.parse_view_tail(
        record.asset_paths, request.match_info.get("tail", "")
    )
    if parsed is None:
        raise _ApiError(404, "NotFound", "asset is not available")
    source_key, relative, slashless = parsed
    if slashless:
        response = web.Response(status=307)
        response.headers["Location"] = radar.view_path(public_id, source_key)
        response.headers["Cache-Control"] = "no-cache"
        return response
    try:
        parts = _safe_asset_parts(relative) if relative else ()
    except _UnsafeAssetPathError:
        raise _ApiError(404, "NotFound", "asset is not available") from None
    normalized = "/".join(parts)
    provider = request.app[_RADAR_KEY]
    try:
        resolution = await asyncio.to_thread(
            provider.resolve, index, record, source_key
        )
    except radar.RadarUnavailableError as error:
        description = (
            "asset is not available"
            if error.status == 404
            else f"radar preview unavailable: {error.reason}"
        )
        raise _ApiError(error.status, _status_code(error.status), description) from None
    if resolution is None:
        raise _ApiError(404, "NotFound", "asset is not available")
    work_dir = request.app[_WORK_DIR_KEY]
    etag = _asset_etag(resolution.fingerprint, source_key, normalized or "/")
    if not normalized:
        payload: bytes | None = radar.manifest_html(public_id, source_key)
        media_type = "text/html; charset=utf-8"
    elif normalized in radar.METADATA_PATHS:
        document = resolution.metadata(normalized)
        if not isinstance(document, Mapping):
            raise _ApiError(404, "NotFound", "asset is not available")
        payload = _strict_dumps(document).encode("utf-8")
        media_type = _JSON_MEDIA_TYPE
    elif normalized.startswith(("x/", "y/")):
        payload = resolution.coordinate_chunk(normalized)
        if not isinstance(payload, bytes):
            raise _ApiError(404, "NotFound", "asset is not available")
        media_type = "application/octet-stream"
    elif normalized.startswith("values/"):
        coordinates = resolution.chunk_index(normalized)
        if coordinates is None:
            raise _ApiError(404, "NotFound", "asset is not available")
        if not _receipt_matches(work_dir, record):
            raise _ApiError(404, "NotFound", "asset is not available")
        size = resolution.chunk_length()
        if _conditional_status(request, etag, resolution.modified) == 304:
            headers = _cache_headers(etag, resolution.modified)
            return web.Response(status=304, headers=headers)
        if request.method == "HEAD":
            return _radar_head_response(
                web,
                request,
                size,
                etag=etag,
                modified=resolution.modified,
            )
        try:
            payload = await asyncio.to_thread(
                provider.read_chunk, resolution, *coordinates
            )
        except radar.RadarUnavailableError as error:
            raise _ApiError(
                error.status,
                _status_code(error.status),
                f"radar preview unavailable: {error.reason}",
            ) from None
        if payload is None or not _receipt_matches(work_dir, record):
            raise _ApiError(404, "NotFound", "asset is not available")
        return _view_bytes_response(
            web,
            request,
            payload,
            etag=etag,
            modified=resolution.modified,
            media_type="application/octet-stream",
        )
    else:
        raise _ApiError(404, "NotFound", "asset is not available")
    if not _receipt_matches(work_dir, record):
        raise _ApiError(404, "NotFound", "asset is not available")
    return _view_bytes_response(
        web,
        request,
        payload,
        etag=etag,
        modified=resolution.modified,
        media_type=media_type,
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
        if not relative:
            try:
                metadata = _open_zarr_metadata(work_dir, product_fd, record, asset_key)
            except (OSError, _UnsafeAssetPathError, _IdentityMismatchError):
                raise _ApiError(404, "NotFound", "asset is not available") from None
            if metadata is not None:
                metadata_name, metadata_fd = metadata
                try:
                    if _read_write_id(product_fd) != str(record.write_id):
                        raise _ApiError(404, "NotFound", "asset is not available")
                finally:
                    os.close(metadata_fd)
                location = _zarr_redirect_location(request, asset_key, metadata_name)
                web = request.app[_WEB_KEY]
                response = web.Response(status=307)
                response.headers["Location"] = location
                response.headers["Cache-Control"] = "no-cache"
                return response
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


async def _close_radar_provider(app: Any) -> None:
    """Close cached radar handles without blocking the event loop.

    A close can wait for an in-flight chunk read that is itself using GDAL over
    this service's HTTP raw-asset route. Running it on a worker thread keeps
    the event loop able to answer that read while shutdown proceeds.
    """
    provider = app.get(_RADAR_KEY)
    if provider is not None:
        await asyncio.to_thread(provider.close)


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
    app[_GEOREF_KEY] = georef.GeorefProvider(
        open_metadata=lambda record, asset_key: _read_native_zarr_metadata(
            work_dir, record, asset_key
        ),
        receipt_valid=lambda record: _receipt_matches(work_dir, record),
        build_view=_build_georeferenced_zarr_view,
    )
    app[_RADAR_KEY] = radar.RadarProvider(
        base_url=base_url,
        open_metadata=lambda record, asset_key: _read_native_zarr_metadata(
            work_dir, record, asset_key
        ),
        receipt_valid=lambda record: _receipt_matches(work_dir, record),
        receipt_fingerprint=lambda record: _receipt_fingerprint(work_dir, record),
    )
    app.on_cleanup.append(_close_radar_provider)
    app.on_response_prepare.append(_normalise_http_error)
    app.router.add_get("/", _landing)
    app.router.add_get("/catalog.json", _root_catalog)
    app.router.add_get("/conformance", _conformance)
    app.router.add_get("/api", _api)
    app.router.add_get("/catalogs/{family}.json", _family_catalog)
    app.router.add_get("/collections", _collections)
    app.router.add_get("/collections/{collection_id}", _collection)
    app.router.add_get("/items/{public_id}", _item)
    app.router.add_get("/zarr/{public_id}/{tail:.*}", _zarr_view)
    app.router.add_get("/radar/{public_id}/{tail:.*}", _radar_view)
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
        with suppress(OSError):
            self._sock.close()
        if self._thread.is_alive():
            logger.warning("STAC service thread did not stop within its timeout")

    def wait(self, timeout: float | None = None) -> None:
        """Block until the server thread stops or the timeout elapses."""
        self._thread.join(timeout)

    @property
    def stopped(self) -> bool:
        """Return whether ``stop`` has been requested."""
        return self._stopped

    def is_terminated(self) -> bool:
        """Return whether the server thread has exited."""
        return not self._thread.is_alive()

    def is_alive(self) -> bool:
        """Return whether this runtime can still accept requests.

        The predicate is local state only: it never probes the port over HTTP.
        A stop request, failed startup, dead thread, stopped loop, or closed
        listener makes the runtime ineligible for reuse.
        """
        with self._lock:
            if self._stopped:
                return False
        if self._startup_error is not None or not self._ready.is_set():
            return False
        if not self._thread.is_alive():
            return False
        loop = self._loop
        if loop is None or not loop.is_running():
            return False
        return self._sock.fileno() != -1


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
        self._stopping = threading.Event()

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
        """Return the latest refresh outcome and effective service identity."""
        with self._lock:
            document = self._report.as_dict()
        document["service"] = {
            "work_dir": str(self._work_dir),
            "host": self._host,
            "port": self._port,
            "persist_catalog": self._persist_catalog,
            "lifecycle": self._lifecycle_state(),
        }
        _strict_dumps(document)
        return document

    def _lifecycle_state(self) -> str:
        """Return the process-local lifecycle label for reporting."""
        if self._stopping.is_set():
            return "stopped" if self._runtime.is_terminated() else "stopping"
        return "running" if self._runtime.is_alive() else "stopped"

    def wait(self, timeout: float | None = None) -> None:
        """Block until the server stops or ``timeout`` elapses."""
        self._runtime.wait(timeout)

    def stop(self) -> None:
        """Stop the service and join the server thread; safe to call twice.

        The endpoint is marked stopping under the registry lock, the runtime is
        stopped and joined without that lock, and only this registration is
        removed after the thread terminates. A concurrent start waits on the
        registration event and never receives a stopping handle.
        """
        self._stopping.set()
        registration = _begin_stop(self)
        self._runtime.stop()
        terminated = self._runtime.is_terminated()
        if registration is not None:
            _finish_stop(registration, terminated)
        if not terminated:
            msg = (
                "the STAC service did not stop within its timeout; its "
                "endpoint remains reserved as stopping"
            )
            raise RuntimeError(msg)

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

    A second call for the same canonical workspace, literal host, and live
    nonzero port returns the existing process-local handle without rebuilding
    the index, refreshing, persisting, or printing the URLs again. The
    original ``persist_catalog`` mode stays in force and is reported when the
    new call asks for a different mode.

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
        If the requested host and port cannot be bound, or the live endpoint
        is already served for a different workspace.
    RuntimeError
        If the application does not become ready.

    """
    if not isinstance(persist_catalog, bool):
        msg = "persist_catalog must be a bool"
        raise TypeError(msg)
    _validate_port(port)
    if not isinstance(host, str) or not host.strip():
        msg = "host must be a non-empty string"
        raise ValueError(msg)
    work_root = Path(work_dir).expanduser().resolve()
    decision = (
        _StartDecision()
        if port == 0
        else _reserve_start(work_root, host, port, persist_catalog)
    )
    if decision.service is not None:
        return decision.service
    registration = decision.registration
    if decision.dead_service is not None:
        try:
            decision.dead_service.stop()
        except BaseException as error:
            if registration is not None:
                _abandon_start(registration, error)
            raise
    if not _is_loopback(host):
        logger.warning(
            "binding the STAC service to non-loopback host %s makes it "
            "reachable beyond this machine",
            host,
        )
    sock: socket.socket | None = None
    runtime: _ServerRuntime | None = None
    try:
        web = _load_web()
        io = _load_io()
        index = io.build_index(work_root)
        sock = _bind_socket(host, port)
        assigned = int(sock.getsockname()[1])
        base_url = _base_url(host, assigned)
        if registration is None:
            registration = _ServiceRegistration(
                key=_service_key(host, assigned),
                work_dir=work_root,
                persist_catalog=persist_catalog,
            )
        state = _IndexState(index)
        app = _create_app(web, io, state, work_root, base_url)
        runtime = _ServerRuntime(web, app, sock)
        runtime.start()
        persistence = _persist(io, index, work_root, persist_catalog)
        report = _build_report(index, persistence)
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
        print(  # noqa: T201
            f"FanInSAR STAC Catalog: {base_url}/catalog.json", flush=True
        )
        print(f"FanInSAR STAC API: {base_url}/api", flush=True)  # noqa: T201
    except BaseException as error:
        if runtime is not None:
            runtime.stop()
        if sock is not None:
            with suppress(OSError):
                sock.close()
        if registration is not None:
            _abandon_start(registration, error)
        raise
    assert registration is not None
    if not _publish_running(registration, service):
        service.stop()
        msg = (
            f"cannot publish the STAC service on {base_url}: another live "
            "service owns that endpoint"
        )
        raise RuntimeError(msg)
    return service
