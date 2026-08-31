# ruff: noqa: E501, EM101, EM102, TRY003
"""Bounded provider transport at the DEM I/O boundary.

The transport deliberately uses ordinary writes.  A destination is useful only
after its complete protocol length and byte budget have been checked.
"""

from __future__ import annotations

import re
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING

from faninsar.logging import setup_logger

from .resources import ResourceBudget, preflight_transfer

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = setup_logger(__name__)

_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.IGNORECASE)


class BoundedTransferError(ValueError):
    """Raised for malformed protocol metadata or a stream over its bound."""


def resolve_cache_path(root: str | Path, relative: str | Path) -> Path:
    """Resolve a cache-relative path and reject traversal on every platform."""
    base = Path(root).expanduser().resolve(strict=False)
    value = str(relative).replace("\\", "/")
    if "\x00" in value:
        raise ValueError("cache path may not contain NUL")
    if value.startswith("/") or re.match(r"^[A-Za-z]:(?:/|$)", value):
        raise ValueError("cache path must be relative")
    candidate = (base / Path(value)).resolve(strict=False)
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise ValueError("cache path escapes configured root") from error
    return candidate


def validate_https_origin(url: str, allowed_hosts: set[str] | frozenset[str]) -> None:
    """Require HTTPS and a registry-owned host before connecting."""
    if "\x00" in url:
        raise ValueError("URL may not contain NUL")
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError:
        port = -1
    if (
        parts.scheme.lower() != "https"
        or not host
        or host not in {h.lower() for h in allowed_hosts}
        or parts.username is not None
        or parts.password is not None
        or port not in (None, 443)
    ):
        raise ValueError(f"URL origin is not registry-owned: {_redact_url(url)}")


def _redact_url(url: str) -> str:
    """Remove URL userinfo, query values, and fragments from diagnostics."""
    parts = urllib.parse.urlsplit(url)
    hostname = parts.hostname or ""
    try:
        port = parts.port
    except ValueError:
        port = "REDACTED"
    netloc = hostname if port in (None, 443) else f"{hostname}:{port}"
    query = ""
    if parts.query:
        query = urllib.parse.urlencode(
            [(key, "REDACTED") for key, _ in urllib.parse.parse_qsl(
                parts.query, keep_blank_values=True
            )]
        )
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, query, ""))


def validate_redirect(
    start_url: str,
    location: str,
    *,
    allowed_hosts: set[str] | frozenset[str],
    credentials: Mapping[str, str] | None = None,
) -> str:
    """Resolve one redirect and forbid credentials on cross-host hops."""
    target = urllib.parse.urljoin(start_url, location)
    start_host = (urllib.parse.urlsplit(start_url).hostname or "").lower()
    target_host = (urllib.parse.urlsplit(target).hostname or "").lower()
    validate_https_origin(target, allowed_hosts)
    if target_host != start_host and credentials:
        raise ValueError("credentials may not cross an HTTPS host boundary")
    return target


def _response_metadata(response: object) -> tuple[int | None, int | None, Mapping[str, str]]:
    headers = getattr(response, "headers", {})
    status = getattr(response, "status_code", None)
    length = headers.get("Content-Length") if hasattr(headers, "get") else None
    try:
        length_value = int(length) if length is not None else None
    except (TypeError, ValueError) as error:
        raise BoundedTransferError("invalid Content-Length") from error
    return status, length_value, headers


def stream_to_cache(
    chunks: Iterable[bytes],
    destination: str | Path,
    *,
    max_bytes: int,
    expected_length: int | None = None,
    status_code: int | None = None,
    content_range: str | None = None,
) -> int:
    """Stream bytes with bounded accounting and exact protocol validation.

    ``status_code`` may be 200 for a complete response or 206 for a range.
    FTP and test transports leave it as ``None`` and may still provide an exact
    ``expected_length``.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise BoundedTransferError("maximum transfer bytes must be positive")
    if expected_length is not None:
        preflight_transfer(expected_length, budget=ResourceBudget(max_fetch_bytes=max_bytes))
    if status_code not in (None, 200, 206):
        raise BoundedTransferError(f"unsupported transfer status {status_code}")
    if status_code == 206:
        if not content_range:
            raise BoundedTransferError("HTTP 206 requires Content-Range")
        match = _CONTENT_RANGE.fullmatch(content_range.strip())
        if match is None:
            raise BoundedTransferError("invalid Content-Range")
        start, end, total = (int(match.group(1)), int(match.group(2)), match.group(3))
        if expected_length is None:
            expected_length = end - start + 1
        if end < start or expected_length != end - start + 1:
            raise BoundedTransferError("HTTP range length does not match Content-Range")
        if total != "*" and int(total) < end + 1:
            raise BoundedTransferError("Content-Range exceeds resource length")
        preflight_transfer(
            expected_length,
            budget=ResourceBudget(max_fetch_bytes=max_bytes),
        )
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("wb") as output:
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise BoundedTransferError("transport yielded a non-bytes chunk")
            written += len(chunk)
            if written > max_bytes:
                raise BoundedTransferError("stream exceeded maximum transfer bytes")
            output.write(chunk)
    if expected_length is not None and written != expected_length:
        raise BoundedTransferError(
            f"transfer length mismatch: expected {expected_length}, got {written}"
        )
    return written


def stream_response_to_cache(
    response: object,
    destination: str | Path,
    *,
    max_bytes: int,
    expected_length: int | None = None,
) -> int:
    """Adapt a requests-like HTTP response to :func:`stream_to_cache`."""
    status, content_length, headers = _response_metadata(response)
    declared = expected_length if expected_length is not None else content_length
    range_value = headers.get("Content-Range") if hasattr(headers, "get") else None
    iterator = response.iter_content(chunk_size=1024 * 1024)
    return stream_to_cache(
        iterator,
        destination,
        max_bytes=max_bytes,
        expected_length=declared,
        status_code=status,
        content_range=range_value,
    )


def download(
    url: str,
    destination: str | Path,
    *,
    allowed_hosts: set[str] | frozenset[str],
    max_bytes: int,
    expected_length: int | None = None,
    session: object | None = None,
) -> Path:
    """Download an approved HTTPS resource with bounded streaming.

    Redirects are followed manually only when every hop remains in the
    registry-owned host set.  No credentials are attached by this primitive.
    """
    import requests

    current = url
    client = session or requests.Session()
    for _ in range(4):
        validate_https_origin(current, allowed_hosts)
        response = client.get(
            current,
            stream=True,
            allow_redirects=False,
            timeout=(10, 120),
        )
        try:
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location:
                    raise BoundedTransferError("redirect response has no Location")
                current = validate_redirect(
                    current, location, allowed_hosts=allowed_hosts
                )
                continue
            response.raise_for_status()
            stream_response_to_cache(
                response,
                destination,
                max_bytes=max_bytes,
                expected_length=expected_length,
            )
            return Path(destination)
        finally:
            response.close()
    raise BoundedTransferError("too many redirects for DEM resource")


def download_ftp(
    url: str,
    destination: str | Path,
    *,
    max_bytes: int,
    expected_length: int | None = None,
) -> Path:
    """Stream an FTP resource with the same bounded accounting contract."""
    import urllib.request

    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() != "ftp" or not parts.hostname:
        raise ValueError("FTP download requires an ftp URL")
    with urllib.request.urlopen(url, timeout=120) as response:
        declared = expected_length
        if declared is None:
            raw_length = response.headers.get("Content-Length")
            declared = int(raw_length) if raw_length else None
        return_value = stream_to_cache(
            iter(lambda: response.read(1024 * 1024), b""),
            destination,
            max_bytes=max_bytes,
            expected_length=declared,
        )
    del return_value
    return Path(destination)


__all__ = [
    "BoundedTransferError",
    "download",
    "download_ftp",
    "resolve_cache_path",
    "stream_response_to_cache",
    "stream_to_cache",
    "validate_https_origin",
    "validate_redirect",
]
