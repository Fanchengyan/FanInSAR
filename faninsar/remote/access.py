"""Endpoint, redirect, and representation access policy helpers."""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .errors import RemoteAccessError, RemoteIntegrityError, _fail

if TYPE_CHECKING:
    from pathlib import Path

    from .protocols import _Adapter
    from .records import RemoteAsset

_SECRET_KEY = re.compile(
    r"(?:authorization|cookie|token|password|passwd|secret|signature|credential|api.?key)",
    re.IGNORECASE,
)
_SIGNED_QUERY_KEY = re.compile(
    r"^(?:sig(?:nature)?|token|expires?|se(?:curity)?|x-amz-|authorization|"
    r"sp|st|sv|sr|spr|sip|si|skoid|sktid|skt|ske|sks|skv|rscc|rscd|rsce|rscl|rsct|"
    r"ss|srt|sdd)$",
    re.IGNORECASE,
)
_AZURE_SAS_KEY = re.compile(
    r"^(?:sp|st|se|sv|sr|spr|sip|si|sig|skoid|sktid|skt|ske|sks|skv|"
    r"rscc|rscd|rsce|rscl|rsct|ss|srt|sdd)$",
    re.IGNORECASE,
)
_ASF_EDL_ORIGIN = "https://urs.earthdata.nasa.gov"
_ASF_AUTH_ORIGIN = "https://cumulus.asf.alaska.edu"
_ASF_DATA_ORIGIN = "https://datapool.asf.alaska.edu"
_ASF_SENTINEL1_ORIGIN = "https://sentinel1.asf.alaska.edu"
_ASF_NISAR_ORIGIN = "https://nisar.asf.earthdatacloud.nasa.gov"
_ASF_NISAR_BUCKET = "sds-n-cumulus-prod-nisar-products.s3.us-west-2.amazonaws.com"
_ASF_EDL_CLIENT_ID = "BO_n7nTIlMljdvU6kRRB3g"
_ASF_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_ASF_CLOUDFRONT_HOST = re.compile(r"^[a-z0-9]+\.cloudfront\.net$")
_ASF_SLC_CLOUDFRONT_PATH = re.compile(
    r"^/s3-[^/]+/asf-ngap2w-p-s1-slc-[a-z0-9]+\."
    r"s3\.us-west-2\.amazonaws\.com/[^/]+\.zip$",
    re.IGNORECASE,
)
_ASF_NISAR_CLOUDFRONT_PATH = re.compile(
    r"^/s3-[^/]+/sds-n-cumulus-prod-nisar-products\.s3\.us-west-2\.amazonaws\.com"
    r"(?P<key>/DEM/v1\.2/.+)$",
    re.IGNORECASE,
)
_ASF_NISAR_GATEWAY_PREFIX = "/NISAR"
_LPDAAC_DATA_ORIGIN = "https://data.lpdaac.earthdatacloud.nasa.gov"
_LPDAAC_BUCKET = "lp-prod-protected"
_LPDAAC_BUCKET_HOST = "lp-prod-protected.s3.us-west-2.amazonaws.com"
_LPDAAC_CLOUDFRONT_PATH = re.compile(
    rf"^/s3-[0-9a-f]{{32}}/{re.escape(_LPDAAC_BUCKET_HOST)}"
    r"(?P<key>/[^/].*)$"
)


def _url_origin(url: str) -> str:
    """Return the normalized origin for an HTTPS URL."""
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname
    if not host:
        message = "URL has no hostname"
        raise ValueError(message)
    port = parsed.port
    return f"https://{host.lower()}" + (f":{port}" if port and port != 443 else "")


def _validate_url(
    url: str,
    adapter: _Adapter,
    *,
    redirect: bool = False,
) -> str:
    """Validate a URL while preserving request-local query credentials.

    Query strings are intentionally not inspected or rewritten here.  A
    provider may need a short-lived signed query on an immediate transfer or
    redirect.  Call :func:`_safe_url` at persistence boundaries to obtain the
    scrubbed canonical representation instead.
    """
    if not isinstance(url, str) or not url:
        _fail(RemoteAccessError, "invalid_endpoint")
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        _fail(RemoteAccessError, "invalid_endpoint")
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        _fail(RemoteAccessError, "invalid_endpoint")
    path_lower = parsed.path.lower()
    if "\\" in parsed.path or "%2f" in path_lower or "%5c" in path_lower:
        _fail(RemoteAccessError, "invalid_endpoint")
    segments = urllib.parse.unquote(parsed.path).split("/")
    if any(segment in {".", ".."} for segment in segments):
        _fail(RemoteAccessError, "invalid_endpoint")
    try:
        origin = _url_origin(url)
    except ValueError:
        _fail(RemoteAccessError, "invalid_endpoint")
    registered_origins: list[str] = []
    origins = getattr(adapter, "redirect_origins", ()) if redirect else adapter.origins
    for item in origins:
        try:
            registered_origin = _url_origin(item)
        except ValueError:
            continue
        registered_origins.append(registered_origin)
    if origin not in registered_origins:
        _fail(RemoteAccessError, "unregistered_endpoint")
    path = parsed.path or "/"
    origin_prefixes = getattr(adapter, "redirect_path_prefixes", {})
    scoped_redirect_paths = (
        redirect and isinstance(origin_prefixes, Mapping) and origin in origin_prefixes
    )
    prefixes = (
        origin_prefixes.get(origin, adapter.path_prefixes)
        if scoped_redirect_paths
        else adapter.path_prefixes
    ) or ("/",)
    allowed_path = False
    for prefix in prefixes:
        normalized_prefix = prefix.rstrip("/") or "/"
        # An explicitly scoped root is an exact root route.  The unscoped
        # default ``/`` remains a wildcard for registered data origins.
        path_matches = (
            path == normalized_prefix
            if scoped_redirect_paths
            else normalized_prefix in {"/", path}
            or path.startswith(normalized_prefix + "/")
        )
        if path_matches:
            allowed_path = True
            break
    if not allowed_path:
        _fail(RemoteAccessError, "unregistered_endpoint")
    # Return the original URL, including its query, for immediate network use.
    return url


def _registered_origin(url: str, adapter: _Adapter, *, redirect: bool) -> bool:
    """Return whether ``url`` belongs to one registered adapter origin."""
    try:
        origin = _url_origin(url)
    except (TypeError, ValueError):
        return False
    origins = getattr(adapter, "redirect_origins", ()) if redirect else adapter.origins
    for candidate in origins:
        try:
            if _url_origin(candidate) == origin:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _validate_anonymous_delivery_url(
    source_url: str,
    target_url: str,
    adapter: _Adapter,
    *,
    source_is_anonymous: bool = False,
) -> str:
    """Validate a credential-free HTTPS object target after a trusted hop.

    The source must be a registered redirect origin, which is the provider
    gateway trust boundary, unless this operation has already crossed that
    boundary anonymously.  Once that boundary has been crossed, the target
    host and storage path are intentionally not registry-owned.  Structural
    URL checks still reject the forms that could change authority or route a
    request through a traversal-like path.
    """
    if not source_is_anonymous and not _registered_origin(
        source_url, adapter, redirect=True
    ):
        _fail(RemoteAccessError, "unregistered_endpoint")
    if not isinstance(target_url, str) or not target_url:
        _fail(RemoteAccessError, "invalid_endpoint")
    try:
        parsed = urllib.parse.urlsplit(target_url)
    except ValueError:
        _fail(RemoteAccessError, "invalid_endpoint")
    try:
        port = parsed.port
    except ValueError:
        _fail(RemoteAccessError, "invalid_endpoint")
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
        or (port is None and ":" in parsed.netloc.rsplit("@", 1)[-1])
    ):
        _fail(RemoteAccessError, "invalid_endpoint")
    path_lower = parsed.path.lower()
    if "\\" in parsed.path or "%2f" in path_lower or "%5c" in path_lower:
        _fail(RemoteAccessError, "invalid_endpoint")
    segments = urllib.parse.unquote(parsed.path).split("/")
    if any(segment in {".", ".."} for segment in segments):
        _fail(RemoteAccessError, "invalid_endpoint")
    try:
        # Accessing ``port`` above catches malformed ``host:port`` values.
        _url_origin(target_url)
    except (TypeError, ValueError):
        _fail(RemoteAccessError, "invalid_endpoint")
    return target_url


def _has_signed_query(url: str) -> bool:
    """Return whether a URL query contains request-capability material."""
    try:
        query = urllib.parse.parse_qsl(
            urllib.parse.urlsplit(url).query,
            keep_blank_values=True,
        )
    except ValueError:
        return False
    return any(
        _SECRET_KEY.search(key) or _SIGNED_QUERY_KEY.fullmatch(key)
        for key, _value in query
    )


def _response_is_complete(response: Any) -> None:
    """Require a complete, un-ranged HTTP response before streaming."""
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(response, "status", None)
    headers = getattr(response, "headers", {})
    has_content_range = isinstance(headers, Mapping) and any(
        str(name).lower() == "content-range" and value
        for name, value in headers.items()
    )
    if status == 206 or has_content_range:
        _fail(RemoteAccessError, "partial_content")
    if status != 200:
        _fail(RemoteAccessError, "unexpected_status")


def _representation_kind(asset: RemoteAsset) -> str | None:
    """Infer a binary representation classifier from safe asset metadata."""
    media_type = (asset.media_type or "").lower().split(";", 1)[0].strip()
    path = urllib.parse.urlsplit(asset.href).path.lower()
    if "zip" in media_type or path.endswith(".zip"):
        return "zip"
    if media_type in {"image/tiff", "image/geotiff"} or path.endswith(
        (".tif", ".tiff", ".geotiff")
    ):
        return "tiff"
    if "hdf" in media_type or path.endswith((".h5", ".hdf", ".hdf5")):
        return "hdf5"
    return None


def _validate_representation(staging: Path, asset: RemoteAsset) -> None:
    """Validate the bounded binary prefix before atomic publication."""
    kind = _representation_kind(asset)
    if kind is None:
        return
    try:
        with staging.open("rb") as stream:
            prefix = stream.read(512)
    except OSError:
        _fail(RemoteIntegrityError, "representation_unreadable")
    probe = prefix.removeprefix(b"\xef\xbb\xbf").lstrip(b" \t\r\n")
    lowered = probe[:16].lower()
    if lowered.startswith((b"<!doctype html", b"<html", b"<?xml")):
        _fail(RemoteIntegrityError, "representation_mismatch")
    magic: dict[str, tuple[bytes, ...]] = {
        "zip": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
        "tiff": (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"),
        "hdf5": (b"\x89HDF\r\n\x1a\n",),
    }
    if not any(prefix.startswith(value) for value in magic[kind]):
        _fail(RemoteIntegrityError, "representation_mismatch")


def _safe_url(
    url: str,
    adapter: _Adapter,
    *,
    redirect: bool = False,
) -> str:
    """Validate and return a scrubbed canonical URL for persistence.

    This compatibility wrapper retains the historical private helper name for
    provider adapters.  Network redirect code uses :func:`_validate_url`
    directly so signed ``Location`` queries survive the hop.
    """
    validated = _validate_url(url, adapter, redirect=redirect)
    parsed = urllib.parse.urlsplit(validated)
    origin = _url_origin(validated)
    return urllib.parse.urlunsplit(
        ("https", origin.removeprefix("https://"), parsed.path or "/", "", "")
    )


def _canonicalize_url(
    url: str,
    adapter: _Adapter,
    *,
    redirect: bool = False,
) -> str:
    """Return the persistence-safe URL after endpoint validation."""
    return _safe_url(url, adapter, redirect=redirect)


def _strip_redirect_credentials(
    request: urllib.request.Request,
) -> None:
    """Remove credential headers before an approved cross-origin redirect."""
    for name in list(request.headers):
        if _SECRET_KEY.search(str(name)):
            del request.headers[name]


def _redirect_headers(
    headers: Mapping[str, str], source_origin: str, target_origin: str
) -> dict[str, str]:
    """Apply the minimal credential policy to one cross-origin hop.

    Bearer authentication is retained only for the ASF datapool-to-Sentinel-1
    handoff.  In particular, Basic Earthdata Login credentials never cross
    URS or reach a CloudFront object URL.
    """
    filtered = {
        key: value for key, value in headers.items() if not _SECRET_KEY.search(key)
    }
    authorization = next(
        (value for key, value in headers.items() if key.lower() == "authorization"),
        None,
    )
    if (
        source_origin == _ASF_DATA_ORIGIN
        and target_origin == _ASF_SENTINEL1_ORIGIN
        and isinstance(authorization, str)
        and authorization.lower().startswith("bearer ")
    ):
        filtered["Authorization"] = authorization
    return filtered


def _asf_redirect_url(
    source_url: str,
    target_url: str,
    asset: RemoteAsset,
    adapter: _Adapter,
    *,
    anonymous: bool = False,
) -> str:
    """Validate one ASF redirect and its anonymous delivery handoff."""
    del asset
    try:
        return _validate_url(target_url, adapter, redirect=True)
    except RemoteAccessError:
        return _validate_anonymous_delivery_url(
            source_url,
            target_url,
            adapter,
            source_is_anonymous=anonymous,
        )


def _lpdaac_redirect_url(
    source_url: str,
    target_url: str,
    asset: RemoteAsset,
    adapter: _Adapter,
    *,
    anonymous: bool = False,
) -> str:
    """Validate one LPDAAC redirect and its anonymous delivery handoff."""
    del asset
    try:
        return _validate_url(target_url, adapter, redirect=True)
    except RemoteAccessError:
        return _validate_anonymous_delivery_url(
            source_url,
            target_url,
            adapter,
            source_is_anonymous=anonymous,
        )


__all__ = [name for name in globals() if name.startswith("_")]
