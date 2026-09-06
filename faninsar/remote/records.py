"""Immutable records and normalization helpers for remote catalogs."""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from .errors import RemoteQueryError, _fail


@dataclass(frozen=True, slots=True)
class RemoteResourceBudget:
    """Finite limits for remote search and complete-file transfer.

    Parameters
    ----------
    max_items : int
        Maximum number of catalog items returned.
    max_requests : int
        Maximum number of transfer attempts.
    max_response_bytes : int
        Maximum bytes accepted from one response.
    max_output_bytes : int
        Maximum bytes written for one asset.
    connect_timeout_seconds, read_timeout_seconds : float
        Standard-library request timeout values.

    """

    max_items: int = 100
    max_requests: int = 256
    max_redirects: int = 5
    max_retries: int = 4
    max_elapsed_seconds: float = 900.0
    max_workers: int = 8
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 120.0
    max_response_bytes: int = 2**31
    max_operation_bytes: int = 2**33
    max_output_bytes: int = 2**31
    max_temporary_bytes: int = 2**33
    max_cache_bytes: int = 2**33

    def __post_init__(self) -> None:
        """Reject non-finite or non-positive resource limits."""
        for name in (
            "max_items",
            "max_requests",
            "max_redirects",
            "max_retries",
            "max_workers",
            "max_response_bytes",
            "max_operation_bytes",
            "max_output_bytes",
            "max_temporary_bytes",
            "max_cache_bytes",
        ):
            if getattr(self, name) <= 0:
                msg = f"{name} must be positive"
                raise ValueError(msg)
        for name in (
            "max_elapsed_seconds",
            "connect_timeout_seconds",
            "read_timeout_seconds",
        ):
            if getattr(self, name) <= 0:
                msg = f"{name} must be positive"
                raise ValueError(msg)


def _freeze(value: Any) -> Any:
    """Recursively make provider metadata immutable and JSON-compatible."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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


def _sanitize(value: Any, key: str | None = None) -> Any:  # noqa: PLR0911
    """Remove credential-bearing values from nested provider data."""
    if key is not None and (_SECRET_KEY.search(key) or _SIGNED_QUERY_KEY.search(key)):
        return None
    if isinstance(value, Mapping):
        return {
            str(k): _sanitize(v, str(k))
            for k, v in value.items()
            if not (_SECRET_KEY.search(str(k)) or _SIGNED_QUERY_KEY.search(str(k)))
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme and parsed.netloc:
            if parsed.username or parsed.password:
                return urllib.parse.urlunsplit(
                    (parsed.scheme, parsed.hostname or "", parsed.path, "", "")
                )
            query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
            # An Azure SAS URL signs the complete query string.  Retaining an
            # unrecognised parameter alongside the SAS fields can still leak
            # signed request material, so drop the entire query whenever a
            # SAS marker is present.  The provider may retain a request-local
            # signed URL separately for the immediate transfer.
            if any(_AZURE_SAS_KEY.fullmatch(k) for k, _ in query):
                return urllib.parse.urlunsplit(
                    (parsed.scheme, parsed.netloc, parsed.path, "", "")
                )
            query = [
                (k, v)
                for k, v in query
                if not (_SIGNED_QUERY_KEY.search(k) or _SECRET_KEY.search(k))
            ]
            clean_query = urllib.parse.urlencode(query)
            return urllib.parse.urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path, clean_query, "")
            )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _utc(value: Any) -> datetime | None:
    """Normalize an optional datetime to timezone-aware UTC."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            _fail(RemoteQueryError, "invalid_datetime", str(exc))
    if not isinstance(value, datetime):
        msg = "acquisition time is not a datetime"
        _fail(RemoteQueryError, "invalid_datetime", msg)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class AcquisitionMetadata:
    """Small normalized description of an acquisition."""

    acquisition_id: str
    start: datetime | None = None
    end: datetime | None = None
    platform: str | None = None
    instrument: str | None = None
    mode: str | None = None
    orbit: int | None = None
    processing_level: str | None = None
    geometry: Mapping[str, object] | None = None
    properties: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize times and protect nested metadata from mutation."""
        object.__setattr__(self, "start", _utc(self.start))
        object.__setattr__(self, "end", _utc(self.end))
        if self.start is not None and self.end is not None and self.end < self.start:
            _fail(RemoteQueryError, "invalid_acquisition_interval")
        geometry = (
            _freeze(_sanitize(self.geometry)) if self.geometry is not None else None
        )
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "properties", _freeze(_sanitize(self.properties)))


@dataclass(frozen=True, slots=True)
class RemoteAsset:
    """An immutable, registered remote asset descriptor."""

    provider: str
    catalog: str
    collection: str | None
    item_id: str
    key: str
    href: str
    media_type: str | None = None
    size_bytes: int | None = None
    checksum: str | None = None
    version: str | None = None
    representation: str = "complete"
    auth_profile: str = "anonymous"
    properties: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize checksum and protect asset metadata from mutation."""
        object.__setattr__(self, "checksum", _normalize_checksum(self.checksum))
        # The request-local signed URL, when one exists, belongs to the
        # provider adapter.  A public asset descriptor must never retain its
        # query credentials (including Azure SAS fields).
        object.__setattr__(self, "href", _sanitize(self.href))
        object.__setattr__(self, "properties", _freeze(_sanitize(self.properties)))


@dataclass(frozen=True, slots=True)
class CatalogItem:
    """An immutable normalized catalog item and its assets."""

    provider: str
    catalog: str
    collection: str | None
    item_id: str
    acquisition: AcquisitionMetadata
    assets: Mapping[str, RemoteAsset]
    raw_metadata: Mapping[str, object]
    matched_point_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Protect asset and metadata mappings from mutation."""
        object.__setattr__(self, "assets", MappingProxyType(dict(self.assets)))
        object.__setattr__(self, "raw_metadata", _freeze(_sanitize(self.raw_metadata)))
        object.__setattr__(
            self, "matched_point_indices", tuple(self.matched_point_indices)
        )


def _normalize_checksum(value: Any) -> str | None:
    """Accept only the qualified checksum grammar used by the cache."""
    if not isinstance(value, str) or ":" not in value:
        return None
    algorithm, digest = value.split(":", 1)
    algorithm = algorithm.lower()
    if algorithm not in {"sha256", "sha512"} or not re.fullmatch(
        r"[0-9a-fA-F]+", digest
    ):
        return None
    expected = 64 if algorithm == "sha256" else 128
    return f"{algorithm}:{digest.lower()}" if len(digest) == expected else None


__all__ = ["AcquisitionMetadata", "CatalogItem", "RemoteAsset", "RemoteResourceBudget"]
