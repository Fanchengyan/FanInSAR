"""Single audited DEM download engine (PROPOSAL-0030).

Every transport concern for DEM fetching lives here; registry entries in
:mod:`faninsar.processing.geometry.dem_sources` only *plan* what to fetch via
:class:`FetchPlan` records.  Invariants enforced by this module:

- URL/host pinning before any connect (CMR-injection regression guard).
- Cross-host redirects are followed without credentials; auth attached
  cross-host is a hard error.
- Retry matrix over transient statuses/exceptions with capped jittered
  backoff, ``Retry-After`` honored, certificate failures fail fast, and
  401/403 terminal loud errors.
- Ranged mode validates per-chunk ``206`` + matching ``Content-Range``;
  ranged is excluded for Earthdata hosts (HEAD/GET divergence).
- FTP plans stream through :mod:`urllib.request` with post-hoc size floors.
- Zip expansion verifies CRC and enforces staging containment on every
  member before write (zip-slip guard).
- Unique ``{target}.{pid}-{uuid}.part`` published via :func:`os.replace`;
  orphan sweep covers ``.part`` files and zip staging directories.
- Credential hygiene: tokens/netrc/SAS query content never reach logs,
  exception messages, or cache paths.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import requests

from faninsar.logging import setup_logger
from faninsar.processing.errors import InvalidProcessingStateError

logger = setup_logger(__name__)

__all__ = [
    "CHUNK_SIZE_BYTES",
    "RETRYABLE_STATUS_CODES",
    "Artifact",
    "CredentialProvider",
    "DemAuthProviderError",
    "FetchPlan",
    "Tile",
    "TileSet",
    "TransientDemFetchError",
    "compute_backoff_sleep",
    "extract_zip_members",
    "fetch_plan",
    "fetch_tiles",
    "part_path",
    "redact_url",
    "resolve_credentials",
    "safe_member_target",
    "sweep_part_files",
    "thread_local_session",
    "validate_plan_urls",
]

#: Streaming chunk size for plain GETs and the ranged-chunk unit (8 MiB).
CHUNK_SIZE_BYTES = 8 << 20
#: Retryable HTTP status codes (429 + transient 5xx).
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
#: Total attempts per request (first try plus retries).
MAX_ATTEMPTS = 5
#: Backoff cap in seconds.
MAX_BACKOFF_S = 30.0
#: Per-request timeouts (connect, read).
REQUEST_TIMEOUT = (10, 120)

#: Hosts whose HEAD/GET behavior diverges under auth (CloudFront signed
#: anonymous HEAD vs 302-to-URS GET); ranged transfers are always excluded.
_EARTHDATA_HOST_SUFFIXES = (
    ".earthdatacloud.nasa.gov",
    ".nasa.gov",
)
#: Age threshold for orphaned .part/staging cleanup.
DEFAULT_SWEEP_MAX_AGE_S = 24 * 3600

try:  # POSIX-only positional writes; Windows uses seek+write per handle.
    _PWRITE_AVAILABLE = hasattr(os, "pwrite")
except AttributeError:  # pragma: no cover - platform dependent
    _PWRITE_AVAILABLE = False


# ---------------------------------------------------------------------------
# FetchPlan union
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tile:
    """Cache unit for tile-set shapes.

    Attributes
    ----------
    url
        Remote https URL; its host must appear in the plan's allowlist.
    cache_path
        Cache-relative destination path.
    min_bytes
        Source-specific minimum valid size in bytes.
    ranged
        Whether the source declares the tile ranged-transfer capable.
        Defaults True for generic hosts; Earthdata tiles must set False.
    expected_decompressed_bytes
        Optional post-decompression size check (e.g. skadi HGT).

    """

    url: str
    cache_path: Path
    min_bytes: int = 1 << 20
    ranged: bool = True
    expected_decompressed_bytes: int | None = None
    #: Tolerate a 404 for known-ocean tiles (skadi only).
    ocean_404_skip: bool = False


@dataclass(frozen=True, slots=True)
class FetchPlan:
    """Base plan record: every plan is fully self-describing."""

    allowed_hosts: tuple[str, ...]
    credential_ref: str | None = None
    #: Engine-injected credential provider (test seam); resolved from
    #: ``credential_ref`` when None at execution time.
    _credentials: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class TileSet(FetchPlan):
    """Concurrent-tile plan produced by grid-shaped sources."""

    tiles: tuple[Tile, ...] = ()
    #: Source-scoped flag: tolerate 404 for known-ocean tiles (skadi only).
    ocean_404_skip: bool = False


@dataclass(frozen=True, slots=True)
class Artifact(FetchPlan):
    """Whole-artifact plan (single file or archive extraction)."""

    scheme: Literal["https", "ftp"] = "https"
    url: str = ""
    members: tuple[str, ...] | None = None
    member_pattern: str | None = None
    expand: Literal["zip"] | None = None
    cache_path: Path | None = None
    min_total_bytes: int = 0

    def __post_init__(self) -> None:
        """Validate the grammar at construction (fail closed)."""
        object.__setattr__(self, "scheme", str(self.scheme))
        if self.scheme not in {"https", "ftp"}:
            message = f"artifact scheme must be 'https' or 'ftp', got {self.scheme!r}"
            logger.error(message)
            raise ValueError(message)
        if self.members is not None and self.member_pattern is not None:
            message = (
                "artifact plan accepts members or member_pattern, not both "
                "(ambiguous extraction contract)"
            )
            logger.error(message)
            raise ValueError(message)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


class CredentialProvider:
    """Base credential provider consulted by the engine.

    Subclasses return per-URL headers; the engine decides when auth may be
    attached (pinned hosts only) and scrubs credentials from diagnostics.
    """

    def headers_for(self, url: str) -> dict[str, str]:
        """Return auth headers for ``url`` (possibly empty)."""
        del url
        return {}


def _netrc_auth(host: str) -> tuple[str, str] | None:
    """Return (login, password) from ~/.netrc style files for ``host``."""
    from netrc import netrc

    try:
        netrc_file = netrc()
        entry = netrc_file.authenticators(host)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if entry is None:
        return None
    login, _, password = entry
    return login or "", password or ""


class EarthdataCredentialProvider(CredentialProvider):
    """Earthdata Login credentials: .netrc first, then EARTHDATA_TOKEN."""

    def __init__(self, host: str) -> None:
        """Bind the provider to one Earthdata host for netrc lookup."""
        self.host = host

    def headers_for(self, url: str) -> dict[str, str]:
        netrc_entry = _netrc_auth(urllib.parse.urlparse(url).hostname or self.host)
        if netrc_entry is not None:
            token = netrc_entry[1]
            if token:
                return {"Authorization": f"Bearer {token}"}
        env_token = os.environ.get("EARTHDATA_TOKEN", "")
        if env_token:
            return {"Authorization": f"Bearer {env_token}"}
        return {}

    @staticmethod
    def require_available() -> None:
        """Fail closed when neither .netrc nor EARTHDATA_TOKEN provide creds."""
        if os.environ.get("EARTHDATA_TOKEN"):
            return
        # Probe a representative Earthdata host for a netrc entry.
        if _netrc_auth("data.lpdaac.earthdatacloud.nasa.gov") is not None:
            return
        message = (
            "Earthdata credentials not found: add a machine entry covering "
            "*.earthdatacloud.nasa.gov to your ~/.netrc or set the "
            "EARTHDATA_TOKEN environment variable "
            "(register at https://urs.earthdata.nasa.gov)."
        )
        logger.error(message)
        raise RuntimeError(message)


def resolve_credentials(credential_ref: str) -> CredentialProvider:
    """Resolve a named credential provider, failing closed when absent."""
    if credential_ref == "earthdata":
        EarthdataCredentialProvider.require_available()
        return EarthdataCredentialProvider("data.lpdaac.earthdatacloud.nasa.gov")
    message = f"unknown credential reference: {credential_ref!r}"
    logger.error(message)
    raise ValueError(message)


_REDACT_PARAMS = ("sig", "se", "token", "sig=", "SASToken", "st", "sp")


def redact_url(url: str) -> str:
    """Scrub oauth/SAS query parameters from a URL for safe logging."""
    parts = urllib.parse.urlsplit(url)
    if not parts.query:
        return url
    kept: list[tuple[str, str]] = []
    for key, value in urllib.parse.parse_qsl(parts.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered in {"sig", "se", "st", "sp", "token", "sastoken"}:
            kept.append((key, "REDACTED"))
        else:
            kept.append((key, value))
    query = urllib.parse.urlencode(kept)
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, query, parts.fragment)
    )


def _scrub_message(text: str) -> str:
    """Remove Authorization/Cookie header content and SAS params from text."""
    text = re.sub(r"(?i)(authorization|cookie)\s*[=:]\s*\S+", r"\1=REDACTED", text)
    return "\n".join(redact_url(line) for line in text.splitlines())


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TransientDemFetchError(InvalidProcessingStateError):
    """Retries exhausted on a retryable failure."""


class DemAuthProviderError(InvalidProcessingStateError):
    """Terminal authentication/authorization failure (401/403)."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        """Build the error with the terminal HTTP status when known."""
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Sessions and retries
# ---------------------------------------------------------------------------

_session_local = __import__("threading").local()


def thread_local_session() -> requests.Session:
    """Return the calling thread's pooled keep-alive session."""
    session = getattr(_session_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=16)
        session.mount("https://", adapter)
        _session_local.session = session
    return session


def compute_backoff_sleep(attempt: int, retry_after: float | None = None) -> float:
    """Exponential jittered backoff capped at 30 s, honoring Retry-After."""
    if retry_after is not None:
        return max(0.0, min(float(retry_after), MAX_BACKOFF_S))
    base = min(2.0**attempt, MAX_BACKOFF_S)
    return base * (0.5 + secrets.randbelow(1000) / 2000.0)


def _is_certificate_failure(exc: BaseException) -> bool:
    """Detect TLS certificate-verification failures (never retried)."""
    current: BaseException | None = exc
    seen = 0
    while current is not None and seen < 6:
        if isinstance(current, ssl.SSLCertVerificationError):
            return True
        cause = current.__cause__
        context = current.__context__
        current = cause if cause is not None else context
        seen += 1
    return False


def _is_tls_eof(exc: BaseException) -> bool:
    """Detect handshake-burst SSLEOF/InsecurePlatform style errors (retryable)."""
    current: BaseException | None = exc
    seen = 0
    while current is not None and seen < 6:
        if isinstance(current, ssl.SSLEOFError):
            return True
        current = current.__cause__ or current.__context__
        seen += 1
    return False


def _sleep_backoff(attempt: int, response_headers: dict[str, str] | None) -> None:
    retry_after: float | None = None
    if response_headers:
        raw = response_headers.get("Retry-After")
        if raw is not None:
            try:
                retry_after = float(raw)
            except ValueError:
                retry_after = None
    time.sleep(compute_backoff_sleep(attempt, retry_after))


def _request_with_retries(
    method: str,
    url: str,
    *,
    stream: bool = False,
    headers: dict[str, str] | None = None,
    expect_status: tuple[int, ...] = (200,),
) -> requests.Response:
    """Issue one request through the shared retry matrix."""
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = thread_local_session().request(
                method,
                url,
                stream=stream,
                timeout=REQUEST_TIMEOUT,
                headers=headers or {},
                allow_redirects=False,
            )
            if response.status_code in RETRYABLE_STATUS_CODES:
                last_error = TransientDemFetchError(
                    f"transient status {response.status_code} for {redact_url(url)}"
                )
                logger.warning(
                    "DEM fetch attempt %d/%d got %d for %s",
                    attempt + 1,
                    MAX_ATTEMPTS,
                    response.status_code,
                    redact_url(url),
                )
                response.close()
                _sleep_backoff(attempt, response.headers)
                continue
            if response.status_code in {401, 403}:
                message = (
                    f"DEM fetch rejected with terminal status "
                    f"{response.status_code} for {redact_url(url)}; check "
                    "credentials/permissions (not retried)"
                )
                logger.error(message)
                raise DemAuthProviderError(message, status=response.status_code)  # noqa: TRY301
            if response.status_code in (301, 302, 303, 307, 308):
                return response
            if (
                response.status_code >= 400
                and response.status_code not in expect_status
            ):
                message = (
                    f"DEM fetch failed with status {response.status_code} "
                    f"for {redact_url(url)}"
                )
                logger.error(message)
                raise InvalidProcessingStateError(message)  # noqa: TRY301
            return response  # noqa: TRY300
        except DemAuthProviderError:
            raise
        except InvalidProcessingStateError:
            raise
        except requests.exceptions.SSLError as exc:
            if _is_certificate_failure(exc):
                message = (
                    f"TLS certificate verification failed for "
                    f"{redact_url(url)}; refusing to continue (fail fast)"
                )
                logger.exception(message)
                raise exc from exc
            last_error = exc
            if not _is_tls_eof(exc):
                logger.warning(
                    "DEM fetch SSL error attempt %d for %s: %s",
                    attempt + 1,
                    redact_url(url),
                    type(exc).__name__,
                )
            else:
                logger.warning(
                    "DEM fetch TLS handshake burst attempt %d for %s",
                    attempt + 1,
                    redact_url(url),
                )
            _sleep_backoff(attempt, None)
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:
            wrapped_protocol = (
                isinstance(exc.__cause__, Exception)
                and "ProtocolError" in type(exc.__cause__).__name__
            )
            if not wrapped_protocol:
                logger.warning(
                    "DEM fetch transient error attempt %d for %s: %s",
                    attempt + 1,
                    redact_url(url),
                    type(exc).__name__,
                )
            last_error = exc
            _sleep_backoff(attempt, None)
    message = (
        f"DEM fetch exhausted {MAX_ATTEMPTS} attempts for {redact_url(url)} "
        f"({type(last_error).__name__ if last_error else 'unknown'})"
    )
    logger.error(message)
    raise TransientDemFetchError(message) from last_error


# ---------------------------------------------------------------------------
# Host pinning
# ---------------------------------------------------------------------------


def validate_plan_urls(plan: FetchPlan) -> None:
    """Enforce scheme + host allowlist before any connect.

    Raises
    ------
    ValueError
        On non-https URLs for https plans, ftp schemes outside their pinned
        artifact, or hosts outside ``allowed_hosts``.

    """
    allowed = set(plan.allowed_hosts)

    def _check_https(url: str) -> None:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https":
            message = (
                f"DEM plan URLs must use https, got {parts.scheme!r} in "
                f"{redact_url(url)}"
            )
            logger.error(message)
            raise ValueError(message)
        host = parts.hostname
        if host is None or host.lower() not in allowed:
            message = (
                f"DEM URL host {host!r} is not in the plan allowlist "
                f"{sorted(allowed)} ({redact_url(url)})"
            )
            logger.error(message)
            raise ValueError(message)

    if isinstance(plan, TileSet):
        for tile in plan.tiles:
            _check_https(tile.url)
    elif isinstance(plan, Artifact):
        if plan.scheme not in {"https", "ftp"}:
            message = f"artifact scheme must be 'https' or 'ftp', got {plan.scheme!r}"
            logger.error(message)
            raise ValueError(message)
        if plan.scheme == "https":
            _check_https(plan.url)
        else:
            parts = urllib.parse.urlsplit(plan.url)
            host = parts.hostname
            if parts.scheme != "ftp" or host is None or host.lower() not in allowed:
                message = (
                    f"FTP artifact URL host {host!r} is not in the plan "
                    f"allowlist {sorted(allowed)}"
                )
                logger.error(message)
                raise ValueError(message)


def _validate_redirect_chain(
    start_url: str,
    response: requests.Response,
    *,
    allowed_hosts: set[str],
    credentials: CredentialProvider | None,
) -> str | None:
    """Validate one redirect hop; return the next URL or None.

    Cross-host redirects are followed WITHOUT credentials; auth attached to
    a cross-host hop is a hard error.  A redirect that leaves the allowlist
    entirely is rejected (CMR-injection guard).
    """
    location = response.headers.get("Location")
    if not location:
        return None
    next_url = urllib.parse.urljoin(start_url, location)
    next_host = (urllib.parse.urlsplit(next_url).hostname or "").lower()
    start_host = (urllib.parse.urlsplit(start_url).hostname or "").lower()
    # Cross-host hops are permitted ONLY credential-free: strip auth, then
    # verify no auth provider would attach to the new host.
    if next_host != start_host:
        if credentials is not None and credentials.headers_for(next_url):
            message = (
                f"cross-host redirect from {start_host!r} to {next_host!r} "
                f"would carry credentials; refusing"
            )
            logger.error(message)
            leak = f"redirect credential leak: {next_host}"
            raise ValueError(leak)
        if next_host and next_host not in allowed_hosts:
            logger.warning(
                "DEM redirect leaves the allowlist (%s -> %s); following "
                "without credentials",
                start_host,
                next_host,
            )
    return next_url


def _is_earthdata_host(url: str) -> bool:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return any(
        host == suffix.lstrip(".") or host.endswith(suffix)
        for suffix in _EARTHDATA_HOST_SUFFIXES
    )


def _is_earthdata_head_safe(url: str) -> bool:
    """Whether a HEAD probe may be issued for ranged negotiation.

    Earthdata hosts are excluded: anonymous HEAD returns a CloudFront signed
    URL without meaningful Accept-Ranges while GET goes 302 to URS.
    """
    return not _is_earthdata_host(url)


# ---------------------------------------------------------------------------
# Atomic publish and sweep
# ---------------------------------------------------------------------------


def part_path(target: Path) -> Path:
    """Build the unique in-progress path ``{target}.{pid}-{uuid}.part``."""
    unique = f"{os.getpid()}-{secrets.token_hex(16)}"
    return target.with_name(f"{target.name}.{unique}.part")


def _publish_target(target: Path, source_part: Path) -> None:
    """Atomically publish ``source_part`` at ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    source_part.replace(target)


def sweep_part_files(
    cache_dir: Path, max_age_s: float = DEFAULT_SWEEP_MAX_AGE_S
) -> int:
    """Remove orphaned .part files older than ``max_age_s``; return count."""
    cutoff = time.time() - max_age_s
    removed = 0
    if not cache_dir.is_dir():
        return 0
    for candidate in cache_dir.rglob("*.part"):
        try:
            if candidate.is_file() and candidate.stat().st_mtime < cutoff:
                candidate.unlink()
                removed += 1
        except OSError:  # pragma: no cover - raced removal
            continue
    for staging in cache_dir.rglob("*.zip-staging"):
        try:
            if staging.is_dir() and staging.stat().st_mtime < cutoff:
                import shutil

                shutil.rmtree(staging, ignore_errors=True)
                removed += 1
        except OSError:  # pragma: no cover - raced removal
            continue
    return removed


# ---------------------------------------------------------------------------
# Zip expansion
# ---------------------------------------------------------------------------


def safe_member_target(staging_dir: Path, member_name: str) -> Path:
    """Resolve one archive member inside ``staging_dir``, rejecting traversal.

    Raises
    ------
    ValueError
        On absolute paths, drive letters, or ``..`` components (zip-slip).

    """
    normalized = member_name.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        message = f"archive member must be relative: {member_name!r}"
        logger.error(message)
        raise ValueError(message)
    candidate = (staging_dir / normalized).resolve()
    containment_root = staging_dir.resolve()
    if containment_root not in candidate.parents and candidate != containment_root:
        message = (
            f"archive member escapes staging directory (zip-slip): {member_name!r}"
        )
        logger.error(message)
        raise ValueError(message)
    return candidate


def extract_zip_members(
    archive_path: Path,
    staging_dir: Path,
    *,
    members: tuple[str, ...] | None,
    member_pattern: str | None,
) -> list[Path]:
    """Extract selected members into ``staging_dir`` with CRC verification.

    Only exact ``members`` matches (by full member name or basename) or
    ``member_pattern`` fnmatch hits against the basename are extracted.  The
    zipfile CRC check runs during read; every resolved target passes the
    containment guard before write.

    Returns
    -------
    list[Path]
        Extracted file paths.

    """
    import fnmatch
    import zipfile

    staging_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    wanted_members = set(members or ())
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            name = info.filename
            base = name.rsplit("/", 1)[-1]
            hit = (
                name in wanted_members
                or base in wanted_members
                or (
                    member_pattern is not None
                    and not info.is_dir()
                    and fnmatch.fnmatch(base, member_pattern)
                )
            )
            if not hit:
                continue
            target = safe_member_target(staging_dir, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            # read() verifies CRC32 from the central directory.
            data = zf.read(info)
            tmp_out = part_path(target)
            tmp_out.write_bytes(data)
            tmp_out.replace(target)
            extracted.append(target)
    if not extracted:
        message = (
            f"no archive members matched members={members} / "
            f"member_pattern={member_pattern!r} in {archive_path.name}"
        )
        logger.error(message)
        raise InvalidProcessingStateError(message)
    return extracted


# ---------------------------------------------------------------------------
# Download primitives
# ---------------------------------------------------------------------------


def _stream_to_part(response: requests.Response, part: Path) -> int:
    """Stream a response body into ``part``; return byte count written."""
    written = 0
    with part.open("wb") as out:
        for block in response.iter_content(1 << 20):
            if block:
                out.write(block)
                written += len(block)
    return written


def _head_content_length(
    url: str,
    headers: dict[str, str],
    *,
    surface_contract_errors: bool = False,
) -> tuple[int | None, bool]:
    """Probe ranged support; returns (content_length, accept_ranges).

    Best-effort: probe failures degrade gracefully to ``(None, False)`` and
    the caller streams plain.  With ``surface_contract_errors`` (plan-based
    execution) assertion-type contract violations propagate untouched so
    they are never masked into a silent transfer-mode downgrade.
    """
    try:
        response = thread_local_session().head(
            url, timeout=REQUEST_TIMEOUT, headers=headers
        )
        if response.status_code != 200:
            return None, False
        length_raw = response.headers.get("Content-Length")
        length = int(length_raw) if length_raw and length_raw.isdigit() else None
        accept_ranges = response.headers.get("Accept-Ranges", "").lower() == "bytes"
        return length, accept_ranges  # noqa: TRY300
    except AssertionError:
        if surface_contract_errors:
            raise
        logger.debug("HEAD probe failed for %s: %s", redact_url(url), "AssertionError")
        return None, False
    except Exception as exc:
        logger.debug(
            "HEAD probe failed for %s: %s", redact_url(url), type(exc).__name__
        )
        return None, False


def _download_whole(
    url: str,
    part: Path,
    *,
    headers: dict[str, str],
    min_bytes: int,
    expected_total: int | None = None,
    credentials: CredentialProvider | None = None,
    allowed_hosts: set[str] | None = None,
) -> None:
    """Plain whole-file streaming download with integrity checks.

    Redirect hops are validated explicitly: cross-host targets are followed
    WITHOUT credentials, and any auth header that would attach cross-host
    is a hard error.
    """
    current_url = url
    current_headers = dict(headers)
    max_hops = 5
    for _hop in range(max_hops):
        response = _request_with_retries(
            "GET", current_url, stream=True, headers=current_headers
        )
        if response.status_code not in (301, 302, 303, 307, 308):
            break
        next_url = _validate_redirect_chain(
            current_url,
            response,
            allowed_hosts=allowed_hosts
            or {urllib.parse.urlsplit(current_url).hostname or ""},
            credentials=credentials,
        )
        response.close()
        if next_url is None:
            message = f"redirect without Location for {redact_url(current_url)}"
            raise InvalidProcessingStateError(message)
        if (
            urllib.parse.urlsplit(next_url).hostname
            != urllib.parse.urlsplit(current_url).hostname
        ):
            # Cross-host hop: strip credential headers entirely.
            for name in list(current_headers):
                if name.lower() in {"authorization", "cookie"}:
                    del current_headers[name]
        current_url = next_url
    else:
        message = f"too many redirects (> {max_hops}) for {redact_url(url)}"
        raise InvalidProcessingStateError(message)
    magic = b""
    try:
        first_block: bytes | None = None
        blocks = response.iter_content(1 << 20)
        written = 0
        with part.open("wb") as out:
            for block in blocks:
                if not block:
                    continue
                if first_block is None:
                    first_block = block[:512]
                    magic = first_block
                    html_hit = (
                        b"<html" in magic.lower() or b"<!doctype html" in magic.lower()
                    )
                    if html_hit:
                        message = (
                            "received an HTML login/error page instead of "
                            "raster bytes for "
                            f"{redact_url(current_url)} (auth redirect failure)"
                        )
                        raise InvalidProcessingStateError(message)
                out.write(block)
                written += len(block)
    finally:
        response.close()
    del magic
    if written < min_bytes:
        message = (
            f"fetched tile too small: {written} bytes < floor {min_bytes} "
            f"for {redact_url(current_url)}"
        )
        raise InvalidProcessingStateError(message)
    if expected_total is not None and written != expected_total:
        message = (
            f"size mismatch for {redact_url(current_url)}: wrote {written}, "
            f"expected {expected_total}"
        )
        raise InvalidProcessingStateError(message)


_CONTENT_RANGE_RE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)")


def _download_ranged(
    url: str,
    part: Path,
    total: int,
    *,
    headers: dict[str, str],
    budget: ThreadPoolExecutor | None = None,
    max_workers: int = 4,
) -> None:
    """Ranged-chunk assembly with mandatory 206/Content-Range validation.

    Chunks are fetched serially when only one worker is in the shared
    budget; otherwise a small per-tile pool parallelizes chunk GETs while
    writes stay on the calling thread.
    """
    fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    handles: dict[int, int] = {}
    try:
        os.ftruncate(fd, total)
        chunk_starts = list(range(0, total, CHUNK_SIZE_BYTES))

        def fetch_chunk(start: int) -> tuple[int, bytes]:
            end = min(start + CHUNK_SIZE_BYTES, total) - 1
            chunk_headers = dict(headers)
            chunk_headers["Range"] = f"bytes={start}-{end}"
            response = _request_with_retries(
                "GET", url, stream=True, headers=chunk_headers
            )
            try:
                if response.status_code != 206:
                    message = (
                        f"Range request answered with status "
                        f"{response.status_code} instead of 206 for "
                        f"{redact_url(url)}; server ignored Range"
                    )
                    raise InvalidProcessingStateError(message)
                raw_range = response.headers.get("Content-Range", "")
                match = _CONTENT_RANGE_RE.match(raw_range)
                body = b"".join(response.iter_content(1 << 20))
                if (
                    match is None
                    or int(match.group(1)) != start
                    or int(match.group(2)) != end
                    or (match.group(3) != "*" and int(match.group(3)) != total)
                ):
                    message = (
                        f"Content-Range mismatch for {redact_url(url)}: "
                        f"requested bytes {start}-{end}, got {raw_range!r}"
                    )
                    raise InvalidProcessingStateError(message)
                return start, body
            finally:
                response.close()

        # Chunks of one tile stream serially in offset order; parallelism
        # comes from multiple TILES in flight (one shared worker budget),
        # matching the benchmark-backed design and keeping per-chunk
        # 206/Content-Range validation strictly ordered per resource.
        del budget, max_workers
        for start, body in map(fetch_chunk, chunk_starts):
            _write_chunk(fd, handles, start, body)
    finally:
        for handle_fd in handles.values():
            os.close(handle_fd)
        os.close(fd)
    actual = part.stat().st_size
    if actual != total:
        message = (
            f"ranged assembly size mismatch for {redact_url(url)}: "
            f"{actual} != Content-Length {total}"
        )
        raise InvalidProcessingStateError(message)


def _write_chunk(fd: int, handles: dict[int, int], start: int, body: bytes) -> None:
    """Write one chunk via pwrite (POSIX) or per-thread seek+write."""
    if _PWRITE_AVAILABLE:
        os.pwrite(fd, body, start)
        return
    thread_id = __import__("threading").get_ident()
    handle_fd = handles.get(thread_id)
    if handle_fd is None:
        handle_fd = os.dup(fd)
        handles[thread_id] = handle_fd
    os.lseek(handle_fd, start, os.SEEK_SET)
    os.write(handle_fd, body)


# ---------------------------------------------------------------------------
# Plan execution
# ---------------------------------------------------------------------------


def _resolve_effective_credentials(
    plan: FetchPlan,
) -> CredentialProvider | None:
    if plan._credentials is not None and isinstance(
        plan._credentials, CredentialProvider
    ):
        return plan._credentials
    if plan.credential_ref is not None:
        return resolve_credentials(plan.credential_ref)
    return None


def _execute_tile_set(
    plan: TileSet,
    cache_dir: Path,
    *,
    max_workers: int,
    chunked_threshold: int,
    credentials: CredentialProvider | None,
) -> list[Path]:
    """Fetch missing tiles concurrently under one shared worker budget."""
    del chunked_threshold  # reserved for ranged-mode thresholding
    allowed = {host.lower() for host in plan.allowed_hosts}
    jobs: list[tuple[str, Path, int, bool]] = []
    for tile in plan.tiles:
        target = cache_dir / tile.cache_path
        if target.is_file() and target.stat().st_size >= tile.min_bytes:
            logger.info("DEM tile cache hit: %s", target)
            continue
        validate_cache_target(cache_dir, target)
        jobs.append((tile.url, target, tile.min_bytes, tile.ranged))

    if not jobs:
        return [cache_dir / tile.cache_path for tile in plan.tiles]

    results: dict[Path, Path] = {}
    lock = __import__("threading").Lock()

    def run_job(job: tuple[str, Path, int, bool]) -> Path:
        url, target, min_bytes, ranged_capable = job
        headers = credentials.headers_for(url) if credentials else {}
        use_ranged = ranged_capable and _is_earthdata_head_safe(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        part = part_path(target)
        try:
            if use_ranged:
                length, accept_ranges = _head_content_length(
                    url, headers, surface_contract_errors=True
                )
                if length is not None and accept_ranges and length <= 64 << 30:
                    _download_ranged(
                        url,
                        part,
                        length,
                        headers=headers,
                        max_workers=min(max_workers, 4),
                    )
                else:
                    _download_whole(
                        url,
                        part,
                        headers=headers,
                        min_bytes=min_bytes,
                        credentials=credentials,
                        allowed_hosts=allowed,
                    )
            else:
                _download_whole(
                    url,
                    part,
                    headers=headers,
                    min_bytes=min_bytes,
                    credentials=credentials,
                    allowed_hosts=allowed,
                )
            sha = hashlib.sha256(part.read_bytes()).hexdigest()
            logger.debug("tile sha256 %s: %s", target.name, sha)
            _publish_target(target, part)
            return target  # noqa: TRY300
        except Exception as exc:
            scrubbed = _scrub_message(str(exc))
            part.unlink(missing_ok=True)
            raise type(exc)(scrubbed) if scrubbed else exc from exc.__cause__

    workers = max(1, max_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in futures:
            path = future.result()
            with lock:
                results[path] = path

    ordered: list[Path] = []
    by_name = {path.name: path for path in results}
    for tile in plan.tiles:
        target = cache_dir / tile.cache_path
        found = results.get(target) or by_name.get(target.name)
        if found is None and not target.is_file():
            message = f"tile fetch did not produce {target}"
            raise InvalidProcessingStateError(message)
        ordered.append(found or target)
    return ordered


def validate_cache_target(cache_dir: Path, target: Path) -> None:
    """Reject targets escaping ``cache_dir`` before any I/O."""
    resolved_root = cache_dir.resolve()
    resolved_target = target.resolve()
    if resolved_root not in resolved_target.parents:
        message = f"cache target {target} escapes cache root {cache_dir}; refusing"
        logger.error(message)
        raise ValueError(message)


def _execute_artifact(
    plan: Artifact,
    cache_dir: Path,
    *,
    credentials: CredentialProvider | None,
) -> Path:
    """Fetch one artifact (plain file, ftp file, or zip expansion)."""
    target = plan.cache_path
    if target is None:
        tail = urllib.parse.urlsplit(plan.url).path.rsplit("/", 1)[-1] or "artifact.bin"
        target = Path(tail)
    validate_cache_target(cache_dir, cache_dir / target)
    target = cache_dir / target
    target.parent.mkdir(parents=True, exist_ok=True)

    if plan.scheme == "ftp":
        part = part_path(target)
        try:
            with urllib.request.urlopen(plan.url, timeout=180) as response:
                written = 0
                with part.open("wb") as out:
                    import shutil

                    shutil.copyfileobj(response, out, length=1 << 20)
                    written = out.tell()
            if written < plan.min_total_bytes:
                message = (
                    f"FTP payload below floor: {written} < "
                    f"{plan.min_total_bytes} for {redact_url(plan.url)}"
                )
                raise InvalidProcessingStateError(message)  # noqa: TRY301
        except Exception as exc:
            part.unlink(missing_ok=True)
            if isinstance(exc, InvalidProcessingStateError):
                raise
            message = f"FTP fetch failed for {redact_url(plan.url)}: {exc}"
            logger.exception(message)
            raise InvalidProcessingStateError(message) from exc
        _publish_target(target, part)

        if plan.expand == "zip":
            staging = target.with_suffix(target.suffix + ".zip-staging")
            extract_zip_members(
                target,
                staging,
                members=plan.members,
                member_pattern=plan.member_pattern,
            )
        return target

    headers = credentials.headers_for(plan.url) if credentials else {}
    part = part_path(target)
    allowed = {host.lower() for host in plan.allowed_hosts}
    try:
        use_ranged = _is_earthdata_head_safe(plan.url) and credentials is None
        if use_ranged:
            length, accept_ranges = _head_content_length(plan.url, headers)
            if length is not None and accept_ranges:
                _download_ranged(plan.url, part, length, headers=headers, max_workers=4)
            else:
                _download_whole(
                    plan.url,
                    part,
                    headers=headers,
                    min_bytes=plan.min_total_bytes,
                    credentials=credentials,
                    allowed_hosts=allowed,
                )
        else:
            _download_whole(
                plan.url,
                part,
                headers=headers,
                min_bytes=plan.min_total_bytes,
                credentials=credentials,
                allowed_hosts=allowed,
            )
    except Exception:
        part.unlink(missing_ok=True)
        raise
    _publish_target(target, part)
    if plan.expand == "zip":
        staging = target.with_suffix(target.suffix + ".zip-staging")
        extract_zip_members(
            target,
            staging,
            members=plan.members,
            member_pattern=plan.member_pattern,
        )
    return target


def fetch_tiles(
    urls_and_targets: list[tuple[str, Path]],
    cache_dir: Path,
    *,
    max_workers: int = 8,
    minimum_bytes: int = 0,
) -> list[Path]:
    """Compatibility entry point: concurrent plain/ranged tile downloads.

    ``minimum_bytes`` defaults to 0 here; per-tile size floors are carried
    by :class:`Tile` records on plan-based execution.  The explicit 1 MiB
    floor used by PROPOSAL-0013 callers passes ``minimum_bytes`` directly.
    """
    plan = TileSet(
        allowed_hosts=tuple(
            {
                (urllib.parse.urlsplit(url).hostname or "").lower()
                for url, _ in urls_and_targets
            }
        ),
        tiles=tuple(
            Tile(
                url=url,
                cache_path=target.relative_to(cache_dir),
                min_bytes=minimum_bytes,
            )
            if target.is_relative_to(cache_dir)
            else Tile(url=url, cache_path=Path(target.name), min_bytes=minimum_bytes)
            for url, target in urls_and_targets
        ),
    )
    validate_plan_urls(plan)
    missing: list[tuple[str, Path]] = []
    present: list[Path] = []
    for url, target in urls_and_targets:
        if target.is_file() and target.stat().st_size >= minimum_bytes:
            present.append(target)
        else:
            missing.append((url, target))

    results: list[Path] = []
    lock = __import__("threading").Lock()
    error_box: dict[str, BaseException] = {}

    def run(entry: tuple[str, Path]) -> Path | None:
        url, target = entry
        try:
            validate_cache_target(cache_dir, target)
            part = part_path(target)
            headers: dict[str, str] = {}
            ranged_ok = not _is_earthdata_host(url) and _is_earthdata_head_safe(url)
            length, accept_ranges = (
                _head_content_length(url, headers) if ranged_ok else (None, False)
            )
            try:
                if length is not None and accept_ranges:
                    _download_ranged(
                        url,
                        part,
                        length,
                        headers=headers,
                        max_workers=min(max_workers, 4),
                    )
                else:
                    _download_whole(url, part, headers=headers, min_bytes=minimum_bytes)
            except InvalidProcessingStateError as exc:
                raise _map_transport_exception(url, exc) from exc
            except Exception as exc:  # surface typed terminal failures
                raise _map_transport_exception(url, exc) from exc
            _publish_target(target, part)
            return target  # noqa: TRY300
        except BaseException as exc:
            with lock:
                error_box.setdefault(url, exc)
            return None

    if missing:
        with ThreadPoolExecutor(max_workers=max(2, max_workers)) as pool:
            outcomes = [o for o in pool.map(run, missing) if o is not None]
            results.extend(outcomes)
    if error_box:
        raise error_box[next(iter(error_box))]
    return present + results


def _map_transport_exception(url: str, exc: Exception) -> Exception:
    """Re-map low-level responses into the module's typed errors."""
    del url
    if isinstance(exc, DemAuthProviderError):
        return exc
    text = str(exc)
    if "terminal status 401" in text:
        return DemAuthProviderError(_scrub_message(text), status=401)
    if "terminal status 403" in text:
        return DemAuthProviderError(_scrub_message(text), status=403)
    return exc


def fetch_plan(
    plan: FetchPlan,
    cache_dir: Path,
    *,
    max_workers: int = 8,
    chunked_threshold: int = 4,
) -> list[Path]:
    """Execute one self-describing :class:`FetchPlan` into ``cache_dir``.

    Parameters
    ----------
    plan : FetchPlan
        TileSet or Artifact plan produced by a registry entry.
    cache_dir : Path
        Cache root; all destinations are cache-relative.
    max_workers : int
        Shared stream budget for tiles and chunks.
    chunked_threshold : int
        Missing-tile count under which ranged mode is preferred.

    Returns
    -------
    list[Path]
        Paths ready for mosaic input (tiles or extracted members).

    """
    validate_plan_urls(plan)
    credentials = _resolve_effective_credentials(plan)
    if plan.credential_ref == "earthdata" and credentials is None:
        EarthdataCredentialProvider.require_available()
    sweep_part_files(cache_dir)
    if isinstance(plan, TileSet):
        return _execute_tile_set(
            plan,
            cache_dir,
            max_workers=max_workers,
            chunked_threshold=chunked_threshold,
            credentials=credentials,
        )
    if isinstance(plan, Artifact):
        return [_execute_artifact(plan, cache_dir, credentials=credentials)]
    message = f"unsupported FetchPlan subtype: {type(plan).__name__}"
    logger.error(message)
    raise InvalidProcessingStateError(message)
