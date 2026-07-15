"""Checksum-enforced access to frozen scientific reference artifacts."""

from __future__ import annotations

import hashlib
import importlib.util
import socket
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

_LIMITS: Final = httpx.Limits(
    max_connections=200,
    max_keepalive_connections=40,
    keepalive_expiry=30.0,
)
_TIMEOUT: Final = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=10.0)
_SOCKET_OPTIONS: Final = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
_HTTP2_ENABLED: Final = importlib.util.find_spec("h2") is not None


@dataclass(frozen=True, slots=True)
class Artifact:
    """Immutable location and integrity metadata for one reference artifact."""

    identifier: str
    url: str
    filename: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ChecksumMismatchError(RuntimeError):
    """Raised when artifact bytes do not match the frozen manifest."""

    artifact: Artifact
    actual_sha256: str
    actual_size_bytes: int

    def __str__(self) -> str:
        """Return the artifact-specific checksum failure."""
        return (
            f"checksum mismatch for {self.artifact.filename}: expected "
            f"{self.artifact.sha256}, got {self.actual_sha256}"
        )


@dataclass(frozen=True, slots=True)
class OfflineCacheMissError(RuntimeError):
    """Raised when offline mode cannot resolve a verified cached artifact."""

    artifact: Artifact

    def __str__(self) -> str:
        """Return the artifact-specific offline failure."""
        return f"offline cache miss for {self.artifact.identifier}"


@dataclass(frozen=True, slots=True)
class DownloadError(RuntimeError):
    """Raised when a transfer reports success without producing bytes."""

    artifact: Artifact

    def __str__(self) -> str:
        """Return the artifact-specific transfer failure."""
        return f"download did not create {self.artifact.filename}"


Fetcher = Callable[[str, Path], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, artifact: Artifact) -> None:
    actual_size = path.stat().st_size
    actual_sha256 = _sha256(path)
    if actual_size != artifact.size_bytes or actual_sha256 != artifact.sha256:
        logger.error(
            "Reference artifact checksum mismatch: %s expected=%s actual=%s",
            artifact.identifier,
            artifact.sha256,
            actual_sha256,
        )
        raise ChecksumMismatchError(artifact, actual_sha256, actual_size)


def _fetch_http(url: str, destination: Path) -> None:
    transport = httpx.HTTPTransport(
        http2=_HTTP2_ENABLED,
        retries=3,
        limits=_LIMITS,
        socket_options=_SOCKET_OPTIONS,
    )
    with (
        httpx.Client(
            transport=transport,
            timeout=_TIMEOUT,
            follow_redirects=True,
        ) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        with destination.open("wb") as stream:
            for chunk in response.iter_bytes():
                stream.write(chunk)


def resolve_artifact(
    artifact: Artifact,
    cache_directory: Path,
    *,
    offline: bool = False,
    fetch: Fetcher = _fetch_http,
) -> Path:
    """Return a verified cache path, downloading atomically when permitted.

    Parameters
    ----------
    artifact
        Frozen artifact metadata.
    cache_directory
        Directory containing or receiving cached bytes.
    offline
        Refuse all network access when ``True``.
    fetch
        Transfer implementation used for cache misses.

    Returns
    -------
    pathlib.Path
        Path to verified artifact bytes.

    Raises
    ------
    ChecksumMismatchError
        If cached or downloaded bytes differ from the manifest.
    OfflineCacheMissError
        If no cached artifact exists in offline mode.

    """
    cache_path = cache_directory / artifact.filename
    if cache_path.exists():
        _verify(cache_path, artifact)
        return cache_path
    if offline:
        logger.error("Reference artifact unavailable offline: %s", artifact.identifier)
        raise OfflineCacheMissError(artifact)

    cache_directory.mkdir(parents=True, exist_ok=True)
    partial_path = cache_directory / f"{artifact.filename}.part"
    partial_path.unlink(missing_ok=True)
    try:
        fetch(artifact.url, partial_path)
    except (OSError, httpx.HTTPError):
        partial_path.unlink(missing_ok=True)
        raise
    if not partial_path.exists():
        logger.error(
            "Reference artifact download produced no file: %s",
            artifact.identifier,
        )
        raise DownloadError(artifact)
    try:
        _verify(partial_path, artifact)
    except ChecksumMismatchError:
        partial_path.unlink(missing_ok=True)
        raise
    partial_path.replace(cache_path)
    return cache_path
