# ruff: noqa: TRY003, EM101, EM102, TRY300, TRY400, D107, SIM105
"""Lazy fetching of the pinned geoid model resources."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

from faninsar.logging import setup_logger

from .cache import (
    ArtifactValidationError,
    cache_artifact_path,
    resolve_cache_root,
    validate_artifact,
)

logger = setup_logger(__name__)


class GeoidResourceError(RuntimeError):
    """Base error for unavailable or invalid geoid resources."""


class GeoidOfflineError(GeoidResourceError):
    """Raised when a required resource is absent while offline."""


class GeoidArtifactError(GeoidResourceError):
    """Raised when a resource cannot satisfy its pinned identity."""


class ChunkTransport(Protocol):
    """Transport seam used by :class:`Fetch` and deterministic tests."""

    def __call__(self, url: str) -> Iterable[bytes]: ...


@dataclass(frozen=True, slots=True)
class GeoidResource:
    """Pinned identity and endpoint for one geoid model."""

    name: str
    filename: str
    url: str | None
    expected_size: int | None
    sha256: str | None
    version: str
    vertical_crs: str

    def __post_init__(self) -> None:
        """Validate the resource descriptor before any I/O."""
        if self.expected_size is not None and self.expected_size <= 0:
            raise ValueError("expected_size must be positive")
        if (self.expected_size is None) != (self.sha256 is None):
            raise ValueError("expected_size and sha256 must be supplied together")


EGM2008_2_5 = GeoidResource(
    name="egm2008",
    filename="us_nga_egm08_25.tif",
    url="https://cdn.proj.org/us_nga_egm08_25.tif",
    expected_size=80_585_622,
    sha256="4191d471eefebf24091b56dbc604353cb3b8cf8cc70e448bb9ae56a272bef17a",
    version="egm2008-2_5",
    vertical_crs="EPSG:3855",
)

# EGM96 is a local ISCE2 coefficient resource.  Its identity is discovered
# from the installed file because distributions package the same coefficient
# set at different paths.  Fetch never downloads or substitutes this model.
EGM96 = GeoidResource(
    name="egm96",
    filename="egm96geoid.dat",
    url=None,
    expected_size=3_145_488,
    sha256="7010d240f863d238d126bfde791650173f72e95ed621d13ad9fb9b783f787aa9",
    version="isce2-egm96",
    vertical_crs="EPSG:5773",
)

DEFAULT_RESOURCES: dict[str, GeoidResource] = {
    "egm96": EGM96,
    "egm2008": EGM2008_2_5,
}


def _requests_transport(url: str) -> Iterable[bytes]:
    """Stream a fixed HTTPS resource without following redirects."""
    import requests

    response = requests.get(url, stream=True, timeout=60, allow_redirects=False)
    if 300 <= response.status_code < 400:
        response.close()
        raise GeoidArtifactError("geoid resource redirects are not accepted")
    try:
        response.raise_for_status()
        yield from response.iter_content(chunk_size=1024 * 1024)
    finally:
        response.close()


def _offline() -> bool:
    """Return whether network-backed resource acquisition is disabled."""
    return os.environ.get("PROJ_NETWORK", "").strip().upper() == "OFF"


class Fetch:
    """Resolve geoid models from a validated local cache, lazily fetching misses.

    Parameters
    ----------
    cache_root : str or pathlib.Path, optional
        Cache root.  Defaults to ``FANINSAR_GEOID_CACHE`` or
        ``~/.cache/faninsar/geoid``.
    transport : callable, optional
        Testable ``transport(url) -> iterable[bytes]`` seam.  The default uses
        HTTPS requests only for the pinned EGM2008 resource.
    resources : mapping, optional
        Additional internal descriptors, useful for deterministic fixtures.

    Notes
    -----
    Cache writes are intentionally ordinary writes: no lock or atomic-write
    guarantee is made.  A partial entry is never loaded and is fetched again.

    """

    def __init__(
        self,
        cache_root: str | Path | None = None,
        *,
        transport: ChunkTransport | None = None,
        resources: dict[str, GeoidResource] | None = None,
    ) -> None:
        self.cache_root = resolve_cache_root(cache_root)
        self.transport = transport or _requests_transport
        self.resources = dict(DEFAULT_RESOURCES)
        self.resources[EGM2008_2_5.version] = EGM2008_2_5
        if resources:
            self.resources.update(resources)

    def resource(self, model: str) -> GeoidResource:
        """Return a registered model or raise a typed error."""
        try:
            return self.resources[model]
        except KeyError as error:
            raise GeoidResourceError(f"unsupported geoid model: {model}") from error

    def path(self, model: str) -> Path:
        """Return the contained cache path for a registered model."""
        resource = self.resource(model)
        return cache_artifact_path(self.cache_root, resource.name, resource.filename)

    def fetch(self, model: str) -> Path:
        """Load a validated cached model, fetching a missing model on demand.

        EGM96 is discovered locally and is never downloaded by this method.
        EGM2008 is downloaded only when this method is called and its cache
        entry is absent or invalid.
        """
        resource = self.resource(model)
        target = self.path(model)
        if resource.url is None:
            return self._discover_local(resource)
        try:
            validate_artifact(
                target,
                expected_size=resource.expected_size,
                expected_sha256=resource.sha256 or "",
            )
            return target
        except ArtifactValidationError as cache_error:
            if _offline():
                message = f"required geoid {model} is not valid in offline cache"
                logger.error(message)
                raise GeoidOfflineError(message) from cache_error
        return self._download(resource, target)

    def _discover_local(self, resource: GeoidResource) -> Path:
        """Find EGM96 using explicit, bundled, and ISCE2 development paths."""
        explicit = os.environ.get("FANINSAR_EGM96_FILE")
        candidates = [Path(explicit).expanduser()] if explicit else []
        candidates.extend(
            [
                Path(__file__).resolve().parents[3]
                / "data"
                / "egm96"
                / resource.filename,
                Path(__file__).resolve().parents[3] / "data" / resource.filename,
                Path.home()
                / "Documents"
                / "GitHub"
                / "isce2"
                / "contrib"
                / "demUtils"
                / "correct_geoid_i2_srtm"
                / resource.filename,
            ]
        )
        for candidate in candidates:
            if candidate.is_file():
                try:
                    return validate_artifact(
                        candidate.resolve(),
                        expected_size=resource.expected_size or 0,
                        expected_sha256=resource.sha256 or "",
                    )
                except ArtifactValidationError as error:
                    logger.warning(
                        "ignoring invalid local %s artifact: %s", resource.name, error
                    )
        message = "EGM96 coefficient file not found; set FANINSAR_EGM96_FILE"
        logger.error(message)
        raise GeoidOfflineError(message)

    def _download(self, resource: GeoidResource, target: Path) -> Path:
        """Download, validate, and cache one fixed resource."""
        if resource.url is None:
            raise GeoidResourceError(f"resource {resource.name} has no download URL")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("wb") as stream:
                for chunk in self.transport(resource.url):
                    if not isinstance(chunk, bytes):
                        raise GeoidArtifactError(
                            "geoid transport returned a non-bytes chunk"
                        )
                    stream.write(chunk)
            validate_artifact(
                target,
                expected_size=resource.expected_size or 0,
                expected_sha256=resource.sha256 or "",
            )
        except (OSError, ArtifactValidationError, GeoidResourceError) as error:
            try:
                target.unlink()
            except OSError:
                pass
            message = f"failed to acquire geoid resource {resource.name}"
            logger.error("%s: %s", message, error)
            if isinstance(error, GeoidResourceError):
                raise
            raise GeoidArtifactError(message) from error
        return target


__all__ = [
    "DEFAULT_RESOURCES",
    "EGM96",
    "EGM2008_2_5",
    "Fetch",
    "GeoidArtifactError",
    "GeoidOfflineError",
    "GeoidResource",
    "GeoidResourceError",
]
