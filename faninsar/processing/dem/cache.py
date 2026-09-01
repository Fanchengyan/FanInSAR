# ruff: noqa: TRY003, EM102
"""Small, deterministic cache helpers for geoid resources.

This module intentionally has no locking or atomic-write promise.  A cache
entry is usable only after its complete content identity has been validated.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from faninsar.logging import setup_logger

logger = setup_logger(__name__)

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class CachePathError(ValueError):
    """Raised when a cache path escapes its configured root."""


class ArtifactValidationError(ValueError):
    """Raised when a cached or downloaded artifact has the wrong identity."""


def resolve_cache_root(cache_root: str | Path | None = None) -> Path:
    """Resolve and validate the geoid cache root.

    Parameters
    ----------
    cache_root : str or pathlib.Path, optional
        Explicit root.  When omitted, ``FANINSAR_GEOID_CACHE`` is honoured,
        followed by ``~/.cache/faninsar/geoid``.

    Returns
    -------
    pathlib.Path
        Absolute, normalized cache root (which need not exist yet).

    """
    configured = cache_root
    if configured is None:
        configured = os.environ.get("FANINSAR_GEOID_CACHE")
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "faninsar" / "geoid"
    )
    return root.resolve(strict=False)


def cache_artifact_path(root: Path, model: str, filename: str) -> Path:
    """Return a model cache path after a narrow containment check."""
    if not _SAFE_COMPONENT.fullmatch(model) or not _SAFE_COMPONENT.fullmatch(filename):
        message = "geoid cache model and filename must be simple path components"
        logger.warning(message)
        raise CachePathError(message)
    model_root = (root / model).resolve(strict=False)
    candidate = (model_root / filename).resolve(strict=False)
    try:
        candidate.relative_to(root.resolve(strict=False))
    except ValueError as error:
        message = f"geoid cache path escapes configured root: {candidate}"
        logger.warning(message)
        raise CachePathError(message) from error
    return candidate


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a file's SHA-256 digest using bounded reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact(path: Path, *, expected_size: int, expected_sha256: str) -> Path:
    """Validate exact size and SHA-256 before an artifact is loaded."""
    if not path.is_file():
        raise ArtifactValidationError(f"geoid artifact is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ArtifactValidationError(
            f"geoid artifact size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_digest = sha256_file(path)
    if actual_digest.lower() != expected_sha256.lower():
        raise ArtifactValidationError(
            "geoid artifact SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_digest}"
        )
    return path
