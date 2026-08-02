"""Persistent torch inductor compile cache resolution."""

from __future__ import annotations

import os
from pathlib import Path


def default_compile_cache_root() -> Path:
    """Return the default cache root under the user home directory."""
    return Path.home() / ".cache" / "faninsar" / "torchinductor"


def resolve_compile_cache_dir() -> Path:
    """Resolve the compile cache directory.

    Prefers ``FANINSAR_COMPILE_CACHE`` when set; otherwise uses
    ``~/.cache/faninsar/torchinductor/``. Never defaults solely to ephemeral
    ``/tmp``.
    """
    override = os.environ.get("FANINSAR_COMPILE_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    return default_compile_cache_root()


def apply_compile_cache_env() -> Path:
    """Set ``TORCHINDUCTOR_CACHE_DIR`` before any ``torch.compile`` call.

    Returns
    -------
    pathlib.Path
        The resolved cache directory (created if missing).

    """
    cache_dir = resolve_compile_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    return cache_dir


__all__ = [
    "apply_compile_cache_env",
    "default_compile_cache_root",
    "resolve_compile_cache_dir",
]
