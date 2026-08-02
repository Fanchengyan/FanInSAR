"""URI-addressed product store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass
class LocalStore:
    """Simple filesystem store implementing the Store protocol."""

    uri: str

    def exists(self, key: str = "") -> bool:
        """Return True if the store path (or key under it) exists."""
        root = Path(self.uri)
        if key:
            return (root / key).exists()
        return root.exists()


def open_store(uri: str, **kwargs: Any) -> LocalStore:
    """Open a store for *uri* (local path for day-1)."""
    del kwargs
    parsed = urlparse(uri)
    if parsed.scheme in {"", "file"}:
        path = parsed.path if parsed.scheme == "file" else uri
        return LocalStore(uri=str(Path(path)))
    # fsspec remote: still return a LocalStore-like handle with the URI string
    return LocalStore(uri=uri)


__all__ = ["LocalStore", "open_store"]
