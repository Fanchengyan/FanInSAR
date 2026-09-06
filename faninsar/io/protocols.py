"""IO backend port — URI store and product read/write boundary."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Store(Protocol):
    """URI-addressed product store (local path or remote fsspec URI)."""

    uri: str

    def exists(self, key: str = "") -> bool:
        """Return True if the store (or *key* inside it) exists."""
        ...


@runtime_checkable
class Catalog(Protocol):
    """STAC catalog surface at product boundaries only."""

    def add_item(self, item: Any) -> None:
        """Register a STAC Item at a product boundary."""
        ...


@runtime_checkable
class Format(Protocol):
    """Serialization format (Zarr, COG, GeoTIFF, …)."""

    name: str


@runtime_checkable
class IOBackend(Protocol):
    """Read/write products through URI-addressed stores."""

    def open_store(self, uri: str, **kwargs: Any) -> Store:
        """Open or create a store at *uri*."""
        ...

    def write(
        self,
        product: Any,
        uri: str,
        *,
        format: str = "cog",  # noqa: A002
    ) -> None:
        """Write *product* to *uri* using *format*."""
        ...

    def read(self, uri: str, **kwargs: Any) -> Any:
        """Read a product from *uri*."""
        ...


__all__ = ["Catalog", "Format", "IOBackend", "Store"]
