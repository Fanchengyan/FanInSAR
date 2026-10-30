"""URI product writers (no Path-only public signatures)."""

from __future__ import annotations

from typing import Any

from faninsar.io.store import open_store
from faninsar.logging import reject


def write_pair(
    product: Any,
    uri: str,
    *,
    format: str = "cog",  # noqa: A002
) -> None:
    """Write a pair product to *uri*.

    Parameters
    ----------
    product : Any
        Pair product or mapping of arrays.
    uri : str
        Destination URI (local path or remote).
    format : str, optional
        ``cog``, ``zarr``, or ``geotiff``.

    """
    store = open_store(uri)
    if format == "zarr":
        import zarr

        root = zarr.open_group(store.uri, mode="w")
        if hasattr(product, "items"):
            for key, value in product.items():
                root[key] = value
        else:
            root.attrs["product_type"] = type(product).__name__
        return
    if format in {"cog", "geotiff"}:
        message = (
            "GeoTIFF/COG write requires raster metadata; use domain product writers"
        )
        reject(message, NotImplementedError)
    message = f"unsupported format {format!r}"
    reject(message, ValueError)


__all__ = ["write_pair"]
