"""Public open_stac / open_zarr readers (URI-based)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def open_stac(uri: str, **kwargs: Any) -> Any:
    """Open a STAC Item or Catalog at *uri*.

    Parameters
    ----------
    uri : str
        Local path or remote URI to a STAC Item/Catalog JSON.
    **kwargs
        Forwarded to ``pystac`` readers.

    Returns
    -------
    Any
        A ``pystac.Item`` or ``pystac.Catalog`` instance.

    """
    import pystac

    path = uri
    # pystac accepts paths; remote URIs go through from_file when local-like
    if uri.startswith(("http://", "https://", "s3://")):
        return pystac.read_file(uri, **kwargs)
    return pystac.read_file(str(Path(path)), **kwargs)


def open_zarr(uri: str, **kwargs: Any) -> Any:
    """Open a Zarr group at *uri* (local path or fsspec URI).

    Parameters
    ----------
    uri : str
        Zarr store URI.
    **kwargs
        Forwarded to ``zarr.open_group``.

    Returns
    -------
    Any
        An open Zarr group.

    """
    import zarr

    mode = kwargs.pop("mode", "r")
    return zarr.open_group(uri, mode=mode, **kwargs)


__all__ = ["open_stac", "open_zarr"]
