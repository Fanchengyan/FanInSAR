"""Remote (HTTP / COG range) view of a frame product (M5).

``RemoteFrame`` is returned by :meth:`Frame.open_remote`. It parses a
remote STAC catalog (via :mod:`pystac`) and exposes ``open()`` methods
that lazily fetch COG windows via GDAL's ``/vsicurl/`` virtual filesystem.
No assets are downloaded eagerly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    import xarray as xr

logger = setup_logger(__name__)


def _vsicurl_url(http_url: str) -> str:
    """Wrap an HTTP URL as a GDAL ``/vsicurl/`` path."""
    return f"/vsicurl/{http_url}"


def _resolve_url(base: str, href: str) -> str:
    """Resolve a possibly-relative href against the catalog base URL."""
    if "://" in href:
        return href
    return urljoin(base + "/" if not base.endswith("/") else base, href)


class RemoteFrame:
    """Lazy HTTP view of a faninsar frame product.

    Returned by :meth:`Frame.open_remote`. Not constructed directly by users.

    """

    def __init__(
        self,
        catalog_url: str,
        *,
        cache_dir: str | Path | None = None,
        anonymous: bool = True,
        **_kwargs: Any,
    ) -> None:
        """Initialise the remote frame view."""
        try:
            import pystac  # noqa: F401
        except ImportError as e:
            msg = (
                "pystac is required for Frame.open_remote(). "
                "Install it with: pip install pystac (or the 'cloud' extra)."
            )
            raise ImportError(msg) from e

        self._catalog_url = catalog_url
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._anonymous = anonymous
        self._catalog = self._load_catalog()
        self._geometry_items: list[Any] = []
        self._interferogram_items: list[Any] = []
        self._classify_items()

    def _load_catalog(self) -> Any:
        import pystac

        if self._anonymous:
            # Disable GDAL HTTP auth for public datasets.
            import os

            os.environ.setdefault("GDAL_HTTP_NETRC", "NO")
            os.environ.setdefault("VSI_FTP_USE_CWD", "NO")
            os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")

        catalog = pystac.Catalog.from_file(self._catalog_url)
        logger.info("Loaded remote STAC catalog: %s", catalog.id or self._catalog_url)
        return catalog

    def _classify_items(self) -> None:
        for child in self._catalog.get_children():
            items = list(child.get_items())
            if child.id == "geometry":
                self._geometry_items = items
            elif child.id == "interferograms":
                self._interferogram_items = items
        if not self._geometry_items and not self._interferogram_items:
            for item in self._catalog.get_all_items():
                props = item.properties or {}
                t = props.get("frame:type", "")
                if t == "FrameGeometry":
                    self._geometry_items.append(item)
                elif t == "FrameInterferogramItem":
                    self._interferogram_items.append(item)

    @property
    def catalog_url(self) -> str:
        """The remote catalog URL."""
        return self._catalog_url

    @property
    def catalog(self) -> Any:
        """The loaded pystac Catalog object."""
        return self._catalog

    @property
    def geometry_item(self) -> Any | None:
        """The first geometry STAC Item, or None."""
        return self._geometry_items[0] if self._geometry_items else None

    @property
    def interferogram_items(self) -> list[Any]:
        """List of interferogram STAC Items."""
        return list(self._interferogram_items)

    def _item_asset_urls(self, item: Any) -> dict[str, str]:
        """Return ``{asset_key: resolved_http_url}`` for an item."""
        out: dict[str, str] = {}
        for key, asset in item.get_assets().items():
            out[key] = _resolve_url(self._catalog_url, asset.href)
        return out

    def open_geometry_asset(
        self,
        name: str,
        *,
        chunks: Any = None,
    ) -> xr.DataArray:
        """Lazily open a remote geometry asset over HTTP.

        Parameters
        ----------
        name : str
            Asset key, e.g. ``"incidence"``.
        chunks : dict, int, ``"auto"``, or None
            Dask chunk sizes.

        """
        import rioxarray  # noqa: F401
        import xarray as xr

        item = self.geometry_item
        if item is None:
            msg = "Remote catalog has no geometry item."
            logger.error(msg)
            raise ValueError(msg)
        urls = self._item_asset_urls(item)
        if name not in urls:
            msg = f"Geometry asset {name!r} not found in remote catalog."
            logger.error(msg)
            raise KeyError(msg)
        gdal_path = _vsicurl_url(urls[name])
        return xr.open_dataarray(gdal_path, engine="rasterio", chunks=chunks)

    def open_interferogram_asset(
        self,
        pair_name: str,
        name: str,
        *,
        chunks: Any = None,
    ) -> xr.DataArray:
        """Lazily open a remote interferogram asset over HTTP.

        Parameters
        ----------
        pair_name : str
            Pair name, e.g. ``"20191115_20200314"``.
        name : str
            Asset key, e.g. ``"unw_phase"``.
        chunks : dict, int, ``"auto"``, or None
            Dask chunk sizes.

        """
        import rioxarray  # noqa: F401
        import xarray as xr

        item = next((i for i in self._interferogram_items if i.id == pair_name), None)
        if item is None:
            msg = f"Pair {pair_name!r} not found in remote catalog."
            logger.error(msg)
            raise KeyError(msg)
        urls = self._item_asset_urls(item)
        if name not in urls:
            msg = f"Asset {name!r} not found for pair {pair_name!r}."
            logger.error(msg)
            raise KeyError(msg)
        gdal_path = _vsicurl_url(urls[name])
        return xr.open_dataarray(gdal_path, engine="rasterio", chunks=chunks)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of the remote frame."""
        geom_keys: list[str] = []
        if self.geometry_item is not None:
            geom_keys = list(self.geometry_item.get_assets().keys())
        return {
            "catalog_url": self._catalog_url,
            "geometry_asset_keys": geom_keys,
            "interferogram_pair_count": len(self._interferogram_items),
            "interferogram_pair_names": [i.id for i in self._interferogram_items],
        }

    def __repr__(self) -> str:
        """Return a short summary string."""
        return (
            f"RemoteFrame(url={self._catalog_url!r}, "
            f"pairs={len(self._interferogram_items)})"
        )


__all__ = ["RemoteFrame"]
