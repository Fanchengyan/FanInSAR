# ruff: noqa: E501, EM101, EM102, TRY003, TID252, PLW2901, RUF005, D105
"""Provider registry and Planetary Computer Copernicus DEM adapters."""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from faninsar.logging import setup_logger

from ..resources import ResourceBudget, preflight_grid
from ..seam import (
    SeamAwareSourceSampler,
    canonical_item_ids,
    plan_query_windows,
)
from ..transport import (
    resolve_cache_path,
    stream_response_to_cache,
    validate_https_origin,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = setup_logger(__name__)

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_STAC_HOST = "planetarycomputer.microsoft.com"
PC_ASSET_HOST = "elevationeuwest.blob.core.windows.net"


class ProviderUnavailableError(RuntimeError):
    """Raised when a registered provider cannot serve a request."""


@dataclass(frozen=True, slots=True)
class SourceResource:
    """One signed source asset and its stable cache/provenance identity."""

    href: str
    cache_path: Path
    collection_id: str
    asset_key: str
    item_id: str
    shape: tuple[int, int]
    crs: str = "EPSG:4326"
    expected_size: int | None = None

    @property
    def identity(self) -> str:
        """Return identity independent of ephemeral SAS query parameters."""
        unsigned = urllib.parse.urlunsplit(urllib.parse.urlsplit(self.href)[:3] + ("", ""))
        return f"{self.collection_id}/{self.item_id}/{self.asset_key}:{unsigned}"


@dataclass(frozen=True, slots=True)
class PcStacSource:
    """Planetary Computer Copernicus DEM source.

    Construction is offline.  ``discover`` is the explicit materialization
    boundary where STAC search, signing, and remote asset enumeration begin.
    """

    product: str
    provider: str = "pc"
    stac_url: str = PC_STAC_URL
    collection_id: str = ""
    asset_key: str = "data"
    tile_shape: tuple[int, int] = (0, 0)
    resolution_m: float = 0.0
    vertical_datum: str = "egm2008"

    def __post_init__(self) -> None:
        if self.provider != "pc":
            raise ValueError("PcStacSource provider must be 'pc'")
        if self.vertical_datum != "egm2008":
            raise ValueError("Copernicus GLO PC source datum is registry-owned")
        specs = {"glo30": ("cop-dem-glo-30", (3600, 3600), 30.0), "glo90": ("cop-dem-glo-90", (1200, 1200), 90.0)}
        try:
            collection, shape, resolution = specs[self.product]
        except KeyError as error:
            raise ValueError(f"unsupported Planetary Computer DEM product: {self.product!r}") from error
        if self.collection_id and self.collection_id != collection:
            raise ValueError("Planetary Computer collection is registry-owned")
        object.__setattr__(self, "collection_id", collection)
        object.__setattr__(self, "tile_shape", shape)
        object.__setattr__(self, "resolution_m", resolution)
        validate_https_origin(self.stac_url, {PC_STAC_HOST})

    def discover(
        self,
        bounds: tuple[float, float, float, float],
        *,
        client: object | None = None,
        signer: Callable[[object], object] | None = None,
    ) -> tuple[SourceResource, ...]:
        """Search and sign assets for automatic source windows."""
        windows = plan_query_windows(bounds)
        if client is None:
            try:
                import planetary_computer
                import pystac_client
            except ImportError as error:
                raise ProviderUnavailableError(
                    "provider 'pc' requires planetary-computer and pystac-client"
                ) from error
            signer = signer or planetary_computer.sign_inplace
            client = pystac_client.Client.open(self.stac_url)
        resources: list[SourceResource] = []
        for window in windows:
            resources.extend(self._discover_window(window, client, signer))
        unique: dict[str, SourceResource] = {}
        for resource in sorted(resources, key=lambda value: value.identity):
            unique.setdefault(resource.identity, resource)
        if not unique:
            raise ProviderUnavailableError(
                f"no {self.collection_id}/{self.asset_key} coverage for {bounds!r}"
            )
        return tuple(unique.values())

    def plan(
        self,
        bounds: tuple[float, float, float, float],
        *,
        client: object | None = None,
        signer: Callable[[object], object] | None = None,
    ) -> tuple[SourceResource, ...]:
        """Alias for discovery retained at the provider adapter boundary."""
        return self.discover(bounds, client=client, signer=signer)

    def _discover_window(
        self,
        bounds: tuple[float, float, float, float],
        client: object,
        signer: Callable[[object], object] | None,
    ) -> list[SourceResource]:
        search = getattr(client, "search", None)
        if search is None:
            collection = getattr(client, "get_collection", lambda _name: client)(self.collection_id)
            search = collection.search
        result = search(collections=[self.collection_id], bbox=list(bounds))
        items = result.items() if hasattr(result, "items") else result
        output: list[SourceResource] = []
        for item in canonical_item_ids(tuple(items)):
            if signer is not None:
                signed_item = signer(item)
                if signed_item is not None:
                    item = signed_item
            assets = getattr(item, "assets", {})
            asset = assets.get(self.asset_key)
            if asset is None:
                continue
            href = str(getattr(asset, "href", ""))
            self._validate_asset_href(href)
            unsigned_name = urllib.parse.urlsplit(href).path.rsplit("/", 1)[-1]
            cache_relative = f"{self.collection_id}/{unsigned_name}"
            # Validate the registry-relative path even though it is retained
            # as a relative value in the provenance record.
            resolve_cache_path(".", cache_relative)
            item_id = str(getattr(item, "id", unsigned_name))
            extra = getattr(asset, "extra_fields", {}) or {}
            size = extra.get("file:size")
            output.append(
                SourceResource(
                    href=href,
                    cache_path=Path(cache_relative),
                    collection_id=self.collection_id,
                    asset_key=self.asset_key,
                    item_id=item_id,
                    shape=self.tile_shape,
                    expected_size=int(size) if size is not None else None,
                )
            )
        return output

    @staticmethod
    def _validate_asset_href(href: str) -> None:
        parts = urllib.parse.urlsplit(href)
        host = (parts.hostname or "").lower()
        if parts.scheme != "https" or host != PC_ASSET_HOST:
            raise ProviderUnavailableError("Planetary Computer asset origin is not approved")


GLO30_PC = PcStacSource("glo30")
GLO90_PC = PcStacSource("glo90")
PC_REGISTRY: dict[str, PcStacSource] = {"glo30:pc": GLO30_PC, "glo90:pc": GLO90_PC}
_PRODUCTS = frozenset(
    {
        "auto", "glo30", "glo90", "nasadem", "alos-dem", "srtm-skadi",
        "terrain-tiles", "arcticdem-10", "arcticdem-32", "arcticdem-2",
        "rema-10", "rema-32", "rema-2", "nisar-glo30",
    }
)


def parse_selection(selection: str) -> tuple[str, str | None]:
    """Parse canonical ``product[:provider]`` selection before any I/O."""
    value = str(selection).strip().lower()
    product, separator, provider = value.partition(":")
    if product not in _PRODUCTS or (separator and not provider) or value.count(":") > 1:
        raise ValueError(f"unsupported DEM source selection {selection!r}")
    return product, provider or None


def get_provider(selection: str) -> PcStacSource:
    """Return the registered PC adapter for a selection."""
    product, provider = parse_selection(selection)
    key = f"{product}:{provider or 'pc'}"
    try:
        return PC_REGISTRY[key]
    except KeyError as error:
        raise ProviderUnavailableError(f"DEM provider is not wired: {selection!r}") from error


def materialize_source(
    source: object,
    grid: object,
    *,
    cache_dir: Path,
    budget: ResourceBudget | None = None,
    client: object | None = None,
    signer: Callable[[object], object] | None = None,
) -> object:
    """Fetch a registered PC source and warp native COGs directly to ``grid``.

    This adapter is intentionally narrow: it owns provider I/O while the
    public DEM class owns the resulting RasterDEM identity.  No geographic
    staging raster is created.
    """
    import numpy as np
    import rasterio
    from affine import Affine

    if not isinstance(source, PcStacSource):
        raise ProviderUnavailableError(f"DEM provider is not wired: {source!r}")
    height, width = int(grid.height), int(grid.width)
    preflight_grid(height, width, budget=budget)
    # Query in WGS84 regardless of the public target projection.
    from pyproj import Transformer

    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    left, bottom, right, top = grid.bounds
    lon_a, lat_a = transformer.transform(left, bottom)
    lon_b, lat_b = transformer.transform(right, top)
    bounds = (min(lon_a, lon_b), min(lat_a, lat_b), max(lon_a, lon_b), max(lat_a, lat_b))
    resources = source.discover(bounds, client=client, signer=signer)
    destination = np.full((height, width), np.nan, dtype=np.float32)
    target_transform = Affine(*grid.transform)
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float64) + 0.5,
        np.arange(height, dtype=np.float64) + 0.5,
    )
    target_x, target_y = target_transform * (columns, rows)
    from faninsar.processing.dem.api import _sample_biquintic

    for resource in resources:
        local = resolve_cache_path(cache_dir, resource.cache_path)
        if not local.is_file():
            fetch_asset(
                resource,
                cache_dir=cache_dir,
                max_bytes=(budget.max_fetch_bytes if budget else 2**33),
            )
        with rasterio.open(local) as dataset:
            source_array = np.asarray(dataset.read(1), dtype=np.float64)
            source_crs = dataset.crs or resource.crs
            if str(source_crs) != str(grid.crs):
                transformer = Transformer.from_crs(
                    grid.crs, source_crs, always_xy=True
                )
                source_x, source_y = transformer.transform(target_x, target_y)
            else:
                source_x, source_y = target_x, target_y
            if str(source_crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
                # Apply the same target-centred unwrapping used during STAC
                # planning before converting geographic coordinates to source
                # pixel indices. This is the materializer's real seam path,
                # rather than a planning-only helper.
                center_longitude = 0.5 * (bounds[0] + bounds[2])
                seam_sampler = SeamAwareSourceSampler(
                    lambda longitude, _latitude: longitude,
                    center_longitude,
                )
                source_x = np.asarray(
                    seam_sampler.sample(
                        np.asarray(source_x).ravel().tolist(),
                        np.asarray(source_y).ravel().tolist(),
                    ),
                    dtype=np.float64,
                ).reshape(np.asarray(source_x).shape)
            source_transform = dataset.transform
            source_columns, source_rows = (~source_transform) * (source_x, source_y)
            sampled = _sample_biquintic(
                source_array,
                np.asarray(source_rows, dtype=np.float64),
                np.asarray(source_columns, dtype=np.float64),
            )
            fill = np.isnan(destination) & np.isfinite(sampled)
            destination[fill] = np.asarray(sampled, dtype=np.float32)[fill]
    from faninsar.processing.dem.api import RasterDEM

    return RasterDEM(
        array=destination,
        grid=grid,
        vertical_datum="egm2008",
        provenance={
            "provider": "pc",
            "collection": source.collection_id,
            "asset": source.asset_key,
            "resampling": "direct-source-target",
        },
    )


def fetch_asset(
    resource: SourceResource,
    *,
    cache_dir: Path,
    max_bytes: int = 2**33,
    session: object | None = None,
) -> Path:
    """Fetch one signed COG into its contained cache path with byte bounds."""
    import requests

    PcStacSource._validate_asset_href(resource.href)
    client = session or requests.Session()
    response = client.get(resource.href, stream=True, allow_redirects=False, timeout=(10, 120))
    try:
        if response.status_code in {301, 302, 303, 307, 308}:
            raise ProviderUnavailableError("Planetary Computer asset redirects are rejected")
        response.raise_for_status()
        destination = resolve_cache_path(cache_dir, resource.cache_path)
        stream_response_to_cache(
            response,
            destination,
            max_bytes=max_bytes,
            expected_length=resource.expected_size,
        )
        return destination
    finally:
        response.close()


__all__ = [
    "GLO30_PC", "GLO90_PC", "PC_REGISTRY", "PC_STAC_URL", "PcStacSource",
    "ProviderUnavailableError", "SourceResource", "get_provider", "materialize_source",
    "parse_selection",
]
