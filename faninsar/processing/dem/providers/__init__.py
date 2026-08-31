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
    SOURCE_KERNEL_SIZE,
    SeamAwareSourceSampler,
    canonical_item_ids,
    plan_query_windows,
    unwrap_longitude,
)
from ..transport import (
    resolve_cache_path,
    stream_response_to_cache,
    validate_https_origin,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

logger = setup_logger(__name__)

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_STAC_HOST = "planetarycomputer.microsoft.com"
PC_ASSET_HOST = "elevationeuwest.blob.core.windows.net"


class ProviderUnavailableError(RuntimeError):
    """Raised when a registered provider cannot serve a request."""


class SourceConflictError(ProviderUnavailableError):
    """Raised when overlapping source windows disagree at a target pixel."""


@dataclass(frozen=True, slots=True)
class _P0030Source:
    """Private adapter around one accepted P0030 product/provider entry."""

    product: str
    provider: str
    entry_name: str

    @property
    def collection_id(self) -> str:
        """Return the stable P0030 selection identity."""
        return f"{self.product}:{self.provider}"


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
    if value == "auto":
        return "auto", None
    if value in PC_REGISTRY:
        product, provider = value.split(":", 1)
        return product, provider
    from faninsar.processing.geometry.dem_sources import parse_selection as admit

    entry = admit(value)
    if ":" not in value:
        return entry.product, None
    return entry.product, entry.provider


def get_provider(selection: str) -> PcStacSource | _P0030Source:
    """Return the admitted provider adapter without network I/O."""
    product, provider = parse_selection(selection)
    if product == "auto":
        return _P0030Source("auto", "p0030", "auto")
    if provider is None:
        from faninsar.processing.geometry.dem_sources import parse_selection as admit

        provider = admit(product).provider
    key = f"{product}:{provider}"
    if key in PC_REGISTRY:
        return PC_REGISTRY[key]
    try:
        from faninsar.processing.geometry.dem_sources import get_dem_source

        entry = get_dem_source(key)
    except (KeyError, ValueError) as error:
        raise ProviderUnavailableError(str(error)) from error
    return _P0030Source(product, str(provider), entry.name)


def _sample_geographic_windows(
    resources: tuple[SourceResource, ...],
    *,
    cache_dir: Path,
    target_longitudes: np.ndarray,
    target_latitudes: np.ndarray,
    target_center_longitude: float,
    max_fetch_bytes: int,
) -> np.ndarray | None:
    """Sample geographic source windows lazily with a cross-window halo.

    Source datasets remain independent.  For each target point this view
    resolves the six-by-six support pixels against the source windows, reads
    only those pixels (with a bounded LRU cache), checks deterministic overlap,
    and evaluates the P0032 stencil once. ``None`` requests the generic
    projected-source path.
    """
    from functools import lru_cache

    import numpy as np
    import rasterio
    from rasterio.windows import Window

    loaded: list[tuple[SourceResource, object, object, float, float, float, float]] = []
    for resource in resources:
        local = resolve_cache_path(cache_dir, resource.cache_path)
        if not local.is_file():
            fetch_asset(resource, cache_dir=cache_dir, max_bytes=max_fetch_bytes)
        dataset = rasterio.open(local)
        source_crs = dataset.crs or resource.crs
        transform = dataset.transform
        if str(source_crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            dataset.close()
            return None
        if abs(float(transform.b)) > 1.0e-12 or abs(float(transform.d)) > 1.0e-12:
            dataset.close()
            return None
        left = unwrap_longitude(float(transform.c), target_center_longitude)
        right = left + abs(float(transform.a)) * int(dataset.width)
        top = float(transform.f)
        bottom = top - abs(float(transform.e)) * int(dataset.height)
        loaded.append((resource, dataset, transform, left, right, bottom, top))
    if not loaded:
        return None
    loaded.sort(key=lambda item: item[0].identity)
    resolution_x = abs(float(loaded[0][2].a))
    resolution_y = abs(float(loaded[0][2].e))
    if resolution_x == 0.0 or resolution_y == 0.0:
        for _resource, dataset, _transform, _left, _right, _bottom, _top in loaded:
            dataset.close()
        return None

    @lru_cache(maxsize=4096)
    def read_pixel(index: int, row: int, column: int) -> float:
        """Read one source pixel while retaining only a bounded tile cache."""
        _resource, dataset, _transform, _left, _right, _bottom, _top = loaded[index]
        if row < 0 or column < 0 or row >= dataset.height or column >= dataset.width:
            return float("nan")
        values = dataset.read(1, window=Window(column, row, 1, 1))
        value = float(values[0, 0])
        nodata = dataset.nodata
        return float("nan") if nodata is not None and np.isclose(value, nodata) else value

    def candidates(longitude: float, latitude: float) -> list[int]:
        """Return sorted source windows covering one logical coordinate."""
        return [
            index
            for index, (_resource, _dataset, _transform, left, right, bottom, top) in enumerate(loaded)
            if left <= longitude < right and bottom <= latitude < top
        ]

    def support_value(longitude: float, latitude: float) -> float:
        """Read one logical support pixel and detect source disagreement."""
        matches = candidates(longitude, latitude)
        if not matches:
            return float("nan")
        values: list[float] = []
        for index in matches:
            _resource, _dataset, transform, left, _right, _bottom, _top = loaded[index]
            shift = left - float(transform.c)
            _column, _row = (~transform) * (longitude - shift, latitude)
            values.append(read_pixel(index, int(np.floor(_row)), int(np.floor(_column))))
        finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)]
        if finite.size > 1 and not np.allclose(finite, finite[0], atol=1e-3, rtol=1e-6):
            identity = loaded[matches[0]][0].identity
            message = f"overlapping DEM source windows disagree; source={identity}"
            logger.error(message)
            raise SourceConflictError(message)
        return values[0]

    seam_sampler = SeamAwareSourceSampler(
        lambda longitude, _latitude: longitude,
        target_center_longitude,
    )
    longitudes = np.asarray(
        seam_sampler.sample(
            np.asarray(target_longitudes).ravel().tolist(),
            np.asarray(target_latitudes).ravel().tolist(),
        ),
        dtype=np.float64,
    ).reshape(np.asarray(target_longitudes).shape)
    output = np.full(longitudes.shape, np.nan, dtype=np.float64)
    from faninsar.processing.dem.api import _natural_spline_six

    for flat_index, (longitude, latitude) in enumerate(
        zip(longitudes.ravel(), np.asarray(target_latitudes).ravel(), strict=True)
    ):
        matches = candidates(float(longitude), float(latitude))
        if not matches:
            continue
        _resource, _dataset, transform, left, _right, _bottom, _top = loaded[matches[0]]
        shift = left - float(transform.c)
        _column, _row = (~transform) * (float(longitude) - shift, float(latitude))
        row_base, col_base = int(np.floor(_row)), int(np.floor(_column))
        window = np.empty((6, 6), dtype=np.float64)
        for row_offset in range(6):
            for col_offset in range(6):
                world_x, world_y = transform * (
                    col_base + col_offset - 0.5,
                    row_base + row_offset - 0.5,
                )
                world_x += shift
                window[row_offset, col_offset] = support_value(world_x, world_y)
        along = _natural_spline_six(window, np.asarray([_column - col_base]))
        value = _natural_spline_six(along, np.asarray([_row - row_base]))
        output.ravel()[flat_index] = float(value[0])
    for _resource, dataset, _transform, _left, _right, _bottom, _top in loaded:
        dataset.close()
    return output


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

    if isinstance(source, _P0030Source):
        from .p0030_adapter import materialize

        return materialize(source, grid, cache_dir=cache_dir, budget=budget)
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
    target_longitudes, target_latitudes = transformer.transform(target_x, target_y)
    logical_samples = _sample_geographic_windows(
        resources,
        cache_dir=cache_dir,
        target_longitudes=np.asarray(target_longitudes, dtype=np.float64),
        target_latitudes=np.asarray(target_latitudes, dtype=np.float64),
        target_center_longitude=0.5 * (bounds[0] + bounds[2]),
        max_fetch_bytes=(budget.max_fetch_bytes if budget else 2**33),
    )
    if logical_samples is not None:
        from faninsar.processing.dem.api import RasterDEM

        return RasterDEM(
            array=np.asarray(logical_samples, dtype=np.float32),
            grid=grid,
            vertical_datum="egm2008",
            provenance={
                "provider": "pc",
                "collection": source.collection_id,
                "asset": source.asset_key,
                "resampling": "direct-source-target",
                "seam_support": "6x6-logical-window",
            },
        )
    # Projected or rotated PC assets use the same bounded source-window
    # sampler as the P0030 adapter.  Keep this fallback lazy as well: a
    # provider resource must never be expanded into a full source mosaic.
    from .p0030_adapter import _sample_dataset

    for resource in resources:
        local = resolve_cache_path(cache_dir, resource.cache_path)
        if not local.is_file():
            fetch_asset(
                resource,
                cache_dir=cache_dir,
                max_bytes=(budget.max_fetch_bytes if budget else 2**33),
            )
        with rasterio.open(local) as dataset:
            source_crs = dataset.crs or resource.crs
            sampled = _sample_dataset(
                dataset,
                np.asarray(target_x),
                np.asarray(target_y),
                target_crs=str(grid.crs),
                source_crs=source_crs,
                source_nodata=dataset.nodata,
            )
            if min(dataset.height, dataset.width) < SOURCE_KERNEL_SIZE:
                logger.debug(
                    "source window is smaller than the qualified %dx%d halo",
                    SOURCE_KERNEL_SIZE,
                    SOURCE_KERNEL_SIZE,
                )
            overlap = np.isfinite(destination) & np.isfinite(sampled)
            if np.any(overlap & ~np.isclose(destination, sampled, atol=1e-3, rtol=1e-6)):
                message = (
                    "overlapping DEM source windows disagree; "
                    f"source={resource.identity}"
                )
                logger.error(message)
                raise SourceConflictError(message)
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
    "ProviderUnavailableError", "SourceConflictError", "SourceResource",
    "get_provider", "materialize_source",
    "parse_selection",
]
