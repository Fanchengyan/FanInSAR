# ruff: noqa: EM101, EM102, TRY003, TID252, PLW2901, D105
"""Provider registry and Planetary Computer Copernicus DEM adapters."""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from faninsar.logging import setup_logger
from faninsar.processing.geometry.dem_sources import (
    DeferredStacPlan,
    _stac_endpoint_identity,
)

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

# Adapter instances are keyed by the injected transport seam.  Keeping this
# small process-local cache avoids duplicate remote catalog registrations while
# preserving the caller's client/signer fixtures.
_PC_ADAPTERS: dict[tuple[str, str, int, int], object] = {}


class _OfflinePcClient:
    """Mark an injected deterministic client as an offline remote fixture."""

    _faninsar_offline = True

    def __init__(self, client: object) -> None:
        self._client = client

    def __getattr__(self, name: str) -> object:
        return getattr(self._client, name)

    def search(self, **kwargs: object) -> object:
        """Call a small fixture search implementation compatibly."""
        search = self._client.search
        try:
            return search(**kwargs)  # type: ignore[operator]
        except TypeError as error:
            if "max_items" not in str(error):
                raise
            kwargs.pop("max_items", None)
            return search(**kwargs)  # type: ignore[operator]


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
    # The provider-neutral remote asset is retained only for the duration of
    # materialization.  Its href is deliberately unsigned; the remote
    # adapter owns the short-lived signed URL and download ledger.
    remote_asset: object | None = field(default=None, repr=False, compare=False)

    @property
    def identity(self) -> str:
        """Return identity independent of ephemeral SAS query parameters."""
        parts = urllib.parse.urlsplit(self.href)
        # Identity never includes query signatures or URL userinfo, even when
        # a resource record is constructed directly in a test/tooling seam.
        hostname = parts.hostname or ""
        netloc = hostname
        try:
            port = parts.port
        except ValueError:
            port = -1
        if port not in (None, 443):
            netloc = f"{hostname}:{port}"
        unsigned = urllib.parse.urlunsplit(
            (parts.scheme.lower(), netloc, parts.path, "", "")
        )
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
        specs = {
            "glo30": ("cop-dem-glo-30", (3600, 3600), 30.0),
            "glo90": ("cop-dem-glo-90", (1200, 1200), 90.0),
        }
        try:
            collection, shape, resolution = specs[self.product]
        except KeyError as error:
            raise ValueError(
                f"unsupported Planetary Computer DEM product: {self.product!r}"
            ) from error
        if self.collection_id and self.collection_id != collection:
            raise ValueError("Planetary Computer collection is registry-owned")
        object.__setattr__(self, "collection_id", collection)
        object.__setattr__(self, "tile_shape", shape)
        object.__setattr__(self, "resolution_m", resolution)
        validate_https_origin(self.stac_url, {PC_STAC_HOST})

    def discover(
        self,
        bounds: tuple[float, float, float, float] | DeferredStacPlan,
        *,
        client: object | None = None,
        signer: Callable[[object], object] | None = None,
        budget: object | None = None,
    ) -> tuple[SourceResource, ...]:
        """Search and sign assets for automatic source windows."""
        plan = bounds if isinstance(bounds, DeferredStacPlan) else self.plan(bounds)
        endpoint, allowed_hosts = _stac_endpoint_identity(self.stac_url)
        if (
            plan.endpoint_identity != endpoint
            or plan.collection != self.collection_id
            or plan.asset != self.asset_key
            or plan.provider != self.provider
            or plan.product != self.product
            or plan.allowed_hosts != allowed_hosts
        ):
            message = "deferred STAC plan does not belong to this source"
            logger.error(message)
            raise ProviderUnavailableError(message)

        # Keep the historical lightweight fixture seam for callers that
        # inject a bare ``search`` object.  Real ``pystac-client`` instances
        # expose ``_stac_io`` and always use the remote boundary below;
        # explicit remote fixtures opt in with ``_faninsar_offline``.
        if (
            client is not None
            and not getattr(client, "_faninsar_offline", False)
            and not hasattr(client, "_stac_io")
        ):
            resources: list[SourceResource] = []
            for window in plan.windows:
                resources.extend(self._discover_window(window, client, signer))
            unique = {resource.identity: resource for resource in resources}
            if not unique:
                raise ProviderUnavailableError(
                    f"no {plan.collection}/{plan.asset} coverage for {plan.bounds!r}"
                )
            return tuple(unique.values())

        # Discovery and signing are intentionally delegated to the remote
        # boundary.  Apart from centralizing URL validation, this is what
        # gives PC discovery one operation-wide request/response ledger.
        try:
            from faninsar.remote import RemoteResourceBudget, search
            from faninsar.remote.providers.planetary_computer import (
                PlanetaryComputerAdapter,
            )
        except ImportError as error:
            raise ProviderUnavailableError(
                "provider 'pc' requires the FanInSAR remote extras"
            ) from error

        # A catalog is registered once per source/client seam.  The adapter
        # keeps signed URLs in memory so the resulting RemoteAsset remains
        # safe to persist and is still downloadable through remote.download.
        cache_key = (self.stac_url, self.collection_id, id(client), id(signer))
        adapter = _PC_ADAPTERS.get(cache_key)
        if adapter is None:
            adapter = PlanetaryComputerAdapter(
                client=(
                    _OfflinePcClient(client)
                    if client is not None
                    and not getattr(client, "_faninsar_offline", False)
                    else client
                ),
                signer=signer,
                endpoint=self.stac_url,
            )
            # Older remote adapters pinned GLO-30/data.  The DEM registry has
            # always exposed the same PC shape for GLO-90, so retain that
            # public selection while allowing the remote adapter to service
            # the registered collection/asset pair.
            object.__setattr__(adapter, "collection", self.collection_id)
            object.__setattr__(adapter, "asset_key", self.asset_key)
            catalog = f"faninsar-dem-pc-{self.collection_id}"
            try:
                adapter.register(catalog)
            except ValueError as error:
                if "already registered" not in str(error):
                    raise
            _PC_ADAPTERS[cache_key] = adapter
        else:
            catalog = f"faninsar-dem-pc-{self.collection_id}"
        if budget is None or not hasattr(budget, "max_items"):
            operation_budget = RemoteResourceBudget(
                max_output_bytes=(
                    int(getattr(budget, "max_fetch_bytes", 2**33))
                    if budget is not None
                    else 2**31
                ),
            )
        else:
            operation_budget = budget
        resources: list[SourceResource] = []
        for window in plan.windows:
            # ``remote.search`` returns immutable RemoteAsset descriptors;
            # no provider SDK object crosses back into the DEM layer.
            from faninsar.query import BoundingBox

            items = search(
                BoundingBox(*window, crs=4326),
                catalog=catalog,
                budget=operation_budget,
            )
            for item in items:
                asset = item.assets.get(self.asset_key)
                if asset is None:
                    continue
                href = str(asset.href)
                self._validate_asset_href(href)
                unsigned_name = urllib.parse.urlsplit(href).path.rsplit("/", 1)[-1]
                cache_relative = f"{self.collection_id}/{unsigned_name}"
                resolve_cache_path(".", cache_relative)
                resources.append(
                    SourceResource(
                        href=href,
                        cache_path=Path(cache_relative),
                        collection_id=self.collection_id,
                        asset_key=self.asset_key,
                        item_id=str(item.item_id),
                        shape=self.tile_shape,
                        remote_asset=asset,
                    )
                )
        unique: dict[str, SourceResource] = {}
        for resource in sorted(resources, key=lambda value: value.identity):
            unique.setdefault(resource.identity, resource)
        if not unique:
            raise ProviderUnavailableError(
                f"no {plan.collection}/{plan.asset} coverage for {plan.bounds!r}"
            )
        return tuple(unique.values())

    def plan(
        self,
        bounds: tuple[float, float, float, float],
        *,
        client: object | None = None,
        signer: Callable[[object], object] | None = None,
    ) -> DeferredStacPlan:
        """Return an immutable query descriptor without opening a socket.

        ``client`` and ``signer`` are accepted for call-shape parity with
        :meth:`discover`, but are intentionally ignored.  They cannot make a
        planning call perform network I/O; discovery belongs exclusively to
        the materialization boundary.
        """
        del client, signer
        raw_bounds = tuple(float(value) for value in bounds)
        endpoint, allowed_hosts = _stac_endpoint_identity(self.stac_url)
        windows = plan_query_windows(raw_bounds)
        return DeferredStacPlan(
            endpoint_identity=endpoint,
            collection=self.collection_id,
            asset=self.asset_key,
            bounds=raw_bounds,
            windows=windows,
            provider=self.provider,
            product=self.product,
            vertical_datum=self.vertical_datum,  # type: ignore[arg-type]
            allowed_hosts=allowed_hosts,
        )

    def _discover_window(
        self,
        bounds: tuple[float, float, float, float],
        client: object,
        signer: Callable[[object], object] | None,
    ) -> list[SourceResource]:
        search = getattr(client, "search", None)
        if search is None:
            collection = getattr(client, "get_collection", lambda _name: client)(
                self.collection_id
            )
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
        try:
            validate_https_origin(href, {PC_ASSET_HOST})
        except ValueError as error:
            raise ProviderUnavailableError(
                "Planetary Computer asset origin is not approved"
            ) from error


GLO30_PC = PcStacSource("glo30")
GLO90_PC = PcStacSource("glo90")
PC_REGISTRY: dict[str, PcStacSource] = {"glo30:pc": GLO30_PC, "glo90:pc": GLO90_PC}
_PRODUCTS = frozenset(
    {
        "auto",
        "glo30",
        "glo90",
        "nasadem",
        "alos-dem",
        "srtm-skadi",
        "terrain-tiles",
        "arcticdem-10",
        "arcticdem-32",
        "arcticdem-2",
        "rema-10",
        "rema-32",
        "rema-2",
        "nisar-glo30",
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
        return (
            float("nan") if nodata is not None and np.isclose(value, nodata) else value
        )

    def candidates(longitude: float, latitude: float) -> list[int]:
        """Return sorted source windows covering one logical coordinate."""
        return [
            index
            for index, (
                _resource,
                _dataset,
                _transform,
                left,
                right,
                bottom,
                top,
            ) in enumerate(loaded)
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
            values.append(
                read_pixel(index, int(np.floor(_row)), int(np.floor(_column)))
            )
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
    bounds = (
        min(lon_a, lon_b),
        min(lat_a, lat_b),
        max(lon_a, lon_b),
        max(lat_a, lat_b),
    )
    resources = source.discover(
        source.plan(bounds), client=client, signer=signer, budget=budget
    )
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
            if np.any(
                overlap & ~np.isclose(destination, sampled, atol=1e-3, rtol=1e-6)
            ):
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
    destination = resolve_cache_path(cache_dir, resource.cache_path)
    if resource.remote_asset is not None:
        from faninsar.remote import RemoteResourceBudget, download

        # ``max_bytes`` is the established DEM budget knob.  Keep it as the
        # remote operation's output/temporary/cache bound so the adapter's
        # ledger remains authoritative for the complete-file transfer.
        remote_budget = RemoteResourceBudget(
            max_output_bytes=max_bytes,
            max_temporary_bytes=max_bytes,
            max_cache_bytes=max_bytes,
        )
        return download(resource.remote_asset, destination, budget=remote_budget)  # type: ignore[arg-type]

    # Compatibility seam for callers constructing SourceResource directly.
    # Production resources always carry a RemoteAsset from ``discover``.
    if session is None:
        message = "PC source resource has no provider-neutral remote asset"
        logger.error(message)
        raise ProviderUnavailableError(message)
    response = session.get(  # type: ignore[attr-defined]
        resource.href,
        stream=True,
        allow_redirects=False,
        timeout=(10, 120),
    )
    try:
        if response.status_code in {301, 302, 303, 307, 308}:
            raise ProviderUnavailableError(
                "Planetary Computer asset redirects are rejected"
            )
        response.raise_for_status()
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
    "GLO30_PC",
    "GLO90_PC",
    "PC_REGISTRY",
    "PC_STAC_URL",
    "PcStacSource",
    "ProviderUnavailableError",
    "SourceConflictError",
    "SourceResource",
    "get_provider",
    "materialize_source",
    "parse_selection",
]
