"""Multi-source water-mask registry (PROPOSAL-0039): products, providers, planning.

Mirrors the PROPOSAL-0030 DEM registry
(:mod:`faninsar.processing.geometry.dem_sources`): a **product group** (what
the mask layer is: resolution, tile grid, extraction polarity) is separated
from a **provider group** (where the bytes live: base URL, tile layout, access
shape).  Every entry's ``name`` is the **bare provider name**
(``"gsw"``, ``"worldcover"``); the selection grammar is
``"water"`` / ``"water:<provider>"`` with :data:`AUTO_SOURCE_NAME` ``"water"``
resolving to the wired default provider (``gsw``).  Each wired entry plans its
fetch as one self-describing
:class:`~faninsar.processing.geometry.dem_transport.FetchPlan` (reusing the
PROPOSAL-0030 ``Tile`` / ``TileSet`` shapes) without touching the network.

Zero network by contract: import, :func:`list_mask_sources`,
:func:`get_mask_source`, and ``plan()`` perform no socket I/O.

Live-verified tile layouts (anonymous HTTP 200 + range support, PROPOSAL-0039
spike 2026-08-29 plus negative-quadrant probes):

- **JRC GSW occurrence 30 m** (auto default): flat pattern
  ``.../downloads/occurrence/occurrence_{lon}{E|W}_{lat}{N|S}.tif`` on a
  10° x 10° grid of lower-left-origin tiles (e.g. ``occurrence_10W_0N.tif``).
  The spike's 404 shapes — the nested ``{lon}E/W_{lat}N/S.tif`` path guess and
  literal-minus naming — are never rendered (pinned by the registry tests).
- **ESA WorldCover 10 m** (v200, 2021): flat pattern
  ``.../v200/2021/map/ESA_WorldCover_10m_2021_v200_{N|S}{lat:02d}{E|W}{lon:03d}_Map.tif``
  on a 3° x 3° grid (zero-padded; the unpadded ``N{lat}E{lon}`` guess 404s).
  Pure-ocean tiles are absent from the bucket, so WorldCover tiles carry the
  transport's ``ocean_404_skip`` flag (skadi precedent).
- **OSM Overpass** (opt-in): registered but **not wired** until the Overpass
  response-size cap and truncation detection bind; ``plan()`` fails closed
  with a clear error and the endpoint is https-enforced at construction.

The mask polarity/extraction metadata lives with the source: GSW is an
occurrence raster (threshold semantics, default 50), WorldCover is categorical
(water class 80 excluded), OSM arrives already vector.
"""

from __future__ import annotations

import math
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from faninsar.data.query import BoundingBox
from faninsar.logging import setup_logger
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry import dem_sources as _dem_sources
from faninsar.processing.geometry.dem_sources import _bounds_tuple
from faninsar.processing.geometry.dem_transport import FetchPlan, Tile, TileSet

logger = setup_logger(__name__)

__all__ = [
    "AUTO_SOURCE_NAME",
    "DEFAULT_OVERPASS_ENDPOINT",
    "DEFAULT_PRODUCT",
    "GSW_BASE_URL",
    "GSW_DEFAULT_THRESHOLD",
    "GSW_TILE_SIZE_DEG",
    "MASK_PRODUCT_DEFAULTS",
    "WATER_CLASSIFICATION_VERSION",
    "WORLDCOVER_BASE_URL",
    "WORLDCOVER_TILE_SIZE_DEG",
    "WORLDCOVER_WATER_CLASS",
    "GswOccurrenceSource",
    "MaskExtraction",
    "MaskSource",
    "MaskSourceUnavailableError",
    "OsmOverpassSource",
    "WorldCoverWaterSource",
    "get_mask_source",
    "list_mask_sources",
    "mask_cache_relative_path",
    "parse_mask_selection",
    "tile_snap",
    "validate_mask_source_name",
]

#: Selection grammar: ``"water"`` / ``"water:<provider>"``; the ``auto``
#: default resolves to the wired default provider (``gsw``).
AUTO_SOURCE_NAME = "water"
DEFAULT_PRODUCT = "water"

#: JRC Global Surface Water occurrence base (flat 10-degree tile layout).
GSW_BASE_URL = (
    "https://storage.googleapis.com/global-surface-water/downloads/occurrence"
)
#: ESA WorldCover v200 (2021) base (flat 3-degree tile layout).
WORLDCOVER_BASE_URL = "https://esa-worldcover.s3.amazonaws.com/v200/2021/map"
#: Configurable Overpass endpoint (https-enforced; OSM stays unwired in v1).
DEFAULT_OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"

GSW_TILE_SIZE_DEG = 10.0
WORLDCOVER_TILE_SIZE_DEG = 3.0
#: GSW occurrence raster is 30 m; WorldCover land-cover is 10 m.
GSW_RESOLUTION_M = 30.0
WORLDCOVER_RESOLUTION_M = 10.0
#: Water polarity defaults: GSW occurrence threshold, WorldCover class code.
GSW_DEFAULT_THRESHOLD = 50
WORLDCOVER_WATER_CLASS = 80
# Internal qualified-label receipt used by the water realization seam.  This
# is metadata, not a selectable classifier protocol.
WATER_CLASSIFICATION_VERSION = "edge-connected-v1"

#: Minimum valid tile size in bytes. GSW coastal tiles run ~15.6 MB and
#: WorldCover land tiles ~10 MB, but all-land/all-water tiles compress far
#: below that; the floor only has to reject error pages and truncations.
GSW_MIN_TILE_BYTES = 1 << 16
WORLDCOVER_MIN_TILE_BYTES = 1 << 16

MaskExtraction = Literal["threshold", "categorical", "vector"]
AuthClass = Literal["none", "token"]

_EXTRACTION_MODES: frozenset[str] = frozenset({"threshold", "categorical", "vector"})

BoundsLike = BoundingBox | tuple[float, float, float, float]


class MaskSourceUnavailableError(InvalidProcessingStateError):
    """A registry mask source cannot serve the request or is not yet wired."""


# ---------------------------------------------------------------------------
# Shared guards (reused from dem_sources) and helpers
# ---------------------------------------------------------------------------


def validate_mask_source_name(name: str) -> None:
    """Validate a mask registry name (delegates to the DEM source guard).

    Names flow into cache partition directories, so the same charset guard as
    :func:`faninsar.processing.geometry.dem_sources._validate_source_name`
    applies verbatim.

    Parameters
    ----------
    name : str
        Registry or selection name candidate.

    Raises
    ------
    ValueError
        If the name is empty or outside the
        ``[a-z0-9]+([-_.@][a-z0-9]+)*`` charset.

    """
    _dem_sources._validate_source_name(name)


def mask_cache_relative_path(relative_path: str, cache_dir: object) -> None:
    """Reject mask cache-relative paths with traversal (DEM guard delegation).

    Parameters
    ----------
    relative_path : str
        Registry-supplied cache-relative path candidate.
    cache_dir : object
        Cache root the path would be joined against (unused for the check
        itself; part of the boundary signature).

    Raises
    ------
    ValueError
        If the path contains a ``..`` component or is absolute.

    """
    _dem_sources.validate_cache_relative_path(relative_path, cache_dir)


def _safe_relative(relative: str) -> Path:
    """Build a cache-relative Path after the traversal guard."""
    mask_cache_relative_path(relative, None)
    return Path(relative)


def tile_snap(band: tuple[float, float], tile_size_deg: float) -> tuple[float, float]:
    """Snap a 1-D degree band outward onto the global fetch-tile grid.

    This is the **one shared snap** for the masking package: the padded fetch
    band (:mod:`faninsar.processing.masking.mask`) and the antimeridian seam
    guard both snap through it, and the registry's tile enumeration is
    expressed on the same grid, so the guarded band is exactly what gets
    fetched.  The semantics reproduce the executed PROPOSAL-0039 round-5
    verification: the grid is anchored at -180° with half-open tiles
    ``[k * size - 180, (k + 1) * size - 180)``; the lower edge snaps DOWN to
    the origin of the tile containing it and the upper edge snaps UP to the
    exclusive end of the tile containing it (e.g. ``165..175`` at 10° snaps to
    ``[160, 180)`` and ``155..169`` to ``[150, 170)``).  The anchor coincides
    with the -90° latitude grid for every tile size dividing 90, so the same
    function serves both axes.

    Parameters
    ----------
    band : tuple of float
        ``(lower, upper)`` degree band (one axis only).
    tile_size_deg : float
        Tile grid size in degrees; must be positive.

    Returns
    -------
    tuple of float
        The snapped ``(lower, upper)`` band covering the input.

    Raises
    ------
    ValueError
        If ``tile_size_deg`` is not positive.

    """
    if tile_size_deg <= 0:
        message = f"tile_size_deg must be positive, got {tile_size_deg!r}"
        logger.error(message)
        raise ValueError(message)
    lower, upper = float(band[0]), float(band[1])
    snapped_lower = math.floor((lower + 180.0) / tile_size_deg) * tile_size_deg - 180.0
    snapped_upper = (
        math.floor((upper + 180.0) / tile_size_deg) * tile_size_deg
        + tile_size_deg
        - 180.0
    )
    return (snapped_lower, snapped_upper)


def _iter_tile_indices(
    bounds: BoundsLike,
    tile_size_deg: float,
) -> list[tuple[int, int]]:
    """Return latitude-major tile grid indices covering ``bounds``.

    Indices are measured on the global grid anchored at (-90°, -180°) with
    half-open tiles ``[k * size, (k + 1) * size)`` degrees.  Indices outside
    the geographic domain are clamped on both ends so no phantom tile beyond
    the poles or the antimeridian is ever planned (a tile origin at +180° or
    +90° does not exist; the +180° seam itself belongs to no tile).

    Parameters
    ----------
    bounds : BoundingBox or tuple
        Requested geographic bounds in degrees.
    tile_size_deg : float
        Tile grid size in degrees.

    Returns
    -------
    list of tuple
        ``(lat_index, lon_index)`` pairs in latitude-major order.

    """
    min_lon, min_lat, max_lon, max_lat = _bounds_tuple(bounds)
    lat_lo = max(0, math.floor((min_lat + 90.0) / tile_size_deg))
    lat_hi = min(
        math.floor((max_lat + 90.0) / tile_size_deg),
        int(180.0 / tile_size_deg) - 1,
    )
    lon_lo = max(0, math.floor((min_lon + 180.0) / tile_size_deg))
    lon_hi = min(
        math.floor((max_lon + 180.0) / tile_size_deg),
        int(360.0 / tile_size_deg) - 1,
    )
    return [
        (lat_index, lon_index)
        for lat_index in range(lat_lo, lat_hi + 1)
        for lon_index in range(lon_lo, lon_hi + 1)
    ]


def _tile_origin(index: int, tile_size_deg: float, domain_min: float) -> int:
    """Return the integer degree origin of one grid cell (exact for 3°/10°)."""
    return round(index * tile_size_deg + domain_min)


def _hemisphere_tag(value: int, positive: str, negative: str) -> str:
    """Return ``f"{abs(value)}{letter}"`` with the hemisphere letter."""
    return f"{abs(value)}{positive if value >= 0 else negative}"


def _hemisphere_letter(value: int, positive: str, negative: str) -> str:
    """Return the hemisphere letter for a signed integer degree origin."""
    return positive if value >= 0 else negative


# ---------------------------------------------------------------------------
# MaskSource ABC + provider subclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MaskSource(ABC):
    """Registry entry describing one selectable water product/provider pair.

    Parameters
    ----------
    name
        Registry selection name (bare provider name).
    description
        Human-readable summary.
    product
        Product-group identifier; every v1 entry is a ``water`` layer.
    provider
        Provider-group identifier (``gsw``, ``worldcover``, ``osm-overpass``).
    resolution_m
        Nominal ground sampling distance in metres (0 for vector sources).
    tile_size_deg
        Fetch-tile grid size in degrees (0 for single-artifact sources).
    extraction
        How the boolean water layer is extracted: ``"threshold"`` (occurrence
        rasters), ``"categorical"`` (class-code rasters), or ``"vector"``
        (already-polygon sources that skip extraction).
    threshold
        Default occurrence threshold for ``"threshold"`` sources (GSW: 50).
    excluded_values
        Class codes counted as water for ``"categorical"`` sources
        (WorldCover: ``(80,)``).
    wired
        Whether this provider is actually reachable in v1; unwired providers
        fail closed at selection and plan time.
    auth
        Access class (all v1 providers are anonymous).

    """

    # selection level
    name: str
    description: str
    # product group
    product: str = DEFAULT_PRODUCT
    provider: str = "gsw"
    resolution_m: float = GSW_RESOLUTION_M
    tile_size_deg: float = GSW_TILE_SIZE_DEG
    # extraction / polarity metadata (value 1 = water / removed)
    extraction: MaskExtraction = "threshold"
    threshold: int | None = None
    excluded_values: tuple[int, ...] = ()
    wired: bool = True
    auth: AuthClass = "none"

    def __post_init__(self) -> None:
        """Validate identity invariants at construction (fail closed)."""
        validate_mask_source_name(self.name)
        if self.tile_size_deg < 0.0:
            message = f"tile_size_deg must be >= 0, got {self.tile_size_deg!r}"
            logger.error(message)
            raise ValueError(message)
        if self.resolution_m < 0.0:
            message = f"resolution_m must be >= 0, got {self.resolution_m!r}"
            logger.error(message)
            raise ValueError(message)
        if self.extraction not in _EXTRACTION_MODES:
            message = (
                f"invalid extraction mode {self.extraction!r}: expected one of "
                f"{sorted(_EXTRACTION_MODES)}"
            )
            logger.error(message)
            raise ValueError(message)
        if self.extraction == "threshold" and self.threshold is None:
            message = (
                f"source {self.name!r} uses 'threshold' extraction but no "
                "default threshold is configured"
            )
            logger.error(message)
            raise ValueError(message)
        if self.extraction == "categorical" and not self.excluded_values:
            message = (
                f"source {self.name!r} uses 'categorical' extraction but no "
                "excluded_values are configured"
            )
            logger.error(message)
            raise ValueError(message)

    @abstractmethod
    def plan(self, bounds: BoundsLike) -> FetchPlan:
        """Return the self-describing fetch plan for ``bounds``.

        No I/O here; planning is a pure function of the registry entry and
        the requested bounds.
        """


@dataclass(frozen=True, slots=True)
class GswOccurrenceSource(MaskSource):
    """JRC Global Surface Water occurrence 30 m (auto default provider).

    Plans the live-verified flat 10° tile layout
    ``occurrence_{lon}{E|W}_{lat}{N|S}.tif`` with lower-left-origin naming
    (``occurrence_10W_0N.tif`` covers ``[-10, 0) x [0, 10)``).  The bucket
    carries the complete in-domain grid (ocean tiles ship occurrence 0), so
    no 404 tolerance is enabled.
    """

    base_url: str = GSW_BASE_URL
    min_bytes: int = GSW_MIN_TILE_BYTES

    def plan(self, bounds: BoundsLike) -> FetchPlan:
        """Enumerate 10-degree occurrence tiles into a TileSet plan."""
        host = urllib.parse.urlsplit(self.base_url).hostname or ""
        tiles: list[Tile] = []
        for lat_index, lon_index in _iter_tile_indices(bounds, self.tile_size_deg):
            lat_origin = _tile_origin(lat_index, self.tile_size_deg, -90.0)
            lon_origin = _tile_origin(lon_index, self.tile_size_deg, -180.0)
            filename = (
                "occurrence_"
                f"{_hemisphere_tag(lon_origin, 'E', 'W')}_"
                f"{_hemisphere_tag(lat_origin, 'N', 'S')}.tif"
            )
            tiles.append(
                Tile(
                    url=f"{self.base_url}/{filename}",
                    cache_path=_safe_relative(
                        f"{self.product}-{self.provider}/{filename}"
                    ),
                    min_bytes=self.min_bytes,
                    ranged=True,
                    ocean_404_skip=False,
                )
            )
        return TileSet(allowed_hosts=(host,), tiles=tuple(tiles))


@dataclass(frozen=True, slots=True)
class WorldCoverWaterSource(MaskSource):
    """ESA WorldCover v200 (2021) water class 80, 10 m on a 3-degree grid.

    Plans the live-verified zero-padded layout
    ``ESA_WorldCover_10m_2021_v200_{N|S}{lat:02d}{E|W}{lon:03d}_Map.tif``.
    The bucket stores only tiles intersecting land, so tiles carry the
    transport's ``ocean_404_skip`` flag: a 404 for a known-empty tile is
    tolerated as "no water here" instead of failing the fetch set.
    """

    base_url: str = WORLDCOVER_BASE_URL
    min_bytes: int = WORLDCOVER_MIN_TILE_BYTES

    def plan(self, bounds: BoundsLike) -> FetchPlan:
        """Enumerate 3-degree land-cover tiles into a TileSet plan."""
        host = urllib.parse.urlsplit(self.base_url).hostname or ""
        tiles: list[Tile] = []
        for lat_index, lon_index in _iter_tile_indices(bounds, self.tile_size_deg):
            lat_origin = _tile_origin(lat_index, self.tile_size_deg, -90.0)
            lon_origin = _tile_origin(lon_index, self.tile_size_deg, -180.0)
            filename = (
                "ESA_WorldCover_10m_2021_v200_"
                f"{_hemisphere_letter(lat_origin, 'N', 'S')}"
                f"{abs(lat_origin):02d}"
                f"{_hemisphere_letter(lon_origin, 'E', 'W')}"
                f"{abs(lon_origin):03d}_Map.tif"
            )
            tiles.append(
                Tile(
                    url=f"{self.base_url}/{filename}",
                    cache_path=_safe_relative(
                        f"{self.product}-{self.provider}/{filename}"
                    ),
                    min_bytes=self.min_bytes,
                    ranged=True,
                    ocean_404_skip=True,
                )
            )
        return TileSet(allowed_hosts=(host,), tiles=tuple(tiles))


@dataclass(frozen=True, slots=True)
class OsmOverpassSource(MaskSource):
    """OpenStreetMap water polygons via the Overpass API (opt-in, unwired).

    When wired, this provider will plan a single Overpass bbox query
    :class:`~faninsar.processing.geometry.dem_transport.Artifact` (the response
    arrives already vector and skips polygonization).  It stays **unwired**
    until the response-size cap and truncated-response detection bind
    (PROPOSAL-0039 pending item): a capped or rate-limited response must never
    be applied as a complete mask.  The endpoint is configurable but
    https-enforced at construction.
    """

    endpoint: str = DEFAULT_OVERPASS_ENDPOINT
    provider: str = "osm-overpass"
    extraction: MaskExtraction = "vector"
    tile_size_deg: float = 0.0
    resolution_m: float = 0.0
    wired: bool = False

    def __post_init__(self) -> None:
        """Validate identity plus the https endpoint rule (fail closed)."""
        MaskSource.__post_init__(self)  # zero-arg super() breaks frozen slots
        if not str(self.endpoint).lower().startswith("https://"):
            message = (
                f"Overpass endpoint must be https (got {self.endpoint!r}); "
                "plain-http endpoints are rejected fail-closed"
            )
            logger.error(message)
            raise ValueError(message)

    def plan(self, bounds: BoundsLike) -> FetchPlan:
        """Fail closed: the Overpass response-size cap is pending."""
        del bounds
        message = (
            "OSM Overpass water provider is not wired in v1: the Overpass "
            "response-size cap and truncation detection are pending "
            "(PROPOSAL-0039); select 'gsw' or 'worldcover' instead"
        )
        logger.error(message)
        raise MaskSourceUnavailableError(message)


# ---------------------------------------------------------------------------
# Selection grammar
# ---------------------------------------------------------------------------


def parse_mask_selection(selection: str) -> MaskSource:
    """Resolve ``"water"`` / ``"water:<provider>"`` to a mask source.

    Fail-closed on unknown products/providers, registered-but-unwired pairs
    (``water:osm-overpass``), and hostile payloads (traversal fragments,
    trailing colons, homoglyphs, surrounding whitespace, extra segments).

    Parameters
    ----------
    selection : str
        Selection expression such as ``water`` or ``water:gsw``.

    Returns
    -------
    MaskSource
        The resolved registry entry.

    Raises
    ------
    TypeError
        If ``selection`` is not a string.
    ValueError
        When any grammar component fails exact-match validation.

    """
    if not isinstance(selection, str):
        message = f"mask selection must be a string, got {type(selection)!r}"
        raise TypeError(message)
    parts = selection.split(":")
    if len(parts) > 2:
        message = (
            f"invalid mask selection {selection!r}: at most one ':' separator "
            "is allowed ('<product>' or '<product>:<provider>')"
        )
        logger.error(message)
        raise ValueError(message)
    product = parts[0]
    provider = parts[1] if len(parts) == 2 else None
    if product not in MASK_PRODUCT_DEFAULTS:
        valid_products = ", ".join(sorted(MASK_PRODUCT_DEFAULTS))
        message = (
            f"unknown mask product {product!r}; valid products are: {valid_products}"
        )
        logger.error(message)
        raise ValueError(message)
    providers = MASK_PRODUCT_DEFAULTS[product]["providers"]
    default_provider = MASK_PRODUCT_DEFAULTS[product]["default"]
    if provider is None:
        return get_mask_source(_PRODUCT_REGISTRY[(product, default_provider)])
    if provider not in providers:
        message = (
            f"unknown mask provider {provider!r} for product {product!r}; "
            f"valid providers are: {', '.join(sorted(providers))}"
        )
        logger.error(message)
        raise ValueError(message)
    pair = (product, provider)
    if pair not in _PRODUCT_REGISTRY:
        message = (
            f"provider {provider!r} is not wired for product {product!r} "
            "(registered but unavailable in this release)"
        )
        logger.error(message)
        raise ValueError(message)
    return get_mask_source(_PRODUCT_REGISTRY[pair])


# ---------------------------------------------------------------------------
# Registry construction
# ---------------------------------------------------------------------------


def _build_registry() -> dict[str, MaskSource]:
    """Construct the wired v1 entries (pure, no I/O)."""
    registry: dict[str, MaskSource] = {}
    registry["gsw"] = GswOccurrenceSource(
        name="gsw",
        description=(
            "JRC Global Surface Water occurrence 30 m (threshold 50, "
            "10-degree tiles, anonymous)"
        ),
        product=DEFAULT_PRODUCT,
        provider="gsw",
        resolution_m=GSW_RESOLUTION_M,
        tile_size_deg=GSW_TILE_SIZE_DEG,
        extraction="threshold",
        threshold=GSW_DEFAULT_THRESHOLD,
        wired=True,
        auth="none",
    )
    registry["worldcover"] = WorldCoverWaterSource(
        name="worldcover",
        description=(
            "ESA WorldCover v200 (2021) 10 m water class 80 (3-degree tiles, "
            "anonymous; ocean tiles absent from the bucket)"
        ),
        product=DEFAULT_PRODUCT,
        provider="worldcover",
        resolution_m=WORLDCOVER_RESOLUTION_M,
        tile_size_deg=WORLDCOVER_TILE_SIZE_DEG,
        extraction="categorical",
        excluded_values=(WORLDCOVER_WATER_CLASS,),
        wired=True,
        auth="none",
    )
    return registry


_REGISTRY: dict[str, MaskSource] = _build_registry()

#: product -> {providers: {provider: {wired, auth}}, default: provider}
MASK_PRODUCT_DEFAULTS: dict[str, dict] = {
    DEFAULT_PRODUCT: {
        "providers": {
            "gsw": {"wired": True, "auth": "none"},
            "worldcover": {"wired": True, "auth": "none"},
            "osm-overpass": {"wired": False, "auth": "none"},
        },
        "default": "gsw",
    },
}

_PRODUCT_REGISTRY: dict[tuple[str, str], str] = {}
for _name, _source in _REGISTRY.items():
    _PRODUCT_REGISTRY[(_source.product, _source.provider)] = _name


# ---------------------------------------------------------------------------
# Public registry surface
# ---------------------------------------------------------------------------


def list_mask_sources() -> list[str]:
    """Return every registered mask source selection name.

    Returns
    -------
    list[str]
        ``["gsw", "water", "worldcover"]``: the ``water`` auto alias plus the
        two wired bare provider names.  The opt-in ``osm-overpass`` provider
        stays reachable through :data:`MASK_PRODUCT_DEFAULTS` (unwired) and is
        rejected by :func:`parse_mask_selection` until it is wired.

    """
    return sorted([AUTO_SOURCE_NAME, *list(_REGISTRY)])


def get_mask_source(name: str) -> MaskSource:
    """Return the registered mask source for ``name``, failing closed.

    Accepts registry names (bare providers), the ``water`` auto alias,
    ``<product>:<provider>`` selections, and bare-product aliases (resolved
    to the wired default provider).

    Parameters
    ----------
    name : str
        Registry or selection name.

    Returns
    -------
    MaskSource
        The registered source entry.

    Raises
    ------
    ValueError
        If the name is unknown; the error lists all valid names.

    """
    if name == AUTO_SOURCE_NAME:
        return parse_mask_selection(DEFAULT_PRODUCT)
    if name in _REGISTRY:
        return _REGISTRY[name]
    if ":" in name and name.count(":") == 1:
        return parse_mask_selection(name)
    message = f"unknown mask source {name!r}; valid sources are: " + ", ".join(
        list_mask_sources()
    )
    logger.error(message)
    raise ValueError(message) from None
