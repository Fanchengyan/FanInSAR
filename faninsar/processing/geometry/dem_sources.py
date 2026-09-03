"""Multi-source DEM registry (PROPOSAL-0030): products, providers, planning.

The registry separates a **product group** (what the raster is: resolution,
vertical datum, derived flag) from a **provider group** (where the bytes live:
base URL, auth class, access shape).  Every entry's ``name`` is the **bare
product name** (``"glo30"``, ``"nasadem"``, ...); the provider dimension lives
in :attr:`DemSource.provider` and in the selection grammar ``"<product>"`` /
``"<product>:<provider>"``.  Each wired entry plans its fetch as one
self-describing :class:`~faninsar.processing.geometry.dem_transport.FetchPlan`
without touching the network — transport and mosaic behavior are owned by the
execution layer and the manager respectively.

Zero network by contract: import, :func:`list_dem_sources`,
:func:`get_dem_source`, :func:`dem_catalog`, and ``plan()`` perform no socket
I/O.
"""

from __future__ import annotations

import fnmatch
import math
import re
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.query import BoundingBox

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from faninsar.processing.geometry.dem_transport import FetchPlan

logger = setup_logger(__name__)

__all__ = [
    "AUTO_SOURCE_NAME",
    "AuthenticatedGranuleSource",
    "DeferredStacPlan",
    "DemSource",
    "DemSourceUnavailableError",
    "FtpZipSource",
    "LatLonGridSource",
    "MosaicRecipe",
    "PcStacSource",
    "PgcQuadSource",
    "QuadEnumerator",
    "RoiClipSource",
    "TerrainPyramidSource",
    "TileSet",
    "dem_catalog",
    "get_dem_source",
    "list_dem_sources",
    "parse_selection",
]

VerticalDatum = Literal["egm2008", "egm96", "mixed-derived", "ellipsoidal"]
AuthClass = Literal["none", "token"]
ProviderName = Literal["aws", "pc", "earthdata", "jaxa-ftp"]

AUTO_SOURCE_NAME = "auto"
DEFAULT_PRODUCT = "glo30"
DEM_SOURCE_ENV = "FANINSAR_DEM_SOURCE"

COPERNICUS_GLO30_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
COPERNICUS_GLO90_BASE = "https://copernicus-dem-90m.s3.amazonaws.com"
TERRAIN_TILES_BASE = "https://elevation-tiles-prod.s3.amazonaws.com"
PGC_OPEN_DATA_BASE = "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com"
JAXA_AW3D30_FTP_BASE = "ftp://ftp.eorc.jaxa.jp/pub/ALOS/ext1/AW3D30/release_v2303"
PC_STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
CMR_API = "https://cmr.earthdata.nasa.gov/search/granules.json"

#: Minimum valid size in bytes for a Copernicus COG tile (~1 MiB).
GLO_MIN_TILE_BYTES = 1 << 20
#: skadi tiles are gzip-compressed SRTM 1-degree grids; even an all-void tile
#: stays in the tens of KB once compressed.
SKADI_MIN_TILE_BYTES = 1024
#: Expected decompressed size of one 1-arc-second HGT grid (3601^2 * int16).
SKADI_EXPECTED_DECOMPRESSED_BYTES = 3601 * 3601 * 2
#: AWS terrain-tiles z12 geotiff tiles are small but never tiny.
TERRAIN_Z12_MIN_TILE_BYTES = 4096
#: PGC quad GeoTIFFs at 32 m run ~1-12 MB per tile.
PGC_QUAD_32M_MIN_BYTES = 1 << 20
#: NASADEM granule zips are ~14 MB; floor well below to avoid false rejections.
NASADEM_GRANULE_MIN_BYTES = 4 << 20

BoundsLike = BoundingBox | tuple[float, float, float, float]


class DemSourceUnavailableError(InvalidProcessingStateError):
    """A registry source cannot serve the requested bounds or environment."""


@dataclass(frozen=True, slots=True)
class DeferredStacPlan:
    """Immutable, zero-network description of a deferred STAC query.

    A registry ``plan()`` call must describe work without opening a socket.
    This record carries the complete admission identity needed by the
    materializer; STAC discovery, asset signing, and byte fetching happen
    only when that materializer explicitly resolves the plan.

    Parameters
    ----------
    endpoint_identity : str
        Canonical HTTPS STAC endpoint identity without credentials, query, or
        fragment components.
    collection : str
        Authoritative STAC collection identifier.
    asset : str
        Authoritative STAC asset key.
    bounds : tuple of float
        Original WGS84 request as ``(west, south, east, north)``.
    windows : tuple of tuple of float
        One or two deterministic query windows, split at the antimeridian
        when necessary.
    provider : str
        Provider identity retained for provenance and cache partitioning.
    product : str
        Product identity retained for provenance and cache partitioning.
    vertical_datum : str
        Registry-owned source vertical datum.
    allowed_hosts : tuple of str
        Hosts permitted for the eventual STAC discovery request.

    """

    endpoint_identity: str
    collection: str
    asset: str
    bounds: tuple[float, float, float, float]
    windows: tuple[tuple[float, float, float, float], ...]
    provider: str
    product: str
    vertical_datum: VerticalDatum
    allowed_hosts: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the descriptor without performing I/O."""
        parts = urllib.parse.urlsplit(self.endpoint_identity)
        if parts.scheme.lower() != "https" or not parts.hostname:
            message = "deferred STAC endpoint must be an HTTPS origin"
            logger.error(message)
            raise ValueError(message)
        if parts.username or parts.password or parts.query or parts.fragment:
            message = "deferred STAC endpoint must not contain credentials or query"
            logger.error(message)
            raise ValueError(message)
        if tuple(self.allowed_hosts) != (parts.hostname,):
            message = "deferred STAC endpoint host does not match allowlist"
            logger.error(message)
            raise ValueError(message)
        if len(self.bounds) != 4 or not all(np.isfinite(self.bounds)):
            message = "deferred STAC bounds must be finite"
            logger.error(message)
            raise ValueError(message)
        if not self.windows:
            message = "deferred STAC plan requires at least one query window"
            logger.error(message)
            raise ValueError(message)


def _stac_endpoint_identity(url: str) -> tuple[str, tuple[str, ...]]:
    """Return a credential-free endpoint identity and its host allowlist."""
    parts = urllib.parse.urlsplit(str(url))
    if parts.scheme.lower() != "https" or not parts.hostname:
        message = "STAC endpoint must be an HTTPS URL with a host"
        logger.error(message)
        raise ValueError(message)
    if parts.username or parts.password or parts.query or parts.fragment:
        message = "STAC endpoint must not contain credentials or query"
        logger.error(message)
        raise ValueError(message)
    host = parts.hostname.lower()
    port = parts.port
    netloc = host if port in (None, 443) else f"{host}:{port}"
    path = parts.path.rstrip("/")
    return urllib.parse.urlunsplit(("https", netloc, path, "", "")), (host,)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _bounds_tuple(bounds: BoundsLike) -> tuple[float, float, float, float]:
    """Normalize a BoundingBox or tuple to (lon_min, lat_min, lon_max, lat_max)."""
    if isinstance(bounds, BoundingBox):
        return (
            float(bounds.left),
            float(bounds.bottom),
            float(bounds.right),
            float(bounds.top),
        )
    min_lon, min_lat, max_lon, max_lat = bounds
    return float(min_lon), float(min_lat), float(max_lon), float(max_lat)


def validate_cache_relative_path(relative_path: str, cache_dir: object) -> None:
    """Reject cache-relative paths with traversal or absolute components.

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
    del cache_dir
    candidate = relative_path.replace("\\\\", "/")
    if candidate.startswith("/"):
        message = f"cache-relative path must not be absolute: {relative_path!r}"
        logger.error(message)
        raise ValueError(message)
    parts = [part for part in candidate.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        message = f"cache-relative path must not contain '..': {relative_path!r}"
        logger.error(message)
        raise ValueError(message)


_SOURCE_NAME_RE = re.compile(r"[a-z0-9]+([-_.@][a-z0-9]+)*")


def _validate_source_name(name: str) -> None:
    """Validate a registry name charset (names flow into partition dirs)."""
    if not name or not _SOURCE_NAME_RE.fullmatch(name):
        message = (
            f"invalid DEM source name {name!r}: names must match the "
            "[a-z0-9]+([-_.@][a-z0-9]+)* charset (they flow into cache "
            "partition directories)"
        )
        logger.error(message)
        raise ValueError(message)


def _cell_tag(latitude_deg: float, longitude_deg: float) -> str:
    """Return ``N{lat}_E{lon}`` style tag for an integer degree cell."""
    lat_tile = int(np.floor(latitude_deg))
    lon_tile = int(np.floor(longitude_deg))
    ns = "N" if lat_tile >= 0 else "S"
    ew = "E" if lon_tile >= 0 else "W"
    return f"{ns}{abs(lat_tile):02d}_{ew}{abs(lon_tile):03d}"


def _iter_degree_cells(
    bounds: BoundsLike,
) -> list[tuple[int, int]]:
    """Return latitude-major integer degree cells intersecting the bounds."""
    min_lon, min_lat, max_lon, max_lat = _bounds_tuple(bounds)
    return [
        (lat_cell, lon_cell)
        for lat_cell in range(int(np.floor(min_lat)), int(np.floor(max_lat)) + 1)
        for lon_cell in range(int(np.floor(min_lon)), int(np.floor(max_lon)) + 1)
    ]


def _slippy_xyz(
    longitude_deg: float,
    latitude_deg: float,
    zoom: int,
) -> tuple[int, int]:
    """Return slippy-map XYZ indices (y measured from north) for a coordinate."""
    n_tiles = 2**zoom
    x_tile = int((longitude_deg + 180.0) / 360.0 * n_tiles)
    clamped = min(max(latitude_deg, -85.05112878), 85.05112878)
    lat_rad = math.radians(clamped)
    y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n_tiles)
    return x_tile, y_tile


# ---------------------------------------------------------------------------
# Mosaic recipe
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MosaicRecipe:
    """Structured mosaic-side behavior driven entirely by registry data.

    Attributes
    ----------
    gdal_open
        GDAL open-prefix template with ``{path}`` (and optionally
        ``{member}``) placeholders, e.g. ``/vsigzip/{path}``.
    source_crs
        Input raster CRS authority string; None means already EPSG:4326.
    warp_target
        ``"none"`` keeps the source grid; ``"epsg4326"`` warps via WarpedVRT.
    resampling
        Resampling used when reprojection/resolution normalization applies.
    nodata
        Declared or hardcoded nodata value masked to NaN; None uses NaN mask.
    mask_to_nan
        Whether void/nodata pixels are converted to NaN before merging.

    """

    gdal_open: str = "{path}"
    source_crs: str | None = None
    warp_target: Literal["none", "epsg4326"] = "none"
    resampling: Literal["bilinear", "cubic"] = "bilinear"
    nodata: float | None = None
    mask_to_nan: bool = True


#: PGC quads ship EPSG:3413/3031 with a hardcoded -9999 void value; mosaic
#: warps onto EPSG:4326 and masks to NaN.
_PGC_MOSAIC_RECIPE = MosaicRecipe(
    gdal_open="/vsicurl/{path}",
    warp_target="epsg4326",
    resampling="bilinear",
    nodata=-9999.0,
)

#: NISAR Mission Modified Copernicus assets are native WGS84 COGs.
_NISAR_MOSAIC_RECIPE = MosaicRecipe(
    gdal_open="{path}",
    source_crs="EPSG:4326",
    warp_target="none",
)


# ---------------------------------------------------------------------------
# QuadEnumerator protocol (PgcQuadSource seam)
# ---------------------------------------------------------------------------


class QuadEnumerator:
    """Protocol object enumerating PGC quad keys for a source.

    Subclasses may implement either :meth:`quads_for_bounds` (semantic
    enumeration) or only :meth:`list_quads` (raw paged listing); sources
    fall back to the raw listing and reconcile it against the derived
    expected-quad set.
    """

    def quads_for_bounds(self, bounds: BoundsLike) -> list[str]:
        """Return quad ids covering ``bounds``."""
        raise NotImplementedError

    def list_quads(self, prefix: str) -> tuple[list[str], bool]:
        """Return (quad ids under prefix, truncated?); raw paged listing."""
        raise NotImplementedError


class S3ListQuadEnumerator(QuadEnumerator):
    """Anonymous S3 XML listing implementation (v1 default).

    Loops over continuation tokens until ``IsTruncated`` is false; callers
    reconcile the result against the independently derived expected set so
    silent quad loss is impossible.
    """

    def __init__(self, bucket_base_url: str) -> None:
        self.bucket_base_url = bucket_base_url.rstrip("/")

    def list_quads(self, prefix: str) -> tuple[list[str], bool]:
        """List quad directory ids under ``prefix`` until exhaustion."""
        quad_re = re.compile(r"(\d\d_\d\d[ns]?)/")
        token: str | None = None
        found: set[str] = set()
        while True:
            query = f"?list-type=2&prefix={urllib.parse.quote(prefix)}&delimiter=/"
            if token is not None:
                query += "&continuation-token=" + urllib.parse.quote(token)
            request = urllib.request.Request(self.bucket_base_url + "/" + query)
            with urllib.request.urlopen(request, timeout=120) as response:
                xml = response.read().decode("utf-8", errors="replace")
            found.update(quad_re.findall(xml))
            if "<IsTruncated>false</IsTruncated>" in xml or (
                "<IsTruncated>true</IsTruncated>" not in xml
            ):
                break
            match = re.search(
                r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml
            )
            if match is None:
                break
            token = match.group(1)
        return sorted(found), False


#: Test/strategy seam: source name -> injected :class:`QuadEnumerator`.
QUAD_ENUMERATOR_OVERRIDES: dict[str, QuadEnumerator] = {}


# ---------------------------------------------------------------------------
# DemSource ABC + shape subclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DemSource(ABC):
    """Registry entry describing one selectable product/provider pair.

    Parameters
    ----------
    name
        Registry selection name (``product`` or auto alias).
    description
        Human-readable summary surfaced by :func:`dem_catalog`.
    product
        Product-group identifier (e.g. ``glo30``); equals ``name`` except
        for the ``auto`` alias.
    provider
        Provider-group identifier (e.g. ``aws``, ``pc``, ``earthdata``).
    wired
        Whether this pair is actually reachable in v1; unwired pairs fail
        closed at selection time.
    resolution_m
        Nominal ground sampling distance in degrees per pixel.
    vertical_datum
        Datum metadata driving the pipeline wrap rule.
    derived
        True for merged/derived products; feeds geometry warnings.

    """

    # selection level
    name: str
    description: str
    # product group
    product: str = DEFAULT_PRODUCT
    provider: str = "aws"
    resolution_m: float = 1.0 / 3600
    vertical_datum: VerticalDatum = "egm2008"
    derived: bool = False
    auth: AuthClass = "none"
    wired: bool = True
    #: Structured mosaic recipe override; None defers to ``mosaic_recipe()``.
    mosaic_recipe_override: MosaicRecipe | None = None

    def __post_init__(self) -> None:
        """Validate identity invariants at construction (fail closed)."""
        _validate_source_name(self.name)

    @abstractmethod
    def plan(self, bounds: BoundsLike) -> FetchPlan:
        """Return the self-describing fetch plan for ``bounds``.

        No I/O here beyond optional read-only remote discovery delegated to
        injected enumerator strategies.
        """

    def coverage(self, bounds: BoundsLike) -> str | None:
        """Return None inside coverage or a fail-closed explanation.

        Parameters
        ----------
        bounds : BoundingBox or tuple
            Requested geographic bounds.

        Returns
        -------
        str or None
            None when fully covered, otherwise a human-readable failure
            message naming the uncovered latitude band.

        """
        del bounds
        return None

    def mosaic_recipe(self) -> MosaicRecipe:
        """Return the structured mosaic recipe (default: plain merge)."""
        if self.mosaic_recipe_override is not None:
            return self.mosaic_recipe_override
        return MosaicRecipe()


@dataclass(frozen=True, slots=True)
class LatLonGridSource(DemSource):
    """1-degree-grid stem-template sources (glo30, glo90, srtm-skadi)."""

    base_url: str = COPERNICUS_GLO30_BASE
    #: ``"copernicus"`` renders ``COG_10_{cell}`` stems; ``"skadi"`` selects
    #: the terrain-tiles HGT layout (``skadi/{N|S}YY/{N|S}YY{E|W}XXX.hgt.gz``).
    remote_layout: Literal["copernicus", "skadi"] = "copernicus"
    stem_prefix: str = "Copernicus_DSM_COG_10"
    suffix: str = ".tif"
    min_bytes: int = GLO_MIN_TILE_BYTES
    expected_decompressed_bytes: int | None = None
    ocean_404_skip: bool = False
    gdal_open_template: str = "{path}"

    def __post_init__(self) -> None:
        """Validate identity invariants at construction (fail closed)."""
        DemSource.__post_init__(self)  # zero-arg super() breaks frozen slots
        if self.remote_layout not in {"copernicus", "skadi"}:
            message = (
                f"invalid remote_layout {self.remote_layout!r}: expected "
                "'copernicus' or 'skadi'"
            )
            logger.error(message)
            raise ValueError(message)

    def plan(self, bounds: BoundsLike) -> FetchPlan:
        """Enumerate degree-cell tiles into a TileSet plan."""
        from faninsar.processing.geometry.dem_transport import Tile

        host = urllib.parse.urlsplit(self.base_url).hostname or ""
        allowed_hosts = (host,)
        tiles: list[Tile] = []
        for lat_cell, lon_cell in _iter_degree_cells(bounds):
            lat_mid = lat_cell + 0.5
            lon_mid = lon_cell + 0.5
            tag = _cell_tag(lat_mid, lon_mid)
            if self.remote_layout == "skadi":
                lat_part, lon_part = tag.split("_")
                hgt_name = f"{lat_part}{lon_part}.hgt.gz"
                relative = f"skadi/{lat_part}/{hgt_name}"
                url = f"{self.base_url}/skadi/{lat_part}/{hgt_name}"
                tiles.append(
                    Tile(
                        url=url,
                        cache_path=_safe_relative(relative),
                        min_bytes=SKADI_MIN_TILE_BYTES,
                        ranged=False,
                        expected_decompressed_bytes=(
                            self.expected_decompressed_bytes
                            or SKADI_EXPECTED_DECOMPRESSED_BYTES
                        ),
                        ocean_404_skip=self.ocean_404_skip,
                    )
                )
                continue
            stem = f"{self.stem_prefix}_{_copernicus_stem_fragment(tag)}_DEM"
            filename = f"{stem}{self.suffix}"
            relative = f"{tag}/{stem}/{filename}"
            url = f"{self.base_url}/{stem}/{filename}"
            tiles.append(
                Tile(
                    url=url,
                    cache_path=_safe_relative(relative),
                    min_bytes=self.min_bytes,
                    ranged=True,
                    ocean_404_skip=False,
                )
            )
        return TileSet(allowed_hosts=allowed_hosts, tiles=tuple(tiles))

    def mosaic_recipe(self) -> MosaicRecipe:
        """Plain file opens; skadi variants open through /vsigzip/."""
        if self.remote_layout == "skadi":
            return MosaicRecipe(gdal_open="/vsigzip/{path}")
        return MosaicRecipe()


def _copernicus_stem_fragment(tag: str) -> str:
    """Convert ``N38_E100`` into ``N38_00_E100_00``."""
    lat_part, lon_part = tag.split("_")
    return f"{lat_part}_00_{lon_part}_00"


def _safe_relative(relative: str) -> Path:
    """Build a cache-relative Path after the traversal guard."""
    validate_cache_relative_path(relative, None)
    from pathlib import Path as _Path

    return _Path(relative)


@dataclass(frozen=True, slots=True)
class PgcQuadSource(DemSource):
    """ArcticDEM / REMA mosaic quads from the PGC open-data bucket."""

    base_url: str = PGC_OPEN_DATA_BASE
    collection: str = "arcticdem"
    version: str = "v4.1"
    resolution_tag: str = "32m"
    epsg: int = 3413
    polar_band: tuple[float, float] | None = None
    quad_enumerator_factory: Callable[[str], QuadEnumerator] | None = None

    @property
    def quad_enumerator(self) -> QuadEnumerator:
        """Injected or default S3-listing enumerator strategy.

        The injection seam is the module-level override table
        :data:`QUAD_ENUMERATOR_OVERRIDES` keyed by source name; frozen
        dataclasses cannot carry mutable per-instance state.
        """
        override = QUAD_ENUMERATOR_OVERRIDES.get(self.name)
        if override is not None:
            return override
        return S3ListQuadEnumerator(self.base_url)

    # -- grid mapping ------------------------------------------------------

    #: North/south quad grids are 100 km cells in EPSG:3413/3031 with the
    #: column index along y and the row index along x (live-derived 2026-08-23
    #: against full bucket listings; see module notes).
    QUAD_GRID_ORIGIN_KM_NORTH = 4100.0
    QUAD_GRID_ORIGIN_KM_SOUTH = 3100.0
    QUAD_GRID_CELL_KM = 100.0

    def _grid_origin_km(self) -> float:
        return (
            self.QUAD_GRID_ORIGIN_KM_NORTH
            if self.epsg == 3413
            else self.QUAD_GRID_ORIGIN_KM_SOUTH
        )

    def bounds_to_quads(self, bounds: BoundsLike) -> list[str]:
        """Map EPSG:4326 bounds onto polar-stereo quad ids.

        Parameters
        ----------
        bounds : BoundingBox or tuple
            Requested geographic bounds in degrees.

        Returns
        -------
        list[str]
            Sorted quad ids like ``07_40`` covering the projected footprint.

        """
        from pyproj import CRS, Transformer

        transformer = Transformer.from_crs(
            CRS.from_epsg(4326),
            CRS.from_epsg(self.epsg),
            always_xy=True,
        )
        min_lon, min_lat, max_lon, max_lat = _bounds_tuple(bounds)
        origin_m = self._grid_origin_km() * 1000.0
        cell_m = self.QUAD_GRID_CELL_KM * 1000.0
        cells: set[tuple[int, int]] = set()
        steps = 24
        for i in range(steps + 1):
            for j in range(steps + 1):
                lon = min_lon + (max_lon - min_lon) * i / steps
                lat = min_lat + (max_lat - min_lat) * j / steps
                x_m, y_m = transformer.transform(lon, lat)
                col = math.floor((y_m + origin_m) / cell_m)
                row = math.floor((x_m + origin_m) / cell_m)
                cells.add((col, row))
        return [f"{col:02d}_{row:02d}" for col, row in sorted(cells)]

    def _expected_quads(self, bounds: BoundsLike) -> list[str]:
        """Derive the expected quad set with listing-completeness checks."""
        expected = self.bounds_to_quads(bounds)
        prefix = f"{self.collection}/mosaics/{self.version}/{self.resolution_tag}m/"
        listed, truncated = self.quad_enumerator.list_quads(prefix)
        if truncated:  # pragma: no cover - defensive; loop should exhaust
            message = f"quad listing for {prefix!r} reported truncation"
            logger.error(message)
            raise DemSourceUnavailableError(message)
        # Listing entries may be quad directory ids ("07_40/") or full file
        # keys (".../07_40/07_40_32m_v4.1_dem.tif"); normalize to quad ids.
        listed_set = {
            entry.rstrip("/").rsplit("/", 1)[-1].split("_2m")[0] for entry in listed
        }
        missing = sorted(set(expected) - listed_set)
        if missing:
            message = (
                f"quad listing gap for {self.name}: expected "
                f"{sorted(expected)} but listing missed {missing}; refusing "
                "silent quad loss"
            )
            logger.error(message)
            raise DemSourceUnavailableError(message)
        return expected

    def coverage(self, bounds: BoundsLike) -> str | None:
        """Fail closed outside the polar bands."""
        _, min_lat, _, max_lat = _bounds_tuple(bounds)
        if self.polar_band is None:
            return None
        south, north = self.polar_band
        if min_lat >= south and max_lat <= north:
            return None
        return (
            f"source {self.name} covers only polar latitudes "
            f"[{min(south, north)}, {max(south, north)}] (EPSG:{self.epsg});"
            f" requested [{min_lat}, {max_lat}] is outside the polar band"
        )

    def plan(self, bounds: BoundsLike) -> FetchPlan:
        """Plan quad downloads through the enumerator + reconciliation.

        Coverage gating is fail-closed: requests outside the declared polar
        band raise before any enumeration.
        """
        from faninsar.processing.geometry.dem_transport import TileSet

        message = self.coverage(bounds)
        if message is not None:
            logger.error(message)
            raise DemSourceUnavailableError(message)
        quads = self.quads_for_bounds(bounds)
        if not quads:
            message = (
                f"no {self.collection} quads intersect bounds {_bounds_tuple(bounds)}"
            )
            logger.error(message)
            raise DemSourceUnavailableError(message)
        host = urllib.parse.urlsplit(self.base_url).hostname or ""
        tiles = [self._quad_tile(quad) for quad in quads]
        return TileSet(allowed_hosts=(host,), tiles=tuple(tiles))

    def quads_for_bounds(self, bounds: BoundsLike) -> list[str]:
        """Resolve quads via the semantic API or reconciled listing."""
        enumerator = self.quad_enumerator
        try:
            return enumerator.quads_for_bounds(bounds)
        except NotImplementedError:
            return self._expected_quads(bounds)

    def _quad_tile(self, quad: str) -> Tile:
        """Build the Tile for one quad id at this tier's key layout."""
        prefix = f"{self.collection}/mosaics/{self.version}/{self.resolution_tag}m/"
        if self.resolution_tag == "2":
            # 2m quads ship a 2x2 row/col sub-tile grid inside each quad
            # directory (live-verified 2026-08-23 against the PGC bucket:
            # keys are {quad}_{r}_{c}_2m_{version}_dem.tif with r indexing
            # south-to-north and c west-to-east; absent land sub-tiles are
            # legitimately missing from the listing).
            tiles: list[Tile] = []
            for row in (1, 2):
                for col in (1, 2):
                    filename = f"{quad}_{row}_{col}_2m_{self.version}_dem.tif"
                    key = f"{prefix}{quad}/{filename}"
                    candidate = self._pgc_tile(key)
                    if candidate is not None:
                        tiles.append(candidate)
            if not tiles:
                message = (
                    f"no 2m sub-tiles found for {self.collection} quad "
                    f"{quad!r} under {prefix}"
                )
                logger.error(message)
                raise DemSourceUnavailableError(message)
            return _MultiTileTile(*tiles)
        key = f"{prefix}{quad}/{quad}_{self.resolution_tag}m_{self.version}_dem.tif"
        tile = self._pgc_tile(key)
        if tile is None:
            message = f"PGC quad key missing: {key!r}"
            logger.error(message)
            raise DemSourceUnavailableError(message)
        return tile

    def _pgc_tile(self, key: str) -> Tile | None:
        """Build one Tile for an exact bucket key, or None when absent.

        Absence is decided from the enumerator's listing when it reports
        file-level keys (2m sub-tile discovery); quad-id listings carry no
        per-file information, so the exact key is planned optimistically and
        the completeness invariant stays in :meth:`_expected_quads`.
        """
        filename = key.rsplit("/", 1)[-1]
        quad_dir = key.rsplit("/", 2)[-2]
        prefix = key.rsplit("/", 1)[0] + "/"
        listed, _truncated = self.quad_enumerator.list_quads(prefix)
        file_level = [entry for entry in listed if entry.endswith(".tif")]
        if file_level and filename not in {
            entry.rsplit("/", 1)[-1] for entry in file_level
        }:
            return None
        return Tile(
            url=f"{self.base_url}/{key}",
            cache_path=_safe_relative(
                f"{self.collection}-{self.version}-{self.resolution_tag}m/"
                f"{quad_dir}/{filename}"
            ),
            min_bytes=PGC_QUAD_32M_MIN_BYTES,
        )


def _pgc_setattr(self: PgcQuadSource, name: str, value: object) -> None:
    """Route ``quad_enumerator`` assignments to the injection seam.

    Frozen slotted dataclasses generate their own ``__setattr__``, so this
    override is attached after class creation: assigning
    ``source.quad_enumerator = strategy`` stores it in the module-level
    :data:`QUAD_ENUMERATOR_OVERRIDES` table keyed by the source name.
    """
    if name == "quad_enumerator":
        QUAD_ENUMERATOR_OVERRIDES[self.name] = value  # type: ignore[assignment]
        return
    object.__setattr__(self, name, value)


PgcQuadSource.__setattr__ = _pgc_setattr  # type: ignore[method-assign]


@dataclass(frozen=True, slots=True)
class TerrainPyramidSource(DemSource):
    """AWS terrain-tiles merged pyramid (pinned z12 XYZ layout)."""

    base_url: str = TERRAIN_TILES_BASE
    zoom: int = 12
    min_bytes: int = TERRAIN_Z12_MIN_TILE_BYTES
    nodata_value: float = -32768.0

    def plan(self, bounds: BoundsLike) -> FetchPlan:
        """Enumerate XYZ geotiff tiles covering the bounds."""
        from faninsar.processing.geometry.dem_transport import Tile, TileSet

        min_lon, min_lat, max_lon, max_lat = _bounds_tuple(bounds)
        n_tiles = 2**self.zoom
        x_min, y_top = _slippy_xyz(min_lon, max_lat, self.zoom)
        x_max, y_bottom = _slippy_xyz(max_lon, min_lat, self.zoom)
        x_min = max(0, x_min)
        y_top = max(0, y_top)
        x_max = min(n_tiles - 1, x_max)
        y_bottom = min(n_tiles - 1, y_bottom)
        host = urllib.parse.urlsplit(self.base_url).hostname or ""
        tiles = [
            Tile(
                url=(f"{self.base_url}/geotiff/{self.zoom}/{x}/{y}.tif"),
                cache_path=_safe_relative(f"geotiff/{self.zoom}/{x}/{y}.tif"),
                min_bytes=self.min_bytes,
                ranged=False,
            )
            for y in range(y_top, y_bottom + 1)
            for x in range(x_min, x_max + 1)
        ]
        return TileSet(allowed_hosts=(host,), tiles=tuple(tiles))

    def mosaic_recipe(self) -> MosaicRecipe:
        """Tiles open through /vsicurl/ and warp onto EPSG:4326."""
        return MosaicRecipe(
            gdal_open="/vsicurl/{path}",
            warp_target="epsg4326",
            resampling="bilinear",
            nodata=self.nodata_value,
        )


@dataclass(frozen=True, slots=True)
class FtpZipSource(DemSource):
    """AW3D30 legacy JAXA FTP channel (5-degree zip blocks)."""

    ftp_base: str = JAXA_AW3D30_FTP_BASE
    member_pattern: str = "ALPSMLC30_*_DSM.tif"

    @staticmethod
    def block_name(min_lon: float, min_lat: float) -> tuple[str, str]:
        """Return (block zip stem, member cell tag) for a SW corner.

        The JAXA AW3D30 v2303 layout is latitude-major: the zip stem starts
        with the lower-latitude corner tag (``N000E005``) followed by the
        upper one (``N005E010``) — live-verified naming.
        """
        lat0 = int(math.floor(min_lat / 5.0) * 5)
        lon0 = int(math.floor(min_lon / 5.0) * 5)
        lat1 = lat0 + 5
        lon1 = lon0 + 5

        def fmt(lat: int, lon: int) -> str:
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            return f"{ns}{abs(lat):03d}{ew}{abs(lon):03d}"

        return (
            f"{fmt(lat0, lon0)}_{fmt(lat1, lon1)}",
            fmt(lat0, lon0),
        )

    def plan(self, bounds: BoundsLike) -> FetchPlan:
        """Plan the 5-degree FTP zip Artifact(s) covering the bounds."""
        from faninsar.processing.geometry.dem_transport import Artifact

        min_lon, min_lat, max_lon, max_lat = _bounds_tuple(bounds)
        host = urllib.parse.urlsplit(self.ftp_base).hostname or ""
        artifacts: list[Artifact] = []
        blocks: set[tuple[int, int]] = set()
        lat_blocks = range(
            math.floor(min_lat / 5.0),
            math.floor(max_lat / 5.0) + 1,
        )
        lon_blocks = range(
            math.floor(min_lon / 5.0),
            math.floor(max_lon / 5.0) + 1,
        )
        for lat_block in lat_blocks:
            for lon_block in lon_blocks:
                blocks.add((lat_block * 5, lon_block * 5))
        for lat0, lon0 in sorted(blocks):
            stem, _ = self.block_name(lon0, lat0)
            url = f"{self.ftp_base}/{stem}.zip"
            artifacts.append(
                Artifact(
                    allowed_hosts=(host,),
                    scheme="ftp",
                    url=url,
                    members=None,
                    member_pattern=self.member_pattern,
                    expand="zip",
                    cache_path=_safe_relative(f"aw3d30-v2303/{stem}.zip"),
                    min_total_bytes=1 << 20,
                )
            )
        if len(artifacts) == 1:
            return artifacts[0]
        return _MultiArtifactPlan(allowed_hosts=(host,), artifacts=tuple(artifacts))


from faninsar.processing.geometry.dem_transport import (  # noqa: E402
    Artifact,
    FetchPlan,
    Tile,
    TileSet,
)


@dataclass(frozen=True, slots=True)
class _MultiArtifactPlan(FetchPlan):
    """Internal multi-artifact wrapper (executed sequentially by the engine).

    The dataclass decorator is load-bearing: without it the inherited
    ``FetchPlan.__init__`` rejects the ``artifacts`` keyword, so plans
    spanning more than one 5-degree JAXA block crash with a TypeError at
    construction.
    """

    artifacts: tuple[Artifact, ...] = ()


@dataclass(frozen=True, slots=True)
class _MultiTileTile(Tile):
    """Internal fan-out of one logical quad into its concrete sub-tiles.

    Carries every sub-tile so the engine fetches each independently while
    callers treat it as one plan entry; ``url`` / ``cache_path`` mirror the
    first sub-tile.
    """

    parts: tuple[Tile, ...] = ()

    def __init__(self, *parts: Tile) -> None:
        if not parts:
            message = "_MultiTileTile requires at least one sub-tile"
            logger.error(message)
            raise ValueError(message)
        # Zero-arg super() breaks inside frozen+slots dataclass __init__;
        # the extra ``parts`` slot is set via object.__setattr__ after the
        # base field initialization.
        Tile.__init__(
            self,
            url=parts[0].url,
            cache_path=parts[0].cache_path,
            min_bytes=min(tile.min_bytes for tile in parts),
            ranged=all(tile.ranged for tile in parts),
            expected_decompressed_bytes=None,
            ocean_404_skip=False,
        )
        object.__setattr__(self, "parts", tuple(parts))


@dataclass(frozen=True, slots=True)
class AuthenticatedGranuleSource(DemSource):
    """CMR-discovered, Earthdata-authenticated granule zips.

    Covers ``nasadem@earthdata`` and ``nisar-glo30@earthdata``.
    """

    cmr_collection: str = "C2763264762-LPCLOUD"
    data_host: str = "data.lpdaac.earthdatacloud.nasa.gov"
    asset_pattern: str = "*.zip"
    title_filter_required: bool = False

    def coverage(self, bounds: BoundsLike) -> str | None:
        """Fail closed outside the product's latitude band."""
        _, min_lat, _, max_lat = _bounds_tuple(bounds)
        if self.cmr_collection == "C3803703055-ASF":
            return None  # global via EPSG 4326/3413/3031 items
        south, north = -56.0, 60.0
        if min_lat >= south and max_lat <= north:
            return None
        return (
            f"source {self.name} does not cover latitude range "
            f"[{min_lat}, {max_lat}]; covered range is [{south}, {north}]"
        )

    def plan(self, bounds: BoundsLike) -> FetchPlan:
        """Query CMR anonymously then build the authenticated TileSet.

        The plan always carries ``credential_ref="earthdata"``; the engine
        resolves ``.netrc`` first then ``EARTHDATA_TOKEN`` and fails closed
        with registration guidance when neither is present (planning itself
        stays zero-network for the credential gate).  Granule GETs are
        executed whole-file — ranged is excluded for Earthdata hosts by the
        engine.
        """
        from faninsar.processing.geometry.dem_transport import Tile, TileSet
        from faninsar.processing.geometry.dem_transport import (
            resolve_credentials as _resolve_credentials,
        )

        # Fail closed BEFORE any network work when credentials are absent.
        try:
            _resolve_credentials("earthdata")
        except RuntimeError as exc:
            message = str(exc)
            logger.exception(message)
            raise DemSourceUnavailableError(message) from None
        min_lon, min_lat, max_lon, max_lat = _bounds_tuple(bounds)
        granules = self._cmr_granules(min_lon, min_lat, max_lon, max_lat)
        if not granules:
            message = (
                f"no CMR granules in collection {self.cmr_collection} for "
                f"bounds {(min_lon, min_lat, max_lon, max_lat)}"
            )
            logger.error(message)
            raise DemSourceUnavailableError(message)
        tiles: list[Tile] = []
        for granule in granules:
            for url in self._granule_asset_urls(granule):
                tail = urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]
                tiles.append(
                    Tile(
                        url=url,
                        cache_path=_safe_relative(
                            f"{self.product}-{self.provider}/granules/{tail}"
                        ),
                        min_bytes=NASADEM_GRANULE_MIN_BYTES,
                        ranged=False,
                    )
                )
        return TileSet(
            allowed_hosts=(self.data_host,),
            tiles=tuple(tiles),
            credential_ref="earthdata",
        )

    def _cmr_granules(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
    ) -> list[dict]:
        """Run the anonymous CMR bounding-box search (read-only https)."""
        params = urllib.parse.urlencode(
            {
                "collection_concept_id": self.cmr_collection,
                "bounding_box": f"{min_lon},{min_lat},{max_lon},{max_lat}",
                "page_size": 200,
            }
        )
        request = urllib.request.Request(f"{CMR_API}?{params}")
        with urllib.request.urlopen(request, timeout=120) as response:
            import json

            payload = json.loads(response.read().decode("utf-8"))
        entries = payload.get("feed", {}).get("entry", [])
        if self.title_filter_required:
            entries = [
                entry
                for entry in entries
                if "EPSG4326" in str(entry.get("title", "")).upper()
                and "-vrt" not in str(entry.get("title", ""))
            ]
        return entries

    def _granule_asset_urls(self, granule: dict) -> list[str]:
        """Extract data-file URLs matching the asset pattern."""
        urls: list[str] = []
        for link in granule.get("links", []):
            href = str(link.get("href", ""))
            rel = str(link.get("rel", ""))
            path = urllib.parse.urlsplit(href).path
            if rel.endswith("#data") and fnmatch.fnmatch(
                path, f"*{self.asset_pattern}"
            ):
                urls.append(href)
        return urls

    def mosaic_recipe(self) -> MosaicRecipe:
        """Return the source-native archive or COG opening recipe."""
        if self.asset_pattern.lower().endswith(".zip"):
            return MosaicRecipe(gdal_open="/vsizip/{path}/{member}")
        return MosaicRecipe(gdal_open="{path}", source_crs="EPSG:4326")


@dataclass(frozen=True, slots=True)
class RoiClipSource(DemSource):
    """OpenTopography ROI-clip shape — reserved, not wired in v1."""

    api_base: str = "https://portal.opentopography.org/API/globaldem"

    def plan(self, bounds: BoundsLike) -> FetchPlan:
        """Reserved: one bounded-ROI GET returning a clipped GeoTIFF."""
        del bounds
        message = (
            "OpenTopography ROI clipping is reserved (unwired in v1); "
            "select a wired provider instead"
        )
        logger.error(message)
        raise DemSourceUnavailableError(message)


@dataclass(frozen=True, slots=True)
class PcStacSource(DemSource):
    """Microsoft Planetary Computer STAC provider (anonymous SAS signing)."""

    stac_api_url: str = PC_STAC_API
    collection_id: str = "cop-dem-glo-30"
    asset_key: str = "dem"
    item_grid_degrees: float = 1.0

    def plan(self, bounds: BoundsLike) -> DeferredStacPlan:
        """Return an immutable STAC query descriptor without network I/O.

        STAC search, anonymous asset signing, and remote asset enumeration are
        deliberately deferred until :meth:`discover` is called by a
        materializer.  This keeps construction, registry listing, catalog
        inspection, and planning safe for dry-run and offline workflows.
        """
        raw_bounds = _bounds_tuple(bounds)
        endpoint, allowed_hosts = _stac_endpoint_identity(self.stac_api_url)
        west, south, east, north = raw_bounds
        if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
            message = "STAC bounds longitude must be within [-180, 180]"
            logger.error(message)
            raise ValueError(message)
        if south > north or not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
            message = "STAC bounds latitude must be ordered within [-90, 90]"
            logger.error(message)
            raise ValueError(message)
        if west > east:
            windows = ((west, south, 180.0, north), (-180.0, south, east, north))
        else:
            windows = (raw_bounds,)
        return DeferredStacPlan(
            endpoint_identity=endpoint,
            collection=self.collection_id,
            asset=self.asset_key,
            bounds=raw_bounds,
            windows=windows,
            provider=self.provider,
            product=self.product,
            vertical_datum=self.vertical_datum,
            allowed_hosts=allowed_hosts,
        )

    def discover(self, plan: DeferredStacPlan) -> FetchPlan:
        """Resolve a deferred plan through STAC and sign the returned assets.

        This is the provider I/O boundary.  Callers must pass a descriptor
        produced by this source's :meth:`plan`; all returned cache paths omit
        ephemeral SAS query parameters.
        """
        from faninsar.processing.geometry.dem_transport import Tile, TileSet

        stack = _import_pc_stack()
        if stack is None:
            message = (
                "provider 'pc' requires the Planetary Computer extras: "
                "pip install 'FanInSAR[pc]' (planetary-computer + "
                "pystac-client)"
            )
            logger.error(message)
            raise DemSourceUnavailableError(message)
        if not isinstance(plan, DeferredStacPlan):
            message = "PcStacSource.discover requires a DeferredStacPlan"
            logger.error(message)
            raise TypeError(message)
        endpoint, allowed_hosts = _stac_endpoint_identity(self.stac_api_url)
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
            raise ValueError(message)
        planetary_computer, stac_client = stack
        catalog = stac_client.Client.open(
            plan.endpoint_identity, modifier=planetary_computer.sign_inplace
        )
        tiles: list[Tile] = []
        seen_hrefs: set[str] = set()
        for window in plan.windows:
            search_kwargs = {
                "collections": [plan.collection],
                "bbox": list(window),
            }
            try:
                collection = catalog.get_collection(plan.collection)
                search = collection.search(**search_kwargs)
            except AttributeError:
                search = catalog.search(**search_kwargs)
            for item in search.items():
                # Sign in place BEFORE reading the href so the URL carries a
                # fresh anonymous SAS token.
                planetary_computer.sign_inplace(item)
                asset = item.assets.get(plan.asset)
                if asset is None:
                    continue
                signed = str(asset.href)
                unsigned = signed.split("?", 1)[0]
                if unsigned in seen_hrefs:
                    continue
                seen_hrefs.add(unsigned)
                tail = urllib.parse.urlsplit(unsigned).path.rsplit("/", 1)[-1]
                tiles.append(
                    Tile(
                        url=signed,
                        cache_path=_safe_relative(f"{plan.collection}/{tail}"),
                        min_bytes=GLO_MIN_TILE_BYTES,
                        ranged=True,
                    )
                )
        if not tiles:
            message = (
                f"no STAC items with asset {plan.asset!r} in collection "
                f"{plan.collection!r} for bounds {plan.bounds}"
            )
            logger.error(message)
            raise DemSourceUnavailableError(message)
        hosts = {
            urllib.parse.urlsplit(tile.url.split("?", 1)[0]).hostname or ""
            for tile in tiles
        }
        return TileSet(allowed_hosts=tuple(sorted(hosts)), tiles=tuple(tiles))


def _import_pc_stack() -> tuple | None:
    """Import planetary_computer + pystac_client; None when absent."""
    try:
        import planetary_computer
        import pystac_client
    except ImportError:
        return None
    return planetary_computer, pystac_client


# ---------------------------------------------------------------------------
# Selection grammar
# ---------------------------------------------------------------------------


def parse_selection(selection: str) -> DemSource:
    """Resolve ``"<product>"`` / ``"<product>:<provider>"`` to a source.

    Fail-closed on unknown products/providers, registered-but-unwired pairs,
    and hostile payloads (traversal fragments, trailing colons, homoglyphs,
    surrounding whitespace, extra segments).

    Parameters
    ----------
    selection : str
        Selection expression such as ``glo30`` or ``glo30:aws``.

    Returns
    -------
    DemSource
        The resolved registry entry.

    Raises
    ------
    ValueError
        When any grammar component fails exact-match validation.

    """
    if not isinstance(selection, str):
        message = f"DEM selection must be a string, got {type(selection)!r}"
        raise TypeError(message)
    parts = selection.split(":")
    if len(parts) > 2:
        message = (
            f"invalid DEM selection {selection!r}: at most one ':' separator "
            "is allowed ('<product>' or '<product>:<provider>')"
        )
        logger.error(message)
        raise ValueError(message)
    product = parts[0]
    provider = parts[1] if len(parts) == 2 else None
    if product not in PRODUCT_DEFAULTS:
        valid_products = ", ".join(sorted(PRODUCT_DEFAULTS))
        message = (
            f"unknown DEM product {product!r}; valid products are: {valid_products}"
        )
        logger.error(message)
        raise ValueError(message)
    providers = PRODUCT_DEFAULTS[product]["providers"]
    default_provider = PRODUCT_DEFAULTS[product]["default"]
    if provider is None:
        return get_dem_source(PRODUCT_REGISTRY[(product, default_provider)])
    if provider not in providers:
        message = (
            f"unknown DEM provider {provider!r} for product {product!r}; "
            f"valid providers are: {', '.join(sorted(providers))}"
        )
        logger.error(message)
        raise ValueError(message)
    pair = (product, provider)
    if pair not in PRODUCT_REGISTRY:
        message = (
            f"provider {provider!r} is not wired for product {product!r} "
            "(registered but unavailable in this release)"
        )
        logger.error(message)
        raise ValueError(message)
    return get_dem_source(PRODUCT_REGISTRY[pair])


# ---------------------------------------------------------------------------
# Registry construction
# ---------------------------------------------------------------------------

_LATLON_SPECS: dict[str, dict] = {
    "glo30": {
        "description": "Copernicus GLO-30 30m DSM (EGM2008)",
        "base_url": COPERNICUS_GLO30_BASE,
        "stem_prefix": "Copernicus_DSM_COG_10",
        "resolution_m": 1.0 / 3600,
        "vertical_datum": "egm2008",
    },
    "glo90": {
        "description": "Copernicus GLO-90 90m DSM (EGM2008)",
        "base_url": COPERNICUS_GLO90_BASE,
        "stem_prefix": "Copernicus_DSM_COG_30",
        "resolution_m": 1.0 / 1200,
        "vertical_datum": "egm2008",
    },
}

_SKADI_SPEC = {
    "description": "SRTM 1-arc-second equivalent (.hgt.gz) from AWS terrain-tiles",
}


def _build_registry() -> dict[str, DemSource]:
    """Construct all thirteen v1 entries (pure, no I/O).

    Registry keys and entry ``name`` fields are **bare product names**; the
    provider dimension lives on :attr:`DemSource.provider` (one wired entry
    per product in v1) and in the selection grammar.
    """
    registry: dict[str, DemSource] = {}

    # -- glo30 / glo90 (LatLonGridSource @ aws) ---------------------------
    for product in ("glo30", "glo90"):
        spec = _LATLON_SPECS[product]
        registry[product] = LatLonGridSource(
            name=product,
            description=spec["description"],
            product=product,
            provider="aws",
            resolution_m=spec["resolution_m"],
            vertical_datum=spec["vertical_datum"],
            derived=False,
            auth="none",
            wired=True,
            base_url=spec["base_url"],
            remote_layout="copernicus",
            stem_prefix=spec["stem_prefix"],
            suffix=".tif",
            min_bytes=GLO_MIN_TILE_BYTES,
        )

    # -- srtm-skadi (LatLonGridSource @ aws, skadi layout) -----------------
    registry["srtm-skadi"] = LatLonGridSource(
        name="srtm-skadi",
        description=(
            "SRTM 1-arc-second equivalent (.hgt.gz) from AWS terrain-tiles (EGM96)"
        ),
        product="srtm-skadi",
        provider="aws",
        resolution_m=1.0 / 3600,
        vertical_datum="egm96",
        derived=False,
        auth="none",
        wired=True,
        base_url=TERRAIN_TILES_BASE,
        remote_layout="skadi",
        suffix=".hgt.gz",
        min_bytes=SKADI_MIN_TILE_BYTES,
        ocean_404_skip=True,
    )

    # -- nasadem / alos-dem defaults (pc) ----------------------------------
    registry["nasadem"] = PcStacSource(
        name="nasadem",
        description="NASADEM 30m DSM via Planetary Computer (EGM96)",
        product="nasadem",
        provider="pc",
        resolution_m=1.0 / 3600,
        vertical_datum="egm96",
        derived=False,
        auth="none",
        wired=True,
        collection_id="nasadem",
        asset_key="elevation",
    )
    registry["alos-dem"] = PcStacSource(
        name="alos-dem",
        description="ALOS World 3D-30m DSM via Planetary Computer (EGM96)",
        product="alos-dem",
        provider="pc",
        resolution_m=1.0 / 3600,
        vertical_datum="egm96",
        derived=False,
        auth="none",
        wired=True,
        collection_id="alos-dem",
        asset_key="data",
    )

    # -- nasadem@earthdata ---------------------------------------------------
    registry["nasadem@earthdata"] = AuthenticatedGranuleSource(
        name="nasadem@earthdata",
        description="NASADEM full-quality granules via Earthdata Login (EGM96)",
        product="nasadem",
        provider="earthdata",
        resolution_m=1.0 / 3600,
        vertical_datum="egm96",
        derived=False,
        auth="token",
        wired=True,
        cmr_collection="C2763264762-LPCLOUD",
    )

    # -- nisar-glo30@earthdata ----------------------------------------------
    registry["nisar-glo30"] = AuthenticatedGranuleSource(
        name="nisar-glo30",
        description=(
            "NISAR Mission Modified Copernicus DEM (GLO-30 re-referenced to "
            "the WGS84 ellipsoid) via ASF Earthdata (derived)"
        ),
        product="nisar-glo30",
        provider="earthdata",
        resolution_m=1.0 / 3600,
        vertical_datum="ellipsoidal",
        derived=True,
        auth="token",
        wired=True,
        cmr_collection="C3803703055-ASF",
        data_host="nisar.asf.earthdatacloud.nasa.gov",
        asset_pattern="*.tif",
        title_filter_required=True,
        mosaic_recipe_override=_NISAR_MOSAIC_RECIPE,
    )

    # -- alos-dem@jaxa-ftp ----------------------------------------------------
    registry["alos-dem@jaxa-ftp"] = FtpZipSource(
        name="alos-dem@jaxa-ftp",
        description="AW3D30 30m DSM via legacy anonymous JAXA FTP zips (EGM96)",
        product="alos-dem",
        provider="jaxa-ftp",
        resolution_m=1.0 / 3600,
        vertical_datum="egm96",
        derived=False,
        auth="none",
        wired=True,
    )

    # -- terrain-tiles@aws -----------------------------------------------------
    registry["terrain-tiles"] = TerrainPyramidSource(
        name="terrain-tiles",
        description=(
            "AWS terrain-tiles merged global pyramid z12 (~38 m/px, "
            "mixed-derived datum, derived product)"
        ),
        product="terrain-tiles",
        provider="aws",
        resolution_m=360.0 / (256 * 2**12),
        vertical_datum="mixed-derived",
        derived=True,
        auth="none",
        wired=True,
        zoom=12,
    )

    # -- ArcticDEM / REMA tiers (PgcQuadSource @ aws) --------------------------
    pgc_specs = {
        ("arcticdem", "10"): {"version": "v4.1", "epsg": 3413},
        ("arcticdem", "32"): {"version": "v4.1", "epsg": 3413},
        ("arcticdem", "2"): {"version": "v4.1", "epsg": 3413},
        ("rema", "10"): {"version": "v2.0", "epsg": 3031},
        ("rema", "32"): {"version": "v2.0", "epsg": 3031},
        ("rema", "2"): {"version": "v2.0", "epsg": 3031},
    }
    for (collection, tier), spec in pgc_specs.items():
        product_name = f"{collection}-{tier}"
        registry[product_name] = PgcQuadSource(
            name=product_name,
            description=(
                f"{collection.capitalize()} {spec['version']} {tier}m mosaic "
                f"quads (EPSG:{spec['epsg']}, ellipsoidal)"
            ),
            product=product_name,
            provider="aws",
            # resolution_m stays in METERS (product metadata, pinned by the
            # pair-matrix test). The meter-to-degree conversion for the
            # EPSG:4326 output grid happens in _mosaic_arrays.
            resolution_m=30.0 if tier == "10" else (32.0 if tier == "32" else 2.0),
            vertical_datum="ellipsoidal",
            derived=False,
            auth="none",
            wired=True,
            collection=collection,
            version=spec["version"],
            resolution_tag=tier,
            epsg=spec["epsg"],
            mosaic_recipe_override=_PGC_MOSAIC_RECIPE,
        )

    return registry


_REGISTRY: dict[str, DemSource] = _build_registry()

#: product -> {providers: {provider: {wired, auth}}, default: provider}
PRODUCT_DEFAULTS: dict[str, dict] = {
    "glo30": {
        "providers": {
            "aws": {"wired": True, "auth": "none"},
            "pc": {"wired": True, "auth": "none"},
            "ot": {"wired": False, "auth": "token"},
        },
        "default": "aws",
    },
    "glo90": {
        "providers": {
            "aws": {"wired": True, "auth": "none"},
            "pc": {"wired": True, "auth": "none"},
            "ot": {"wired": False, "auth": "token"},
        },
        "default": "aws",
    },
    "nasadem": {
        "providers": {
            "pc": {"wired": True, "auth": "none"},
            "earthdata": {"wired": True, "auth": "token"},
            "ot": {"wired": False, "auth": "token"},
        },
        "default": "pc",
    },
    "alos-dem": {
        "providers": {
            "pc": {"wired": True, "auth": "none"},
            "jaxa-ftp": {"wired": True, "auth": "none"},
            "ot": {"wired": False, "auth": "token"},
        },
        "default": "pc",
    },
    "srtm-skadi": {
        "providers": {
            "aws": {"wired": True, "auth": "none"},
        },
        "default": "aws",
    },
    "terrain-tiles": {
        "providers": {
            "aws": {"wired": True, "auth": "none"},
        },
        "default": "aws",
    },
    "arcticdem-10": {
        "providers": {"aws": {"wired": True, "auth": "none"}},
        "default": "aws",
    },
    "arcticdem-32": {
        "providers": {"aws": {"wired": True, "auth": "none"}},
        "default": "aws",
    },
    "arcticdem-2": {
        "providers": {"aws": {"wired": True, "auth": "none"}},
        "default": "aws",
    },
    "rema-10": {
        "providers": {"aws": {"wired": True, "auth": "none"}},
        "default": "aws",
    },
    "rema-32": {
        "providers": {"aws": {"wired": True, "auth": "none"}},
        "default": "aws",
    },
    "rema-2": {"providers": {"aws": {"wired": True, "auth": "none"}}, "default": "aws"},
    "nisar-glo30": {
        "providers": {"earthdata": {"wired": True, "auth": "token"}},
        "default": "earthdata",
    },
}

PRODUCT_REGISTRY: dict[tuple[str, str], str] = {}
for _name, _source in _REGISTRY.items():
    PRODUCT_REGISTRY[(_source.product, _source.provider)] = _name

_AUTO_ENTRY = LatLonGridSource(
    name=AUTO_SOURCE_NAME,
    description=(
        "GLO-30 primary with control-tile-guarded GLO-90 fallback on "
        "withheld cells (manager semantics); product identity resolves to "
        "the glo30 primary"
    ),
    product=DEFAULT_PRODUCT,
    provider="aws",
    resolution_m=1.0 / 3600,
    vertical_datum="egm2008",
    derived=False,
    auth="none",
    wired=True,
    base_url=COPERNICUS_GLO30_BASE,
    remote_layout="copernicus",
    stem_prefix="Copernicus_DSM_COG_10",
    suffix=".tif",
    min_bytes=GLO_MIN_TILE_BYTES,
)


def _compute_selection_aliases() -> dict[str, str]:
    """Map bare products and ``product@provider`` identities to entries."""
    aliases: dict[str, str] = {}
    for name, source in _REGISTRY.items():
        if source.name == name:
            # Canonical bare-name entry: also reachable as product@provider.
            aliases[f"{source.product}@{source.provider}"] = name
        else:
            # Provider-suffixed entry (e.g. alos-dem@jaxa-ftp): the bare
            # product resolves to the wired default instead.
            aliases[source.name] = PRODUCT_REGISTRY[
                (source.product, PRODUCT_DEFAULTS[source.product]["default"])
            ]
    for product, meta in PRODUCT_DEFAULTS.items():
        aliases[product] = PRODUCT_REGISTRY[(product, meta["default"])]
    return aliases


_SELECTION_ALIASES = _compute_selection_aliases()

# Polar bands pinned from live listings (2026-08-23 calibration):
# ArcticDEM v4.1 quads span the arctic band north of ~60N (EPSG:3413 usage
# bbox); REMA v2.0 covers the Antarctic continent south of ~60S (EPSG:3031).
_ARCTICDEM_BAND = (60.0, 90.0)
_REMA_BAND = (-90.0, -60.0)


def _apply_polar_bands() -> None:
    """Attach coverage bands to PGC entries post-construction."""
    for tier in ("10", "32", "2"):
        object.__setattr__(
            _REGISTRY[f"arcticdem-{tier}"],
            "polar_band",
            _ARCTICDEM_BAND,
        )
        object.__setattr__(
            _REGISTRY[f"rema-{tier}"],
            "polar_band",
            _REMA_BAND,
        )


_apply_polar_bands()


# ---------------------------------------------------------------------------
# Public registry surface
# ---------------------------------------------------------------------------


def list_dem_sources() -> list[str]:
    """Return every registered DEM source selection name.

    Returns
    -------
    list[str]
        The 14 selection names: ``auto`` plus the 13 canonical bare product
        names.  Alternative-provider identities (``nasadem@earthdata``,
        ``alos-dem@jaxa-ftp``) stay reachable through :func:`get_dem_source`
        and :func:`parse_selection` but are not part of the default listing.

    """
    return sorted(
        [AUTO_SOURCE_NAME]
        + [name for name, source in _REGISTRY.items() if name == source.product]
    )


def get_dem_source(name: str) -> DemSource:
    """Return the registered source for ``name``, failing closed.

    Accepts registry names (bare products, alternative-provider identities
    such as ``nasadem@earthdata``), ``<product>:<provider>`` selections,
    bare-product aliases (resolved to the wired default provider), and the
    ``auto`` alias.

    Parameters
    ----------
    name : str
        Registry or selection name.

    Returns
    -------
    DemSource
        The registered source entry.

    Raises
    ------
    ValueError
        If the name is unknown; the error lists all valid names.

    """
    if name == AUTO_SOURCE_NAME:
        return _AUTO_ENTRY
    if name in _SELECTION_ALIASES:
        return _REGISTRY[_SELECTION_ALIASES[name]]
    try:
        return _REGISTRY[name]
    except KeyError:
        pass
    if ":" in name and name.count(":") == 1:
        try:
            return parse_selection(name)
        except ValueError:
            logger.exception("selection resolution failed for %r", name)
            raise
    message = f"unknown DEM source {name!r}; valid sources are: " + ", ".join(
        list_dem_sources()
    )
    logger.error(message)
    raise ValueError(message) from None


def dem_catalog() -> dict[str, dict]:
    """Return the structured product x provider matrix (zero network).

    Returns
    -------
    dict
        Per product: ``providers`` (each with ``wired`` and ``auth``),
        ``default`` provider, ``vertical_datum``, and ``resolution_m``.

    """
    catalog: dict[str, dict] = {}
    for product, meta in PRODUCT_DEFAULTS.items():
        default_entry = get_dem_source(_SELECTION_ALIASES[product])
        catalog[product] = {
            "providers": {
                provider: {"wired": info["wired"], "auth": info["auth"]}
                for provider, info in meta["providers"].items()
            },
            "default": meta["default"],
            "vertical_datum": default_entry.vertical_datum,
            "resolution_m": default_entry.resolution_m,
        }
    catalog[AUTO_SOURCE_NAME] = {
        "providers": {
            "aws": {"wired": True, "auth": "none"},
        },
        "default": "aws",
        "vertical_datum": _AUTO_ENTRY.vertical_datum,
        "resolution_m": _AUTO_ENTRY.resolution_m,
        "fallback": "glo90",
    }
    return catalog
