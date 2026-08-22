"""Multi-source DEM registry: product/provider metadata and tile enumeration.

Each wired DEM is described by a frozen :class:`DemSource` that separates a
**product group** (what the raster is: DSM/DTM/derived-merge, resolution,
vertical datum) from a **provider group** (where the bytes live: default base
URL, auth class, on-disk layout).  Selection-level fields (:attr:`name`,
:attr:`description`, :meth:`DemSource.tiles`, :attr:`DemSource.fallback`) tie
the two groups together under one registry name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.query import BoundingBox

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = setup_logger(__name__)

VerticalDatum = Literal["egm2008", "egm96", "mixed-derived", "ellipsoidal"]
ProductKind = Literal["dsm", "dtm", "topo-bathy", "merged-derived"]
DemMethod = Literal[
    "radar-interferometric",
    "optical-photogrammetric",
    "lidar",
    "composite",
]
AuthClass = Literal["none", "token", "registration"]

AUTO_SOURCE_NAME = "auto"
COPERNICUS_GLO30_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
COPERNICUS_GLO90_BASE = "https://copernicus-dem-90m.s3.amazonaws.com"
TERRAIN_TILES_BASE = "https://elevation-tiles-prod.s3.amazonaws.com"

#: Minimum valid size in bytes for a Copernicus COG tile (~1 MiB).
GLO_MIN_TILE_BYTES = 1 << 20
#: Minimum valid size in bytes for an AWS terrain-tiles z12 geotiff tile.
TERRAIN_Z12_MIN_TILE_BYTES = 4096
#: skadi tiles are gzip-compressed SRTM 1-degree grids; even an all-void tile
#: stays in the tens of KB once compressed.
SKADI_MIN_TILE_BYTES = 1024
#: Expected decompressed size of one 1-arc-second HGT grid (3601^2 * int16).
SKADI_EXPECTED_DECOMPRESSED_BYTES = 3601 * 3601 * 2

BoundsLike = BoundingBox | tuple[float, float, float, float]


def validate_cache_relative_path(relative_path: str, cache_dir: object) -> None:
    """Reject cache-relative paths with traversal or absolute components.

    Parameters
    ----------
    relative_path : str
        Registry-supplied cache-relative path candidate.
    cache_dir : object
        Cache root the path would be joined against (unused for the check
        itself; part of the boundary signature for future registry entries).

    Raises
    ------
    ValueError
        If the path contains a ``..`` component or is absolute.

    """
    del cache_dir
    candidate = relative_path.replace("\\", "/")
    if candidate.startswith("/"):
        message = f"cache-relative path must not be absolute: {relative_path!r}"
        logger.error(message)
        raise ValueError(message)
    parts = [part for part in candidate.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        message = f"cache-relative path must not contain '..': {relative_path!r}"
        logger.error(message)
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class DemTile:
    """One remote tile with its cache-relative identity and validity floor."""

    cache_relative_path: str
    remote_url: str
    minimum_bytes: int = GLO_MIN_TILE_BYTES
    expected_decompressed_bytes: int | None = None
    raster_open_recipe: Literal["plain", "vsigzip"] = "plain"
    _open_hook: Callable[[Path], str] | None = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
    )

    def raster_open_path(self, cached_path: Path) -> str:
        """Return the GDAL-openable path for the cached tile."""
        if self._open_hook is not None:
            return self._open_hook(cached_path)
        return str(cached_path)


def _skadi_name(latitude_deg: float, longitude_deg: float) -> str:
    """Return the skadi 1-degree HGT file name for a coordinate."""
    lat_tile = int(np.floor(latitude_deg))
    lon_tile = int(np.floor(longitude_deg))
    ns = "N" if lat_tile >= 0 else "S"
    ew = "E" if lon_tile >= 0 else "W"
    return f"{ns}{abs(lat_tile):02d}{ew}{abs(lon_tile):03d}.hgt.gz"


def _terrain_xyz(
    latitude_deg: float,
    longitude_deg: float,
    zoom: int,
) -> tuple[int, int]:
    """Return slippy-map XYZ indices (y measured from north) for a coordinate."""
    n_tiles = 2**zoom
    x_tile = int((longitude_deg + 180.0) / 360.0 * n_tiles)
    latitude = min(max(latitude_deg, -85.05112878), 85.05112878)
    lat_rad = math.radians(latitude)
    y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n_tiles)
    return x_tile, y_tile


@dataclass(frozen=True, slots=True)
class DemSource:
    """Registry entry describing one selectable DEM product/provider pair.

    Product group
    -------------
    product_kind:
        ``"dsm"`` for surface models, ``"dtm"`` for terrain models,
        ``"topo-bathy"`` for combined topography/bathymetry, and
        ``"merged-derived"`` for merged products mixing sources/datums.
    method:
        Measurement method (``"radar-interferometric"``,
        ``"optical-photogrammetric"``, ``"lidar"``, ``"composite"``); None
        when not applicable.
    hydro_conditioned / void_filled:
        Product modifiers; all v1 entries are False.
    resolution_m:
        Nominal ground sampling distance in degrees per pixel.
    vertical_datum / derived:
        Datum metadata; the pipeline wrap rule reads these.

    Provider group
    --------------
    default_base_url:
        Immutable public default base (https enforced on overrides).
    auth:
        Authentication class of the provider endpoint.
    layout_id:
        Identifier of the remote/cache layout recipe.

    Selection level
    ---------------
    name / description / tiles / fallback bind the groups into one
    user-selectable registry entry.
    """

    # selection-level
    name: str
    description: str
    fallback: str | None = None

    # product group
    product_kind: ProductKind = "dsm"
    resolution_m: float = 1.0 / 3600
    vertical_datum: VerticalDatum = "egm2008"
    derived: bool = False
    method: DemMethod | None = None
    hydro_conditioned: bool = False
    void_filled: bool = False

    # provider group
    default_base_url: str = COPERNICUS_GLO30_BASE
    auth: AuthClass = "none"
    layout_id: str = "copernicus-cog-stem"

    coverage_bounds: tuple[float, float, float, float] | None = None
    tiles_fn: Callable[[BoundsLike], list[DemTile]] | None = None

    @property
    def base_url(self) -> str:
        """Return the effective base URL (default until overridden)."""
        return self.default_base_url

    def with_base_url(
        self,
        base_url: str,
        *,
        warn: Callable[[str], None] | None = None,
    ) -> DemSource:
        """Return a copy bound to an explicit https-only base override.

        Parameters
        ----------
        base_url : str
            Replacement base URL for this source's provider only.
        warn : callable, optional
            Sink receiving a warning message when the override is applied.

        Returns
        -------
        DemSource
            Copy of this source with ``base_url`` set.

        Raises
        ------
        ValueError
            If the override scheme is not https.

        """
        if not base_url.lower().startswith("https://"):
            message = (
                f"DEM base URL override must use https, got: {base_url!r} "
                f"(source {self.name})"
            )
            logger.error(message)
            raise ValueError(message)
        if warn is not None:
            warn(f"DEM source {self.name} base URL overridden to {base_url}")
        return replace(self, default_base_url=base_url.rstrip("/"))

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
            message naming the uncovered latitude.

        """
        min_lat, max_lat = _bounds_tuple(bounds)[1], _bounds_tuple(bounds)[3]
        if self.coverage_bounds is None:
            return None
        south, north = self.coverage_bounds[1], self.coverage_bounds[3]
        if min_lat >= south and max_lat <= north:
            return None
        return (
            f"source {self.name} does not cover latitude range "
            f"[{min_lat}, {max_lat}]; covered range is [{south}, {north}]"
        )

    def tiles(self, bounds: BoundsLike) -> list[DemTile]:
        """Enumerate the tiles covering the requested bounds.

        Parameters
        ----------
        bounds : BoundingBox or tuple
            Geographic bounds in EPSG:4326 degrees.

        Returns
        -------
        list[DemTile]
            One entry per intersecting integer degree cell (or XYZ tile).

        Raises
        ------
        RuntimeError
            If the registry entry was built without an enumeration recipe.

        """
        if self.tiles_fn is None:
            message = f"source {self.name} has no tile enumeration recipe"
            logger.error(message)
            raise RuntimeError(message)
        return self.tiles_fn(bounds)


def _bounds_tuple(bounds: BoundsLike) -> tuple[float, float, float, float]:
    """Normalize a BoundingBox or plain tuple to (lon, lat, lon, lat)."""
    if isinstance(bounds, BoundingBox):
        return (
            float(bounds.left),
            float(bounds.bottom),
            float(bounds.right),
            float(bounds.top),
        )
    min_lon, min_lat, max_lon, max_lat = bounds
    return float(min_lon), float(min_lat), float(max_lon), float(max_lat)


def _enumerate_cells(
    bounds: BoundsLike,
    *,
    minimum_bytes: int,
    base_url: str,
    stem_prefix: str,
) -> list[DemTile]:
    """Enumerate degree-cell Copernicus-style COG tiles over bounds.

    Remote keys are ``{stem}/{stem}.tif`` at the bucket root; the cache keeps
    the per-tile directory layout (``{cell}/{stem}/{stem}.tif``).
    """
    min_lon, min_lat, max_lon, max_lat = _bounds_tuple(bounds)
    tiles: list[DemTile] = []
    for lat_cell in range(int(np.floor(min_lat)), int(np.floor(max_lat)) + 1):
        for lon_cell in range(int(np.floor(min_lon)), int(np.floor(max_lon)) + 1):
            lat_mid = lat_cell + 0.5
            lon_mid = lon_cell + 0.5
            lat_tile = int(np.floor(lat_mid))
            lon_tile = int(np.floor(lon_mid))
            ns = "N" if lat_tile >= 0 else "S"
            ew = "E" if lon_tile >= 0 else "W"
            directory = f"{ns}{abs(lat_tile):02d}_{ew}{abs(lon_tile):03d}"
            stem = (
                f"{stem_prefix}_{ns}{abs(lat_tile):02d}_00_"
                f"{ew}{abs(lon_tile):03d}_00_DEM"
            )
            filename = f"{stem}.tif"
            relative = f"{directory}/{stem}/{filename}"
            url = f"{base_url}/{stem}/{filename}"
            tiles.append(
                DemTile(
                    cache_relative_path=relative,
                    remote_url=url,
                    minimum_bytes=minimum_bytes,
                )
            )
    return tiles


def _copernicus_tiles_factory(
    base: str,
    prefix: str,
    minimum_bytes: int,
) -> Callable[[BoundsLike], list[DemTile]]:
    """Build a tiles callable for one Copernicus bucket/stem variant."""

    def _tiles(bounds: BoundsLike) -> list[DemTile]:
        return _enumerate_cells(
            bounds,
            minimum_bytes=minimum_bytes,
            base_url=base,
            stem_prefix=prefix,
        )

    return _tiles


def _skadi_tiles(bounds: BoundsLike) -> list[DemTile]:
    """Enumerate skadi HGT tiles over bounds."""
    min_lon, min_lat, max_lon, max_lat = _bounds_tuple(bounds)

    def _make(lat_cell: int, lon_cell: int) -> DemTile:
        ns = "N" if lat_cell >= 0 else "S"
        ew = "E" if lon_cell >= 0 else "W"
        name = f"{ns}{abs(lat_cell):02d}{ew}{abs(lon_cell):03d}.hgt.gz"
        relative = f"skadi/{ns}{abs(lat_cell):02d}/{name}"
        url = f"{TERRAIN_TILES_BASE}/skadi/{ns}{abs(lat_cell):02d}/{name}"

        def open_hook(cached_path: Path) -> str:
            return f"/vsigzip/{cached_path}"

        return DemTile(
            cache_relative_path=relative,
            remote_url=url,
            minimum_bytes=SKADI_MIN_TILE_BYTES,
            expected_decompressed_bytes=SKADI_EXPECTED_DECOMPRESSED_BYTES,
            raster_open_recipe="vsigzip",
            _open_hook=open_hook,
        )

    return [
        _make(lat_cell, lon_cell)
        for lat_cell in range(int(np.floor(min_lat)), int(np.floor(max_lat)) + 1)
        for lon_cell in range(int(np.floor(min_lon)), int(np.floor(max_lon)) + 1)
    ]


TERRAIN_TILES_DEFAULT_ZOOM = 12
_TERRAIN_Z12_METERS_PER_PIXEL = 360.0 / (256 * 2**TERRAIN_TILES_DEFAULT_ZOOM)


def _terrain_tiles_factory(
    zoom: int,
) -> Callable[[BoundsLike], list[DemTile]]:
    """Build a tiles callable for the terrain-tiles geotiff pyramid."""

    def _tiles(bounds: BoundsLike) -> list[DemTile]:
        min_lon, min_lat, max_lon, max_lat = _bounds_tuple(bounds)
        n_tiles = 2**zoom
        x_min = int((min_lon + 180.0) / 360.0 * n_tiles)
        x_max = int((max_lon + 180.0) / 360.0 * n_tiles)
        top = min(max(max_lat, -85.05112878), 85.05112878)
        bottom = min(max(min_lat, -85.05112878), 85.05112878)
        lat_rad_top = math.radians(top)
        lat_rad_bottom = math.radians(bottom)
        y_min = int((1.0 - math.asinh(math.tan(lat_rad_top)) / math.pi) / 2.0 * n_tiles)
        y_max = int(
            (1.0 - math.asinh(math.tan(lat_rad_bottom)) / math.pi) / 2.0 * n_tiles
        )
        tiles: list[DemTile] = []
        for x_tile in range(x_min, x_max + 1):
            for y_tile in range(y_min, y_max + 1):
                relative = f"geotiff/{zoom}/{x_tile}/{y_tile}.tif"
                url = f"{TERRAIN_TILES_BASE}/geotiff/{zoom}/{x_tile}/{y_tile}.tif"
                tiles.append(
                    DemTile(
                        cache_relative_path=relative,
                        remote_url=url,
                        minimum_bytes=TERRAIN_Z12_MIN_TILE_BYTES,
                    )
                )
        return tiles

    return _tiles


_SOURCES: dict[str, DemSource] = {
    "copernicus-30": DemSource(
        name="copernicus-30",
        description="Copernicus GLO-30 DSM (EGM2008), default source",
        product_kind="dsm",
        resolution_m=1.0 / 3600,
        vertical_datum="egm2008",
        derived=False,
        method="radar-interferometric",
        default_base_url=COPERNICUS_GLO30_BASE,
        auth="none",
        layout_id="copernicus-cog-stem",
        tiles_fn=_copernicus_tiles_factory(
            COPERNICUS_GLO30_BASE, "Copernicus_DSM_COG_10", GLO_MIN_TILE_BYTES
        ),
    ),
    "copernicus-90": DemSource(
        name="copernicus-90",
        description=(
            "Copernicus GLO-90 DSM (EGM2008); covers cells withheld from GLO-30"
        ),
        product_kind="dsm",
        resolution_m=1.0 / 1200,
        vertical_datum="egm2008",
        derived=False,
        method="radar-interferometric",
        default_base_url=COPERNICUS_GLO90_BASE,
        auth="none",
        layout_id="copernicus-cog-stem",
        tiles_fn=_copernicus_tiles_factory(
            COPERNICUS_GLO90_BASE, "Copernicus_DSM_COG_30", GLO_MIN_TILE_BYTES
        ),
    ),
    AUTO_SOURCE_NAME: DemSource(
        name=AUTO_SOURCE_NAME,
        description=(
            "Copernicus GLO-30 with validated per-cell GLO-90 fallback "
            "on withheld cells"
        ),
        fallback="copernicus-90",
        product_kind="dsm",
        resolution_m=1.0 / 3600,
        vertical_datum="egm2008",
        derived=False,
        method="radar-interferometric",
        default_base_url=COPERNICUS_GLO30_BASE,
        auth="none",
        layout_id="copernicus-cog-stem",
        tiles_fn=_copernicus_tiles_factory(
            COPERNICUS_GLO30_BASE, "Copernicus_DSM_COG_10", GLO_MIN_TILE_BYTES
        ),
    ),
    "srtm-skadi": DemSource(
        name="srtm-skadi",
        description="SRTM 1-arc-second equivalent (.hgt.gz) from AWS terrain-tiles",
        fallback=None,
        product_kind="dsm",
        resolution_m=1.0 / 3600,
        vertical_datum="egm96",
        derived=False,
        method="radar-interferometric",
        default_base_url=TERRAIN_TILES_BASE,
        auth="none",
        layout_id="skadi-hgt-gz",
        coverage_bounds=(-180.0, -56.0, 180.0, 60.0),
        tiles_fn=_skadi_tiles,
    ),
    "terrain-tiles": DemSource(
        name="terrain-tiles",
        description="AWS terrain-tiles merged global pyramid at zoom 12 "
        "(~38 m/px, mixed-derived datum)",
        fallback=None,
        product_kind="merged-derived",
        resolution_m=_TERRAIN_Z12_METERS_PER_PIXEL,
        vertical_datum="mixed-derived",
        derived=True,
        method="composite",
        default_base_url=TERRAIN_TILES_BASE,
        auth="none",
        layout_id="terrain-zxy",
        tiles_fn=_terrain_tiles_factory(TERRAIN_TILES_DEFAULT_ZOOM),
    ),
}


def list_dem_sources() -> list[str]:
    """Return every registered DEM source name.

    Returns
    -------
    list[str]
        Sorted registry names.

    """
    return sorted(_SOURCES)


def get_dem_source(name: str) -> DemSource:
    """Return the registered source for ``name``, failing closed.

    Parameters
    ----------
    name : str
        Registry name such as ``copernicus-30``.

    Returns
    -------
    DemSource
        The registered source entry.

    Raises
    ------
    ValueError
        If the name is unknown; the error lists all valid names.

    """
    try:
        return _SOURCES[name]
    except KeyError:
        message = f"unknown DEM source {name!r}; valid sources are: " + ", ".join(
            list_dem_sources()
        )
        logger.exception(message)
        raise ValueError(message) from None
