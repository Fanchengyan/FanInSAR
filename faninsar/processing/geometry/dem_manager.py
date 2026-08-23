"""Multi-source automatic DEM management (PROPOSAL-0030).

Resolves a selectable DEM source (``glo30``, ``glo90``, ``nasadem``, ... or
``auto``), executes its :class:`~faninsar.processing.geometry.dem_transport.FetchPlan`
through the audited transport engine, caches tiles under per-identity
partitions (``<product>-<provider>/``), and mosaics the result into one
float32 EPSG:4326 GeoTIFF with provenance tags.

The ``auto`` selection is GLO-30 with a control-tile-guarded per-cell GLO-90
fallback on withheld cells: the fallback is trusted only after a known-present
control tile validates the primary base, an all-withheld ROI legitimately
rescues via the canonical out-of-ROI control cell, and a control failure is a
loud mirror-misconfiguration error instead of a silent whole-ROI downgrade.
"""

from __future__ import annotations

import math
import os
import urllib.parse
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry.dem_sources import (
    AUTO_SOURCE_NAME,
    DEFAULT_PRODUCT,
    PRODUCT_DEFAULTS,
    DemSource,
    MosaicRecipe,
    PcStacSource,
    _bounds_tuple,
    get_dem_source,
    parse_selection,
)
from faninsar.processing.geometry.dem_transport import (
    MAX_ATTEMPTS,
    Artifact,
    FetchPlan,
    Tile,
    TileSet,
    fetch_plan,
)
from faninsar.query import BoundingBox

logger = setup_logger(__name__)

__all__ = [
    "AUTO_CONTROL_CELL",
    "COPERNICUS_GLO30_URL",
    "DEMManager",
    "DEMProviderUnavailableError",
    "copernicus_tile_name",
    "default_dem_name",
    "get_dem_manager",
]

COPERNICUS_GLO30_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
DEM_CACHE_ENV = "FANINSAR_DEM_CACHE_DIR"
#: Selection grammar environment (``<product>`` / ``<product>:<provider>``).
DEM_SELECTION_ENV = "FANINSAR_DEM_SOURCE"
#: Base-URL override for the primary selected source (https enforced).
DEM_SOURCE_ENV = "FANINSAR_DEM_SOURCE_URL"
DEM_NAME_ENV = "FANINSAR_DEM_NAME"
DEFAULT_DEM_NAME = "dem.tif"

#: Canonical out-of-ROI GLO-30 control cell (lat, lon degrees, Po Valley) used
#: by ``auto`` when every ROI cell is withheld on the primary base.
AUTO_CONTROL_CELL = (45, 9)

#: Fraction of rescued cells above which ``auto`` warns loudly.
_AUTO_FALLBACK_WARN_FRACTION = 0.25

Bounds = BoundingBox | tuple[float, float, float, float]


class DEMProviderUnavailableError(InvalidProcessingStateError):
    """Structured outage contract for an exhausted DEM provider.

    Attributes
    ----------
    product, provider, host
        Identity of the failing selection.
    failure_class
        One of ``auth``, ``forbidden``, ``coverage``, ``upstream-outage``.
    attempts
        Transport attempts spent on the failing request(s).
    alternatives
        Wired same-product alternative providers with their auth cost;
        unwired providers are excluded in v1. Switching is always manual.

    """

    def __init__(
        self,
        message: str,
        *,
        product: str,
        provider: str,
        host: str,
        failure_class: str,
        attempts: int,
        alternatives: dict[str, str],
    ) -> None:
        """Build the structured outage error."""
        super().__init__(message)
        self.product = product
        self.provider = provider
        self.host = host
        self.failure_class = failure_class
        self.attempts = attempts
        self.alternatives = alternatives


def copernicus_tile_name(latitude_deg: float, longitude_deg: float) -> tuple[str, str]:
    """Return the Copernicus GLO-30 tile directory and file name for a coordinate.

    Parameters
    ----------
    latitude_deg, longitude_deg : float
        Geodetic coordinates in degrees.

    Returns
    -------
    tuple[str, str]
        Tile directory and file name following the COG convention.

    """
    lat_tile = int(np.floor(latitude_deg))
    lon_tile = int(np.floor(longitude_deg))
    ns = "N" if lat_tile >= 0 else "S"
    ew = "E" if lon_tile >= 0 else "W"
    lat_abs = abs(lat_tile)
    lon_abs = abs(lon_tile)
    filename = (
        f"Copernicus_DSM_COG_10_{ns}{lat_abs:02d}_00_{ew}{lon_abs:03d}_00_DEM.tif"
    )
    return f"{ns}{lat_abs:02d}_{ew}{lon_abs:03d}", filename


def default_dem_name() -> str:
    """Return the configured output DEM file name or dem.tif."""
    return os.environ.get(DEM_NAME_ENV, DEFAULT_DEM_NAME)


def _host_of(url: str) -> str:
    """Return the lowercase hostname of ``url``."""
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def _source_base(entry: DemSource) -> str:
    """Return the entry's primary base URL for host tagging."""
    base = getattr(entry, "base_url", None) or getattr(entry, "ftp_base", None)
    if base is None and isinstance(entry, PcStacSource):
        base = entry.stac_api_url
    return base or ""


def _wired_alternatives(product: str, provider: str) -> dict[str, str]:
    """Enumerate wired same-product alternative providers with auth cost."""
    meta = PRODUCT_DEFAULTS.get(product, {})
    alternatives: dict[str, str] = {}
    for name, info in meta.get("providers", {}).items():
        if name == provider or not info.get("wired", False):
            # Unwired providers stay excluded in v1 (never silent fallbacks).
            continue
        auth = info.get("auth", "none")
        cost = "no authentication" if auth == "none" else f"requires {auth}"
        alternatives[name] = cost
    return alternatives


def _classify_failure(exc: BaseException) -> str:
    """Map a transport failure into the structured failure taxonomy."""
    text = str(exc)
    status = getattr(exc, "status", None)
    if status == 401 or "terminal status 401" in text:
        return "auth"
    if status == 403 or "terminal status 403" in text:
        return "forbidden"
    if "status 404" in text:
        return "coverage"
    return "upstream-outage"


def _probe_status(url: str, attempts: int = 3) -> int | None:
    """Best-effort HEAD probe returning the HTTP status or None.

    Used only by the ``auto`` control-tile guard; transient probe failures
    return ``None`` after a couple of quiet retries.  The session is resolved
    through :mod:`dem_transport` on every call so test fakes (and any future
    session instrumentation) apply uniformly.
    """
    from faninsar.processing.geometry import dem_transport as _transport

    last: int | None = None
    for attempt in range(attempts):
        try:
            response = _transport.thread_local_session().head(
                url, timeout=_transport.REQUEST_TIMEOUT
            )
            status = int(response.status_code)
            response.close()
        except AssertionError:
            raise
        except Exception as exc:
            last = None
            logger.debug(
                "auto probe attempt %d failed for %s: %s",
                attempt + 1,
                url,
                type(exc).__name__,
            )
        else:
            return status
    return last


def _open_target(path: Path, gdal_open: str) -> list[str]:
    """Resolve one cached file into openable GDAL dataset name(s).

    Local cache copies never need ``/vsicurl/``; ``/vsigzip/`` and
    ``/vsizip/`` prefixes are honored, with ``{member}`` templates expanded
    against the archive's raster members.
    """
    if "{member}" in gdal_open:
        import zipfile

        with zipfile.ZipFile(path) as zf:
            members = [
                name
                for name in zf.namelist()
                if name.lower().endswith((".tif", ".tiff", ".hgt"))
            ]
        if not members:
            message = f"no raster members found in archive {path}"
            logger.error(message)
            raise InvalidProcessingStateError(message)
        return [gdal_open.format(path=path, member=name) for name in members]
    if gdal_open.startswith("/vsicurl/"):
        return [str(path)]
    return [gdal_open.format(path=path)]


def _resampling(name: str) -> object:
    """Translate a recipe resampling name into a rasterio Resampling enum."""
    from rasterio.enums import Resampling

    try:
        return Resampling[name]
    except KeyError as exc:
        message = f"unsupported mosaic resampling {name!r}"
        logger.exception(message)
        raise InvalidProcessingStateError(message) from exc


def resolution_m_to_degrees(resolution_m: float) -> float:
    """Convert a meters resolution to EPSG:4326 degrees.

    Uses a cos(75 deg) mid-band factor documented for the PGC polar mosaics;
    deterministic so tests can pin the exact value.
    """
    return resolution_m / (111_320.0 * math.cos(math.radians(75.0)))


def _mosaic_arrays(
    paths: list[Path],
    recipe: MosaicRecipe,
    resolution_m: float,
) -> tuple[np.ndarray, object]:
    """Reproject every input onto one explicit EPSG:4326 grid.

    The output pixel size is exactly ``resolution_m`` regardless of the first
    dataset's native resolution, so a coarse fallback tile can never silently
    downgrade the mosaic grid.
    """
    import rasterio
    from rasterio import warp
    from rasterio.transform import from_origin

    left = bottom = right = top = None
    opened: list[tuple[rasterio.DatasetReader, str]] = []
    try:
        for path in paths:
            for target in _open_target(path, recipe.gdal_open):
                dataset = rasterio.open(target)
                opened.append((dataset, target))
                ds_left, ds_bottom, ds_right, ds_top = warp.transform_bounds(
                    dataset.crs,
                    rasterio.crs.CRS.from_epsg(4326),
                    *dataset.bounds,
                    densify_pts=21,
                )
                left = ds_left if left is None else min(left, ds_left)
                bottom = ds_bottom if bottom is None else min(bottom, ds_bottom)
                right = ds_right if right is None else max(right, ds_right)
                top = ds_top if top is None else max(top, ds_top)
        if not opened or left is None:
            message = "no DEM rasters available for mosaicking"
            logger.error(message)
            raise InvalidProcessingStateError(message)

        width = max(1, int(np.ceil((right - left) / resolution_m)))
        height = max(1, int(np.ceil((top - bottom) / resolution_m)))
        transform = from_origin(left, top, resolution_m, resolution_m)
        mosaic = np.full((height, width), np.nan, dtype=np.float32)
        resampling = _resampling(recipe.resampling)
        for dataset, _target in opened:
            band = dataset.read(1).astype(np.float32, copy=False)
            src_nodata = recipe.nodata if recipe.nodata is not None else dataset.nodata
            dst = np.full((height, width), np.nan, dtype=np.float32)
            warp.reproject(
                source=band,
                destination=dst,
                src_transform=dataset.transform,
                src_crs=dataset.crs,
                src_nodata=src_nodata,
                dst_transform=transform,
                dst_crs=rasterio.crs.CRS.from_epsg(4326),
                dst_nodata=np.nan,
                resampling=resampling,
            )
            known = (
                np.isfinite(dst)
                if src_nodata is not None
                else np.ones_like(dst, dtype=bool)
            )
            mosaic[known] = dst[known]
    finally:
        for dataset, _target in opened:
            dataset.close()
    return mosaic, transform


@dataclass(slots=True)
class DEMManager:
    """Resolve, cache, download, and mosaic a selectable DEM source.

    Parameters
    ----------
    cache_dir : Path
        Root of the raw tile cache. Downloads land under
        ``<cache_dir>/<product>-<provider>/``; legacy flat GLO-30 layouts
        remain valid hits for ``glo30@aws`` lookups only.
    source : str or DemSource, optional
        ``"<product>"``, ``"<product>:<provider>"``, ``"auto"``, or a raw
        :class:`~faninsar.processing.geometry.dem_sources.DemSource`.
        ``None`` (the default) defers to ``FANINSAR_DEM_SOURCE`` and then to
        ``"glo30"``; passing any explicit value (including ``"glo30"``)
        overrides the environment.
    max_workers : int
        Shared stream budget handed to the transport engine.
    chunked_threshold : int
        Missing-tile count under which ranged mode is preferred.
    base_url : str, optional
        https base-URL override for the selected source (mirror escape
        hatch); rejected for shapes without a base URL (FTP, Earthdata).

    """

    cache_dir: Path
    source: str | DemSource | None = None
    max_workers: int = 8
    chunked_threshold: int = 4
    base_url: str | None = None
    source_entry: DemSource = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve the selection and apply the base override (fail closed)."""
        self.cache_dir = Path(self.cache_dir)
        if isinstance(self.source, DemSource):
            entry: DemSource = self.source
        elif self.source is not None:
            entry = self._resolve_selection(str(self.source).strip())
        else:
            # No explicit kwarg: env first, then the glo30 default.
            selection = os.environ.get(DEM_SELECTION_ENV, DEFAULT_PRODUCT).strip()
            entry = self._resolve_selection(selection)
        if self.base_url is not None:
            entry = self._apply_base_override(entry)
        object.__setattr__(self, "source_entry", entry)

    def _resolve_selection(self, selection: str) -> DemSource:
        """Resolve a selection string, handling the ``auto`` alias fail-closed."""
        if selection == AUTO_SOURCE_NAME:
            return get_dem_source(AUTO_SOURCE_NAME)
        if selection.split(":")[0] == AUTO_SOURCE_NAME:
            message = (
                f"invalid DEM source {selection!r}: 'auto' accepts no "
                "provider override; select the underlying product "
                "directly (e.g. 'glo90')"
            )
            logger.error(message)
            raise ValueError(message)
        return parse_selection(selection)

    def _apply_base_override(self, entry: DemSource) -> DemSource:
        """Clone ``entry`` with the https base override applied."""
        if not self.base_url.lower().startswith("https://"):
            message = (
                f"FANINSAR_DEM_SOURCE_URL-style overrides must use https, "
                f"got {self.base_url!r}"
            )
            logger.error(message)
            raise InvalidProcessingStateError(message)
        if hasattr(entry, "base_url"):
            return replace(entry, base_url=self.base_url)
        if isinstance(entry, PcStacSource):
            return replace(entry, stac_api_url=self.base_url)
        message = (
            f"source {entry.name!r} ({entry.provider}) does not accept a "
            "base URL override; its endpoint is pinned"
        )
        logger.error(message)
        raise InvalidProcessingStateError(message)

    # -- identity -----------------------------------------------------------

    @property
    def product(self) -> str:
        """Product group of the resolved selection."""
        return self.source_entry.product

    @property
    def provider(self) -> str:
        """Provider group of the resolved selection."""
        return self.source_entry.provider

    @property
    def vertical_datum(self) -> str:
        """Vertical datum declared by the resolved registry entry.

        Drives the shared datum-aware wrap rule in
        :func:`faninsar.processing.pipeline.production.resolve_auto_dem`;
        ``ellipsoidal`` sources are returned unwrapped.
        """
        return self.source_entry.vertical_datum

    @property
    def partition_name(self) -> str:
        """Opaque hyphen-joined cache partition label for this identity."""
        return f"{self.source_entry.product}-{self.source_entry.provider}"

    @property
    def partition_dir(self) -> Path:
        """Directory holding this identity's downloads."""
        return self.cache_dir / self.partition_name

    # -- tile planning ------------------------------------------------------

    def required_tiles(self, bounds: Bounds) -> list[Tile]:
        """Return the sorted :class:`Tile` records covering the bounds.

        Meaningful only for TileSet-shaped sources; Artifact-shaped sources
        (whole-archive units such as AW3D30 FTP zips) raise a clear error.

        Raises
        ------
        InvalidProcessingStateError
            When the resolved source plans Artifacts instead of tiles.

        """
        plan = self.source_entry.plan(bounds)
        if isinstance(plan, TileSet):
            return [self._partition_tile(tile) for tile in plan.tiles]
        message = (
            f"source {self.source_entry.name!r} plans whole-artifact units "
            f"({type(plan).__name__}); required_tiles is undefined for it"
        )
        logger.error(message)
        raise InvalidProcessingStateError(message)

    # -- cache lookups ------------------------------------------------------

    def _legacy_hit(self, tile: Tile) -> Path | None:
        """Probe legacy flat GLO-30 layouts (bound to glo30@aws lookups).

        Existing DATA2 caches store tiles flat or under ``<tag>/``
        subdirectories directly below the cache root; other identities must
        never adopt those stems as hits because they collide across cache
        directories.
        """
        if (self.source_entry.product, self.source_entry.provider) not in {
            ("glo30", "aws"),
            (AUTO_SOURCE_NAME, "aws"),
        }:
            return None
        parts = tile.cache_path.parts
        if len(parts) < 2:
            return None
        tag_dir, filename = parts[0], parts[-1]
        candidates = [
            self.cache_dir / filename,
            self.cache_dir / tag_dir / filename,
        ]
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size >= tile.min_bytes:
                return candidate
        return None

    def _tile_hit(self, tile: Tile) -> Path | None:
        """Probe the partitioned target, then the legacy layout, for a tile."""
        target = self.cache_dir / tile.cache_path
        if target.is_file():
            return target
        return self._legacy_hit(tile)

    def _resolve_tile_hits(self, plan: TileSet) -> list[Path]:
        """Resolve every planned tile against the cache, fetching only misses.

        Hits come from the identity partition directory or (for ``glo30@aws``
        lookups only) the legacy flat layout; the remaining tiles are wrapped
        into the partition and handed to the transport engine together.
        """
        resolved: list[Path] = []
        missing: list[Tile] = []
        seen_paths: set[str] = set()
        seen_targets: set[str] = set()
        for raw in plan.tiles:
            tile = self._partition_tile(raw)
            hit = self._tile_hit(tile)
            if hit is not None:
                key = str(hit)
                if key not in seen_paths:
                    seen_paths.add(key)
                    resolved.append(hit)
                continue
            key = str(tile.cache_path)
            if key not in seen_targets:
                seen_targets.add(key)
                missing.append(tile)
        if missing:
            executed = self._execute(replace(plan, tiles=tuple(missing)))
            for path in executed:
                key = str(path)
                if key not in seen_paths:
                    seen_paths.add(key)
                    resolved.append(path)
        return resolved

    # -- plan execution -----------------------------------------------------

    def _partitioned(self, tile: Tile, *labels: str) -> Tile:
        """Clone ``tile`` with its cache path nested under partition labels."""
        wrapped = Path(*labels) / tile.cache_path
        return replace(tile, cache_path=wrapped)

    def _partition_tile(self, tile: Tile) -> Tile:
        """Wrap one planned tile into this identity's partition directory."""
        return self._partitioned(tile, self.partition_name)

    def _outage_error(
        self,
        exc: BaseException,
        *,
        host: str | None = None,
        extra: str = "",
    ) -> DEMProviderUnavailableError:
        """Wrap a transport failure into the structured outage contract."""
        entry = self.source_entry
        failure_class = _classify_failure(exc)
        attempts = 1 if failure_class in {"auth", "forbidden"} else MAX_ATTEMPTS
        alternatives = _wired_alternatives(entry.product, entry.provider)
        alt_text = ", ".join(
            f"{name} ({cost})" for name, cost in sorted(alternatives.items())
        )
        hint = (
            "Completed tiles stay cached and a later retry resumes; check "
            "cached tiles under the cache partition directories before "
            "switching providers manually."
        )
        message = (
            f"DEM provider unavailable: {entry.product}@{entry.provider} "
            f"({failure_class}) after {attempts} attempt(s){extra}: "
            f"{exc}. Same-product alternatives: {alt_text or 'none'}. {hint}"
        )
        logger.error(message)
        return DEMProviderUnavailableError(
            message,
            product=entry.product,
            provider=entry.provider,
            host=host or _host_of(_source_base(entry)),
            failure_class=failure_class,
            attempts=attempts,
            alternatives=alternatives,
        )

    def _execute(self, plan: FetchPlan) -> list[Path]:
        """Execute one plan, translating engine failures to the contract."""
        try:
            return fetch_plan(
                plan,
                self.cache_dir,
                max_workers=self.max_workers,
                chunked_threshold=self.chunked_threshold,
            )
        except (InvalidProcessingStateError, Exception) as exc:
            if isinstance(exc, AssertionError):
                raise
            raise self._outage_error(exc) from exc

    def _artifact_members(self, plan: Artifact) -> list[Path]:
        """Return extracted/extractable raster members for one artifact."""
        if plan.cache_path is None:
            return []
        staging = (
            self.cache_dir
            / plan.cache_path.parent
            / (plan.cache_path.name + ".zip-staging")
        )
        pattern = plan.member_pattern
        if pattern is None and plan.members:
            return [
                self.cache_dir / plan.cache_path.parent / name for name in plan.members
            ]
        if staging.is_dir():
            import fnmatch

            return [
                path
                for path in sorted(staging.rglob("*"))
                if path.is_file() and fnmatch.fnmatch(path.name, pattern or "*")
            ]
        archive = self.cache_dir / plan.cache_path
        return [archive] if archive.is_file() else []

    def _paths_for_plan(self, plan: FetchPlan, executed: list[Path]) -> list[Path]:
        """Collect mosaic inputs after executing ``plan``.

        Tile lookups probe the partitioned target first, then the legacy
        flat layout (``glo30@aws`` only); every planned tile must resolve,
        otherwise the fetch already failed loudly inside ``_execute``.
        """
        if isinstance(plan, TileSet):
            paths: list[Path] = []
            for tile in plan.tiles:
                target = self.cache_dir / tile.cache_path
                if target.is_file():
                    paths.append(target)
                    continue
                legacy = self._legacy_hit(tile)
                if legacy is not None:
                    paths.append(legacy)
                    continue
                message = (
                    f"tile fetch did not produce {tile.cache_path}; refusing "
                    "to mosaic an incomplete set"
                )
                logger.error(message)
                raise InvalidProcessingStateError(message)
            return paths
        if isinstance(plan, Artifact):
            return executed
        members: list[Path] = []
        for sub in getattr(plan, "artifacts", ()):
            members.extend(executed if executed else self._artifact_members(sub))
        return members

    def _write_mosaic(
        self,
        paths: list[Path],
        output_path: Path,
        *,
        tags: dict[str, str],
    ) -> Path:
        """Write the merged float32 EPSG:4326 tiled deflate GeoTIFF."""
        import rasterio

        recipe = self.source_entry.mosaic_recipe()
        # PGC-style sources keep resolution_m in METERS on their native
        # polar-stereo grid; when the recipe reprojects to EPSG:4326 the
        # output grid is sized in degrees (cos(75 deg) mid-band factor,
        # deterministic per the proposal). Orthometric degree-native sources
        # (glo30 etc.) use their resolution directly.
        resolution_deg = (
            resolution_m_to_degrees(self.source_entry.resolution_m)
            if recipe.warp_target == "epsg4326"
            and self.source_entry.vertical_datum == "ellipsoidal"
            else self.source_entry.resolution_m
        )
        mosaic, transform = _mosaic_arrays(paths, recipe, resolution_deg)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            self._warn_contradicting_tags(output_path)
        profile = {
            "driver": "GTiff",
            "height": mosaic.shape[0],
            "width": mosaic.shape[1],
            "count": 1,
            "dtype": "float32",
            "nodata": np.nan,
            "crs": rasterio.crs.CRS.from_epsg(4326),
            "transform": transform,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic, 1)
            dst.update_tags(**tags)
        logger.info(
            "DEM mosaic written: %s shape=%s source=%s@%s",
            output_path,
            mosaic.shape,
            tags.get("dem_product"),
            tags.get("dem_provider"),
        )
        return output_path

    def _warn_contradicting_tags(self, output_path: Path) -> None:
        """Warn when an existing mosaic's stamps contradict the selection.

        Tags are advisory provenance only: the file is still overwritten and
        used (use-after-warning is safe under the product-identity invariant;
        datum/wrap authority stays with live registry metadata).
        """
        import rasterio

        try:
            with rasterio.open(output_path) as dataset:
                tags = dataset.tags()
        except Exception:
            return
        stamped_product = str(tags.get("dem_product", ""))
        stamped_provider = str(tags.get("dem_provider", ""))
        if stamped_product and (
            stamped_product != self.source_entry.product
            or (stamped_provider and stamped_provider != self.source_entry.provider)
        ):
            logger.warning(
                "existing DEM mosaic %s carries contradicting provenance tags "
                "(%s@%s, current selection %s@%s); overwriting and using it "
                "(advisory tags never drive wrap decisions)",
                output_path,
                stamped_product,
                stamped_provider,
                self.source_entry.product,
                self.source_entry.provider,
            )

    def _provenance_tags(self, *, fallback: str | None = None) -> dict[str, str]:
        """Build the advisory provenance tag set for the output mosaic."""
        entry = self.source_entry
        tags = {
            "dem_product": entry.product
            if entry.name != AUTO_SOURCE_NAME
            else DEFAULT_PRODUCT,
            "dem_provider": entry.provider,
            "dem_vertical_datum": entry.vertical_datum,
            "dem_host": _host_of(_source_base(entry)),
            "dem_retrieved": datetime.now(tz=UTC).date().isoformat(),
        }
        if fallback is not None:
            tags["dem_fallback"] = fallback
        return tags

    # -- public entry points -------------------------------------------------

    def fetch_dem(self, bounds: Bounds, output_path: str | Path | None = None) -> Path:
        """Execute the source plan for the bounds and write one merged GeoTIFF.

        Fetching is resumable rather than transactional: completed tiles
        persist in the cache if a later tile fails (the mosaic is then
        absent). ``auto`` adds the control-tile-guarded GLO-90 per-cell
        fallback with an explicit primary-resolution merge.

        Parameters
        ----------
        bounds : BoundingBox or tuple
            (min_lon, min_lat, max_lon, max_lat) in EPSG:4326 or a BoundingBox.
        output_path : path, optional
            Destination GeoTIFF. Defaults to
            <cache parent>/dem/<FANINSAR_DEM_NAME or dem.tif>.

        Returns
        -------
        Path
            The written mosaic path.

        """
        if output_path is None:
            output_path = self.cache_dir.parent / "dem" / default_dem_name()
        out = Path(output_path)
        if self.source_entry.name == AUTO_SOURCE_NAME:
            return self._fetch_auto(_bounds_tuple(bounds), out)
        plan = self.source_entry.plan(bounds)
        if isinstance(plan, TileSet):
            paths = self._resolve_tile_hits(plan)
        else:
            executed = self._execute(plan)
            paths = self._paths_for_plan(plan, executed)
        return self._write_mosaic(paths, out, tags=self._provenance_tags())

    # -- auto: control-tile-guarded GLO-90 fallback ---------------------------

    def _fetch_auto(self, bounds: tuple[float, float, float, float], out: Path) -> Path:
        """Fetch GLO-30 with per-cell GLO-90 rescue on withheld cells."""
        primary = self.source_entry
        glo90 = get_dem_source("glo90")
        primary_plan = primary.plan(bounds)
        glo90_plan = glo90.plan(bounds)
        host = _host_of(_source_base(primary))

        statuses: dict[str, int | None] = {}
        for tile in primary_plan.tiles:
            statuses[tile.url] = _probe_status(tile.url)
        validated = any(status == 200 for status in statuses.values())
        if not validated:
            # Every ROI cell is missing-or-error: consult the canonical
            # out-of-ROI control cell before trusting any fallback.
            lat_cell, lon_cell = AUTO_CONTROL_CELL
            _, control_file = copernicus_tile_name(lat_cell + 0.5, lon_cell + 0.5)
            control_stem = control_file[: -len(".tif")]
            control_url = f"{_source_base(primary)}/{control_stem}/{control_file}"
            control_status = _probe_status(control_url)
            if control_status != 200:
                message = (
                    "auto DEM control tile failed: primary mirror "
                    f"{host} returned {control_status} for the canonical "
                    "control cell and every ROI cell is missing; refusing to "
                    "silently degrade the ROI to GLO-90 (mirror "
                    "misconfiguration)"
                )
                logger.error(message)
                raise DEMProviderUnavailableError(
                    message,
                    product=DEFAULT_PRODUCT,
                    provider=primary.provider,
                    host=host,
                    failure_class="upstream-outage",
                    attempts=MAX_ATTEMPTS,
                    alternatives=_wired_alternatives(DEFAULT_PRODUCT, primary.provider),
                )
            validated = True
        del validated

        withheld: set[str] = {url for url, status in statuses.items() if status == 404}
        total = len(primary_plan.tiles)
        fallback_tiles: list[Tile] = []
        primary_tiles: list[Tile] = []
        for tile in primary_plan.tiles:
            if tile.url in withheld:
                # Match the withheld GLO-30 cell to its GLO-90 twin by tag.
                tag = tile.cache_path.parts[0]
                twin = next(
                    (t for t in glo90_plan.tiles if t.cache_path.parts[0] == tag),
                    None,
                )
                if twin is not None:
                    fallback_tiles.append(
                        self._partitioned(twin, AUTO_SOURCE_NAME, "glo90")
                    )
                continue
            primary_tiles.append(self._partitioned(tile, AUTO_SOURCE_NAME, "glo30"))
        if total and len(fallback_tiles) / total > _AUTO_FALLBACK_WARN_FRACTION:
            logger.warning(
                "auto DEM fallback fraction %.0f%% (%d/%d cells) exceeds "
                "%.0f%%: large portions of this ROI are withheld on GLO-30 "
                "and were rescued from GLO-90",
                100.0 * len(fallback_tiles) / total,
                len(fallback_tiles),
                total,
                100.0 * _AUTO_FALLBACK_WARN_FRACTION,
            )

        paths: list[Path] = []
        if primary_tiles:
            primary_hits = self._execute(
                TileSet(
                    allowed_hosts=primary_plan.allowed_hosts,
                    tiles=tuple(primary_tiles),
                )
            )
            paths.extend(primary_hits)
        if fallback_tiles:
            logger.warning(
                "%d withheld GLO-30 cell(s) rescued from GLO-90 fallback tiles",
                len(fallback_tiles),
            )
            paths.extend(
                self._execute(
                    TileSet(
                        allowed_hosts=glo90_plan.allowed_hosts,
                        tiles=tuple(fallback_tiles),
                    )
                )
            )
        resolved = list(dict.fromkeys(paths))
        # Legacy flat hits participate for the primary cells as well.
        for tile in primary_plan.tiles:
            if tile.url in withheld:
                continue
            legacy = self._legacy_hit(tile)
            if legacy is not None and str(legacy) not in {str(p) for p in resolved}:
                resolved.append(legacy)
        tags = self._provenance_tags(fallback="glo90" if fallback_tiles else None)
        return self._write_mosaic(resolved, out, tags=tags)


def get_dem_manager(*, source: str | None = None) -> DEMManager:
    """Return a DEMManager configured from environment variables.

    Reads ``FANINSAR_DEM_CACHE_DIR`` (required), ``FANINSAR_DEM_SOURCE``
    (optional ``<product>`` / ``<product>:<provider>`` compound grammar), and
    ``FANINSAR_DEM_SOURCE_URL`` (https-enforced primary-base override; warned
    when combined with a non-default source).

    Parameters
    ----------
    source : str, optional
        Explicit selection overriding ``FANINSAR_DEM_SOURCE``; ``None``
        defers to the environment and then the ``glo30`` default.

    Returns
    -------
    DEMManager
        The environment-configured manager.

    Raises
    ------
    InvalidProcessingStateError
        If ``FANINSAR_DEM_CACHE_DIR`` is unset or the URL override is not
        https / unsupported for the selected source shape.

    """
    cache_dir = os.environ.get(DEM_CACHE_ENV)
    if not cache_dir:
        message = (
            f"{DEM_CACHE_ENV} is not set; cannot resolve the automatic DEM. "
            "Point it at a folder for the raw DEM tile download cache."
        )
        logger.error(message)
        raise InvalidProcessingStateError(message)
    selection = (
        source
        if source is not None
        else os.environ.get(DEM_SELECTION_ENV, DEFAULT_PRODUCT)
    )
    base_url = os.environ.get(DEM_SOURCE_ENV)
    manager = DEMManager(
        cache_dir=Path(cache_dir),
        source=selection,
        base_url=base_url,
    )
    if base_url is not None:
        default_pair = PRODUCT_DEFAULTS.get(manager.product, {}).get("default")
        if (manager.product, manager.provider) != (DEFAULT_PRODUCT, default_pair):
            logger.warning(
                "FANINSAR_DEM_SOURCE_URL override combined with non-default "
                "source %s@%s; verify the mirror serves the identical "
                "product grid and datum",
                manager.product,
                manager.provider,
            )
    return manager
