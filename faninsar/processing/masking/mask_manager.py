"""Automatic water-mask manager (PROPOSAL-0039, Slice C).

Mirrors the PROPOSAL-0030 DEM manager
(:mod:`faninsar.processing.geometry.dem_manager`): a selectable water source
(:mod:`faninsar.processing.masking.mask_sources`) is executed through the
audited PROPOSAL-0030 transport engine
(:func:`faninsar.processing.geometry.dem_transport.fetch_plan` — reused as-is,
no new socket code), raw tiles cache under ``<product>-<provider>/``, and the
water layer is extracted, polygonized once, and cached as GeoJSON with
provenance under ``<product>-<provider>/vectors/<identity>/``.

Pipeline (water category, PROPOSAL-0039 "MaskManager pipeline"):

1. **Fetch** (:meth:`MaskManager.fetch_water`) — atomic at the SET level: any
   tile failure fails the whole call with the structured
   :class:`MaskProviderUnavailableError`; a partial tile set is never
   mosaicked or vectorized (completed tiles stay cached and a retry resumes).
2. **Extract** — boolean water: ``value >= threshold`` (GSW occurrence,
   default 50) and/or membership in ``excluded_values`` (WorldCover water
   class 80); ``invert`` flips the predicate last; NoData cells never count
   as water.
3. **Vectorize** — :func:`rasterio.features.shapes` + shapely
   :func:`~shapely.union_all` dissolve + simplification (metric tolerance,
   default 30 m) + minimum-area filtering (default 1 km2), both applied in
   the ROI's auto-UTM zone via :func:`~faninsar.processing.masking.mask.\
buffer_land_utm_km`'s zone convention.
4. **Cache** — GeoJSON written with temp-file + ``os.replace`` under
   ``<product>-<provider>/vectors/<identity>/`` with a provenance sidecar;
   the ``identity`` is a sha256 hexdigest over a canonical (sorted-keys JSON)
   serialization of ``(source, normalized padded bounds, resolved plan
   URL/version, threshold, excluded_values, invert, simplify_tolerance_m,
   min_area_km2)``. The resolved version comes from the transport response
   ETag/Last-Modified (weak-validator prefix and quotes stripped); when both
   are absent the version is explicitly ``"unversioned"``. Any parameter or
   source-version change produces a new identity and re-extracts; bounds are
   normalized (tile-snapped) before hashing so float jitter cannot defeat
   cross-run reuse.
5. **Consume** (:meth:`MaskManager.resolve_auto_mask`) — cached vector ->
   UTM planar land buffer -> rasterize onto the DEM mosaic transform/shape
   -> uint8 0/1 GeoTIFF (1 = water/removed, 255 = invalid where the DEM has
   no data). The rasterized mask is cached keyed by (vector-layer digest,
   buffer_km, grid) so unchanged configuration never re-rasterizes.

Failure contract: provider failures always raise the structured error. There
is no warning/skip branch in the v1 manager; a missing water layer must not be
silently mistaken for a successful scientific result.

Governing proposals: PROPOSAL-0039 (mask manager pipeline, owner-directed
UTM planar buffer design, three-condition antimeridian seam guard, fetch
atomicity); PROPOSAL-0030 (transport engine, cache-partition and structured
outage patterns mirrored here).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry.dem_sources import _bounds_tuple
from faninsar.processing.geometry.dem_transport import (
    MAX_ATTEMPTS,
    REQUEST_TIMEOUT,
    FetchPlan,
    TileSet,
    expand_tile_parts,
    fetch_plan,
    redact_url,
)
from faninsar.processing.masking.mask import (
    LonLatBounds,
    _utm_transformers,
    antimeridian_seam_guard,
    padded_fetch_band,
    rasterize_to_grid,
    snap_band,
)
from faninsar.processing.masking.mask_sources import (
    AUTO_SOURCE_NAME,
    WATER_CLASSIFICATION_VERSION,
    MaskSource,
    MaskSourceUnavailableError,
    get_mask_source,
)
from faninsar.query import BoundingBox

if TYPE_CHECKING:
    from collections.abc import Sequence

    from affine import Affine
    from rasterio.crs import CRS
    from shapely.geometry.base import BaseGeometry

logger = setup_logger(__name__)

__all__ = [
    "DEFAULT_MASK_NAME",
    "INLAND_WATER_BUFFER_M",
    "MASK_BUFFER_ENV",
    "MASK_CACHE_ENV",
    "MASK_SOURCE_ENV",
    "MASK_SOURCE_URL_ENV",
    "OCEAN_SHORE_KEEP_M",
    "UNVERSIONED",
    "FetchResult",
    "MaskManager",
    "MaskProviderUnavailableError",
    "WaterLayer",
    "get_mask_manager",
    "realize_water",
    "realize_water_mask",
    "resolve_auto_mask",
]

#: Required raw-download cache root environment (per-identity partitions).
MASK_CACHE_ENV = "FANINSAR_MASK_CACHE_DIR"
#: Selection grammar environment (``water`` / ``water:<provider>``).
MASK_SOURCE_ENV = "FANINSAR_MASK_SOURCE"
#: Base-URL override for the selected source (https enforced, no userinfo).
MASK_SOURCE_URL_ENV = "FANINSAR_MASK_SOURCE_URL"
#: Land-buffer width override in kilometres (``MaskManager.buffer_km``).
MASK_BUFFER_ENV = "FANINSAR_MASK_BUFFER_KM"
#: Output file name under ``<output_dir>/mask/``.
DEFAULT_MASK_NAME = "water_mask.tif"
#: Explicit version recorded when the transport carries no validator.
UNVERSIONED = "unversioned"

Bounds = BoundingBox | tuple[float, float, float, float]
FailurePolicy = Literal["error"]
_ON_FAILURE_POLICIES = frozenset({"error"})

# Water-class operation constants.  The negative ocean operation keeps a
# one-kilometre shore strip in the effective land ROI; inland water is kept at
# its authoritative footprint.  They are deliberately fixed in v1 rather
# than exposed as a public classifier/protocol surface.
OCEAN_SHORE_KEEP_M = 1000.0
INLAND_WATER_BUFFER_M = 0.0
_CLASSIFICATION_VERSION = WATER_CLASSIFICATION_VERSION


class MaskProviderUnavailableError(InvalidProcessingStateError):
    """Structured outage contract for an exhausted mask provider.

    Attributes
    ----------
    product, provider, host
        Identity of the failing selection.
    failure_class
        One of ``auth``, ``forbidden``, ``coverage``, ``upstream-outage``.
    attempts
        Transport attempts spent on the failing request(s).

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
    ) -> None:
        """Build the structured outage error."""
        super().__init__(message)
        self.product = product
        self.provider = provider
        self.host = host
        self.failure_class = failure_class
        self.attempts = attempts


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Result of one atomic set-level water fetch.

    Attributes
    ----------
    band
        Normalized (tile-snapped) fetch band that was covered.
    tiles
        Resolved tile paths (cache hits and fresh fetches alike).
    source_version
        Normalized transport version (ETag/Last-Modified) or
        :data:`UNVERSIONED`.

    """

    band: LonLatBounds
    tiles: tuple[Path, ...]
    source_version: str


@dataclass(frozen=True, slots=True)
class WaterLayer:
    """Cached vectorized water layer.

    Attributes
    ----------
    path
        GeoJSON path under ``<product>-<provider>/vectors/<identity>/``.
    identity
        sha256 identity digest of the layer (see module docstring).
    source_version
        Resolved source version or :data:`UNVERSIONED`.
    band
        Normalized (tile-snapped) padded fetch band covered by the layer.
    from_cache
        Whether this call reused an existing cached layer.
    feature_count
        Number of GeoJSON features (polygon parts after dissolve,
        simplification, and minimum-area filtering).

    """

    path: Path
    identity: str
    source_version: str
    band: LonLatBounds
    from_cache: bool
    feature_count: int
    categories: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _host_of(url: str) -> str:
    """Return the lowercase hostname of ``url``."""
    return (urllib.parse.urlsplit(url).hostname or "").lower()


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


def _canonical_json(payload: object) -> str:
    """Serialize a payload canonically (sorted keys, compact separators)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex(text: str) -> str:
    """Return the sha256 hexdigest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_version(etag: str | None, last_modified: str | None) -> str:
    """Normalize ETag/Last-Modified into one source-version string.

    The weak-validator prefix (``W/``) and surrounding quotes are stripped
    (PROPOSAL-0039 round-4 revision); when both headers are absent the
    version is explicitly :data:`UNVERSIONED`.
    """
    if etag:
        normalized = etag.strip()
        if normalized[:2] in {"W/", "w/"}:
            normalized = normalized[2:]
        normalized = normalized.strip('"').strip()
        if normalized:
            return normalized
    if last_modified:
        normalized = last_modified.strip()
        if normalized:
            return normalized
    return UNVERSIONED


def _polygon_parts(geometry: BaseGeometry) -> list[BaseGeometry]:
    """Return the non-empty polygon parts of a (possibly multi) geometry."""
    if geometry.is_empty:
        return []
    members = getattr(geometry, "geoms", None)
    if members is None:
        return [geometry] if geometry.geom_type in {"Polygon", "MultiPolygon"} else []
    return [
        part
        for part in members
        if not part.is_empty and part.geom_type in {"Polygon", "MultiPolygon"}
    ]


def _tile_crs_to_lonlat(crs: object) -> object | None:
    """Return a lon/lat projector for a non-geographic tile CRS (else None).

    The returned callable maps one shapely geometry from the tile CRS to
    EPSG:4326; geographic (EPSG:4326) tiles need no projection.
    """
    rasterio_crs = getattr(crs, "to_epsg", None)
    if callable(rasterio_crs) and rasterio_crs() == 4326:
        return None
    if crs is None:
        return None
    import pyproj
    import shapely.ops

    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return lambda geometry: shapely.ops.transform(transformer.transform, geometry)


def _geojson_payload(
    parts: Sequence[BaseGeometry], categories: Sequence[str] | None = None
) -> dict[str, object]:
    """Build a GeoJSON FeatureCollection with optional water categories.

    ``category`` is intentionally a plain feature property rather than a
    public classifier API.  It is the authoritative label attached to the
    materialized water result and is covered by the layer identity.
    """
    import shapely

    if categories is None:
        categories = ("water",) * len(parts)
    if len(categories) != len(parts):
        message = "water geometry categories must align with polygons"
        logger.error(message)
        raise ValueError(message)
    features = [
        {
            "type": "Feature",
            "properties": {"category": category},
            "geometry": json.loads(shapely.to_geojson(part)),
        }
        for part, category in zip(parts, categories, strict=True)
    ]
    return {"type": "FeatureCollection", "features": features}


def _count_features(path: Path) -> int:
    """Count the features of a cached GeoJSON layer."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    features = payload.get("features", [])
    return len(features) if isinstance(features, list) else 0


def _load_layer_geometries(path: Path) -> BaseGeometry:
    """Load a cached GeoJSON layer back into one dissolved geometry."""
    from shapely.geometry import Polygon, shape

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    geoms = []
    for feature in payload.get("features", []):
        geom = shape(feature["geometry"])
        if not geom.is_empty:
            geoms.append(geom)
    if not geoms:
        return Polygon()
    import shapely

    return shapely.union_all(geoms)


def _load_layer_features(path: Path) -> tuple[list[BaseGeometry], tuple[str, ...]]:
    """Load materialized water geometries and their qualified categories."""
    from shapely.geometry import shape

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    geometries: list[BaseGeometry] = []
    categories: list[str] = []
    for feature in payload.get("features", []):
        geometry = shape(feature["geometry"])
        if geometry.is_empty:
            continue
        properties = feature.get("properties", {})
        # Keep a neutral label for legacy/manual layers.  The manager decides
        # whether an unqualified feature is usable under the active policies.
        category = properties.get("category", "water")
        if category not in {"ocean", "inland", "water"}:
            message = f"invalid water layer category {category!r}"
            logger.error(message)
            raise InvalidProcessingStateError(message)
        geometries.append(geometry)
        categories.append(str(category))
    return geometries, tuple(categories)


def _padded_water_bounds(
    raw: LonLatBounds, manager: MaskManager, tile_size_deg: float
) -> LonLatBounds:
    """Return the fetch band padded for every category operation."""
    buffer_km = max(
        float(manager.buffer_km),
        float(manager.ocean_shore_keep_m) / 1000.0,
        float(manager.inland_water_buffer_m) / 1000.0,
    )
    zone_lon = (raw[0] + raw[2]) / 2.0
    zone_lat = (raw[1] + raw[3]) / 2.0
    padded = padded_fetch_band(raw, buffer_km, zone_lon=zone_lon, zone_lat=zone_lat)
    band = snap_band(padded, tile_size_deg)
    manager._guard_seam(raw, band)
    return padded


def _atomic_write_json(path: Path, payload: object) -> None:
    """Atomically write JSON via a unique temp file + ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{secrets.token_hex(8)}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_raster(
    path: Path,
    mask: np.ndarray,
    transform: Affine,
    crs: CRS | None,
    *,
    tags: dict[str, str],
) -> None:
    """Atomically write the uint8 mask GeoTIFF (nodata 255)."""
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{secrets.token_hex(8)}.tmp")
    profile = {
        "driver": "GTiff",
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "count": 1,
        "dtype": "uint8",
        "nodata": 255,
        "crs": crs if crs is not None else rasterio.crs.CRS.from_epsg(4326),
        "transform": transform,
        "compress": "deflate",
    }
    try:
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(mask, 1)
            if tags:
                dst.update_tags(**tags)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _tag_value(value: object) -> str:
    """Render one provenance value as a GeoTIFF tag string."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set, frozenset)):
        return ",".join(str(item) for item in sorted(value))  # type: ignore[type-var, call-overload]
    return str(value)


def _raster_cache_key(
    buffer_km: float,
    transform: Affine,
    shape: tuple[int, int],
    crs: CRS | None,
    validity: np.ndarray,
    roi_bounds: LonLatBounds | None = None,
) -> str:
    """Digest the rasterized-mask cache key (vector digest added by caller).

    The grid component covers the affine transform, shape, CRS, and the
    DEM's valid-data plane (the 255 cells of the product are a function of
    the DEM NoData pattern, so a changed validity plane must never reuse a
    cached raster rendered under the old one).
    """
    validity_digest = hashlib.sha256(
        np.ascontiguousarray(validity, dtype=np.uint8).tobytes()
    ).hexdigest()
    payload = {
        "buffer_km": float(buffer_km),
        "transform": [
            float(v)
            for v in (
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f,
            )
        ],
        "shape": [int(shape[0]), int(shape[1])],
        "crs": str(crs) if crs is not None else "",
        "validity": validity_digest,
        "roi_bounds": None
        if roi_bounds is None
        else [round(float(value), 9) for value in roi_bounds],
    }
    return _sha256_hex(_canonical_json(payload))


# ---------------------------------------------------------------------------
# MaskManager
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MaskManager:
    """Resolve, cache, vectorize, and rasterize an automatic water mask.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Root of the raw tile / vector / raster cache. Tiles land under
        ``<cache_dir>/<product>-<provider>/`` (the registry bakes the
        partition into the planned cache paths).
    source : str, optional
        Selection grammar value (``water`` / ``water:<provider>``).
        ``None`` (the default) defers to ``FANINSAR_MASK_SOURCE`` and then
        to the ``water`` auto alias (GSW occurrence).
    on_failure : "error"
        Provider failures always raise the structured
        :class:`MaskProviderUnavailableError` in v1.
    buffer_km : float
        Land-buffer width in kilometres owned by the water pipeline (default
        1.0; ``get_mask_manager`` overrides it from
        ``FANINSAR_MASK_BUFFER_KM``). The padded fetch band, the UTM planar
        buffer, and the raster-cache key all derive from this single value.
    threshold : float, optional
        Occurrence threshold override; ``None`` resolves the source default
        (GSW 50).
    excluded_values : frozenset of int, optional
        Class-code override; ``None`` resolves the source default
        (WorldCover ``{80}``).
    invert : bool
        Flip the extraction predicate (applied last).
    simplify_tolerance_m : float
        Metric simplification tolerance applied in the auto-UTM zone
        (proposal-pinned default 30 m).
    min_area_km2 : float
        Minimum polygon area kept after simplification (proposal-pinned
        default 1 km2).
    max_workers : int
        Shared stream budget handed to the transport engine.
    chunked_threshold : int
        Transport-engine parameter (reserved; mirrors the DEM manager).
    base_url : str, optional
        https base-URL override for the selected source (the
        ``FANINSAR_MASK_SOURCE_URL`` mirror escape); mirror URLs must not
        embed credentials (userinfo netlocs are rejected fail-closed).

    Raises
    ------
    ValueError
        If the selection, ``on_failure`` policy, or source name is invalid.
    InvalidProcessingStateError
        If the base-URL override is not https or embeds credentials.

    """

    cache_dir: Path
    source: str | None = None
    on_failure: FailurePolicy = "error"
    buffer_km: float = 1.0
    threshold: float | None = None
    excluded_values: frozenset[int] | None = None
    invert: bool = False
    simplify_tolerance_m: float = 30.0
    min_area_km2: float = 1.0
    ocean_shore_keep_m: float = OCEAN_SHORE_KEEP_M
    inland_water_buffer_m: float = INLAND_WATER_BUFFER_M
    max_workers: int = 8
    chunked_threshold: int = 4
    base_url: str | None = None
    source_entry: MaskSource = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve the selection, validate the buffer, apply the override."""
        self.cache_dir = Path(self.cache_dir)
        if not float(self.buffer_km) >= 0.0:
            message = f"buffer_km must be a number >= 0; got {self.buffer_km!r}"
            logger.error(message)
            raise ValueError(message)
        self.buffer_km = float(self.buffer_km)
        for field_name in ("ocean_shore_keep_m", "inland_water_buffer_m"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0.0:
                message = f"{field_name} must be a finite number >= 0; got {value!r}"
                logger.error(message)
                raise ValueError(message)
            setattr(self, field_name, value)
        if self.on_failure not in _ON_FAILURE_POLICIES:
            message = (
                f"invalid on_failure policy {self.on_failure!r}: expected one "
                f"of {sorted(_ON_FAILURE_POLICIES)}"
            )
            logger.error(message)
            raise ValueError(message)
        if self.source is not None:
            entry = get_mask_source(str(self.source))
        else:
            selection = os.environ.get(MASK_SOURCE_ENV, AUTO_SOURCE_NAME)
            entry = get_mask_source(selection)
        if self.base_url is not None:
            entry = self._apply_base_override(entry)
        object.__setattr__(self, "source_entry", entry)

    def _apply_base_override(self, entry: MaskSource) -> MaskSource:
        """Clone ``entry`` with the https, credential-free base override."""
        base = str(self.base_url)
        if not base.lower().startswith("https://"):
            message = (
                f"{MASK_SOURCE_URL_ENV}-style overrides must use https, "
                f"got {self.base_url!r}"
            )
            logger.error(message)
            raise InvalidProcessingStateError(message)
        parts = urllib.parse.urlsplit(base)
        if parts.username is not None or parts.password is not None:
            message = (
                f"{MASK_SOURCE_URL_ENV} mirror URLs must not embed "
                "credentials: a userinfo (user:pass@host) netloc is rejected "
                "fail-closed at runtime"
            )
            logger.error(message)
            raise InvalidProcessingStateError(message)
        if hasattr(entry, "base_url"):
            return replace(entry, base_url=base)
        message = (
            f"mask source {entry.name!r} ({entry.provider}) does not accept "
            "a base URL override; its endpoint is pinned"
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
    def partition_name(self) -> str:
        """Opaque hyphen-joined cache partition label for this identity."""
        return f"{self.source_entry.product}-{self.source_entry.provider}"

    @property
    def partition_dir(self) -> Path:
        """Directory holding this identity's downloads, vectors, rasters."""
        return self.cache_dir / self.partition_name

    @property
    def effective_threshold(self) -> float | None:
        """Resolved occurrence threshold (manager override or source default)."""
        if self.threshold is not None:
            return float(self.threshold)
        if self.source_entry.threshold is None:
            return None
        return float(self.source_entry.threshold)

    @property
    def effective_excluded_values(self) -> frozenset[int]:
        """Resolved class codes (manager override or source default)."""
        if self.excluded_values is not None:
            return frozenset(self.excluded_values)
        return frozenset(self.source_entry.excluded_values)

    def _identity(self, band: LonLatBounds, version: str) -> str:
        """Digest the vector-layer identity over its canonical payload.

        The payload covers (source, normalized padded bounds, resolved plan
        URL/version, threshold, excluded_values, invert, simplification and
        minimum-area settings); bounds are tile-snapped and rounded so float
        jitter cannot defeat cross-run cache reuse (PROPOSAL-0039 G3).
        """
        entry = self.source_entry
        payload = {
            "source": f"{entry.product}@{entry.provider}",
            "bounds": [round(float(v), 9) for v in band],
            "plan_url": getattr(entry, "base_url", "") or "",
            "source_version": version,
            "threshold": self.effective_threshold,
            "excluded_values": sorted(int(v) for v in self.effective_excluded_values),
            "invert": bool(self.invert),
            "simplify_tolerance_m": float(self.simplify_tolerance_m),
            "min_area_km2": float(self.min_area_km2),
            "ocean_shore_keep_m": float(self.ocean_shore_keep_m),
            "inland_water_buffer_m": float(self.inland_water_buffer_m),
            "classification_version": _CLASSIFICATION_VERSION,
        }
        return _sha256_hex(_canonical_json(payload))

    def _provenance(self, version: str) -> dict[str, object]:
        """Build the provenance sidecar payload for one vector layer."""
        entry = self.source_entry
        return {
            "mask_product": entry.product,
            "mask_provider": entry.provider,
            "mask_retrieved": datetime.now(tz=UTC).date().isoformat(),
            "threshold": self.effective_threshold,
            "excluded_values": sorted(int(v) for v in self.effective_excluded_values),
            "invert": bool(self.invert),
            "source_version": version,
            "classification_version": _CLASSIFICATION_VERSION,
            "ocean_shore_keep_m": float(self.ocean_shore_keep_m),
            "inland_water_buffer_m": float(self.inland_water_buffer_m),
        }

    # -- planning and guard -------------------------------------------------

    def _guard_seam(
        self,
        raw: tuple[float, float, float, float],
        band: LonLatBounds,
    ) -> None:
        """Fail closed on antimeridian conditions before any fetch."""
        ok, reason = antimeridian_seam_guard(raw, padded_band=band)
        if not ok:
            message = (
                "water-mask fetch rejected by the antimeridian seam guard "
                f"({reason}); bounds={raw} fail closed before any tile "
                "fetch (PROPOSAL-0039)"
            )
            logger.error(message)
            raise InvalidProcessingStateError(message)

    def _prepare(self, bounds: Bounds) -> tuple[LonLatBounds, FetchPlan, str]:
        """Normalize bounds, guard the seam, plan, and resolve the version.

        Planning runs on the raw (padded) bounds — the registry's tile
        enumeration is inclusive on the upper edge, exactly matching the
        tiles spanned by the exclusive-end snapped band that normalizes the
        identity (the Slice-B registry test pins this equality).
        """
        entry = self.source_entry
        raw = _bounds_tuple(bounds)
        band = snap_band(raw, entry.tile_size_deg)
        self._guard_seam(raw, band)
        plan = entry.plan(raw)
        etag, last_modified = self._probe_version(plan)
        return band, plan, _normalize_version(etag, last_modified)

    def _probe_version(self, plan: FetchPlan) -> tuple[str | None, str | None]:
        """Best-effort HEAD probe of the first planned tile for validators.

        The GSW/WorldCover URLs carry no version, so the resolved version is
        derived from the transport response (ETag / Last-Modified). The
        session resolves through :mod:`dem_transport` on every call so test
        fakes apply uniformly; probe failures degrade to ``(None, None)``
        (explicitly unversioned) instead of blocking cached-tile reuse.
        """
        if not isinstance(plan, TileSet) or not plan.tiles:
            return None, None
        url = expand_tile_parts(plan.tiles[0])[0].url
        from faninsar.processing.geometry import dem_transport as _transport

        try:
            response = _transport.thread_local_session().head(
                url, timeout=REQUEST_TIMEOUT
            )
            status = int(response.status_code)
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
            response.close()
        except AssertionError:
            raise
        except Exception as exc:
            logger.debug(
                "mask source-version probe failed for %s: %s",
                redact_url(url),
                type(exc).__name__,
            )
            return None, None
        if status != 200:
            return None, None
        return etag, last_modified

    # -- plan execution -----------------------------------------------------

    def _execute(self, plan: FetchPlan) -> list[Path]:
        """Execute one plan, translating engine failures to the contract."""
        try:
            return fetch_plan(
                plan,
                self.cache_dir,
                max_workers=self.max_workers,
                chunked_threshold=self.chunked_threshold,
            )
        except AssertionError:
            raise
        except Exception as exc:
            raise self._outage_error(exc) from exc

    def _outage_error(self, exc: BaseException) -> MaskProviderUnavailableError:
        """Wrap a transport failure into the structured outage contract.

        The structured outage is logged before it is raised.
        """
        entry = self.source_entry
        failure_class = _classify_failure(exc)
        attempts = 1 if failure_class in {"auth", "forbidden"} else MAX_ATTEMPTS
        host = _host_of(getattr(entry, "base_url", "") or "")
        message = (
            f"Mask provider unavailable: {entry.product}@{entry.provider} "
            f"({failure_class}) after {attempts} attempt(s): {exc}. Fetch is "
            "atomic at the tile-set level: completed tiles stay cached and a "
            "later retry resumes; switch providers manually via the source "
            "selection if needed."
        )
        if self.on_failure == "error":
            logger.error(message)
        else:
            logger.debug(message)
        return MaskProviderUnavailableError(
            message,
            product=entry.product,
            provider=entry.provider,
            host=host,
            failure_class=failure_class,
            attempts=attempts,
        )

    def _resolve_tiles(self, plan: FetchPlan) -> list[Path]:
        """Resolve every planned tile against the cache, fetching only misses.

        The raw tiles cache under ``<product>-<provider>/`` (the registry
        plans partitioned cache paths). Any tile failure raises through
        :meth:`_execute`, so the returned set is always complete.
        """
        resolved: list[Path] = []
        missing: list = []
        seen: set[str] = set()
        for raw_tile in plan.tiles:
            for unit in expand_tile_parts(raw_tile):
                target = self.cache_dir / unit.cache_path
                if target.is_file():
                    key = str(target)
                    if key not in seen:
                        seen.add(key)
                        resolved.append(target)
                    continue
                key = str(unit.cache_path)
                if key not in seen:
                    seen.add(key)
                    missing.append(unit)
        if missing:
            executed = self._execute(replace(plan, tiles=tuple(missing)))
            for path in executed:
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    resolved.append(path)
        return resolved

    # -- extraction and vectorization ----------------------------------------

    def _tile_water_polygons(self, path: Path) -> list[BaseGeometry]:
        """Extract the boolean water polygons of one cached tile.

        ``threshold`` (values >= threshold, GSW occurrence) and
        ``excluded_values`` membership combine as OR; ``invert`` flips the
        predicate last; NoData cells never count as water (mirroring the
        :class:`~faninsar.processing.masking.mask.RasterMask` fill
        semantics, where the NoData rule is structural, not part of the
        predicate).
        """
        import rasterio
        from rasterio import features as rio_features
        from shapely.geometry import shape

        threshold = self.effective_threshold
        excluded = self.effective_excluded_values
        with rasterio.open(path) as dataset:
            values = dataset.read(1)
            transform = dataset.transform
            crs = dataset.crs
            nodata = dataset.nodata

        if threshold is not None:
            matched = np.asarray(values, dtype=np.float64) >= float(threshold)
        elif excluded:
            matched = np.isin(values, np.asarray(sorted(excluded), dtype=values.dtype))
        else:
            matched = np.asarray(values) != 0
        if self.invert:
            matched = np.logical_not(matched)
        if nodata is not None:
            if np.issubdtype(values.dtype, np.floating):
                is_nodata = np.isclose(
                    np.asarray(values, dtype=np.float64), float(nodata)
                )
            else:
                is_nodata = values == nodata
            matched = np.logical_and(matched, np.logical_not(is_nodata))

        to_lonlat = _tile_crs_to_lonlat(crs)
        polygons: list[BaseGeometry] = []
        for geom, _value in rio_features.shapes(
            matched.astype("uint8"), mask=matched, transform=transform
        ):
            polygon = shape(geom)
            if to_lonlat is not None:
                polygon = to_lonlat(polygon)
            if not polygon.is_empty:
                polygons.append(polygon)
        return polygons

    def _vectorize(self, tiles: list[Path], band: LonLatBounds) -> BaseGeometry:
        """Classify, repair, operate, and dissolve water polygons.

        Source-qualified classes are assigned before any dissolve.  Raster
        products without an authoritative ocean/inland field fail closed when
        category policies differ; no fetch-edge shoreline inference is done. The
        category-specific operation is then applied in one local UTM zone.
        Narrow inland components are retained because geometry aspect and area
        are not a scientific classification rule.

        The method returns a geometry for compatibility with the P0039
        private seam.  Category-bearing output is available through
        :meth:`_vectorize_features` and is what gets cached.
        """
        parts, _categories = self._vectorize_features(tiles, band)
        import shapely
        from shapely.geometry import Polygon

        return shapely.union_all(parts) if parts else Polygon()

    def _vectorize_features(
        self, tiles: list[Path], band: LonLatBounds
    ) -> tuple[list[BaseGeometry], tuple[str, ...]]:
        """Return operated water polygons and source-qualified categories.

        Raster products do not encode an ocean/inland distinction.  They fail
        closed when category policies differ; no fetch-boundary inference or
        implicit inland classification is performed.  A source adapter may
        return ``(geometry, category)`` records from ``_tile_water_polygons``;
        those categories are validated before any dissolve and remain the
        only basis for category-specific operations.
        """
        import shapely
        import shapely.ops

        records: list[tuple[BaseGeometry, str]] = []
        for path in tiles:
            for item in self._tile_water_polygons(path):
                category = None
                geometry = item
                if isinstance(item, tuple) and len(item) == 2:
                    geometry, category = item
                if category in {None, "water"}:
                    if float(self.ocean_shore_keep_m) != float(
                        self.inland_water_buffer_m
                    ):
                        message = (
                            "water source did not qualify ocean versus inland "
                            "geometry, but category policies differ"
                        )
                        logger.error(message)
                        raise InvalidProcessingStateError(message)
                    category = "inland"
                if category not in {"ocean", "inland"}:
                    message = (
                        f"water source returned an unqualified category {category!r}"
                    )
                    logger.error(message)
                    raise InvalidProcessingStateError(message)
                if not hasattr(geometry, "is_empty"):
                    message = "water source returned a non-geometric polygon"
                    logger.error(message)
                    raise InvalidProcessingStateError(message)
                if not geometry.is_empty:
                    records.append((geometry, category))
        if not records:
            return [], ()

        # Classify before dissolve.  Dissolving first can merge separately
        # qualified classes and make their policy choice unknowable.
        classified: dict[str, list[BaseGeometry]] = {"ocean": [], "inland": []}
        for geometry, category in records:
            candidate = shapely.make_valid(geometry)
            if not candidate.is_empty:
                classified[category].append(candidate)

        zone_lon = (band[0] + band[2]) / 2.0
        zone_lat = (band[1] + band[3]) / 2.0
        forward, reverse = _utm_transformers(zone_lon, zone_lat)
        result: list[BaseGeometry] = []
        categories: list[str] = []
        operations = {
            "ocean": -float(self.ocean_shore_keep_m),
            "inland": float(self.inland_water_buffer_m),
        }
        min_area_m2 = float(self.min_area_km2) * 1e6
        for category in ("ocean", "inland"):
            if not classified[category]:
                continue
            projected = [
                shapely.ops.transform(forward.transform, geometry)
                for geometry in classified[category]
            ]
            operated = [
                shapely.make_valid(geometry.buffer(operations[category]))
                for geometry in projected
            ]
            dissolved = shapely.make_valid(shapely.union_all(operated))
            simplified = dissolved.simplify(
                float(self.simplify_tolerance_m), preserve_topology=True
            )
            # Do not use area/aspect-ratio guesses to decide whether an
            # unqualified narrow river is real.  The source classification is
            # authoritative; minimum-area filtering applies only to explicitly
            # qualified ocean polygons.
            if category == "inland":
                min_area_m2 = 0.0
            for part in _polygon_parts(shapely.make_valid(simplified)):
                if category == "ocean" and part.area < min_area_m2:
                    continue
                geographic = shapely.ops.transform(reverse.transform, part)
                if not geographic.is_empty:
                    result.append(geographic)
                    categories.append(category)
        return result, tuple(categories)

    # -- public entry points -------------------------------------------------

    def fetch_water(self, bounds: Bounds) -> FetchResult:
        """Execute the source plan for the bounds; atomic at the set level.

        Any tile failure (404, timeout, terminal error) fails the whole call
        with :class:`MaskProviderUnavailableError`; a partial tile set is
        never returned. Completed tiles stay cached under
        ``<product>-<provider>/`` so a later retry resumes.

        Parameters
        ----------
        bounds : BoundingBox or tuple
            Padded fetch bounds ``(min_lon, min_lat, max_lon, max_lat)`` in
            EPSG:4326. :meth:`resolve_auto_mask` passes
            ``snap(padded_fetch_band(roi_bounds, buffer_km))``; direct
            callers should do the same so the padding folds into the
            vector-cache identity.

        Returns
        -------
        FetchResult
            The normalized band, resolved tile paths, and resolved source
            version.

        Raises
        ------
        InvalidProcessingStateError
            If the antimeridian seam guard rejects the bounds (structural).
        MaskProviderUnavailableError
            When any tile of the set fails.

        """
        band, plan, version = self._prepare(bounds)
        tiles = self._resolve_tiles(plan)
        return FetchResult(band=band, tiles=tuple(tiles), source_version=version)

    def get_water_layer(self, bounds: Bounds) -> WaterLayer:
        """Fetch, extract, vectorize, and cache the water layer for bounds.

        The vector layer caches at
        ``<product>-<provider>/vectors/<identity>/layer.geojson`` with a
        provenance sidecar; the write is atomic (temp file + os.replace). A
        cache hit skips the tile fetch and the vectorization entirely (the
        identity resolution itself is a single best-effort HEAD probe).

        Parameters
        ----------
        bounds : BoundingBox or tuple
            Padded fetch bounds in EPSG:4326 (see :meth:`fetch_water`).

        Returns
        -------
        WaterLayer
            The cached (or freshly written) vector layer.

        Raises
        ------
        InvalidProcessingStateError
            If the antimeridian seam guard rejects the bounds (structural).
        MaskProviderUnavailableError
            When any tile of the fetch set fails.
        MaskSourceUnavailableError
            For sources that arrive already vector (not wired in v1).

        """
        entry = self.source_entry
        band, plan, version = self._prepare(bounds)
        identity = self._identity(band, version)
        vector_dir = self.partition_dir / "vectors" / identity
        layer_path = vector_dir / "layer.geojson"
        if layer_path.is_file():
            _, cached_categories = _load_layer_features(layer_path)
            if "water" in cached_categories and float(self.ocean_shore_keep_m) != float(
                self.inland_water_buffer_m
            ):
                message = (
                    "cached water layer lacks authoritative ocean/inland category, "
                    "but category policies differ"
                )
                logger.error(message)
                raise InvalidProcessingStateError(message)
            return WaterLayer(
                path=layer_path,
                identity=identity,
                source_version=version,
                band=band,
                from_cache=True,
                feature_count=_count_features(layer_path),
                categories=cached_categories,
            )
        if entry.extraction == "vector":
            message = (
                f"mask source {entry.name!r} arrives already vector; raster "
                "extraction/polygonization is not applicable and the "
                "Overpass provider is not wired in v1"
            )
            logger.error(message)
            raise MaskSourceUnavailableError(message)
        tiles = self._resolve_tiles(plan)
        parts, categories = self._vectorize_features(tiles, band)
        _atomic_write_json(layer_path, _geojson_payload(parts, categories))
        _atomic_write_json(vector_dir / "provenance.json", self._provenance(version))
        logger.info(
            "water layer vectorized: %s features=%d identity=%s version=%s",
            layer_path,
            len(parts),
            identity,
            version,
        )
        return WaterLayer(
            path=layer_path,
            identity=identity,
            source_version=version,
            band=band,
            from_cache=False,
            feature_count=len(parts),
            categories=categories,
        )

    def resolve_auto_mask(
        self,
        roi_bounds: Bounds,
        *,
        dem_path: str | Path,
        output_dir: str | Path,
    ) -> Path:
        """Resolve the buffered binary water mask onto the DEM mosaic grid.

        Cached vector -> UTM planar buffer (:func:`buffer_land_utm_km` in
        the ROI's auto-UTM zone) -> :func:`rasterize_to_grid` on the DEM
        mosaic transform/shape -> uint8 0/1 GeoTIFF (1 = water/removed,
        255 = invalid where the DEM has no data), written under
        ``<output_dir>/mask/`` with provenance tags. The rasterized mask is
        cached under ``<product>-<provider>/rasters/<identity>/`` keyed by
        (vector-layer digest, :attr:`buffer_km`, grid + DEM data mask), so an
        unchanged configuration never re-rasterizes. The antimeridian seam
        guard runs on the raw ROI bounds with the padded band BEFORE any
        fetch.

        Parameters
        ----------
        roi_bounds : BoundingBox or tuple
            Raw ROI bounds ``(min_lon, min_lat, max_lon, max_lat)`` in
            EPSG:4326 (padding is derived here, in UTM metres).
        dem_path : path
            The DEM mosaic GeoTIFF whose transform/shape define the target
            grid (EPSG:4326; other CRS fail closed in v1).
        output_dir : path
            Run output directory receiving ``mask/<DEFAULT_MASK_NAME>``.

        Returns
        -------
        pathlib.Path
            The written mask path.

        Raises
        ------
        InvalidProcessingStateError
            If the seam guard rejects the ROI (structural, independent of
            ``on_failure``) or the DEM grid CRS is not EPSG:4326.
        MaskProviderUnavailableError
            When the provider fails and :attr:`on_failure` is ``error``.

        """
        import rasterio

        entry = self.source_entry
        # Fetch context must contain every category operation's neighbourhood.
        # ``buffer_km`` remains accepted as a compatibility alias for callers
        # that used the P0039 manager, but can never make the context smaller
        # than the pinned ocean shoreline keep distance.
        raw = _bounds_tuple(roi_bounds)
        padded = _padded_water_bounds(raw, self, entry.tile_size_deg)
        buffer_km = max(
            float(self.buffer_km),
            float(self.ocean_shore_keep_m) / 1000.0,
            float(self.inland_water_buffer_m) / 1000.0,
        )

        dem_file = Path(dem_path)
        with rasterio.open(dem_file) as dataset:
            crs = dataset.crs
            if crs is not None and crs.to_epsg() != 4326:
                message = (
                    f"DEM grid CRS {crs} is not EPSG:4326; the v1 water "
                    "mask rasterizes onto EPSG:4326 DEM mosaics only "
                    "(fail closed)"
                )
                logger.error(message)
                raise InvalidProcessingStateError(message)
            transform = dataset.transform
            grid_shape = (dataset.height, dataset.width)
            validity = dataset.read_masks(1) != 0

        layer = self.get_water_layer(padded)

        raster_key = _raster_cache_key(
            buffer_km, transform, grid_shape, crs, validity, raw
        )
        cached = self.partition_dir / "rasters" / layer.identity / f"{raster_key}.tif"
        if cached.is_file():
            with rasterio.open(cached) as src:
                mask = src.read(1)
            logger.info("water mask raster cache hit: %s", cached)
        else:
            geometries, _categories = _load_layer_features(layer.path)
            import shapely
            from shapely.geometry import Polygon

            roi_geometry = shapely.box(*raw)
            clipped = [geometry.intersection(roi_geometry) for geometry in geometries]
            operated = shapely.union_all(clipped) if clipped else Polygon()
            mask = rasterize_to_grid(
                [operated], transform, grid_shape, validity=validity
            )
            _atomic_write_raster(cached, mask, transform, crs, tags={})

        out_path = Path(output_dir) / "mask" / DEFAULT_MASK_NAME
        _atomic_write_raster(
            out_path,
            mask,
            transform,
            crs,
            tags=self._raster_tags(layer, buffer_km),
        )
        logger.info(
            "water mask resolved: %s (identity=%s, buffer_km=%s)",
            out_path,
            layer.identity,
            buffer_km,
        )
        return out_path

    def _raster_tags(self, layer: WaterLayer, buffer_km: float) -> dict[str, str]:
        """Build the GeoTIFF provenance tags for one resolved mask."""
        tags = {
            key: _tag_value(value)
            for key, value in self._provenance(layer.source_version).items()
        }
        tags["mask_buffer_km"] = str(float(buffer_km))
        tags["ocean_shore_keep_m"] = str(float(self.ocean_shore_keep_m))
        tags["inland_water_buffer_m"] = str(float(self.inland_water_buffer_m))
        tags["classification_version"] = _CLASSIFICATION_VERSION
        tags["mask_identity"] = layer.identity
        return tags


def get_mask_manager(
    *,
    source: str | None = None,
    on_failure: FailurePolicy = "error",
    buffer_km: float | None = None,
) -> MaskManager:
    """Return a MaskManager configured from environment variables.

    Reads ``FANINSAR_MASK_CACHE_DIR`` (required), ``FANINSAR_MASK_SOURCE``
    (optional ``water`` / ``water:<provider>`` grammar),
    ``FANINSAR_MASK_SOURCE_URL`` (https-enforced primary-base override;
    mirror URLs embedding userinfo ``user:pass@host`` are rejected
    fail-closed at runtime), and ``FANINSAR_MASK_BUFFER_KM`` (optional land
    buffer in kilometres; must be >= 0, garbage or negative values fail
    closed).

    Parameters
    ----------
    source : str, optional
        Explicit selection overriding ``FANINSAR_MASK_SOURCE``; ``None``
        defers to the environment and then the ``water`` default.
    on_failure : "error"
        Provider failures always raise the structured outage error.
    buffer_km : float, optional
        Explicit land-buffer width overriding ``FANINSAR_MASK_BUFFER_KM``;
        ``None`` defers to the environment and then the 1.0 default.

    Returns
    -------
    MaskManager
        The environment-configured manager.

    Raises
    ------
    InvalidProcessingStateError
        If ``FANINSAR_MASK_CACHE_DIR`` is unset or the URL override is not
        https / embeds credentials / is unsupported for the source shape.
    ValueError
        If the buffer override (explicit or environmental) is not a number
        >= 0.

    """
    cache_dir = os.environ.get(MASK_CACHE_ENV)
    if not cache_dir:
        message = (
            f"{MASK_CACHE_ENV} is not set; cannot resolve the automatic "
            "water mask. Point it at a folder for the raw mask tile "
            "download cache."
        )
        logger.error(message)
        raise InvalidProcessingStateError(message)
    selection = source if source is not None else os.environ.get(MASK_SOURCE_ENV)
    raw_buffer = buffer_km if buffer_km is not None else os.environ.get(MASK_BUFFER_ENV)
    try:
        resolved_buffer = 1.0 if raw_buffer is None else float(raw_buffer)
    except (TypeError, ValueError) as error:
        message = f"{MASK_BUFFER_ENV} must be a number >= 0; got {raw_buffer!r}"
        logger.exception(message)
        raise ValueError(message) from error
    return MaskManager(
        cache_dir=Path(cache_dir),
        source=selection,
        on_failure=on_failure,
        buffer_km=resolved_buffer,
        base_url=os.environ.get(MASK_SOURCE_URL_ENV),
    )


def realize_water(
    recipe: object,
    *,
    grid: object | None = None,
    cache_dir: str | Path | None = None,
) -> object:
    """Materialize a deferred ``Mask.from_water`` recipe.

    A recipe must carry finite bounds; provider selection and extraction
    policy are resolved here, so constructing a deferred recipe remains a
    zero-I/O operation.  With no ``grid`` a category-bearing ``VectorMask``
    is returned.  Supplying the mask core's ``GridSpec`` returns its
    nearest-neighbour ``RasterMask`` instead.

    Parameters
    ----------
    recipe : object
        A recipe mapping or a deferred ``VectorMask`` created by
        :meth:`faninsar.processing.masking.mask.Mask.from_water`.
    grid : object, optional
        Target ``GridSpec``.  Omit to receive the authoritative vector
        realization.
    cache_dir : pathlib.Path, optional
        Raw/vector cache root.  Defaults to ``FANINSAR_MASK_CACHE_DIR``.

    Returns
    -------
    object
        A ``VectorMask`` or ``RasterMask`` from the mask core.

    Raises
    ------
    ValueError
        If bounds are absent, non-finite, or malformed.
    InvalidProcessingStateError
        If no cache directory is configured.

    """
    deferred = getattr(recipe, "_water_recipe", None)
    if deferred is None:
        deferred = recipe
    if not isinstance(deferred, Mapping):
        message = "water realization requires a deferred water recipe mapping"
        logger.error(message)
        raise TypeError(message)
    bounds = deferred.get("bounds")
    if bounds is None:
        message = "deferred water recipe requires finite bounds at realization"
        logger.error(message)
        raise ValueError(message)
    try:
        raw_bounds = tuple(float(value) for value in _bounds_tuple(bounds))
    except (TypeError, ValueError) as error:
        message = "deferred water recipe bounds must contain four finite numbers"
        logger.exception(message)
        raise ValueError(message) from error
    if len(raw_bounds) != 4 or not np.all(np.isfinite(raw_bounds)):
        message = "deferred water recipe bounds must contain four finite numbers"
        logger.error(message)
        raise ValueError(message)
    provider = str(deferred.get("provider", "auto"))
    if provider == "auto":
        selection = AUTO_SOURCE_NAME
    elif provider == AUTO_SOURCE_NAME or provider.startswith(f"{AUTO_SOURCE_NAME}:"):
        selection = provider
    else:
        selection = f"{AUTO_SOURCE_NAME}:{provider}"
    policy = deferred.get("policy", {})
    if policy is None:
        policy = {}
    if not isinstance(policy, Mapping):
        message = "deferred water recipe policy must be a mapping"
        logger.error(message)
        raise TypeError(message)
    accepted = {
        "threshold",
        "excluded_values",
        "invert",
        "simplify_tolerance_m",
        "min_area_km2",
        "ocean_shore_keep_m",
        "inland_water_buffer_m",
        "max_workers",
        "chunked_threshold",
    }
    unknown = sorted(set(policy) - accepted)
    if unknown:
        message = (
            f"deferred water recipe policy contains unsupported fields: {unknown!r}"
        )
        logger.error(message)
        raise ValueError(message)
    options = {key: value for key, value in policy.items() if key in accepted}
    if cache_dir is None and not os.environ.get(MASK_CACHE_ENV):
        message = f"{MASK_CACHE_ENV} is not set; cannot realize deferred water mask"
        logger.error(message)
        raise InvalidProcessingStateError(message)
    manager = MaskManager(
        cache_dir=Path(cache_dir)
        if cache_dir is not None
        else Path(os.environ[MASK_CACHE_ENV]),
        source=selection,
        **options,
    )
    padded_bounds = _padded_water_bounds(
        raw_bounds, manager, manager.source_entry.tile_size_deg
    )
    layer = manager.get_water_layer(padded_bounds)
    geometries, categories = _load_layer_features(layer.path)
    # The padded band is fetch context only.  The deferred recipe's original
    # ROI is the authoritative output extent and identity input.
    import shapely

    roi_geometry = shapely.box(*raw_bounds)
    clipped = [geometry.intersection(roi_geometry) for geometry in geometries]
    from faninsar.processing.masking.mask import Mask

    vector = Mask.from_vector(
        clipped,
        crs="EPSG:4326",
        categories=categories,
        provenance={
            "mask_identity": layer.identity,
            "source_version": layer.source_version,
            "classification_version": _CLASSIFICATION_VERSION,
            "roi_bounds": raw_bounds,
        },
    )
    if grid is None:
        return vector
    return vector.to_raster(grid)


def realize_water_mask(
    recipe: object,
    *,
    grid: object | None = None,
    cache_dir: str | Path | None = None,
) -> object:
    """Alias for :func:`realize_water` used by integration callers."""
    return realize_water(recipe, grid=grid, cache_dir=cache_dir)


def resolve_auto_mask(
    bounds: Bounds,
    *,
    dem_path: str | Path,
    output_dir: str | Path,
    buffer_km: float | None = None,
    on_failure: FailurePolicy = "error",
    **kwargs: object,
) -> Path:
    """Module-level automatic water-mask convenience (mirrors resolve_auto_dem).

    Builds the manager via :func:`get_mask_manager` (environment-driven,
    with ``on_failure`` and ``buffer_km`` applied), resolves the buffered
    binary mask onto the DEM grid. Provider failures always raise
    :class:`MaskProviderUnavailableError`.

    Parameters
    ----------
    bounds : BoundingBox or tuple
        Raw ROI bounds ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
    dem_path : path
        The DEM mosaic GeoTIFF defining the target grid.
    output_dir : path
        Run output directory receiving ``mask/<DEFAULT_MASK_NAME>``.
    buffer_km : float, optional
        Land buffer width in kilometres; ``None`` defers to
        ``FANINSAR_MASK_BUFFER_KM`` and then the 1.0 default.
    on_failure : "error"
        Provider failures always raise the structured outage error.
    **kwargs
        Extra :class:`MaskManager` configuration fields (``threshold``,
        ``excluded_values``, ``invert``, ``simplify_tolerance_m``,
        ``min_area_km2``, ``max_workers``, ``chunked_threshold``) applied on
        top of the environment-built manager.

    Returns
    -------
    pathlib.Path
        The written mask path.

    """
    source = kwargs.pop("source", None)
    manager = get_mask_manager(
        source=source,  # type: ignore[arg-type]
        on_failure=on_failure,
        buffer_km=buffer_km,
    )
    if kwargs:
        manager = replace(manager, **kwargs)  # type: ignore[arg-type]
    return manager.resolve_auto_mask(bounds, dem_path=dem_path, output_dir=output_dir)
