"""Geometry core of the general mask operator (PROPOSAL-0039, Slice A).

Provides the :class:`MaskSampler` protocol (``sample(lat, lon) -> bool`` with
``True`` = keep), the :class:`RasterMask` / :class:`VectorMask` adapters, the
:class:`MaskOperator` composition, and the consumption helpers shared with the
water-mask manager: the UTM planar land buffer :func:`buffer_land_utm_km`, the
padded fetch band :func:`padded_fetch_band`, the shared tile-grid snap
:func:`snap_band`, the three-condition antimeridian seam guard
:func:`antimeridian_seam_guard`, :func:`resample_mask_to_grid` (nearest
neighbour only), and :func:`rasterize_to_grid` (uint8 0/1 with 255 = invalid).

Two array conventions are used consistently:

- **Sampler planes** are boolean with ``True`` = keep (valid), mirroring
  :class:`faninsar.processing.geometry.dem.DEMSampler`.
- **Mask products** are uint8 with ``0`` = valid keep, ``1`` = removed
  (water / masked, the ISCE3 ``mask == 1 -> invalid`` convention) and
  ``255`` = invalid where no data exists.

Governing proposals: PROPOSAL-0039 (mask operator, owner-directed UTM planar
buffer design, three-condition seam guard); PROPOSAL-0038 consumes the mask
as a support input at the IFG/unwrap seams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from affine import Affine
    from geopandas import GeoDataFrame, GeoSeries
    from pyproj import Transformer
    from rasterio.io import DatasetReader
    from shapely.geometry.base import BaseGeometry

logger = setup_logger(__name__)

LonLatBounds = tuple[float, float, float, float]
"""Bounds tuple ``(min_lon, min_lat, max_lon, max_lat)`` in degrees."""

_SEAM_TOLERANCE_DEG = 1e-9
"""Padded-band tolerance for the "reaches the +/-180 seam" condition."""


@runtime_checkable
class MaskSampler(Protocol):
    """Sample a boolean mask plane at geodetic coordinates.

    Mirrors :class:`faninsar.processing.geometry.dem.DEMSampler`: samplers
    report ``True`` for **kept** (valid) cells and ``False`` for cells removed
    by the mask (or invalid under it).
    """

    def sample(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> np.ndarray:
        """Return the boolean keep plane (``True`` = keep) for each sample."""


@dataclass(slots=True)
class RasterMask:
    """Sample a single-band raster mask with the pinned fill semantics.

    A cell is *removed* when its source value is excluded by the configured
    predicate. ``threshold`` (values ``>= threshold`` are excluded, e.g. GSW
    occurrence) and ``excluded_values`` membership (e.g. WorldCover water
    class 80) both apply when both are configured; with neither configured the
    raster is treated as a boolean mask and every nonzero value is excluded.
    ``invert=True`` flips the predicate only.

    Fill semantics are pinned by PROPOSAL-0039 G1: target cells outside the
    raster's extent resolve to **valid** (the user's exclusion intent never
    silently deletes data), NoData cells resolve to **invalid**, and
    resampling is **nearest-neighbour only** (bilinear would invent fractional
    values at coastlines).

    Parameters
    ----------
    path : pathlib.Path
        Path to a single-band GeoTIFF (any CRS; non-geographic CRS are
        reprojected on the fly for point lookups).
    excluded_values : frozenset[int]
        Source values whose cells are removed (class-code semantics).
    invert : bool
        Flip the exclusion predicate (apply last, after threshold/membership).
    threshold : float or None
        Continuous-raster knob: values ``>= threshold`` are excluded.
    nodata : float or None
        NoData override; when ``None`` the dataset's own NoData value is used.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.

    """

    path: Path
    excluded_values: frozenset[int] = field(default_factory=frozenset)
    invert: bool = False
    threshold: float | None = None
    nodata: float | None = None
    _dataset: DatasetReader | None = None
    _band: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate that the raster exists and normalize field types."""
        self.path = Path(self.path)
        if not self.path.exists():
            message = f"mask raster does not exist: {self.path}"
            logger.error(message)
            raise FileNotFoundError(message)
        self.excluded_values = frozenset(self.excluded_values)
        self.invert = bool(self.invert)

    def __getstate__(self) -> dict[str, object]:
        """Return a picklable state without the open rasterio handle."""
        return {
            "path": self.path,
            "excluded_values": self.excluded_values,
            "invert": self.invert,
            "threshold": self.threshold,
            "nodata": self.nodata,
            "_band": self._band,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore the sampler state; the dataset reopens lazily."""
        self.path = Path(state["path"])
        self.excluded_values = frozenset(state["excluded_values"])  # type: ignore[arg-type]
        self.invert = bool(state["invert"])
        self.threshold = state["threshold"]  # type: ignore[assignment]
        self.nodata = state["nodata"]  # type: ignore[assignment]
        self._band = state["_band"]  # type: ignore[assignment]
        self._dataset = None

    def _open(self) -> tuple[DatasetReader, np.ndarray]:
        """Open the raster lazily and cache the single band."""
        if self._band is None:
            import rasterio

            dataset = rasterio.open(self.path)
            self._dataset = dataset
            self._band = dataset.read(1)
        dataset = self._dataset
        band = self._band
        if dataset is None or band is None:
            message = f"failed to open mask raster: {self.path}"
            logger.error(message)
            raise RuntimeError(message)
        return dataset, band

    def _extract(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(in_extent, is_nodata, excluded)`` planes at the points.

        Nearest-neighbour lookups use the pixel-area convention (the
        containing pixel, ``floor`` of the fractional pixel coordinate), so an
        identical target grid maps exactly onto itself.
        """
        lat = np.asarray(latitude_deg, dtype=np.float64)
        lon = np.asarray(longitude_deg, dtype=np.float64)
        lat_b, lon_b = np.broadcast_arrays(lat, lon)
        out_shape = lat_b.shape
        xs = lon_b.ravel().astype(np.float64, copy=True)
        ys = lat_b.ravel().astype(np.float64, copy=True)

        dataset, band = self._open()
        crs = dataset.crs
        if crs is not None and crs.to_epsg() != 4326:
            import pyproj as _pyproj

            source_crs = _pyproj.CRS.from_user_input(crs.to_wkt())
            transformer = _pyproj.Transformer.from_crs(
                "EPSG:4326", source_crs, always_xy=True
            )
            xs, ys = transformer.transform(xs, ys)
            xs = np.asarray(xs, dtype=np.float64)
            ys = np.asarray(ys, dtype=np.float64)

        inverse = ~dataset.transform
        cols_f, rows_f = inverse * (xs, ys)
        cols_f = np.asarray(cols_f, dtype=np.float64)
        rows_f = np.asarray(rows_f, dtype=np.float64)
        height, width = band.shape
        in_extent = (
            (cols_f >= 0.0)
            & (cols_f <= float(width))
            & (rows_f >= 0.0)
            & (rows_f <= float(height))
        )
        cols = np.floor(np.clip(cols_f, 0.0, float(width - 1))).astype(np.int64)
        rows = np.floor(np.clip(rows_f, 0.0, float(height - 1))).astype(np.int64)
        values = band[rows, cols]

        nodata = self.nodata if self.nodata is not None else dataset.nodata
        if nodata is None:
            is_nodata = np.zeros(values.shape, dtype=bool)
        elif np.issubdtype(band.dtype, np.floating):
            is_nodata = np.isclose(values, float(nodata))
        else:
            is_nodata = values == nodata

        has_threshold = self.threshold is not None
        has_values = bool(self.excluded_values)
        matched = np.zeros(values.shape, dtype=bool)
        if has_threshold:
            matched |= np.asarray(values, dtype=np.float64) >= float(
                self.threshold  # type: ignore[arg-type]
            )
        if has_values:
            matched |= np.isin(
                values,
                np.asarray(sorted(self.excluded_values), dtype=np.float64),
            )
        if not has_threshold and not has_values:
            matched = np.asarray(values) != 0
        if self.invert:
            matched = np.logical_not(matched)

        # Structural fill rules: out-of-extent cells are valid, never excluded
        # and never NoData (the clipped lookup values are meaningless there).
        is_nodata = np.logical_and(is_nodata, in_extent)
        matched = np.logical_and(matched, in_extent)
        return (
            in_extent.reshape(out_shape),
            is_nodata.reshape(out_shape),
            matched.reshape(out_shape),
        )

    def sample(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> np.ndarray:
        """Return the boolean keep plane for the requested coordinates.

        Parameters
        ----------
        latitude_deg, longitude_deg : numpy.ndarray
            Geodetic sample coordinates in degrees (broadcast against each
            other).

        Returns
        -------
        numpy.ndarray
            Boolean plane with ``True`` = keep. Cells outside the raster
            extent are kept (valid); NoData and excluded cells are not.

        """
        in_extent, is_nodata, excluded = self._extract(latitude_deg, longitude_deg)
        keep = (~in_extent) | (~is_nodata & ~excluded)
        return np.asarray(keep, dtype=bool)

    def close(self) -> None:
        """Close the underlying raster dataset if open."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None
        self._band = None


def _as_shapely_geometries(geometries: object) -> list[BaseGeometry]:
    """Normalize GeoDataFrame/GeoSeries/geometry/sequence input to a list."""
    from shapely.geometry.base import BaseGeometry as _BaseGeometry

    if isinstance(geometries, _BaseGeometry):
        return [geometries]  # type: ignore[list-item]
    geometry_attr = getattr(geometries, "geometry", None)
    if geometry_attr is not None:
        # geopandas GeoDataFrame / GeoSeries, duck-typed so the geometry core
        # never needs a hard geopandas import
        return [geom for geom in geometry_attr if geom is not None]
    if isinstance(geometries, (list, tuple)):
        flattened: list[BaseGeometry] = []
        for item in geometries:
            flattened.extend(_as_shapely_geometries(item))
        return flattened
    message = f"unsupported vector-mask geometries: {type(geometries).__name__}"
    logger.error(message)
    raise TypeError(message)


@dataclass(slots=True)
class VectorMask:
    """Vector geometries marking the removed (masked) region.

    The geometries are interpreted as the region to remove (water polygons for
    the automatic water mask). :meth:`rasterize` burns them onto an exact
    target grid as uint8 ``1`` (removed) / ``0`` (kept); :meth:`sample` reports
    ``True`` = keep for points outside every geometry (boundary points count
    as covered). v1 expects lon/lat (EPSG:4326) geometries, which is what the
    water manager's cached vector layer stores.

    Parameters
    ----------
    geometries : geopandas.GeoDataFrame or geopandas.GeoSeries or \
            shapely geometry or sequence
        The mask geometries. GeoDataFrame/GeoSeries input is consumed via its
        ``geometry`` column (duck-typed, so geopandas is not imported here).

    Raises
    ------
    ValueError
        If the geometries contain no non-empty geometry when used.

    """

    geometries: GeoDataFrame | GeoSeries | BaseGeometry | Sequence[BaseGeometry]
    _union_geometry: BaseGeometry | None = None

    def _cached_union(self) -> BaseGeometry:
        """Union the geometries once and prepare it for repeated queries."""
        import shapely

        if self._union_geometry is None:
            geoms = _as_shapely_geometries(self.geometries)
            geoms = [geom for geom in geoms if not geom.is_empty]
            if not geoms:
                message = "vector mask has no non-empty geometries"
                logger.error(message)
                raise ValueError(message)
            union = shapely.union_all(geoms)
            shapely.prepare(union)
            self._union_geometry = union
        return self._union_geometry

    def rasterize(
        self,
        transform: Affine,
        shape: tuple[int, int],
        *,
        all_touched: bool = False,
    ) -> np.ndarray:
        """Rasterize the geometries onto an exact target grid.

        Parameters
        ----------
        transform : affine.Affine
            Affine transform of the target grid (EPSG:4326 degrees).
        shape : tuple[int, int]
            ``(height, width)`` of the target grid.
        all_touched : bool
            GDAL burn rule: ``False`` burns cells whose center is inside the
            geometry; ``True`` burns every cell the geometry touches (the
            coastline tie-break).

        Returns
        -------
        numpy.ndarray
            uint8 array with ``1`` = covered by the geometries (removed) and
            ``0`` = outside (kept); the shape/transform match the target grid
            exactly.

        """
        from rasterio import features as rio_features

        return np.asarray(
            rio_features.rasterize(
                [(self._cached_union(), 1)],
                out_shape=tuple(shape),
                transform=transform,
                fill=0,
                all_touched=all_touched,
                dtype="uint8",
            ),
            dtype=np.uint8,
        )

    def sample(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> np.ndarray:
        """Return the boolean keep plane for the requested coordinates.

        Parameters
        ----------
        latitude_deg, longitude_deg : numpy.ndarray
            Geodetic sample coordinates in degrees (broadcast against each
            other).

        Returns
        -------
        numpy.ndarray
            Boolean plane with ``True`` = keep (outside every geometry).

        """
        import shapely

        lat = np.asarray(latitude_deg, dtype=np.float64)
        lon = np.asarray(longitude_deg, dtype=np.float64)
        lat_b, lon_b = np.broadcast_arrays(lat, lon)
        covered = shapely.intersects_xy(self._cached_union(), lon_b, lat_b)
        return np.logical_not(np.asarray(covered, dtype=bool))


@dataclass(frozen=True, slots=True)
class MaskOperator:
    """Compose masks by their removed regions (union / intersection / invert).

    A mask sampler reports ``True`` for kept cells; the *mask* region is the
    complement (removed cells, the uint8 ``1`` of the product convention).
    Composition follows the removed-region algebra used throughout
    PROPOSAL-0039 (masks combine as ``valid_mask &= ~mask``):

    - ``union``: removed = union of the constituents' removed regions — a cell
      is kept only if **every** constituent keeps it (AND of keep planes).
    - ``intersection``: removed = intersection of the removed regions — a cell
      is kept if **any** constituent keeps it (OR of keep planes).
    - ``invert``: the complement mask (kept cells become removed and vice
      versa).

    Operators nest: a :class:`MaskOperator` is itself a :class:`MaskSampler`.

    Raises
    ------
    ValueError
        If the mode/operand combination is invalid.

    """

    mode: Literal["union", "intersection", "invert"]
    masks: tuple[MaskSampler, ...]

    def __post_init__(self) -> None:
        """Validate the composition mode and operand count."""
        if self.mode not in {"union", "intersection", "invert"}:
            message = f"unsupported MaskOperator mode: {self.mode!r}"
            logger.error(message)
            raise ValueError(message)
        if self.mode == "invert":
            if len(self.masks) != 1:
                message = "invert requires exactly one mask"
                logger.error(message)
                raise ValueError(message)
            return
        if not self.masks:
            message = f"{self.mode} requires at least one mask"
            logger.error(message)
            raise ValueError(message)

    @classmethod
    def union(cls, *masks: MaskSampler) -> MaskOperator:
        """Return the union of the masks' removed regions."""
        return cls("union", tuple(masks))

    @classmethod
    def intersection(cls, *masks: MaskSampler) -> MaskOperator:
        """Return the intersection of the masks' removed regions."""
        return cls("intersection", tuple(masks))

    @classmethod
    def invert(cls, *masks: MaskSampler) -> MaskOperator:
        """Return the complement of ``mask`` (exactly one mask required)."""
        return cls("invert", masks)

    def sample(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> np.ndarray:
        """Return the composed boolean keep plane for the coordinates.

        Parameters
        ----------
        latitude_deg, longitude_deg : numpy.ndarray
            Geodetic sample coordinates in degrees (broadcast against each
            other).

        Returns
        -------
        numpy.ndarray
            Boolean plane with ``True`` = keep under the composition.

        """
        planes = [
            np.asarray(mask.sample(latitude_deg, longitude_deg), dtype=bool)
            for mask in self.masks
        ]
        if self.mode == "union":
            composed = np.logical_and.reduce(planes)
        elif self.mode == "intersection":
            composed = np.logical_or.reduce(planes)
        else:
            composed = np.logical_not(planes[0])
        return np.asarray(composed, dtype=bool)


# ---------------------------------------------------------------------------
# UTM planar buffer, padded fetch band, tile snap, seam guard (PROPOSAL-0039
# G4; ports of the executed round6_utm_design_audit.py / round5_seam_guard.py)
# ---------------------------------------------------------------------------


def _utm_crs(longitude_deg: float, latitude_deg: float) -> str:
    """Return the auto-UTM CRS for a zone anchor (EPSG:326xx / EPSG:327xx)."""
    zone = int((longitude_deg + 180.0) // 6.0) + 1
    return f"EPSG:{32600 + zone}" if latitude_deg >= 0 else f"EPSG:{32700 + zone}"


def _utm_transformers(
    longitude_deg: float,
    latitude_deg: float,
) -> tuple[Transformer, Transformer]:
    """Return the EPSG:4326 <-> auto-UTM transformer pair (always_xy)."""
    import pyproj

    crs = _utm_crs(longitude_deg, latitude_deg)
    forward = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    reverse = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return forward, reverse


def _buffer_projected(geom: BaseGeometry, radius_m: float) -> BaseGeometry:
    """Buffer a projected geometry, guarding boundary-less parts.

    shapely 2.x reports ``boundary is None`` for GeometryCollection objects
    (verified on shapely 2.1.2, empty and non-empty alike); such parts stay on
    the identity path instead of crashing the buffer primitive
    (PROPOSAL-0039 round-4 blocker guard).
    """
    from shapely.geometry import GeometryCollection

    if isinstance(geom, GeometryCollection):
        parts: list[BaseGeometry] = []
        for member in geom.geoms:
            buffered = _buffer_projected(member, radius_m)
            if buffered is not None and not buffered.is_empty:
                parts.append(buffered)
        if not parts:
            return GeometryCollection([])
        import shapely

        return shapely.union_all(parts)  # type: ignore[return-value]
    if geom.is_empty or geom.boundary is None:
        return geom
    return geom.buffer(radius_m)


def buffer_land_utm_km(
    geom: BaseGeometry,
    buffer_km: float,
    *,
    zone_lon: float,
    zone_lat: float,
) -> BaseGeometry:
    """Planar-buffer a lon/lat geometry in the auto-UTM zone.

    The buffer runs in projected metres (EPSG:326xx north / EPSG:327xx south
    selected from ``zone_lon`` / ``zone_lat``, the Stack merge-grid zone
    convention), so no cos(lat) degree conversion, latitude cap, or geodesic
    disk-sum machinery is involved (PROPOSAL-0039 G4, owner-directed UTM
    design). Executed self-audit: agreement with a high-fidelity geodesic
    reference is 0.019 % at 70 N and ``buffer_km=0`` is the identity within
    UTM round-trip float noise.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        Lon/lat (EPSG:4326) geometry; Polygon, MultiPolygon, and
        GeometryCollection inputs are supported (a part whose boundary is
        ``None`` under shapely 2.x is kept unchanged instead of crashing).
    buffer_km : float
        Buffer width in kilometres; ``0`` is the identity (the geometry is
        still round-tripped through the UTM zone like every other width).
    zone_lon, zone_lat : float
        Zone anchor (the ROI / footprint centroid in degrees).

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        The buffered geometry transformed back to lon/lat.

    Raises
    ------
    ValueError
        If ``buffer_km`` is negative.

    """
    import shapely.ops

    if buffer_km < 0:
        message = f"buffer_km must be >= 0, got {buffer_km}"
        logger.error(message)
        raise ValueError(message)
    if geom.is_empty:
        return geom
    forward, reverse = _utm_transformers(zone_lon, zone_lat)
    projected = shapely.ops.transform(forward.transform, geom)
    if buffer_km > 0:
        projected = _buffer_projected(projected, float(buffer_km) * 1e3)
    return shapely.ops.transform(reverse.transform, projected)


def padded_fetch_band(
    roi_bounds: LonLatBounds | Sequence[float],
    buffer_km: float,
    *,
    zone_lon: float,
    zone_lat: float,
) -> LonLatBounds:
    """Expand ROI bounds by ``buffer_km`` in UTM metres, back to lon/lat.

    The band fetched for the vector water layer is
    ``snap(ROI bounds expanded by buffer_km in UTM)`` (PROPOSAL-0039 G4): the
    padding folds into the vector-cache identity and guarantees buffered water
    from a neighboring tile is never silently missing at ROI edges (executed
    round-5 blocker fix: a 944 m strip at a tile boundary is now masked).

    Parameters
    ----------
    roi_bounds : LonLatBounds or sequence of float
        Raw ROI bounds ``(min_lon, min_lat, max_lon, max_lat)`` in degrees.
    buffer_km : float
        Padding width in kilometres, applied in UTM metres on every side.
    zone_lon, zone_lat : float
        Zone anchor (the ROI / footprint centroid in degrees).

    Returns
    -------
    LonLatBounds
        The padded bounds ``(min_lon, min_lat, max_lon, max_lat)`` in degrees;
        ``buffer_km=0`` reproduces the input within UTM round-trip noise.

    .. note::
        When the single-zone UTM round trip is degenerate (a near-hemisphere
        ROI projects past its zone's transverse-Mercator singularity and the
        back-transform yields non-finite values), the padding falls back to
        the round-5 degree-space approximation
        (``buffer_km / (111.32 * cos(max |lat|))`` in longitude) so the band
        stays finite for the seam guard.

    """
    forward, reverse = _utm_transformers(zone_lon, zone_lat)
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in roi_bounds)
    xs, ys = forward.transform(
        [min_lon, max_lon, min_lon, max_lon],
        [min_lat, min_lat, max_lat, max_lat],
    )
    pad_m = float(buffer_km) * 1e3
    xs = np.asarray(xs, dtype=np.float64) + np.array(
        [-pad_m, pad_m, -pad_m, pad_m], dtype=np.float64
    )
    ys = np.asarray(ys, dtype=np.float64) + np.array(
        [-pad_m, -pad_m, pad_m, pad_m], dtype=np.float64
    )
    lons, lats = reverse.transform(xs, ys)
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    if not np.all(np.isfinite(lons)) or not np.all(np.isfinite(lats)):
        # Degenerate single-zone UTM round trip: a near-hemisphere ROI sits
        # far outside its zone's usable extent (e.g. an exact-180 span around
        # lon 0 projects past the transverse-Mercator singularity). Fall back
        # to the round-5 degree-space padding so the band stays finite and
        # the seam guard receives well-defined inputs.
        max_abs_lat = min(max(abs(min_lat), abs(max_lat)), 89.0)
        lon_pad = float(buffer_km) / (111.32 * math.cos(math.radians(max_abs_lat)))
        lat_pad = float(buffer_km) / 111.32
        lons = np.array([min_lon - lon_pad, max_lon + lon_pad], dtype=np.float64)
        lats = np.array([min_lat - lat_pad, max_lat + lat_pad], dtype=np.float64)
    return (
        float(min(lons)),
        float(min(lats)),
        float(max(lons)),
        float(max(lats)),
    )


def snap_band(
    band: LonLatBounds | Sequence[float],
    tile_size_deg: float,
) -> LonLatBounds:
    """Snap a lon/lat band outwards onto the source tile grid.

    The single shared snap used by both the padded fetch band and the seam
    guard, so fetch coverage and fail-closed reasoning can never disagree
    (PROPOSAL-0039 G4).

    Parameters
    ----------
    band : LonLatBounds or sequence of float
        Bounds ``(min_lon, min_lat, max_lon, max_lat)`` in degrees.
    tile_size_deg : float
        Tile size of the source grid in degrees (10 for GSW, 3 for
        WorldCover).

    Returns
    -------
    LonLatBounds
        Tile-grid-aligned bounds covering ``band`` (snapped outwards on every
        side).

    """
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in band)
    tile = float(tile_size_deg)
    s_left = math.floor((min_lon + 180.0) / tile) * tile - 180.0
    s_right = math.floor((max_lon + 180.0) / tile) * tile + tile - 180.0
    s_bottom = math.floor((min_lat + 90.0) / tile) * tile - 90.0
    s_top = math.floor((max_lat + 90.0) / tile) * tile + tile - 90.0
    return (s_left, s_bottom, s_right, s_top)


def antimeridian_seam_guard(
    raw_bounds: LonLatBounds | Sequence[float],
    *,
    padded_band: LonLatBounds | Sequence[float],
) -> tuple[bool, str]:
    """Fail-closed antimeridian guard evaluated on raw bounds.

    Evaluated in order **before** bounds normalization and tile enumeration
    (PROPOSAL-0039 G4; ports the executed ``round5_seam_guard.py`` predicate
    onto the padded band):

    - **(c)** any raw longitude outside ``[-180, 180]`` (unwrapped-lon ROI)
      fails closed;
    - **(a)** a raw planar longitude span > 180 deg (bbox wrap evader) fails
      closed;
    - **(b)** the tile-snapped padded fetch band reaching +/-180 fails closed
      (the near-seam consumption crash: buffered disks wrap the seam, an
      intermittent GEOS topology failure).

    Parameters
    ----------
    raw_bounds : LonLatBounds or sequence of float
        Raw ROI bounds ``(min_lon, min_lat, max_lon, max_lat)`` in degrees.
    padded_band : LonLatBounds or sequence of float
        The tile-snapped padded fetch band, i.e.
        ``snap_band(padded_fetch_band(...))`` — not the raw bounds.

    Returns
    -------
    tuple[bool, str]
        ``(ok, reason)``; ``ok=False`` means the caller must raise a
        structured fail-closed error before any buffering happens.

    """
    min_lon, _min_lat, max_lon, _max_lat = (float(v) for v in raw_bounds)
    # (c) unwrapped / out-of-range longitudes
    if min_lon < -180.0 or max_lon > 180.0:
        return False, "out-of-range longitude (outside [-180, 180])"
    # (a) planar wrap evader
    if (max_lon - min_lon) > 180.0:
        return False, "planar longitude span > 180 deg (seam wrap)"
    # (b) padded fetch band reaching the seam
    band_min_lon, _band_min_lat, band_max_lon, _band_max_lat = (
        float(v) for v in padded_band
    )
    if (
        180.0 - band_max_lon <= _SEAM_TOLERANCE_DEG
        or band_min_lon + 180.0 <= _SEAM_TOLERANCE_DEG
    ):
        return False, "padded fetch band reaches the +/-180 seam"
    return True, "ok"


# ---------------------------------------------------------------------------
# Grid consumption helpers
# ---------------------------------------------------------------------------


def resample_mask_to_grid(
    raster_mask: RasterMask,
    transform: Affine,
    shape: tuple[int, int],
) -> np.ndarray:
    """Extract a :class:`RasterMask` onto an exact target grid.

    Nearest-neighbour only (the globally pinned rule for boolean masks —
    bilinear would invent fractional values at coastlines). A target grid
    identical to the source grid (same transform/shape) maps exactly onto
    itself.

    Parameters
    ----------
    raster_mask : RasterMask
        The user raster mask (``excluded_values`` / ``threshold`` /
        ``invert`` configure the predicate).
    transform : affine.Affine
        Affine transform of the target grid (EPSG:4326 degrees).
    shape : tuple[int, int]
        ``(height, width)`` of the target grid.

    Returns
    -------
    numpy.ndarray
        uint8 mask with ``0`` = valid keep, ``1`` = removed, ``255`` =
        invalid (NoData). Cells whose center falls outside the raster extent
        resolve to ``0`` (valid), never to ``255``.

    """
    rows, cols = np.indices(tuple(shape), dtype=np.float64)
    xs, ys = transform * (cols + 0.5, rows + 0.5)
    in_extent, is_nodata, excluded = raster_mask._extract(ys, xs)
    out = np.zeros(in_extent.shape, dtype=np.uint8)
    out[in_extent & is_nodata] = 255
    out[in_extent & ~is_nodata & excluded] = 1
    return out


def rasterize_to_grid(
    geometries: GeoDataFrame | GeoSeries | BaseGeometry | Sequence[BaseGeometry],
    transform: Affine,
    shape: tuple[int, int],
    *,
    all_touched: bool = False,
    validity: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterize vector geometries onto an exact target grid.

    The consumption step of the water pipeline (PROPOSAL-0039): the buffered
    water vector is rasterized directly onto the DEM mosaic grid
    (``mask_resolution_m`` overrides) as a uint8 binary mask.

    Parameters
    ----------
    geometries : geopandas.GeoDataFrame or geopandas.GeoSeries or \
            shapely geometry or sequence
        Geometries marking the removed region.
    transform : affine.Affine
        Affine transform of the target grid (EPSG:4326 degrees).
    shape : tuple[int, int]
        ``(height, width)`` of the target grid.
    all_touched : bool
        GDAL burn rule (see :meth:`VectorMask.rasterize`).
    validity : numpy.ndarray or None
        Optional boolean plane of the target grid (``True`` = valid, e.g. DEM
        cells with data). Cells marked invalid become ``255``.

    Returns
    -------
    numpy.ndarray
        uint8 mask with ``1`` = covered by the geometries (water / removed),
        ``0`` = valid keep, and ``255`` = invalid where the validity plane
        marks no data.

    Raises
    ------
    ValueError
        If ``validity`` does not match the target grid shape.

    """
    from rasterio import features as rio_features

    shapes = [
        (geom, 1) for geom in _as_shapely_geometries(geometries) if not geom.is_empty
    ]
    out = np.asarray(
        rio_features.rasterize(
            shapes,
            out_shape=tuple(shape),
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype="uint8",
        ),
        dtype=np.uint8,
    )
    if validity is not None:
        valid_plane = np.asarray(validity, dtype=bool)
        if valid_plane.shape != out.shape:
            message = (
                f"validity plane shape {valid_plane.shape} does not match "
                f"the target grid {out.shape}"
            )
            logger.error(message)
            raise ValueError(message)
        out = np.where(valid_plane, out, np.uint8(255)).astype(np.uint8)
    return out
