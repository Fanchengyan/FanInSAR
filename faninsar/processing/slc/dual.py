"""Dual-coordinate SLC objects with transform-cache integration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.coordinates import CoordinateSystem, GeoGrid, RadarGrid
from faninsar.processing.dem import DEM, ConstantDEM
from faninsar.processing.errors import reject_invalid_state
from faninsar.processing.geometry import (
    RadarGeometryModel,
    TransformCacheKey,
    read_transform_cache,
    write_transform_cache,
)
from faninsar.processing.geometry.prepare_production import run_geo2rdr, run_rdr2geo

if TYPE_CHECKING:
    from pathlib import Path

    from faninsar.processing.contracts import SLCProduct
    from faninsar.processing.geometry.transforms import TransformResult
    from faninsar.processing.runtime.types import DeviceLike

logger = setup_logger(__name__)

ProcessingGridChoice = Literal["radar", "geo", "auto"]


@dataclass(frozen=True, slots=True)
class RadarSLC:
    """Radar-coordinate SLC with optional transform-cache path."""

    product: SLCProduct
    samples: np.ndarray
    cache_root: Path | None = None

    def __post_init__(self) -> None:
        """Validate radar grid identity and complex sample shape."""
        if self.product.coordinate_system is not CoordinateSystem.RADAR:
            reject_invalid_state("RadarSLC requires a radar-grid product")
        if self.samples.ndim != 2:
            reject_invalid_state("RadarSLC samples must be 2-D")
        if self.samples.shape != self.product.grid.shape:
            reject_invalid_state("RadarSLC samples must match the product grid")
        if not np.iscomplexobj(self.samples):
            reject_invalid_state("RadarSLC samples must be complex")

    @property
    def grid(self) -> RadarGrid:
        """Return the radar grid metadata."""
        grid = self.product.grid
        if not isinstance(grid, RadarGrid):
            reject_invalid_state("RadarSLC product grid is not a RadarGrid")
        return grid

    def rdr2geo(
        self,
        *,
        device: DeviceLike,
        dem: DEM | None = None,
        geo_grid: GeoGrid | None = None,
        use_cache: bool = True,
    ) -> GeoSLC:
        """Geocode this SLC onto a geographic grid using rdr2geo LUTs.

        Parameters
        ----------
        device : DeviceLike
            Required production device (``auto`` resolves to cpu or cuda).
        dem : DEM, optional
            Height sampler. Defaults to a zero-height ellipsoid.
        geo_grid : GeoGrid, optional
            Target geographic grid. When omitted, a coarse grid is derived from
            the radar-corner geocoding of the current burst window.
        use_cache : bool, optional
            Persist and reuse transform LUTs under ``cache_root`` when set.

        Returns
        -------
        GeoSLC
            Geocoded complex SLC on the target geographic grid.

        Notes
        -----
        Geographic conversion is a phase-safe resampling operation with residual
        loss, not a mathematically lossless inverse of radar coordinates.

        """
        dem_sampler = dem if dem is not None else ConstantDEM(0.0)
        model = RadarGeometryModel.from_radar_grid(self.grid, self.product.orbit)
        height, width = self.grid.shape
        az, rg = np.meshgrid(
            np.arange(height, dtype=np.float64),
            np.arange(width, dtype=np.float64),
            indexing="ij",
        )

        cache_key = TransformCacheKey(
            product_id=self.product.acquisition_id,
            direction="rdr2geo",
            dem_identity=type(dem_sampler).__name__,
            orbit_source=self.product.orbit.source,
            grid_shape=self.grid.shape,
        )
        transform = self._load_or_build_rdr2geo(
            model=model,
            az=az,
            rg=rg,
            dem_sampler=dem_sampler,
            cache_key=cache_key,
            use_cache=use_cache,
            device=device,
        )

        target_grid = geo_grid or _geo_grid_from_transform(transform)
        geocoded = _radar_to_geo_resample(
            self.samples,
            transform,
            target_grid,
        )
        geo_product = replace(
            self.product,
            grid=target_grid,
            samples=replace(
                self.product.samples,
                uri=f"{self.product.samples.uri}#geo",
                shape=target_grid.shape,
            ),
        )
        return GeoSLC(
            product=geo_product,
            samples=geocoded,
            cache_root=self.cache_root,
            source_radar_id=self.product.acquisition_id,
        )

    def _load_or_build_rdr2geo(
        self,
        *,
        model: RadarGeometryModel,
        az: np.ndarray,
        rg: np.ndarray,
        dem_sampler: DEM,
        cache_key: TransformCacheKey,
        use_cache: bool,
        device: DeviceLike,
    ) -> TransformResult:
        if use_cache and self.cache_root is not None:
            store = self.cache_root / f"{cache_key.as_path_stem()}.zarr"
            if store.exists():
                _, cached = read_transform_cache(store)
                return cached
        transform = run_rdr2geo(model, az, rg, dem_sampler, device=device)
        if use_cache and self.cache_root is not None:
            write_transform_cache(self.cache_root, cache_key, transform)
        return transform


@dataclass(frozen=True, slots=True)
class GeoSLC:
    """Geographic-coordinate SLC with optional inverse transform path."""

    product: SLCProduct
    samples: np.ndarray
    cache_root: Path | None = None
    source_radar_id: str | None = None

    def __post_init__(self) -> None:
        """Validate geographic grid identity and complex sample shape."""
        if self.product.coordinate_system is not CoordinateSystem.GEO:
            reject_invalid_state("GeoSLC requires a geographic-grid product")
        if self.samples.ndim != 2:
            reject_invalid_state("GeoSLC samples must be 2-D")
        if self.samples.shape != self.product.grid.shape:
            reject_invalid_state("GeoSLC samples must match the product grid")
        if not np.iscomplexobj(self.samples):
            reject_invalid_state("GeoSLC samples must be complex")

    @property
    def grid(self) -> GeoGrid:
        """Return the geographic grid metadata."""
        grid = self.product.grid
        if not isinstance(grid, GeoGrid):
            reject_invalid_state("GeoSLC product grid is not a GeoGrid")
        return grid

    def geo2rdr(
        self,
        radar_grid: RadarGrid,
        *,
        device: DeviceLike,
        dem: DEM | None = None,
        height_m: float = 0.0,
    ) -> RadarSLC:
        """Resample this geocoded SLC back onto a radar grid.

        Parameters
        ----------
        radar_grid : RadarGrid
            Target radar grid.
        device : DeviceLike
            Required production device (``auto`` resolves to cpu or cuda).
        dem : DEM, optional
            Unused placeholder for DEM-aware inverse paths.
        height_m : float, optional
            Constant height used when projecting geo centres to radar.

        Returns
        -------
        RadarSLC
            Complex SLC on the requested radar grid.

        Notes
        -----
        This is a phase-safe resampling with quantified residual loss, never a
        lossless mathematical inverse of ``rdr2geo``.

        """
        _ = dem  # reserved for DEM-aware inverse composition
        model = RadarGeometryModel.from_radar_grid(radar_grid, self.product.orbit)
        lat, lon = _geo_centres(self.grid)
        transform = run_geo2rdr(model, lat, lon, height_m, device=device)
        radar_samples = _geo_to_radar_resample(
            self.samples,
            transform,
            radar_grid.shape,
        )
        radar_product = replace(
            self.product,
            grid=radar_grid,
            samples=replace(
                self.product.samples,
                uri=f"{self.product.samples.uri}#radar",
                shape=radar_grid.shape,
            ),
        )
        return RadarSLC(
            product=radar_product,
            samples=radar_samples,
            cache_root=self.cache_root,
        )


def choose_processing_grid(
    preference: ProcessingGridChoice,
    *,
    has_radar: bool,
    has_geo: bool,
) -> Literal["radar", "geo"]:
    """Resolve a processing-grid preference to an explicit branch."""
    if preference == "radar":
        if not has_radar:
            reject_invalid_state("radar processing requested but no RadarSLC available")
        return "radar"
    if preference == "geo":
        if not has_geo:
            reject_invalid_state("geo processing requested but no GeoSLC available")
        return "geo"
    if has_radar:
        return "radar"
    if has_geo:
        return "geo"
    return reject_invalid_state("auto processing grid requires a radar or geo SLC")


def _geo_grid_from_transform(transform: TransformResult) -> GeoGrid:
    valid = transform.converged
    if not np.any(valid):
        reject_invalid_state("cannot build geo grid without converged rdr2geo samples")
    lat = transform.latitude_deg[valid]
    lon = transform.longitude_deg[valid]
    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    height, width = transform.latitude_deg.shape
    # Affine: (a, b, c, d, e, f) = (x_res, 0, x_min, 0, -y_res, y_max)
    x_res = (lon_max - lon_min) / max(width - 1, 1)
    y_res = (lat_max - lat_min) / max(height - 1, 1)
    return GeoGrid(
        shape=(height, width),
        crs="EPSG:4326",
        transform=(x_res, 0.0, lon_min, 0.0, -y_res, lat_max),
    )


def _geo_centres(grid: GeoGrid) -> tuple[np.ndarray, np.ndarray]:
    height, width = grid.shape
    a, _b, c, _d, e, f = grid.transform
    cols = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    lon = c + cols * a
    lat = f + rows * e
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return lat_grid, lon_grid


def _radar_to_geo_resample(
    samples: np.ndarray,
    transform: TransformResult,
    target_grid: GeoGrid,
) -> np.ndarray:
    """Nearest-neighbour place radar samples onto a geo grid via rdr2geo LUTs."""
    height, width = target_grid.shape
    out = np.zeros((height, width), dtype=samples.dtype)
    a, _b, c, _d, e, f = target_grid.transform
    valid = transform.converged
    if not np.any(valid):
        return out
    lat = transform.latitude_deg[valid]
    lon = transform.longitude_deg[valid]
    normalized_crs = target_grid.crs.upper().replace(" ", "")
    if normalized_crs in {"EPSG:4326", "OGC:CRS84", "CRS84"}:
        x_coordinates = lon
        y_coordinates = lat
    else:
        try:
            from pyproj import Transformer

            transformer = Transformer.from_crs(
                "EPSG:4326",
                target_grid.crs,
                always_xy=True,
            )
            x_coordinates, y_coordinates = transformer.transform(lon, lat)
        except Exception as error:
            logger.exception("RadarSLC target coordinates cannot be projected")
            reject_invalid_state(
                f"RadarSLC target CRS cannot project WGS84 coordinates: {error}"
            )
        x_coordinates = np.asarray(x_coordinates, dtype=np.float64)
        y_coordinates = np.asarray(y_coordinates, dtype=np.float64)
    az = transform.azimuth_index[valid]
    rg = transform.range_index[valid]
    finite_coordinates = np.isfinite(x_coordinates) & np.isfinite(y_coordinates)
    col = np.rint(np.where(finite_coordinates, (x_coordinates - c) / a, 0.0)).astype(
        np.int64
    )
    row = np.rint(np.where(finite_coordinates, (y_coordinates - f) / e, 0.0)).astype(
        np.int64
    )
    src_az = np.rint(az).astype(np.int64)
    src_rg = np.rint(rg).astype(np.int64)
    src_h, src_w = samples.shape
    keep = (
        (row >= 0)
        & (row < height)
        & (col >= 0)
        & (col < width)
        & (src_az >= 0)
        & (src_az < src_h)
        & (src_rg >= 0)
        & (src_rg < src_w)
        & finite_coordinates
    )
    out[row[keep], col[keep]] = samples[src_az[keep], src_rg[keep]]
    return out


def _geo_to_radar_resample(
    samples: np.ndarray,
    transform: TransformResult,
    radar_shape: tuple[int, int],
) -> np.ndarray:
    """Nearest-neighbour place geo samples onto a radar grid via geo2rdr LUTs."""
    out = np.zeros(radar_shape, dtype=samples.dtype)
    valid = transform.converged
    if not np.any(valid):
        return out
    az = np.rint(transform.azimuth_index[valid]).astype(np.int64)
    rg = np.rint(transform.range_index[valid]).astype(np.int64)
    # source indices are geo pixel centres in row-major order of transform arrays
    src_rows, src_cols = np.indices(transform.latitude_deg.shape)
    src_rows = src_rows[valid]
    src_cols = src_cols[valid]
    height, width = radar_shape
    keep = (az >= 0) & (az < height) & (rg >= 0) & (rg < width)
    out[az[keep], rg[keep]] = samples[src_rows[keep], src_cols[keep]]
    return out
