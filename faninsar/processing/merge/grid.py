"""Common projected grid spec for burst merge products."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = setup_logger(__name__)

# (west, south, east, north) in lon/lat degrees when building from footprints.
LonLatFootprint: tuple[float, float, float, float]

__all__ = ["GeoGridSpec", "build_geo_grid"]


@dataclass(frozen=True, slots=True)
class GeoGridSpec:
    """Regular north-up projected grid shared by all merge products.

    Parameters
    ----------
    crs : str
        EPSG code or WKT string (e.g. ``"EPSG:32633"``).
    transform : tuple of float
        Affine mapping ``(x0, dx, 0, y0, 0, -dy)`` in CRS units, north-up.
    width, height : int
        Grid dimensions in pixels.
    resolution_m : tuple of float
        ``(dx, dy)`` pixel sizes in CRS units (metres for projected CRS).
    bbox : tuple of float, optional
        ``(west, south, east, north)`` in CRS units. When omitted it is
        derived from ``transform`` and ``shape``.

    """

    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int
    resolution_m: tuple[float, float]
    bbox: tuple[float, float, float, float] = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Validate dimensions and fill a default bbox from the transform."""
        if self.width <= 0 or self.height <= 0:
            msg = "GeoGridSpec width and height must be positive"
            raise ValueError(msg)
        if not self.crs:
            msg = "GeoGridSpec crs must not be empty"
            raise ValueError(msg)
        x0, dx, _, y0, _, dy = self.transform
        if dx == 0.0 or dy == 0.0:
            msg = "GeoGridSpec pixel sizes must be non-zero"
            raise ValueError(msg)
        if self.bbox is None:
            west = x0
            north = y0
            east = x0 + self.width * dx
            south = y0 + self.height * dy  # dy is negative for north-up
            object.__setattr__(self, "bbox", (west, south, east, north))

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(height, width)`` consistent with numpy array ordering."""
        return (self.height, self.width)

    def xy_pixel_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Return meshgrid of pixel-center X and Y coordinates in CRS units.

        Returns
        -------
        x, y : numpy.ndarray
            Both of shape ``(height, width)``.

        """
        x0, dx, _, y0, _, dy = self.transform
        xs = x0 + dx * (0.5 + np.arange(self.width))
        ys = y0 + dy * (0.5 + np.arange(self.height))
        return np.meshgrid(xs, ys)


def _utm_zone_for_lonlat(lon: float, lat: float) -> str:
    """Return the EPSG code for the UTM zone containing a lon/lat point."""
    zone = int((lon + 180.0) // 6.0) + 1
    if lat >= 0.0:
        return f"EPSG:{32600 + zone}"
    return f"EPSG:{32700 + zone}"


def _to_utm_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
    crs: str,
) -> tuple[float, float, float, float]:
    """Transform a lon/lat bounding box to the target projected CRS."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs: list[float] = []
    ys: list[float] = []
    for lon in (west, east):
        for lat in (south, north):
            x, y = transformer.transform(lon, lat)
            xs.append(x)
            ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys))


def build_geo_grid(
    footprints: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    *,
    resolution_m: tuple[float, float],
    crs: str = "auto_utm",
    margin_m: float = 500.0,
) -> GeoGridSpec:
    """Build a common :class:`GeoGridSpec` from a union of burst footprints.

    Parameters
    ----------
    footprints : sequence of ((west, south), (east, north))
        Each footprint is a pair of lon/lat corners.
    resolution_m : tuple of float
        ``(dx, dy)`` pixel sizes in the target CRS units.
    crs : str, optional
        ``"auto_utm"`` picks the UTM zone of the footprint centroid,
        otherwise an explicit EPSG code (e.g. ``"EPSG:3857"``).
    margin_m : float, optional
        Margin added around the union extent in CRS units.

    Returns
    -------
    GeoGridSpec
        North-up grid covering the union of all footprints.

    """
    if not footprints:
        msg = "build_geo_grid requires at least one footprint"
        raise ValueError(msg)

    wests = [min(fp[0][0], fp[1][0]) for fp in footprints]
    easts = [max(fp[0][0], fp[1][0]) for fp in footprints]
    souths = [min(fp[0][1], fp[1][1]) for fp in footprints]
    norths = [max(fp[0][1], fp[1][1]) for fp in footprints]

    lon_w = min(wests)
    lon_e = max(easts)
    lat_s = min(souths)
    lat_n = max(norths)
    lon_c = 0.5 * (lon_w + lon_e)
    lat_c = 0.5 * (lat_s + lat_n)

    target_crs = _utm_zone_for_lonlat(lon_c, lat_c) if crs == "auto_utm" else crs

    x_min, y_min, x_max, y_max = _to_utm_bbox(lon_w, lat_s, lon_e, lat_n, target_crs)
    x_min -= margin_m
    y_min -= margin_m
    x_max += margin_m
    y_max += margin_m

    dx, dy = resolution_m
    width = int(np.ceil((x_max - x_min) / dx))
    height = int(np.ceil((y_max - y_min) / dy))

    # North-up: y0 = y_max, dy negative.
    transform = (x_min, dx, 0.0, y_max, 0.0, -dy)
    bbox = (x_min, y_min, x_max, y_max)
    logger.info(
        "built GeoGridSpec crs=%s shape=(%d, %d) bbox=%s",
        target_crs,
        height,
        width,
        bbox,
    )
    return GeoGridSpec(
        crs=target_crs,
        transform=transform,
        width=width,
        height=height,
        resolution_m=resolution_m,
        bbox=bbox,
    )
