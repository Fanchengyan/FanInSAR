"""Automatic Copernicus GLO-30 DEM management: tile query, cache, download, mosaic."""

from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.query import BoundingBox

logger = setup_logger(__name__)

COPERNICUS_GLO30_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
DEM_CACHE_ENV = "FANINSAR_DEM_CACHE_DIR"
DEM_SOURCE_ENV = "FANINSAR_DEM_SOURCE_URL"
DEM_NAME_ENV = "FANINSAR_DEM_NAME"
DEFAULT_DEM_NAME = "dem.tif"
_MIN_TILE_BYTES = 1 << 20
_DOWNLOAD_ATTEMPTS = 3

Bounds = BoundingBox | tuple[float, float, float, float]


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


def _bounds_tuple(bounds: Bounds) -> tuple[float, float, float, float]:
    """Normalize a BoundingBox or plain tuple to lon/lat bounds."""
    if isinstance(bounds, BoundingBox):
        return (
            float(bounds.left),
            float(bounds.bottom),
            float(bounds.right),
            float(bounds.top),
        )
    min_lon, min_lat, max_lon, max_lat = bounds
    return float(min_lon), float(min_lat), float(max_lon), float(max_lat)


@dataclass(slots=True)
class DEMManager:
    """Resolve, cache, download, and mosaic Copernicus GLO-30 DEM tiles.

    Parameters
    ----------
    cache_dir : Path
        Folder holding raw Copernicus GLO-30 tiles.  Flat files and per-tile
        subdirectory layouts are both accepted as cache hits.
    source_url : str, optional
        Base URL for tile downloads.  Defaults to the public AWS S3 bucket.

    """

    cache_dir: Path
    source_url: str = COPERNICUS_GLO30_URL

    def __post_init__(self) -> None:
        """Convert the cache directory to a Path."""
        self.cache_dir = Path(self.cache_dir)

    def required_tiles(self, bounds: Bounds) -> list[tuple[str, str]]:
        """Return the sorted tile identifiers covering the requested bounds.

        Parameters
        ----------
        bounds : BoundingBox or tuple
            (min_lon, min_lat, max_lon, max_lat) in EPSG:4326 or a BoundingBox.

        Returns
        -------
        list[tuple[str, str]]
            (tile_dir, filename) pairs for every integer degree cell the
            bounds intersect, in latitude-major order.

        """
        min_lon, min_lat, max_lon, max_lat = _bounds_tuple(bounds)
        return [
            copernicus_tile_name(lat_tile + 0.5, lon_tile + 0.5)
            for lat_tile in range(int(np.floor(min_lat)), int(np.floor(max_lat)) + 1)
            for lon_tile in range(int(np.floor(min_lon)), int(np.floor(max_lon)) + 1)
        ]

    def _cache_candidates(self, tile: tuple[str, str]) -> list[Path]:
        tile_dir, filename = tile
        return [
            self.cache_dir / filename,
            self.cache_dir / tile_dir / filename,
            *sorted(self.cache_dir.glob(f"*/{tile_dir}/{filename}")),
            *sorted(self.cache_dir.glob(f"**/{filename}")),
        ]

    def _find_cached(self, tile: tuple[str, str]) -> Path | None:
        for candidate in self._cache_candidates(tile):
            if candidate.is_file() and candidate.stat().st_size >= _MIN_TILE_BYTES:
                return candidate
        return None

    def _download(self, tile: tuple[str, str]) -> Path:
        tile_dir, filename = tile
        target_dir = self.cache_dir / tile_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        url = f"{self.source_url.rstrip('/')}/{tile_dir}/{filename}"
        temporary = target.with_suffix(target.suffix + ".part")
        last_error: Exception | None = None
        for attempt in range(_DOWNLOAD_ATTEMPTS):
            try:
                with urllib.request.urlopen(url, timeout=180) as response, (
                    temporary.open("wb")
                ) as out:
                    shutil.copyfileobj(response, out, length=1 << 20)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "DEM tile fetch attempt %d failed for %s: %r",
                    attempt + 1,
                    url,
                    exc,
                )
                temporary.unlink(missing_ok=True)
                continue
            if temporary.stat().st_size < _MIN_TILE_BYTES:
                last_error = RuntimeError(
                    f"tile too small: {temporary.stat().st_size} bytes"
                )
                logger.warning("%s", last_error)
                temporary.unlink(missing_ok=True)
                continue
            temporary.replace(target)
            logger.info(
                "DEM tile fetched: %s (%d bytes)", url, target.stat().st_size
            )
            return target
        message = (
            f"DEM tile download failed after {_DOWNLOAD_ATTEMPTS} attempts: "
            f"{url} ({last_error!r})"
        )
        logger.error(message)
        raise InvalidProcessingStateError(message)

    def _ensure_tiles(self, tiles: list[tuple[str, str]]) -> list[Path]:
        paths: list[Path] = []
        for tile in tiles:
            cached = self._find_cached(tile)
            if cached is None:
                cached = self._download(tile)
            else:
                logger.info("DEM tile cache hit: %s", cached)
            paths.append(cached)
        return paths

    def fetch_dem(self, bounds: Bounds, output_path: str | Path | None = None) -> Path:
        """Ensure all tiles covering the bounds and write one merged GeoTIFF.

        Parameters
        ----------
        bounds : BoundingBox or tuple
            (min_lon, min_lat, max_lon, max_lat) in EPSG:4326 or a BoundingBox.
        output_path : path, optional
            Destination GeoTIFF.  Defaults to
            <cache parent>/dem/<FANINSAR_DEM_NAME or dem.tif>.

        Returns
        -------
        Path
            The written mosaic path.

        """
        tiles = self.required_tiles(bounds)
        tile_paths = self._ensure_tiles(tiles)
        if output_path is None:
            output_path = self.cache_dir.parent / "dem" / default_dem_name()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        import rasterio
        from rasterio.merge import merge

        datasets = [rasterio.open(path) for path in tile_paths]
        try:
            mosaic, transform = merge(datasets, nodata=np.nan, dtype="float32")
        finally:
            for dataset in datasets:
                dataset.close()
        profile = {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
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
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(mosaic[0], 1)
        logger.info("DEM mosaic written: %s shape=%s", out, mosaic.shape)
        return out


def get_dem_manager() -> DEMManager:
    """Return a DEMManager configured from environment variables.

    Returns
    -------
    DEMManager
        Manager with FANINSAR_DEM_CACHE_DIR as the raw tile cache and the
        optional FANINSAR_DEM_SOURCE_URL base URL.

    Raises
    ------
    InvalidProcessingStateError
        If FANINSAR_DEM_CACHE_DIR is not set.

    """
    cache_dir = os.environ.get(DEM_CACHE_ENV)
    if not cache_dir:
        message = (
            f"{DEM_CACHE_ENV} is not set; cannot resolve the automatic DEM. "
            "Point it at a folder of Copernicus GLO-30 tiles (the download cache)."
        )
        logger.error(message)
        raise InvalidProcessingStateError(message)
    return DEMManager(
        cache_dir=Path(cache_dir),
        source_url=os.environ.get(DEM_SOURCE_ENV, COPERNICUS_GLO30_URL),
    )
