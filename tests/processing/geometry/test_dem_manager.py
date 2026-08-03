"""Tests for the automatic Copernicus DEM manager."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from affine import Affine

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry.dem_manager import (
    _MIN_TILE_BYTES,
    DEMManager,
    copernicus_tile_name,
    default_dem_name,
    get_dem_manager,
)
from faninsar.query import BoundingBox


def _write_tile(
    path: Path,
    *,
    latitude: int,
    longitude: int,
    rows: int = 512,
    cols: int = 512,
    value: float = 100.0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = Affine.translation(float(longitude), float(latitude + 1)) * Affine.scale(
        1.0 / cols, -1.0 / rows
    )
    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((rows, cols), value, dtype="float32"), 1)
    return path


def test_copernicus_tile_name_convention() -> None:
    """Tile naming follows the Copernicus GLO-30 COG convention."""
    assert copernicus_tile_name(38.5, 100.5) == (
        "N38_E100",
        "Copernicus_DSM_COG_10_N38_00_E100_00_DEM.tif",
    )
    assert copernicus_tile_name(-2.5, -71.5) == (
        "S03_W072",
        "Copernicus_DSM_COG_10_S03_00_W072_00_DEM.tif",
    )


def test_required_tiles_spans_all_intersecting_cells() -> None:
    """Bounds spanning several degrees resolve one tile per integer cell."""
    manager = DEMManager(Path("/tmp/unused-cache"))
    tiles = manager.required_tiles((97.5, 38.7, 101.4, 40.0))
    assert tiles == [
        copernicus_tile_name(lat + 0.5, lon + 0.5)
        for lat in (38, 39, 40)
        for lon in (97, 98, 99, 100, 101)
    ]


def test_required_tiles_single_cell_at_degree_boundary() -> None:
    """A point exactly on an integer degree still resolves one tile."""
    manager = DEMManager(Path("/tmp/unused-cache"))
    assert manager.required_tiles((100.0, 38.0, 100.0, 38.0)) == [
        copernicus_tile_name(38.5, 100.5)
    ]


def test_fetch_dem_uses_flat_cache_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing flat tiles are reused and no network fetch happens."""
    cache = tmp_path / "cache"
    first = _write_tile(
        cache / "Copernicus_DSM_COG_10_N38_00_E100_00_DEM.tif",
        latitude=38,
        longitude=100,
    )
    second = _write_tile(
        cache / "Copernicus_DSM_COG_10_N39_00_E100_00_DEM.tif",
        latitude=39,
        longitude=100,
    )
    assert first.stat().st_size >= _MIN_TILE_BYTES
    assert second.stat().st_size >= _MIN_TILE_BYTES

    def fail_download(_tile: tuple[str, str]) -> Path:
        raise AssertionError("download must not be called on a cache hit")

    manager = DEMManager(cache)
    monkeypatch.setattr(DEMManager, "_download", fail_download)
    out = tmp_path / "out" / "dem.tif"
    result = manager.fetch_dem((100.2, 38.2, 100.8, 39.8), out)

    assert result == out
    assert out.exists()
    with rasterio.open(out) as dataset:
        assert dataset.dtypes == ("float32",)
        assert dataset.crs.to_epsg() == 4326
        band = dataset.read(1)
        assert np.isfinite(band).all()
        assert float(np.nanmin(band)) == 100.0


def test_fetch_dem_downloads_missing_tile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing tile is fetched from the source URL into the cache."""
    cache = tmp_path / "cache"
    tile_bytes = _write_tile(
        tmp_path / "seed" / "tile.tif",
        latitude=38,
        longitude=100,
        rows=512,
        cols=512,
        value=42.0,
    ).read_bytes()
    assert len(tile_bytes) >= _MIN_TILE_BYTES
    requests: list[str] = []

    def fake_urlopen(url: str, timeout: int) -> SimpleNamespace:
        requests.append(url)
        return contextlib.nullcontext(io.BytesIO(tile_bytes))

    import faninsar.processing.geometry.dem_manager as dem_manager_module

    monkeypatch.setattr(dem_manager_module.urllib.request, "urlopen", fake_urlopen)
    manager = DEMManager(cache)
    out = tmp_path / "out" / "dem.tif"
    manager.fetch_dem((100.2, 38.2, 100.8, 38.8), out)

    expected_url = (
        "https://copernicus-dem-30m.s3.amazonaws.com/N38_E100/"
        "Copernicus_DSM_COG_10_N38_00_E100_00_DEM.tif"
    )
    assert requests == [expected_url]
    downloaded = cache / "N38_E100" / "Copernicus_DSM_COG_10_N38_00_E100_00_DEM.tif"
    assert downloaded.is_file()
    assert downloaded.stat().st_size == len(tile_bytes)
    assert out.exists()


def test_fetch_dem_accepts_bounding_box(tmp_path: Path) -> None:
    """A BoundingBox is accepted as the fetch bounds."""
    cache = tmp_path / "cache"
    _write_tile(
        cache / "Copernicus_DSM_COG_10_N38_00_E100_00_DEM.tif",
        latitude=38,
        longitude=100,
    )
    manager = DEMManager(cache)
    out = tmp_path / "out" / "dem.tif"
    result = manager.fetch_dem(BoundingBox(100.2, 38.2, 100.8, 38.8), out)
    assert result == out


def test_get_dem_manager_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_dem_manager reads cache and source URL from the environment."""
    monkeypatch.setenv("FANINSAR_DEM_CACHE_DIR", "/tmp/dem-cache")
    monkeypatch.setenv("FANINSAR_DEM_SOURCE_URL", "https://example.test/dem")
    manager = get_dem_manager()
    assert manager.cache_dir == Path("/tmp/dem-cache")
    assert manager.source_url == "https://example.test/dem"


def test_get_dem_manager_requires_cache_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_dem_manager raises when the cache environment is missing."""
    monkeypatch.delenv("FANINSAR_DEM_CACHE_DIR", raising=False)
    with pytest.raises(InvalidProcessingStateError):
        get_dem_manager()


def test_default_dem_name_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FANINSAR_DEM_NAME overrides the default output name."""
    monkeypatch.setenv("FANINSAR_DEM_NAME", "merged.tif")
    assert default_dem_name() == "merged.tif"
    monkeypatch.delenv("FANINSAR_DEM_NAME")
    assert default_dem_name() == "dem.tif"
