"""PROPOSAL-0079: one managed WGS84 DEM with bounds-only reuse and growth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from faninsar.processing.geometry import DEM, GridSpec, RasterDEM, SourceDEM
from faninsar.processing.geometry.managed_dem import (
    DEMCoverageInsufficientError,
    interpolation_margin_deg,
    is_canonical_managed_dem,
    raster_covers_bounds,
    resolve_managed_wgs84_dem,
)


def _grid(
    west: float = 10.0,
    north: float = 40.1,
    size: int = 12,
    spacing: float = 0.01,
) -> GridSpec:
    return GridSpec(
        "EPSG:4326",
        Affine(spacing, 0.0, west, 0.0, -spacing, north),
        height=size,
        width=size,
    )


def _raster(grid: GridSpec, height: float = 12.0) -> RasterDEM:
    return RasterDEM(
        array=np.full(grid.shape, height, dtype=np.float32),
        grid=grid,
        vertical_datum="ellipsoidal",
        provenance={"product": "glo30", "provider": "pc"},
    )


def _fake_to_raster(grids: list[GridSpec]):
    def _to_raster(self: SourceDEM, grid: GridSpec, **_kwargs: object) -> RasterDEM:
        grids.append(grid)
        return _raster(grid)

    return _to_raster


def test_reuse_without_roi_does_not_call_source_or_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing dem.tif without an ROI is reused without a coverage check."""
    managed = tmp_path / "dem.tif"
    original = _raster(_grid())
    original.save(managed)
    before = managed.read_bytes()
    grids: list[GridSpec] = []
    monkeypatch.setattr(SourceDEM, "to_raster", _fake_to_raster(grids))
    source = DEM.from_source("glo30", cache_dir=tmp_path / "cache")
    resolved = resolve_managed_wgs84_dem(
        managed_path=managed,
        source=source,
        required_bounds=(9.0, 39.0, 12.0, 42.0),
        check_coverage=False,
        cache_dir=tmp_path / "cache",
        scratch_dir=tmp_path / "tmp",
    )
    assert grids == []
    assert managed.read_bytes() == before
    assert resolved.grid == original.grid
    assert not (tmp_path / "dem_wgs84.tif").exists()
    assert not (tmp_path / "dem" / "rgv_stack.tif").exists()


def test_covered_roi_reads_header_only_and_leaves_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A covered explicit ROI inspects bounds and does not rewrite the DEM."""
    managed = tmp_path / "dem.tif"
    original = _raster(_grid())
    original.save(managed)
    before = managed.read_bytes()
    grids: list[GridSpec] = []
    monkeypatch.setattr(SourceDEM, "to_raster", _fake_to_raster(grids))
    reads: list[int] = []
    import rasterio

    real_open = rasterio.open

    class _HeaderOnly:
        def __init__(self, dataset: object) -> None:
            self._dataset = dataset

        def __enter__(self) -> object:
            dataset = self._dataset.__enter__()

            def _forbidden(*_args: object, **_kwargs: object) -> object:
                reads.append(1)
                raise AssertionError("covered ROI reuse must not read DEM pixels")

            dataset.read = _forbidden  # type: ignore[method-assign]
            return dataset

        def __exit__(self, *args: object) -> object:
            return self._dataset.__exit__(*args)

    monkeypatch.setattr(
        rasterio, "open", lambda *args, **kwargs: _HeaderOnly(real_open(*args, **kwargs))
    )
    assert raster_covers_bounds(managed, (10.02, 40.02, 10.08, 40.08))
    monkeypatch.setattr(rasterio, "open", real_open)
    source = DEM.from_source("glo30", cache_dir=tmp_path / "cache")
    resolved = resolve_managed_wgs84_dem(
        managed_path=managed,
        source=source,
        required_bounds=(10.02, 40.02, 10.08, 40.08),
        check_coverage=True,
        cache_dir=tmp_path / "cache",
        scratch_dir=tmp_path / "tmp",
    )
    assert grids == []
    assert reads == []
    assert managed.read_bytes() == before
    assert resolved.grid == original.grid


def test_uncovered_roi_grows_union_and_keeps_prior_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An uncovered ROI regenerates the same file over the union extent."""
    managed = tmp_path / "dem.tif"
    original = _raster(_grid(west=10.0, north=40.1))
    original.save(managed)
    grids: list[GridSpec] = []
    monkeypatch.setattr(SourceDEM, "to_raster", _fake_to_raster(grids))
    source = DEM.from_source("glo30", cache_dir=tmp_path / "cache")
    requested = (9.5, 39.5, 10.2, 40.05)
    resolved = resolve_managed_wgs84_dem(
        managed_path=managed,
        source=source,
        required_bounds=requested,
        check_coverage=True,
        cache_dir=tmp_path / "cache",
        scratch_dir=tmp_path / "tmp",
        product="glo30",
    )
    assert len(grids) == 1
    west, south, east, north = resolved.grid.bounds
    assert west <= min(original.grid.bounds[0], requested[0])
    assert south <= min(original.grid.bounds[1], requested[1])
    assert east >= max(original.grid.bounds[2], requested[2])
    assert north >= max(original.grid.bounds[3], requested[3])
    assert managed.is_file()
    assert not (tmp_path / "dem_wgs84.tif").exists()
    assert abs(resolved.grid.transform[0] - original.grid.transform[0]) < 1.0e-12


def test_failed_fetch_leaves_previous_managed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed expansion keeps the previous dem.tif and raises the real error."""
    managed = tmp_path / "dem.tif"
    original = _raster(_grid())
    original.save(managed)
    before = managed.read_bytes()

    def _boom(_self: SourceDEM, _grid: GridSpec, **_kwargs: object) -> RasterDEM:
        raise OSError("injected source outage")

    monkeypatch.setattr(SourceDEM, "to_raster", _boom)
    source = DEM.from_source("glo30", cache_dir=tmp_path / "cache")
    with pytest.raises(OSError, match="injected source outage"):
        resolve_managed_wgs84_dem(
            managed_path=managed,
            source=source,
            required_bounds=(9.0, 39.0, 12.0, 42.0),
            check_coverage=True,
            cache_dir=tmp_path / "cache",
            scratch_dir=tmp_path / "tmp",
        )
    assert managed.read_bytes() == before


def test_raster_only_shortage_reports_no_source(tmp_path: Path) -> None:
    """External rasters without a source cannot be expanded in place."""
    managed = tmp_path / "dem.tif"
    _raster(_grid()).save(managed)
    with pytest.raises(DEMCoverageInsufficientError, match="no source"):
        resolve_managed_wgs84_dem(
            managed_path=managed,
            source=None,
            required_bounds=(9.0, 39.0, 12.0, 42.0),
            check_coverage=True,
            scratch_dir=tmp_path / "tmp",
        )
    assert interpolation_margin_deg(1.0 / 3600.0) > 0.0


def test_projected_legacy_raster_is_normalized_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leftover projected raster is rewritten once onto EPSG:4326 ellipsoidal."""
    managed = tmp_path / "dem.tif"
    utm = GridSpec(
        "EPSG:32633",
        Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4200000.0),
        height=12,
        width=12,
    )
    original = _raster(utm)
    original.save(managed)
    calls: list[tuple[str, str]] = []

    def _to_raster(
        self: RasterDEM, grid: GridSpec, **kwargs: object
    ) -> RasterDEM:
        datum = str(kwargs.get("vertical_datum", "ellipsoidal"))
        calls.append((str(grid.crs), datum))
        return _raster(grid)

    monkeypatch.setattr(RasterDEM, "to_raster", _to_raster)
    resolved = resolve_managed_wgs84_dem(
        managed_path=managed,
        existing=original,
        check_coverage=False,
        scratch_dir=tmp_path / "tmp",
    )
    assert calls == [("EPSG:4326", "ellipsoidal")]
    assert is_canonical_managed_dem(resolved.grid, resolved.vertical_datum)
    assert str(resolved.crs) == "EPSG:4326"
    assert resolved.vertical_datum == "ellipsoidal"


def test_canonical_file_is_not_rewritten_on_prep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An already canonical ellipsoidal dem.tif is left on its existing lattice."""
    managed = tmp_path / "dem.tif"
    original = _raster(_grid())
    original.save(managed)
    before = managed.read_bytes()

    def _boom(self: RasterDEM, grid: GridSpec, **_kwargs: object) -> RasterDEM:
        raise AssertionError("canonical managed DEM must not be renormalized")

    monkeypatch.setattr(RasterDEM, "to_raster", _boom)
    resolved = resolve_managed_wgs84_dem(
        managed_path=managed,
        check_coverage=False,
        scratch_dir=tmp_path / "tmp",
    )
    assert managed.read_bytes() == before
    assert resolved.grid == original.grid
    assert is_canonical_managed_dem(resolved.grid, resolved.vertical_datum)
