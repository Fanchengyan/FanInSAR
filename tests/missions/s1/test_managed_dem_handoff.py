"""S1 acquisition hands DEM materialization to the shared workspace raster."""

# The repository intentionally keeps tests as a namespace package.
# ruff: noqa: INP001

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from affine import Affine

from faninsar.data.query.bbox import BoundingBox
from faninsar.missions.s1.acquisition import _fetch_dem_for_roi
from faninsar.processing.geometry import GridSpec, RasterDEM, SourceDEM

if TYPE_CHECKING:
    from pathlib import Path


def test_fetch_dem_for_roi_writes_root_dem_tif(
    tmp_path: Path, monkeypatch: object
) -> None:
    """ROI acquisition persists ``dem.tif`` rather than a nested working copy."""
    grids: list[GridSpec] = []

    def _fake_to_raster(
        _self: SourceDEM, grid: GridSpec, **_kwargs: object
    ) -> RasterDEM:
        grids.append(grid)
        return RasterDEM(
            array=np.full(grid.shape, 80.0, dtype=np.float32),
            grid=grid,
            vertical_datum="ellipsoidal",
            provenance={"product": "glo30", "provider": "pc"},
        )

    monkeypatch.setattr(SourceDEM, "to_raster", _fake_to_raster)
    roi = BoundingBox(10.0, 40.0, 10.05, 40.05, crs="EPSG:4326")
    raster = _fetch_dem_for_roi(roi, tmp_path)
    managed = tmp_path / "dem.tif"
    assert managed.is_file()
    assert not (tmp_path / "dem_wgs84.tif").exists()
    assert not (tmp_path / "dem" / "rgv_stack.tif").exists()
    assert str(raster.crs) == "EPSG:4326"
    assert raster.vertical_datum == "ellipsoidal"
    assert raster.path == managed
    assert grids
    assert all(str(grid.crs) == "EPSG:4326" for grid in grids)
    west, south, east, north = raster.grid.bounds
    assert west <= 10.0
    assert south <= 40.0
    assert east >= 10.05
    assert north >= 40.05
