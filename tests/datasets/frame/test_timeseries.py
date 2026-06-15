"""Tests for FrameTimeSeries (M2).

Uses synthetic displacement.zarr + velocity COG written by the
``FrameTimeSeries.from_displacement`` seed helper. No real inversion runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from pyproj.crs import CRS

from faninsar.datasets.frame import Frame, FrameTimeSeries
from faninsar.datasets.geogrid import GeoGrid


def _write_tiff(path: Path, bounds: tuple, arr: np.ndarray) -> None:
    crs = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    )
    h, w = arr.shape
    transform = rasterio.transform.from_bounds(*bounds, w, h)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype=arr.dtype, crs=crs, transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(arr, 1)


def _make_synthetic_displacement(shape=(5, 4, 4)):
    """Return an xarray.DataArray (time, y, x) with synthetic displacement."""
    rng = np.random.default_rng(42)
    times = np.array(
        [np.datetime64("2019-11-15"), np.datetime64("2020-03-14"),
         np.datetime64("2020-07-08"), np.datetime64("2020-10-31"),
         np.datetime64("2021-02-22")]
    )
    data = rng.normal(0, 0.005, shape).astype(np.float32)
    da = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={
            "time": times,
            "y": np.linspace(46.0, 45.0, shape[1]),
            "x": np.linspace(10.0, 11.0, shape[2]),
        },
        name="displacement",
    )
    return da


@pytest.fixture
def ts_dir(tmp_path: Path) -> Path:
    """A frame dir with a populated timeseries/ subfolder."""
    bounds = (10.0, 45.0, 11.0, 46.0)
    disp = _make_synthetic_displacement()
    velocity = np.random.default_rng(42).uniform(-0.02, 0.02, (4, 4)).astype(np.float32)

    # Build a reference GeoGrid for the velocity COG
    grid = GeoGrid.from_bounds(bounds, crs="EPSG:4326", shape=(4, 4), tight=True)

    FrameTimeSeries.from_displacement(
        out_dir=tmp_path / "frame",
        displacement=disp,
        velocity=velocity,
        reference=grid,
        method="NSBAS",
        overwrite=True,
    )
    return tmp_path / "frame"


class TestFrameTimeSeriesBasic:
    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            FrameTimeSeries(tmp_path / "nope")

    def test_root(self, ts_dir: Path) -> None:
        ts = FrameTimeSeries(ts_dir / "timeseries")
        assert ts.root.name == "timeseries"

    def test_has_displacement(self, ts_dir: Path) -> None:
        ts = FrameTimeSeries(ts_dir / "timeseries")
        assert ts.has_displacement() is True
        assert ts.displacement_path().name == "displacement.zarr"

    def test_velocity_exists(self, ts_dir: Path) -> None:
        ts = FrameTimeSeries(ts_dir / "timeseries")
        assert ts.exists("velocity") is True
        assert ts.exists("velocity_std") is False

    def test_path(self, ts_dir: Path) -> None:
        ts = FrameTimeSeries(ts_dir / "timeseries")
        assert ts.path("velocity").name == "velocity.cog.tif"

    def test_metadata(self, ts_dir: Path) -> None:
        ts = FrameTimeSeries(ts_dir / "timeseries")
        meta = ts.metadata
        assert meta is not None
        assert meta["method"] == "NSBAS"
        assert meta["date_count"] == 5

    def test_summary(self, ts_dir: Path) -> None:
        ts = FrameTimeSeries(ts_dir / "timeseries")
        s = ts.summary()
        assert s["type"] == "FrameTimeSeries"
        assert s["has_displacement"] is True
        assert s["velocity"] is True
        assert s["date_count"] == 5

    def test_repr(self, ts_dir: Path) -> None:
        ts = FrameTimeSeries(ts_dir / "timeseries")
        r = repr(ts)
        assert "FrameTimeSeries" in r
        assert "displacement=True" in r


class TestFrameTimeSeriesOpen:
    def test_open_displacement(self, ts_dir: Path) -> None:
        ts = FrameTimeSeries(ts_dir / "timeseries")
        ds = ts.open_displacement()
        assert "displacement" in ds.data_vars
        assert ds["displacement"].shape == (5, 4, 4)

    def test_open_velocity(self, ts_dir: Path) -> None:
        ts = FrameTimeSeries(ts_dir / "timeseries")
        da = ts.open("velocity")
        assert da.shape[-2:] == (4, 4)

    def test_open_missing_displacement_raises(self, tmp_path: Path) -> None:
        (tmp_path / "timeseries").mkdir()
        ts = FrameTimeSeries(tmp_path / "timeseries")
        with pytest.raises(FileNotFoundError):
            ts.open_displacement()

    def test_open_missing_asset_raises(self, tmp_path: Path) -> None:
        (tmp_path / "timeseries").mkdir()
        ts = FrameTimeSeries(tmp_path / "timeseries")
        with pytest.raises(Exception):
            ts.open("velocity")


class TestFrameIntegration:
    def test_frame_timeseries_property(self, ts_dir: Path) -> None:
        frame = Frame(ts_dir)
        assert frame.timeseries is not None
        assert frame.timeseries.has_displacement()

    def test_frame_summary_includes_timeseries(self, ts_dir: Path) -> None:
        frame = Frame(ts_dir)
        s = frame.summary()
        assert "timeseries" in s
        assert s["timeseries"]["date_count"] == 5

    def test_frame_without_timeseries(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        frame = Frame(empty)
        assert frame.timeseries is None
