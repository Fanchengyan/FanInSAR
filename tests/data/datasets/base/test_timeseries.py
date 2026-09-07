"""Tests for TimeSeriesDataset base class."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj.crs import CRS
from rasterio.transform import from_bounds

from faninsar.core.acquisition import Acquisition
from faninsar.data.datasets.base import TimeSeriesDataset


def _write_tile(
    path: Path,
    bounds: tuple[float, float, float, float],
    value: float,
) -> None:
    """Create a small GeoTIFF tile for testing."""
    height = width = 4
    transform = from_bounds(*bounds, width, height)
    data = np.full((height, width), value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=0.0,
    ) as dst:
        dst.write(data, 1)


class SampleTimeSeriesDataset(TimeSeriesDataset):
    """Time-series dataset that parses dates from filenames."""

    pattern = "*.tif"

    @classmethod
    def parse_dates(cls, paths: list[str | Path]) -> Acquisition:
        """Parse YYYYMMDD dates from filename stems."""
        dates = [
            np.datetime64(
                datetime.strptime(Path(path).stem.split("_")[0], "%Y%m%d"), "ns"
            )
            for path in paths
        ]
        return Acquisition(dates)


@pytest.fixture
def time_series_root(tmp_path: Path) -> Path:
    """Create tiles with YYYYMMDD prefixes for time-series parsing."""
    bounds = (0.0, 0.0, 4.0, 4.0)
    for date, value in [("20210101", 1.0), ("20210113", 2.0)]:
        _write_tile(tmp_path / f"{date}_scene.tif", bounds, value=value)
    return tmp_path


def test_time_series_dataset_parses_dates(time_series_root: Path) -> None:
    """TimeSeriesDataset attaches parsed Acquisition metadata."""
    ds = SampleTimeSeriesDataset(root_dir=time_series_root, verbose=False)
    expected = [
        np.datetime64("2021-01-01T00:00:00"),
        np.datetime64("2021-01-13T00:00:00"),
    ]
    assert list(ds.dates.values) == expected
    assert ds.file_dim_name == "date"
