"""Tests for pair parsing: PairDataset base and InterferogramDataset."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from pyproj.crs import CRS
from rasterio.transform import from_bounds

from faninsar.core.pairs import Pairs
from faninsar.data.datasets.base import PairDataset
from faninsar.data.datasets.ifg import InterferogramDataset

if TYPE_CHECKING:
    from collections.abc import Iterable


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


class SamplePairDataset(PairDataset):
    """Pair dataset that parses primary/secondary dates from filenames."""

    pattern = "*.tif"

    @classmethod
    def parse_pairs(cls, paths: Iterable[str | Path]) -> Pairs:
        """Parse YYYYMMDD_YYYYMMDD pair names from file stems."""
        parsed: list[tuple[np.datetime64, np.datetime64]] = []
        for path in paths:
            parts = Path(path).stem.split("_")
            primary = np.datetime64(datetime.strptime(parts[0], "%Y%m%d"), "ns")
            secondary = np.datetime64(datetime.strptime(parts[1], "%Y%m%d"), "ns")
            parsed.append((primary, secondary))
        return Pairs(parsed)


@pytest.fixture
def pair_root(tmp_path: Path) -> Path:
    """Create tiles encoding interferometric pair names."""
    bounds = (0.0, 0.0, 4.0, 4.0)
    stems = ["20210101_20210113_pair", "20210113_20210125_pair"]
    for stem in stems:
        _write_tile(tmp_path / f"{stem}.tif", bounds, value=1.0)
    return tmp_path


def test_pair_dataset_parses_pairs(pair_root: Path) -> None:
    """PairDataset should store primary and secondary date metadata."""
    ds = SamplePairDataset(root_dir=pair_root, verbose=False)
    assert list(ds.pairs.to_names()) == [
        "20210101_20210113",
        "20210113_20210125",
    ]
    assert ds.file_dim_name == "pair"


class TestInterferogramParsePairs:
    """Test InterferogramDataset.parse_pairs with various date formats."""

    def test_yyyymmdd(self) -> None:
        """Plain YYYYMMDD dates."""
        paths = [
            "/data/S1A_IW_20200314_20200326_unw_phase.tif",
            "/data/S1A_IW_20200326_20200407_unw_phase.tif",
        ]
        pairs = InterferogramDataset.parse_pairs(paths)
        assert list(pairs.to_names()) == [
            "20200314_20200326",
            "20200326_20200407",
        ]

    def test_yyyymmddthhmmss(self) -> None:
        """HyP3 naming convention."""
        paths = [
            "/data/S1A_IW_20200314T170000_20200326T170000_VVP000_unw_phase.tif",
            "/data/S1A_IW_20200326T170000_20200407T170000_VVP000_unw_phase.tif",
        ]
        pairs = InterferogramDataset.parse_pairs(paths)
        assert list(pairs.to_names()) == [
            "20200314_20200326",
            "20200326_20200407",
        ]

    def test_yyyy_mm_dd(self) -> None:
        """Hyphenated YYYY-MM-DD dates."""
        paths = [
            "/data/S1A_IW_2020-03-14_2020-03-26_unw_phase.tif",
        ]
        pairs = InterferogramDataset.parse_pairs(paths)
        assert list(pairs.to_names()) == ["20200314_20200326"]

    def test_iso_with_fractional_seconds(self) -> None:
        """ISO datetime with fractional seconds."""
        paths = [
            "/data/S1A_IW_20200314T170000.000_20200326T170000.000.tif",
        ]
        pairs = InterferogramDataset.parse_pairs(paths)
        assert list(pairs.to_names()) == ["20200314_20200326"]

    def test_invalid_filename_raises(self) -> None:
        """Filename without dates should raise ValueError."""
        paths = ["/data/no_dates_here.tif"]
        with pytest.raises(ValueError, match="Cannot parse pair dates"):
            InterferogramDataset.parse_pairs(paths)
