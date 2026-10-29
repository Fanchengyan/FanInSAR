"""Public-seam tests for GACOS adjacent APS increments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from pyproj.crs import CRS
from rasterio.transform import from_bounds

from faninsar.data.datasets import GACOS
from faninsar.data.query import BoundingBox, Points

if TYPE_CHECKING:
    from pathlib import Path


def _write_ztd(path: Path, data: np.ndarray) -> None:
    """Write a 4x4 GACOS-like GeoTIFF covering (0, 0, 4, 4)."""
    height, width = data.shape
    transform = from_bounds(0.0, 0.0, 4.0, 4.0, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data.astype(np.float32), 1)


def _gacos_root(tmp_path: Path) -> Path:
    """Create three dated ZTD maps with independent pixel values."""
    # north-up 4x4: row 0 is y in (3, 4), col 0 is x in (0, 1)
    _write_ztd(
        tmp_path / "20210101.ztd.tif",
        np.array(
            [
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
                [30.0, 31.0, 32.0, 33.0],
                [40.0, 41.0, 42.0, 43.0],
            ]
        ),
    )
    _write_ztd(
        tmp_path / "20210113.ztd.tif",
        np.array(
            [
                [15.0, 16.0, 17.0, 18.0],
                [25.0, 26.0, 27.0, 28.0],
                [35.0, 36.0, 37.0, 38.0],
                [50.0, 51.0, 52.0, 53.0],
            ]
        ),
    )
    _write_ztd(
        tmp_path / "20210125.ztd.tif",
        np.array(
            [
                [21.0, 22.0, 23.0, 24.0],
                [31.0, 32.0, 33.0, 34.0],
                [41.0, 42.0, 43.0, 44.0],
                [70.0, 71.0, 72.0, 73.0],
            ]
        ),
    )
    return tmp_path


@pytest.fixture
def gacos_ds(tmp_path: Path) -> GACOS:
    """GACOS dataset with three dated ZTD files."""
    _gacos_root(tmp_path)
    return GACOS(root_dir=tmp_path, verbose=False, nodata=np.nan)


def test_increments_match_adjacent_ztd_minus_reference(gacos_ds: GACOS) -> None:
    """Adjacent increments are later-minus-earlier ZTD after reference subtraction."""
    dates = ["20210101", "20210113", "20210125"]
    query = Points([(0.5, 0.5), (3.5, 3.5)], crs=CRS.from_epsg(4326))
    ref_points = Points([(0.5, 0.5)], crs=CRS.from_epsg(4326))

    increments = gacos_ds.increments(dates, query=query, ref_points=ref_points)

    # pixel (0.5, 0.5) -> 40, 50, 70 m; pixel (3.5, 3.5) -> 13, 18, 24 m
    # raw diffs [[10, 5], [20, 6]] m; subtract ref diffs [10, 20] m; convert to mm
    expected = np.array([[0.0, -5000.0], [0.0, -14000.0]], dtype=np.float64)
    np.testing.assert_allclose(increments, expected, rtol=1e-6, atol=1e-6)
    assert increments.shape == (2, 2)


def test_increments_divide_by_los_ratio(gacos_ds: GACOS) -> None:
    """LOS conversion divides zenith increments by the direction cosine."""
    dates = ["20210101", "20210113", "20210125"]
    query = Points([(0.5, 0.5), (3.5, 3.5)], crs=CRS.from_epsg(4326))
    ref_points = Points([(0.5, 0.5)], crs=CRS.from_epsg(4326))

    increments = gacos_ds.increments(
        dates,
        query=query,
        ref_points=ref_points,
        los_ratio=0.5,
    )

    expected = np.array([[0.0, -10000.0], [0.0, -28000.0]], dtype=np.float64)
    np.testing.assert_allclose(increments, expected, rtol=1e-6, atol=1e-6)


def test_increments_can_prepend_reference_zero(gacos_ds: GACOS) -> None:
    """insert_ref_zero prepends a zero row for cumulative time series."""
    dates = ["20210101", "20210113", "20210125"]
    query = Points([(0.5, 0.5), (3.5, 3.5)], crs=CRS.from_epsg(4326))
    ref_points = Points([(0.5, 0.5)], crs=CRS.from_epsg(4326))

    increments = gacos_ds.increments(
        dates,
        query=query,
        ref_points=ref_points,
        insert_ref_zero=True,
    )

    expected = np.array(
        [[0.0, 0.0], [0.0, -5000.0], [0.0, -14000.0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(increments, expected, rtol=1e-6, atol=1e-6)
    assert increments.shape == (3, 2)


def test_correct_subtracts_increments_from_nsbas_incs(gacos_ds: GACOS) -> None:
    """correct() subtracts APS increments from NSBAS incs of shape (n_date-1, n_pt)."""
    dates = ["20210101", "20210113", "20210125"]
    query = Points([(0.5, 0.5), (3.5, 3.5)], crs=CRS.from_epsg(4326))
    ref_points = Points([(0.5, 0.5)], crs=CRS.from_epsg(4326))
    incs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    corrected = gacos_ds.correct(
        incs,
        dates,
        query=query,
        ref_points=ref_points,
    )

    expected = np.array([[1.0, 5002.0], [3.0, 14004.0]], dtype=np.float64)
    np.testing.assert_allclose(corrected, expected, rtol=1e-6, atol=1e-6)


def test_correct_rejects_cumulative_series_with_leading_zero(
    gacos_ds: GACOS,
) -> None:
    """A cumulative series of length n_date must not be passed as incs."""
    dates = ["20210101", "20210113", "20210125"]
    query = Points([(0.5, 0.5), (3.5, 3.5)], crs=CRS.from_epsg(4326))
    ref_points = Points([(0.5, 0.5)], crs=CRS.from_epsg(4326))
    cumulative = np.zeros((3, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="n_date - 1"):
        gacos_ds.correct(
            cumulative,
            dates,
            query=query,
            ref_points=ref_points,
        )


def test_increments_fail_closed_on_missing_date(gacos_ds: GACOS) -> None:
    """Missing GACOS dates must raise instead of inserting zeros."""
    dates = ["20210101", "20210119", "20210125"]
    query = Points([(0.5, 0.5)], crs=CRS.from_epsg(4326))
    ref_points = Points([(0.5, 0.5)], crs=CRS.from_epsg(4326))

    with pytest.raises(ValueError, match="missing"):
        gacos_ds.increments(dates, query=query, ref_points=ref_points)


def test_increments_from_bbox_flatten_to_pixel_columns(gacos_ds: GACOS) -> None:
    """A bbox query returns (n_date-1, n_pixel) after flattening y, x."""
    dates = ["20210101", "20210113"]
    bbox = BoundingBox(0.0, 0.0, 4.0, 4.0, crs=CRS.from_epsg(4326))
    ref_points = Points([(0.5, 0.5)], crs=CRS.from_epsg(4326))

    increments = gacos_ds.increments(dates, query=bbox, ref_points=ref_points)

    assert increments.shape == (1, 16)
    # southwest pixel (0.5, 0.5) is last row, first column -> index 12 in C-order
    np.testing.assert_allclose(increments[0, 12], 0.0, atol=1e-6)
    # northeast pixel (3.5, 3.5) is first row, last column -> index 3
    np.testing.assert_allclose(increments[0, 3], -5000.0, atol=1e-6)
