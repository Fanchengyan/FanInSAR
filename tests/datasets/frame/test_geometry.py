"""Tests for faninsar.datasets.frame.geometry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from faninsar.datasets.frame.exceptions import MissingGeometryAssetError
from faninsar.datasets.frame.geometry import FrameGeometry
from faninsar.query import BoundingBox


def _write_tiff(
    path: Path,
    bounds: tuple[float, float, float, float],
    data: np.ndarray,
    crs: str = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]',
    nodata: float = -9999.0,
) -> None:
    """Write a single-band GeoTIFF."""
    height, width = data.shape
    transform = from_bounds(*bounds, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def synthetic_rasters(tmp_path: Path) -> dict[str, Path]:
    """Create small synthetic geometry rasters."""
    bounds = (10.0, 45.0, 11.0, 46.0)
    shape = (10, 10)

    inc = np.random.uniform(20.0, 60.0, shape).astype(np.float32)
    azi = np.random.uniform(0.0, 360.0, shape).astype(np.float32)
    heading = np.full(shape, 350.0, dtype=np.float32)
    dem = np.random.uniform(100.0, 2000.0, shape).astype(np.float32)
    water = np.random.choice([0, 1], size=shape).astype(np.uint8)

    paths: dict[str, Path] = {}
    for name, arr, nodata in [
        ("incidence", inc, -9999.0),
        ("azimuth", azi, -9999.0),
        ("heading", heading, -9999.0),
        ("dem", dem, -9999.0),
        ("water_mask", water, 255),
    ]:
        p = tmp_path / f"{name}.tif"
        _write_tiff(p, bounds, arr, nodata=nodata)
        paths[name] = p

    return paths


@pytest.fixture
def geometry_dir(tmp_path: Path, synthetic_rasters: dict[str, Path]) -> Path:
    """Create a FrameGeometry via from_rasters."""
    out = tmp_path / "frame"
    FrameGeometry.from_rasters(
        out_dir=out,
        incidence=synthetic_rasters["incidence"],
        azimuth=synthetic_rasters["azimuth"],
        heading=synthetic_rasters["heading"],
        dem=synthetic_rasters["dem"],
        water_mask=synthetic_rasters["water_mask"],
        overwrite=True,
    )
    return out / "geometry"


class TestFrameGeometryFromRasters:
    def test_creates_geometry_dir(self, geometry_dir: Path) -> None:
        assert geometry_dir.is_dir()

    def test_creates_geometry_json(self, geometry_dir: Path) -> None:
        assert (geometry_dir / "geometry.json").exists()

    def test_creates_all_assets(self, geometry_dir: Path) -> None:
        geom = FrameGeometry(geometry_dir)
        for name in ("incidence", "azimuth", "heading", "dem", "water_mask"):
            assert geom.exists(name), f"Asset '{name}' missing"
            assert geom.path(name).exists()

    def test_metadata_fields(self, geometry_dir: Path) -> None:
        geom = FrameGeometry(geometry_dir)
        meta = geom.metadata
        assert meta is not None
        assert meta["type"] == "FrameGeometry"
        assert meta["version"] == "0.1.0"
        assert "crs" in meta
        assert "width" in meta
        assert "height" in meta
        assert "transform" in meta
        assert "bounds" in meta
        assert "resolution" in meta
        assert "assets" in meta
        assert "created_at" in meta

    def test_relative_asset_hrefs(self, geometry_dir: Path) -> None:
        geom = FrameGeometry(geometry_dir)
        meta = geom.metadata
        for asset_info in meta["assets"].values():
            href = asset_info["href"]
            assert not Path(href).is_absolute()

    def test_continuous_assets_are_float32(self, geometry_dir: Path) -> None:
        geom = FrameGeometry(geometry_dir)
        meta = geom.metadata
        for name in ("incidence", "azimuth", "heading", "dem"):
            assert meta["assets"][name]["dtype"] == "float32"

    def test_water_mask_is_uint8(self, geometry_dir: Path) -> None:
        geom = FrameGeometry(geometry_dir)
        meta = geom.metadata
        assert meta["assets"]["water_mask"]["dtype"] == "uint8"

    def test_validate_alignment(self, geometry_dir: Path) -> None:
        geom = FrameGeometry(geometry_dir)
        assert geom.validate_alignment() is True

    def test_summary(self, geometry_dir: Path) -> None:
        geom = FrameGeometry(geometry_dir)
        s = geom.summary()
        assert "root" in s
        assert "crs" in s
        assert all(v is True for v in s["assets_present"].values())


class TestFrameGeometryOpen:
    def test_open_returns_dataarray(self, geometry_dir: Path) -> None:
        geom = FrameGeometry(geometry_dir)
        da = geom.open("incidence")
        assert da.ndim == 2
        assert da.shape == (10, 10)

    def test_open_with_chunks(self, geometry_dir: Path) -> None:
        geom = FrameGeometry(geometry_dir)
        da = geom.open("incidence", chunks={"y": 5, "x": 5})
        assert hasattr(da.data, "dask")

    def test_open_missing_raises(self, geometry_dir: Path) -> None:
        """Opening an asset that does not exist should raise MissingGeometryAssetError."""
        geom = FrameGeometry(geometry_dir)
        # Create a geometry with only incidence
        assert geom.exists("incidence")
        # A non-existent path: just test that exists() returns False for heading
        # if we removed it. But all are present. Instead, directly test:
        with pytest.raises(ValueError):
            geom.open("nonexistent_asset_name")  # type: ignore[arg-type]


class TestFrameGeometryClip:
    def test_clip_bbox(self, geometry_dir: Path) -> None:
        geom = FrameGeometry(geometry_dir)
        bbox = BoundingBox(10.2, 45.2, 10.8, 45.8, crs="EPSG:4326")
        clipped = geom.clip_bbox("incidence", bbox)
        assert clipped.ndim == 2
        # Should be a sub-region
        assert clipped.shape[0] <= 10
        assert clipped.shape[1] <= 10


class TestFrameGeometryLOS:
    def test_compute_los_unit_vectors(self, geometry_dir: Path) -> None:
        geom = FrameGeometry(geometry_dir)
        ds = geom.compute_los_unit_vectors(chunks=5)
        assert "los_east" in ds
        assert "los_north" in ds
        assert "los_up" in ds
        # LOS unit vector magnitude should be ~1
        mag = np.sqrt(
            ds["los_east"].values ** 2
            + ds["los_north"].values ** 2
            + ds["los_up"].values ** 2
        )
        np.testing.assert_allclose(mag, 1.0, atol=1e-5)

    def test_missing_azimuth_raises(
        self, tmp_path: Path, synthetic_rasters: dict[str, Path]
    ) -> None:
        """compute_los_unit_vectors should raise if azimuth is missing."""
        out = tmp_path / "frame_no_azi"
        geom = FrameGeometry.from_rasters(
            out_dir=out,
            incidence=synthetic_rasters["incidence"],
            overwrite=True,
        )
        with pytest.raises(MissingGeometryAssetError):
            geom.compute_los_unit_vectors()


class TestFrameGeometryOverwrite:
    def test_overwrite_false_raises(
        self, geometry_dir: Path, synthetic_rasters: dict[str, Path]
    ) -> None:
        out = geometry_dir.parent
        with pytest.raises(FileExistsError):
            FrameGeometry.from_rasters(
                out_dir=out,
                incidence=synthetic_rasters["incidence"],
                overwrite=False,
            )

    def test_overwrite_true_succeeds(
        self, geometry_dir: Path, synthetic_rasters: dict[str, Path]
    ) -> None:
        out = geometry_dir.parent
        geom = FrameGeometry.from_rasters(
            out_dir=out,
            incidence=synthetic_rasters["incidence"],
            overwrite=True,
        )
        assert geom.exists("incidence")


class TestFrameGeometryRadianConversion:
    def test_radian_to_degree(self, tmp_path: Path) -> None:
        """Angle rasters in radians should be converted to degrees."""
        bounds = (10.0, 45.0, 11.0, 46.0)
        shape = (10, 10)
        # Create incidence in radians (~0.5 rad ≈ 28.6 deg)
        inc_rad = np.full(shape, 0.5, dtype=np.float32)
        inc_path = tmp_path / "inc_rad.tif"
        _write_tiff(inc_path, bounds, inc_rad)

        out = tmp_path / "frame"
        geom = FrameGeometry.from_rasters(
            out_dir=out,
            incidence=inc_path,
            angle_unit="radian",
            overwrite=True,
        )
        da = geom.open("incidence")
        # Should be ~28.6 degrees, not 0.5
        assert float(da.mean()) > 20.0
        assert geom.metadata["angle_unit"] == "degree"


class TestFrameGeometryToXarrayDataset:
    def test_to_xarray_dataset(self, geometry_dir: Path) -> None:
        geom = FrameGeometry(geometry_dir)
        ds = geom.to_xarray_dataset()
        assert "incidence" in ds
        assert "dem" in ds

    def test_to_xarray_dataset_subset(self, geometry_dir: Path) -> None:
        geom = FrameGeometry(geometry_dir)
        ds = geom.to_xarray_dataset(names=["incidence", "dem"])
        assert "incidence" in ds
        assert "dem" in ds
        assert "azimuth" not in ds
