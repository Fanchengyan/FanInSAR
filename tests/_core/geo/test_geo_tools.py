from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio import Affine
from rasterio.profiles import Profile as RasterioProfile

from faninsar._core.geo.geo_tools import GeoGrid, Profile

profile = Profile(200, 300, Affine(*list(range(6))))


@pytest.fixture
def data_dir() -> Path:
    """Fixture for the data directory."""
    return Path(__file__).parents[2] / "data" / "ascii"


@pytest.fixture
def ascii_file_center(data_dir: Path) -> Path:
    """Fixture for the center ASCII file."""
    return data_dir / "lower_left_center.asc"


@pytest.fixture
def ascii_file_corner(data_dir: Path) -> Path:
    """Fixture for the corner ASCII file."""
    return data_dir / "lower_left_corner.asc"


class TestProfile:
    def test_init(self):
        profile = Profile(200, 300, Affine(*list(range(6))))
        assert profile["height"] == 200
        assert profile.width == 300
        assert profile.transform == Affine(*list(range(6)))

    def test_post_init(self):
        profile = Profile(200, 300, Affine(*list(range(6))), crs="EPSG:4326")
        assert profile.crs.to_string() == "EPSG:4326"


    def test_bounds_property(self):
        profile = Profile(200, 300, Affine(*list(range(6))))
        assert profile.bounds == (2.0, 805.0, 2.0, 5.0)

    def test_res_property(self):
        profile = Profile(200, 300, Affine(*list(range(6))))
        assert profile.res == (0, 4)


    def test_from_raster_file(self, tmp_path:Path):
        tif_file = tmp_path / "dummy.tif"
        with rasterio.open(tif_file, "w", driver="GTiff", height=200, width=300, count=1, dtype="float32") as dst:
            dst.write(np.zeros((200, 300), dtype="float32"),1)

        profile = Profile.from_raster_file(tif_file)
        assert profile["height"] == 200
        assert profile.width == 300

    def test_from_raster_file_with_kwargs(self, tmp_path: Path):
        tif_file = tmp_path / "dummy.tif"
        with rasterio.open(
            tif_file,
            "w",
            driver="GTiff",
            height=200,
            width=300,
            count=1,
            dtype="float32",
        ) as dst:
            dst.write(np.zeros((200, 300), dtype="float32"), 1)

        profile = Profile.from_raster_file(tif_file, count=3, tiled=True)
        assert profile.count == 3
        assert profile["tiled"] is True


    def test_from_ascii_header_file(self, ascii_file_corner, ascii_file_center):
        profile_corner = Profile.from_ascii_header_file(ascii_file_corner)
        assert profile_corner.height == 200
        assert profile_corner.width == 100
        assert profile_corner.nodata == -32768
        assert profile_corner.transform == Affine(30.0, 0.0, 300.0,
       0.0, -30.0, 6430.0)
        profile_center = Profile.from_ascii_header_file(ascii_file_center)
        assert profile_center.height == 200
        assert profile_center.width == 100
        assert profile_center.nodata == -32768
        assert profile_center.transform == Affine(30.0, 0.0, 285.0,
       0.0, -30.0, 6415.0)

    def test_from_ascii_header_file_with_kwargs(self, ascii_file_corner):
        profile = Profile.from_ascii_header_file(
            ascii_file_corner,
            count=2,
            dtype="float32",
        )
        assert profile.count == 2
        assert profile.dtype == "float32"

    def test_from_profile_file(self, ascii_file_corner, tmp_path:Path):
        profile_corner = Profile.from_ascii_header_file(ascii_file_corner)
        # save profile to tmp file
        profile_file = tmp_path / "dummy.profile"
        profile_corner.to_file(profile_file)
        # read profile from file
        profile = Profile.from_profile_file(profile_file)
        assert profile["height"] == 200
        assert profile.width == 100
        assert profile.nodata == -32768
        assert profile.transform == Affine(30.0, 0.0, 300.0, 0.0, -30.0, 6430.0)

    def test_from_profile_file_with_kwargs(self, ascii_file_corner, tmp_path: Path):
        profile_corner = Profile.from_ascii_header_file(ascii_file_corner)
        profile_file = tmp_path / "dummy.profile"
        profile_corner.to_file(profile_file)

        profile = Profile.from_profile_file(profile_file, count=4, tiled=True)
        assert profile.count == 4
        assert profile["tiled"] is True

    def test_from_bounds(self):
        profile = Profile.from_bounds((0, 0, 300, 200), 1)
        assert profile["height"] == 200
        assert profile.width == 300

    def test_from_bounds_with_kwargs(self):
        profile = Profile.from_bounds((0, 0, 300, 200), 1, count=2, tiled=True)
        assert profile.count == 2
        assert profile["tiled"] is True

    def test_from_geogrid(self):
        geogrid = GeoGrid.from_bounds((0, 0, 300, 200), res=1, crs="EPSG:4326")
        profile = Profile.from_geogrid(
            geogrid,
            nodata=-9999.0,
            count=2,
            tiled=True,
        )

        assert profile["height"] == 200
        assert profile.width == 300
        assert profile.crs.to_string() == "EPSG:4326"
        assert profile.nodata == -9999.0
        assert profile.count == 2
        assert profile["tiled"] is True

    def test_from_bounds_with_signed_resolution(self):
        profile = Profile.from_bounds((0, 0, 300, 200), (1, -1))
        assert profile["height"] == 200
        assert profile.width == 300
        assert profile.transform == Affine(1.0, 0.0, 0.0, 0.0, -1.0, 200.0)

    def test_from_xy_with_kwargs(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([2.0, 1.0, 0.0])
        profile = Profile.from_xy(x, y, crs="EPSG:4326", count=5, tiled=True)
        assert profile.count == 5
        assert profile["tiled"] is True
        assert profile.crs.to_string() == "EPSG:4326"

    def test_setitem_updates_geometry(self):
        profile = Profile.from_bounds((0, 0, 4, 2), 1)
        profile["width"] = 8
        profile["crs"] = "EPSG:4326"

        assert profile.width == 8
        assert profile.shape == (2, 8)
        assert profile.bounds == (0.0, 0.0, 8.0, 2.0)
        assert profile.crs.to_string() == "EPSG:4326"

    def test_metadata_properties(self):
        profile = Profile.from_bounds((0, 0, 4, 2), 1)
        profile.nodata = -9999
        profile.count = 3
        profile.driver = "COG"
        profile.dtype = "float32"
        profile.kwargs = {"compress": "lzw"}
        profile["tiled"] = True

        assert profile.nodata == -9999.0
        assert profile.count == 3
        assert profile.driver == "COG"
        assert profile.dtype == "float32"
        assert profile.kwargs["compress"] == "lzw"
        assert profile.kwargs["tiled"] is True

    def test_to_dict(self):
        profile = Profile(200, 300, Affine(*list(range(6))))
        profile_dict = profile.to_dict()
        assert profile_dict["height"] == 200
        assert profile_dict["width"] == 300
        assert profile_dict["transform"] == Affine(*list(range(6)))

    def test_mapping_unpacking(self):
        profile = Profile.from_bounds((0, 0, 4, 2), 1, count=2, tiled=True)

        def collect_kwargs(**kwargs):
            return kwargs

        collected = collect_kwargs(**profile)
        assert collected["height"] == 2
        assert collected["width"] == 4
        assert collected["count"] == 2
        assert collected["tiled"] is True
        assert dict(profile)["count"] == 2

    def test_to_file(self, tmp_path:Path):
        profile_file = tmp_path / "dummy.profile"
        profile = Profile(200, 300, Affine(*list(range(6))))
        profile.to_file(profile_file)
        assert profile_file.exists()
        assert profile_file.read_text() == str(profile.to_dict())
        # with different file extension
        profile_file = tmp_path / "dummy.tif"
        profile.to_file(profile_file)
        assert profile_file.with_suffix(".tif.profile").exists()

    def test_to_rasterio_profile(self):
        profile = Profile(200, 300, Affine(*list(range(6))))
        rasterio_profile = profile.to_rasterio_profile()
        assert rasterio_profile["height"] == 200
        assert rasterio_profile["width"] == 300
        assert isinstance(rasterio_profile, RasterioProfile)

    def test_get_xy(self,ascii_file_center):
        profile = Profile.from_ascii_header_file(ascii_file_center)
        x, y = profile.get_xy()
        assert len(x) == 100
        assert len(y) == 200
        assert y[-1] == 400 + 30
        assert y[0] == 400 + 30 * len(y)
        assert x[0] == 300
        assert x[-1] == 300 + 30 * (len(x)-1)
