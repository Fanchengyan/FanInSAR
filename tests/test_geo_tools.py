from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio import Affine
from rasterio.profiles import Profile as RasterioProfile

from faninsar.utils.geo_tools import Profile

profile = Profile(200, 300, Affine(*list(range(6))))


@pytest.fixture
def data_dir() -> Path:
    """Fixture for the data directory."""
    return Path(__file__).parent / "data" / "ascii"


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
        assert profile.height == 200
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
        assert profile.height == 200
        assert profile.width == 300


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



    def test_from_profile_file(self, ascii_file_corner, tmp_path:Path):
        profile_corner = Profile.from_ascii_header_file(ascii_file_corner)
        # save profile to tmp file
        profile_file = tmp_path / "dummy.profile"
        profile_corner.to_file(profile_file)
        # read profile from file
        profile = Profile.from_profile_file(profile_file)
        assert profile.height == 200
        assert profile.width == 100
        assert profile.nodata == -32768
        assert profile.transform == Affine(30.0, 0.0, 300.0, 0.0, -30.0, 6430.0)

    def test_from_bounds_res(self):
        profile = Profile.from_bounds_res((0, 0, 300, 200), 1)
        assert profile.height == 200
        assert profile.width == 300

    def test_to_dict(self):
        profile = Profile(200, 300, Affine(*list(range(6))))
        profile_dict = profile.to_dict()
        assert profile_dict["height"] == 200
        assert profile_dict["width"] == 300
        assert profile_dict["transform"] == Affine(*list(range(6)))

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

    def test_to_latlon(self,ascii_file_center):
        profile = Profile.from_ascii_header_file(ascii_file_center)
        lat, lon = profile.to_latlon()
        print(profile)
        print(lat, lon)
        assert len(lat) == 200
        assert len(lon) == 100
        assert lat[-1] == 400 + 30
        assert lat[0] == 400 + 30 * len(lat)
        assert lon[0] == 300
        assert lon[-1] == 300 + 30 * (len(lon)-1)
