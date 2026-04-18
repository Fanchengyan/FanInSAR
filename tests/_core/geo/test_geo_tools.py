import io
import zipfile
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest
import rasterio
import xarray as xr
from lxml import etree
from rasterio import Affine
from rasterio.profiles import Profile as RasterioProfile

from faninsar._core.geo import (
    GeoGrid,
    Profile,
    array2kml,
    array2kmz,
    bounds_from_xy,
    dataarray2kml,
    dataarray2kmz,
)

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
        assert tuple(profile.bounds) == (2.0, 5.0, 202.0, 1705.0)

    def test_res_property(self):
        profile = Profile(200, 300, Affine(*list(range(6))))
        assert profile.res == (0, 4)

    def test_from_raster_file(self, tmp_path: Path):
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
        assert profile_corner.transform == Affine(30.0, 0.0, 300.0, 0.0, -30.0, 6430.0)
        profile_center = Profile.from_ascii_header_file(ascii_file_center)
        assert profile_center.height == 200
        assert profile_center.width == 100
        assert profile_center.nodata == -32768
        assert profile_center.transform == Affine(30.0, 0.0, 285.0, 0.0, -30.0, 6415.0)

    def test_from_ascii_header_file_with_kwargs(self, ascii_file_corner):
        profile = Profile.from_ascii_header_file(
            ascii_file_corner,
            count=2,
            dtype="float32",
        )
        assert profile.count == 2
        assert profile.dtype == "float32"

    def test_from_profile_file(self, ascii_file_corner, tmp_path: Path):
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
        assert tuple(profile.bounds) == (0.0, 0.0, 8.0, 2.0)
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

    def test_to_file(self, tmp_path: Path):
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

    def test_get_xy(self, ascii_file_center):
        profile = Profile.from_ascii_header_file(ascii_file_center)
        x, y = profile.get_xy()
        assert len(x) == 100
        assert len(y) == 200
        assert y[-1] == 400 + 30
        assert y[0] == 400 + 30 * len(y)
        assert x[0] == 300
        assert x[-1] == 300 + 30 * (len(x) - 1)


def _read_kmz_xml(kmz_path: Path, member: str) -> etree._Element:
    """Read an XML member from a KMZ archive."""
    with zipfile.ZipFile(kmz_path) as kmz:
        return etree.fromstring(kmz.read(member))


def _kml_text(element: etree._Element, tag_name: str) -> str:
    """Extract the first text value for a KML tag."""
    return element.xpath(f"string(.//*[local-name()='{tag_name}'][1])")


def _spatial_dataarray(crs: str = "EPSG:4326") -> xr.DataArray:
    """Create a small data array with rioxarray geospatial metadata."""
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    data_array = xr.DataArray(
        arr,
        coords={"latitude": [2.5, 1.5, 0.5], "longitude": [10.5, 11.5, 12.5, 13.5]},
        dims=("latitude", "longitude"),
    )
    data_array = data_array.rio.set_spatial_dims(
        x_dim="longitude",
        y_dim="latitude",
    )
    return data_array.rio.write_crs(crs)


def test_array2kmz_tiled_writes_multilevel_kmz(tmp_path: Path) -> None:
    """Tiled KMZ export should produce a multilevel tile pyramid."""
    arr = np.arange(520 * 520, dtype=np.float32).reshape(520, 520)
    out_file = tmp_path / "multilevel_tiled_kmz.kmz"
    bounds = (-10.0, 20.0, 10.0, 40.0)

    array2kmz(
        arr,
        out_file,
        bounds,
        cbar_kwargs={"label": "Velocity"},
        render_scale=1,
        verbose=False,
        tiled=True,
    )

    with zipfile.ZipFile(out_file) as kmz:
        names = set(kmz.namelist())

    assert "doc.kml" in names
    assert "legend/colorbar.png" in names
    assert "tiles/0/0/0.kml" in names
    assert "tiles/0/0/0.png" in names
    assert "tiles/1/0/0.kml" in names
    assert "tiles/1/0/0.png" in names
    assert "tiles/2/0/0.kml" in names
    assert "tiles/2/0/0.png" in names

    root_doc = _read_kmz_xml(out_file, "doc.kml")
    assert root_doc.xpath("//*[local-name()='NetworkLink']")
    assert root_doc.xpath("//*[local-name()='ScreenOverlay']")

    root_tile = _read_kmz_xml(out_file, "tiles/0/0/0.kml")
    assert root_tile.xpath("//*[local-name()='GroundOverlay']")
    assert root_tile.xpath("//*[local-name()='Region']")
    assert root_tile.xpath("//*[local-name()='Lod']")
    assert len(root_tile.xpath("//*[local-name()='NetworkLink']")) == 4

    root_overlay = root_tile.xpath("//*[local-name()='GroundOverlay']")[0]
    assert float(_kml_text(root_overlay, "west")) == pytest.approx(bounds[0])
    assert float(_kml_text(root_overlay, "south")) == pytest.approx(bounds[1])
    assert float(_kml_text(root_overlay, "east")) == pytest.approx(bounds[2])
    assert float(_kml_text(root_overlay, "north")) == pytest.approx(bounds[3])

    child_tile = _read_kmz_xml(out_file, "tiles/1/0/0.kml")
    child_overlay = child_tile.xpath("//*[local-name()='GroundOverlay']")[0]
    assert float(_kml_text(child_overlay, "west")) == pytest.approx(-10.0)
    assert float(_kml_text(child_overlay, "south")) == pytest.approx(30.0)
    assert float(_kml_text(child_overlay, "east")) == pytest.approx(0.0)
    assert float(_kml_text(child_overlay, "north")) == pytest.approx(40.0)

    leaf_tile = _read_kmz_xml(out_file, "tiles/2/0/0.kml")
    assert leaf_tile.xpath("//*[local-name()='GroundOverlay']")
    assert leaf_tile.xpath("//*[local-name()='Region']")
    assert leaf_tile.xpath("//*[local-name()='Lod']")
    assert not leaf_tile.xpath("//*[local-name()='NetworkLink']")


def test_array2kmz_tiled_writes_single_level_kmz(tmp_path: Path) -> None:
    """Small inputs should only emit a single tile level."""
    arr = np.arange(80 * 64, dtype=np.float32).reshape(64, 80)
    out_file = tmp_path / "single_level_tiled_kmz.kmz"

    array2kmz(
        arr,
        out_file,
        (0.0, 0.0, 1.0, 1.0),
        render_scale=1,
        verbose=False,
        tiled=True,
    )

    with zipfile.ZipFile(out_file) as kmz:
        names = set(kmz.namelist())

    kml_members = {name for name in names if name.endswith(".kml")}
    png_members = {name for name in names if name.endswith(".png")}

    assert kml_members == {"doc.kml", "tiles/0/0/0.kml"}
    assert png_members == {"legend/colorbar.png", "tiles/0/0/0.png"}


def test_dataarray2kml_uses_rioxarray_spatial_metadata(tmp_path: Path) -> None:
    """DataArray KML export should derive bounds from configured xy dimensions."""
    data_array = _spatial_dataarray()
    out_file = tmp_path / "dataarray_overlay.kml"

    dataarray2kml(data_array, out_file, verbose=False)

    assert out_file.exists()
    assert out_file.with_suffix(".png").exists()
    assert out_file.with_name("dataarray_overlay_cbar.png").exists()

    root = etree.fromstring(out_file.read_bytes())
    expected_bounds = bounds_from_xy(data_array.longitude, data_array.latitude)
    assert float(_kml_text(root, "west")) == pytest.approx(expected_bounds[0])
    assert float(_kml_text(root, "south")) == pytest.approx(expected_bounds[1])
    assert float(_kml_text(root, "east")) == pytest.approx(expected_bounds[2])
    assert float(_kml_text(root, "north")) == pytest.approx(expected_bounds[3])


def test_dataarray2kmz_reprojects_to_wgs84(tmp_path: Path) -> None:
    """DataArray KMZ export should reproject non-WGS84 arrays before writing."""
    data_array = _spatial_dataarray().rio.reproject("EPSG:3857")
    out_file = tmp_path / "dataarray_overlay.kmz"

    dataarray2kmz(data_array, out_file, verbose=False)

    with zipfile.ZipFile(out_file) as kmz:
        root = etree.fromstring(kmz.read("dataarray_overlay.kml"))

    expected_bounds = bounds_from_xy(data_array.x, data_array.y, crs=data_array.rio.crs)
    expected_bounds = expected_bounds.to_crs("EPSG:4326")
    assert float(_kml_text(root, "west")) == pytest.approx(expected_bounds[0], abs=0.1)
    assert float(_kml_text(root, "south")) == pytest.approx(expected_bounds[1], abs=0.1)
    assert float(_kml_text(root, "east")) == pytest.approx(expected_bounds[2], abs=0.1)
    assert float(_kml_text(root, "north")) == pytest.approx(expected_bounds[3], abs=0.1)


def test_dataarray_fis_accessor_writes_kmz(tmp_path: Path) -> None:
    """The fis DataArray accessor should expose KMZ export."""
    data_array = _spatial_dataarray()
    out_file = tmp_path / "accessor_overlay.kmz"

    data_array.fis.to_kmz(out_file, render_scale=1, verbose=False, tiled=True)

    with zipfile.ZipFile(out_file) as kmz:
        names = set(kmz.namelist())

    assert "doc.kml" in names
    assert "legend/colorbar.png" in names
    assert "tiles/0/0/0.kml" in names


def test_array2kml_and_array2kmz_regression(tmp_path: Path) -> None:
    """Existing single-overlay exporters should keep their asset layout."""
    arr = np.arange(25, dtype=np.float32).reshape(5, 5)
    bounds = (0.0, 0.0, 1.0, 1.0)

    kml_file = tmp_path / "single_overlay.kml"
    array2kml(arr, kml_file, bounds, verbose=False)
    assert kml_file.exists()
    assert kml_file.with_suffix(".png").exists()
    assert kml_file.with_name("single_overlay_cbar.png").exists()
    kml_image = mpimg.imread(kml_file.with_suffix(".png"))
    assert kml_image.shape[:2] == (20, 20)

    kmz_file = tmp_path / "single_overlay_archive.kmz"
    array2kmz(arr, kmz_file, bounds, verbose=False)
    with zipfile.ZipFile(kmz_file) as kmz:
        names = set(kmz.namelist())
        kmz_image = mpimg.imread(io.BytesIO(kmz.read("single_overlay_archive.png")))

    assert names == {
        "single_overlay_archive.kml",
        "single_overlay_archive.png",
        "single_overlay_archive_cbar.png",
    }
    assert kmz_image.shape[:2] == (20, 20)
    assert not kmz_file.with_suffix(".kml").exists()
    assert not kmz_file.with_suffix(".png").exists()
    assert not kmz_file.with_name("single_overlay_archive_cbar.png").exists()


def test_array2kml_render_scale_repeats_source_pixels(tmp_path: Path) -> None:
    """Render scaling should repeat source pixels instead of interpolating them."""
    arr = np.arange(4, dtype=np.float32).reshape(2, 2)
    out_file = tmp_path / "scaled_overlay.kml"

    array2kml(
        arr,
        out_file,
        (0.0, 0.0, 1.0, 1.0),
        render_scale=3,
        verbose=False,
    )

    image = mpimg.imread(out_file.with_suffix(".png"))

    assert image.shape[:2] == (6, 6)
    for row_start in range(0, 6, 3):
        for col_start in range(0, 6, 3):
            pixel_block = image[row_start : row_start + 3, col_start : col_start + 3]
            assert np.all(pixel_block == pixel_block[0, 0])


def test_array2kml_render_scale_rejects_non_integer_values(tmp_path: Path) -> None:
    """Render scaling should only accept integer pixel repeat values."""
    arr = np.arange(4, dtype=np.float32).reshape(2, 2)

    with pytest.raises(ValueError, match="positive integer"):
        array2kml(
            arr,
            tmp_path / "scaled_overlay.kml",
            (0.0, 0.0, 1.0, 1.0),
            render_scale=1.5,
            verbose=False,
        )
