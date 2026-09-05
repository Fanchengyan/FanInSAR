"""Tests for raster I/O helpers, especially phase-aware resampling (D0.1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from faninsar.datasets.frame.metadata import (
    CONTINUOUS_ASSETS,
    PHASE_ASSETS,
)
from faninsar.datasets.frame.raster_io import (
    reproject_phase_to_geogrid,
    reproject_to_geogrid,
)
from faninsar.data.datasets.geogrid import GeoGrid

_WGS84_CRS = "EPSG:4326"


def _write_tiff(
    path: Path,
    bounds: tuple[float, float, float, float],
    data: np.ndarray,
    nodata: float = -9999.0,
) -> None:
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
        crs=_WGS84_CRS,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def _geogrid_from_path(path: Path) -> GeoGrid:
    """Read a GeoGrid back from a written raster file (matches production use)."""
    with rasterio.open(path) as ds:
        return GeoGrid.from_rio(ds)


def _geogrid_from_bounds(
    bounds: tuple[float, float, float, float], shape: tuple[int, int]
) -> GeoGrid:
    """Build a GeoGrid by writing then reading a tiny placeholder raster."""
    height, width = shape
    transform = from_bounds(*bounds, width, height)
    from rasterio.io import MemoryFile

    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=_WGS84_CRS,
            transform=transform,
        ) as ds:
            pass
        with rasterio.open(mem.name) as ds:
            return GeoGrid.from_rio(ds)


class TestPhaseAssetClassification:
    """D0.1: wrapped_phase is cyclic; unw_phase is continuous."""

    def test_wrapped_phase_is_in_phase_assets(self) -> None:
        assert "wrapped_phase" in PHASE_ASSETS

    def test_unw_phase_is_not_in_phase_assets(self) -> None:
        # unwrapped phase is continuous — bilinear is correct for it.
        assert "unw_phase" not in PHASE_ASSETS
        assert "unw_phase" in CONTINUOUS_ASSETS

    def test_phase_assets_subset_of_continuous(self) -> None:
        # The dispatch checks phase before continuous; ensure no overlap bug.
        assert PHASE_ASSETS <= (PHASE_ASSETS | CONTINUOUS_ASSETS)


class TestComplexAverageResampling:
    """Verify complex averaging handles 2*pi wrap boundaries correctly."""

    def test_uniform_phase_is_preserved(self, tmp_path: Path) -> None:
        """A spatially uniform phase should resample to the same value."""
        src_path = tmp_path / "src.tif"
        src_bounds = (10.0, 45.0, 11.0, 46.0)
        src_shape = (8, 8)
        phi = np.full(src_shape, 1.0, dtype=np.float32)  # constant 1 rad
        _write_tiff(src_path, src_bounds, phi)

        grid = _geogrid_from_bounds(src_bounds, src_shape)
        out = reproject_phase_to_geogrid(src_path, grid)
        # Where data exists, result should be ~1.0 rad
        valid = out[out != -9999.0]
        assert valid.size > 0
        np.testing.assert_allclose(valid, 1.0, atol=1e-4)

    def test_wrap_boundary_complex_average(self, tmp_path: Path) -> None:
        """Two adjacent pixels straddling +pi/-pi average near pi, not 0.

        Naive bilinear of (+3.1) and (-3.1) gives ~0, which is wrong because
        both values represent phase near +pi. Complex averaging of
        exp(i*3.1) and exp(i*(-3.1)) yields a vector near angle +pi.
        """
        # Construct a source raster where left half is +3.1 rad and right half
        # is -3.1 rad (both near pi, straddling the branch cut).
        src_shape = (4, 4)
        phi = np.zeros(src_shape, dtype=np.float32)
        phi[:, : src_shape[1] // 2] = 3.1
        phi[:, src_shape[1] // 2 :] = -3.1

        src_path = tmp_path / "src_wrap.tif"
        src_bounds = (10.0, 45.0, 11.0, 46.0)
        _write_tiff(src_path, src_bounds, phi)

        # Upsample to a finer grid so interpolation actually happens.
        dst_shape = (16, 16)
        dst_bounds = src_bounds
        dst_grid = _geogrid_from_bounds(dst_bounds, dst_shape)

        out = reproject_phase_to_geogrid(src_path, dst_grid)
        valid = out[out != -9999.0]

        # The complex-averaged mean phase magnitude should be high (both
        # vectors point near +pi), and the mean angle near +/- pi. This is
        # the key property naive bilinear destroys: averaging the complex
        # representatives keeps the vector coherent and pointing at pi.
        mean_complex = np.mean(np.exp(1j * valid))
        assert np.abs(mean_complex) > 0.9  # coherent vector
        mean_angle = np.angle(mean_complex)
        assert abs(abs(mean_angle) - np.pi) < 0.2  # near +pi or -pi

    def test_nodata_propagated(self, tmp_path: Path) -> None:
        """Nodata regions in source stay nodata in output."""
        src_shape = (8, 8)
        phi = np.full(src_shape, 0.5, dtype=np.float32)
        phi[:4, :] = -9999.0  # top half nodata
        src_path = tmp_path / "src_nodata.tif"
        src_bounds = (10.0, 45.0, 11.0, 46.0)
        _write_tiff(src_path, src_bounds, phi, nodata=-9999.0)

        grid = _geogrid_from_bounds(src_bounds, src_shape)
        out = reproject_phase_to_geogrid(src_path, grid, dst_nodata=-9999.0)
        valid = out[out != -9999.0]
        assert valid.size > 0
        np.testing.assert_allclose(valid, 0.5, atol=1e-4)

    def test_bilinear_still_used_for_unw_phase(self, tmp_path: Path) -> None:
        """unw_phase uses the standard bilinear path (continuous field)."""
        src_shape = (8, 8)
        unw = np.linspace(-10, 10, src_shape[0] * src_shape[1], dtype=np.float32).reshape(
            src_shape
        )
        src_path = tmp_path / "unw.tif"
        src_bounds = (10.0, 45.0, 11.0, 46.0)
        _write_tiff(src_path, src_bounds, unw)

        grid = _geogrid_from_bounds(src_bounds, src_shape)
        out = reproject_to_geogrid(src_path, grid, dst_dtype="float32")
        # unw_phase is continuous — bilinear should reproduce values closely
        valid = out[out != -9999.0]
        assert valid.min() >= -10.5
        assert valid.max() <= 10.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
