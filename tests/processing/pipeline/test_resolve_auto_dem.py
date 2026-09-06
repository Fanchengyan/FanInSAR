"""Tests for the shared datum-aware auto-DEM resolution helper (PROPOSAL-0030).

Covers :func:`faninsar.processing.stages.resolve_auto_dem`: the single wrap
rule used by Stack interferogram production and source selection via
``dem_source``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from faninsar.processing.geometry import DEM, RasterDEM
from faninsar.processing.stages import resolve_auto_dem


def _make_raster(path: Path) -> Path:
    """Create a tiny valid GeoTIFF for RasterDEM construction."""
    import numpy as np
    import rasterio
    from affine import Affine

    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 4,
        "count": 1,
        "dtype": "float32",
        "nodata": float("nan"),
        "crs": "EPSG:4326",
        "transform": Affine.translation(100.0, 39.0)
        * Affine.scale(1.0 / 3600, -1.0 / 3600),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((4, 4), 12.5, dtype="float32"), 1)
    return path


def _install_fake_manager(
    monkeypatch: pytest.MonkeyPatch,
    raster: Path,
    *,
    vertical_datum: str,
    seen: dict[str, object] | None = None,
) -> None:
    """Patch the unified DEM factory with an offline source fixture."""
    datum_value = vertical_datum

    class _FakeSource:
        product = "glo30"
        provider = "aws"

        def to_raster(self, grid: object, **kwargs: object) -> RasterDEM:
            if seen is not None:
                seen["grid"] = grid
            import numpy as np

            from faninsar.processing.geometry import GridSpec

            assert isinstance(grid, GridSpec)
            return RasterDEM(
                array=np.full(grid.shape, 12.5, dtype=np.float32),
                grid=grid,
                vertical_datum=str(kwargs.get("vertical_datum", datum_value)),
            )

    def fake_from_source(cls: type[DEM], source: str, **_kwargs: object) -> _FakeSource:
        if seen is not None:
            seen["source"] = source
        return _FakeSource()

    monkeypatch.setattr(DEM, "from_source", classmethod(fake_from_source))


class TestResolveAutoDem:
    """Datum-aware wrap rule and source forwarding."""

    def test_ellipsoidal_source_skips_wrap_even_with_geoid_correction(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ellipsoidal sources (e.g. arcticdem-32) are returned unwrapped."""
        raster = _make_raster(tmp_path / "mosaic.tif")
        seen: dict[str, object] = {}
        _install_fake_manager(
            monkeypatch, raster, vertical_datum="ellipsoidal", seen=seen
        )
        sampler = resolve_auto_dem(
            (45.2, 38.2, 45.8, 39.8),
            output_dir=tmp_path / "out",
            geoid_correction=True,
            dem_source="arcticdem-32",
        )
        assert isinstance(sampler, RasterDEM)
        assert type(sampler).__name__ != "GeoidAdjustedDEM"
        assert seen["source"] == "arcticdem-32"

    def test_wraps_orthometric_source_when_geoid_correction_enabled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Orthometric sources are materialized directly at ellipsoidal datum."""
        raster = _make_raster(tmp_path / "mosaic.tif")
        _install_fake_manager(monkeypatch, raster, vertical_datum="egm2008")
        wrapped = resolve_auto_dem(
            (45.2, 38.2, 45.8, 39.8),
            output_dir=tmp_path / "out",
            geoid_correction=True,
            dem_source="glo30",
        )
        assert isinstance(wrapped, RasterDEM)
        assert wrapped.vertical_datum == "ellipsoidal"

    def test_geoid_correction_false_skips_wrap_for_orthometric_source(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """geoid_correction=False keeps an orthometric DEM raw."""
        raster = _make_raster(tmp_path / "mosaic.tif")
        _install_fake_manager(monkeypatch, raster, vertical_datum="egm96")
        sampler = resolve_auto_dem(
            (45.2, 38.2, 45.8, 39.8),
            output_dir=tmp_path / "out",
            geoid_correction=False,
        )
        assert isinstance(sampler, RasterDEM)
        assert type(sampler).__name__ != "GeoidAdjustedDEM"

    def test_omitted_dem_source_defers_to_environment(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No explicit source kwarg reaches the manager factory."""
        raster = _make_raster(tmp_path / "mosaic.tif")
        seen: dict[str, object] = {}
        _install_fake_manager(monkeypatch, raster, vertical_datum="egm96", seen=seen)
        resolve_auto_dem(
            (45.2, 38.2, 45.8, 39.8),
            output_dir=tmp_path / "out",
            geoid_correction=False,
        )
        assert seen["source"] == "glo30"

    def test_requires_cache_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing FANINSAR_DEM_CACHE_DIR raises the structured error."""
        from faninsar.processing.errors import InvalidProcessingStateError

        monkeypatch.delenv("FANINSAR_DEM_CACHE_DIR", raising=False)
        with pytest.raises(InvalidProcessingStateError):
            resolve_auto_dem(
                (45.2, 38.2, 45.8, 39.8),
                output_dir=tmp_path / "out",
                geoid_correction=False,
            )
