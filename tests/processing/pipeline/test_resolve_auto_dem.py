"""Tests for the shared datum-aware auto-DEM resolution helper (PROPOSAL-0030).

Covers :func:`faninsar.processing.pipeline.production.resolve_auto_dem`: the
single wrap rule used by ``run_pair``, ``_run_pair_sweep``, and
``cli.frame.run_frame_cli``; source selection via ``dem_source``; and the
CLI fail-closed contract for unwired providers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from faninsar.processing.geometry.dem import RasterDEM
from faninsar.processing.pipeline.production import resolve_auto_dem


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
    """Patch get_dem_manager to return a fake manager with fixed datum."""
    datum_value = vertical_datum

    class _FakeManager:
        product = "glo30"
        provider = "aws"

        @property
        def vertical_datum(self) -> str:
            return datum_value

        def fetch_dem(
            self,
            bounds: tuple[float, float, float, float],
            output_path: Path,  # noqa: ARG002 - protocol signature
        ) -> Path:
            if seen is not None:
                seen["bounds"] = bounds
            return raster

    def fake_get_dem_manager(*, source: str | None = None) -> _FakeManager:
        if seen is not None:
            seen["source"] = source
        return _FakeManager()

    monkeypatch.setattr(
        "faninsar.processing.geometry.dem_manager.get_dem_manager",
        fake_get_dem_manager,
    )


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
        """Orthometric sources get GeoidAdjustedDEM when requested."""
        raster = _make_raster(tmp_path / "mosaic.tif")
        _install_fake_manager(monkeypatch, raster, vertical_datum="egm2008")
        wrapped = resolve_auto_dem(
            (45.2, 38.2, 45.8, 39.8),
            output_dir=tmp_path / "out",
            geoid_correction=True,
            dem_source="glo30",
        )
        assert type(wrapped).__name__ == "GeoidAdjustedDEM"

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
        assert seen["source"] is None

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


class TestCliDemSourceOption:
    """CLI fail-closed contract for unwired providers."""

    def test_unwired_dem_source_fails_closed(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--dem-source glo30:ot must exit non-zero citing unwired status."""
        from faninsar.cli.main import main

        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "frame",
                    "--reference",
                    "r.SAFE",
                    "--secondary",
                    "s.SAFE",
                    "--output",
                    str(tmp_path / "out"),
                    "--dem-source",
                    "glo30:ot",
                ]
            )
        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        err = (captured.err + captured.out).lower()
        assert "unwired" in err or "not wired" in err

    def test_unknown_product_fails_closed(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--dem-source not-a-product exits non-zero listing valid names."""
        from faninsar.cli.main import main

        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "frame",
                    "--reference",
                    "r.SAFE",
                    "--secondary",
                    "s.SAFE",
                    "--output",
                    str(tmp_path / "out"),
                    "--dem-source",
                    "not-a-product",
                ]
            )
        assert excinfo.value.code != 0
        captured = capsys.readouterr()
        assert "not-a-product" in (captured.err + captured.out)
