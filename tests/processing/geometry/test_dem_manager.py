"""Tests for the multi-source automatic DEM manager (PROPOSAL-0030).

Covers the reshaped :class:`DEMManager`: selection resolution, ``auto``
control-tile-guarded fallback, cache partitions, mosaic provenance tags,
structured outage errors, and environment parsing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry import dem_manager as dm
from faninsar.processing.geometry.dem_manager import (
    DEMManager,
    DEMProviderUnavailableError,
    copernicus_tile_name,
    default_dem_name,
    get_dem_manager,
)
from faninsar.data.query import BoundingBox


def _write_tile(
    path: Path,
    *,
    latitude: int,
    longitude: int,
    rows: int = 512,
    cols: int = 512,
    value: float = 100.0,
) -> Path:
    """Write one uncompressed EPSG:4326 float32 tile (>= 1 MiB on disk)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = Affine.translation(
        float(longitude), float(latitude + 1)
    ) * Affine.scale(1.0 / cols, -1.0 / rows)
    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((rows, cols), value, dtype="float32"), 1)
    return path


# ---------------------------------------------------------------------------
# Fake HTTP layer (mirrors the transport-test conventions)
# ---------------------------------------------------------------------------


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(
        self,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ) -> None:
        """Build the fake response."""
        self.status_code = status
        self.headers = headers or {}
        self._body = body

    def iter_content(self, chunk_size: int) -> object:
        """Yield the scripted body in one block."""
        del chunk_size
        yield self._body

    def close(self) -> None:
        """No-op close."""


class HttpFake:
    """Scripted host serving registered URLs with ranged-GET semantics."""

    def __init__(self) -> None:
        """Initialize empty scripting tables."""
        self.bodies: dict[str, bytes] = {}
        self.statuses: dict[str, int] = {}
        self.calls: list[tuple[str, str]] = []

    def register_ok(self, url: str, body: bytes) -> None:
        """Serve 200 with ``body`` for ``url``."""
        self.statuses[url] = 200
        self.bodies[url] = body

    def register_missing(self, url: str) -> None:
        """Serve 404 for ``url``."""
        self.statuses[url] = 404

    def register_error(self, url: str, status: int) -> None:
        """Serve the given error status for ``url``."""
        self.statuses[url] = status

    def _respond(self, method: str, url: str, headers: dict[str, str]) -> FakeResponse:
        self.calls.append((method, url))
        status = self.statuses.get(url, 404)
        if status != 200:
            return FakeResponse(
                status=status,
                headers={"Content-Type": "application/xml"},
                body=b"<Error><Code>NoSuchKey</Code></Error>",
            )
        body = self.bodies[url]
        range_header = headers.get("Range")
        if method == "GET" and range_header:
            start_s, end_s = range_header.removeprefix("bytes=").split("-")
            start, end = int(start_s), int(end_s)
            return FakeResponse(
                status=206,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{len(body)}",
                    "Content-Length": str(end - start + 1),
                },
                body=body[start : end + 1],
            )
        if method == "HEAD":
            return FakeResponse(
                status=200,
                headers={
                    "Content-Length": str(len(body)),
                    "Accept-Ranges": "bytes",
                },
            )
        return FakeResponse(
            status=200,
            headers={"Content-Length": str(len(body))},
            body=body,
        )

    def install(self, monkeypatch: pytest.MonkeyPatch) -> HttpFake:
        """Route transport sessions through this fake."""
        monkeypatch.setattr(
            "faninsar.processing.geometry.dem_transport.thread_local_session",
            lambda: self,
        )
        return self

    def head(self, url: str, **kwargs: object) -> FakeResponse:
        """Serve a scripted HEAD."""
        del kwargs
        return self._respond("HEAD", url, {})

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        """Serve a scripted GET."""
        return self._respond("GET", url, kwargs.get("headers") or {})

    def request(self, method: str, url: str, **kwargs: object) -> FakeResponse:
        """Serve a scripted request of any method."""
        return self._respond(method, url, kwargs.get("headers") or {})


def _seed_body(tmp_path: Path, latitude: int, longitude: int, value: float) -> bytes:
    """Build a >= 1 MiB valid COG-shaped tile body for fake downloads."""
    path = _write_tile(
        tmp_path / "seed" / f"{latitude}_{longitude}.tif",
        latitude=latitude,
        longitude=longitude,
        value=value,
    )
    return path.read_bytes()


def _copernicus_urls(base: str, stem_prefix: str, lat: int, lon: int) -> str:
    """Build the bucket URL with the given COG stem prefix.

    Use 10 for GLO-30 and 30 for GLO-90, matching the registry's
    LatLonGridSource layout.
    """
    lat_abs = abs(lat)
    lon_abs = abs(lon)
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    stem = f"{stem_prefix}_{ns}{lat_abs:02d}_00_{ew}{lon_abs:03d}_00_DEM"
    filename = f"{stem}.tif"
    return f"{base}/{stem}/{filename}"


GLO30_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
GLO90_BASE = "https://copernicus-dem-90m.s3.amazonaws.com"


def _glo90_url(lat: int, lon: int) -> str:
    return _copernicus_urls(GLO90_BASE, "Copernicus_DSM_COG_30", lat, lon)


def _glo30_url_at(lat: int, lon: int) -> str:
    return _copernicus_urls(GLO30_BASE, "Copernicus_DSM_COG_10", lat, lon)


# ---------------------------------------------------------------------------
# Constructor and selection resolution
# ---------------------------------------------------------------------------


class TestConstructorSourceResolution:
    def test_accepts_product_compound_and_instance(self, tmp_path: Path) -> None:
        assert DEMManager(tmp_path, source="glo30").product == "glo30"
        assert DEMManager(tmp_path, source="glo30:aws").provider == "aws"
        from faninsar.processing.geometry.dem_sources import get_dem_source

        entry = get_dem_source("glo90")
        manager = DEMManager(tmp_path, source=entry)
        assert manager.product == "glo90"
        assert manager.provider == "aws"

    def test_resolution_kwarg_beats_env_beats_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FANINSAR_DEM_SOURCE", "glo90")
        assert DEMManager(tmp_path).product == "glo90"
        assert DEMManager(tmp_path, source="glo30").product == "glo30"
        monkeypatch.delenv("FANINSAR_DEM_SOURCE")
        assert DEMManager(tmp_path).product == "glo30"

    def test_env_carries_compound_value(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FANINSAR_DEM_SOURCE", "alos-dem:jaxa-ftp")
        manager = DEMManager(tmp_path)
        assert manager.product == "alos-dem"
        assert manager.provider == "jaxa-ftp"

    def test_auto_with_non_default_provider_rejected_loudly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(dm.logger, "error", lambda *a, **k: None)
        with pytest.raises(ValueError, match="auto"):
            DEMManager(tmp_path, source="auto:pc")

    def test_unwired_pair_fails_closed_at_construction(
        self,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="[Uu]nwired|not wired"):
            DEMManager(tmp_path, source="glo30:ot")


# ---------------------------------------------------------------------------
# Cache partitions and legacy flat probe
# ---------------------------------------------------------------------------


class TestCachePartitions:
    def test_downloads_land_in_partition_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = HttpFake().install(monkeypatch)
        body = _seed_body(tmp_path, 38, 100, 42.0)
        fake.register_ok(_glo30_url_at(38, 100), body)
        manager = DEMManager(tmp_path / "cache")
        manager.fetch_dem((100.2, 38.2, 100.8, 38.8), tmp_path / "out" / "dem.tif")
        downloaded = (
            tmp_path
            / "cache"
            / "glo30-aws"
            / "N38_E100"
            / "Copernicus_DSM_COG_10_N38_00_E100_00_DEM"
            / "Copernicus_DSM_COG_10_N38_00_E100_00_DEM.tif"
        )
        assert downloaded.is_file()
        assert downloaded.stat().st_size == len(body)

    def test_legacy_flat_cache_hit_bound_to_glo30_aws(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache = tmp_path / "cache"
        _write_tile(
            cache / "Copernicus_DSM_COG_10_N38_00_E100_00_DEM.tif",
            latitude=38,
            longitude=100,
        )

        def fail(*args: object, **kwargs: object) -> None:
            raise AssertionError("network fetch must not run on a legacy hit")

        monkeypatch.setattr(
            "faninsar.processing.geometry.dem_transport.thread_local_session",
            fail,
        )
        manager = DEMManager(cache, source="glo30")
        out = tmp_path / "out" / "dem.tif"
        manager.fetch_dem((100.2, 38.2, 100.8, 38.8), out)
        assert out.exists()

    def test_legacy_flat_probe_not_used_for_other_sources(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache = tmp_path / "cache"
        # A flat file carrying the GLO-90 stem must not be adopted as a hit
        # through the legacy probe (legacy probe binds to glo30@aws only);
        # the manager must go to the network for the glo90 selection.
        flat = cache / "Copernicus_DSM_COG_30_N38_00_E100_00_DEM.tif"
        _write_tile(flat, latitude=38, longitude=100)

        class Boom(HttpFake):
            def head(self, url: str, **kwargs: object):
                raise AssertionError(f"unexpected request {url}")

        fake = Boom()
        fake.install(monkeypatch)
        manager = DEMManager(cache, source="glo90")
        with pytest.raises(AssertionError, match="unexpected request"):
            manager.fetch_dem((100.2, 38.2, 100.8, 38.8), tmp_path / "o.tif")

    def test_partition_labels_are_opaque_write_targets(self, tmp_path: Path) -> None:
        manager = DEMManager(tmp_path / "cache", source="alos-dem:jaxa-ftp")
        assert manager.partition_name == "alos-dem-jaxa-ftp"
        assert manager.partition_dir == tmp_path / "cache" / "alos-dem-jaxa-ftp"


# ---------------------------------------------------------------------------
# auto: control-tile-guarded GLO-90 per-cell fallback
# ---------------------------------------------------------------------------

CONTROL_LAT, CONTROL_LON = dm.AUTO_CONTROL_CELL


class TestAutoFallback:
    def test_withheld_cell_rescues_via_glo90_and_stays_30m(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        fake = HttpFake().install(monkeypatch)
        # Cell N38E045 is withheld on GLO-30; N39E045 is present.
        fake.register_missing(_glo30_url_at(38, 45))
        fake.register_ok(_glo30_url_at(39, 45), _seed_body(tmp_path, 39, 45, 55.0))
        fake.register_ok(_glo90_url(38, 45), _seed_body(tmp_path, 38, 45, 77.0))

        manager = DEMManager(tmp_path / "cache", source="auto")
        out = tmp_path / "out" / "dem.tif"
        with caplog.at_level(
            logging.WARNING, logger="faninsar.processing.geometry.dem_manager"
        ):
            result = manager.fetch_dem((45.2, 38.2, 45.8, 39.8), out)

        assert result == out
        # Fallback tile cached under the auto selection directory.
        assert (tmp_path / "cache" / "auto" / "glo90").is_dir()
        assert (tmp_path / "cache" / "auto" / "glo30").is_dir()
        # Explicit merge resolution: the 90 m fallback must not downgrade
        # the mosaic away from the 30 m primary resolution.
        with rasterio.open(out) as dataset:
            assert dataset.transform.a == pytest.approx(1.0 / 3600)
            band = dataset.read(1)
        assert np.isfinite(band).any()
        # 50% fallback fraction crosses the 25% warning threshold.
        assert any("fallback" in record.message.lower() for record in caplog.records)

    def test_all_withheld_roi_rescues_via_canonical_control_tile(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = HttpFake().install(monkeypatch)
        fake.register_missing(_glo30_url_at(38, 45))
        fake.register_missing(_glo30_url_at(39, 45))
        # Canonical out-of-ROI control cell is present on the primary base.
        fake.register_ok(
            _glo30_url_at(CONTROL_LAT, CONTROL_LON),
            _seed_body(tmp_path, CONTROL_LAT, CONTROL_LON, 10.0),
        )
        fake.register_ok(_glo90_url(38, 45), _seed_body(tmp_path, 38, 45, 77.0))
        fake.register_ok(_glo90_url(39, 45), _seed_body(tmp_path, 39, 45, 88.0))

        manager = DEMManager(tmp_path / "cache", source="auto")
        out = tmp_path / "out" / "dem.tif"
        manager.fetch_dem((45.2, 38.2, 45.8, 39.8), out)
        with rasterio.open(out) as dataset:
            assert dataset.transform.a == pytest.approx(1.0 / 3600)
            band = dataset.read(1)
        assert float(np.nanmax(band)) == pytest.approx(88.0)

    def test_control_tile_failure_is_loud_mirror_misconfiguration(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = HttpFake().install(monkeypatch)
        fake.register_missing(_glo30_url_at(38, 45))
        fake.register_missing(_glo30_url_at(CONTROL_LAT, CONTROL_LON))

        manager = DEMManager(tmp_path / "cache", source="auto")
        with pytest.raises(DEMProviderUnavailableError, match="control|mirror"):
            manager.fetch_dem((45.2, 38.2, 45.8, 38.8), tmp_path / "out" / "dem.tif")
        # The rescue must never start: no GLO-90 requests were made.
        assert not any("copernicus-dem-90m" in url for _, url in fake.calls)

    def test_direct_selection_caches_under_own_partition_not_auto(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = HttpFake().install(monkeypatch)
        fake.register_ok(_glo90_url(38, 45), _seed_body(tmp_path, 38, 45, 9.0))
        manager = DEMManager(tmp_path / "cache", source="glo90")
        manager.fetch_dem((45.2, 38.2, 45.8, 38.8), tmp_path / "out" / "dem.tif")
        assert (tmp_path / "cache" / "glo90-aws").is_dir()
        assert not (tmp_path / "cache" / "auto").exists()


# ---------------------------------------------------------------------------
# fetch_dem: mosaic, provenance tags, structured errors
# ---------------------------------------------------------------------------


class TestFetchDemContract:
    def test_provenance_tags_stamped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = HttpFake().install(monkeypatch)
        fake.register_ok(_glo30_url_at(38, 100), _seed_body(tmp_path, 38, 100, 5.0))
        manager = DEMManager(tmp_path / "cache")
        out = tmp_path / "out" / "dem.tif"
        manager.fetch_dem((100.2, 38.2, 100.8, 38.8), out)
        with rasterio.open(out) as dataset:
            tags = dataset.tags()
        assert tags["dem_product"] == "glo30"
        assert tags["dem_provider"] == "aws"
        assert tags["dem_vertical_datum"] == "egm2008"
        assert tags["dem_host"] == "copernicus-dem-30m.s3.amazonaws.com"
        assert tags["dem_retrieved"]

    def test_contradicting_output_tags_warn_but_are_used(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        cache = tmp_path / "cache"
        _write_tile(
            cache / "Copernicus_DSM_COG_10_N38_00_E100_00_DEM.tif",
            latitude=38,
            longitude=100,
        )
        out = tmp_path / "out" / "dem.tif"
        out.parent.mkdir(parents=True)
        # Pre-existing mosaic stamped with a foreign product identity.
        with rasterio.open(
            out,
            "w",
            driver="GTiff",
            height=8,
            width=8,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=Affine.translation(100.0, 39.0),
        ) as dst:
            dst.write(np.full((8, 8), 1.0, dtype="float32"), 1)
            dst.update_tags(dem_product="glo90", dem_provider="aws")

        manager = DEMManager(cache)
        with caplog.at_level(
            logging.WARNING, logger="faninsar.processing.geometry.dem_manager"
        ):
            manager.fetch_dem((100.2, 38.2, 100.8, 38.8), out)
        assert any("contradict" in record.message.lower() for record in caplog.records)
        # Use-after-warning: the fetch completed and rewrote the mosaic.
        with rasterio.open(out) as dataset:
            assert dataset.tags().get("dem_product") == "glo30"

    def test_resumable_not_transactional(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = HttpFake().install(monkeypatch)
        fake.register_ok(_glo30_url_at(38, 100), _seed_body(tmp_path, 38, 100, 5.0))
        fake.register_error(_glo30_url_at(39, 100), 500)
        monkeypatch.setattr(
            "faninsar.processing.geometry.dem_transport.compute_backoff_sleep",
            lambda *_a, **_k: 0.0,
        )
        manager = DEMManager(tmp_path / "cache")
        with pytest.raises(Exception):  # noqa: B017, PT011 - transport error type
            manager.fetch_dem((100.2, 38.2, 100.8, 39.8), tmp_path / "out" / "dem.tif")
        completed = (
            tmp_path
            / "cache"
            / "glo30-aws"
            / "N38_E100"
            / "Copernicus_DSM_COG_10_N38_00_E100_00_DEM"
            / "Copernicus_DSM_COG_10_N38_00_E100_00_DEM.tif"
        )
        # Completed tiles persist even though the mosaic was never written.
        assert completed.is_file()
        assert not (tmp_path / "out" / "dem.tif").exists()

    def test_structured_outage_error_excludes_unwired_alternatives(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = HttpFake().install(monkeypatch)
        fake.register_error(_glo30_url_at(38, 100), 500)
        monkeypatch.setattr(
            "faninsar.processing.geometry.dem_transport.compute_backoff_sleep",
            lambda *_a, **_k: 0.0,
        )
        manager = DEMManager(tmp_path / "cache")
        with pytest.raises(DEMProviderUnavailableError) as excinfo:
            manager.fetch_dem((100.2, 38.2, 100.8, 38.8), tmp_path / "o.tif")
        error = excinfo.value
        assert error.product == "glo30"
        assert error.provider == "aws"
        assert error.host == "copernicus-dem-30m.s3.amazonaws.com"
        assert error.failure_class == "upstream-outage"
        assert error.attempts >= 2
        # Unwired providers (ot) must never appear as v1 alternatives.
        assert "ot" not in error.alternatives
        assert "pc" in error.alternatives
        assert "cache" in str(error).lower()

    def test_terminal_403_maps_to_forbidden_failure_class(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = HttpFake().install(monkeypatch)
        fake.register_error(_glo30_url_at(38, 100), 403)
        manager = DEMManager(tmp_path / "cache")
        with pytest.raises(DEMProviderUnavailableError) as excinfo:
            manager.fetch_dem((100.2, 38.2, 100.8, 38.8), tmp_path / "o.tif")
        assert excinfo.value.failure_class == "forbidden"

    def test_required_tiles_returns_tile_records(self, tmp_path: Path) -> None:
        from faninsar.processing.geometry.dem_transport import Tile

        manager = DEMManager(tmp_path)
        tiles = manager.required_tiles((100.2, 38.2, 101.4, 38.8))
        assert len(tiles) == 2
        assert all(isinstance(tile, Tile) for tile in tiles)
        assert tiles[0].url.endswith(
            "Copernicus_DSM_COG_10_N38_00_E100_00_DEM/"
            "Copernicus_DSM_COG_10_N38_00_E100_00_DEM.tif"
        )

    def test_required_tiles_rejects_artifact_shapes(self, tmp_path: Path) -> None:
        manager = DEMManager(tmp_path, source="alos-dem:jaxa-ftp")
        with pytest.raises(InvalidProcessingStateError, match="[Aa]rtifact"):
            manager.required_tiles((6.2, 0.2, 6.4, 0.4))


# ---------------------------------------------------------------------------
# get_dem_manager environment contract
# ---------------------------------------------------------------------------


class TestGetDemManagerEnvironment:
    def test_reads_cache_and_source_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FANINSAR_DEM_CACHE_DIR", "/tmp/dem-cache")
        monkeypatch.setenv("FANINSAR_DEM_SOURCE", "glo90:aws")
        monkeypatch.setenv("FANINSAR_DEM_SOURCE_URL", "https://mirror.test/dem")
        manager = get_dem_manager()
        assert manager.cache_dir == Path("/tmp/dem-cache")
        assert manager.product == "glo90"
        assert manager.source_entry.base_url == "https://mirror.test/dem"

    def test_requires_cache_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FANINSAR_DEM_CACHE_DIR", raising=False)
        with pytest.raises(InvalidProcessingStateError):
            get_dem_manager()

    def test_source_url_must_be_https(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FANINSAR_DEM_CACHE_DIR", "/tmp/dem-cache")
        monkeypatch.setenv("FANINSAR_DEM_SOURCE_URL", "http://insecure.test/dem")
        with pytest.raises(InvalidProcessingStateError, match="https"):
            get_dem_manager()

    def test_source_url_override_warns_on_non_default_source(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("FANINSAR_DEM_CACHE_DIR", "/tmp/dem-cache")
        monkeypatch.setenv("FANINSAR_DEM_SOURCE", "glo90")
        monkeypatch.setenv("FANINSAR_DEM_SOURCE_URL", "https://mirror.test/dem")
        with caplog.at_level(
            logging.WARNING, logger="faninsar.processing.geometry.dem_manager"
        ):
            get_dem_manager()
        assert any(
            "override" in record.message.lower()
            or "non-default" in record.message.lower()
            for record in caplog.records
        )

    def test_source_url_override_rejects_unsupported_shape(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FANINSAR_DEM_CACHE_DIR", "/tmp/dem-cache")
        monkeypatch.setenv("FANINSAR_DEM_SOURCE", "alos-dem:jaxa-ftp")
        monkeypatch.setenv("FANINSAR_DEM_SOURCE_URL", "https://mirror.test/dem")
        with pytest.raises(InvalidProcessingStateError):
            get_dem_manager()


# ---------------------------------------------------------------------------
# BLOCKER-0030-B2: 2m-tier fan-out consumes every part end-to-end
# ---------------------------------------------------------------------------


def _seed_pgc_2m_body(tmp_path: Path, value: float) -> bytes:
    """Write a >= 1 MiB EPSG:3413 GeoTIFF body for a 2m quad sub-tile."""
    path = tmp_path / f"seed-2m-{int(value)}.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": 512,
        "width": 512,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3413",
        "transform": Affine.translation(-512.0, -1_668_976.0) * Affine.scale(2.0, -2.0),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((512, 512), value, dtype="float32"), 1)
    return path.read_bytes()


class TestMultiTileTwoMeterTier:
    def test_arcticdem_2m_fetch_dem_consumes_every_part(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """BLOCKER-0030-B2: 2m fetch_dem fetches and mosaics all four sub-tiles.

        The plan's ``_MultiTileTile`` fan-out must be expanded by the
        manager: every ``{quad}_{r}_{c}_2m_v4.1_dem.tif`` part is fetched
        into the partition cache and a second run is a full cache hit.
        """
        from faninsar.processing.geometry.dem_sources import get_dem_source

        class _ScriptedQuadEnumerator:
            def __init__(self, quads: list[str]) -> None:
                self.quads = quads

            def quads_for_bounds(self, bounds) -> list[str]:
                return list(self.quads)

            def list_quads(self, prefix: str) -> tuple[list[str], bool]:
                del prefix
                return list(self.quads), False

        source = get_dem_source("arcticdem-2")
        source.quad_enumerator = _ScriptedQuadEnumerator(["07_40"])
        fake = HttpFake().install(monkeypatch)
        bodies: dict[str, bytes] = {}
        for row in (1, 2):
            for col in (1, 2):
                filename = f"07_40_{row}_{col}_2m_v4.1_dem.tif"
                url = (
                    "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/"
                    f"arcticdem/mosaics/v4.1/2m/07_40/{filename}"
                )
                body = _seed_pgc_2m_body(tmp_path, float(100 * row + col))
                fake.register_ok(url, body)
                bodies[url] = body

        manager = DEMManager(tmp_path / "cache", source=source)
        out = tmp_path / "out" / "dem.tif"
        manager.fetch_dem((-70.0, 75.0, -60.0, 80.0), out)

        gets = [url for method, url in fake.calls if method == "GET"]
        fetched = set(bodies) & set(gets)
        assert fetched == set(bodies)
        assert len(fetched) == 4
        for filename in (
            "07_40_1_1_2m_v4.1_dem.tif",
            "07_40_1_2_2m_v4.1_dem.tif",
            "07_40_2_1_2m_v4.1_dem.tif",
            "07_40_2_2_2m_v4.1_dem.tif",
        ):
            cached = (
                tmp_path
                / "cache"
                / "arcticdem-2-aws"
                / "arcticdem-v4.1-2m"
                / "07_40"
                / filename
            )
            assert cached.is_file(), filename

        # A second run must be a complete cache hit: no new requests.
        fake.calls.clear()
        manager.fetch_dem((-70.0, 75.0, -60.0, 80.0), out)
        assert fake.calls == []

        with rasterio.open(out) as dataset:
            band = dataset.read(1)
        assert np.isfinite(band).any()


# ---------------------------------------------------------------------------
# NOTE-p30 hygiene: _paths_for_plan TileSet branch stays self-consistent
# ---------------------------------------------------------------------------


class TestPathsForPlanTileSetBranch:
    def test_tileset_branch_expands_multi_tile_parts(self, tmp_path: Path) -> None:
        """The TileSet branch expands ``_MultiTileTile`` fan-out records.

        Every concrete sub-tile of every planned tile must resolve to a
        mosaic input; without ``expand_tile_parts`` the 2m sub-tiles
        would be silently dropped (the retired BLOCKER-0030-B2 shape).
        """
        from faninsar.processing.geometry.dem_sources import _MultiTileTile
        from faninsar.processing.geometry.dem_transport import Tile, TileSet

        manager = DEMManager(tmp_path, source="arcticdem-2")
        parts = tuple(
            Tile(
                url=(
                    "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/"
                    f"arcticdem/mosaics/v4.1/2m/07_40/07_40_{row}_{col}"
                    "_2m_v4.1_dem.tif"
                ),
                cache_path=Path("arcticdem-v4.1-2m/07_40")
                / f"07_40_{row}_{col}_2m_v4.1_dem.tif",
                min_bytes=1,
            )
            for row in (1, 2)
            for col in (1, 2)
        )
        plain = Tile(
            url=(
                "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/"
                "arcticdem/mosaics/v4.1/2m/07_40/07_40_plain.tif"
            ),
            cache_path=Path("arcticdem-v4.1-2m/07_40/07_40_plain.tif"),
            min_bytes=1,
        )
        for tile in (*parts, plain):
            cached = tmp_path / tile.cache_path
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(b"dem")
        plan = TileSet(
            allowed_hosts=("pgc-opendata-dems.s3.us-west-2.amazonaws.com",),
            tiles=(_MultiTileTile(*parts), plain),
        )

        paths = manager._paths_for_plan(plan, executed=[])

        expected = {tmp_path / tile.cache_path for tile in (*parts, plain)}
        assert set(paths) == expected
        assert len(paths) == len(expected)

    def test_tileset_branch_prefers_executed_paths(self, tmp_path: Path) -> None:
        """Executed paths win over cache probing and are deduplicated."""
        from faninsar.processing.geometry.dem_sources import _MultiTileTile
        from faninsar.processing.geometry.dem_transport import Tile, TileSet

        manager = DEMManager(tmp_path, source="arcticdem-2")
        fan_out = _MultiTileTile(
            Tile(
                url="https://example.test/07_40_1_1_2m_v4.1_dem.tif",
                cache_path=Path("arcticdem-v4.1-2m/07_40")
                / "07_40_1_1_2m_v4.1_dem.tif",
                min_bytes=1,
            ),
            Tile(
                url="https://example.test/07_40_1_2_2m_v4.1_dem.tif",
                cache_path=Path("arcticdem-v4.1-2m/07_40")
                / "07_40_1_2_2m_v4.1_dem.tif",
                min_bytes=1,
            ),
        )
        plan = TileSet(
            allowed_hosts=("example.test",),
            tiles=(fan_out,),
        )
        first = tmp_path / "arcticdem-v4.1-2m/07_40/07_40_1_1_2m_v4.1_dem.tif"
        second = tmp_path / "arcticdem-v4.1-2m/07_40/07_40_1_2_2m_v4.1_dem.tif"

        paths = manager._paths_for_plan(plan, executed=[first, first, second])

        assert paths == [first, second]


# ---------------------------------------------------------------------------
# Shared helpers and misc
# ---------------------------------------------------------------------------


def test_copernicus_tile_name_convention() -> None:
    """Tile naming follows the Copernicus GLO-30 COG convention."""
    assert copernicus_tile_name(38.5, 100.5) == (
        "N38_E100",
        "Copernicus_DSM_COG_10_N38_00_E100_00_DEM.tif",
    )
    assert copernicus_tile_name(-2.5, -71.5) == (
        "S03_W072",
        "Copernicus_DSM_COG_10_S03_00_W072_00_DEM.tif",
    )


def test_default_dem_name_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """FANINSAR_DEM_NAME overrides the default output name."""
    monkeypatch.setenv("FANINSAR_DEM_NAME", "merged.tif")
    assert default_dem_name() == "merged.tif"
    monkeypatch.delenv("FANINSAR_DEM_NAME")
    assert default_dem_name() == "dem.tif"


def test_flatten_delegates_to_shared_lookup(tmp_path: Path) -> None:
    """flatten.copernicus_glo30_dem probes through the shared lookup."""
    from faninsar.processing.interferometry.flatten import copernicus_glo30_dem

    nested = tmp_path / "N38_E100" / "Copernicus_DSM_COG_10_N38_00_E100_00_DEM.tif"
    _write_tile(nested, latitude=38, longitude=100)
    dem = copernicus_glo30_dem(38.5, 100.5, base_path=tmp_path, device="cpu")
    assert dem.path == nested
    with pytest.raises(FileNotFoundError):
        copernicus_glo30_dem(40.5, 100.5, base_path=tmp_path, device="cpu")
