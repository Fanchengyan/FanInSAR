"""Focused manager tests for the PROPOSAL-0040 water-mask seam.

Covers TDD-plan items 12-17 against
``faninsar/processing/masking/mask_manager.py``:

12. Fetch atomicity: any tile failure fails the whole ``fetch_water``; a
    partial tile set is never mosaicked/vectorized (offline fake transport).
13. Water extraction + vectorization: GSW occurrence >= threshold 50 ->
    boolean -> polygonize + dissolve + simplify (30 m) + min-area (1 km2);
    pinned constants; OSM fails closed (unwired in v1).
14. Cache identity digest: sha256 over the canonical serialization;
    parameter or version change -> new identity; ETag normalization
    (quotes / weak validator stripped); missing headers -> explicitly
    unversioned; bounds normalization defeats float jitter; the padded
    fetch band folds into the identity.
15. Provider failures always raise ``MaskProviderUnavailableError``; there is
    no warning/skip continuation that could silently change scientific output.
16. ``resolve_auto_mask`` end-to-end on a synthetic DEM grid
    (transform/shape exact match, uint8 values {0, 1, 255}, water cells
    where the buffered polygon is) plus the rasterized-mask cache
    (identical inputs never re-rasterize, asserted via call counting).
17. GeoJSON atomic write: temp file + ``os.replace``, no ``.tmp``/``.part``
    leftovers.

Plus the ``get_mask_manager`` environment contract (cache dir required,
https-enforced mirror URL, runtime fail-closed userinfo rejection).

All tests are offline: the transport session is faked the same way
``tests/processing/geometry/test_dem_manager.py`` fakes it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
import shapely
from rasterio.transform import from_origin
from shapely.geometry import shape

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry.dem_transport import Tile
from faninsar.processing.masking import mask_manager as mm
from faninsar.processing.masking.mask_manager import (
    FetchResult,
    MaskManager,
    MaskProviderUnavailableError,
    WaterLayer,
    get_mask_manager,
    resolve_auto_mask,
)
from faninsar.processing.masking.mask_sources import GSW_BASE_URL

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Big lake inside the fixture tile: 0.8 deg x 0.4 deg (~3100 km2 at 38 N).
BIG_LAKE = (100.2, 38.2, 101.0, 38.6)
#: GSW NoData patch (occurrence 255) that must never count as water.
NODATA_PATCH = (100.1, 37.9, 100.15, 37.95)
#: ROI for the resolve_auto_mask tests (inside the single 100E/30N tile).
ROI = (100.1, 38.1, 100.5, 38.5)


# ---------------------------------------------------------------------------
# Offline transport fake (mirrors test_dem_manager.py conventions)
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
        self.head_headers: dict[str, dict[str, str]] = {}
        self.calls: list[tuple[str, str]] = []

    def register_ok(
        self,
        url: str,
        body: bytes,
        *,
        head_headers: dict[str, str] | None = None,
    ) -> None:
        """Serve 200 with ``body`` (and optional HEAD headers) for ``url``."""
        self.statuses[url] = 200
        self.bodies[url] = body
        if head_headers:
            self.head_headers[url] = head_headers

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
            head = {"Content-Length": str(len(body)), "Accept-Ranges": "bytes"}
            head.update(self.head_headers.get(url, {}))
            return FakeResponse(status=200, headers=head)
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


class NetworkBoom(HttpFake):
    """Fake that fails loudly on any scripted response (network touched)."""

    def _respond(self, method: str, url: str, headers: dict[str, str]) -> FakeResponse:
        del method, headers
        message = f"network touched: {url}"
        raise AssertionError(message)


# ---------------------------------------------------------------------------
# Offline fixtures
# ---------------------------------------------------------------------------


def _gsw_url(lon_origin: int, lat_origin: int) -> str:
    """Build the flat live-verified GSW occurrence URL (NE quadrant)."""
    return f"{GSW_BASE_URL}/occurrence_{lon_origin}E_{lat_origin}N.tif"


def _write_occurrence_tile(
    path: Path,
    *,
    water_boxes: Sequence[tuple[float, float, float, float]] = (),
    nodata_boxes: Sequence[tuple[float, float, float, float]] = (),
    px: float = 0.005,
    origin_lon: float = 100.0,
    top_lat: float = 39.0,
    rows: int = 300,
    cols: int = 300,
) -> bytes:
    """Write a >= 64 KiB GSW-like occurrence tile and return its bytes.

    Background occurrence is 10 (land), water boxes carry occurrence 90, and
    NoData patches carry 255 with the dataset NoData value set accordingly.
    """
    values = np.full((rows, cols), 10, dtype="uint8")
    for min_lon, min_lat, max_lon, max_lat in water_boxes:
        r0 = round((top_lat - max_lat) / px)
        r1 = round((top_lat - min_lat) / px)
        c0 = round((min_lon - origin_lon) / px)
        c1 = round((max_lon - origin_lon) / px)
        values[r0:r1, c0:c1] = 90
    for min_lon, min_lat, max_lon, max_lat in nodata_boxes:
        r0 = round((top_lat - max_lat) / px)
        r1 = round((top_lat - min_lat) / px)
        c0 = round((min_lon - origin_lon) / px)
        c1 = round((max_lon - origin_lon) / px)
        values[r0:r1, c0:c1] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(origin_lon, top_lat, px, px),
        "nodata": 255,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)
    return path.read_bytes()


def _tile_body(tmp_path: Path, name: str, **kwargs: object) -> bytes:
    """Build one fixture tile body (>= GSW_MIN_TILE_BYTES on disk)."""
    path = tmp_path / "fixtures" / name
    return _write_occurrence_tile(path, **kwargs)  # type: ignore[arg-type]


def _write_dem(path: Path, *, nodata_rows: int = 2) -> Path:
    """Write a small synthetic EPSG:4326 float32 DEM grid (30 x 30)."""
    rows, cols = 30, 30
    values = np.full((rows, cols), 50.0, dtype="float32")
    if nodata_rows:
        values[rows - nodata_rows :, :] = -9999.0
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(100.0, 38.9, 0.02, 0.02),
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)
    return path


def _layer_geometry(layer: WaterLayer) -> object:
    """Union the GeoJSON features of a water layer into one geometry."""
    features = json.loads(layer.path.read_text(encoding="utf-8"))["features"]
    assert features, "expected at least one GeoJSON feature"
    return shapely.union_all([shape(feature["geometry"]) for feature in features])


def _seed_single_tile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    head_headers: dict[str, str] | None = None,
) -> HttpFake:
    """Install a fake serving one OK GSW tile covering the test ROI."""
    fake = HttpFake().install(monkeypatch)
    body = _tile_body(tmp_path, "occurrence_100E_30N.tif", water_boxes=[BIG_LAKE])
    fake.register_ok(_gsw_url(100, 30), body, head_headers=head_headers)
    return fake


def _qualified_manager(cache_dir: Path, **kwargs: object) -> MaskManager:
    """Build a manager with an explicit, unambiguous raster category policy."""
    kwargs.setdefault("ocean_shore_keep_m", 0.0)
    kwargs.setdefault("inland_water_buffer_m", 0.0)
    return MaskManager(cache_dir, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 12. Fetch atomicity (set-level; partial sets never vectorized)
# ---------------------------------------------------------------------------


class TestFetchAtomicity:
    """Set-level fetch atomicity: partial sets never vectorize (item 12)."""

    def test_one_tile_failure_fails_whole_fetch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Any tile failure fails the whole fetch_water call (structured)."""
        fake = HttpFake().install(monkeypatch)
        body = _tile_body(tmp_path, "ok.tif", water_boxes=[BIG_LAKE])
        fake.register_ok(_gsw_url(100, 30), body)
        fake.register_missing(_gsw_url(110, 30))
        manager = MaskManager(tmp_path / "cache")

        with pytest.raises(MaskProviderUnavailableError) as excinfo:
            manager.fetch_water((100.2, 38.2, 111.8, 38.8))
        error = excinfo.value
        assert isinstance(error, InvalidProcessingStateError)
        assert error.product == "water"
        assert error.provider == "gsw"
        assert error.host == "storage.googleapis.com"
        assert error.failure_class == "coverage"
        assert error.attempts >= 1
        # Completed tiles stay cached (resumable), but nothing was vectorized.
        assert (tmp_path / "cache" / "water-gsw" / "occurrence_100E_30N.tif").is_file()
        assert not (tmp_path / "cache" / "water-gsw" / "vectors").exists()

    def test_partial_tile_set_is_never_vectorized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_water_layer raises and writes no vector cache on tile failure."""
        fake = HttpFake().install(monkeypatch)
        body = _tile_body(tmp_path, "ok.tif", water_boxes=[BIG_LAKE])
        fake.register_ok(_gsw_url(100, 30), body)
        fake.register_missing(_gsw_url(110, 30))
        manager = MaskManager(tmp_path / "cache")

        with pytest.raises(MaskProviderUnavailableError):
            manager.get_water_layer((100.2, 38.2, 111.8, 38.8))
        partition = tmp_path / "cache" / "water-gsw"
        assert not list(partition.glob("vectors/*"))
        assert not list(partition.rglob("*.geojson"))

    def test_successful_fetch_returns_tiles_and_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A complete fetch returns tile paths, band, and normalized version."""
        fake = _seed_single_tile(
            tmp_path, monkeypatch, head_headers={"ETag": '"abc123"'}
        )
        manager = MaskManager(tmp_path / "cache")

        result = manager.fetch_water((100.2, 38.2, 100.8, 38.8))
        assert isinstance(result, FetchResult)
        assert result.band == (100.0, 30.0, 110.0, 40.0)
        assert result.source_version == "abc123"
        assert result.tiles == (
            tmp_path / "cache" / "water-gsw" / "occurrence_100E_30N.tif",
        )
        assert all(isinstance(tile, Path) for tile in result.tiles)
        # Second fetch is a raw-tile cache hit: no GET requests, same paths.
        fake.calls.clear()
        again = manager.fetch_water((100.2, 38.2, 100.8, 38.8))
        assert again.tiles == result.tiles
        assert [url for method, url in fake.calls if method == "GET"] == []

    def test_fetch_hit_returns_tile_records(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cached tiles resolve to Tile-shaped cache targets under the partition."""
        _seed_single_tile(tmp_path, monkeypatch)
        manager = MaskManager(tmp_path / "cache")
        manager.fetch_water((100.2, 38.2, 100.8, 38.8))
        tiles = manager.source_entry.plan((100.2, 38.2, 100.8, 38.8)).tiles
        assert len(tiles) == 1
        assert isinstance(tiles[0], Tile)
        assert str(tiles[0].cache_path).startswith("water-gsw/")

    def test_terminal_403_maps_to_forbidden(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Terminal 403 maps to the forbidden failure class with one attempt."""
        fake = HttpFake().install(monkeypatch)
        fake.register_error(_gsw_url(100, 30), 403)
        manager = MaskManager(tmp_path / "cache")
        with pytest.raises(MaskProviderUnavailableError) as excinfo:
            manager.fetch_water((100.2, 38.2, 100.8, 38.8))
        assert excinfo.value.failure_class == "forbidden"
        assert excinfo.value.attempts == 1


# ---------------------------------------------------------------------------
# 13. Extraction + vectorization (threshold 50, simplify 30 m, min-area 1 km2)
# ---------------------------------------------------------------------------


class TestWaterExtractionAndVectorize:
    """Extraction and category handling fail closed where required."""

    def test_raster_water_category_is_ambiguous_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raster tile cannot be treated as inland water implicitly."""
        _seed_single_tile(tmp_path, monkeypatch)
        manager = MaskManager(tmp_path / "cache")
        with pytest.raises(InvalidProcessingStateError, match="category"):
            manager.get_water_layer((100.2, 38.2, 100.8, 38.8))

    def test_qualified_raster_category_can_be_materialized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller may explicitly make equal category policies applicable."""
        fake = _seed_single_tile(tmp_path, monkeypatch)
        manager = MaskManager(
            tmp_path / "cache", ocean_shore_keep_m=0.0, inland_water_buffer_m=0.0
        )
        layer = manager.get_water_layer((100.2, 38.2, 100.8, 38.8))
        assert layer.feature_count == 1
        assert layer.categories == ("inland",)
        fake.calls.clear()
        assert manager.get_water_layer((100.2, 38.2, 100.8, 38.8)).from_cache

    def test_pinned_defaults_and_resolved_polarity(self, tmp_path: Path) -> None:
        """Pinned proposal constants and source-default polarity resolution."""
        manager = MaskManager(tmp_path)
        assert manager.simplify_tolerance_m == 30.0
        assert manager.min_area_km2 == 1.0
        assert manager.invert is False
        assert manager.on_failure == "error"
        assert manager.threshold is None
        assert manager.effective_threshold == 50.0
        assert manager.effective_excluded_values == frozenset()

        worldcover = MaskManager(tmp_path, source="water:worldcover")
        assert worldcover.effective_threshold is None
        assert worldcover.effective_excluded_values == frozenset({80})

        custom = MaskManager(
            tmp_path,
            threshold=60,
            excluded_values=frozenset({1, 2}),
            invert=True,
        )
        assert custom.effective_threshold == 60.0
        assert custom.effective_excluded_values == frozenset({1, 2})
        assert custom.invert is True
        assert custom.partition_name == "water-gsw"
        assert custom.partition_dir == tmp_path / "water-gsw"

    def test_invalid_on_failure_rejected(self, tmp_path: Path) -> None:
        """Unknown on_failure policies fail closed at construction."""
        with pytest.raises(ValueError, match="on_failure"):
            MaskManager(tmp_path, on_failure="explode")  # type: ignore[arg-type]

    @pytest.mark.parametrize("policy", ["warning", "skip"])
    def test_continuation_policies_are_removed(
        self, tmp_path: Path, policy: str
    ) -> None:
        """Provider failures cannot be downgraded to warning/skip."""
        with pytest.raises(ValueError, match="on_failure"):
            MaskManager(tmp_path, on_failure=policy)  # type: ignore[arg-type]

    def test_osm_overpass_selection_fails_closed(self, tmp_path: Path) -> None:
        """The unwired OSM provider is rejected at selection time (v1)."""
        with pytest.raises(ValueError, match="wired"):
            MaskManager(tmp_path, source="water:osm-overpass")


# ---------------------------------------------------------------------------
# 13b. Manager-owned buffer (buffer ownership moved to the water pipeline)
# ---------------------------------------------------------------------------


class TestManagerBufferField:
    """``MaskManager.buffer_km`` owns the land buffer (default/validation)."""

    def test_default_and_validation(self, tmp_path: Path) -> None:
        """Default 1.0; negative and NaN rejected fail-closed; 0 admitted."""
        manager = MaskManager(tmp_path)
        assert manager.buffer_km == 1.0
        with pytest.raises(ValueError, match="buffer_km"):
            MaskManager(tmp_path, buffer_km=-0.5)
        with pytest.raises(ValueError, match="buffer_km"):
            MaskManager(tmp_path, buffer_km=float("nan"))
        zero = MaskManager(tmp_path, buffer_km=0.0)
        assert zero.buffer_km == 0.0

    def test_resolve_auto_mask_uses_the_manager_buffer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The padded band, buffer, and provenance tag derive from buffer_km."""
        _seed_single_tile(tmp_path, monkeypatch, head_headers={"ETag": '"lake-v2"'})
        dem_path = _write_dem(tmp_path / "dem.tif")
        manager = _qualified_manager(tmp_path / "cache", buffer_km=2.0)

        out = manager.resolve_auto_mask(
            ROI, dem_path=dem_path, output_dir=tmp_path / "run"
        )
        assert out is not None
        with rasterio.open(out) as dataset:
            assert dataset.tags()["mask_buffer_km"] == "2.0"


# ---------------------------------------------------------------------------
# 14. Cache identity digest
# ---------------------------------------------------------------------------


class TestCacheIdentity:
    """Vector-cache identity digest behavior (item 14)."""

    def test_parameter_change_produces_new_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A threshold change re-extracts under a new identity."""
        _seed_single_tile(tmp_path, monkeypatch)
        manager = _qualified_manager(tmp_path / "cache")
        first = manager.get_water_layer((100.2, 38.2, 100.8, 38.8))

        stricter = _qualified_manager(tmp_path / "cache", threshold=60)
        second = stricter.get_water_layer((100.2, 38.2, 100.8, 38.8))
        assert second.identity != first.identity
        assert second.from_cache is False
        # The original identity's cache entry is untouched and still hits.
        assert manager.get_water_layer((100.2, 38.2, 100.8, 38.8)).from_cache is True

    def test_etag_normalization_quoted_vs_weak(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Quoted and weak-quoted ETags normalize to the same identity."""
        url = _gsw_url(100, 30)
        _seed_single_tile(tmp_path, monkeypatch, head_headers={"ETag": '"abc123"'})
        manager = _qualified_manager(tmp_path / "cache")
        quoted = manager.get_water_layer((100.2, 38.2, 100.8, 38.8))
        assert quoted.source_version == "abc123"

        # Re-install a fresh fake that serves the weak-quoted validator.
        weak_fake = HttpFake()
        weak_fake.install(monkeypatch)
        weak_fake.register_ok(
            url,
            _tile_body(tmp_path, "weak.tif", water_boxes=[BIG_LAKE]),
            head_headers={"ETag": 'W/"abc123"'},
        )
        weak = manager.get_water_layer((100.2, 38.2, 100.8, 38.8))
        assert weak.source_version == "abc123"
        assert weak.identity == quoted.identity
        assert weak.from_cache is True

    def test_last_modified_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without ETag the Last-Modified header becomes the version."""
        _seed_single_tile(
            tmp_path,
            monkeypatch,
            head_headers={"Last-Modified": "Tue, 15 Nov 2099 12:45:26 GMT"},
        )
        manager = _qualified_manager(tmp_path / "cache")
        layer = manager.get_water_layer((100.2, 38.2, 100.8, 38.8))
        assert layer.source_version == "Tue, 15 Nov 2099 12:45:26 GMT"

    def test_missing_headers_explicitly_unversioned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No ETag/Last-Modified records the explicit unversioned state."""
        _seed_single_tile(tmp_path, monkeypatch)
        manager = _qualified_manager(tmp_path / "cache")
        layer = manager.get_water_layer((100.2, 38.2, 100.8, 38.8))
        assert layer.source_version == "unversioned"
        assert manager.get_water_layer((100.2, 38.2, 100.8, 38.8)).from_cache is True

    def test_bounds_normalization_defeats_float_jitter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same ROI at different float precision reuses the same identity."""
        _seed_single_tile(tmp_path, monkeypatch)
        manager = _qualified_manager(tmp_path / "cache")
        first = manager.get_water_layer((100.2, 38.2, 100.8, 38.8))
        jittered = manager.get_water_layer(
            (100.200000001, 38.200000001, 100.800000001, 38.800000001)
        )
        assert jittered.identity == first.identity
        assert jittered.from_cache is True

    def test_padded_band_folds_into_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Padding that reaches the neighbor tile changes the identity."""
        fake = HttpFake().install(monkeypatch)
        body = _tile_body(tmp_path, "tile.tif", water_boxes=[BIG_LAKE])
        fake.register_ok(_gsw_url(100, 30), body)
        fake.register_ok(_gsw_url(90, 30), body)
        manager = _qualified_manager(tmp_path / "cache")

        roi = manager.get_water_layer((100.0, 38.2, 100.8, 38.8))
        assert roi.band == (100.0, 30.0, 110.0, 40.0)
        padded = manager.get_water_layer((99.99, 38.2, 100.8, 38.8))
        assert padded.band[0] == pytest.approx(90.0)
        assert padded.identity != roi.identity
        # The neighbor tile [90, 100) was fetched for the padded band
        # (round-5 blocker regression at the manager level).
        assert (tmp_path / "cache" / "water-gsw" / "occurrence_90E_30N.tif").is_file()


# ---------------------------------------------------------------------------
# 15. resolve_auto_mask end-to-end + rasterized-mask cache
# ---------------------------------------------------------------------------


class TestResolveAutoMask:
    """resolve_auto_mask end-to-end and its raster cache (item 16)."""

    def test_end_to_end_rasterized_mask(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Buffered vector rasterizes exactly onto the DEM grid (uint8 0/1/255)."""
        _seed_single_tile(tmp_path, monkeypatch, head_headers={"ETag": '"lake-v1"'})
        dem_path = _write_dem(tmp_path / "dem.tif")
        manager = _qualified_manager(tmp_path / "cache")

        out = manager.resolve_auto_mask(
            ROI, dem_path=dem_path, output_dir=tmp_path / "run"
        )
        assert out == tmp_path / "run" / "mask" / "water_mask.tif"
        with rasterio.open(dem_path) as dem_ds:
            dem_transform = dem_ds.transform
            dem_shape = (dem_ds.height, dem_ds.width)
        with rasterio.open(out) as dataset:
            assert dataset.transform == dem_transform
            assert (dataset.height, dataset.width) == dem_shape
            assert dataset.dtypes[0] == "uint8"
            band = dataset.read(1)
            tags = dataset.tags()
        assert set(np.unique(band)).issubset({0, 1, 255})
        # Water is represented as excluded labels and remains distinct from
        # invalid DEM coverage; exact footprint is owned by the source grid.
        assert np.count_nonzero(band == 1) > 0
        assert band[0, 0] == 0
        # DEM NoData rows resolve to invalid (255), even over the lake.
        assert np.all(band[28:30, :] == 255)
        # Provenance tags are stamped on the product.
        assert tags["mask_product"] == "water"
        assert tags["mask_provider"] == "gsw"
        assert tags["source_version"] == "lake-v1"
        assert tags["mask_buffer_km"] == "1.0"
        assert tags["mask_identity"]

    def test_rasterized_mask_cache_hit_does_not_rerasterize(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Identical inputs reuse the cached raster; a buffer change re-runs."""
        _seed_single_tile(tmp_path, monkeypatch)
        dem_path = _write_dem(tmp_path / "dem.tif")
        manager = _qualified_manager(tmp_path / "cache")

        calls = {"n": 0}
        real = mm.rasterize_to_grid

        def counting(*args: object, **kwargs: object) -> object:
            calls["n"] += 1
            return real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(mm, "rasterize_to_grid", counting)

        first = manager.resolve_auto_mask(
            ROI, dem_path=dem_path, output_dir=tmp_path / "run1"
        )
        second = manager.resolve_auto_mask(
            ROI, dem_path=dem_path, output_dir=tmp_path / "run2"
        )
        assert calls["n"] == 1
        assert first is not None
        assert second is not None
        with rasterio.open(first) as a, rasterio.open(second) as b:
            assert np.array_equal(a.read(1), b.read(1))

        # A different buffer is a different cache key: it re-rasterizes once.
        wider = _qualified_manager(tmp_path / "cache", buffer_km=2.0)
        wider.resolve_auto_mask(ROI, dem_path=dem_path, output_dir=tmp_path / "run3")
        assert calls["n"] == 2

    @pytest.mark.parametrize(
        "bad_bounds",
        [
            (178.0, 8.0, 179.9, 10.0),  # padded band reaches the seam
            (175.0, 8.0, 185.0, 10.0),  # unwrapped / out-of-range longitude
        ],
    )
    def test_seam_guard_fails_closed_before_any_fetch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_bounds: tuple
    ) -> None:
        """The antimeridian guard rejects ROIs before any network request."""
        fake = NetworkBoom()
        fake.install(monkeypatch)
        dem_path = _write_dem(tmp_path / "dem.tif")
        manager = _qualified_manager(tmp_path / "cache")
        with pytest.raises(InvalidProcessingStateError, match=r"seam|guard"):
            manager.resolve_auto_mask(
                bad_bounds,
                dem_path=dem_path,
                output_dir=tmp_path / "run",
            )
        assert fake.calls == []

    def test_module_level_buffer_km_defers_to_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Module-level ``buffer_km=None`` reads ``FANINSAR_MASK_BUFFER_KM``."""
        _seed_single_tile(tmp_path, monkeypatch, head_headers={"ETag": '"lake-v3"'})
        dem_path = _write_dem(tmp_path / "dem.tif")
        monkeypatch.setenv("FANINSAR_MASK_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setenv("FANINSAR_MASK_BUFFER_KM", "2.0")

        out = resolve_auto_mask(
            ROI,
            dem_path=dem_path,
            output_dir=tmp_path / "run",
            ocean_shore_keep_m=0.0,
            inland_water_buffer_m=0.0,
        )

        assert out is not None
        with rasterio.open(out) as dataset:
            assert dataset.tags()["mask_buffer_km"] == "2.0"


# ---------------------------------------------------------------------------
# 17. GeoJSON atomic write + provenance sidecar
# ---------------------------------------------------------------------------


class TestGeojsonAtomicWrite:
    """Atomic GeoJSON write + provenance sidecar (item 17)."""

    def test_atomic_write_with_provenance_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """layer.geojson + provenance.json are atomic; no tmp/part leftovers."""
        fake = _seed_single_tile(tmp_path, monkeypatch, head_headers={"ETag": '"v9"'})
        manager = _qualified_manager(tmp_path / "cache")
        layer = manager.get_water_layer((100.2, 38.2, 100.8, 38.8))

        vector_dir = layer.path.parent
        assert layer.path.name == "layer.geojson"
        assert vector_dir.name == layer.identity
        assert (vector_dir / "provenance.json").is_file()
        leftovers = [
            path
            for path in (tmp_path / "cache").rglob("*")
            if path.name.endswith(".tmp") or path.name.endswith(".part")
        ]
        assert leftovers == []

        provenance = json.loads((vector_dir / "provenance.json").read_text())
        for key in (
            "mask_product",
            "mask_provider",
            "mask_retrieved",
            "threshold",
            "excluded_values",
            "invert",
            "source_version",
        ):
            assert key in provenance
        assert provenance["mask_product"] == "water"
        assert provenance["mask_provider"] == "gsw"
        assert provenance["threshold"] == 50.0
        assert provenance["excluded_values"] == []
        assert provenance["invert"] is False
        assert provenance["source_version"] == "v9"
        del fake


# ---------------------------------------------------------------------------
# get_mask_manager environment contract
# ---------------------------------------------------------------------------


class TestGetMaskManagerEnvironment:
    """get_mask_manager environment contract (cache, source, URL)."""

    def test_requires_cache_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing FANINSAR_MASK_CACHE_DIR fails closed."""
        monkeypatch.delenv("FANINSAR_MASK_CACHE_DIR", raising=False)
        with pytest.raises(
            InvalidProcessingStateError, match="FANINSAR_MASK_CACHE_DIR"
        ):
            get_mask_manager()

    def test_reads_cache_and_source_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env selection grammar resolves; an explicit kwarg beats the env."""
        monkeypatch.setenv("FANINSAR_MASK_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.delenv("FANINSAR_MASK_SOURCE", raising=False)
        default = get_mask_manager()
        assert default.product == "water"
        assert default.provider == "gsw"

        monkeypatch.setenv("FANINSAR_MASK_SOURCE", "water:worldcover")
        assert get_mask_manager().provider == "worldcover"
        assert get_mask_manager(source="gsw").provider == "gsw"

    def test_buffer_km_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``FANINSAR_MASK_BUFFER_KM`` feeds buffer_km; an explicit kwarg wins."""
        monkeypatch.setenv("FANINSAR_MASK_CACHE_DIR", str(tmp_path))
        monkeypatch.delenv("FANINSAR_MASK_BUFFER_KM", raising=False)
        assert get_mask_manager().buffer_km == 1.0

        monkeypatch.setenv("FANINSAR_MASK_BUFFER_KM", "2.5")
        assert get_mask_manager().buffer_km == 2.5
        assert get_mask_manager(buffer_km=0.5).buffer_km == 0.5

    def test_buffer_km_env_must_be_a_non_negative_number(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Garbage and negative buffer overrides fail closed."""
        monkeypatch.setenv("FANINSAR_MASK_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("FANINSAR_MASK_BUFFER_KM", "not-a-number")
        with pytest.raises(ValueError, match="FANINSAR_MASK_BUFFER_KM"):
            get_mask_manager()
        monkeypatch.setenv("FANINSAR_MASK_BUFFER_KM", "-1.0")
        with pytest.raises(ValueError, match="buffer_km"):
            get_mask_manager()

    def test_source_url_must_be_https(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Plain-http mirror URLs are rejected fail-closed."""
        monkeypatch.setenv("FANINSAR_MASK_CACHE_DIR", "/tmp/mask-cache")
        monkeypatch.setenv("FANINSAR_MASK_SOURCE_URL", "http://mirror.test/gsw")
        with pytest.raises(InvalidProcessingStateError, match="https"):
            get_mask_manager()

    def test_source_url_rejects_userinfo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """user:pass@host mirror URLs are rejected at runtime (fail closed)."""
        monkeypatch.setenv("FANINSAR_MASK_CACHE_DIR", "/tmp/mask-cache")
        monkeypatch.setenv(
            "FANINSAR_MASK_SOURCE_URL", "https://user:pass@mirror.test/gsw"
        )
        with pytest.raises(InvalidProcessingStateError, match="credential"):
            get_mask_manager()

    def test_source_url_override_applied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A clean https mirror URL overrides the source base."""
        monkeypatch.setenv("FANINSAR_MASK_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("FANINSAR_MASK_SOURCE_URL", "https://mirror.test/gsw")
        assert get_mask_manager().source_entry.base_url == "https://mirror.test/gsw"
