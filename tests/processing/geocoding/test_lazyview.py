"""Tests for the lazy windowed radar geolocation backend.

The fixtures use real Zarr v3 stores read through rasterio/GDAL. The identity
case has an exact analytic value field; the cropped multilook case pins the
``first_center - step / 2 + 0.5`` bilinear geolocation crop and compares the
warp with an independent ``rasterio.warp.reproject`` reference. The HTTP case
proves that open reads no value chunks and that a small window stays bounded.
"""

from __future__ import annotations

import gc
import http.server
import math
import os
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

import numpy as np
import pytest
import rasterio
import zarr
from rasterio.crs import CRS
from rasterio.warp import reproject

from faninsar.processing.geocoding.lazyview import (
    LazyGeoViewError,
    RadarGeolocationSource,
    _build_geoposition,
    open_lazy_radar_view,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

LON0 = 10.0
LAT0 = 20.0
DELTA = 0.001


def _write_zarr(
    path: Path,
    values: np.ndarray,
    *,
    chunks: tuple[int, int] = (8, 8),
) -> Path:
    """Write one real Zarr v3 array and return its path."""
    path.mkdir(parents=True, exist_ok=True)
    array = zarr.create_array(
        store=str(path),
        shape=values.shape,
        chunks=chunks,
        dtype=values.dtype,
        compressors=None,
        zarr_format=3,
    )
    array[:] = values
    return path


def _uri(path: Path) -> str:
    """Return a native GDAL Zarr source filename."""
    return f'ZARR:"{path}"'


def _identity_arrays(
    height: int = 32,
    width: int = 40,
    *,
    dtype: type[np.floating] = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a linear value field and exact integer-centre geolocation."""
    rows = np.arange(height, dtype=np.float64)[:, None]
    cols = np.arange(width, dtype=np.float64)[None, :]
    values = (1000.0 * rows + cols).astype(dtype)
    longitude = np.broadcast_to(LON0 + cols * DELTA, (height, width)).copy()
    latitude = np.broadcast_to(LAT0 - rows * DELTA, (height, width)).copy()
    return values, longitude, latitude


def _identity_source(
    root: Path,
    *,
    height: int = 32,
    width: int = 40,
    values_dtype: type[np.floating] = np.float32,
    fingerprint: str = "",
) -> RadarGeolocationSource:
    """Write an identity product and return its source description."""
    values, longitude, latitude = _identity_arrays(height, width, dtype=values_dtype)
    values_path = _write_zarr(root / "values.zarr", values)
    longitude_path = _write_zarr(root / "longitude.zarr", longitude)
    latitude_path = _write_zarr(root / "latitude.zarr", latitude)
    return RadarGeolocationSource(
        values_uri=_uri(values_path),
        longitude_uri=_uri(longitude_path),
        latitude_uri=_uri(latitude_path),
        shape=(height, width),
        geolocation_shape=(height, width),
        data_type=np.dtype(values_dtype).name,
        source_fingerprint=fingerprint,
    )


def _crop_fixture(root: Path) -> tuple[RadarGeolocationSource, np.ndarray]:
    """Write a 64x64 full map with an 8x10, even-look product grid."""
    height, width = 8, 10
    first_row, first_col = 6.5, 7.5
    step_row, step_col = 4.0, 2.0
    row_centres = first_row + step_row * np.arange(height)
    col_centres = first_col + step_col * np.arange(width)
    product = (1000.0 * row_centres[:, None] + col_centres[None, :]).astype(np.float32)
    rows = np.arange(64, dtype=np.float64)[:, None]
    cols = np.arange(64, dtype=np.float64)[None, :]
    longitude = np.broadcast_to(LON0 + cols * DELTA, (64, 64)).copy()
    latitude = np.broadcast_to(LAT0 - rows * DELTA, (64, 64)).copy()
    values_path = _write_zarr(root / "product.zarr", product, chunks=(2, 5))
    longitude_path = _write_zarr(root / "full_longitude.zarr", longitude)
    latitude_path = _write_zarr(root / "full_latitude.zarr", latitude)
    source = RadarGeolocationSource(
        values_uri=_uri(values_path),
        longitude_uri=_uri(longitude_path),
        latitude_uri=_uri(latitude_path),
        shape=(height, width),
        geolocation_shape=(64, 64),
        data_type="float32",
        first_center=(first_row, first_col),
        step=(step_row, step_col),
        kind="unwrapped_phase",
    )
    return source, product


def _transform_bounds(view: object) -> tuple[float, float, float, float]:
    """Return ``(left, bottom, right, top)`` from a view transform."""
    transform = view.transform  # type: ignore[attr-defined]
    left = transform[2]
    top = transform[5]
    right = left + transform[0] * view.width  # type: ignore[attr-defined]
    bottom = top + transform[4] * view.height  # type: ignore[attr-defined]
    return left, bottom, right, top


def _reference_warp(
    view: object,
    product: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    *,
    step: tuple[float, float],
) -> np.ndarray:
    """Reproject a product independently with a half-pixel corner convention."""
    shifted_longitude = longitude - (step[1] / 2.0) * DELTA
    shifted_latitude = latitude + (step[0] / 2.0) * DELTA
    reference = np.full(
        (view.height, view.width),  # type: ignore[attr-defined]
        np.nan,
        dtype=np.float64,
    )
    reproject(
        product.astype(np.float64),
        reference,
        src_geoloc_array=np.stack([shifted_longitude, shifted_latitude]),
        src_crs=CRS.from_epsg(4326),
        dst_transform=rasterio.Affine(*view.transform),  # type: ignore[attr-defined]
        dst_crs=CRS.from_epsg(4326),
        resampling=rasterio.enums.Resampling.bilinear,
        dst_nodata=np.nan,
    )
    return reference


class _WhitelistHandler(http.server.BaseHTTPRequestHandler):
    """Serve only a fixed file whitelist with byte ranges."""

    root: Path
    allowed: set[str]
    log: list[tuple[str, str]]

    def do_HEAD(self) -> None:  # noqa: N802
        """Serve a HEAD request."""
        self._serve(head=True)

    def do_GET(self) -> None:  # noqa: N802
        """Serve a GET request."""
        self._serve(head=False)

    def _serve(self, *, head: bool) -> None:
        """Serve one whitelisted file, honoring one byte range."""
        relative = unquote(urlsplit(self.path).path).lstrip("/")
        self.log.append((self.command, self.path))
        if relative not in self.allowed:
            self.send_error(404, "not whitelisted")
            return
        payload = (self.root / relative).read_bytes()
        start, end = 0, len(payload) - 1
        requested = self.headers.get("Range")
        if requested and requested.startswith("bytes="):
            first, _, second = requested[6:].split(",")[0].partition("-")
            if first:
                start = int(first)
            if second:
                end = int(second)
            start = max(0, min(start, len(payload) - 1))
            end = max(start, min(end, len(payload) - 1))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
        else:
            self.send_response(200)
        body = payload[start : end + 1]
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        """Suppress server logging."""


@contextmanager
def _serve(root: Path) -> Iterator[tuple[str, list[tuple[str, str]]]]:
    """Serve a Zarr fixture over loopback HTTP with request logging."""
    allowed = {
        str(path.relative_to(root))
        for store in ("data.zarr", "longitude.zarr", "latitude.zarr")
        for path in (root / store).rglob("*")
        if path.is_file()
    }
    log: list[tuple[str, str]] = []
    handler = type(
        "BoundHandler",
        (_WhitelistHandler,),
        {"root": root, "allowed": allowed, "log": log},
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", log
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _chunk_requests(log: list[tuple[str, str]], store: str) -> set[str]:
    """Return unique chunk paths requested for one Zarr store."""
    return {path for _, path in log if f"/{store}/c/" in path}


def test_identity_linear_window_matches_analytic_reference(
    tmp_path: Path,
) -> None:
    """A float32 identity view preserves the exact linear value field."""
    source = _identity_source(tmp_path, fingerprint="identity-fingerprint")
    view = open_lazy_radar_view(source, block_size=8)
    try:
        assert view.shape == (32, 40)
        assert (view.width, view.height) == (40, 32)
        assert len(view.transform) == 6
        assert view.crs == "EPSG:4326"
        assert view.data_type == "float32"
        assert math.isnan(view.fill_value)
        assert view.chunk_shape == (8, 8)
        assert view.source_fingerprint == "identity-fingerprint"

        actual = view.read_window((0, 32, 0, 40))
        expected = 1000.0 * np.arange(32)[:, None] + np.arange(40)[None, :]
        assert actual.dtype == np.float32
        assert np.allclose(actual, expected, atol=1e-4, rtol=0.0)

        window = view.read_window((10, 14, 20, 24))
        assert window.shape == (4, 4)
        assert np.allclose(window, expected[10:14, 20:24], atol=1e-4)
    finally:
        view.close()


def test_float64_data_type_is_preserved(tmp_path: Path) -> None:
    """A float64 product keeps float64 output and metadata."""
    source = _identity_source(tmp_path, values_dtype=np.float64)
    view = open_lazy_radar_view(source, block_size=8)
    try:
        assert view.data_type == "float64"
        actual = view.read_window((0, 8, 0, 8))
        assert actual.dtype == np.float64
        assert np.allclose(
            actual,
            1000.0 * np.arange(8)[:, None] + np.arange(8)[None, :],
            atol=1e-10,
        )
    finally:
        view.close()


def test_crop_geolocation_vrt_samples_exact_float64_centres(
    tmp_path: Path,
) -> None:
    """The bilinear crop lands on ``first_center + i * step`` exactly."""
    source, _ = _crop_fixture(tmp_path)
    owners: list[object] = []
    try:
        geoposition = _build_geoposition(
            height=source.shape[0],
            width=source.shape[1],
            geo_height=source.geolocation_shape[0],
            geo_width=source.geolocation_shape[1],
            first_row=source.first_center[0],
            first_col=source.first_center[1],
            step_row=source.step[0],
            step_col=source.step[1],
            longitude_uri=source.longitude_uri,
            latitude_uri=source.latitude_uri,
            owners=owners,
        )
        with rasterio.open(geoposition.longitude_dataset) as dataset:
            longitude = dataset.read(1)
        with rasterio.open(geoposition.latitude_dataset) as dataset:
            latitude = dataset.read(1)
    finally:
        for owner in reversed(owners):
            owner.close()  # type: ignore[attr-defined]

    row_centres = 6.5 + 4.0 * np.arange(8)
    col_centres = 7.5 + 2.0 * np.arange(10)
    expected_longitude = LON0 + col_centres[None, :] * DELTA
    expected_latitude = LAT0 - row_centres[:, None] * DELTA
    assert longitude.dtype == np.float64
    assert latitude.dtype == np.float64
    assert float(longitude[0, 0]) == pytest.approx(10.0075, abs=1e-12)
    assert float(longitude[-1, -1]) == pytest.approx(10.0255, abs=1e-12)
    assert float(latitude[0, 0]) == pytest.approx(19.9935, abs=1e-12)
    assert float(latitude[-1, -1]) == pytest.approx(19.9655, abs=1e-12)
    assert np.max(np.abs(longitude - expected_longitude)) < 1e-12
    assert np.max(np.abs(latitude - expected_latitude)) < 1e-12


def test_crop_footprint_is_product_scoped_and_values_match_reference(
    tmp_path: Path,
) -> None:
    """An even-look crop is not stretched to the full geolocation frame."""
    source, product = _crop_fixture(tmp_path)
    view = open_lazy_radar_view(source, block_size=4)
    try:
        left, bottom, right, top = _transform_bounds(view)
        target_x = view.transform[0]
        target_y = abs(view.transform[4])
        assert abs(left - 10.0065) <= target_x
        assert abs(bottom - 19.9635) <= target_y
        assert abs(right - 10.0265) <= target_x
        assert abs(top - 19.9955) <= target_y
        assert right - left < 0.04
        assert top - bottom < 0.04
        assert view.width < 64
        assert view.height < 64

        actual = view.read_window((0, view.height, 0, view.width))
        row_centres = 6.5 + 4.0 * np.arange(8)
        col_centres = 7.5 + 2.0 * np.arange(10)
        reference = _reference_warp(
            view,
            product,
            np.broadcast_to(LON0 + col_centres[None, :] * DELTA, (8, 10)),
            np.broadcast_to(LAT0 - row_centres[:, None] * DELTA, (8, 10)),
            step=(4.0, 2.0),
        )
        finite = np.isfinite(actual) & np.isfinite(reference)
        assert finite.any()
        assert np.max(np.abs(actual[finite] - reference[finite])) < 0.1
        assert np.isnan(actual).sum() == 0
    finally:
        view.close()


def test_validity_mask_masks_zero_and_keeps_valid_zero(
    tmp_path: Path,
) -> None:
    """Masked pixels become NaN while a valid coherence zero stays zero."""
    values, longitude, latitude = _identity_arrays(32, 40)
    values[5:7, 5:8] = 0.0
    valid = np.ones((32, 40), dtype=np.uint8)
    valid[10:14, 12:16] = 0
    values_path = _write_zarr(tmp_path / "values.zarr", values)
    longitude_path = _write_zarr(tmp_path / "longitude.zarr", longitude)
    latitude_path = _write_zarr(tmp_path / "latitude.zarr", latitude)
    valid_path = _write_zarr(tmp_path / "valid.zarr", valid)
    source = RadarGeolocationSource(
        values_uri=_uri(values_path),
        longitude_uri=_uri(longitude_path),
        latitude_uri=_uri(latitude_path),
        shape=(32, 40),
        geolocation_shape=(32, 40),
        data_type="float32",
        kind="coherence",
        valid_uri=_uri(valid_path),
    )
    view = open_lazy_radar_view(source, block_size=8)
    try:
        actual = view.read_window((0, 32, 0, 40))
        valid_zero = actual[5:7, 5:8]
        assert np.isfinite(valid_zero).all()
        assert np.allclose(valid_zero, 0.0, atol=1e-6)
        assert np.isnan(actual[11:13, 13:15]).all()
        assert np.isnan(actual[10:14, 12:16]).sum() > 0
    finally:
        view.close()


def test_http_initialization_reads_no_values_and_window_is_bounded(
    tmp_path: Path,
) -> None:
    """Open reads only geolocation arrays; one window reads few value chunks."""
    height = width = 64
    values, longitude, latitude = _identity_arrays(height, width)
    _write_zarr(tmp_path / "data.zarr", values, chunks=(8, 8))
    _write_zarr(tmp_path / "longitude.zarr", longitude, chunks=(8, 8))
    _write_zarr(tmp_path / "latitude.zarr", latitude, chunks=(8, 8))

    with _serve(tmp_path) as (base_url, log):
        source = RadarGeolocationSource(
            values_uri=f'ZARR:"/vsicurl/{base_url}/data.zarr"',
            longitude_uri=f'ZARR:"/vsicurl/{base_url}/longitude.zarr"',
            latitude_uri=f'ZARR:"/vsicurl/{base_url}/latitude.zarr"',
            shape=(height, width),
            geolocation_shape=(height, width),
            data_type="float32",
        )
        before_open = len(log)
        view = open_lazy_radar_view(source, block_size=8)
        try:
            initialization = log[before_open:]
            assert _chunk_requests(initialization, "data.zarr") == set()
            assert len(_chunk_requests(initialization, "longitude.zarr")) == 64
            assert len(_chunk_requests(initialization, "latitude.zarr")) == 64

            before_read = len(log)
            actual = view.read_window((24, 28, 28, 32))
            read_log = log[before_read:]
            assert len(_chunk_requests(read_log, "data.zarr")) < 16
            assert _chunk_requests(read_log, "longitude.zarr") == set()
            assert _chunk_requests(read_log, "latitude.zarr") == set()

            transform = view.transform
            rows = transform[5] + (np.arange(24, 28) + 0.5) * transform[4]
            cols = transform[2] + (np.arange(28, 32) + 0.5) * transform[0]
            expected = (
                1000.0 * ((LAT0 - rows) / DELTA)[:, None]
                + ((cols - LON0) / DELTA)[None, :]
            )
            assert np.allclose(actual, expected, atol=0.1, rtol=0.0)
        finally:
            view.close()


def test_geolocation_cap_fails_before_opening(tmp_path: Path) -> None:
    """The pixel cap is enforced before any GDAL source is opened."""
    source = RadarGeolocationSource(
        values_uri="",
        longitude_uri="",
        latitude_uri="",
        shape=(4, 4),
        geolocation_shape=(64, 64),
        data_type="float32",
    )
    with pytest.raises(LazyGeoViewError) as exc_info:
        open_lazy_radar_view(source, max_geolocation_pixels=100)
    assert exc_info.value.reason == "geolocation_too_large"
    assert "4096" in exc_info.value.detail


@pytest.mark.parametrize(
    ("kind", "data_type"),
    [
        ("label", "float32"),
        ("slc", "float32"),
        ("complex", "float32"),
        ("wrapped_phase", "float32"),
        ("coherence", "int16"),
        ("coherence", "complex64"),
    ],
)
def test_unsupported_scientific_assets_rejected(
    kind: str,
    data_type: str,
) -> None:
    """Label, complex, integer, and wrapped-phase assets are rejected early."""
    source = RadarGeolocationSource(
        values_uri="",
        longitude_uri="",
        latitude_uri="",
        shape=(4, 4),
        geolocation_shape=(4, 4),
        data_type=data_type,
        kind=kind,
    )
    with pytest.raises(LazyGeoViewError) as exc_info:
        open_lazy_radar_view(source)
    assert exc_info.value.reason == "unsupported_scientific_asset"


def test_mapping_outside_full_geolocation_map_rejected() -> None:
    """Product centres may not point outside the full geolocation array."""
    source = RadarGeolocationSource(
        values_uri="",
        longitude_uri="",
        latitude_uri="",
        shape=(4, 4),
        geolocation_shape=(8, 8),
        data_type="float32",
        first_center=(6.0, 6.0),
        step=(1.0, 1.0),
    )
    with pytest.raises(LazyGeoViewError) as exc_info:
        open_lazy_radar_view(source)
    assert exc_info.value.reason == "invalid_geolocation"
    assert "outside" in exc_info.value.detail


@pytest.mark.parametrize(
    ("longitude_start", "latitude_start"),
    [
        (179.9, 20.0),
        (10.0, 89.5),
    ],
)
def test_dateline_and_polar_footprints_fail_closed(
    tmp_path: Path,
    longitude_start: float,
    latitude_start: float,
) -> None:
    """Unsafe dateline or polar auto bounds are rejected, never faked."""
    height = width = 16
    rows = np.arange(height, dtype=np.float64)[:, None]
    cols = np.arange(width, dtype=np.float64)[None, :]
    values = np.zeros((height, width), dtype=np.float32)
    longitude = np.broadcast_to(longitude_start + cols * 0.02, (height, width)).copy()
    latitude = np.broadcast_to(latitude_start - rows * 0.001, (height, width)).copy()
    values_path = _write_zarr(tmp_path / "values.zarr", values)
    longitude_path = _write_zarr(tmp_path / "longitude.zarr", longitude)
    latitude_path = _write_zarr(tmp_path / "latitude.zarr", latitude)
    source = RadarGeolocationSource(
        values_uri=_uri(values_path),
        longitude_uri=_uri(longitude_path),
        latitude_uri=_uri(latitude_path),
        shape=(height, width),
        geolocation_shape=(height, width),
        data_type="float32",
    )
    with pytest.raises(LazyGeoViewError) as exc_info:
        open_lazy_radar_view(source)
    assert exc_info.value.reason == "antimeridian_or_polar"


def test_gdal_cache_max_is_configured_in_bytes_and_restored(
    tmp_path: Path,
) -> None:
    """The open view sets a 64 MiB GDAL cache and restores the caller value."""
    source = _identity_source(tmp_path)
    before = rasterio.env.get_gdal_config("GDAL_CACHEMAX")
    with open_lazy_radar_view(source, block_size=8) as view:
        assert view.shape == (32, 40)
        assert rasterio.env.get_gdal_config("GDAL_CACHEMAX") == 64 * 1024 * 1024
    assert rasterio.env.get_gdal_config("GDAL_CACHEMAX") == before


def test_close_is_idempotent_and_releases_owners(tmp_path: Path) -> None:
    """Close is safe to repeat, blocks reads, and does not leak descriptors."""
    descriptor_root = "/dev/fd"
    before = len(os.listdir(descriptor_root)) if os.path.isdir(descriptor_root) else 0
    source = _identity_source(tmp_path)
    view = open_lazy_radar_view(source, block_size=8)
    assert view.read_window((0, 1, 0, 1)).shape == (1, 1)
    view.close()
    view.close()
    with pytest.raises(LazyGeoViewError) as exc_info:
        view.read_window((0, 1, 0, 1))
    assert exc_info.value.reason == "view_closed"
    gc.collect()
    if os.path.isdir(descriptor_root):
        after = len(os.listdir(descriptor_root))
        assert after <= before + 2


def test_context_manager_closes_view(tmp_path: Path) -> None:
    """The context manager closes the view on exit."""
    source = _identity_source(tmp_path)
    with open_lazy_radar_view(source, block_size=8) as view:
        assert view.read_window((0, 2, 0, 2)).shape == (2, 2)
    with pytest.raises(LazyGeoViewError) as exc_info:
        view.read_window((0, 1, 0, 1))
    assert exc_info.value.reason == "view_closed"


@pytest.mark.parametrize(
    "window",
    [
        (0, 1, 0),
        (0, 0, 0, 1),
        (0, 33, 0, 1),
        (-1, 1, 0, 1),
        (0, 1, 0, 41),
    ],
)
def test_invalid_windows_rejected(
    tmp_path: Path,
    window: tuple[int, ...],
) -> None:
    """Malformed or out-of-bounds windows are rejected."""
    source = _identity_source(tmp_path)
    view = open_lazy_radar_view(source, block_size=8)
    try:
        with pytest.raises(LazyGeoViewError):
            view.read_window(window)  # type: ignore[arg-type]
    finally:
        view.close()
