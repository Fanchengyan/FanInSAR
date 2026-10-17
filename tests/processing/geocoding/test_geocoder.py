"""Unit tests for the thin Geocoder shell and GeocodedRaster value object."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT, build_geo2rdr_lut
from faninsar.processing.geocoding.geo_resample import sample
from faninsar.processing.geocoding.geocoder import GeocodedRaster, Geocoder
from faninsar.processing.geometry import RadarGeometryModel
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.mosaicking.grid import GeoGridSpec

if TYPE_CHECKING:
    from pathlib import Path


GRID_SHAPE = (4, 5)
RADAR_SHAPE = (64, 128)


def _toy_geometry(shape: tuple[int, int] = RADAR_SHAPE) -> RadarGeometryModel:
    epoch = datetime(2020, 1, 1, tzinfo=UTC)
    vectors = tuple(
        OrbitStateVector(
            time=epoch + timedelta(seconds=float(i)),
            position_m=(7000_000.0, 0.0, 0.0 + 100.0 * i),
            velocity_m_s=(0.0, 7500.0, 0.0),
        )
        for i in range(40)
    )
    orbit = OrbitMetadata(reference_frame="ECFT", source="test", vectors=vectors)
    rgrid = RadarGrid(
        shape=shape,
        starting_slant_range_m=800_000.0,
        range_spacing_m=2.3,
        sensing_start=epoch,
        azimuth_time_interval_s=0.002,
        wavelength_m=0.0555,
        look_direction="right",
    )
    return RadarGeometryModel.from_radar_grid(rgrid, orbit)


def _projected_grid(pixel_size_m: float = 200.0) -> GeoGridSpec:
    return GeoGridSpec(
        crs="EPSG:32631",
        transform=(500_000.0, pixel_size_m, 0.0, 1_000.0, 0.0, -pixel_size_m),
        width=GRID_SHAPE[1],
        height=GRID_SHAPE[0],
        resolution_m=(pixel_size_m, pixel_size_m),
    )


def _synthetic_lut() -> Geo2RdrLUT:
    """Return an all-valid LUT with in-window fractional radar coordinates."""
    azimuth, shift = np.meshgrid(
        np.linspace(3.0, 6.0, GRID_SHAPE[0]),
        np.linspace(5.0, 9.0, GRID_SHAPE[1]),
        indexing="ij",
    )
    return Geo2RdrLUT(
        az_full=azimuth,
        rg_full=shift,
        valid=np.ones(GRID_SHAPE, dtype=bool),
        full_radar_shape=RADAR_SHAPE,
        height_m=0.0,
    )


def _geocoder(**overrides: object) -> Geocoder:
    kwargs: dict[str, object] = {
        "dem": None,
        "full_radar_shape": RADAR_SHAPE,
        "device": "cpu",
        "lut": _synthetic_lut(),
    }
    kwargs.update(overrides)
    return Geocoder(_toy_geometry(), _projected_grid(), **kwargs)


def _complex_layer() -> np.ndarray:
    rows, columns = np.indices(RADAR_SHAPE)
    return np.exp(1j * (0.05 * rows + 0.02 * columns)).astype(np.complex64)


def _offset_field(shape: tuple[int, int], azimuth: float, shift: float) -> object:
    from faninsar.processing.coregistration.offsets import OffsetFieldResult

    return OffsetFieldResult(
        range_offset_px=np.full(shape, shift, dtype=np.float32),
        azimuth_offset_px=np.full(shape, azimuth, dtype=np.float32),
        coverage=np.ones(shape, dtype=bool),
        uncertainty_px=np.zeros(shape, dtype=np.float32),
    )


def test_geocoder_apply_matches_coordinates_plus_sample() -> None:
    """Apply is exactly coordinates()+sample() on the shared frame."""
    geocoder = _geocoder()
    layer = _complex_layer()
    raster = geocoder.apply(layer, kind="complex")
    expected, expected_valid = sample(
        layer,
        geocoder.coordinates(),
        kind="complex",
        device="cpu",
    )
    assert isinstance(raster, GeocodedRaster)
    np.testing.assert_array_equal(raster.values, expected)
    np.testing.assert_array_equal(raster.valid, expected_valid)
    assert raster.grid.shape == GRID_SHAPE
    assert raster.grid.crs == _projected_grid().crs


def test_geocoder_apply_chunked_out_and_on_chunk(tmp_path: Path) -> None:
    """Chunked application into caller storage matches the full frame."""
    geocoder = _geocoder()
    layer = _complex_layer()
    out = np.memmap(
        tmp_path / "values.complex64",
        mode="w+",
        dtype=np.complex64,
        shape=GRID_SHAPE,
    )
    chunks: list[slice] = []
    raster = geocoder.apply(
        layer,
        kind="complex",
        out=out,
        rows_per_chunk=2,
        on_chunk=chunks.append,
    )
    assert chunks == [slice(0, 2), slice(2, 4)]
    assert raster.values is out
    full = geocoder.apply(layer, kind="complex")
    np.testing.assert_array_equal(np.asarray(out), full.values)
    np.testing.assert_array_equal(raster.valid, full.valid)


def test_geocoder_correction_is_applied_per_chunk() -> None:
    """A per-chunk correction (offset composition) matches manual composition."""
    geocoder = _geocoder()
    layer = _complex_layer()
    offsets = _offset_field(RADAR_SHAPE, azimuth=0.125, shift=-0.25)
    raster = geocoder.apply(
        layer,
        kind="complex",
        correction=lambda coords: coords.minus_offsets(offsets, device="cpu"),
        rows_per_chunk=2,
    )
    coordinates = geocoder.coordinates().minus_offsets(offsets, device="cpu")
    expected, expected_valid = sample(
        layer,
        coordinates,
        kind="complex",
        device="cpu",
    )
    np.testing.assert_array_equal(raster.values, expected)
    np.testing.assert_array_equal(raster.valid, expected_valid)


def test_geocoder_window_coordinates_slice_the_lut() -> None:
    """Window coordinates are the exact LUT sub-block."""
    geocoder = _geocoder()
    lut = geocoder.lut
    coords = geocoder.coordinates(window=(1, 3, 2, 5))
    np.testing.assert_array_equal(coords.az, lut.az_full[1:3, 2:5])
    np.testing.assert_array_equal(coords.rg, lut.rg_full[1:3, 2:5])
    np.testing.assert_array_equal(coords.valid, lut.valid[1:3, 2:5])


def test_geocoder_looks_take_box_average_centres() -> None:
    """Look blocks sample at (index - (L - 1) / 2) / L, even factors included."""
    lut = _synthetic_lut()
    geocoder = _geocoder(lut=lut)
    coords = geocoder.coordinates(window=(0, 2, 0, 4), looks=(2, 2))
    # Look block (0, 0) averages rows {0, 1} x columns {0, 1}.
    expected_az = 0.25 * lut.az_full[0:2, 0:2].sum()
    expected_rg = 0.25 * lut.rg_full[0:2, 0:2].sum()
    assert coords.shape == (1, 2)
    assert float(coords.az[0, 0]) == pytest.approx((expected_az - 0.5) / 2.0)
    assert float(coords.rg[0, 0]) == pytest.approx((expected_rg - 0.5) / 2.0)
    # Second look column averages columns 2 and 3.
    expected_rg_1 = 0.25 * lut.rg_full[0:2, 2:4].sum()
    assert float(coords.rg[0, 1]) == pytest.approx((expected_rg_1 - 0.5) / 2.0)


def test_geocoder_applies_boolean_mask_at_application_time() -> None:
    """A boolean mask narrows the returned validity without touching the LUT."""
    geocoder = _geocoder()
    field = (
        np.linspace(0.0, 1.0, RADAR_SHAPE[0] * RADAR_SHAPE[1])
        .reshape(RADAR_SHAPE)
        .astype(np.float32)
    )
    mask = np.zeros(GRID_SHAPE, dtype=bool)
    mask[1:3, 1:4] = True
    raster = geocoder.apply(field, kind="continuous", mask=mask)
    expected, expected_valid = sample(
        field,
        geocoder.coordinates(),
        kind="continuous",
    )
    assert not raster.valid[~mask].any()
    np.testing.assert_array_equal(raster.valid, expected_valid & mask)
    np.testing.assert_array_equal(
        raster.values[mask & expected_valid],
        expected[mask & expected_valid],
    )


def test_geocoder_reuses_one_lut_across_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    """One engine builds exactly one LUT no matter how many layers it applies."""
    from faninsar.processing.geocoding import geo_lut

    builds: list[int] = []
    real_build = geo_lut.build_geo2rdr_lut

    def counting_build(**kwargs: object) -> Geo2RdrLUT:
        builds.append(1)
        return real_build(**kwargs)

    monkeypatch.setattr(geo_lut, "build_geo2rdr_lut", counting_build)
    geocoder = _geocoder(lut=None)
    first = geocoder.apply(_complex_layer(), kind="complex")
    second = geocoder.apply(_complex_layer(), kind="complex")
    assert len(builds) == 1
    np.testing.assert_array_equal(first.values, second.values)
    # A pre-built LUT bypasses the builder entirely.
    prebuilt = _synthetic_lut()
    prebuilt_geocoder = _geocoder(lut=prebuilt)
    coords = prebuilt_geocoder.coordinates()
    np.testing.assert_array_equal(coords.az, prebuilt.az_full)
    assert len(builds) == 1


def test_geocoder_forwards_device_to_lut_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured device reaches the geometric LUT solve unchanged."""
    from faninsar.processing.geocoding import geo_lut

    captured: dict[str, object] = {}

    def fake_build(**kwargs: object) -> Geo2RdrLUT:
        captured.update(kwargs)
        return _synthetic_lut()

    monkeypatch.setattr(geo_lut, "build_geo2rdr_lut", fake_build)
    geocoder = _geocoder(lut=None, device="cuda")
    assert geocoder.device == "cuda"
    coordinates = geocoder.coordinates()
    assert captured["device"] == "cuda"
    assert captured["geometry"] is geocoder._geometry
    np.testing.assert_array_equal(coordinates.az, _synthetic_lut().az_full)


def test_geocoder_forwards_device_to_resampling_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured device reaches the kind-driven resampling kernels."""
    from faninsar.processing.geocoding import geo_resample

    calls: list[dict[str, object]] = []
    real_sample = geo_resample.sample

    def fake_sample(
        layer: np.ndarray,
        coords: object,
        *,
        kind: str,
        device: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append({"kind": kind, "device": device})
        return real_sample(layer, coords, kind=kind, device="cpu")  # type: ignore[arg-type]

    monkeypatch.setattr(geo_resample, "sample", fake_sample)
    geocoder = _geocoder(device="cuda")
    raster = geocoder.apply(_complex_layer(), kind="complex")
    assert calls == [{"kind": "complex", "device": "cuda"}]
    assert isinstance(raster, GeocodedRaster)


def test_geocoder_rasterizes_polygon_mask_on_destination() -> None:
    """A shapely ROI is rasterized at application time and ANDed into validity."""
    from shapely.geometry import box

    from faninsar.processing.geocoding.geo_lut import grid_lonlat

    geocoder = _geocoder()
    field = (
        np.linspace(0.0, 1.0, RADAR_SHAPE[0] * RADAR_SHAPE[1])
        .reshape(RADAR_SHAPE)
        .astype(np.float32)
    )
    latitude, longitude = grid_lonlat(_projected_grid())
    roi = box(
        float(np.min(longitude)),
        float(np.min(latitude)),
        float(np.max(longitude)),
        float(np.max(latitude)),
    )
    covered = geocoder.apply(field, kind="continuous", mask=roi)
    uncovered = geocoder.apply(field, kind="continuous")
    np.testing.assert_array_equal(covered.valid, uncovered.valid)
    np.testing.assert_array_equal(covered.values, uncovered.values)


def test_geocoder_label_values_keep_layer_dtype() -> None:
    """Label kinds allocate the layer dtype and zero the invalid lanes."""
    geocoder = _geocoder()
    labels = np.zeros(RADAR_SHAPE, dtype=np.uint8)
    labels[2:8, 4:12] = 3
    raster = geocoder.apply(labels, kind="label")
    assert raster.values.dtype == np.uint8
    assert int(np.nanmax(raster.values[raster.valid])) == 3
    assert not raster.valid[~raster.valid].any()


def test_geocoder_rejects_misconfigured_out_and_caching(tmp_path: Path) -> None:
    """Malformed destinations and half-configured caching fail closed."""
    from faninsar.processing.errors import InvalidProcessingStateError

    geocoder = _geocoder()
    with pytest.raises(InvalidProcessingStateError):
        geocoder.apply(
            _complex_layer(),
            kind="complex",
            out=np.zeros((1, 1), dtype=np.complex64),
        )
    with pytest.raises(InvalidProcessingStateError):
        _geocoder(cache_dir=tmp_path / "cache")
    with pytest.raises(InvalidProcessingStateError):
        _geocoder(bounds=(0, 1))
    with pytest.raises(InvalidProcessingStateError):
        geocoder.coordinates(window=(0, 99, 0, 5))
