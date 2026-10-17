"""Tests for the on-demand ``Stack.geocode`` entry point.

The entry point composes the shared geocoding engine with the Stack grid,
radar context, and DEM. These tests use synthetic right-looking geometry and
pin the public contract: kind inference from the asset vocabulary, the
geographic no-op warning, kernel rejection, and the unified ``overwrite``
persistence semantics (PROPOSAL-0033).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.core.pair import Pairs
from faninsar.network.products import AssetKind
from faninsar.processing.geocoding import RadarDomainProduct
from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT
from faninsar.processing.geocoding.geocoder import Geocoder
from faninsar.processing.geometry import ConstantDEM, RadarGeometryModel
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.geometry.prepare_production import run_rdr2geo
from faninsar.processing.mosaicking.grid import GeoGridSpec
from faninsar.stack import Stack, StackConfig
from faninsar.stack.catalog import SceneCatalog

if TYPE_CHECKING:
    from pathlib import Path

RADAR_SHAPE = (64, 128)
UTM = "EPSG:32631"


def _toy_geometry() -> RadarGeometryModel:
    """Return a synthetic right-looking geometry with a straight orbit."""
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
        shape=RADAR_SHAPE,
        starting_slant_range_m=800_000.0,
        range_spacing_m=2.3,
        sensing_start=epoch,
        azimuth_time_interval_s=0.002,
        wavelength_m=0.0555,
        look_direction="right",
    )
    return RadarGeometryModel.from_radar_grid(rgrid, orbit)


def _scene_grid(model: RadarGeometryModel) -> GeoGridSpec:
    """Return a UTM grid covering the toy burst footprint plus margin."""
    from pyproj import Transformer

    az = np.array([0.0, 0.0, 63.0, 63.0, 0.0, 63.0, 31.5, 31.5], dtype=np.float64)
    rg = np.array([0.0, 127.0, 0.0, 127.0, 63.5, 63.5, 0.0, 127.0], dtype=np.float64)
    geo = run_rdr2geo(model, az, rg, ConstantDEM(0.0), device="cpu")
    ok = np.isfinite(geo.latitude_deg) & np.isfinite(geo.longitude_deg)
    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    ux, uy = to_utm.transform(geo.longitude_deg[ok], geo.latitude_deg[ok])
    margin = 600.0
    west, east = float(np.min(ux)) - margin, float(np.max(ux)) + margin
    south, north = float(np.min(uy)) - margin, float(np.max(uy)) + margin
    return GeoGridSpec(
        crs=UTM,
        transform=(west, 10.0, 0.0, north, 0.0, -10.0),
        width=int(np.ceil((east - west) / 10.0)),
        height=int(np.ceil((north - south) / 10.0)),
        resolution_m=(10.0, 10.0),
    )


@pytest.fixture
def stack(tmp_path: Path) -> Stack:
    """Return a bare Stack with synthetic radar geometry in its context."""
    paths = []
    for day in ("20240101", "20240113"):
        safe = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000.SAFE"
        safe.mkdir()
        paths.append(safe)
    model = _toy_geometry()
    instance = Stack(
        catalog=SceneCatalog.from_paths(paths),
        config=StackConfig(
            work_dir=tmp_path / "work",
            activation_mode="reference",
            grid=_scene_grid(model),
            dem=ConstantDEM(0.0),
            device="cpu",
        ),
        pairs=Pairs.from_names(["20240101_20240113"]),
        misreg_pairs=Pairs.from_names(["20240101_20240113"]),
        reference="20240101",
    )
    instance._radar_projection_context = {
        "geometry": model,
        "full_radar_shape": RADAR_SHAPE,
    }
    return instance


def _forbid_lut_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if any geocoding path rebuilds the lookup table."""
    from faninsar.processing.geocoding import geo_lut

    def _forbidden(**_kwargs: object) -> Geo2RdrLUT:
        raise AssertionError("geocode rebuilt the lookup table unexpectedly")

    monkeypatch.setattr(geo_lut, "build_geo2rdr_lut", _forbidden)


def test_geocode_complex_product_roundtrip_and_reuse(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complex product geocodes once, commits COGs, and reuses on re-call."""
    values = np.full(RADAR_SHAPE, 0.25 + 0.75j, dtype=np.complex64)
    product = RadarDomainProduct(
        name="pair_ifg",
        values=values,
        full_radar_shape=RADAR_SHAPE,
        asset_kind=AssetKind.COMPLEX_INTERFEROGRAM,
    )
    raster = stack.geocode(product)
    assert raster.values.dtype == np.complex64
    assert raster.valid.dtype == np.bool_
    assert raster.valid.any()
    assert raster.grid.crs == UTM
    target = stack.config.work_dir / "geocode"
    assert (target / "pair_ifg.tif").is_file()
    assert (target / "pair_ifg_valid.tif").is_file()
    assert (target / "pair_ifg.json").is_file()
    assert (stack.config.work_dir / "cache" / "geo-lut").is_dir()

    # The entry point must match a directly constructed engine exactly (the
    # LUT is served from the content cache, so this build is cheap).
    reference = Geocoder(
        stack._radar_projection_context["geometry"],
        stack.config.grid,
        dem=ConstantDEM(0.0),
        full_radar_shape=RADAR_SHAPE,
        device="cpu",
    )
    expected = reference.apply(values, kind="complex")
    np.testing.assert_array_equal(raster.valid, expected.valid)
    np.testing.assert_allclose(
        np.asarray(raster.values)[raster.valid],
        np.asarray(expected.values)[expected.valid],
        rtol=0.0,
        atol=1e-6,
    )

    _forbid_lut_rebuild(monkeypatch)
    reused = stack.geocode(product)
    np.testing.assert_array_equal(np.asarray(reused.values), np.asarray(raster.values))
    np.testing.assert_array_equal(reused.valid, raster.valid)
    assert reused.grid == raster.grid


def test_geocode_phase_product_returns_wrapped_radians(stack: Stack) -> None:
    """Wrapped phase infers the cyclic kernel and returns float32 radians."""
    wrapped = np.full(RADAR_SHAPE, 0.7, dtype=np.float32)
    product = RadarDomainProduct(
        name="wrapped",
        values=wrapped,
        full_radar_shape=RADAR_SHAPE,
        asset_kind=AssetKind.WRAPPED_PHASE,
    )
    raster = stack.geocode(product)
    assert raster.values.dtype == np.float32
    assert raster.valid.any()
    sampled = raster.values[raster.valid]
    assert np.all(np.abs(sampled) <= np.pi + 1e-6)
    np.testing.assert_allclose(sampled, 0.7, rtol=0.0, atol=1e-5)


def test_geocode_continuous_and_label_products(stack: Stack) -> None:
    """Smooth fields ride bilinear via UNWRAPPED_PHASE; labels need kind=."""
    field = np.full(RADAR_SHAPE, 3.5, dtype=np.float32)
    unwrapped = RadarDomainProduct(
        name="unw",
        values=field,
        full_radar_shape=RADAR_SHAPE,
        asset_kind=AssetKind.UNWRAPPED_PHASE,
    )
    raster = stack.geocode(unwrapped)
    assert raster.values.dtype == np.float32
    np.testing.assert_allclose(raster.values[raster.valid], 3.5, rtol=0.0, atol=1e-6)

    labelled = RadarDomainProduct(
        name="conncomp",
        values=np.full(RADAR_SHAPE, 3, dtype=np.uint8),
        full_radar_shape=RADAR_SHAPE,
    )
    with pytest.raises(ValueError, match="asset_kind"):
        stack.geocode(labelled)
    label_raster = stack.geocode(labelled, kind="label")
    assert label_raster.values.dtype == np.uint8
    np.testing.assert_array_equal(label_raster.values[label_raster.valid], 3)


def test_geocode_geographic_product_warns_and_returns_unchanged(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An already geographic product is a warned no-op."""
    _forbid_lut_rebuild(monkeypatch)
    product = RadarDomainProduct(
        name="already_geo",
        values=np.zeros(RADAR_SHAPE, dtype=np.float32),
        full_radar_shape=RADAR_SHAPE,
        crs="EPSG:4326",
    )
    with caplog.at_level("WARNING"):
        returned = stack.geocode(product)
    assert returned is product
    assert any("already georeferenced" in record.message for record in caplog.records)


def test_geocode_kind_lookup_failures(stack: Stack) -> None:
    """Unknown asset vocabularies and kernels fail with actionable errors."""
    values = np.zeros(RADAR_SHAPE, dtype=np.float32)
    with pytest.raises(ValueError, match="asset_kind"):
        stack.geocode(
            RadarDomainProduct(
                name="no_kind",
                values=values,
                full_radar_shape=RADAR_SHAPE,
                asset_kind=None,
            )
        )
    with pytest.raises(ValueError, match="asset_kind"):
        stack.geocode(
            RadarDomainProduct(
                name="bad_kind",
                values=values,
                full_radar_shape=RADAR_SHAPE,
                asset_kind="not_an_asset",
            )
        )
    product = RadarDomainProduct(
        name="explicit",
        values=values,
        full_radar_shape=RADAR_SHAPE,
        asset_kind=AssetKind.COHERENCE,
    )
    with pytest.raises(ValueError, match="kind must be one of"):
        stack.geocode(product, kind="bogus")


def test_geocode_overwrite_semantics(stack: Stack) -> None:
    """Matching identities reuse; changed or corrupt stores fail; True replaces."""
    product = RadarDomainProduct(
        name="unw",
        values=np.full(RADAR_SHAPE, 1.5, dtype=np.float32),
        full_radar_shape=RADAR_SHAPE,
        asset_kind=AssetKind.UNWRAPPED_PHASE,
    )
    stack.geocode(product)
    target = stack.config.work_dir / "geocode"

    # A changed source payload yields a different identity: refuse, then replace.
    changed = RadarDomainProduct(
        name="unw",
        values=np.full(RADAR_SHAPE, 2.5, dtype=np.float32),
        full_radar_shape=RADAR_SHAPE,
        asset_kind=AssetKind.UNWRAPPED_PHASE,
    )
    with pytest.raises(ValueError, match="different identity"):
        stack.geocode(changed)
    replaced = stack.geocode(changed, overwrite=True)
    np.testing.assert_allclose(
        np.asarray(replaced.values)[replaced.valid],
        2.5,
        rtol=0.0,
        atol=1e-6,
    )

    # A corrupt sidecar fails actionably without overwrite and heals with it.
    (target / "unw.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="identity sidecar"):
        stack.geocode(changed)
    stack.geocode(changed, overwrite=True)
    record = json.loads((target / "unw.json").read_text(encoding="utf-8"))
    assert record["identity"]

    # A missing values COG makes the store absent: it is rebuilt and commits.
    (target / "unw.tif").unlink()
    rebuilt = stack.geocode(changed)
    np.testing.assert_array_equal(
        np.asarray(rebuilt.values)[rebuilt.valid],
        np.asarray(replaced.values)[replaced.valid],
    )


def test_geocode_applies_origin_and_multilook_mapping(stack: Stack) -> None:
    """Crop origin and look factors fold through one to_layer_frame mapping.

    The oracle builds the designed layer-frame coordinates directly — full
    radar coordinates through
    ``RadarCoordinates.to_layer_frame(origin, looks)`` — instead of
    replicating the entry point's composition, so a chained
    ``(x - (L - 1) / 2) / L - origin`` drift of ``origin * (L - 1) / L``
    layer pixels cannot pass.
    """
    from faninsar.processing.geocoding.geo_resample import sample

    model = stack._radar_projection_context["geometry"]
    reference = Geocoder(
        model,
        stack.config.grid,
        dem=ConstantDEM(0.0),
        full_radar_shape=RADAR_SHAPE,
        device="cpu",
    )
    layer_shape = (
        (RADAR_SHAPE[0] - 2) // 2,
        (RADAR_SHAPE[1] - 4) // 2,
    )
    layer = np.arange(
        layer_shape[0] * layer_shape[1], dtype=np.float32
    ).reshape(layer_shape)
    product = RadarDomainProduct(
        name="ml",
        values=layer,
        full_radar_shape=RADAR_SHAPE,
        asset_kind=AssetKind.AMPLITUDE,
        multilook=(2, 2),
        origin=(2, 4),
    )
    raster = stack.geocode(product)

    # The destination grid is never thinned by the product multilook: the
    # raster covers the requested frame pixel for pixel.
    full_frame = reference.coordinates()
    assert raster.values.shape == full_frame.shape
    assert raster.grid.shape == stack.config.grid.shape

    layer_coords = full_frame.to_layer_frame(origin=(2, 4), looks=(2, 2))
    expected_values, expected_valid = sample(layer, layer_coords, kind="continuous")
    assert raster.values.shape == expected_values.shape
    np.testing.assert_array_equal(raster.valid, expected_valid)
    np.testing.assert_allclose(
        np.asarray(raster.values)[raster.valid],
        expected_values[expected_valid],
        rtol=0.0,
        atol=1e-5,
    )


def test_geocode_rejects_frame_inconsistent_product(stack: Stack) -> None:
    """A layer whose shape contradicts its crop/multilook frame is rejected."""
    from faninsar.processing.errors import InvalidProcessingStateError

    with pytest.raises(InvalidProcessingStateError, match="layer frame"):
        RadarDomainProduct(
            name="duck",
            values=np.zeros(RADAR_SHAPE, dtype=np.float32),
            full_radar_shape=RADAR_SHAPE,
            multilook=(2, 2),
            origin=(2, 4),
        )

    class _DuckProduct:
        """Duck-typed product bypassing the frozen carrier validation."""

        name = "duck"
        asset_kind = AssetKind.AMPLITUDE
        crs = None
        values = np.zeros(RADAR_SHAPE, dtype=np.float32)
        full_radar_shape = RADAR_SHAPE
        multilook = (2, 2)
        origin = (2, 4)

    with pytest.raises(ValueError, match="layer frame"):
        stack.geocode(_DuckProduct())
