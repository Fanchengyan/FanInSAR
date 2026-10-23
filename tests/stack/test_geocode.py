"""Tests for the on-demand ``Stack.geocode`` entry point.

The entry point composes the shared geocoding engine with the Stack grid,
radar context, and DEM. These tests use synthetic right-looking geometry and
pin the public contract: kind inference from the asset vocabulary, the
geographic no-op warning, kernel rejection, and the unified ``overwrite``
persistence semantics (PROPOSAL-0033).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.core.pair import Pairs
from faninsar.io.storage.scene_store import write_scene
from faninsar.network.products import AssetKind
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geocoding import RadarDomainProduct
from faninsar.processing.geocoding.geo_lut import Geo2RdrLUT
from faninsar.processing.geocoding.geocoder import Geocoder
from faninsar.processing.geometry import ConstantDEM, RadarGeometryModel
from faninsar.processing.geometry.coordinates import RadarGrid
from faninsar.processing.geometry.prepare_production import run_rdr2geo
from faninsar.processing.geometry.record import radar_geometry_record
from faninsar.processing.mosaicking.grid import GeoGridSpec
from faninsar.stack import Stack, StackConfig
from faninsar.stack.catalog import SceneCatalog

if TYPE_CHECKING:
    from faninsar.io.storage.ifg_store import InterferogramArtifactStore

RADAR_SHAPE = (64, 128)
UTM = "EPSG:32631"


def _toy_grid_and_orbit() -> tuple[RadarGrid, OrbitMetadata]:
    """Return a synthetic right-looking radar grid with a straight orbit."""
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
    return rgrid, orbit


def _toy_geometry() -> RadarGeometryModel:
    """Return the synthetic geometry the persisted Reference record encodes."""
    return RadarGeometryModel.from_radar_grid(*_toy_grid_and_orbit())


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
    """Return a Stack whose committed Reference scene carries the geometry."""
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
            multilook=(1, 1),
            dem=ConstantDEM(0.0),
            device="cpu",
        ),
        pairs=Pairs.from_names(["20240101_20240113"]),
        misreg_pairs=Pairs.from_names(["20240101_20240113"]),
        reference="20240101",
    )
    write_scene(
        instance.config.work_dir / "coreg" / instance.reference,
        date_id=instance.reference,
        reference_id=instance.reference,
        domain="radar",
        grid_shape=RADAR_SHAPE,
        units=[
            {
                "tag": "f0_IW1_b0",
                "payload": np.ones(RADAR_SHAPE, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 0,
            }
        ],
        properties={
            "faninsar:radar_geometry": radar_geometry_record(*_toy_grid_and_orbit())
        },
    )
    return instance


def _forbid_lut_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if any geocoding path rebuilds the lookup table."""
    from faninsar.processing.geocoding import geo_lut

    def _forbidden(**_kwargs: object) -> Geo2RdrLUT:
        raise AssertionError("geocode rebuilt the lookup table unexpectedly")

    monkeypatch.setattr(geo_lut, "build_geo2rdr_lut", _forbidden)


def _write_ifg(
    stack: Stack, *, value: float = 0.7, overwrite: bool = False
) -> InterferogramArtifactStore:
    """Commit one full-frame Pair whose phase and amplitude are known."""
    from faninsar.io.storage.ifg_store import write_ifg_artifact

    reference = stack.reference_scene
    assert reference is not None
    secondary = stack.scenes.get("20240113")
    if secondary is None:
        secondary = write_scene(
            stack.config.work_dir / "coreg" / "20240113",
            date_id="20240113",
            reference_id=stack.reference,
            domain="radar",
            grid_shape=RADAR_SHAPE,
            units=[
                {
                    "tag": "f0_IW1_b0",
                    "payload": np.ones(RADAR_SHAPE, dtype=np.complex64),
                    "row_origin": 0,
                    "col_origin": 0,
                }
            ],
            inputs={"reference": reference.write_id},
        )
    return write_ifg_artifact(
        stack.config.work_dir / "ifg" / "ml_1x1" / "20240101_20240113",
        pair=("20240101", "20240113"),
        looks=(1, 1),
        grid_identity=reference.grid_identity,
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={
            "primary": reference.write_id,
            "secondary": secondary.write_id,
        },
        complex_ifg=np.full(RADAR_SHAPE, np.exp(1j * value), dtype=np.complex64),
        coherence=np.full(RADAR_SHAPE, 0.5, dtype=np.float32),
        valid_mask=np.ones(RADAR_SHAPE, dtype=bool),
        replace_existing=overwrite,
    )


def test_pair_phase_attributes_retain_wrapped_and_unwrapped_semantics(
    stack: Stack,
) -> None:
    """Short phase attributes carry the type and source required for geocoding."""
    from faninsar.io.storage.ifg_store import write_unwrapped_artifact

    store = _write_ifg(stack)
    write_unwrapped_artifact(
        store.root,
        unwrapped_phase=np.full(RADAR_SHAPE, 10 * np.pi + 0.7, dtype=np.float32),
        connected_components=np.ones(RADAR_SHAPE, dtype=np.uint32),
        method="test",
        method_parameters={},
        ifg_manifest_digest=store.write_id,
    )
    ifg = stack.interferograms["20240101_20240113"]
    assert str(ifg.wrapped.asset_kind) == "wrapped_phase"
    assert str(ifg.unwrapped.asset_kind) == "unwrapped_phase"
    assert ifg.wrapped.path is None
    assert ifg.amplitude.path is None
    assert ifg.complex_ifg.path.is_dir()
    wrapped = stack.geocode(ifg.wrapped)
    unwrapped = stack.geocode(ifg.unwrapped)
    np.testing.assert_allclose(wrapped.values[wrapped.valid], 0.7, atol=1e-5)
    np.testing.assert_allclose(
        unwrapped.values[unwrapped.valid], 10 * np.pi + 0.7, atol=1e-5
    )
    assert (store.root / "geocoded" / "wrapped_phase" / "item.json").is_file()
    assert (store.unwrap_root / "geocoded" / "unwrapped_phase" / "item.json").is_file()


def test_flat_layer_surface_lists_names_and_keeps_virtual_paths_none(
    stack: Stack,
) -> None:
    """``ifg.layers`` lists accessible names and virtual handles expose no path."""
    from faninsar.io.storage.ifg_store import write_unwrapped_artifact

    store = _write_ifg(stack)
    ifg = stack.interferograms["20240101_20240113"]
    assert ifg.layers == ["amplitude", "coherence", "complex_ifg", "wrapped"]
    assert ifg.wrapped.path is None
    assert ifg.amplitude.path is None
    assert ifg.complex_ifg.path.is_dir()
    np.testing.assert_allclose(ifg.wrapped.values, 0.7, atol=1e-5)
    np.testing.assert_allclose(ifg.amplitude.values, 1.0, atol=1e-6)
    assert ifg.wrapped.geocode_kind == "phase"
    # Deriving values must never materialize a virtual payload on disk.
    assert not (store.root / "wrapped_phase.zarr").exists()
    assert not (store.root / "amplitude.zarr").exists()

    write_unwrapped_artifact(
        store.root,
        unwrapped_phase=np.full(RADAR_SHAPE, 0.7, dtype=np.float32),
        connected_components=np.ones(RADAR_SHAPE, dtype=np.uint32),
        method="test",
        method_parameters={},
        ifg_manifest_digest=store.write_id,
    )
    assert stack.interferograms["20240101_20240113"].layers == [
        "amplitude",
        "coherence",
        "complex_ifg",
        "connected_components",
        "unwrapped",
        "wrapped",
    ]
    assert ifg.unwrapped.path.is_dir()
    assert str(ifg.unwrapped.asset_kind) == "unwrapped_phase"
    assert ifg.connected_components.geocode_kind == "label"
    np.testing.assert_allclose(ifg.connected_components.values, 1, rtol=0.0, atol=1e-6)


def test_missing_optional_layers_raise_file_not_found(stack: Stack) -> None:
    """Unestimated optional layers name the missing result without staging runs."""
    from faninsar.io.storage.ifg_store import write_ifg_artifact
    from faninsar.io.storage.product import open_product

    reference = stack.reference_scene
    assert reference is not None
    secondary_root = stack.config.work_dir / "coreg" / "20240113"
    write_scene(
        secondary_root,
        date_id="20240113",
        reference_id=stack.reference,
        domain="radar",
        grid_shape=RADAR_SHAPE,
        units=[
            {
                "tag": "f0_IW1_b0",
                "payload": np.ones(RADAR_SHAPE, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 0,
            }
        ],
        inputs={"reference": reference.write_id},
    )
    secondary = open_product(secondary_root)
    write_ifg_artifact(
        stack.config.work_dir / "ifg" / "ml_1x1" / "20240101_20240113",
        pair=("20240101", "20240113"),
        looks=(1, 1),
        grid_identity=reference.grid_identity,
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={
            "primary": reference.write_id,
            "secondary": secondary.write_id,
        },
        complex_ifg=np.ones(RADAR_SHAPE, dtype=np.complex64),
        coherence=None,
        valid_mask=None,
    )
    ifg = stack.interferograms["20240101_20240113"]
    assert ifg.layers == ["amplitude", "complex_ifg", "wrapped"]
    with pytest.raises(FileNotFoundError, match="coherence"):
        ifg.coherence
    with pytest.raises(FileNotFoundError, match="unwrapped"):
        ifg.unwrapped
    with pytest.raises(FileNotFoundError, match="unwrapped result"):
        ifg.connected_components
    assert not stack.interferograms.exists("20240101_20240113", "coherence")
    # Item/path adapters must not invent a file for an unestimated optional
    # payload: no coherence href may appear without a persisted coherence.zarr.
    with pytest.raises(FileNotFoundError, match="coherence"):
        stack.interferograms.path("20240101_20240113", "coherence")
    item = stack.interferograms.item("20240101_20240113")
    assert set(item["assets"]) == {"complex_ifg"}
    assert all(Path(asset["href"]).exists() for asset in item["assets"].values())


def test_collection_items_and_paths_never_invent_virtual_files(stack: Stack) -> None:
    """Item assets enumerate real payloads only; short public names stay readable."""
    from faninsar.io.storage.ifg_store import write_unwrapped_artifact

    store = _write_ifg(stack)
    collection = stack.interferograms
    pair = "20240101_20240113"
    item = collection.item(pair)
    assert set(item["assets"]) == {"complex_ifg", "coherence"}
    for name in ("wrapped", "wrapped_phase", "amplitude"):
        with pytest.raises(FileNotFoundError, match="virtual"):
            collection.path(pair, name)
    np.testing.assert_allclose(
        collection.open(pair, "wrapped").values,
        collection.open(pair, "wrapped_phase").values,
    )
    write_unwrapped_artifact(
        store.root,
        unwrapped_phase=np.full(RADAR_SHAPE, 0.2, dtype=np.float32),
        connected_components=np.ones(RADAR_SHAPE, dtype=np.uint32),
        method="test",
        method_parameters={},
        ifg_manifest_digest=store.write_id,
    )
    assert set(collection.item(pair)["assets"]) == {
        "complex_ifg",
        "coherence",
        "unw_phase",
    }
    assert collection.exists(pair, "unwrapped")
    assert collection.path(pair, "unwrapped") == collection.path(pair, "unw_phase")
    np.testing.assert_allclose(
        collection.open_stack("unwrapped").values,
        collection.open_stack("unw_phase").values,
    )


def test_geocode_complex_product_roundtrip_and_reuse(
    stack: Stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complex product geocodes once, commits COGs, and reuses on re-call."""
    store = _write_ifg(stack)
    product = stack.interferograms["20240101_20240113"].complex_ifg
    values = product.read()
    raster = stack.geocode(product)
    assert raster.values.dtype == np.complex64
    assert raster.valid.dtype == np.bool_
    assert raster.valid.any()
    assert raster.grid.crs == UTM
    target = store.root / "geocoded" / "complex_ifg"
    assert (target / "values.tif").is_file()
    assert (target / "valid.tif").is_file()
    assert (target / "item.json").is_file()
    assert (stack.config.work_dir / "cache" / "geo-lut").is_dir()

    # The entry point must match a directly constructed engine exactly (the
    # LUT is served from the content cache, so this build is cheap).
    reference = Geocoder(
        _toy_geometry(),
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


def test_memory_geocode_path_requires_explicit_replacement(stack: Stack) -> None:
    """A destination never substitutes for the identity of an in-memory array."""
    from faninsar.io.storage.product import open_product

    values = np.full(RADAR_SHAPE, 1.5, dtype=np.float32)
    product = RadarDomainProduct(
        name="derived",
        values=values,
        full_radar_shape=RADAR_SHAPE,
        asset_kind=AssetKind.UNWRAPPED_PHASE,
    )
    target = stack.config.work_dir / "results" / "derived"
    first = stack.geocode(product, path=target)
    original_id = open_product(target).write_id
    values[:] = 9.0
    with pytest.raises(FileExistsError, match="overwrite=True"):
        stack.geocode(product, path=target)
    assert open_product(target).write_id == original_id
    replacement = stack.geocode(product, path=target, overwrite=True)
    assert open_product(target).write_id != original_id
    np.testing.assert_allclose(first.values[first.valid], 1.5, atol=1e-6)
    np.testing.assert_allclose(replacement.values[replacement.valid], 9.0, atol=1e-6)
    assert (target / "values.tif").is_file()
    assert (target / "valid.tif").is_file()
    assert open_product(target).inputs == {}


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
    """Smooth fields ride bilinear via UNWRAPPED_PHASE; labels use LABEL."""
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
        asset_kind=AssetKind.LABEL,
    )
    label_raster = stack.geocode(labelled)
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
    """Unknown asset vocabularies fail with actionable errors."""
    values = np.zeros(RADAR_SHAPE, dtype=np.float32)
    with pytest.raises(InvalidProcessingStateError, match="asset_kind"):
        RadarDomainProduct(
            name="no_kind",
            values=values,
            full_radar_shape=RADAR_SHAPE,
            asset_kind=None,
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


def test_geocode_rejects_removed_override_arguments(stack: Stack) -> None:
    """Kind and grid-construction overrides belong to the product and grid."""
    product = RadarDomainProduct(
        name="explicit",
        values=np.zeros(RADAR_SHAPE, dtype=np.float32),
        full_radar_shape=RADAR_SHAPE,
        asset_kind=AssetKind.COHERENCE,
    )
    with pytest.raises(TypeError):
        stack.geocode(product, kind="continuous")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        stack.geocode(product, bounds=(0.0, 0.0, 1.0, 1.0))  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        stack.geocode(product, resolution_m=10.0)  # type: ignore[call-arg]


def test_resolve_grid_applies_resolution_override(stack: Stack) -> None:
    """Automatic grid construction owns the spacing knob, not geocode."""
    from dataclasses import replace
    from types import SimpleNamespace

    auto = SimpleNamespace(config=replace(stack.config, grid="utm_ups", roi=None))
    native = Stack.resolve_grid(auto, roi=(10.0, 40.0, 11.0, 41.0))
    coarser = Stack.resolve_grid(
        auto, roi=(10.0, 40.0, 11.0, 41.0), resolution_m=20.0
    )
    assert native.crs == coarser.crs
    assert native.resolution_m == (30.0, 30.0)
    assert coarser.resolution_m == (20.0, 20.0)
    assert native.shape != coarser.shape


def test_geocode_overwrite_semantics(stack: Stack) -> None:
    """Source-attached children follow source replacement and explicit overwrite."""
    from faninsar.io.storage.product import open_product
    from faninsar.processing.errors import InvalidProcessingStateError

    store = _write_ifg(stack)
    product = stack.interferograms["20240101_20240113"].wrapped
    target = store.root / "geocoded" / "wrapped_phase"
    committed = stack.geocode(product)
    original = open_product(target)
    assert original.inputs["source"] == store.write_id
    reused = stack.geocode(product)
    assert open_product(target).write_id == original.write_id
    np.testing.assert_array_equal(committed.values, reused.values)

    fresh_store = _write_ifg(stack, value=1.2, overwrite=True)
    assert not target.exists()
    with pytest.raises(InvalidProcessingStateError, match="replaced source"):
        stack.geocode(product)
    fresh = stack.interferograms["20240101_20240113"].wrapped
    replaced = stack.geocode(fresh)
    np.testing.assert_allclose(replaced.values[replaced.valid], 1.2, atol=1e-5)
    assert open_product(target).inputs["source"] == fresh_store.write_id

    (target / "item.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(InvalidProcessingStateError, match="item.json"):
        stack.geocode(fresh)
    stack.geocode(fresh, overwrite=True)
    assert open_product(target).properties["faninsar:geocode"]


@pytest.mark.parametrize("crs", ["EPSG:4326", UTM])
def test_georeferenced_input_does_not_ignore_requested_output(
    stack: Stack, crs: str
) -> None:
    """Pass-through is allowed only without an output or regridding request."""
    product = RadarDomainProduct(
        name="map_phase",
        values=np.ones(RADAR_SHAPE, dtype=np.float32),
        full_radar_shape=RADAR_SHAPE,
        crs=crs,
    )
    assert stack.geocode(product) is product
    with pytest.raises(ValueError, match="already georeferenced"):
        stack.geocode(product, path=stack.config.work_dir / "map_copy")
    with pytest.raises(ValueError, match="already georeferenced"):
        stack.geocode(product, grid=stack.grid)


def test_layer_cache_is_lazy_and_tracks_dem_context(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed hit reads no input raster and a changed DEM prevents reuse."""
    from faninsar.io.storage.product import Product, open_product

    store = _write_ifg(stack)
    product = stack.interferograms["20240101_20240113"].wrapped
    raster = stack.geocode(product)
    target = store.root / "geocoded" / "wrapped_phase"
    original = open_product(target).write_id

    def forbid_payload(*_args, **_kwargs):
        raise AssertionError("cached layer read input-array payloads")

    monkeypatch.setattr(Product, "read", forbid_payload)
    cached = stack.geocode(stack.interferograms["20240101_20240113"].wrapped)
    np.testing.assert_array_equal(cached.values, raster.values)
    assert open_product(target).write_id == original
    stack.config.dem = ConstantDEM(20.0)
    with pytest.raises(ValueError, match="different"):
        stack.geocode(product)


def test_alternate_grid_and_layer_outputs_do_not_replace_each_other(
    stack: Stack,
) -> None:
    """Per-layer children are independent, and alternate grids are one-off."""
    from faninsar.io.storage.product import open_product

    store = _write_ifg(stack)
    ifg = stack.interferograms["20240101_20240113"]
    stack.geocode(ifg.wrapped)
    phase_target = store.root / "geocoded" / "wrapped_phase"
    original = open_product(phase_target).write_id
    stack.geocode(ifg.coherence)
    assert (store.root / "geocoded" / "coherence" / "item.json").is_file()
    alternate = GeoGridSpec(
        crs=stack.grid.crs,
        transform=stack.grid.transform,
        width=stack.grid.width - 1,
        height=stack.grid.height,
        resolution_m=stack.grid.resolution_m,
    )
    raster = stack.geocode(ifg.wrapped, grid=alternate)
    assert raster.grid == alternate
    assert open_product(phase_target).write_id == original
    custom = stack.config.work_dir / "results" / "phase_alternate"
    stack.geocode(ifg.wrapped, grid=alternate, path=custom)
    assert open_product(custom).inputs["source"] == store.write_id
    with pytest.raises(ValueError, match="source|ancestor"):
        stack.geocode(ifg.wrapped, path=store.root, overwrite=True)
    assert open_product(store.root).write_id == store.write_id


def test_unwrapped_support_and_labels_follow_the_source(stack: Stack) -> None:
    """Finite phase in excluded components must never become valid output."""
    from faninsar.io.storage.ifg_store import write_unwrapped_artifact

    store = _write_ifg(stack)
    write_unwrapped_artifact(
        store.root,
        unwrapped_phase=np.full(RADAR_SHAPE, 9.0, dtype=np.float32),
        connected_components=np.zeros(RADAR_SHAPE, dtype=np.uint32),
        method="test",
        method_parameters={},
        ifg_manifest_digest=store.write_id,
        extra_layers={"valid_mask": np.zeros(RADAR_SHAPE, dtype=bool)},
    )
    unwrapped = stack.interferograms["20240101_20240113"].unwrapped
    raster = stack.geocode(unwrapped)
    assert not raster.valid.any()
    assert np.isnan(raster.values).all()
    labels = stack.geocode(
        stack.interferograms["20240101_20240113"].connected_components
    )
    assert labels.values.dtype == np.uint32
    assert not labels.valid.any()


def test_layer_values_are_detached_from_committed_provenance(stack: Stack) -> None:
    """Changing a loaded array cannot silently change a source-bound request."""
    _write_ifg(stack)
    layer = stack.interferograms["20240101_20240113"].wrapped
    detached = layer.read()
    detached[:] = -1.0
    assert not layer.values.flags.writeable
    raster = stack.geocode(layer)
    np.testing.assert_allclose(raster.values[raster.valid], 0.7, atol=1e-5)


def test_layer_rejects_replacement_of_its_reference_frame(stack: Stack) -> None:
    """A same-shaped replacement Reference does not make an old layer current."""
    from faninsar.processing.errors import InvalidProcessingStateError

    _write_ifg(stack)
    layer = stack.interferograms["20240101_20240113"].wrapped
    reference = stack.reference_scene
    assert reference is not None
    write_scene(
        reference.root,
        date_id=stack.reference,
        reference_id=stack.reference,
        domain="radar",
        grid_shape=RADAR_SHAPE,
        units=[
            {
                "tag": "f0_IW1_b0",
                "payload": np.ones(RADAR_SHAPE, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 0,
            }
        ],
        properties={
            "faninsar:radar_geometry": radar_geometry_record(*_toy_grid_and_orbit())
        },
        overwrite=True,
    )
    with pytest.raises(InvalidProcessingStateError, match="replaced|Reference"):
        stack.geocode(layer)
    with pytest.raises(InvalidProcessingStateError, match="replaced|Reference"):
        stack.interferograms["20240101_20240113"]


def test_layer_rejects_a_different_stack_reference(
    stack: Stack, tmp_path: Path
) -> None:
    """Equal array shapes do not make two Reference products interchangeable."""
    from faninsar.processing.errors import InvalidProcessingStateError

    _write_ifg(stack)
    layer = stack.interferograms["20240101_20240113"].wrapped
    other = Stack(
        catalog=stack.catalog,
        config=StackConfig(
            work_dir=tmp_path / "other",
            activation_mode="reference",
            grid=stack.grid,
            dem=ConstantDEM(0.0),
            device="cpu",
        ),
        reference=stack.reference,
        pairs=stack.pairs,
        misreg_pairs=stack.misreg_pairs,
    )
    write_scene(
        other.config.work_dir / "coreg" / other.reference,
        date_id=other.reference,
        reference_id=other.reference,
        domain="radar",
        grid_shape=RADAR_SHAPE,
        units=[
            {
                "tag": "f0_IW1_b0",
                "payload": np.ones(RADAR_SHAPE, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 0,
            }
        ],
        properties={
            "faninsar:radar_geometry": radar_geometry_record(*_toy_grid_and_orbit())
        },
    )
    with pytest.raises(InvalidProcessingStateError, match="Reference"):
        other.geocode(layer)


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

    model = _toy_geometry()
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
    layer = np.arange(layer_shape[0] * layer_shape[1], dtype=np.float32).reshape(
        layer_shape
    )
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


def test_geocode_on_a_fresh_stack_reads_the_persisted_reference(
    stack: Stack,
) -> None:
    """A new Stack object geocodes from disk alone (no coregistration run).

    Regression: ``geocode`` used to require the in-memory projection context
    of a coregistration run in the same process.
    """
    fresh = Stack(
        catalog=stack.catalog,
        config=stack.config,
        pairs=stack.pairs,
        misreg_pairs=stack.misreg_pairs,
        reference=stack.reference,
    )
    product = RadarDomainProduct(
        name="fresh",
        values=np.ones(RADAR_SHAPE, dtype=np.float32),
        full_radar_shape=RADAR_SHAPE,
        asset_kind=AssetKind.AMPLITUDE,
    )
    raster = fresh.geocode(product)
    assert raster.valid.any()


def test_geocode_without_reference_scene_names_the_missing_stage(
    stack: Stack,
) -> None:
    """Without a committed Reference scene the error says what to run."""
    import shutil

    shutil.rmtree(stack.config.work_dir / "coreg")
    product = RadarDomainProduct(
        name="none",
        values=np.ones(RADAR_SHAPE, dtype=np.float32),
        full_radar_shape=RADAR_SHAPE,
        asset_kind=AssetKind.AMPLITUDE,
    )
    with pytest.raises(ValueError, match="coregister_scenes"):
        stack.geocode(product)


def test_geocode_rejects_a_shape_that_differs_from_the_reference(
    stack: Stack,
) -> None:
    """A product declaring another full radar shape fails closed."""
    shape = (RADAR_SHAPE[0] + 2, RADAR_SHAPE[1])
    product = RadarDomainProduct(
        name="bad",
        values=np.ones(shape, dtype=np.float32),
        full_radar_shape=shape,
        asset_kind=AssetKind.AMPLITUDE,
    )
    with pytest.raises(ValueError, match="Reference scene grid"):
        stack.geocode(product)


def _toy_grid(
    *,
    shape: tuple[int, int],
    starting_slant_range_m: float,
    range_spacing_m: float,
    azimuth_time_interval_s: float,
    sensing_start: datetime,
) -> RadarGrid:
    """Return one synthetic swath grid for the merged-geocode test."""
    return RadarGrid(
        shape=shape,
        starting_slant_range_m=starting_slant_range_m,
        range_spacing_m=range_spacing_m,
        sensing_start=sensing_start,
        azimuth_time_interval_s=azimuth_time_interval_s,
        wavelength_m=0.0555,
        look_direction="right",
    )


def test_merged_multi_swath_geocode_agrees_with_per_burst_within_one_pixel() -> None:
    """Merged-frame geocode matches per-burst geocode across all boundaries.

    The residual is measured in pixels on the geo2rdr LUTs themselves: the
    merged frame (first-swath timing and range, integer per-swath range
    offsets) versus each unit's own geometry, sampled within four pixels of
    every unit edge — that is, on both sides of every swath and burst
    boundary (PROPOSAL-0059 Watch 13).
    """
    from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
    from faninsar.processing.geocoding.geo_lut import build_geo2rdr_lut
    from faninsar.processing.mosaicking.grid import GeoGridSpec
    from pyproj import Transformer

    epoch = datetime(2020, 1, 1, tzinfo=UTC)
    # A straight, constant-velocity orbit: band footprints tile cleanly.
    vectors = tuple(
        OrbitStateVector(
            time=epoch + timedelta(seconds=float(i)),
            position_m=(7000_000.0, 0.0, 0.0),
            velocity_m_s=(0.0, 7500.0, 0.0),
        )
        for i in range(6)
    )
    orbit = OrbitMetadata(reference_frame="ECFT", source="test", vectors=vectors)
    lines, samples = 32, 64
    bursts = 2
    frame_rows = lines * bursts

    def grid_for(
        *,
        rows: int,
        cols: int,
        start: float,
        dr: float,
        dt: float,
        sensing: datetime,
    ) -> RadarGrid:
        return RadarGrid(
            shape=(rows, cols),
            starting_slant_range_m=start,
            range_spacing_m=dr,
            sensing_start=sensing,
            azimuth_time_interval_s=dt,
            wavelength_m=0.0555,
            look_direction="right",
        )

    # Real IW swaths differ in timing/range parameters at the 1e-4 level.
    swath_start = (800_000.0, 800_050.0)
    swath_dr = (2.3, 2.3005)
    swath_dt = (0.002, 0.0020002)
    range_offset = int(round((swath_start[1] - swath_start[0]) / swath_dr[0]))
    frame_cols = samples + range_offset
    merged_grid = grid_for(
        rows=frame_rows,
        cols=frame_cols,
        start=swath_start[0],
        dr=swath_dr[0],
        dt=swath_dt[0],
        sensing=epoch,
    )
    merged_model = RadarGeometryModel.from_radar_grid(merged_grid, orbit)
    # Destination grid: the merged footprint's corners plus margin.
    az = np.array([0.0, 0.0, frame_rows - 1.0, frame_rows - 1.0], dtype=np.float64)
    rg = np.array([0.0, frame_cols - 1.0, 0.0, frame_cols - 1.0], dtype=np.float64)
    geo = run_rdr2geo(merged_model, az, rg, ConstantDEM(0.0), device="cpu")
    ok = np.isfinite(geo.latitude_deg) & np.isfinite(geo.longitude_deg)
    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    ux, uy = to_utm.transform(geo.longitude_deg[ok], geo.latitude_deg[ok])
    margin = 400.0
    grid = GeoGridSpec(
        crs=UTM,
        transform=(
            float(np.min(ux)) - margin,
            10.0,
            0.0,
            float(np.max(uy)) + margin,
            0.0,
            -10.0,
        ),
        width=int(np.ceil((np.max(ux) - np.min(ux) + 2 * margin) / 10.0)),
        height=int(np.ceil((np.max(uy) - np.min(uy) + 2 * margin) / 10.0)),
        resolution_m=(10.0, 10.0),
    )

    def build(model: RadarGeometryModel, shape: tuple[int, int]) -> Geo2RdrLUT:
        return build_geo2rdr_lut(
            geometry=model,
            grid=grid,
            full_radar_shape=shape,
            height_m=0.0,
            device="cpu",
        )

    merged = build(merged_model, (frame_rows, frame_cols))
    worst = 0.0
    for swath, column_offset in enumerate((0, range_offset)):
        for burst in range(bursts):
            row_offset = burst * lines
            unit_grid = grid_for(
                rows=lines,
                cols=samples,
                start=swath_start[swath],
                dr=swath_dr[swath],
                dt=swath_dt[swath],
                sensing=epoch + timedelta(seconds=row_offset * swath_dt[0]),
            )
            unit = build(
                RadarGeometryModel.from_radar_grid(unit_grid, orbit),
                (lines, samples),
            )
            both = merged.valid & unit.valid
            # Edge bands of the unit in its own radar coordinates: four
            # pixels on each side of the burst row boundary and the swath
            # column boundary.
            near_edge = (
                (unit.az_full < 4)
                | (unit.az_full > lines - 4)
                | (unit.rg_full < 4)
                | (unit.rg_full > samples - 4)
            )
            mask = both & near_edge
            assert mask.any(), (swath, burst)
            row_residual = merged.az_full[mask] - (unit.az_full[mask] + row_offset)
            column_residual = merged.rg_full[mask] - (
                unit.rg_full[mask] + column_offset
            )
            residual = float(
                np.max(np.abs(np.concatenate([row_residual, column_residual])))
            )
            worst = max(worst, residual)
    print(f"merged-vs-per-burst max boundary residual: {worst:.4f} px")
    assert worst <= 1.0


def test_trimmed_orbit_geocode_matches_full_orbit(tmp_path: Path) -> None:
    """Geocoding with the persisted trimmed orbit equals the full orbit."""
    model = _toy_geometry()
    grid = _scene_grid(model)
    values = np.full(RADAR_SHAPE, 0.3 + 0.4j, dtype=np.complex64)
    from faninsar.processing.geometry.record import radar_geometry_from_record
    from faninsar.processing.geometry.record import radar_geometry_record

    radar_grid, _orbit = _toy_grid_and_orbit()
    epoch = radar_grid.sensing_start
    # A long acquisition makes the span trim actually drop vectors.
    from faninsar.core.orbit import OrbitStateVector

    long_vectors = tuple(
        OrbitStateVector(
            time=epoch + timedelta(seconds=float(i)),
            position_m=(7000_000.0, 0.0, 0.0 + 100.0 * i),
            velocity_m_s=(0.0, 7500.0, 0.0),
        )
        for i in range(300)
    )
    long_orbit = OrbitMetadata(
        reference_frame="ECFT", source="test", vectors=long_vectors
    )
    record = radar_geometry_record(radar_grid, long_orbit)
    trimmed_model = radar_geometry_from_record(record)

    full = Geocoder(
        model, grid, dem=ConstantDEM(0.0), full_radar_shape=RADAR_SHAPE, device="cpu"
    ).apply(values, kind="complex")
    trimmed = Geocoder(
        trimmed_model,
        grid,
        dem=ConstantDEM(0.0),
        full_radar_shape=RADAR_SHAPE,
        device="cpu",
    ).apply(values, kind="complex")

    np.testing.assert_array_equal(full.valid, trimmed.valid)
    np.testing.assert_array_equal(
        np.asarray(full.values)[full.valid], np.asarray(trimmed.values)[trimmed.valid]
    )
    # The persisted orbit is actually a trim, not the full vector set.
    assert len(record["orbit"]["vectors"]) < len(long_vectors)


def test_reopened_stack_geocode_shares_one_lut_entry_and_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reopened Stack geocodes identically and hits the same LUT entry."""
    from faninsar.stack.identity import save_stack_identity

    work = tmp_path / "work"
    paths = []
    for day in ("20240101", "20240113"):
        safe = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000.SAFE"
        safe.mkdir()
        paths.append(safe)
    model = _toy_geometry()
    stack = Stack(
        catalog=SceneCatalog.from_paths(paths),
        config=StackConfig(
            work_dir=work,
            activation_mode="reference",
            grid=_scene_grid(model),
            multilook=(1, 1),
            dem=ConstantDEM(0.0),
            device="cpu",
        ),
        pairs=Pairs.from_names(["20240101_20240113"]),
        misreg_pairs=Pairs.from_names(["20240101_20240113"]),
        reference="20240101",
    )
    write_scene(
        work / "coreg" / stack.reference,
        date_id=stack.reference,
        reference_id=stack.reference,
        domain="radar",
        grid_shape=RADAR_SHAPE,
        units=[
            {
                "tag": "f0_IW1_b0",
                "payload": np.ones(RADAR_SHAPE, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 0,
            }
        ],
        properties={
            "faninsar:radar_geometry": radar_geometry_record(*_toy_grid_and_orbit())
        },
    )
    looked_shape = (31, 62)
    product = RadarDomainProduct(
        name="looked",
        values=np.arange(np.prod(looked_shape), dtype=np.float32).reshape(looked_shape),
        full_radar_shape=RADAR_SHAPE,
        asset_kind=AssetKind.UNWRAPPED_PHASE,
        multilook=(2, 2),
        origin=(2, 4),
    )
    in_process = stack.geocode(product)

    save_stack_identity(stack)
    reopened = Stack.open(work, device="cpu")
    from faninsar.processing.geocoding import geo_lut

    def forbid_solve(*_args, **_kwargs):
        raise AssertionError(
            "reopened geocode solved geometry instead of reusing the LUT"
        )

    monkeypatch.setattr(geo_lut, "run_geo2rdr", forbid_solve)
    reopened_geocode = reopened.geocode(product)

    np.testing.assert_array_equal(
        np.asarray(in_process.values), np.asarray(reopened_geocode.values)
    )
    np.testing.assert_array_equal(in_process.valid, reopened_geocode.valid)
    namespaces = [
        path.name
        for path in (work / "cache" / "geo-lut").iterdir()
        if path.is_dir() and path.name != "storage"
    ]
    assert namespaces == ["stack_20240101"]


def test_bound_layer_geocode_reuses_the_stack_child(stack: Stack) -> None:
    """A bound handle geocodes without repeating product or kernel."""
    import dataclasses

    from faninsar.io.storage.product import open_product
    from faninsar.network.layers import ProductLayer

    store = _write_ifg(stack)
    ifg = stack.interferograms["20240101_20240113"]
    session_field = next(
        field for field in dataclasses.fields(ProductLayer) if field.name == "_session"
    )
    assert session_field.compare is False

    raster = ifg.wrapped.geocode()
    np.testing.assert_allclose(raster.values[raster.valid], 0.7, atol=1e-5)
    child = store.root / "geocoded" / "wrapped_phase"
    assert (child / "item.json").is_file()

    first_id = open_product(child).write_id
    reused = stack.geocode(ifg.wrapped)
    assert open_product(child).write_id == first_id
    np.testing.assert_allclose(reused.values[reused.valid], 0.7, atol=1e-5)


def test_bound_layer_geocode_routes_unwrap_layers_to_their_product(
    stack: Stack,
) -> None:
    """Unwrap children live with the unwrap product, exactly as before."""
    from faninsar.io.storage.ifg_store import write_unwrapped_artifact

    store = _write_ifg(stack)
    write_unwrapped_artifact(
        store.root,
        unwrapped_phase=np.full(RADAR_SHAPE, 1.25, dtype=np.float32),
        connected_components=np.ones(RADAR_SHAPE, dtype=np.uint32),
        method="test",
        method_parameters={},
        ifg_manifest_digest=store.write_id,
    )
    ifg = stack.interferograms["20240101_20240113"]
    raster = ifg.unwrapped.geocode()
    np.testing.assert_allclose(raster.values[raster.valid], 1.25, atol=1e-5)
    assert (
        store.unwrap_root / "geocoded" / "unwrapped_phase" / "item.json"
    ).is_file()

    labels = ifg.connected_components.geocode()
    assert labels.values.dtype == np.uint32
    assert labels.valid.any()


def test_unbound_layer_handle_fails_closed(stack: Stack) -> None:
    """A hand-built handle has no session and must not guess one."""
    from faninsar.io.storage.ifg_store import IFG_KIND
    from faninsar.io.storage.product import open_product
    from faninsar.network.layers import InterferogramLayers

    store = _write_ifg(stack)
    handle = InterferogramLayers(
        open_product(store.root, kind=IFG_KIND), store.shape, store.looks
    )
    with pytest.raises(InvalidProcessingStateError, match="not bound"):
        handle.wrapped.geocode()


def test_bound_layer_geocode_honours_grid_and_overwrite(stack: Stack) -> None:
    """Handle geocoding forwards grid and overwrite to the Stack verb."""
    from faninsar.io.storage.product import open_product

    store = _write_ifg(stack)
    ifg = stack.interferograms["20240101_20240113"]
    child = store.root / "geocoded" / "wrapped_phase"
    ifg.wrapped.geocode()
    first_id = open_product(child).write_id
    ifg.wrapped.geocode(overwrite=True)
    assert open_product(child).write_id != first_id

    alternate = GeoGridSpec(
        crs=stack.grid.crs,
        transform=stack.grid.transform,
        width=stack.grid.width - 1,
        height=stack.grid.height,
        resolution_m=stack.grid.resolution_m,
    )
    raster = ifg.coherence.geocode(alternate)
    assert raster.grid == alternate
    assert not (store.root / "geocoded" / "coherence" / "item.json").exists()
