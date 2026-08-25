"""Executable bounded NISAR provider tests (PROPOSAL-0035)."""

# Test modules intentionally import runtime fixtures directly.

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from faninsar.missions.nisar import NisarSensor
from faninsar.processing.coordinates import GeoGrid
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry.dem import ConstantHeightDEM
from faninsar.processing.merge.grid import GeoGridSpec
from faninsar.processing.slc import GeoSLC, RadarSLC
from faninsar.processing.stack import NISARStack
from faninsar.processing.stack.nisar_provider import (
    _dense_secondary_mapping,
    _full_stack_reference_bounds,
    _geo_tile_for_radar_crop,
    _geometry_shared_radar_window,
    _lanczos_source_coverage,
    _radar_crop,
    _tile_resume_identity,
    make_nisar_scene_provider,
)
from faninsar.processing.stack.scene_store import CoregisteredSceneStore

from .test_nisar_stack import _result


class _Dataset:
    """Small reader-like dataset that records bounded selections."""

    shape = (4, 5)
    dtype = np.dtype("complex64")

    def __init__(self, value: complex) -> None:
        self.samples = np.full(self.shape, value, dtype=np.complex64)
        self.selections: list[tuple[slice, slice]] = []

    def __getitem__(self, selection: tuple[slice, slice]) -> np.ndarray:
        self.selections.append(selection)
        return self.samples[selection]


def _identity_dense_mapping(
    bounds: tuple[int, int, int, int],
) -> SimpleNamespace:
    """Return an identity dense mapping for one synthetic radar tile."""
    row_start, row_stop, col_start, col_stop = bounds
    azimuth, range_index = np.meshgrid(
        np.arange(row_stop - row_start, dtype=np.float64),
        np.arange(col_stop - col_start, dtype=np.float64),
        indexing="ij",
    )
    return SimpleNamespace(
        azimuth=azimuth,
        range_index=range_index,
        valid=np.ones(azimuth.shape, dtype=bool),
        source_bounds=bounds,
    )


def _stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    domain: str = "radar",
) -> tuple[NISARStack, tuple[Path, Path], dict[Path, SimpleNamespace]]:
    """Construct a Stack with two fake lazily-read RSLCs."""
    paths = tuple(
        tmp_path / f"NISAR_RSLC_{date_id}.h5" for date_id in ("20240101", "20240113")
    )
    for path in paths:
        path.touch()
    handles = {
        path: SimpleNamespace(
            filename=str(path),
            dataset=_Dataset(index + 1),
        )
        for index, path in enumerate(paths)
    }
    monkeypatch.setattr(
        NisarSensor,
        "open_product",
        lambda _sensor, uri, **_kwargs: handles[Path(uri)],
    )

    def read_product(
        _sensor: object,
        handle: object,
        **_kwargs: object,
    ) -> object:
        result = _result(
            Path(handle.filename),  # type: ignore[attr-defined]
            Path(handle.filename).stem.rsplit("_", 1)[-1],  # type: ignore[attr-defined]
        )
        # Keep the ordinary lifecycle fixture on one relative radar timeline;
        # the dedicated regression below covers absolute sensing-time offsets.
        grid = replace(
            result.product.grid,
            sensing_start=datetime(2024, 1, 1, tzinfo=UTC),
        )
        orbit = replace(
            result.product.orbit,
            vectors=tuple(
                replace(
                    vector,
                    time=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=index),
                )
                for index, vector in enumerate(result.product.orbit.vectors)
            ),
        )
        return replace(result, product=replace(result.product, grid=grid, orbit=orbit))

    monkeypatch.setattr(NisarSensor, "to_slc_product", read_product)
    monkeypatch.setattr(
        NisarSensor,
        "read_slc_window",
        lambda _sensor, handle, window, **_kwargs: handle.dataset[window],
    )
    # The tiny lifecycle fixture has synthetic orbit metadata; the geometry
    # seam itself is covered by dedicated tests below.
    monkeypatch.setattr(
        "faninsar.processing.stack.nisar_provider._geometry_shared_radar_window",
        lambda *_args, **_kwargs: (1, 4, 1, 5),
    )
    config: dict[str, object] = {
        "extra": {"nisar_window": (1, 4, 1, 5)},
        "nisar_admission": {
            "trusted_roots": [tmp_path],
            "max_size_bytes": 1024,
        },
        "dem": ConstantHeightDEM(0.0),
        "multilook": (1, 1),
        "goldstein_alpha": 0.0,
        "coregistration_grid": domain,
    }
    if domain == "geo":
        config["geo_grid"] = GeoGridSpec(
            crs="EPSG:32633",
            transform=(100.0, 10.0, 0.0, 200.0, 0.0, -10.0),
            width=3,
            height=2,
            resolution_m=(10.0, 10.0),
        )
    stack = NISARStack.from_rslc(paths, work_dir=tmp_path / "work", **config)
    return stack, paths, handles


def test_nisar_radar_provider_runs_shared_stack_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Radar RSLC windows publish scenes consumed by shared IFG formation."""
    stack, paths, handles = _stack(tmp_path, monkeypatch)

    stack.prepare_scenes().coregister_scenes().form_interferograms(multilook=(1, 1))

    store = CoregisteredSceneStore.open(
        tmp_path / "work" / "coreg" / "20240113" / "scenes"
    )
    assert store.domain == "radar"
    assert store.grid_shape == (3, 4)
    assert stack.ifg_dirs == [
        tmp_path / "work" / "ifg" / "ml_1x1" / "20240101_20240113"
    ]
    expected = (slice(1, 4), slice(1, 5))
    assert handles[paths[0]].dataset.selections == [expected]
    assert handles[paths[1]].dataset.selections == [expected]


def test_nisar_geo_provider_uses_one_shared_geo_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Geo RSLC windows are converted through the common RadarSLC API."""
    calls: list[tuple[tuple[int, int], str, object]] = []

    def fake_rdr2geo(
        slc: RadarSLC,
        *,
        geo_grid: GeoGrid,
        dem: object,
        **_kwargs: object,
    ) -> GeoSLC:
        calls.append((geo_grid.shape, geo_grid.crs, dem))
        product = replace(
            slc.product,
            grid=geo_grid,
            samples=replace(slc.product.samples, shape=geo_grid.shape),
        )
        return GeoSLC(product=product, samples=np.ones(geo_grid.shape, np.complex64))

    monkeypatch.setattr(RadarSLC, "rdr2geo", fake_rdr2geo)
    stack, _paths, _handles = _stack(tmp_path, monkeypatch, domain="geo")

    stack.prepare_scenes().coregister_scenes().form_interferograms(multilook=(1, 1))

    store = CoregisteredSceneStore.open(
        tmp_path / "work" / "coreg" / "20240113" / "scenes"
    )
    assert store.domain == "geo"
    assert store.grid_shape == (2, 3)
    assert calls == [
        ((2, 3), "EPSG:32633", stack.config.dem),
        ((2, 3), "EPSG:32633", stack.config.dem),
    ]


def test_nisar_provider_maps_secondary_physical_window_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Secondary reads follow sensing time and update crop grid metadata."""
    reference_path = tmp_path / "NISAR_RSLC_20240101.h5"
    secondary_path = tmp_path / "NISAR_RSLC_20240113.h5"
    reference_path.touch()
    secondary_path.touch()
    reference_result = _result(reference_path, "20240101")
    secondary_result = _result(secondary_path, "20240113")
    reference_grid = replace(
        reference_result.product.grid,
        shape=(8, 8),
        sensing_start=datetime(2024, 1, 1, tzinfo=UTC),
    )
    secondary_grid = replace(
        secondary_result.product.grid,
        shape=(8, 10),
        starting_slant_range_m=799_997.1,
        sensing_start=datetime(2024, 1, 1, 0, 0, 0, 4000, tzinfo=UTC),
    )
    reference_product = replace(
        reference_result.product,
        grid=reference_grid,
        samples=replace(reference_result.product.samples, shape=reference_grid.shape),
    )
    secondary_product = replace(
        secondary_result.product,
        grid=secondary_grid,
        samples=replace(secondary_result.product.samples, shape=secondary_grid.shape),
    )
    reference_array = np.arange(64, dtype=np.float32).reshape(8, 8).astype(np.complex64)
    secondary_array = (
        np.arange(80, dtype=np.float32).reshape(8, 10).astype(np.complex64)
    )
    handles = {"reference": object(), "secondary": object()}
    selections: list[tuple[str, tuple[slice, slice]]] = []

    class Sensor:
        """Fake normalized NISAR window reader."""

        def read_slc_window(
            self,
            handle: object,
            window: tuple[slice, slice],
            **_kwargs: object,
        ) -> np.ndarray:
            key = "reference" if handle is handles["reference"] else "secondary"
            selections.append((key, window))
            return (reference_array if key == "reference" else secondary_array)[window]

    mapping_inputs: dict[str, object] = {}

    def fake_mapping(*_args: object, **kwargs: object) -> tuple[int, int, int, int]:
        mapping_inputs.update(kwargs)
        return (3, 6, 3, 7)

    monkeypatch.setattr(
        "faninsar.processing.stack.nisar_provider._geometry_shared_radar_window",
        fake_mapping,
    )
    configured_dem = ConstantHeightDEM(55.0)
    callback = make_nisar_scene_provider(
        sensor=Sensor(),
        handles={
            reference_path: handles["reference"],
            secondary_path: handles["secondary"],
        },
        products={"20240101": reference_product, "20240113": secondary_product},
        lineage={
            "20240101": str(reference_path),
            "20240113": str(secondary_path),
        },
        master="20240101",
        channel=("B", "HH"),
        configured_window=(5, 8, 2, 6),
        configured_dem=configured_dem,
    )

    callback(
        reference_path,
        secondary_path,
        output_dir=tmp_path / "pair",
        options={"coregistration_grid": "radar"},
    )

    assert mapping_inputs["dem"] is configured_dem
    assert mapping_inputs["height_m"] is None
    assert selections == [
        ("reference", (slice(5, 8), slice(2, 6))),
        ("secondary", (slice(3, 6), slice(3, 7))),
    ]
    secondary_crop = _radar_crop(
        secondary_product,
        secondary_array[3:6, 3:7],
        (3, 6, 3, 7),
    )
    assert secondary_crop.grid.shape == (3, 4)
    assert secondary_crop.grid.starting_slant_range_m == pytest.approx(800_004.0)
    assert secondary_crop.grid.sensing_start == datetime(
        2024, 1, 1, 0, 0, 0, 10_000, tzinfo=UTC
    )
    np.testing.assert_array_equal(secondary_crop.samples, secondary_array[3:6, 3:7])

    manifest = (tmp_path / "pair" / "scenes" / "manifest.json").read_text()
    assert "source_digest" in manifest
    assert "source_id" in manifest
    assert "dem_identity" in manifest
    assert "B/HH" in manifest


def test_nisar_geometry_mapping_passes_dem_and_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Radar→Geo→Radar seam uses the admitted DEM and device unchanged."""
    reference = _result(tmp_path / "reference.h5", "20240101").product
    secondary = _result(tmp_path / "secondary.h5", "20240113").product
    dem = ConstantHeightDEM(123.0)
    seen: dict[str, object] = {}

    def fake_rdr2geo(*args: object, **kwargs: object) -> object:
        seen["dem"] = args[3]
        seen["rdr2geo_device"] = kwargs["device"]
        seen["rdr2geo_doppler_tol_hz"] = kwargs["doppler_tol_hz"]
        return SimpleNamespace(
            latitude_deg=np.array([[10.0]]),
            longitude_deg=np.array([[20.0]]),
            height_m=np.array([[123.0]]),
            converged=np.array([[True]]),
        )

    def fake_geo2rdr(*args: object, **kwargs: object) -> object:
        seen["height"] = args[3]
        seen["geo2rdr_device"] = kwargs["device"]
        seen["geo2rdr_doppler_tol_hz"] = kwargs["doppler_tol_hz"]
        return SimpleNamespace(
            azimuth_index=np.array([[2.0]]),
            range_index=np.array([[2.0]]),
            converged=np.array([[True]]),
        )

    # The helpers are imported lazily, so patch their defining module.
    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.run_rdr2geo",
        fake_rdr2geo,
    )
    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.run_geo2rdr",
        fake_geo2rdr,
    )

    bounds = _geometry_shared_radar_window(
        reference,
        secondary,
        (1, 3, 1, 3),
        device="cuda:0",
        dem=dem,
    )

    assert bounds == (1, 3, 1, 3)
    assert seen["dem"] is dem
    np.testing.assert_array_equal(seen["height"], np.array([123.0]))
    assert seen["rdr2geo_device"] == "cuda:0"
    assert seen["geo2rdr_device"] == "cuda:0"
    assert seen["rdr2geo_doppler_tol_hz"] == 0.1
    assert seen["geo2rdr_doppler_tol_hz"] == 0.1


def test_nisar_geometry_mapping_materializes_noncontiguous_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NISAR normalizes strided Radar→Geo outputs before Geo→Radar dispatch."""
    reference = _result(tmp_path / "reference.h5", "20240101").product
    secondary = _result(tmp_path / "secondary.h5", "20240113").product
    seen: dict[str, object] = {}

    def strided(value: float) -> np.ndarray:
        """Return a non-contiguous one-pixel float64 array."""
        backing = np.full((2, 2), value, dtype=np.float64)
        return backing[::2, ::2]

    def fake_rdr2geo(*_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(
            latitude_deg=strided(10.0),
            longitude_deg=strided(20.0),
            height_m=strided(123.0),
            converged=np.array([[True]]),
        )

    def fake_geo2rdr(*args: object, **_kwargs: object) -> object:
        geometry_inputs = args[1:4]
        seen["inputs"] = geometry_inputs
        assert all(
            isinstance(value, np.ndarray)
            and value.ndim == 1
            and value.shape == (1,)
            and value.flags.c_contiguous
            for value in geometry_inputs
        )
        return SimpleNamespace(
            azimuth_index=np.array([[2.0]]),
            range_index=np.array([[2.0]]),
            converged=np.array([[True]]),
        )

    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.run_rdr2geo",
        fake_rdr2geo,
    )
    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.run_geo2rdr",
        fake_geo2rdr,
    )

    bounds = _geometry_shared_radar_window(
        reference,
        secondary,
        (1, 3, 1, 3),
        device="cuda:0",
        height_m=123.0,
    )

    assert bounds == (1, 3, 1, 3)
    assert len(seen["inputs"]) == 3  # type: ignore[arg-type]


def test_nisar_geometry_mapping_rejects_full_window_outside_secondary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mapped crop outside the secondary grid is rejected without clamping."""
    reference = _result(tmp_path / "reference.h5", "20240101").product
    secondary = _result(tmp_path / "secondary.h5", "20240113").product

    def fake_rdr2geo(*_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(
            latitude_deg=np.array([[10.0]]),
            longitude_deg=np.array([[20.0]]),
            height_m=np.array([[0.0]]),
            converged=np.array([[True]]),
        )

    def fake_geo2rdr(*_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(
            azimuth_index=np.array([[99.0]]),
            range_index=np.array([[99.0]]),
            converged=np.array([[True]]),
        )

    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.run_rdr2geo",
        fake_rdr2geo,
    )
    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.run_geo2rdr",
        fake_geo2rdr,
    )

    with pytest.raises(InvalidProcessingStateError, match="outside the source grid"):
        _geometry_shared_radar_window(
            reference,
            secondary,
            (1, 3, 1, 3),
            device="cpu",
            height_m=0.0,
        )


def test_nisar_geometry_mapping_rejects_missing_height_and_nonconvergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing height and failed geometry lanes produce explicit errors."""
    reference = _result(tmp_path / "reference.h5", "20240101").product
    secondary = _result(tmp_path / "secondary.h5", "20240113").product

    with pytest.raises(InvalidProcessingStateError, match="explicit DEM or height"):
        _geometry_shared_radar_window(
            reference,
            secondary,
            (1, 3, 1, 3),
            device="cpu",
        )

    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.run_rdr2geo",
        lambda *_args, **_kwargs: SimpleNamespace(converged=np.array([[False]])),
    )
    with pytest.raises(
        InvalidProcessingStateError, match="did not converge in rdr2geo"
    ):
        _geometry_shared_radar_window(
            reference,
            secondary,
            (1, 3, 1, 3),
            device="cpu",
            height_m=0.0,
        )


def test_nisar_provider_promotes_full_scene_without_bounded_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting the bounded window publishes the common full-scene overlap."""
    stack, paths, _handles = _stack(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "faninsar.processing.stack.nisar_provider._dense_secondary_mapping",
        lambda _reference, _secondary, bounds, **_kwargs: _identity_dense_mapping(
            bounds
        ),
    )
    monkeypatch.setattr(
        "faninsar.processing.stack.nisar_provider._lanczos_source_coverage",
        lambda _samples, azimuth, _range_index: np.ones(azimuth.shape, dtype=bool),
    )
    stack.scene_provider(
        paths[0],
        paths[1],
        output_dir=tmp_path / "pair",
        options={
            "nisar_window": None,
            "coregistration_grid": "radar",
        },
    )

    store = CoregisteredSceneStore.open(tmp_path / "pair" / "scenes")
    assert store.grid_shape == (4, 5)
    assert [unit.tag for unit in store.units] == ["NISAR_b000000"]
    assert store.units[0].row_origin == 0
    assert store.units[0].col_origin == 0


def test_nisar_full_scene_uses_deterministic_row_major_tiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full-scene tile reads and manifest placements cover the common grid."""
    reference_path = tmp_path / "NISAR_RSLC_20240101.h5"
    secondary_path = tmp_path / "NISAR_RSLC_20240113.h5"
    reference_path.write_bytes(b"reference")
    secondary_path.write_bytes(b"secondary")
    reference_result = _result(reference_path, "20240101")
    secondary_result = _result(secondary_path, "20240113")
    handles = {reference_path: object(), secondary_path: object()}
    selections: list[tuple[slice, slice]] = []

    class Sensor:
        """Record every normalized window read."""

        def read_slc_window(
            self,
            _handle: object,
            window: tuple[slice, slice],
            **_kwargs: object,
        ) -> np.ndarray:
            selections.append(window)
            return np.ones(
                (window[0].stop - window[0].start, window[1].stop - window[1].start),
                dtype=np.complex64,
            )

    monkeypatch.setattr(
        "faninsar.processing.stack.nisar_provider._dense_secondary_mapping",
        lambda _reference, _secondary, bounds, **_kwargs: _identity_dense_mapping(
            bounds
        ),
    )
    monkeypatch.setattr(
        "faninsar.processing.stack.nisar_provider._lanczos_source_coverage",
        lambda _samples, azimuth, _range_index: np.ones(azimuth.shape, dtype=bool),
    )
    callback = make_nisar_scene_provider(
        sensor=Sensor(),
        handles=handles,
        products={
            "20240101": reference_result.product,
            "20240113": secondary_result.product,
        },
        lineage={
            "20240101": str(reference_path),
            "20240113": str(secondary_path),
        },
        master="20240101",
        channel=("B", "HH"),
        configured_tile_shape=(2, 3),
        configured_height=0.0,
    )
    callback(
        reference_path,
        secondary_path,
        output_dir=tmp_path / "full",
        options={"coregistration_grid": "radar", "device": "cpu"},
    )

    store = CoregisteredSceneStore.open(tmp_path / "full" / "scenes")
    assert store.grid_shape == (4, 5)
    assert [unit.tag for unit in store.units] == [
        "NISAR_b000000",
        "NISAR_b000001",
        "NISAR_b000002",
        "NISAR_b000003",
    ]
    assert [(unit.row_origin, unit.col_origin, unit.shape) for unit in store.units] == [
        (0, 0, (2, 3)),
        (0, 3, (2, 2)),
        (2, 0, (2, 3)),
        (2, 3, (2, 2)),
    ]
    assert len(selections) == 8

    callback(
        reference_path,
        secondary_path,
        output_dir=tmp_path / "full",
        options={"coregistration_grid": "radar", "device": "cpu"},
    )
    assert len(selections) == 8


def test_nisar_full_stack_uses_complete_master_grid(tmp_path: Path) -> None:
    """All coregistered dates share the complete master-grid extent."""
    products = {
        date_id: _result(tmp_path / f"{date_id}.h5", date_id).product
        for date_id in ("20240101", "20240113", "20240125")
    }
    assert _full_stack_reference_bounds(
        products,
        "20240101",
        device="cpu",
        dem=None,
        height_m=0.0,
    ) == (0, 4, 0, 5)


def test_nisar_dense_mapping_uses_spatially_varying_per_pixel_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dense mapping retains fractional, spatially varying secondary indices."""
    reference = _result(tmp_path / "reference.h5", "20240101").product
    secondary = _result(tmp_path / "secondary.h5", "20240113").product

    def fake_rdr2geo(
        _model: object,
        azimuth: np.ndarray,
        range_index: np.ndarray,
        _dem: object,
        **_kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            converged=np.ones(azimuth.shape, dtype=bool),
            latitude_deg=np.asarray(azimuth, dtype=np.float64),
            longitude_deg=np.asarray(range_index, dtype=np.float64),
            height_m=np.zeros(azimuth.shape, dtype=np.float64),
        )

    def fake_geo2rdr(
        _model: object,
        latitude: np.ndarray,
        longitude: np.ndarray,
        _height: np.ndarray,
        **_kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            converged=np.ones(latitude.shape, dtype=bool),
            azimuth_index=latitude + 0.1 * longitude,
            range_index=longitude + 0.1 * latitude,
        )

    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.run_rdr2geo",
        fake_rdr2geo,
    )
    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.run_geo2rdr",
        fake_geo2rdr,
    )

    mapping = _dense_secondary_mapping(
        reference,
        secondary,
        (1, 3, 1, 4),
        device="cpu",
        dem=ConstantHeightDEM(0.0),
    )

    assert mapping.source_bounds == (0, 4, 0, 5)
    assert np.all(mapping.valid)
    assert mapping.azimuth[0, 0] == pytest.approx(1.1)
    assert mapping.azimuth[0, -1] == pytest.approx(1.3)
    assert mapping.range_index[0, 0] == pytest.approx(1.1)
    assert mapping.range_index[-1, 0] == pytest.approx(1.2)


def test_nisar_tile_resume_identity_binds_scientific_configuration() -> None:
    """DEM, channel, geometry, and tile changes invalidate tile resume."""
    base = {
        "dem_identity": "constant:0",
        "channel": ["B", "HH"],
        "geometry": "per_pixel_rdr2geo_geo2rdr_lanczos4",
        "tile_shape": [2048, 2048],
    }
    identity = _tile_resume_identity(base)
    for key, value in (
        ("dem_identity", "constant:10"),
        ("channel", ["B", "VV"]),
        ("geometry", "different"),
        ("tile_shape", [1024, 1024]),
    ):
        changed = dict(base)
        changed[key] = value
        assert _tile_resume_identity(changed) != identity


def test_nisar_lanczos_coverage_rejects_invalid_source_support() -> None:
    """Dense resampling masks a destination touching invalid SLC support."""
    samples = np.ones((20, 20), dtype=np.complex64)
    samples[10, 10] = 0.0
    azimuth = np.array([[5.0, 10.0, 15.0]], dtype=np.float64)
    ranges = np.array([[5.0, 10.0, 15.0]], dtype=np.float64)

    coverage = _lanczos_source_coverage(samples, azimuth, ranges)

    assert coverage.tolist() == [[True, False, True]]


def test_nisar_geo_tile_preserves_projected_global_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A UTM tile keeps its placement in the configured full Geo grid."""
    from pyproj import Transformer

    longitude = np.array([-147.001, -147.0, -146.999], dtype=np.float64)
    latitude = np.array([65.001, 65.0, 64.999], dtype=np.float64)
    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
    x_coordinates, y_coordinates = Transformer.from_crs(
        "EPSG:4326", "EPSG:32606", always_xy=True
    ).transform(longitude_grid, latitude_grid)
    x_min = float(np.min(x_coordinates)) - 100.0
    y_max = float(np.max(y_coordinates)) + 100.0
    target = GeoGrid(
        shape=(30, 30),
        crs="EPSG:32606",
        transform=(20.0, 0.0, x_min, 0.0, -20.0, y_max),
    )
    monkeypatch.setattr(
        "faninsar.processing.geometry.prepare_production.run_rdr2geo",
        lambda *_args, **_kwargs: SimpleNamespace(
            converged=np.ones(9, dtype=bool),
            latitude_deg=latitude_grid.reshape(-1),
            longitude_deg=longitude_grid.reshape(-1),
        ),
    )

    local, row_origin, col_origin = _geo_tile_for_radar_crop(
        _result(tmp_path / "reference.h5", "20240101").product,
        (0, 4, 0, 5),
        target,
        device="cpu",
        dem=ConstantHeightDEM(0.0),
    )

    assert row_origin > 0
    assert col_origin > 0
    assert local.transform[2] == pytest.approx(
        target.transform[2] + col_origin * target.transform[0]
    )
    assert local.transform[5] == pytest.approx(
        target.transform[5] + row_origin * target.transform[4]
    )
    assert row_origin + local.shape[0] <= target.shape[0]
    assert col_origin + local.shape[1] <= target.shape[1]


def test_nisar_three_date_multitile_radar_and_projected_geo_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three dates complete Radar and projected-Geo lifecycle without gaps."""
    dates = ("20240101", "20240113", "20240125")
    paths = tuple(tmp_path / f"NISAR_RSLC_{date_id}.h5" for date_id in dates)
    for path in paths:
        path.write_bytes(path.name.encode())
    handles = {
        path: SimpleNamespace(
            filename=str(path),
            dataset=_Dataset(complex(index + 1)),
        )
        for index, path in enumerate(paths)
    }
    monkeypatch.setattr(
        NisarSensor,
        "open_product",
        lambda _sensor, uri, **_kwargs: handles[Path(uri)],
    )
    monkeypatch.setattr(
        NisarSensor,
        "to_slc_product",
        lambda _sensor, handle, **_kwargs: _result(
            Path(handle.filename), Path(handle.filename).stem.rsplit("_", 1)[-1]
        ),
    )
    monkeypatch.setattr(
        NisarSensor,
        "read_slc_window",
        lambda _sensor, handle, window, **_kwargs: handle.dataset[window],
    )
    mapping_calls: list[tuple[str, tuple[int, int, int, int]]] = []

    def varying_dense_mapping(
        _reference: object,
        secondary: object,
        bounds: tuple[int, int, int, int],
        **_kwargs: object,
    ) -> SimpleNamespace:
        mapping_calls.append((secondary.acquisition_id, bounds))
        row_start, row_stop, col_start, col_stop = bounds
        rows, cols = np.meshgrid(
            np.arange(row_start, row_stop, dtype=np.float64),
            np.arange(col_start, col_stop, dtype=np.float64),
            indexing="ij",
        )
        date_shift = 0.15 if secondary.acquisition_id == "20240113" else 0.3
        return SimpleNamespace(
            azimuth=rows + date_shift * (cols / 5.0),
            range_index=cols + date_shift * (rows / 4.0),
            valid=np.ones(rows.shape, dtype=bool),
            source_bounds=(0, 4, 0, 5),
        )

    def fake_resample(
        source: np.ndarray,
        azimuth: np.ndarray,
        range_index: np.ndarray,
        *,
        valid: np.ndarray,
        **_kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert np.ptp(azimuth) > 0.0 or np.ptp(range_index) > 0.0
        value = np.complex64(source[0, 0])
        return np.full(azimuth.shape, value, np.complex64), valid.copy()

    monkeypatch.setattr(
        "faninsar.processing.stack.nisar_provider._dense_secondary_mapping",
        varying_dense_mapping,
    )
    monkeypatch.setattr(
        "faninsar.processing.stack.nisar_provider._lanczos_source_coverage",
        lambda _samples, azimuth, _range_index: np.ones(azimuth.shape, dtype=bool),
    )
    monkeypatch.setattr(
        "faninsar.processing.pipeline.geo_resample.resample_complex_at_coordinates",
        fake_resample,
    )

    radar_stack = NISARStack.from_rslc(
        paths,
        work_dir=tmp_path / "radar_work",
        dem=ConstantHeightDEM(0.0),
        multilook=(1, 1),
        goldstein_alpha=0.0,
        coregistration_grid="radar",
        extra={"nisar_tile_shape": (2, 3)},
    )
    radar_stack.prepare_scenes().coregister_scenes().form_interferograms(
        multilook=(1, 1)
    )
    assert len(radar_stack.ifg_dirs) == 3
    for date_id in dates:
        store = CoregisteredSceneStore.open(
            tmp_path / "radar_work" / "coreg" / date_id / "scenes"
        )
        assert store.grid_shape == (4, 5)
        assert len(store.units) == 4

    full_geo_grid = GeoGrid(
        shape=(4, 5),
        crs="EPSG:32606",
        transform=(20.0, 0.0, 500_000.0, 0.0, -20.0, 7_200_000.0),
    )

    def overlapping_geo_tile(
        _product: object,
        bounds: tuple[int, int, int, int],
        target: GeoGrid,
        **_kwargs: object,
    ) -> tuple[GeoGrid, int, int]:
        row_start = max(0, bounds[0] - 1)
        row_stop = min(target.shape[0], bounds[1] + 1)
        col_start = max(0, bounds[2] - 1)
        col_stop = min(target.shape[1], bounds[3] + 1)
        return (
            GeoGrid(
                shape=(row_stop - row_start, col_stop - col_start),
                crs=target.crs,
                transform=(
                    target.transform[0],
                    0.0,
                    target.transform[2] + col_start * target.transform[0],
                    0.0,
                    target.transform[4],
                    target.transform[5] + row_start * target.transform[4],
                ),
            ),
            row_start,
            col_start,
        )

    monkeypatch.setattr(
        "faninsar.processing.stack.nisar_provider._geo_tile_for_radar_crop",
        overlapping_geo_tile,
    )

    def fake_geocode_aligned(
        _product: object,
        reference: np.ndarray,
        secondary: np.ndarray,
        bounds: tuple[int, int, int, int],
        target: GeoGrid,
        **_kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        reference_value = reference[np.isfinite(reference)][0]
        secondary_value = secondary[np.isfinite(secondary)][0]
        reference_geo = np.full(target.shape, reference_value, np.complex64)
        secondary_geo = np.full(target.shape, secondary_value, np.complex64)
        if bounds[2] == 0:
            # Reference-only validity at this overlap must not claim half a pair;
            # the following range tile owns the pixel jointly instead.
            secondary_geo[:, -1] = np.complex64(np.nan + 1j * np.nan)
        return reference_geo, secondary_geo

    monkeypatch.setattr(
        "faninsar.processing.stack.nisar_provider._geocode_aligned_radar_tile",
        fake_geocode_aligned,
    )
    geo_stack = NISARStack.from_rslc(
        paths,
        work_dir=tmp_path / "geo_work",
        dem=ConstantHeightDEM(0.0),
        geo_grid=GeoGridSpec(
            crs=full_geo_grid.crs,
            transform=(500_000.0, 20.0, 0.0, 7_200_000.0, 0.0, -20.0),
            width=5,
            height=4,
            resolution_m=(20.0, 20.0),
        ),
        multilook=(1, 1),
        goldstein_alpha=0.0,
        coregistration_grid="geo",
        extra={"nisar_tile_shape": (2, 3)},
    )
    geo_stack.prepare_scenes().coregister_scenes().form_interferograms(multilook=(1, 1))
    assert len(geo_stack.ifg_dirs) == 3
    for date_id in dates:
        store = CoregisteredSceneStore.open(
            tmp_path / "geo_work" / "coreg" / date_id / "scenes"
        )
        reference_coverage = np.zeros(store.grid_shape, dtype=np.int8)
        secondary_coverage = np.zeros(store.grid_shape, dtype=np.int8)
        for unit in store.units:
            reference, secondary, _ = store.read(unit.tag)
            row_slice = slice(unit.row_origin, unit.row_origin + unit.shape[0])
            col_slice = slice(unit.col_origin, unit.col_origin + unit.shape[1])
            reference_coverage[row_slice, col_slice] += np.isfinite(reference.real)
            secondary_coverage[row_slice, col_slice] += np.isfinite(secondary.real)
            assert np.array_equal(
                np.isfinite(reference.real),
                np.isfinite(secondary.real),
            )
            assert unit.phase_state is not None
            assert unit.phase_state["coverage_policy"] == (
                "joint_first_valid_row_major_v1"
            )
            assert unit.phase_state["pair_valid_pixels"] == int(
                np.sum(np.isfinite(reference.real))
            )
        assert np.all(reference_coverage == 1)
        assert np.all(secondary_coverage == 1)
    assert {date_id for date_id, _bounds in mapping_calls} == {
        "20240113",
        "20240125",
    }


def test_nisar_provider_rejects_source_content_mutation_after_admission(
    tmp_path: Path,
) -> None:
    """A provider snapshot prevents stale scene reuse after source mutation."""
    reference_path = tmp_path / "NISAR_RSLC_20240101.h5"
    secondary_path = tmp_path / "NISAR_RSLC_20240113.h5"
    reference_path.write_bytes(b"reference-v1")
    secondary_path.write_bytes(b"secondary-v1")
    reference_result = _result(reference_path, "20240101")
    secondary_result = _result(secondary_path, "20240113")
    handles = {reference_path: object(), secondary_path: object()}

    class Sensor:
        """Fake normalized NISAR window reader."""

        def read_slc_window(
            self,
            _handle: object,
            window: tuple[slice, slice],
            **_kwargs: object,
        ) -> np.ndarray:
            return np.ones(
                (window[0].stop - window[0].start, window[1].stop - window[1].start),
                dtype=np.complex64,
            )

    callback = make_nisar_scene_provider(
        sensor=Sensor(),
        handles=handles,
        products={
            "20240101": reference_result.product,
            "20240113": secondary_result.product,
        },
        lineage={
            "20240101": str(reference_path),
            "20240113": str(secondary_path),
        },
        master="20240101",
        channel=("B", "HH"),
        configured_window=(0, 2, 0, 2),
    )
    secondary_path.write_bytes(b"secondary-v2")
    with pytest.raises(InvalidProcessingStateError, match="changed after admission"):
        callback(
            reference_path,
            secondary_path,
            output_dir=tmp_path / "pair",
            options={"coregistration_grid": "radar", "height": 0.0},
        )
