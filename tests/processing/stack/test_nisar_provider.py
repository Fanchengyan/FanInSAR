"""Executable bounded NISAR provider tests (PROPOSAL-0035)."""

# Test modules intentionally import runtime fixtures directly.
# ruff: noqa: TC001

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
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
    _geometry_shared_radar_window,
    _radar_crop,
    make_nisar_scene_provider,
)
from faninsar.processing.stack.provider import UnsupportedStackCapabilityError
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
        lambda _sensor, uri: handles[Path(uri)],
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
        return replace(result, product=replace(result.product, grid=grid))

    monkeypatch.setattr(NisarSensor, "to_slc_product", read_product)
    monkeypatch.setattr(
        NisarSensor,
        "read_slc_window",
        lambda _sensor, handle, window, **_kwargs: handle.dataset[window],
    )
    config: dict[str, object] = {
        "extra": {"nisar_window": (1, 4, 1, 5)},
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
    calls: list[tuple[tuple[int, int], str]] = []

    def fake_rdr2geo(
        slc: RadarSLC,
        *,
        geo_grid: GeoGrid,
        **_kwargs: object,
    ) -> GeoSLC:
        calls.append((geo_grid.shape, geo_grid.crs))
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
    assert calls == [((2, 3), "EPSG:32633"), ((2, 3), "EPSG:32633")]


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

    monkeypatch.setattr(
        "faninsar.processing.stack.nisar_provider._geometry_shared_radar_window",
        lambda *_args, **_kwargs: (3, 6, 3, 7),
    )
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
    )

    callback(
        reference_path,
        secondary_path,
        output_dir=tmp_path / "pair",
        options={"coregistration_grid": "radar", "height": 0.0},
    )

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
        return SimpleNamespace(
            latitude_deg=np.array([[10.0]]),
            longitude_deg=np.array([[20.0]]),
            height_m=np.array([[123.0]]),
            converged=np.array([[True]]),
        )

    def fake_geo2rdr(*args: object, **kwargs: object) -> object:
        seen["height"] = args[3]
        seen["geo2rdr_device"] = kwargs["device"]
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
    np.testing.assert_array_equal(seen["height"], np.array([[123.0]]))
    assert seen["rdr2geo_device"] == "cuda:0"
    assert seen["geo2rdr_device"] == "cuda:0"


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


def test_nisar_provider_fails_closed_without_bounded_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unbounded scene promotion remains an explicit capability failure."""
    stack, paths, _handles = _stack(tmp_path, monkeypatch)
    with pytest.raises(UnsupportedStackCapabilityError, match="scene-production"):
        stack.scene_provider(
            paths[0],
            paths[1],
            output_dir=tmp_path / "pair",
            options={"nisar_window": None},
        )


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
