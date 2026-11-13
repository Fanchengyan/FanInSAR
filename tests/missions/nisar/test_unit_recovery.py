"""NISAR per-tile checkpoint recovery tests (PROPOSAL-0088)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from faninsar.io.storage.scene_store import CoregisteredSceneStore
from faninsar.io.storage.unit_checkpoint import UnitCheckpointStore
from faninsar.missions.nisar import NisarSensor, NISARStack
from faninsar.missions.nisar.provider import (
    _specification_identity,
    _tile_checkpoint_specification,
    _tile_unit_locator,
    make_nisar_scene_provider,
)
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry import ConstantDEM
from faninsar.stack.session import Stack

from .test_provider import _Dataset, _identity_dense_mapping
from .test_stack import _result


class _Sensor:
    """Deterministic tiny NISAR RSLC window reader."""

    def __init__(self) -> None:
        self.reads: list[tuple[slice, slice]] = []

    def read_slc_window(
        self,
        handle: SimpleNamespace,
        window: tuple[slice, slice],
        **_kwargs: object,
    ) -> np.ndarray:
        self.reads.append(window)
        shape = (
            window[0].stop - window[0].start,
            window[1].stop - window[1].start,
        )
        return np.full(shape, handle.value, dtype=np.complex64)


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    height: float = 0.0,
    checkpoint_root: Path | None = None,
    integrity_root: Path | None = None,
    verification: str = "fast",
    fail_on_mapping_call: int | None = None,
) -> tuple[object, _Sensor, list[int]]:
    """Build one real NISAR provider over the tiny deterministic fixture."""
    reference_path = tmp_path / "NISAR_RSLC_20240101.h5"
    secondary_path = tmp_path / "NISAR_RSLC_20240113.h5"
    reference_path.write_bytes(b"reference")
    secondary_path.write_bytes(b"secondary")
    products = {
        "20240101": _result(reference_path, "20240101").product,
        "20240113": _result(secondary_path, "20240113").product,
    }
    handles = {
        reference_path: SimpleNamespace(value=np.complex64(1 + 1j)),
        secondary_path: SimpleNamespace(value=np.complex64(2 + 2j)),
    }
    sensor = _Sensor()
    calls: list[int] = []

    def mapping(
        _reference: object,
        _secondary: object,
        bounds: tuple[int, int, int, int],
        **_kwargs: object,
    ) -> object:
        calls.append(1)
        if fail_on_mapping_call is not None and len(calls) == fail_on_mapping_call:
            message = "injected NISAR tile interruption"
            raise RuntimeError(message)
        return _identity_dense_mapping(bounds)

    monkeypatch.setattr(
        "faninsar.missions.nisar.provider._dense_secondary_mapping", mapping
    )
    monkeypatch.setattr(
        "faninsar.missions.nisar.provider._lanczos_source_coverage",
        lambda _samples, azimuth, _range_index: np.ones(azimuth.shape, dtype=bool),
    )
    callback = make_nisar_scene_provider(
        sensor=sensor,
        handles=handles,
        products=products,
        lineage={
            "20240101": str(reference_path),
            "20240113": str(secondary_path),
        },
        reference="20240101",
        channel=("B", "HH"),
        configured_tile_shape=(2, 3),
        configured_height=height,
        checkpoint_root=checkpoint_root,
        integrity_root=integrity_root,
        verification=verification,
    )
    return callback, sensor, calls


def _run(callback: object, tmp_path: Path, name: str = "pair") -> None:
    """Run one full-scene Radar pair through the public provider callback."""
    callback(
        tmp_path / "NISAR_RSLC_20240101.h5",
        tmp_path / "NISAR_RSLC_20240113.h5",
        output_dir=tmp_path / name,
        options={"coregistration_grid": "radar", "device": "cpu"},
    )


def test_interrupted_nisar_tile_reuses_the_committed_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after tile one keeps that commit and skips its numerics."""
    callback, sensor, calls = _fixture(
        tmp_path,
        monkeypatch,
        checkpoint_root=tmp_path / "checkpoints",
        fail_on_mapping_call=2,
    )
    with pytest.raises(RuntimeError, match="injected NISAR tile interruption"):
        _run(callback, tmp_path)
    assert len(calls) == 2
    assert len(sensor.reads) == 2
    assert not (tmp_path / "pair" / "item.json").is_file()

    callback, sensor, calls = _fixture(
        tmp_path,
        monkeypatch,
        checkpoint_root=tmp_path / "checkpoints",
    )
    _run(callback, tmp_path)

    # Only the three missing tiles are recomputed; the committed prefix is not.
    assert len(calls) == 3
    assert len(sensor.reads) == 6
    store = CoregisteredSceneStore.open(tmp_path / "pair")
    assert [unit.tag for unit in store.units] == [
        "NISAR_b000000",
        "NISAR_b000001",
        "NISAR_b000002",
        "NISAR_b000003",
    ]


def test_repeat_secondaries_reuse_reference_and_secondary_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeated pair reads no source windows and keeps the Reference id."""
    callback, sensor, calls = _fixture(
        tmp_path,
        monkeypatch,
        checkpoint_root=tmp_path / "checkpoints",
    )
    _run(callback, tmp_path)
    reference_id = CoregisteredSceneStore.open(tmp_path / "20240101").write_id
    checkpoint_ids = sorted(
        path.name for path in (tmp_path / "checkpoints").iterdir() if path.is_dir()
    )
    assert len(calls) == 4
    assert len(sensor.reads) == 8

    _run(callback, tmp_path)

    assert len(calls) == 4
    assert len(sensor.reads) == 8
    assert CoregisteredSceneStore.open(tmp_path / "20240101").write_id == reference_id
    assert (
        sorted(
            path.name for path in (tmp_path / "checkpoints").iterdir() if path.is_dir()
        )
        == checkpoint_ids
    )


def test_dem_and_correction_changes_miss_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed DEM recomputes tiles without a payload-wide identity."""
    callback, _sensor, calls = _fixture(
        tmp_path,
        monkeypatch,
        checkpoint_root=tmp_path / "checkpoints",
    )
    _run(callback, tmp_path)
    assert len(calls) == 4

    callback, _sensor, calls = _fixture(
        tmp_path,
        monkeypatch,
        height=10.0,
        checkpoint_root=tmp_path / "checkpoints",
    )
    _run(callback, tmp_path)
    assert len(calls) == 4

    base = {
        "role": "secondary",
        "date_id": "20240113",
        "reference_id": "20240101",
        "domain": "radar",
        "tile_bounds": (0, 2, 0, 3),
        "tile_output_shape": (2, 3),
        "tile_shape": (2, 3),
        "grid_shape": (4, 5),
        "grid_identity": "radar:4x5",
        "channel": ("B", "HH"),
        "dem_identity": "constant:0",
        "geometry": "per_pixel_rdr2geo_geo2rdr_lanczos4",
        "flatten_stage": "coregistration",
        "source": "/data/20240113.h5",
        "admission": {"source_id": "a"},
        "orbit": {"reference_frame": "ITRF", "source": "orbit"},
        "reference_unit": {"locator": "ref", "specification_id": "spec"},
    }
    identity = _specification_identity(_tile_checkpoint_specification(**base))
    for key, value in (
        ("source", "/data/other.h5"),
        ("admission", {"source_id": "b"}),
        ("orbit", {"reference_frame": "ITRF", "source": "other-orbit"}),
        ("dem_identity", "constant:10"),
        ("flatten_stage", "interferogram"),
    ):
        changed = dict(base)
        changed[key] = value
        assert _specification_identity(_tile_checkpoint_specification(**changed)) != (
            identity
        )


def test_fast_mode_never_hashes_and_strict_requires_a_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fast reuse is hash-free; strict recomputes baseline-less units once."""
    calls_to_integrity: list[str] = []

    def fail_integrity(*_args: object, **_kwargs: object) -> object:
        calls_to_integrity.append("called")
        message = "fast mode must not hash payloads"
        raise AssertionError(message)

    monkeypatch.setattr("faninsar.stack.integrity.establish_integrity", fail_integrity)
    monkeypatch.setattr("faninsar.stack.integrity.verify_integrity", fail_integrity)
    callback, _sensor, calls = _fixture(
        tmp_path,
        monkeypatch,
        checkpoint_root=tmp_path / "checkpoints",
        integrity_root=tmp_path / "integrity",
    )
    _run(callback, tmp_path)
    _run(callback, tmp_path)
    assert calls_to_integrity == []
    assert len(calls) == 4

    monkeypatch.undo()
    callback, _sensor, calls = _fixture(
        tmp_path,
        monkeypatch,
        checkpoint_root=tmp_path / "checkpoints",
        integrity_root=tmp_path / "integrity",
        verification="strict",
    )
    _run(callback, tmp_path)
    assert len(calls) == 4

    callback, _sensor, calls = _fixture(
        tmp_path,
        monkeypatch,
        checkpoint_root=tmp_path / "checkpoints",
        integrity_root=tmp_path / "integrity",
        verification="strict",
    )
    _run(callback, tmp_path)
    assert len(calls) == 0

    with pytest.raises(InvalidProcessingStateError, match="checkpoint_root"):
        _fixture(
            tmp_path,
            monkeypatch,
            checkpoint_root=None,
            integrity_root=tmp_path / "integrity",
            verification="strict",
        )


def test_corrupt_checkpoint_recomputes_without_deleting_the_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A structurally corrupt unit is recomputed through the same locator."""
    callback, _sensor, calls = _fixture(
        tmp_path,
        monkeypatch,
        checkpoint_root=tmp_path / "checkpoints",
    )
    _run(callback, tmp_path)
    assert len(calls) == 4
    units = sorted(
        path for path in (tmp_path / "checkpoints").iterdir() if path.is_dir()
    )
    assert len(units) == 8
    chunk = next(
        path
        for path in units[0].rglob("*")
        if path.is_file() and path.name not in {"item.json", "zarr.json", ".zarray"}
    )
    chunk.write_bytes(b"")

    callback, _sensor, calls = _fixture(
        tmp_path,
        monkeypatch,
        checkpoint_root=tmp_path / "checkpoints",
    )
    _run(callback, tmp_path)

    # Only the corrupt unit's tile is recomputed; the same locator is reused.
    assert len(calls) == 1
    assert (
        sorted(path for path in (tmp_path / "checkpoints").iterdir() if path.is_dir())
        == units
    )
    assert UnitCheckpointStore(tmp_path / "checkpoints").root.is_dir()


def test_tile_locator_uses_the_absolute_source_window() -> None:
    """Tile index movement cannot change the stable physical locator."""
    first = _tile_unit_locator("secondary", "20240113", ("B", "HH"), (0, 2, 0, 3))
    shifted = _tile_unit_locator("secondary", "20240113", ("B", "HH"), (0, 2, 0, 3))
    other = _tile_unit_locator("secondary", "20240113", ("B", "HH"), (2, 4, 0, 3))
    assert first == shifted
    assert first != other


def test_open_restores_workspace_checkpoint_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reopened NISAR Stack reuses units from its runtime work directory."""
    paths = tuple(
        tmp_path / f"NISAR_RSLC_{date_id}.h5" for date_id in ("20240101", "20240113")
    )
    handles = {
        path: SimpleNamespace(
            filename=str(path),
            dataset=_Dataset(index + 1),
        )
        for index, path in enumerate(paths)
    }
    for path in paths:
        path.write_bytes(path.name.encode())
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
    calls: list[int] = []

    def mapping(
        _reference: object,
        _secondary: object,
        bounds: tuple[int, int, int, int],
        **_kwargs: object,
    ) -> object:
        calls.append(1)
        return _identity_dense_mapping(bounds)

    monkeypatch.setattr(
        "faninsar.missions.nisar.provider._dense_secondary_mapping", mapping
    )
    monkeypatch.setattr(
        "faninsar.missions.nisar.provider._lanczos_source_coverage",
        lambda _samples, azimuth, _range_index: np.ones(azimuth.shape, dtype=bool),
    )
    work_dir = tmp_path / "work"
    stack = NISARStack.from_rslc(
        paths,
        work_dir=work_dir,
        dem=ConstantDEM(0.0),
        multilook=(1, 1),
        goldstein_alpha=0.0,
        coregistration_grid="radar",
        extra={"nisar_tile_shape": (2, 3)},
    )
    stack.prepare_data()
    stack.coregister_scenes()
    assert (work_dir / "unit-checkpoints" / "nisar").is_dir()
    assert len(calls) == 4

    reopened = Stack.open(work_dir)
    calls.clear()
    reopened.scene_provider(
        paths[0],
        paths[1],
        output_dir=work_dir / "coreg" / "20240113",
        options={"coregistration_grid": "radar", "device": "cpu"},
    )
    assert calls == []
