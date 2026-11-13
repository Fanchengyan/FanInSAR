"""Reference/secondary unit-checkpoint dependencies for P0088 coregistration.

These tests use the existing tiny deterministic S1 coregistration fixture and
real committed checkpoint products. They assert consumed arrays and local
publication bindings rather than only that a checkpoint helper was called.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from faninsar.data.query import BoundingBox, Polygons
from faninsar.missions.s1 import coregister as coreg_module
from faninsar.missions.s1.coregister import (
    coregister_s1_secondary,
    prepare_s1_reference,
)
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry import ConstantDEM
from tests.missions.s1.test_coregister import (
    _COVERING_ROI,
    REFERENCE,
    SECONDARIES,
    _fake_scene,
    _install_fake_s1,
    _multi_swath_product,
    _prepare,
)


def _checkpoint_item(root: Path, pattern: str) -> dict[str, object]:
    """Return the single committed checkpoint item matching ``pattern``."""
    matches = list(root.glob(pattern))
    assert len(matches) == 1, matches
    return json.loads((matches[0] / "item.json").read_text())


def _first_payload_chunk(target: Path, payload: str = "slc.zarr") -> Path:
    """Return one stored chunk file of the named committed payload."""
    return next(
        path
        for path in (target / payload).rglob("*")
        if path.is_file() and path.name != "zarr.json"
    )


def _republish_as_legacy_sparse_fill(path: Path, *, shape: tuple[int, int]) -> None:
    """Rewrite a boolean payload as an older writer: an all-fill chunk is absent."""
    import zarr

    shutil.rmtree(path)
    chunks = tuple(max(1, min(int(size), 256)) for size in shape)
    array = zarr.create_array(
        store=str(path),
        shape=shape,
        chunks=chunks,
        dtype=np.bool_,
        fill_value=True,
    )
    array[...] = np.ones(shape, dtype=bool)


def test_reference_and_secondary_unit_replay_is_payload_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged local publications replay the exact numerical unit payloads."""
    recorder = _install_fake_s1(tmp_path, monkeypatch)
    checkpoint_root = tmp_path / "unit-checkpoints"
    context = _prepare(tmp_path, checkpoint_root=checkpoint_root)

    unit = context.units["f0_IW1_b0"]
    assert unit.publication_id is not None
    assert unit.specification_id is not None
    reference_item = json.loads((context.product.root / "item.json").read_text())
    assert reference_item["properties"]["faninsar:reference_unit_publications"] == {
        "f0_IW1_b0": unit.publication_id
    }
    reference_checkpoint = _checkpoint_item(checkpoint_root, "reference*_IW1_b0_*")

    output = tmp_path / "coreg" / "20200113"
    first = coregister_s1_secondary(
        context,
        tmp_path / SECONDARIES[0],
        output_scene=output,
        misreg_az_px=0.01,
        misreg_rg_px=-0.02,
        checkpoint_root=checkpoint_root,
    )
    first_payload = first.read("f0_IW1_b0")[0]
    secondary_checkpoint = _checkpoint_item(checkpoint_root, "20200113*_IW1_b0_*")
    secondary_specification = secondary_checkpoint["properties"][
        "faninsar:specification"
    ]
    assert secondary_specification["reference"]["unit_publication_id"] == (
        unit.publication_id
    )
    assert secondary_specification["reference"]["unit_specification_id"] == (
        unit.specification_id
    )
    assert secondary_specification["reference"]["support_key"] == unit.support_key
    assert secondary_specification["secondary"]["source"]["product_id"] == (
        Path(SECONDARIES[0]).stem
    )
    assert secondary_specification["correction"] == {
        "misreg_az_px": 0.01,
        "misreg_rg_px": -0.02,
    }

    original_prepare = coreg_module._prepare_radar_unit

    def no_reference_compute(unit: object, **kwargs: object) -> object:
        if kwargs.get("compute"):
            pytest.fail("recomputed a committed Reference unit")
        return original_prepare(unit, **kwargs)

    def no_secondary_compute(*_args: object, **_kwargs: object) -> object:
        pytest.fail("recomputed a committed secondary unit")

    monkeypatch.setattr(coreg_module, "_prepare_radar_unit", no_reference_compute)
    monkeypatch.setattr(coreg_module, "_coregister_unit", no_secondary_compute)
    reads_before = recorder.reference_reads
    reopened = _prepare(tmp_path, checkpoint_root=checkpoint_root)
    assert recorder.reference_reads == reads_before
    assert reopened.product.write_id == context.product.write_id
    assert reopened.units["f0_IW1_b0"].publication_id == unit.publication_id
    second = coregister_s1_secondary(
        reopened,
        tmp_path / SECONDARIES[0],
        output_scene=output,
        misreg_az_px=0.01,
        misreg_rg_px=-0.02,
        overwrite=True,
        checkpoint_root=checkpoint_root,
    )
    np.testing.assert_array_equal(first_payload, second.read("f0_IW1_b0")[0])
    assert second.units[0].burst_id == context.product.units[0].burst_id
    assert reference_checkpoint["properties"]["faninsar:specification"]["support_key"]


def test_changed_reference_support_does_not_replay_the_old_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed ROI window creates a distinct local unit, never a stale replay."""
    recorder = _install_fake_s1(tmp_path, monkeypatch)
    window = [(0, 8, 0, 8)]
    monkeypatch.setattr(
        coreg_module, "_roi_burst_window", lambda *_args, **_kwargs: window[0]
    )
    monkeypatch.setattr(
        coreg_module, "reference_window_crop", lambda crop, _shape: crop
    )
    checkpoint_root = tmp_path / "unit-checkpoints"
    first = _prepare(
        tmp_path,
        swaths=("IW1",),
        bursts={"IW1": [0]},
        roi=_COVERING_ROI,
        checkpoint_root=checkpoint_root,
    )
    first_id = first.units["f0_IW1_b0"].publication_id
    assert first.product.read("f0_IW1_b0")[0].shape == (8, 8)

    window[0] = (2, 6, 2, 6)
    second = _prepare(
        tmp_path,
        swaths=("IW1",),
        bursts={"IW1": [0]},
        roi=_COVERING_ROI,
        checkpoint_root=checkpoint_root,
    )
    second_id = second.units["f0_IW1_b0"].publication_id
    assert second_id != first_id
    assert second.product.read("f0_IW1_b0")[0].shape == (4, 4)
    assert recorder.reference_reads == 2
    assert len(list(checkpoint_root.glob("reference*_IW1_b0_*"))) == 2

    original_prepare = coreg_module._prepare_radar_unit

    def no_reference_compute(unit: object, **kwargs: object) -> object:
        if kwargs.get("compute"):
            pytest.fail("recomputed the unchanged second ROI unit")
        return original_prepare(unit, **kwargs)

    monkeypatch.setattr(coreg_module, "_prepare_radar_unit", no_reference_compute)
    third = _prepare(
        tmp_path,
        swaths=("IW1",),
        bursts={"IW1": [0]},
        roi=_COVERING_ROI,
        checkpoint_root=checkpoint_root,
    )
    assert recorder.reference_reads == 2
    assert third.units["f0_IW1_b0"].publication_id == second_id


def test_changed_orbit_or_source_invalidates_the_reference_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Orbit stat and physical product identity participate in local reuse."""
    recorder = _install_fake_s1(tmp_path, monkeypatch)
    checkpoint_root = tmp_path / "unit-checkpoints"
    orbit = tmp_path / "reference.eof"
    orbit.write_text("a")
    first = prepare_s1_reference(
        tmp_path / REFERENCE,
        output_scene=tmp_path / "coreg" / "20200101",
        dem=ConstantDEM(0.0),
        orbit_path=orbit,
        checkpoint_root=checkpoint_root,
    )
    first_id = first.units["f0_IW1_b0"].publication_id

    orbit.write_text("bb")
    second = prepare_s1_reference(
        tmp_path / REFERENCE,
        output_scene=tmp_path / "coreg" / "20200101",
        dem=ConstantDEM(0.0),
        orbit_path=orbit,
        checkpoint_root=checkpoint_root,
    )
    assert second.units["f0_IW1_b0"].publication_id != first_id
    assert recorder.reference_reads == 2

    base_open = coreg_module.open_acquisition_products

    def changed_source(paths: object, **kwargs: object) -> object:
        products = base_open(paths, **kwargs)
        return [
            replace(product, product_id=f"{product.product_id}_moved")
            for product in products
        ]

    monkeypatch.setattr(coreg_module, "open_acquisition_products", changed_source)
    third = prepare_s1_reference(
        tmp_path / REFERENCE,
        output_scene=tmp_path / "coreg" / "20200101",
        dem=ConstantDEM(0.0),
        orbit_path=orbit,
        checkpoint_root=checkpoint_root,
    )
    assert (
        third.units["f0_IW1_b0"].publication_id
        != second.units["f0_IW1_b0"].publication_id
    )
    assert recorder.reference_reads == 3


def test_strict_verification_never_reuses_an_unbaselined_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strict mode recomputes when the committed unit has no baseline."""
    recorder = _install_fake_s1(tmp_path, monkeypatch)
    checkpoint_root = tmp_path / "unit-checkpoints"
    integrity_root = tmp_path / "integrity"
    first = _prepare(
        tmp_path,
        checkpoint_root=checkpoint_root,
        verification="strict",
        integrity_root=integrity_root,
    )
    first_id = first.units["f0_IW1_b0"].publication_id
    assert list(integrity_root.rglob("*.json"))

    second = _prepare(
        tmp_path,
        checkpoint_root=checkpoint_root,
        verification="strict",
        integrity_root=integrity_root,
    )
    assert second.units["f0_IW1_b0"].publication_id == first_id
    assert recorder.reference_reads == 1

    missing_baselines = tmp_path / "integrity-empty"
    third = _prepare(
        tmp_path,
        checkpoint_root=checkpoint_root,
        verification="strict",
        integrity_root=missing_baselines,
    )
    assert third.units["f0_IW1_b0"].publication_id != first_id
    assert recorder.reference_reads == 2
    assert list(missing_baselines.rglob("*.json"))


def test_strict_verification_requires_both_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strict policy is not silently accepted as an ignored argument."""
    _install_fake_s1(tmp_path, monkeypatch)
    with pytest.raises(InvalidProcessingStateError, match="integrity_root"):
        _prepare(
            tmp_path,
            checkpoint_root=tmp_path / "unit-checkpoints",
            verification="strict",
        )


def test_appending_a_reference_burst_reuses_unchanged_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new sibling burst does not invalidate an already committed burst."""
    recorder = _install_fake_s1(tmp_path, monkeypatch)
    product = _multi_swath_product(tmp_path / REFERENCE)

    monkeypatch.setattr(
        coreg_module,
        "open_acquisition_products",
        lambda paths, **_kwargs: [product for _ in paths],
    )

    def load_multi(path: Path, **kwargs: object) -> object:
        if Path(path).name != REFERENCE:
            return _fake_scene(Path(path), "20200113")
        if kwargs.get("read_samples", True):
            recorder.reference_reads += 1
        scene = _fake_scene(Path(path), "20200101")
        swath = product.find_swath(str(kwargs.get("swath", "IW1")))
        assert swath is not None
        scene.swath = swath
        scene.burst = swath.bursts[int(kwargs.get("burst_index", 0))]
        return scene

    monkeypatch.setattr(coreg_module, "load_production_scene", load_multi)
    checkpoint_root = tmp_path / "unit-checkpoints"
    first = _prepare(
        tmp_path,
        swaths=("IW1",),
        bursts={"IW1": [0]},
        checkpoint_root=checkpoint_root,
    )
    first_unit = first.units["f0_IW1_b0"]
    assert first.grid_shape == (8, 8)
    first_payload = first.product.read("f0_IW1_b0")[0]
    first_reads = recorder.reference_reads

    original_prepare = coreg_module._prepare_radar_unit

    def guarded(unit: object, **kwargs: object) -> object:
        if kwargs.get("compute") and getattr(unit, "tag", "") == "f0_IW1_b0":
            pytest.fail("recomputed unchanged reference burst b0")
        return original_prepare(unit, **kwargs)

    monkeypatch.setattr(coreg_module, "_prepare_radar_unit", guarded)
    second = _prepare(
        tmp_path,
        swaths=("IW1",),
        bursts={"IW1": [0, 1]},
        checkpoint_root=checkpoint_root,
    )
    assert second.grid_shape != first.grid_shape
    assert second.grid_shape[1] == first.grid_shape[1]
    assert set(second.units) == {"f0_IW1_b0", "f0_IW1_b1"}
    assert second.units["f0_IW1_b0"].publication_id == first_unit.publication_id
    assert (
        second.units["f0_IW1_b1"].publication_id
        != second.units["f0_IW1_b0"].publication_id
    )
    np.testing.assert_array_equal(first_payload, second.product.read("f0_IW1_b0")[0])
    assert recorder.reference_reads == first_reads + 1
    assert len(list(checkpoint_root.glob("reference*_IW1_b0_*"))) == 1
    assert len(list(checkpoint_root.glob("reference*_IW1_b1_*"))) == 1


def test_disjoint_roi_addition_recomputes_conservatively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Radar keeps the full canonical ROI; any ROI change safely recomputes."""
    recorder = _install_fake_s1(tmp_path, monkeypatch)
    monkeypatch.setattr(
        coreg_module, "_roi_burst_window", lambda *_args, **_kwargs: (0, 8, 0, 8)
    )
    monkeypatch.setattr(
        coreg_module, "reference_window_crop", lambda crop, _shape: crop
    )
    checkpoint_root = tmp_path / "unit-checkpoints"
    first = _prepare(
        tmp_path,
        swaths=("IW1",),
        bursts={"IW1": [0]},
        roi=_COVERING_ROI,
        checkpoint_root=checkpoint_root,
    )
    first_id = first.units["f0_IW1_b0"].publication_id

    union = Polygons(
        gpd.GeoDataFrame(
            {
                "geometry": [
                    box(9.99, 44.99, 10.08, 45.08),
                    box(20.0, 20.0, 20.1, 20.1),
                ]
            },
            crs="EPSG:4326",
        )
    )
    second = _prepare(
        tmp_path,
        swaths=("IW1",),
        bursts={"IW1": [0]},
        roi=union,
        checkpoint_root=checkpoint_root,
    )
    assert second.units["f0_IW1_b0"].publication_id != first_id
    assert recorder.reference_reads == 2
    assert len(list(checkpoint_root.glob("reference*_IW1_b0_*"))) == 2


def test_radar_support_key_uses_full_canonical_roi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nonlinear rdr2geo edge cannot hide behind a sampled radar hull."""
    _install_fake_s1(tmp_path, monkeypatch)
    context = _prepare(tmp_path)
    unit = context.units["f0_IW1_b0"]
    narrow = BoundingBox(-1, -1, 5, 4.1)
    extended = BoundingBox(-1, -1, 5, 4.6)
    first = coreg_module._reference_support_metadata(
        unit,
        coregistration_grid="radar",
        roi=narrow,
        geo_grid=None,
        geo_grid_identity=None,
    )
    second = coreg_module._reference_support_metadata(
        unit,
        coregistration_grid="radar",
        roi=extended,
        geo_grid=None,
        geo_grid_identity=None,
    )
    assert first["roi"] != second["roi"]
    assert coreg_module._support_key(first) != coreg_module._support_key(second)


def test_corrupt_secondary_checkpoint_chunk_recomputes_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A truncated committed secondary unit is rebuilt at its own workdir."""
    _install_fake_s1(tmp_path, monkeypatch)
    checkpoint_root = tmp_path / "unit-checkpoints"
    context = _prepare(tmp_path, checkpoint_root=checkpoint_root)
    output = tmp_path / "coreg" / "20200113"
    first = coregister_s1_secondary(
        context,
        tmp_path / SECONDARIES[0],
        output_scene=output,
        misreg_az_px=0.01,
        misreg_rg_px=-0.02,
        checkpoint_root=checkpoint_root,
    )
    payload = first.read("f0_IW1_b0")[0]
    (unit_dir,) = checkpoint_root.glob("20200113*_IW1_b0_*")
    original_item = (unit_dir / "item.json").read_bytes()
    _first_payload_chunk(unit_dir).unlink()

    original_coregister = coreg_module._coregister_unit
    recomputed: list[object] = []

    def spy(*args: object, **kwargs: object) -> object:
        recomputed.append(args)
        return original_coregister(*args, **kwargs)

    monkeypatch.setattr(coreg_module, "_coregister_unit", spy)
    caplog.set_level(logging.WARNING, logger="faninsar.missions.s1.coregister")
    caplog.clear()
    second = coregister_s1_secondary(
        context,
        tmp_path / SECONDARIES[0],
        output_scene=output,
        misreg_az_px=0.01,
        misreg_rg_px=-0.02,
        overwrite=True,
        checkpoint_root=checkpoint_root,
    )

    assert len(recomputed) == 1
    np.testing.assert_array_equal(second.read("f0_IW1_b0")[0], payload)
    assert list(checkpoint_root.glob("20200113*_IW1_b0_*")) == [unit_dir]
    assert (unit_dir / "item.json").is_file()
    assert (unit_dir / "item.json").read_bytes() != original_item
    assert any(
        unit_dir.name in record.getMessage() and "missing chunk" in record.getMessage()
        for record in caplog.records
    )


def test_legacy_sparse_fill_reference_recomputes_without_its_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An older sparse-fill publication is replaced without touching siblings."""
    recorder = _install_fake_s1(tmp_path, monkeypatch)
    product = _multi_swath_product(tmp_path / REFERENCE)

    monkeypatch.setattr(
        coreg_module,
        "open_acquisition_products",
        lambda paths, **_kwargs: [product for _ in paths],
    )

    def load_multi(path: Path, **kwargs: object) -> object:
        if Path(path).name != REFERENCE:
            return _fake_scene(Path(path), "20200113")
        if kwargs.get("read_samples", True):
            recorder.reference_reads += 1
        scene = _fake_scene(Path(path), "20200101")
        swath = product.find_swath(str(kwargs.get("swath", "IW1")))
        assert swath is not None
        scene.swath = swath
        scene.burst = swath.bursts[int(kwargs.get("burst_index", 0))]
        return scene

    monkeypatch.setattr(coreg_module, "load_production_scene", load_multi)
    checkpoint_root = tmp_path / "unit-checkpoints"
    first = _prepare(
        tmp_path,
        swaths=("IW1",),
        bursts={"IW1": [0, 1]},
        checkpoint_root=checkpoint_root,
    )
    b0_payload = first.product.read("f0_IW1_b0")[0]
    b1_payload = first.product.read("f0_IW1_b1")[0]
    b0 = first.units["f0_IW1_b0"]
    b1 = first.units["f0_IW1_b1"]
    reads = recorder.reference_reads
    (b0_dir,) = checkpoint_root.glob("reference*_IW1_b0_*")
    (b1_dir,) = checkpoint_root.glob("reference*_IW1_b1_*")
    _republish_as_legacy_sparse_fill(
        b0_dir / "valid.zarr",
        shape=(int(b0_payload.shape[0]), int(b0_payload.shape[1])),
    )

    caplog.set_level(logging.WARNING, logger="faninsar.missions.s1.coregister")
    caplog.clear()
    second = _prepare(
        tmp_path,
        swaths=("IW1",),
        bursts={"IW1": [0, 1]},
        checkpoint_root=checkpoint_root,
    )

    assert second.units["f0_IW1_b0"].publication_id != b0.publication_id
    assert second.units["f0_IW1_b1"].publication_id == b1.publication_id
    np.testing.assert_array_equal(second.product.read("f0_IW1_b0")[0], b0_payload)
    np.testing.assert_array_equal(second.product.read("f0_IW1_b1")[0], b1_payload)
    assert list(checkpoint_root.glob("reference*_IW1_b0_*")) == [b0_dir]
    assert list(checkpoint_root.glob("reference*_IW1_b1_*")) == [b1_dir]
    assert recorder.reference_reads == reads + 1
    assert any(
        b0_dir.name in record.getMessage() and "missing chunk" in record.getMessage()
        for record in caplog.records
    )


def test_foreign_product_at_a_unit_path_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovery never relabels or deletes a non-checkpoint product."""
    _install_fake_s1(tmp_path, monkeypatch)
    checkpoint_root = tmp_path / "unit-checkpoints"
    _prepare(tmp_path, checkpoint_root=checkpoint_root)
    (unit_dir,) = list(checkpoint_root.glob("reference*_IW1_b0_*"))
    item_path = unit_dir / "item.json"
    item = json.loads(item_path.read_text())
    item["properties"]["faninsar:kind"] = "scene"
    item_path.write_text(json.dumps(item))
    preserved = sorted(path.name for path in unit_dir.iterdir())

    with pytest.raises(InvalidProcessingStateError, match="existing product kind"):
        _prepare(tmp_path, checkpoint_root=checkpoint_root)

    assert json.loads(item_path.read_text())["properties"]["faninsar:kind"] == "scene"
    assert sorted(path.name for path in unit_dir.iterdir()) == preserved
