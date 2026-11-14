"""Pair-owned IFG products, ion placement, and sibling lifetime (P0080)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.io.storage.ifg_store import (
    InterferogramArtifactStore,
    complex_ion_root,
    ifg_artifact_dir,
    unwrap_ion_root,
    unwrap_root,
    write_complex_ion_artifact,
    write_ifg_artifact,
    write_unwrapped_artifact,
)
from faninsar.io.storage.ion_store import (
    IonosphereArtifactStore,
    ion_dir,
    write_ionosphere_artifact,
)
from faninsar.io.storage.product import open_product
from faninsar.processing.errors import InvalidProcessingStateError

if TYPE_CHECKING:
    from pathlib import Path

_DIGEST = "a" * 64
_PAIR = ("20240101", "20240113")
_PAIR_ID = "20240101_20240113"


def _ifg_arrays(shape: tuple[int, int] = (2, 2)) -> dict[str, np.ndarray]:
    return {
        "complex": np.full(shape, 2.0 * np.exp(1j * 0.3), dtype=np.complex64),
        "coherence": np.full(shape, 0.8, dtype=np.float32),
        "valid_mask": np.ones(shape, dtype=bool),
    }


def _write_ifg(
    work: Path,
    *,
    looks: tuple[int, int] = (2, 2),
    shape: tuple[int, int] = (2, 2),
    look_origin: tuple[int, int] = (0, 0),
    row_origin: int = 0,
    col_origin: int = 0,
    replace_existing: bool = False,
) -> InterferogramArtifactStore:
    arrays = _ifg_arrays(shape)
    return write_ifg_artifact(
        ifg_artifact_dir(work, looks, _PAIR),
        pair=_PAIR,
        looks=looks,
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"primary": _DIGEST, "secondary": "b" * 64},
        complex_ifg=arrays["complex"],
        coherence=arrays["coherence"],
        valid_mask=arrays["valid_mask"],
        row_origin=row_origin,
        col_origin=col_origin,
        look_origin=look_origin,
        grid_shape=shape,
        replace_existing=replace_existing,
    )


def _write_ion(
    work: Path,
    *,
    looks: tuple[int, int] = (2, 2),
    shape: tuple[int, int] = (2, 2),
    native_origin: tuple[int, int] = (0, 0),
    phase: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    replace_existing: bool = False,
) -> IonosphereArtifactStore:
    screen = (
        np.full(shape, 0.1, dtype=np.float32)
        if phase is None
        else np.asarray(phase, dtype=np.float32)
    )
    return write_ionosphere_artifact(
        ion_dir(work, _PAIR_ID, looks),
        pair=_PAIR,
        looks=looks,
        domain="radar",
        wavelength_m=0.236,
        grid_identity=_DIGEST,
        method_name="ionosphere_split_spectrum",
        method_parameters={"looks": list(looks)},
        degraded=False,
        degradation_reason=None,
        runtime_fingerprint=_DIGEST,
        devices=("cpu",),
        source_manifest_digests={"primary": _DIGEST, "secondary": "b" * 64},
        ionosphere_phase=screen,
        nondispersive_phase=np.zeros(shape, dtype=np.float32),
        weight=np.ones(shape, dtype=np.float32) if weight is None else weight,
        native_origin=native_origin,
        replace_existing=replace_existing,
    )


def _write_unwrap(
    store: InterferogramArtifactStore,
    *,
    source: str = "complex",
    phase: float = 0.3,
    replace_existing: bool = False,
) -> None:
    write_unwrapped_artifact(
        store.root,
        unwrapped_phase=np.full(store.shape, phase, dtype=np.float32),
        connected_components=np.full(store.shape, 1 if source == "complex" else 7, dtype=np.int32),
        method="SpatialIRLS",
        method_parameters={},
        ifg_manifest_digest=store.write_id,
        source=source,
        replace_existing=replace_existing,
    )


def test_ordinary_unwrap_and_ion_live_under_ifg(tmp_path: Path) -> None:
    """Ordinary unwrap is a pair sibling; ion estimates live under ifg/ion."""
    work = tmp_path / "work"
    store = _write_ifg(work)
    ion = _write_ion(work)
    _write_unwrap(store)

    pair_root = ifg_artifact_dir(work, (2, 2), _PAIR)
    assert unwrap_root(pair_root) == pair_root / "unwrap"
    assert complex_ion_root(pair_root) == pair_root / "complex_ion"
    assert unwrap_ion_root(pair_root) == pair_root / "unwrap_ion"
    assert ion_dir(work, _PAIR_ID, (2, 2)) == work / "ifg" / "ion" / "ml_2x2" / _PAIR_ID
    assert (pair_root / "unwrap" / "item.json").is_file()
    assert ion.root == work / "ifg" / "ion" / "ml_2x2" / _PAIR_ID
    assert not (work / "unwrap").exists()
    assert not (work / "ion").exists()


def test_burst_unwrap_helper_is_pair_owned(tmp_path: Path) -> None:
    """Burst unwrap is a sibling of the burst IFG when explicitly allowed."""
    work = tmp_path / "work"
    burst_root = ifg_artifact_dir(
        work, (1, 1), _PAIR, representation="burst", burst_id="IW1_b0"
    )
    write_ifg_artifact(
        burst_root,
        pair=_PAIR,
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"primary": _DIGEST, "secondary": "b" * 64},
        complex_ifg=np.ones((1, 2), dtype=np.complex64),
        representation="burst",
        burst_id="IW1_b0",
        row_origin=4,
        col_origin=0,
        grid_shape=(8, 2),
    )
    with pytest.raises(InvalidProcessingStateError, match="allow_burst"):
        unwrap_root(burst_root)
    assert unwrap_root(burst_root, allow_burst=True) == burst_root / "unwrap"


def test_ion_grouping_directory_is_not_an_ifg_collection(tmp_path: Path) -> None:
    """ifg/ion is reserved for estimates and is skipped by ml_* IFG discovery."""
    work = tmp_path / "work"
    _write_ifg(work)
    _write_ion(work)
    ifg_root = work / "ifg"
    looks = sorted(path.name for path in ifg_root.glob("ml_*"))
    assert looks == ["ml_2x2"]
    assert (ifg_root / "ion").is_dir()
    with pytest.raises(InvalidProcessingStateError):
        InterferogramArtifactStore.open(ifg_root / "ion" / "ml_2x2" / _PAIR_ID)


def test_ion_persists_and_round_trips_sample_placement(tmp_path: Path) -> None:
    """Ion products record explicit native-pixel anchors and spacing."""
    store = _write_ion(tmp_path / "work", looks=(4, 2), shape=(3, 5), native_origin=(6, 1))
    assert store.sample_center_anchor == (6.5, 2.5)
    assert store.sample_spacing == (2.0, 4.0)
    item = json.loads((store.root / "item.json").read_text())
    assert item["properties"]["faninsar:sample_center_anchor"] == [6.5, 2.5]
    assert item["properties"]["faninsar:sample_spacing"] == [2.0, 4.0]
    reopened = IonosphereArtifactStore.open(store.root)
    assert reopened.sample_center_anchor == store.sample_center_anchor
    assert reopened.sample_spacing == store.sample_spacing


def test_ion_open_rejects_missing_placement(tmp_path: Path) -> None:
    """Interpolation cannot consume an ion product without placement metadata."""
    store = _write_ion(tmp_path / "work")
    path = store.root / "item.json"
    item = json.loads(path.read_text())
    del item["properties"]["faninsar:sample_center_anchor"]
    path.write_text(json.dumps(item))
    with pytest.raises(InvalidProcessingStateError, match="sample_center_anchor"):
        IonosphereArtifactStore.open(store.root)


def test_complex_ion_commits_before_unwrap_and_preserves_amplitude(
    tmp_path: Path,
) -> None:
    """Corrected complex IFG is a sibling product storing complex.zarr."""
    work = tmp_path / "work"
    ifg = _write_ifg(work, looks=(1, 1), shape=(2, 2))
    ion = _write_ion(work, looks=(1, 1), shape=(2, 2))
    original = ifg.read().complex
    phi = ion.read().ionosphere_phase
    corrected = original * np.exp(-1j * phi)
    product = write_complex_ion_artifact(
        ifg.root,
        complex_ifg=corrected,
        valid_mask=np.ones(ifg.shape, dtype=bool),
        ion_write_id=ion.write_id,
        ion_looks=ion.looks,
        interpolation="masked_bilinear_unwrapped",
        source_anchor=ion.sample_center_anchor,
        source_spacing=ion.sample_spacing,
        target_anchor=(0.0, 0.0),
        target_spacing=(1.0, 1.0),
    )
    assert product.path == ifg.root / "complex_ion"
    assert product.kind == "complex_ion"
    assert product.inputs == {"ifg": ifg.write_id, "ion": ion.write_id}
    assert sorted(path.name for path in product.path.iterdir() if path.name != "item.json") == [
        "complex.zarr",
        "valid_mask.zarr",
    ]
    stored = open_product(product.path, kind="complex_ion").read("complex")
    np.testing.assert_allclose(np.abs(stored), np.abs(original), atol=1e-6)
    np.testing.assert_allclose(np.angle(stored), np.angle(original) - phi, atol=1e-6)
    assert not (ifg.root / "unwrap").exists()


def test_sibling_overwrite_independence_and_shared_ion_survival(
    tmp_path: Path,
) -> None:
    """Replacement follows the W3 sibling lifetime matrix."""
    work = tmp_path / "work"
    fine = _write_ifg(work, looks=(1, 1), shape=(4, 4))
    coarse = _write_ifg(work, looks=(2, 2), shape=(2, 2))
    ion = _write_ion(work, looks=(2, 2), shape=(2, 2))
    _write_unwrap(fine, phase=0.4)
    fine_ion = write_complex_ion_artifact(
        fine.root,
        complex_ifg=fine.read().complex * np.exp(-1j * 0.1),
        valid_mask=np.ones(fine.shape, dtype=bool),
        ion_write_id=ion.write_id,
        ion_looks=ion.looks,
        interpolation="masked_bilinear_unwrapped",
        source_anchor=ion.sample_center_anchor,
        source_spacing=ion.sample_spacing,
        target_anchor=(0.0, 0.0),
        target_spacing=(1.0, 1.0),
    )
    coarse_ion = write_complex_ion_artifact(
        coarse.root,
        complex_ifg=coarse.read().complex * np.exp(-1j * 0.1),
        valid_mask=np.ones(coarse.shape, dtype=bool),
        ion_write_id=ion.write_id,
        ion_looks=ion.looks,
        interpolation="masked_bilinear_unwrapped",
        source_anchor=ion.sample_center_anchor,
        source_spacing=ion.sample_spacing,
        target_anchor=(0.5, 0.5),
        target_spacing=(2.0, 2.0),
    )
    _write_unwrap(fine, source="complex_ion", phase=0.2)
    ordinary_id = fine.unwrap_product().write_id
    ion_unwrap_id = open_product(unwrap_ion_root(fine.root), kind="unwrap_ion").write_id

    _write_unwrap(fine, phase=0.9, replace_existing=True)
    assert fine.unwrap_product().write_id != ordinary_id
    assert open_product(fine_ion.path, kind="complex_ion").write_id == fine_ion.write_id
    assert open_product(unwrap_ion_root(fine.root), kind="unwrap_ion").write_id == ion_unwrap_id

    replaced_complex = write_complex_ion_artifact(
        fine.root,
        complex_ifg=fine.read().complex * np.exp(-1j * 0.2),
        valid_mask=np.ones(fine.shape, dtype=bool),
        ion_write_id=ion.write_id,
        ion_looks=ion.looks,
        interpolation="masked_bilinear_unwrapped",
        source_anchor=ion.sample_center_anchor,
        source_spacing=ion.sample_spacing,
        target_anchor=(0.0, 0.0),
        target_spacing=(1.0, 1.0),
        replace_existing=True,
    )
    assert replaced_complex.write_id != fine_ion.write_id
    with pytest.raises(InvalidProcessingStateError, match="complex_ion"):
        fine.read_unwrapped(source="complex_ion")
    assert fine.has_unwrapped()

    new_ion = _write_ion(work, looks=(2, 2), shape=(2, 2), replace_existing=True)
    assert new_ion.write_id != ion.write_id
    with pytest.raises(InvalidProcessingStateError, match="ion"):
        InterferogramArtifactStore.open(fine.root).read_layer("complex_ion")
    with pytest.raises(InvalidProcessingStateError, match="ion"):
        InterferogramArtifactStore.open(coarse.root).read_layer("complex_ion")
    assert coarse_ion.path.is_dir()
    assert fine.has_unwrapped()

    parent_id = fine.write_id
    replaced = _write_ifg(work, looks=(1, 1), shape=(4, 4), replace_existing=True)
    assert replaced.write_id != parent_id
    assert not (replaced.root / "unwrap").exists()
    assert not (replaced.root / "complex_ion").exists()
    assert not (replaced.root / "unwrap_ion").exists()
    assert IonosphereArtifactStore.open(new_ion.root).write_id == new_ion.write_id
