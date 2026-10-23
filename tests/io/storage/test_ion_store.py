"""Tests for the ionosphere products (PROPOSAL-0036, PROPOSAL-0059)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from faninsar.io.storage.ifg_store import (
    InterferogramArtifactStore,
    write_ifg_artifact,
    write_unwrapped_artifact,
)
from faninsar.io.storage.ion_store import (
    IonosphereArtifactStore,
    ion_dir,
    read_ion_correction_artifact,
    write_ion_correction_artifact,
    write_ionosphere_artifact,
)
from faninsar.processing.errors import InvalidProcessingStateError

_DIGEST = "a" * 64
_PAIR = "20240101_20240113"


def _expected_screen() -> np.ndarray:
    return np.linspace(-0.1, 0.1, 16, dtype=np.float32).reshape(4, 4)


def _write(
    root,
    *,
    degraded: bool = False,
    degradation_reason: str | None = None,
    replace_existing: bool = False,
) -> IonosphereArtifactStore:
    return write_ionosphere_artifact(
        root,
        pair=("20240101", "20240113"),
        looks=(1, 1),
        domain="radar",
        wavelength_m=0.236,
        grid_identity=_DIGEST,
        method_name="ionosphere_split_spectrum",
        method_parameters={"f0_hz": 1257.5e6, "looks": [1, 1]},
        degraded=degraded,
        degradation_reason=degradation_reason,
        runtime_fingerprint=_DIGEST,
        devices=("cpu",),
        source_manifest_digests={"primary": _DIGEST, "secondary": _DIGEST},
        ionosphere_phase=_expected_screen(),
        nondispersive_phase=np.zeros((4, 4), dtype=np.float32),
        weight=np.ones((4, 4), dtype=np.float32),
        replace_existing=replace_existing,
    )


def _unwrapped_ifg(work) -> InterferogramArtifactStore:
    """Commit one IFG and its unwrap product; return the IFG store."""
    phase = np.full((4, 4), 0.3, dtype=np.float32)
    store = write_ifg_artifact(
        work / "ifg" / "ml_1x1" / _PAIR,
        pair=("20240101", "20240113"),
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"primary": _DIGEST, "secondary": _DIGEST},
        complex_ifg=np.exp(1j * phase.astype(np.float64)),
        coherence=np.ones((4, 4), dtype=np.float32),
    )
    write_unwrapped_artifact(
        store.root,
        unwrapped_phase=phase,
        connected_components=np.zeros((4, 4), dtype=np.int32),
        method="SpatialIRLS",
        method_parameters={},
        ifg_manifest_digest=store.write_id,
    )
    return store


def test_ion_artifact_roundtrip(tmp_path) -> None:
    root = ion_dir(tmp_path, _PAIR, (1, 1))
    store = _write(root)
    assert store.pair == ("20240101", "20240113")
    assert store.looks == (1, 1)
    assert store.degraded is False
    assert store.degradation_reason is None
    assert store.runtime_fingerprint == _DIGEST
    assert store.devices == ("cpu",)
    assert store.method_name == "ionosphere_split_spectrum"
    assert store.method_parameters["f0_hz"] == 1257.5e6
    assert store.source_manifest_digests == {"primary": _DIGEST, "secondary": _DIGEST}
    artifact = store.read()
    np.testing.assert_allclose(artifact.ionosphere_phase, _expected_screen(), atol=1e-7)
    np.testing.assert_allclose(artifact.weight, 1.0)
    item = json.loads((root / "item.json").read_text())
    assert item["properties"]["faninsar:kind"] == "ion"
    with pytest.raises(InvalidProcessingStateError, match="already exists"):
        _write(root)


def test_degraded_metadata_contract(tmp_path) -> None:
    with pytest.raises(InvalidProcessingStateError, match="degradation reason"):
        _write(tmp_path / "a", degraded=True)
    with pytest.raises(InvalidProcessingStateError, match="degradation reason"):
        _write(tmp_path / "b", degradation_reason="narrow-band mode")
    store = _write(tmp_path / "c", degraded=True, degradation_reason="narrow-band mode")
    assert store.degraded is True
    assert store.degradation_reason == "narrow-band mode"


def test_ion_correction_is_a_child_of_the_unwrap_product(tmp_path) -> None:
    store = _unwrapped_ifg(tmp_path)
    ion = _write(ion_dir(tmp_path, _PAIR, (1, 1)))
    unwrap = store.unwrap_product()
    corrected = np.full((4, 4), -0.05, dtype=np.float32)
    with pytest.raises(InvalidProcessingStateError, match="binding"):
        write_ion_correction_artifact(
            store.root,
            unwrapped_phase=corrected,
            unwrap_write_id="0" * 26,
            ion_write_id=ion.write_id,
            ion_looks=(1, 1),
            degraded_consumed=False,
        )
    product = write_ion_correction_artifact(
        store.root,
        unwrapped_phase=corrected,
        unwrap_write_id=unwrap.write_id,
        ion_write_id=ion.write_id,
        ion_looks=(1, 1),
        degraded_consumed=False,
    )
    assert product.path == store.unwrap_root / "ion_corrected"
    assert product.inputs == {"unwrap": unwrap.write_id, "ion": ion.write_id}
    np.testing.assert_allclose(read_ion_correction_artifact(store.root), corrected)
    with pytest.raises(InvalidProcessingStateError, match="already exists"):
        write_ion_correction_artifact(
            store.root,
            unwrapped_phase=corrected,
            unwrap_write_id=unwrap.write_id,
            ion_write_id=ion.write_id,
            ion_looks=(1, 1),
            degraded_consumed=False,
        )


def test_replacing_the_unwrap_removes_its_correction(tmp_path) -> None:
    store = _unwrapped_ifg(tmp_path)
    ion = _write(ion_dir(tmp_path, _PAIR, (1, 1)))
    write_ion_correction_artifact(
        store.root,
        unwrapped_phase=np.zeros((4, 4), dtype=np.float32),
        unwrap_write_id=store.unwrap_product().write_id,
        ion_write_id=ion.write_id,
        ion_looks=(1, 1),
        degraded_consumed=False,
    )
    write_unwrapped_artifact(
        store.root,
        unwrapped_phase=np.ones((4, 4), dtype=np.float32),
        connected_components=np.zeros((4, 4), dtype=np.int32),
        method="SpatialIRLS",
        method_parameters={},
        ifg_manifest_digest=store.write_id,
        replace_existing=True,
    )
    assert not (store.unwrap_root / "ion_corrected").exists()
    with pytest.raises(InvalidProcessingStateError):
        read_ion_correction_artifact(store.root)
