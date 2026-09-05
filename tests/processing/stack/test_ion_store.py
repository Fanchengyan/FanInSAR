"""Tests for the transactional ionosphere artifact stores (PROPOSAL-0036)."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.stack.ifg_store import (
    InterferogramArtifactStore,
    write_ifg_artifact,
)
from faninsar.stack.ion_store import (
    IonosphereArtifactStore,
    read_ion_correction_artifact,
    write_ion_correction_artifact,
    write_ionosphere_artifact,
)

_DIGEST = "a" * 64


def _expected_screen() -> np.ndarray:
    return np.linspace(-0.1, 0.1, 16, dtype=np.float32).reshape(4, 4)


def _write(
    root,
    *,
    degraded: bool = False,
    degradation_reason: str | None = None,
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
    )


def _write_ifg_pair(root) -> str:
    """Publish one synthetic IFG pair artifact; return its manifest digest."""
    phase = np.full((4, 4), 0.3, dtype=np.float32)
    write_ifg_artifact(
        root,
        pair=("20240101", "20240113"),
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"scenes": _DIGEST},
        complex_ifg=np.exp(1j * phase.astype(np.float64)),
        coherence=np.ones((4, 4), dtype=np.float32),
        wrapped_phase=phase,
        amplitude=np.ones((4, 4), dtype=np.float32),
    )
    store = InterferogramArtifactStore.open(root)
    try:
        return store.manifest_digest
    finally:
        store.close()


def test_ion_artifact_roundtrip(tmp_path) -> None:
    root = tmp_path / "ion" / "ml_1x1"
    store = _write(root)
    try:
        assert store.pair == ("20240101", "20240113")
        assert store.looks == (1, 1)
        assert store.degraded is False
        assert store.degradation_reason is None
        assert store.runtime_fingerprint == _DIGEST
        assert store.devices == ("cpu",)
        assert store.method_name == "ionosphere_split_spectrum"
        assert store.method_parameters["f0_hz"] == 1257.5e6
        artifact = store.read()
        np.testing.assert_allclose(artifact.ionosphere_phase, _expected_screen(), atol=1e-7)
        np.testing.assert_allclose(artifact.weight, 1.0)
    finally:
        store.close()
    with pytest.raises(InvalidProcessingStateError, match="already published"):
        _write(root)


def test_degraded_metadata_contract(tmp_path) -> None:
    with pytest.raises(InvalidProcessingStateError, match="degradation reason"):
        _write(tmp_path / "a", degraded=True)
    with pytest.raises(InvalidProcessingStateError, match="degradation reason"):
        _write(tmp_path / "b", degradation_reason="narrow-band mode")
    store = _write(tmp_path / "c", degraded=True, degradation_reason="narrow-band mode")
    try:
        assert store.degraded is True
        assert store.degradation_reason == "narrow-band mode"
    finally:
        store.close()


def test_payload_tampering_is_rejected(tmp_path) -> None:
    root = tmp_path / "ion"
    store = _write(root)
    store.close()
    target = next(root.rglob("weight.npy"))
    target.write_bytes(target.read_bytes()[:-1] + b"\x00")
    reopened = IonosphereArtifactStore.open(root)
    try:
        with pytest.raises(InvalidProcessingStateError, match="digest mismatch"):
            reopened.read()
    finally:
        reopened.close()


def test_ion_correction_roundtrip_binding_and_republish(tmp_path) -> None:
    ifg_root = tmp_path / "pair"
    ifg_digest = _write_ifg_pair(ifg_root)
    corrected = np.full((4, 4), -0.05, dtype=np.float32)
    with pytest.raises(InvalidProcessingStateError):
        write_ion_correction_artifact(
            ifg_root,
            unwrapped_phase=corrected,
            ion_manifest_digest=_DIGEST,
            ifg_manifest_digest="b" * 64,
            degraded_consumed=False,
        )
    write_ion_correction_artifact(
        ifg_root,
        unwrapped_phase=corrected,
        ion_manifest_digest=_DIGEST,
        ifg_manifest_digest=ifg_digest,
        degraded_consumed=False,
    )
    np.testing.assert_allclose(read_ion_correction_artifact(ifg_root), corrected, atol=1e-7)
    with pytest.raises(InvalidProcessingStateError, match="already published"):
        write_ion_correction_artifact(
            ifg_root,
            unwrapped_phase=corrected,
            ion_manifest_digest=_DIGEST,
            ifg_manifest_digest=ifg_digest,
            degraded_consumed=False,
        )
