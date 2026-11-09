"""Shared branch-aware IFG layer resolution (PROPOSAL-0080 W2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.io.storage.ifg_store import (
    InterferogramArtifactStore,
    resolve_ifg_layer,
    write_complex_ion_artifact,
    write_ifg_artifact,
    write_unwrapped_artifact,
)
from faninsar.io.storage.ion_store import ion_dir, write_ionosphere_artifact
from faninsar.network.layers import InterferogramLayers
from faninsar.io.storage.product import open_product
from faninsar.processing.errors import InvalidProcessingStateError

if TYPE_CHECKING:
    from pathlib import Path

_DIGEST = "a" * 64
_PAIR = ("20240101", "20240113")
_PAIR_ID = "20240101_20240113"


def _commit_branches(tmp_path: Path) -> InterferogramArtifactStore:
    """Commit distinguishable ordinary and ion-corrected pair products."""
    root = tmp_path / "work" / "ifg" / "ml_1x1" / _PAIR_ID
    ordinary = np.full((2, 2), 3.0 + 0.0j, dtype=np.complex64)
    corrected = np.full((2, 2), 1.5 * np.exp(1j * -0.4), dtype=np.complex64)
    valid_ordinary = np.array([[True, True], [True, False]])
    valid_ion = np.array([[True, False], [True, True]])
    store = write_ifg_artifact(
        root,
        pair=_PAIR,
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"primary": _DIGEST, "secondary": "b" * 64},
        complex_ifg=ordinary,
        coherence=np.full((2, 2), 0.6, dtype=np.float32),
        valid_mask=valid_ordinary,
    )
    ion = write_ionosphere_artifact(
        ion_dir(tmp_path / "work", _PAIR_ID, (1, 1)),
        pair=_PAIR,
        looks=(1, 1),
        domain="radar",
        wavelength_m=0.236,
        grid_identity=_DIGEST,
        method_name="ionosphere_split_spectrum",
        method_parameters={"looks": [1, 1]},
        degraded=False,
        degradation_reason=None,
        runtime_fingerprint=_DIGEST,
        devices=("cpu",),
        source_manifest_digests={"primary": _DIGEST, "secondary": "b" * 64},
        ionosphere_phase=np.full((2, 2), 0.4, dtype=np.float32),
        nondispersive_phase=np.zeros((2, 2), dtype=np.float32),
        weight=np.ones((2, 2), dtype=np.float32),
        native_origin=(0, 0),
    )
    write_complex_ion_artifact(
        store.root,
        complex_ifg=corrected,
        valid_mask=valid_ion,
        ion_write_id=ion.write_id,
        ion_looks=(1, 1),
        interpolation="masked_bilinear_unwrapped",
        source_anchor=ion.sample_center_anchor,
        source_spacing=ion.sample_spacing,
        target_anchor=(0.0, 0.0),
        target_spacing=(1.0, 1.0),
    )
    write_unwrapped_artifact(
        store.root,
        unwrapped_phase=np.full((2, 2), 0.11, dtype=np.float32),
        connected_components=np.full((2, 2), 1, dtype=np.int32),
        method="SpatialIRLS",
        method_parameters={},
        ifg_manifest_digest=store.write_id,
        source="complex",
    )
    write_unwrapped_artifact(
        store.root,
        unwrapped_phase=np.full((2, 2), -0.4, dtype=np.float32),
        connected_components=np.full((2, 2), 9, dtype=np.int32),
        method="SpatialIRLS",
        method_parameters={},
        ifg_manifest_digest=store.write_id,
        source="complex_ion",
    )
    return InterferogramArtifactStore.open(store.root)


def test_resolver_maps_public_names_to_branch_products(tmp_path: Path) -> None:
    """One seam maps ordinary and ion names onto product, asset, and derivation."""
    store = _commit_branches(tmp_path)
    complex_ion = resolve_ifg_layer(store.root, "complex_ion")
    wrapped_ion = resolve_ifg_layer(store.root, "wrapped_ion")
    unwrapped_ion = resolve_ifg_layer(store.root, "unwrapped_ion")
    coherence = resolve_ifg_layer(store.root, "coherence")
    assert complex_ion.product_root == store.root / "complex_ion"
    assert complex_ion.real_asset == "complex"
    assert complex_ion.derivation == "identity"
    assert wrapped_ion.product_root == store.root / "complex_ion"
    assert wrapped_ion.derivation == "angle"
    assert unwrapped_ion.product_root == store.root / "unwrap_ion"
    assert unwrapped_ion.real_asset == "unwrapped_phase"
    assert coherence.product_root == store.root
    assert resolve_ifg_layer(store.root, "wrapped").real_asset == "complex"
    assert resolve_ifg_layer(store.root, "unwrapped").product_root == store.root / "unwrap"


def test_virtual_ion_layers_read_corrected_complex_and_write_nothing(
    tmp_path: Path,
) -> None:
    """wrapped_ion and amplitude_ion derive from complex_ion without payloads."""
    store = _commit_branches(tmp_path)
    wrapped = store.read_layer("wrapped_ion")
    amplitude = store.read_layer("amplitude_ion")
    np.testing.assert_allclose(wrapped[0, 0], -0.4, atol=1e-5)
    np.testing.assert_allclose(amplitude[0, 0], 1.5, atol=1e-5)
    assert np.isnan(wrapped[0, 1])
    ordinary = store.read_layer("wrapped")
    np.testing.assert_allclose(ordinary[0, 0], 0.0, atol=1e-5)
    names = sorted(path.name for path in (store.root / "complex_ion").iterdir())
    assert names == ["complex.zarr", "item.json", "valid_mask.zarr"]


def test_unwrapped_ion_components_do_not_follow_ordinary_unwrap(
    tmp_path: Path,
) -> None:
    """Selected unwrap validity and labels stay on their own product."""
    store = _commit_branches(tmp_path)
    layers = InterferogramLayers(
        open_product(store.root, kind="ifg"), store.shape, store.looks
    )
    np.testing.assert_array_equal(
        layers.connected_components.read(), np.full((2, 2), 1, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        layers.connected_components_ion.read(), np.full((2, 2), 9, dtype=np.int32)
    )
    np.testing.assert_allclose(layers.unwrapped_ion.read()[0, 0], -0.4)


def test_missing_named_branch_does_not_fall_back(tmp_path: Path) -> None:
    """An explicit ion layer raises when the sibling product is absent."""
    root = tmp_path / "work" / "ifg" / "ml_1x1" / _PAIR_ID
    store = write_ifg_artifact(
        root,
        pair=_PAIR,
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"primary": _DIGEST, "secondary": "b" * 64},
        complex_ifg=np.ones((2, 2), dtype=np.complex64),
    )
    with pytest.raises(InvalidProcessingStateError, match="complex_ion"):
        store.read_layer("complex_ion")
    with pytest.raises(InvalidProcessingStateError, match="unwrap_ion"):
        store.read_unwrapped(source="complex_ion")
    np.testing.assert_array_equal(store.read_layer("complex"), np.ones((2, 2), dtype=np.complex64))
