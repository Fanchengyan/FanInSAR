"""Tests for Stack interferogram and unwrap products (PROPOSAL-0059)."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.io.storage.ifg_store import (
    IFG_KIND,
    InterferogramArtifactStore,
    unwrap_root,
    write_ifg_artifact,
    write_unwrapped_artifact,
)
from faninsar.io.storage.product import open_product, verify_product
from faninsar.network.layers import InterferogramLayers
from faninsar.processing.errors import InvalidProcessingStateError

if TYPE_CHECKING:
    from pathlib import Path

_SOURCES = {"primary": "a" * 64, "secondary": "b" * 64}


def _ifg_dir(tmp_path: Path) -> Path:
    return tmp_path / "work" / "ifg" / "ml_2x3" / "20240101_20240113"


def _ifg_arrays(shape: tuple[int, int] = (2, 3)) -> dict[str, np.ndarray]:
    """Return the persisted layers of one small interferogram product.

    ``wrapped_phase`` and ``amplitude`` are virtual layers and are never
    persisted, so the fixture only carries the canonical complex IFG and the
    optional coherence.
    """
    complex_ifg = np.full(shape, 1.0 + 2.0j, dtype=np.complex64)
    return {
        "complex_ifg": complex_ifg,
        "coherence": np.full(shape, 0.75, dtype=np.float32),
    }


def _write_base(root: Path, **overrides: object) -> InterferogramArtifactStore:
    """Commit one complete IFG product."""
    kwargs: dict[str, object] = {
        "pair": ("20240101", "20240113"),
        "looks": (2, 3),
        "filter_name": "goldstein",
        "filter_parameters": {"alpha": 0.5},
        "source_manifest_digests": dict(_SOURCES),
        **_ifg_arrays(),
    }
    kwargs.update(overrides)
    return write_ifg_artifact(root, **kwargs)  # type: ignore[arg-type]


def _unwrap(store: InterferogramArtifactStore, **overrides: object) -> None:
    kwargs: dict[str, object] = {
        "unwrapped_phase": np.arange(6, dtype=np.float32).reshape(2, 3),
        "connected_components": np.ones((2, 3), dtype=np.int32),
        "method": "snaphu",
        "method_parameters": {"cost_mode": "smooth"},
        "ifg_manifest_digest": store.write_id,
    }
    kwargs.update(overrides)
    write_unwrapped_artifact(store.root, **kwargs)  # type: ignore[arg-type]


def _layers(store: InterferogramArtifactStore) -> InterferogramLayers:
    """Return the flat per-Pair layer view of one committed IFG product."""
    return InterferogramLayers(
        open_product(store.root, kind=IFG_KIND), store.shape, store.looks
    )


def _edit_item(root: Path, **properties: object) -> None:
    path = root / "item.json"
    item = json.loads(path.read_text())
    item["properties"].update(properties)
    path.write_text(json.dumps(item))


def test_ifg_artifact_round_trip_records_provenance(tmp_path: Path) -> None:
    """Persisted layers round-trip; virtual layers derive without new assets."""
    root = _ifg_dir(tmp_path)
    store = _write_base(root)

    product = store.read()
    expected = _ifg_arrays()
    np.testing.assert_array_equal(product.complex_ifg, expected["complex_ifg"])
    np.testing.assert_array_equal(product.coherence, expected["coherence"])
    expected_wrapped = np.angle(expected["complex_ifg"]).astype(np.float32)
    expected_amplitude = np.abs(expected["complex_ifg"]).astype(np.float32)
    np.testing.assert_array_equal(product.wrapped_phase, expected_wrapped)
    np.testing.assert_array_equal(product.amplitude, expected_amplitude)
    assert product.wrapped_phase.dtype == np.float32
    assert product.amplitude.dtype == np.float32
    assert store.pair == ("20240101", "20240113")
    assert store.looks == (2, 3)
    assert store.filter_name == "goldstein"
    assert store.filter_parameters == {"alpha": 0.5}
    assert {key: store.source_manifest_digests[key] for key in _SOURCES} == _SOURCES
    item = json.loads((root / "item.json").read_text())
    assert item["properties"]["faninsar:kind"] == "ifg"
    assert item["properties"]["faninsar:inputs"] == _SOURCES
    assert set(item["assets"]) == {"coherence", "complex_ifg"}
    assert sorted(path.name for path in root.iterdir()) == [
        "coherence.zarr",
        "complex_ifg.zarr",
        "item.json",
    ]
    assert not [path for path in root.parent.iterdir() if path.name.startswith(".")]


def test_read_layer_derives_virtual_layers_and_writes_nothing(tmp_path: Path) -> None:
    """Virtual layers derive from the canonical complex IFG without writing."""
    root = _ifg_dir(tmp_path)
    complex_ifg = np.array(
        [[1.0 + 2.0j, np.nan + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]],
        dtype=np.complex64,
    )
    valid_mask = np.array([[True, True], [True, False]])
    store = _write_base(
        root,
        complex_ifg=complex_ifg,
        coherence=np.full((2, 2), 0.5, dtype=np.float32),
        valid_mask=valid_mask,
    )

    wrapped_phase = store.read_layer("wrapped_phase")
    amplitude = store.read_layer("amplitude")
    expected_wrapped = np.angle(complex_ifg)
    expected_amplitude = np.abs(complex_ifg)
    expected_wrapped[~valid_mask] = np.nan
    expected_amplitude[~valid_mask] = np.nan
    np.testing.assert_array_equal(wrapped_phase, expected_wrapped)
    np.testing.assert_array_equal(amplitude, expected_amplitude)
    assert wrapped_phase.dtype == np.float32
    assert amplitude.dtype == np.float32
    # Zero complex samples keep exact zeros; NaN complex samples propagate NaN.
    assert wrapped_phase[1, 0] == 0.0
    assert amplitude[1, 0] == 0.0
    assert np.isnan(wrapped_phase[0, 1]) and np.isnan(amplitude[0, 1])
    # Invalid support becomes NaN even where the complex sample is finite.
    assert np.isnan(wrapped_phase[1, 1]) and np.isnan(amplitude[1, 1])

    artifact = store.read()
    np.testing.assert_array_equal(artifact.wrapped_phase, wrapped_phase)
    np.testing.assert_array_equal(artifact.amplitude, amplitude)
    np.testing.assert_array_equal(artifact.valid_mask, valid_mask)
    # Real assets only; virtual reads never write wrapped/amplitude stores.
    assert sorted(path.name for path in root.iterdir()) == [
        "coherence.zarr",
        "complex_ifg.zarr",
        "item.json",
        "valid_mask.zarr",
    ]
    item = json.loads((root / "item.json").read_text())
    assert set(item["assets"]) == {"coherence", "complex_ifg", "valid_mask"}


def test_read_layer_focus_reads_only_the_requested_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A focused layer read must not load unrelated payloads."""
    import faninsar.io.storage.ifg_store as ifg_store_module

    root = _ifg_dir(tmp_path)
    store = _write_base(root, valid_mask=np.ones((2, 3), dtype=bool))
    seen: list[str] = []
    original = ifg_store_module.read_zarr_array

    def recording(path: Path) -> np.ndarray:
        seen.append(path.name)
        return original(path)

    monkeypatch.setattr(ifg_store_module, "read_zarr_array", recording)
    store.read_layer("coherence")
    assert seen == ["coherence.zarr"]
    seen.clear()
    store.read_layer("wrapped_phase")
    assert seen == ["complex_ifg.zarr", "valid_mask.zarr"]
    seen.clear()
    store.read()
    # The canonical complex IFG is loaded once per read bundle.
    assert seen == ["complex_ifg.zarr", "coherence.zarr", "valid_mask.zarr"]


def test_read_layer_rejects_unknown_missing_and_damaged_layers(
    tmp_path: Path,
) -> None:
    """Unknown names, absent optionals, and damaged payloads fail closed."""
    import zarr

    root = _ifg_dir(tmp_path)
    store = _write_base(root)
    with pytest.raises(InvalidProcessingStateError):
        store.read_layer("phase")
    with pytest.raises(InvalidProcessingStateError):
        store.read_layer("valid_mask")

    without_coherence = _write_base(tmp_path / "bare", coherence=None)
    with pytest.raises(InvalidProcessingStateError):
        without_coherence.read_layer("coherence")
    assert without_coherence.read().coherence is None

    # A dtype-damaged canonical complex layer invalidates derived reads too.
    zarr.open_array(
        store=str(root / "complex_ifg.zarr"), mode="w", shape=(2, 3), dtype=np.float32
    )
    for name in ("complex_ifg", "wrapped_phase", "amplitude"):
        with pytest.raises(InvalidProcessingStateError):
            store.read_layer(name)
    with pytest.raises(InvalidProcessingStateError):
        store.read()


def test_read_layer_rejects_wrong_shape_mask_payload(tmp_path: Path) -> None:
    """A wrong-shape ``valid_mask`` fails closed for mask and virtual reads."""
    import zarr

    root = _ifg_dir(tmp_path)
    store = _write_base(root, valid_mask=np.ones((2, 3), dtype=bool))
    zarr.open_array(
        store=str(root / "valid_mask.zarr"), mode="w", shape=(1, 1), dtype=bool
    )
    with pytest.raises(InvalidProcessingStateError):
        store.read_layer("valid_mask")
    with pytest.raises(InvalidProcessingStateError):
        store.read_layer("amplitude")


def test_unwrap_is_a_separate_product_bound_to_the_ifg(tmp_path: Path) -> None:
    """Unwrap lives under ``unwrap/ml_AxR/<pair>`` and records the IFG write id."""
    root = _ifg_dir(tmp_path)
    store = _write_base(root)
    _unwrap(store)

    target = unwrap_root(root)
    assert target == tmp_path / "work" / "unwrap" / "ml_2x3" / "20240101_20240113"
    item = json.loads((target / "item.json").read_text())
    assert item["properties"]["faninsar:kind"] == "unwrap"
    assert item["properties"]["faninsar:inputs"] == {"ifg": store.write_id}
    assert not list(tmp_path.rglob("*.npy"))
    assert not (root / "unwrapped_phase.zarr").exists()


def test_ifg_artifact_records_phase_screen_lineage(tmp_path: Path) -> None:
    """Phase-screen metadata is explicit in the Item."""
    root = _ifg_dir(tmp_path)
    store = _write_base(
        root,
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        flatten_stage="interferogram",
        phase_screen_model="nisar_ellipsoidal_v1",
        phase_screen_digests={"secondary": {"IW1_b1": "c" * 64}},
    )

    assert store.flatten_stage == "interferogram"
    assert store.phase_screen_model == "nisar_ellipsoidal_v1"
    assert store.phase_screen_digests == {"secondary": {"IW1_b1": "c" * 64}}
    assert store.phase_screen_domain == store.domain
    assert store.phase_screen_grid_identity == store.grid_identity
    properties = json.loads((root / "item.json").read_text())["properties"]
    assert properties["faninsar:phase_screen_model"] == "nisar_ellipsoidal_v1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("faninsar:flatten_stage", "invalid"),
        ("faninsar:phase_screen_model", "invalid"),
        ("faninsar:phase_screen_digests", {"primary": {"IW1_b1": "not-a-digest"}}),
        ("faninsar:looks", [0, 1]),
        ("faninsar:pair", ["20240101"]),
        ("faninsar:filter", {"parameters": {}}),
    ],
)
def test_ifg_open_rejects_invalid_item_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    """An Item edited into an inconsistent state fails at open."""
    root = _ifg_dir(tmp_path)
    _write_base(root)
    _edit_item(root, **{field: value})

    with pytest.raises(InvalidProcessingStateError):
        InterferogramArtifactStore.open(root)


@pytest.mark.parametrize(
    ("phase_screen_model", "phase_screen_digests"),
    [
        (None, None),
        ("nisar_ellipsoidal_v1", {}),
    ],
)
def test_ifg_artifact_rejects_deferred_without_screen_lineage(
    tmp_path: Path,
    phase_screen_model: str | None,
    phase_screen_digests: dict[str, dict[str, str]] | None,
) -> None:
    """Deferred flattening requires an explicit non-empty screen lineage."""
    with pytest.raises(InvalidProcessingStateError, match="non-empty NISAR"):
        _write_base(
            _ifg_dir(tmp_path),
            flatten_stage="interferogram",
            phase_screen_model=phase_screen_model,
            phase_screen_digests=phase_screen_digests,
        )
    assert not _ifg_dir(tmp_path).exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("flatten_stage", "invalid"),
        ("phase_screen_model", "invalid"),
        ("phase_screen_digests", {"primary": {"IW1_b1": "not-a-digest"}}),
    ],
)
def test_ifg_artifact_rejects_invalid_phase_lineage(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Invalid phase-lineage enums and digests are rejected before commit."""
    kwargs: dict[str, object] = {
        "flatten_stage": "interferogram",
        "phase_screen_model": "nisar_ellipsoidal_v1",
        "phase_screen_digests": {"primary": {"IW1_b1": "c" * 64}},
    }
    kwargs[field] = value
    with pytest.raises(InvalidProcessingStateError):
        _write_base(_ifg_dir(tmp_path), **kwargs)


@pytest.mark.parametrize("failure", ["missing", "corrupt", "shape"])
def test_ifg_artifact_payload_damage_fails_closed(
    tmp_path: Path,
    failure: str,
) -> None:
    """Missing, corrupt, and wrong-shape payloads cannot be consumed."""
    root = _ifg_dir(tmp_path)
    _write_base(root)
    payload = root / "coherence.zarr"
    if failure == "missing":
        shutil.rmtree(payload)
    elif failure == "corrupt":
        (payload / "zarr.json").write_bytes(b"not a zarr array")
    else:
        import zarr

        zarr.open_array(store=str(payload), mode="w", shape=(1, 1), dtype=np.float32)

    with pytest.raises(InvalidProcessingStateError):
        InterferogramArtifactStore.open(root).read()


def test_verify_detects_modified_payload(tmp_path: Path) -> None:
    """Normal reads skip hashing; ``verify_product`` re-hashes every asset."""
    root = _ifg_dir(tmp_path)
    _write_base(root)
    assert verify_product(root) == []
    import zarr

    array = zarr.open_array(store=str(root / "coherence.zarr"), mode="r+")
    array[0, 0] = 99.0
    assert verify_product(root) == ["coherence"]


def test_ifg_artifact_rejects_inconsistent_arrays(tmp_path: Path) -> None:
    """A product whose layers do not share shape fails before commit."""
    arrays = _ifg_arrays()
    arrays["coherence"] = np.ones((1, 1), dtype=np.float32)
    with pytest.raises(InvalidProcessingStateError):
        _write_base(_ifg_dir(tmp_path), **arrays)
    assert not _ifg_dir(tmp_path).exists()


def test_ifg_overwrite_requires_explicit_replace(tmp_path: Path) -> None:
    """A committed IFG is replaced only with ``replace_existing=True``."""
    root = _ifg_dir(tmp_path)
    first = _write_base(root)
    with pytest.raises(InvalidProcessingStateError, match="already exists"):
        _write_base(root)
    second = _write_base(root, replace_existing=True)
    assert second.write_id != first.write_id


def test_unwrapped_artifact_round_trip_is_bound_to_ifg(tmp_path: Path) -> None:
    """Unwrap layers commit only against the exact current IFG."""
    root = _ifg_dir(tmp_path)
    store = _write_base(root)
    _unwrap(store)

    result = InterferogramArtifactStore.open(root).read_unwrapped()
    np.testing.assert_array_equal(
        result.unwrapped_phase, np.arange(6, dtype=np.float32).reshape(2, 3)
    )
    np.testing.assert_array_equal(
        result.connected_components, np.ones((2, 3), dtype=np.int32)
    )
    assert result.method == "snaphu"
    assert result.method_parameters == {"cost_mode": "smooth"}
    assert result.ifg_manifest_digest == store.write_id
    assert store.has_unwrapped()


def test_unwrapped_artifact_rejects_wrong_ifg_binding(tmp_path: Path) -> None:
    """A caller cannot attach an unwrap result to a different IFG."""
    root = _ifg_dir(tmp_path)
    store = _write_base(root)
    with pytest.raises(InvalidProcessingStateError):
        _unwrap(store, ifg_manifest_digest="f" * 64)
    assert not unwrap_root(root).exists()


def test_replacing_the_ifg_makes_its_unwrap_stale(tmp_path: Path) -> None:
    """A recomputed IFG gets a new write id; the old unwrap no longer binds."""
    root = _ifg_dir(tmp_path)
    store = _write_base(root)
    _unwrap(store)
    replaced = _write_base(root, replace_existing=True)

    assert not replaced.has_unwrapped()
    stale = _layers(replaced)
    assert stale.layers == ["amplitude", "coherence", "complex_ifg", "wrapped"]
    with pytest.raises(InvalidProcessingStateError, match="replaced IFG"):
        stale.unwrapped
    with pytest.raises(InvalidProcessingStateError, match="different IFG"):
        replaced.read_unwrapped()
    with pytest.raises(InvalidProcessingStateError, match="already exists"):
        _unwrap(replaced)
    _unwrap(replaced, replace_existing=True)
    assert replaced.has_unwrapped()
    assert _layers(replaced).layers == [
        "amplitude",
        "coherence",
        "complex_ifg",
        "connected_components",
        "unwrapped",
        "wrapped",
    ]


def test_unwrapped_artifact_damage_fails_closed(tmp_path: Path) -> None:
    """A missing connected-component layer invalidates the unwrap product."""
    root = _ifg_dir(tmp_path)
    store = _write_base(root)
    _unwrap(store)
    shutil.rmtree(unwrap_root(root) / "connected_components.zarr")

    with pytest.raises(InvalidProcessingStateError):
        InterferogramArtifactStore.open(root).read_unwrapped()


def test_unwrapped_artifact_requires_matching_shape_and_integer_labels(
    tmp_path: Path,
) -> None:
    """Unwrap arrays must match the IFG grid and components must be integral."""
    store = _write_base(_ifg_dir(tmp_path))
    with pytest.raises(InvalidProcessingStateError):
        _unwrap(store, connected_components=np.ones((2, 2), dtype=np.int32))
    with pytest.raises(InvalidProcessingStateError):
        _unwrap(store, connected_components=np.ones((2, 3), dtype=np.float32))


def test_earlier_layout_fails_with_rebuild_message(tmp_path: Path) -> None:
    """A pre-PROPOSAL-0059 IFG directory fails with one actionable message."""
    root = _ifg_dir(tmp_path)
    root.mkdir(parents=True)
    (root / "manifest.json").write_text("{}")
    with pytest.raises(InvalidProcessingStateError, match="item.json"):
        InterferogramArtifactStore.open(root)
