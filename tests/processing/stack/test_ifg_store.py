"""Tests for manifest-bound Stack interferogram artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.stack.ifg_store import (
    ArtifactResourceLimits,
    InterferogramArtifactStore,
    write_ifg_artifact,
    write_unwrapped_artifact,
)

if TYPE_CHECKING:
    from pathlib import Path


def _ifg_arrays(shape: tuple[int, int] = (2, 3)) -> dict[str, np.ndarray]:
    """Return one internally consistent small interferogram product."""
    complex_ifg = np.full(shape, 1.0 + 2.0j, dtype=np.complex64)
    return {
        "complex_ifg": complex_ifg,
        "coherence": np.full(shape, 0.75, dtype=np.float32),
        "wrapped_phase": np.angle(complex_ifg).astype(np.float32),
        "amplitude": np.abs(complex_ifg).astype(np.float32),
    }


def _write_base(root: Path) -> InterferogramArtifactStore:
    """Write and reopen one complete base artifact."""
    return write_ifg_artifact(
        root,
        pair=("20240101", "20240113"),
        looks=(2, 3),
        filter_name="goldstein",
        filter_parameters={"alpha": 0.5},
        source_manifest_digests={
            "reference": "a" * 64,
            "secondary": "b" * 64,
        },
        **_ifg_arrays(),
    )


def _redigest_ifg_manifest(
    root: Path,
    update: dict[str, object] | None = None,
    remove: str | None = None,
) -> None:
    """Apply a manifest mutation and refresh every canonical pointer digest."""
    current_path = root / "CURRENT"
    current = json.loads(current_path.read_text())
    generation_path = root / ".ifg_generations" / current["generation_id"]
    manifest_paths = [root / "manifest.json", generation_path / "manifest.json"]
    digest = ""
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text())
        if update:
            manifest.update(update)
        if remove is not None:
            manifest.pop(remove)
        unsigned = dict(manifest)
        unsigned.pop("manifest_digest", None)
        digest = hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        manifest["manifest_digest"] = digest
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    current["manifest_digest"] = digest
    unsigned_current = dict(current)
    unsigned_current.pop("control_digest", None)
    current["control_digest"] = hashlib.sha256(
        json.dumps(
            unsigned_current,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    current_path.write_text(json.dumps(current), encoding="utf-8")


def test_ifg_artifact_round_trip_records_provenance(tmp_path: Path) -> None:
    """All IFG layers and their processing provenance round-trip exactly."""
    root = tmp_path / "ifg"
    store = _write_base(root)

    product = store.read()
    expected = _ifg_arrays()
    np.testing.assert_array_equal(product.complex_ifg, expected["complex_ifg"])
    np.testing.assert_array_equal(product.coherence, expected["coherence"])
    np.testing.assert_array_equal(product.wrapped_phase, expected["wrapped_phase"])
    np.testing.assert_array_equal(product.amplitude, expected["amplitude"])
    assert store.pair == ("20240101", "20240113")
    assert store.looks == (2, 3)
    assert store.filter_name == "goldstein"
    assert store.filter_parameters == {"alpha": 0.5}
    assert store.source_manifest_digests == {
        "reference": "a" * 64,
        "secondary": "b" * 64,
    }

    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert set(manifest["payloads"]) == {
        "amplitude",
        "coherence",
        "complex_ifg",
        "wrapped_phase",
    }
    assert not list(root.glob("*.tmp"))


def test_ifg_artifact_records_phase_screen_lineage(tmp_path: Path) -> None:
    """Phase-screen metadata is explicit and bound into the manifest digest."""
    root = tmp_path / "ifg"
    store = write_ifg_artifact(
        root,
        pair=("20240101", "20240113"),
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"primary": "a" * 64, "secondary": "b" * 64},
        flatten_stage="interferogram",
        phase_screen_model="nisar_ellipsoidal_v1",
        phase_screen_digests={"secondary": {"IW1_b1": "c" * 64}},
        **_ifg_arrays(),
    )

    assert store.flatten_stage == "interferogram"
    assert store.phase_screen_model == "nisar_ellipsoidal_v1"
    assert store.phase_screen_digests == {"secondary": {"IW1_b1": "c" * 64}}
    assert store.phase_screen_domain == store.domain
    assert store.phase_screen_grid_identity == store.grid_identity
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["phase_screen_model"] == "nisar_ellipsoidal_v1"
    assert manifest["phase_screen_digests"] == {"secondary": {"IW1_b1": "c" * 64}}


def test_ifg_artifact_rejects_tampered_phase_screen_lineage(tmp_path: Path) -> None:
    """Changing phase-screen lineage makes reopening fail closed."""
    root = tmp_path / "ifg"
    write_ifg_artifact(
        root,
        pair=("20240101", "20240113"),
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"primary": "a" * 64},
        flatten_stage="interferogram",
        phase_screen_model="nisar_ellipsoidal_v1",
        phase_screen_digests={"primary": {"IW1_b1": "c" * 64}},
        **_ifg_arrays(),
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["phase_screen_digests"]["primary"]["IW1_b1"] = "d" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InvalidProcessingStateError, match="digest mismatch"):
        InterferogramArtifactStore.open(root)


@pytest.mark.parametrize(
    "missing_field",
    [
        "flatten_stage",
        "phase_screen_model",
        "phase_screen_digests",
        "phase_screen_domain",
        "phase_screen_grid_identity",
    ],
)
def test_ifg_artifact_rejects_redigested_missing_phase_lineage(
    tmp_path: Path,
    missing_field: str,
) -> None:
    """A valid digest cannot legitimize a manifest missing lineage fields."""
    root = tmp_path / "ifg"
    _write_base(root)
    _redigest_ifg_manifest(root, remove=missing_field)

    with pytest.raises(InvalidProcessingStateError, match="missing explicit"):
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
        write_ifg_artifact(
            tmp_path / "ifg",
            pair=("20240101", "20240113"),
            looks=(1, 1),
            filter_name="none",
            filter_parameters={},
            source_manifest_digests={"primary": "a" * 64},
            flatten_stage="interferogram",
            phase_screen_model=phase_screen_model,
            phase_screen_digests=phase_screen_digests,
            **_ifg_arrays(),
        )


def test_ifg_artifact_open_rejects_redigested_deferred_without_screen(
    tmp_path: Path,
) -> None:
    """Re-digesting a deferred artifact cannot remove its screen lineage."""
    root = tmp_path / "ifg"
    write_ifg_artifact(
        root,
        pair=("20240101", "20240113"),
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"primary": "a" * 64},
        flatten_stage="interferogram",
        phase_screen_model="nisar_ellipsoidal_v1",
        phase_screen_digests={"primary": {"IW1_b1": "c" * 64}},
        **_ifg_arrays(),
    )
    _redigest_ifg_manifest(root, update={"phase_screen_digests": {}})

    with pytest.raises(InvalidProcessingStateError, match="non-empty NISAR"):
        InterferogramArtifactStore.open(root)


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
    """Invalid phase-lineage enums and digests are rejected before publish."""
    kwargs: dict[str, object] = {
        "flatten_stage": "interferogram",
        "phase_screen_model": "nisar_ellipsoidal_v1",
        "phase_screen_digests": {"primary": {"IW1_b1": "c" * 64}},
    }
    kwargs[field] = value
    with pytest.raises(InvalidProcessingStateError):
        write_ifg_artifact(
            tmp_path / "ifg",
            pair=("20240101", "20240113"),
            looks=(1, 1),
            filter_name="none",
            filter_parameters={},
            source_manifest_digests={"primary": "a" * 64},
            **kwargs,
            **_ifg_arrays(),
        )


@pytest.mark.parametrize("failure", ["missing", "corrupt", "shape"])
def test_ifg_artifact_payload_damage_fails_closed(
    tmp_path: Path,
    failure: str,
) -> None:
    """Missing, corrupt, and wrong-shape payloads cannot be consumed."""
    root = tmp_path / "ifg"
    _write_base(root)
    payload = InterferogramArtifactStore.open(root).generation_root / "coherence.npy"
    if failure == "missing":
        payload.unlink()
    elif failure == "corrupt":
        payload.write_bytes(b"not a numpy array")
    else:
        with payload.open("wb") as stream:
            np.save(stream, np.ones((1, 1), dtype=np.float32), allow_pickle=False)

    with pytest.raises(InvalidProcessingStateError):
        InterferogramArtifactStore.open(root).read()


def test_ifg_artifact_rejects_inconsistent_arrays(tmp_path: Path) -> None:
    """A product whose layers do not share shape and dtype fails before publish."""
    arrays = _ifg_arrays()
    arrays["amplitude"] = np.ones((1, 1), dtype=np.float32)
    with pytest.raises(InvalidProcessingStateError):
        write_ifg_artifact(
            tmp_path / "ifg",
            pair=("20240101", "20240113"),
            looks=(1, 1),
            filter_name="none",
            filter_parameters={},
            source_manifest_digests={"reference": "a" * 64},
            **arrays,
        )
    assert not (tmp_path / "ifg" / "manifest.json").exists()


def test_ifg_artifact_rejects_manifest_tampering(tmp_path: Path) -> None:
    """Rewriting provenance without its matching digest invalidates the store."""
    root = tmp_path / "ifg"
    _write_base(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["looks"] = [1, 1]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(InvalidProcessingStateError):
        InterferogramArtifactStore.open(root)


def test_unwrapped_artifact_round_trip_is_bound_to_ifg_digest(tmp_path: Path) -> None:
    """Unwrapped layers publish only against the exact complete IFG generation."""
    root = tmp_path / "ifg"
    store = _write_base(root)
    unwrapped = np.arange(6, dtype=np.float32).reshape(2, 3)
    components = np.ones((2, 3), dtype=np.int32)

    write_unwrapped_artifact(
        root,
        unwrapped_phase=unwrapped,
        connected_components=components,
        method="snaphu",
        method_parameters={"cost_mode": "smooth"},
        ifg_manifest_digest=store.manifest_digest,
    )
    result = InterferogramArtifactStore.open(root).read_unwrapped()
    np.testing.assert_array_equal(result.unwrapped_phase, unwrapped)
    np.testing.assert_array_equal(result.connected_components, components)
    assert result.method == "snaphu"
    assert result.method_parameters == {"cost_mode": "smooth"}
    assert result.ifg_manifest_digest == store.manifest_digest


def test_unwrapped_artifact_rejects_wrong_ifg_binding(tmp_path: Path) -> None:
    """A caller cannot attach an unwrap result to a different IFG digest."""
    root = tmp_path / "ifg"
    _write_base(root)
    with pytest.raises(InvalidProcessingStateError):
        write_unwrapped_artifact(
            root,
            unwrapped_phase=np.ones((2, 3), dtype=np.float32),
            connected_components=np.ones((2, 3), dtype=np.int32),
            method="snaphu",
            method_parameters={},
            ifg_manifest_digest="f" * 64,
        )
    assert not (root / "unwrap_manifest.json").exists()


def test_unwrapped_artifact_damage_fails_closed(tmp_path: Path) -> None:
    """Connected-component corruption invalidates the entire unwrap generation."""
    root = tmp_path / "ifg"
    store = _write_base(root)
    write_unwrapped_artifact(
        root,
        unwrapped_phase=np.ones((2, 3), dtype=np.float32),
        connected_components=np.ones((2, 3), dtype=np.int32),
        method="snaphu",
        method_parameters={},
        ifg_manifest_digest=store.manifest_digest,
    )
    unwrap_current = json.loads((root / "UNWRAP_CURRENT").read_text())
    unwrap_root = root / ".unwrap_generations" / unwrap_current["generation_id"]
    (unwrap_root / "connected_components.npy").unlink()

    with pytest.raises(InvalidProcessingStateError):
        InterferogramArtifactStore.open(root).read_unwrapped()


def test_unwrapped_artifact_requires_matching_shape_and_integer_labels(
    tmp_path: Path,
) -> None:
    """Unwrap arrays must match the IFG grid and components must be integral."""
    root = tmp_path / "ifg"
    store = _write_base(root)
    with pytest.raises(InvalidProcessingStateError):
        write_unwrapped_artifact(
            root,
            unwrapped_phase=np.ones((2, 3), dtype=np.float32),
            connected_components=np.ones((2, 2), dtype=np.int32),
            method="snaphu",
            method_parameters={},
            ifg_manifest_digest=store.manifest_digest,
        )
    with pytest.raises(InvalidProcessingStateError):
        write_unwrapped_artifact(
            root,
            unwrapped_phase=np.ones((2, 3), dtype=np.float32),
            connected_components=np.ones((2, 3), dtype=np.float32),
            method="snaphu",
            method_parameters={},
            ifg_manifest_digest=store.manifest_digest,
        )


def test_ifg_publication_uses_immutable_generation_and_bounded_current(
    tmp_path: Path,
) -> None:
    """Publication exposes only an atomic pointer, never direct payload files."""
    root = tmp_path / "ifg"
    store = _write_base(root)

    assert store.generation_root.parent == root / ".ifg_generations"
    assert store.generation_root.name == store.generation_id
    assert (root / "CURRENT").stat().st_size < 4096
    assert not (root / "complex_ifg.npy").exists()
    assert (store.generation_root / "manifest.json").is_file()


def test_ifg_current_tampering_and_partial_staging_fail_closed(tmp_path: Path) -> None:
    """A forged pointer or abandoned staging directory is never readable."""
    root = tmp_path / "ifg"
    _write_base(root)
    current = json.loads((root / "CURRENT").read_text())
    current["generation_id"] = "0" * 32
    (root / "CURRENT").write_text(json.dumps(current), encoding="utf-8")

    with pytest.raises(InvalidProcessingStateError, match="CURRENT"):
        InterferogramArtifactStore.open(root)

    partial = tmp_path / "partial"
    (partial / ".ifg_staging" / ("1" * 32)).mkdir(parents=True)
    (partial / ".ifg_staging" / ("1" * 32) / "manifest.json").write_text("{}")
    with pytest.raises(InvalidProcessingStateError):
        InterferogramArtifactStore.open(partial)


def test_ifg_reader_lease_prevents_generation_collection(tmp_path: Path) -> None:
    """An open reader pins its generation across a CURRENT replacement."""
    from faninsar.stack.artifact_transaction import collect_generations

    root = tmp_path / "ifg"
    first = _write_base(root)
    second = write_ifg_artifact(
        root,
        pair=("20240101", "20240113"),
        looks=(2, 3),
        filter_name="goldstein",
        filter_parameters={"alpha": 0.5},
        source_manifest_digests={"reference": "a" * 64, "secondary": "b" * 64},
        replace_existing=True,
        **_ifg_arrays(),
    )

    assert first.generation_id != second.generation_id
    assert collect_generations(root, "ifg") == ()
    first.close()
    assert collect_generations(root, "ifg") == (first.generation_id,)


def test_ifg_quota_rejects_before_any_generation_is_published(tmp_path: Path) -> None:
    """Insufficient declared quota fails before payload or CURRENT publication."""
    root = tmp_path / "ifg"
    with pytest.raises(InvalidProcessingStateError, match="quota"):
        write_ifg_artifact(
            root,
            pair=("20240101", "20240113"),
            looks=(1, 1),
            filter_name="none",
            filter_parameters={},
            source_manifest_digests={"reference": "a" * 64},
            resource_limits=ArtifactResourceLimits(
                max_final_bytes=1,
                max_temporary_bytes=1,
                min_free_bytes=0,
            ),
            **_ifg_arrays(),
        )
    assert not (root / "CURRENT").exists()
    assert not list((root / ".ifg_generations").glob("*"))


def test_legacy_direct_ifg_payload_is_quarantined(tmp_path: Path) -> None:
    """Pre-transaction direct payload layouts cannot masquerade as generations."""
    root = tmp_path / "ifg"
    root.mkdir()
    (root / "complex_ifg.npy").write_bytes(b"legacy")

    with pytest.raises(InvalidProcessingStateError, match="legacy direct"):
        InterferogramArtifactStore.open(root)
