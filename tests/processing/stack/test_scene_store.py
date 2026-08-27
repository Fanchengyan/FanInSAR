"""Tests for manifest-bound Stack scene reuse."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.stack.scene_store import (
    CoregisteredSceneStore,
    form_merged_scene_interferogram,
    form_scene_interferograms,
    write_scene_unit,
)


def test_scene_store_round_trip_and_ifg_uses_aligned_payloads(tmp_path: Path) -> None:
    """Persisted Reference-aligned arrays round-trip into one exact IFG."""
    reference = np.ones((2, 3), dtype=np.complex64)
    secondary = np.full((2, 3), 1.0 + 2.0j, dtype=np.complex64)
    write_scene_unit(
        tmp_path / "reference",
        date_id="20240101",
        reference_id="20240101",
        domain="radar",
        tag="IW1_b0",
        primary=reference,
        secondary=secondary,
        row_origin=0,
        col_origin=0,
    )
    write_scene_unit(
        tmp_path / "secondary",
        date_id="20240113",
        reference_id="20240101",
        domain="radar",
        tag="IW1_b0",
        primary=reference,
        secondary=secondary,
        row_origin=0,
        col_origin=0,
    )
    first = CoregisteredSceneStore.open(tmp_path / "reference")
    second = CoregisteredSceneStore.open(tmp_path / "secondary")
    output = form_scene_interferograms(first, second)
    np.testing.assert_array_equal(output["IW1_b0"], np.conj(secondary))


def test_scene_store_rejects_manifest_tampering(tmp_path: Path) -> None:
    """A changed manifest cannot be consumed as a complete generation."""
    data = np.ones((2, 2), dtype=np.complex64)
    root = tmp_path / "scene"
    write_scene_unit(
        root,
        date_id="20240113",
        reference_id="20240101",
        domain="radar",
        tag="IW1_b0",
        primary=data,
        secondary=data,
        row_origin=0,
        col_origin=0,
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        manifest.read_text().replace('"status": "complete"', '"status": "bad"')
    )
    with pytest.raises(InvalidProcessingStateError):
        CoregisteredSceneStore.open(root)


def test_scene_store_rejects_legacy_master_manifest_key(tmp_path: Path) -> None:
    """Old Stack manifests fail closed instead of being silently migrated."""
    data = np.ones((2, 2), dtype=np.complex64)
    root = tmp_path / "scene"
    write_scene_unit(
        root,
        date_id="20240113",
        reference_id="20240101",
        domain="radar",
        tag="IW1_b0",
        primary=data,
        secondary=data,
        row_origin=0,
        col_origin=0,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["master_id"] = manifest.pop("reference_id")
    unsigned = dict(manifest)
    unsigned.pop("manifest_digest")
    manifest["manifest_digest"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InvalidProcessingStateError, match="master terminology"):
        CoregisteredSceneStore.open(root)


def test_scene_store_rejects_symlinked_root(tmp_path: Path) -> None:
    """A symlinked generation root must fail closed before manifest access."""
    data = np.ones((2, 2), dtype=np.complex64)
    real_root = tmp_path / "real"
    write_scene_unit(
        real_root,
        date_id="20240113",
        reference_id="20240101",
        domain="radar",
        tag="IW1_b0",
        primary=data,
        secondary=data,
        row_origin=0,
        col_origin=0,
    )
    link_root = tmp_path / "link"
    link_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(InvalidProcessingStateError):
        CoregisteredSceneStore.open(link_root)


def test_form_interferograms_supports_multiple_units(tmp_path: Path) -> None:
    """All complete burst units are formed without a single-unit shortcut."""
    reference = np.ones((2, 2), dtype=np.complex64)
    secondary = np.full((2, 2), 1.0 + 2.0j, dtype=np.complex64)
    for date_id, root in (("20240101", "reference"), ("20240113", "secondary")):
        for tag in ("IW1_b0", "IW1_b1"):
            write_scene_unit(
                tmp_path / root,
                date_id=date_id,
                reference_id="20240101",
                domain="radar",
                tag=tag,
                primary=reference,
                secondary=secondary,
                row_origin=0,
                col_origin=0,
            )
    first = CoregisteredSceneStore.open(tmp_path / "reference")
    second = CoregisteredSceneStore.open(tmp_path / "secondary")
    output = form_scene_interferograms(first, second)
    assert set(output) == {"IW1_b0", "IW1_b1"}


def test_form_interferograms_has_no_legacy_pair_call() -> None:
    """Stack formation must not import or invoke the Pair engine."""
    source = Path("faninsar/processing/stack/session.py").read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "form_interferograms"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == "run_pair"
        for node in ast.walk(method)
    )


def test_merged_scene_interferogram_uses_global_origins_before_looks(
    tmp_path: Path,
) -> None:
    """Overlapping bursts are merged in complex space on the common grid."""
    reference = np.ones((2, 4), dtype=np.complex64)
    upper_secondary = np.full((2, 4), 1.0j, dtype=np.complex64)
    lower_secondary = np.full((2, 4), -1.0j, dtype=np.complex64)
    for date_id, root in (("20240101", "reference"), ("20240113", "secondary")):
        for tag, row_origin, secondary in (
            ("f0_IW1_b0", 0, upper_secondary),
            ("f0_IW1_b1", 2, lower_secondary),
        ):
            write_scene_unit(
                tmp_path / root,
                date_id=date_id,
                reference_id="20240101",
                domain="radar",
                tag=tag,
                primary=reference,
                secondary=secondary,
                row_origin=row_origin,
                col_origin=0,
                grid_shape=(4, 4),
            )
    first = CoregisteredSceneStore.open(tmp_path / "reference")
    second = CoregisteredSceneStore.open(tmp_path / "secondary")

    product = form_merged_scene_interferogram(
        first,
        second,
        secondary_role="secondary",
        multilook=(2, 2),
    )

    expected = np.vstack(
        (
            np.full((1, 2), -1.0j, dtype=np.complex64),
            np.full((1, 2), 1.0j, dtype=np.complex64),
        )
    )
    np.testing.assert_array_equal(product.complex_ifg, expected)
    np.testing.assert_array_equal(product.coherence, np.ones((2, 2), np.float32))


def test_merged_scene_interferogram_rejects_mixed_grid_placement(
    tmp_path: Path,
) -> None:
    """The same burst tag cannot move between acquisition generations."""
    data = np.ones((2, 2), dtype=np.complex64)
    for root, origin in (("reference", 0), ("secondary", 1)):
        write_scene_unit(
            tmp_path / root,
            date_id="20240101" if root == "reference" else "20240113",
            reference_id="20240101",
            domain="radar",
            tag="f0_IW1_b0",
            primary=data,
            secondary=data,
            row_origin=origin,
            col_origin=0,
            grid_shape=(4, 2),
        )
    with pytest.raises(InvalidProcessingStateError):
        form_merged_scene_interferogram(
            CoregisteredSceneStore.open(tmp_path / "reference"),
            CoregisteredSceneStore.open(tmp_path / "secondary"),
        )


def test_merged_scene_interferogram_rejects_equal_shape_shifted_grid(
    tmp_path: Path,
) -> None:
    """Equal array shapes cannot hide different coordinate-grid identities."""
    data = np.ones((2, 2), dtype=np.complex64)
    for root, grid_identity in (("reference", "a" * 64), ("secondary", "b" * 64)):
        write_scene_unit(
            tmp_path / root,
            date_id="20240101" if root == "reference" else "20240113",
            reference_id="20240101",
            domain="geo",
            tag="f0_IW1_b0",
            primary=data,
            secondary=data,
            row_origin=0,
            col_origin=0,
            grid_shape=(2, 2),
            grid_identity=grid_identity,
        )
    with pytest.raises(InvalidProcessingStateError, match="coordinate grids"):
        form_merged_scene_interferogram(
            CoregisteredSceneStore.open(tmp_path / "reference"),
            CoregisteredSceneStore.open(tmp_path / "secondary"),
        )


def test_scene_store_rejects_geo_manifest_without_grid_identity(
    tmp_path: Path,
) -> None:
    """A legacy Geo manifest cannot substitute array shape for coordinates."""
    data = np.ones((2, 2), dtype=np.complex64)
    root = tmp_path / "geo"
    write_scene_unit(
        root,
        date_id="20240101",
        reference_id="20240101",
        domain="geo",
        tag="f0_IW1_b0",
        primary=data,
        secondary=data,
        row_origin=0,
        col_origin=0,
        grid_shape=(2, 2),
        grid_identity="a" * 64,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("manifest_digest")
    manifest.pop("grid_identity")
    manifest["manifest_digest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InvalidProcessingStateError, match="grid_identity"):
        CoregisteredSceneStore.open(root)


def test_scene_writer_requires_explicit_geo_grid_identity(tmp_path: Path) -> None:
    """Geo publication cannot derive coordinates from array shape alone."""
    data = np.ones((2, 2), dtype=np.complex64)
    with pytest.raises(InvalidProcessingStateError, match="explicit grid_identity"):
        write_scene_unit(
            tmp_path / "geo",
            date_id="20240101",
            reference_id="20240101",
            domain="geo",
            tag="f0_IW1_b0",
            primary=data,
            secondary=data,
            row_origin=0,
            col_origin=0,
            grid_shape=(2, 2),
        )


def test_scene_writer_rejects_corrupted_existing_generation(tmp_path: Path) -> None:
    """Adding a unit cannot bless corrupted bytes with a fresh manifest."""
    data = np.ones((2, 2), dtype=np.complex64)
    root = tmp_path / "scene"
    write_scene_unit(
        root,
        date_id="20240101",
        reference_id="20240101",
        domain="radar",
        tag="f0_IW1_b0",
        primary=data,
        secondary=data,
        row_origin=0,
        col_origin=0,
        grid_shape=(4, 2),
    )
    (root / "f0_IW1_b0.secondary.npy").write_bytes(b"corrupt")

    with pytest.raises(InvalidProcessingStateError, match="payload digest"):
        write_scene_unit(
            root,
            date_id="20240101",
            reference_id="20240101",
            domain="radar",
            tag="f0_IW1_b1",
            primary=data,
            secondary=data,
            row_origin=2,
            col_origin=0,
            grid_shape=(4, 2),
        )


def test_merged_scene_interferogram_rejects_different_references(
    tmp_path: Path,
) -> None:
    """Equal grids cannot hide scene generations aligned to different References."""
    data = np.ones((2, 2), dtype=np.complex64)
    for root, date_id, reference_id in (
        ("reference", "20240101", "20240101"),
        ("secondary", "20240113", "20231220"),
    ):
        write_scene_unit(
            tmp_path / root,
            date_id=date_id,
            reference_id=reference_id,
            domain="radar",
            tag="f0_IW1_b0",
            primary=data,
            secondary=data,
            row_origin=0,
            col_origin=0,
            grid_shape=(2, 2),
        )

    with pytest.raises(InvalidProcessingStateError, match="References"):
        form_merged_scene_interferogram(
            CoregisteredSceneStore.open(tmp_path / "reference"),
            CoregisteredSceneStore.open(tmp_path / "secondary"),
        )
