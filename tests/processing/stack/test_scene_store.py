"""Tests for manifest-bound Stack scene reuse."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.stack.scene_store import (
    CoregisteredSceneStore,
    form_scene_interferograms,
    write_scene_unit,
)


def test_scene_store_round_trip_and_ifg_uses_aligned_payloads(tmp_path: Path) -> None:
    """Persisted master-aligned arrays round-trip into one exact IFG."""
    reference = np.ones((2, 3), dtype=np.complex64)
    secondary = np.full((2, 3), 1.0 + 2.0j, dtype=np.complex64)
    write_scene_unit(
        tmp_path / "reference",
        date_id="20240101",
        master_id="20240101",
        domain="radar",
        tag="IW1_b0",
        reference=reference,
        secondary=secondary,
        row_origin=0,
        col_origin=0,
    )
    write_scene_unit(
        tmp_path / "secondary",
        date_id="20240113",
        master_id="20240101",
        domain="radar",
        tag="IW1_b0",
        reference=reference,
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
        master_id="20240101",
        domain="radar",
        tag="IW1_b0",
        reference=data,
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


def test_scene_store_rejects_symlinked_root(tmp_path: Path) -> None:
    """A symlinked generation root must fail closed before manifest access."""
    data = np.ones((2, 2), dtype=np.complex64)
    real_root = tmp_path / "real"
    write_scene_unit(
        real_root,
        date_id="20240113",
        master_id="20240101",
        domain="radar",
        tag="IW1_b0",
        reference=data,
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
                master_id="20240101",
                domain="radar",
                tag=tag,
                reference=reference,
                secondary=secondary,
                row_origin=0,
                col_origin=0,
            )
    first = CoregisteredSceneStore.open(tmp_path / "reference")
    second = CoregisteredSceneStore.open(tmp_path / "secondary")
    output = form_scene_interferograms(first, second)
    assert set(output) == {"IW1_b0", "IW1_b1"}


def test_form_interferograms_has_no_run_pair_call() -> None:
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
