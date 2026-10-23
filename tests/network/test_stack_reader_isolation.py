"""Single-Pair query isolation for the Stack interferogram reader (PROPOSAL-0059).

A per-Pair query (``open``/``exists``/``path``/``item``) and an explicit
``open_stack`` subset must touch only the selected Pair products. A missing or
damaged product of another planned Pair must not block them. Whole-network
reads without a subset keep the strict planned-set check.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.io.storage import ifg_store
from faninsar.io.storage.ifg_store import InterferogramArtifactStore, write_ifg_artifact
from faninsar.missions.s1 import S1Stack
from faninsar.network.readers.stack import StackInterferogramCollection
from faninsar.processing.errors import InvalidProcessingStateError

if TYPE_CHECKING:
    from faninsar.stack import Stack

TARGET = "20240101_20240113"
OTHER = "20240113_20240125"
THIRD = "20240101_20240125"
UNPLANNED = "20240101_20240201"
PLANNED = (TARGET, OTHER, THIRD)


def _planned_stack(tmp_path: Path) -> Stack:
    """Create a prepared three-date Stack with an explicit triangle network."""
    from faninsar import Pairs

    paths = []
    for date_id in ("20240101", "20240113", "20240125"):
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000.SAFE"
        path.mkdir()
        paths.append(path)
    stack = S1Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        activation_mode="reference",
        pairs=Pairs.from_names(list(PLANNED)),
        multilook=(1, 1),
    )
    stack = stack.prepare_scenes()
    stack._ensure_precise_orbits = lambda: None  # type: ignore[method-assign]
    return stack


def _write_pair_product(
    stack: Stack,
    pair_id: str,
    *,
    phase: np.ndarray,
    coherence: np.ndarray | None = None,
    looks: tuple[int, int] = (1, 1),
    pair: tuple[str, str] | None = None,
) -> Path:
    """Commit one small synthetic IFG product for the planned network."""
    root = stack.config.work_dir / "ifg" / "ml_1x1" / pair_id
    if coherence is None:
        coherence = np.full(phase.shape, 0.75, dtype=np.float32)
    write_ifg_artifact(
        root,
        pair=pair or tuple(pair_id.split("_")),  # type: ignore[arg-type]
        looks=looks,
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"primary": "a" * 64, "secondary": "b" * 64},
        complex_ifg=np.exp(1j * phase).astype(np.complex64),
        coherence=coherence,
        replace_existing=True,
    )
    return root


def _target_product(
    stack: Stack,
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Write the target Pair product and return it with its known layers."""
    phase = np.full((3, 4), 0.2, dtype=np.float32)
    coherence = np.full((3, 4), 0.75, dtype=np.float32)
    root = _write_pair_product(stack, TARGET, phase=phase, coherence=coherence)
    return root, phase, coherence


def _stack_with_missing_other_pair(
    tmp_path: Path,
) -> tuple[Stack, np.ndarray, np.ndarray]:
    """Plan three Pairs, commit two, and leave the third product missing."""
    stack = _planned_stack(tmp_path)
    root, phase, coherence = _target_product(stack)
    _write_pair_product(stack, THIRD, phase=phase)
    assert root.is_dir()
    assert not (stack.config.work_dir / "ifg" / "ml_1x1" / OTHER).exists()
    return stack, phase, coherence


def _stack_with_corrupt_other_pair(
    tmp_path: Path,
) -> tuple[Stack, np.ndarray, np.ndarray]:
    """Plan three Pairs and damage another Pair's product metadata."""
    stack = _planned_stack(tmp_path)
    root, phase, coherence = _target_product(stack)
    for pair_id in (OTHER, THIRD):
        _write_pair_product(stack, pair_id, phase=phase)
    broken = stack.config.work_dir / "ifg" / "ml_1x1" / OTHER / "item.json"
    broken.write_text("{ not valid json", encoding="utf-8")
    assert root.is_dir()
    return stack, phase, coherence


def test_pair_queries_ignore_missing_other_pair_product(tmp_path: Path) -> None:
    """Per-Pair open/exists/path/item succeed while another Pair is missing."""
    stack, _phase, coherence = _stack_with_missing_other_pair(tmp_path)
    collection = StackInterferogramCollection(stack)

    assert collection.exists(TARGET, "complex_ifg") is True
    assert collection.exists(TARGET, "coherence") is True
    np.testing.assert_allclose(
        collection.open(TARGET, "coherence").values,
        coherence,
    )
    complex_path = collection.path(TARGET, "complex_ifg")
    assert complex_path == collection.root / TARGET / "complex_ifg.zarr"
    assert complex_path.is_dir()
    assets = collection.item(TARGET)["assets"]
    assert {"complex_ifg", "coherence"}.issubset(assets)
    assert "wrapped_phase" not in assets


def test_pair_queries_ignore_corrupt_other_pair_product(tmp_path: Path) -> None:
    """Per-Pair open/exists/path/item succeed while another Pair is corrupt."""
    stack, _phase, coherence = _stack_with_corrupt_other_pair(tmp_path)
    collection = StackInterferogramCollection(stack)

    assert collection.exists(TARGET, "complex_ifg") is True
    np.testing.assert_allclose(
        collection.open(TARGET, "coherence").values,
        coherence,
    )
    assert collection.path(TARGET, "complex_ifg").is_dir()
    assert "complex_ifg" in collection.item(TARGET)["assets"]


def test_exists_reads_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The existence check never loads a raster payload."""
    stack, _phase, _coherence = _stack_with_corrupt_other_pair(tmp_path)
    collection = StackInterferogramCollection(stack)

    def boom(_path: object) -> object:
        message = "exists must not read raster payloads"
        raise AssertionError(message)

    # The check observes only product metadata; any payload read is a bug.
    monkeypatch.setattr(ifg_store, "read_zarr_array", boom)
    assert collection.exists(TARGET, "complex_ifg") is True
    assert collection.exists(TARGET, "coherence") is True
    assert collection.exists(TARGET, "wrapped_phase") is True
    assert collection.exists(TARGET, "unw_phase") is False


def test_open_reads_only_the_target_raster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening one Pair reads exactly that Pair's product and raster."""
    stack, phase, _coherence = _stack_with_missing_other_pair(tmp_path)
    collection = StackInterferogramCollection(stack)

    opened: list[Path] = []
    reads: list[Path] = []
    real_open = InterferogramArtifactStore.open
    real_read = ifg_store.read_zarr_array

    def recording_open(root: Path) -> InterferogramArtifactStore:
        opened.append(Path(root))
        return real_open(root)

    def recording_read(path: Path) -> np.ndarray:
        reads.append(Path(path))
        return real_read(path)

    monkeypatch.setattr(InterferogramArtifactStore, "open", recording_open)
    monkeypatch.setattr(ifg_store, "read_zarr_array", recording_read)
    np.testing.assert_allclose(
        collection.open(TARGET, "complex_ifg").values,
        np.exp(1j * phase).astype(np.complex64),
    )

    assert opened == [collection.root / TARGET]
    assert reads
    assert all(path.is_relative_to(collection.root / TARGET) for path in reads)


def test_unknown_pair_is_rejected_without_touching_products(tmp_path: Path) -> None:
    """An unplanned Pair keeps KeyError/False and never opens a product."""
    stack, _phase, _coherence = _stack_with_missing_other_pair(tmp_path)
    collection = StackInterferogramCollection(stack)

    with pytest.raises(KeyError):
        collection.open(UNPLANNED, "complex_ifg")
    with pytest.raises(KeyError):
        collection.path(UNPLANNED, "complex_ifg")
    with pytest.raises(KeyError):
        collection.item(UNPLANNED)
    assert collection.exists(UNPLANNED, "complex_ifg") is False


def test_missing_target_product_still_fails(tmp_path: Path) -> None:
    """A planned Pair without its own product still fails every query."""
    stack, _phase, _coherence = _stack_with_missing_other_pair(tmp_path)
    collection = StackInterferogramCollection(stack)

    with pytest.raises(InvalidProcessingStateError):
        collection.open(OTHER, "complex_ifg")
    with pytest.raises(InvalidProcessingStateError):
        collection.path(OTHER, "complex_ifg")
    with pytest.raises(InvalidProcessingStateError):
        collection.item(OTHER)
    with pytest.raises(InvalidProcessingStateError):
        collection.exists(OTHER, "complex_ifg")


def test_whole_network_reads_remain_strict(tmp_path: Path) -> None:
    """Reads without a subset still enforce the exact planned Pair set."""
    stack, _phase, _coherence = _stack_with_missing_other_pair(tmp_path)
    collection = StackInterferogramCollection(stack)

    with pytest.raises(InvalidProcessingStateError, match="does not exactly match"):
        collection.open_stack("complex_ifg")
    with pytest.raises(InvalidProcessingStateError, match="does not exactly match"):
        _ = collection.index_metadata
    with pytest.raises(InvalidProcessingStateError, match="does not exactly match"):
        _ = collection.spatial_metadata


def test_whole_network_reads_reject_corrupt_members(tmp_path: Path) -> None:
    """Whole-network reads still open and validate every planned product."""
    stack, _phase, _coherence = _stack_with_corrupt_other_pair(tmp_path)
    collection = StackInterferogramCollection(stack)

    with pytest.raises(InvalidProcessingStateError):
        collection.open_stack("complex_ifg")
    with pytest.raises(InvalidProcessingStateError):
        _ = collection.index_metadata


def test_open_stack_explicit_subset_reads_only_selected_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit subset opens only the selected Pair products."""
    stack, phase, _coherence = _stack_with_missing_other_pair(tmp_path)
    collection = StackInterferogramCollection(stack)

    from faninsar import Pairs

    subset = Pairs.from_names([TARGET, THIRD])
    opened: list[Path] = []
    reads: list[Path] = []
    real_open = InterferogramArtifactStore.open
    real_read = ifg_store.read_zarr_array

    def recording_open(root: Path) -> InterferogramArtifactStore:
        opened.append(Path(root))
        return real_open(root)

    def recording_read(path: Path) -> np.ndarray:
        reads.append(Path(path))
        return real_read(path)

    monkeypatch.setattr(InterferogramArtifactStore, "open", recording_open)
    monkeypatch.setattr(ifg_store, "read_zarr_array", recording_read)
    opened_data = collection.open_stack("complex_ifg", pairs=subset)

    expected = np.stack(
        [np.exp(1j * phase).astype(np.complex64)] * 2,
        axis=0,
    )
    np.testing.assert_allclose(opened_data.values, expected)
    assert opened_data.coords["pair"].values.tolist() == [TARGET, THIRD]
    assert sorted(path.name for path in opened) == sorted((TARGET, THIRD))
    assert reads
    assert all(
        path.is_relative_to(collection.root / TARGET)
        or path.is_relative_to(collection.root / THIRD)
        for path in reads
    )

    with pytest.raises(InvalidProcessingStateError):
        collection.open_stack("complex_ifg", pairs=Pairs.from_names([OTHER]))
    with pytest.raises(KeyError):
        collection.open_stack("complex_ifg", pairs=Pairs.from_names([UNPLANNED]))


def test_target_pair_constraints_still_enforced(tmp_path: Path) -> None:
    """A broken target product is rejected on its own Pair/looks identity."""
    stack = _planned_stack(tmp_path)
    phase = np.full((3, 4), 0.2, dtype=np.float32)
    collection = StackInterferogramCollection(stack)

    _write_pair_product(stack, TARGET, phase=phase, looks=(2, 2))
    with pytest.raises(InvalidProcessingStateError, match="looks"):
        collection.open(TARGET, "complex_ifg")

    _write_pair_product(
        stack,
        TARGET,
        phase=phase,
        pair=("20240113", "20240125"),
    )
    with pytest.raises(InvalidProcessingStateError, match="pair"):
        collection.open(TARGET, "complex_ifg")
