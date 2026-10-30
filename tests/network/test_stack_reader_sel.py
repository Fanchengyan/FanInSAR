"""sel() multilook and Pair-subset views on StackInterferogramCollection.

``sel(looks=)`` binds the collection to a committed ``ifg/ml_AxR`` view;
``sel(pairs=)`` narrows the read view to a subset of the planned network
when given a :class:`Pairs`, or demotes to that pair's
``InterferogramLayers`` when given a scalar ``Pair``/``str`` — the
scalar-vs-collection rule mirrors xarray ``sel``. Selection is immutable
and never relaxes the strict planned-network check on product discovery
(PROPOSAL-0059 AC#6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.core import Pairs
from faninsar.io.storage.scene_store import write_scene
from faninsar.network.layers import InterferogramLayers
from faninsar.network.readers.stack import StackInterferogramCollection
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geometry import ConstantDEM
from faninsar.processing.geometry.record import radar_geometry_record
from faninsar.stack import Stack, StackConfig
from faninsar.stack.catalog import SceneCatalog
from tests.stack.test_geocode import (
    RADAR_SHAPE,
    _scene_grid,
    _toy_geometry,
    _toy_grid_and_orbit,
    _write_ifg,
)

from .test_stack_reader_isolation import (
    OTHER,
    PLANNED,
    TARGET,
    THIRD,
    UNPLANNED,
    _planned_stack,
    _write_pair_product,
)

if TYPE_CHECKING:
    from pathlib import Path


def _commit_all(stack: Stack, looks: tuple[int, int] = (1, 1)) -> np.ndarray:
    """Commit a synthetic IFG product for every planned Pair at one looks."""
    phase = np.full((3, 4), 0.2, dtype=np.float32)
    for pair_id in PLANNED:
        _write_pair_product(stack, pair_id, phase=phase, looks=looks)
    return phase


def _stack_two_multilooks(
    tmp_path: Path,
) -> tuple[Stack, np.ndarray, np.ndarray]:
    """Commit the full planned network at both (1, 1) and (10, 4)."""
    stack = _planned_stack(tmp_path)
    phase_1x1 = _commit_all(stack, looks=(1, 1))
    phase_10x4 = np.full((2, 2), 0.4, dtype=np.float32)
    for pair_id in PLANNED:
        _write_pair_product(stack, pair_id, phase=phase_10x4, looks=(10, 4))
    return stack, phase_1x1, phase_10x4


@pytest.fixture
def coregistered_stack(tmp_path: Path) -> Stack:
    """Return a two-PAIR Stack whose committed Reference backs real layers.

    Mirrors ``tests/stack/test_geocode.py``'s ``stack`` fixture so
    ``sel(pair=)`` exercises the full ``_load_layers`` validation path
    (Reference, lineage, scene digests), not just the planned-network
    metadata shell the isolation fixture provides.
    """
    paths = []
    for day in ("20240101", "20240113"):
        safe = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000.SAFE"
        safe.mkdir()
        paths.append(safe)
    model = _toy_geometry()
    instance = Stack(
        catalog=SceneCatalog.from_paths(paths),
        config=StackConfig(
            work_dir=tmp_path / "work",
            grid=_scene_grid(model),
            multilook=(1, 1),
            dem=ConstantDEM(0.0),
            device="cpu",
        ),
        pairs=Pairs.from_names(["20240101_20240113"]),
        misreg_pairs=Pairs.from_names(["20240101_20240113"]),
        reference="20240101",
    )
    write_scene(
        instance.config.work_dir / "coreg" / instance.reference,
        date_id=instance.reference,
        reference_id=instance.reference,
        domain="radar",
        grid_shape=RADAR_SHAPE,
        units=[
            {
                "tag": "f0_IW1_b0",
                "payload": np.ones(RADAR_SHAPE, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 0,
            }
        ],
        properties={
            "faninsar:radar_geometry": radar_geometry_record(*_toy_grid_and_orbit())
        },
    )
    return instance


def test_available_looks_lists_committed_views(tmp_path: Path) -> None:
    """available_looks enumerates committed ml_AxR trees, sorted."""
    stack, _p1, _p10 = _stack_two_multilooks(tmp_path)
    collection = StackInterferogramCollection(stack)

    assert collection.available_looks == [(1, 1), (10, 4)]


def test_available_looks_empty_when_no_products(tmp_path: Path) -> None:
    """No committed ifg directory yields an empty list."""
    stack = _planned_stack(tmp_path)
    collection = StackInterferogramCollection(stack)

    assert collection.available_looks == []


def test_sel_looks_switches_the_read_view(tmp_path: Path) -> None:
    """sel(looks=) reads the other committed multilook tree."""
    stack, _p1, phase_10x4 = _stack_two_multilooks(tmp_path)
    collection = StackInterferogramCollection(stack)

    assert collection.looks == (1, 1)
    view = collection.sel(looks=(10, 4))
    assert isinstance(view, StackInterferogramCollection)
    assert view.looks == (10, 4)
    assert view.root.name == "ml_10x4"
    # Original view is untouched.
    assert collection.looks == (1, 1)
    assert collection.root.name == "ml_1x1"

    np.testing.assert_allclose(
        view.open(TARGET, "complex").values,
        np.exp(1j * phase_10x4).astype(np.complex64),
    )


def test_sel_looks_open_stack_uses_the_selected_tree(tmp_path: Path) -> None:
    """open_stack on a looks-selected view stacks the selected ml tree."""
    stack, _p1, phase_10x4 = _stack_two_multilooks(tmp_path)
    collection = StackInterferogramCollection(stack)
    view = collection.sel(looks=(10, 4))

    cube = view.open_stack("complex")
    assert cube.sizes["pair"] == len(PLANNED)
    np.testing.assert_allclose(
        cube.values[0],
        np.exp(1j * phase_10x4).astype(np.complex64),
    )


def test_sel_pair_scalar_demotes_to_layers(coregistered_stack: Stack) -> None:
    """sel(pairs=scalar) returns the same InterferogramLayers as __getitem__."""
    _write_ifg(coregistered_stack)
    collection = StackInterferogramCollection(coregistered_stack)
    pair = coregistered_stack.pairs[0]

    via_str = collection.sel(pairs="20240101_20240113")
    via_pair = collection.sel(pairs=pair)
    via_getitem = collection["20240101_20240113"]
    assert isinstance(via_str, InterferogramLayers)
    assert isinstance(via_pair, InterferogramLayers)
    assert isinstance(via_getitem, InterferogramLayers)
    np.testing.assert_allclose(
        via_str.wrapped_phase.values, via_getitem.wrapped_phase.values
    )
    np.testing.assert_allclose(
        via_pair.wrapped_phase.values, via_getitem.wrapped_phase.values
    )


def test_sel_pairs_narrows_the_read_view(tmp_path: Path) -> None:
    """sel(pairs=) restricts open/open_stack/exists/path/item/__getitem__."""
    stack, _p1, _p10 = _stack_two_multilooks(tmp_path)
    collection = StackInterferogramCollection(stack)
    subset = Pairs.from_names([TARGET, THIRD])
    view = collection.sel(pairs=subset)

    assert view.pairs == subset
    cube = view.open_stack("complex")
    assert cube.coords["pair"].values.tolist() == [TARGET, THIRD]
    assert cube.sizes["pair"] == 2
    assert view.exists(TARGET, "complex") is True
    assert view.path(TARGET, "complex").is_dir()
    assert "complex" in view.item(TARGET)["assets"]
    # A pair outside the subset is invisible to the view.
    with pytest.raises(KeyError):
        view.open(OTHER, "complex")
    with pytest.raises(KeyError):
        view[OTHER]


def test_sel_pairs_then_looks_composes(tmp_path: Path) -> None:
    """sel(pairs=).sel(looks=) applies both dims to the new view."""
    stack, _p1, phase_10x4 = _stack_two_multilooks(tmp_path)
    collection = StackInterferogramCollection(stack)
    subset = Pairs.from_names([TARGET, THIRD])
    view = collection.sel(pairs=subset).sel(looks=(10, 4))

    assert view.looks == (10, 4)
    assert view.pairs == subset
    np.testing.assert_allclose(
        view.open(TARGET, "complex").values,
        np.exp(1j * phase_10x4).astype(np.complex64),
    )


def test_sel_pairs_requires_subset_of_plan(tmp_path: Path) -> None:
    """pairs= outside the planned network or empty is rejected."""
    stack = _planned_stack(tmp_path)
    collection = StackInterferogramCollection(stack)

    with pytest.raises(ValueError, match="subset"):
        collection.sel(pairs=Pairs.from_names([UNPLANNED]))
    with pytest.raises(ValueError, match="subset"):
        collection.sel(pairs=Pairs.from_names([]))


def test_sel_rejects_invalid_args(tmp_path: Path) -> None:
    """Non-Pairs/non-scalar pairs=, empty sel, and malformed looks= fail."""
    stack = _planned_stack(tmp_path)
    collection = StackInterferogramCollection(stack)

    with pytest.raises(TypeError, match="Pairs, Pair, or str"):
        collection.sel(pairs=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Pairs, Pair, or str"):
        collection.sel(pairs=[TARGET])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        collection.sel()
    with pytest.raises(ValueError, match="looks"):
        collection.sel(looks=(0, 4))
    with pytest.raises(ValueError, match="looks"):
        collection.sel(looks=(10, 4, 2))  # type: ignore[arg-type]


def test_sel_looks_on_incomplete_view_fails_closed(tmp_path: Path) -> None:
    """sel(looks=) on a partially committed multilook fails closed."""
    stack = _planned_stack(tmp_path)
    _commit_all(stack, looks=(1, 1))
    # Only TARGET committed at (10, 4); OTHER/THIRD missing.
    _write_pair_product(
        stack, TARGET, phase=np.full((2, 2), 0.4, dtype=np.float32), looks=(10, 4)
    )
    collection = StackInterferogramCollection(stack)
    view = collection.sel(looks=(10, 4))

    # Single-pair read still works.
    np.testing.assert_allclose(
        view.open(TARGET, "complex").values,
        np.exp(1j * np.full((2, 2), 0.4, dtype=np.float32)).astype(np.complex64),
    )
    # Whole-network reads on that view fail closed.
    with pytest.raises(InvalidProcessingStateError):
        view.open_stack("complex")
    with pytest.raises(InvalidProcessingStateError):
        _ = view.index_metadata


def test_default_view_unchanged(tmp_path: Path) -> None:
    """stack.interferograms without sel keeps the config.multilook view."""
    stack, _p1, _p10 = _stack_two_multilooks(tmp_path)
    collection = StackInterferogramCollection(stack)

    assert collection.looks == (1, 1)
    assert collection.root.name == "ml_1x1"
    assert collection.pairs.to_names().tolist() == list(PLANNED)
    np.testing.assert_allclose(
        collection.open_stack("complex").values[0],
        np.exp(1j * np.full((3, 4), 0.2, dtype=np.float32)).astype(np.complex64),
    )


def test_sel_returns_independent_views(tmp_path: Path) -> None:
    """Two sel() calls yield views that do not share mutable selection state."""
    stack, _p1, _p10 = _stack_two_multilooks(tmp_path)
    collection = StackInterferogramCollection(stack)

    v1 = collection.sel(looks=(10, 4))
    v2 = collection.sel(pairs=Pairs.from_names([TARGET]))
    assert v1.looks == (10, 4)
    assert v2.looks == (1, 1)
    assert v2.pairs.to_names().tolist() == [TARGET]
    assert collection.looks == (1, 1)
    assert collection.pairs.to_names().tolist() == list(PLANNED)


def test_collection_repr_shows_pairs_table(tmp_path: Path) -> None:
    """Repr shows the view header plus the Pairs table."""
    stack, _p1, _p10 = _stack_two_multilooks(tmp_path)
    collection = StackInterferogramCollection(stack)

    text = repr(collection)
    assert "StackInterferogramCollection(3 pairs, looks=1x1" in text
    assert "primary" in text
    assert "secondary" in text
    assert "2024-01-01" in text
    assert "2024-01-25" in text
    assert str(collection) == "StackInterferogramCollection(3 pairs, looks=1x1)"

    subset = collection.sel(pairs=Pairs.from_names([TARGET]))
    assert "(1 pairs" in repr(subset)
    assert OTHER not in repr(subset)


def test_interferogram_layers_repr(coregistered_stack: Stack) -> None:
    """InterferogramLayers repr names the pair and accessible layers."""
    _write_ifg(coregistered_stack)
    layers = coregistered_stack.interferograms["20240101_20240113"]
    text = repr(layers)
    assert "InterferogramLayers(pair='20240101_20240113'" in text
    assert "wrapped_phase" in text
