"""StackSceneCollection session-binding tests (PROPOSAL-0063)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.io.storage.scene_store import CoregisteredSceneStore
from faninsar.missions.s1 import S1Stack
from faninsar.network.readers.stack import StackSceneCollection

if TYPE_CHECKING:
    from pathlib import Path


def _stack_with_scenes(tmp_path: Path) -> S1Stack:
    """Build a two-date S1 Stack with committed scene products."""
    from faninsar import Pairs

    dates = ("20240101", "20240113")
    paths = []
    for date_id in dates:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000.SAFE"
        path.mkdir()
        paths.append(path)
    stack = S1Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        pairs=Pairs.from_names(["20240101_20240113"]),
        multilook=(1, 1),
    )

    def units() -> list[dict[str, object]]:
        # Two bursts side by side: union bbox (8, 16) == declared grid.
        return [
            {
                "tag": "f0_IW1_b0",
                "payload": np.ones((8, 8), np.complex64),
                "row_origin": 0,
                "col_origin": 0,
            },
            {
                "tag": "f0_IW2_b0",
                "payload": np.ones((8, 8), np.complex64),
                "row_origin": 0,
                "col_origin": 8,
            },
        ]

    from faninsar.io.storage.scene_store import write_scene
    from tests.support.scene_stores import coreg_dir

    reference_root = coreg_dir(stack.config.work_dir, stack.reference)
    reference = write_scene(
        reference_root,
        date_id=stack.reference,
        reference_id=stack.reference,
        domain="radar",
        units=units(),
        grid_shape=(8, 16),
    )
    for date_id in dates:
        if date_id == stack.reference:
            continue
        write_scene(
            coreg_dir(stack.config.work_dir, date_id),
            date_id=date_id,
            reference_id=stack.reference,
            domain="radar",
            units=units(),
            grid_shape=(8, 16),
            inputs={"reference": reference.write_id},
        )
    return stack


def test_scenes_returns_collection_and_binds_session(tmp_path: Path) -> None:
    stack = _stack_with_scenes(tmp_path)
    scenes = stack.scenes
    assert isinstance(scenes, StackSceneCollection)
    store = scenes["20240113"]
    assert isinstance(store, CoregisteredSceneStore)
    layers = store.layers
    assert layers._session is stack
    assert layers.slc._session is stack


def test_scenes_getitem_acquisition_index(tmp_path: Path) -> None:
    import pandas as pd

    from faninsar.core.acquisition import Acquisition

    stack = _stack_with_scenes(tmp_path)
    index = Acquisition(pd.DatetimeIndex(["2024-01-01", "2024-01-13"]))
    out = stack.scenes[index]
    assert set(out) == {"20240101", "20240113"}
    for view in out.values():
        assert view._session is stack


def test_scenes_getitem_str_binds_geocode(tmp_path: Path) -> None:
    stack = _stack_with_scenes(tmp_path)
    store = stack.scenes["20240113"]
    # The bound store's layers reach geocode through the injected session.
    assert store.layers.slc._session is stack
    assert store.layers.slc.extent == (8, 16)  # union bbox of both bursts
    assert store.layers.slc.values.shape == (8, 16)


def test_unbound_store_layers_geocode_raises(tmp_path: Path) -> None:
    stack = _stack_with_scenes(tmp_path)
    store = CoregisteredSceneStore.open(stack.config.work_dir / "coreg" / "20240113")
    layer = store.layers.slc
    assert layer._session is None
    with pytest.raises(Exception, match="not bound to a Stack session"):
        layer.geocode()


def test_scene_layer_extent_and_origin(tmp_path: Path) -> None:
    stack = _stack_with_scenes(tmp_path)
    sel = stack.scenes["20240113"].layers.select(bursts=("f0_IW1_b0",))
    assert sel.slc.extent == (8, 8)
    assert sel.slc.origin == (0, 0)
    assert sel.slc.values.shape == (8, 8)
