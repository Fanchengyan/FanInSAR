"""StackSceneCollection session-binding tests (PROPOSAL-0063)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.io.storage.scene_store import CoregisteredSceneStore, SceneLayers
from faninsar.missions.s1 import S1Stack
from faninsar.network.readers.stack import StackSceneCollection

if TYPE_CHECKING:
    from pathlib import Path


def _stack_with_scenes(tmp_path: Path) -> S1Stack:
    """Build a two-date S1 Stack with committed scene products."""
    from faninsar import Pairs

    dates = ("20240101", "20240113", "20240125")
    paths = []
    for date_id in dates:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000.SAFE"
        path.mkdir()
        paths.append(path)
    stack = S1Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        pairs=Pairs.from_names(["20240101_20240113", "20240113_20240125"]),
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
    """Scenes returns the collection with session-bound layer views."""
    stack = _stack_with_scenes(tmp_path)
    scenes = stack.scenes
    assert isinstance(scenes, StackSceneCollection)
    store = scenes["20240113"]
    assert isinstance(store, CoregisteredSceneStore)
    layers = store.layers
    assert layers._session is stack
    assert layers.slc._session is stack


def test_scenes_getitem_acquisition_index(tmp_path: Path) -> None:
    """An Acquisition index returns per-date SceneLayers keyed on dates."""
    import pandas as pd

    from faninsar.core.acquisition import Acquisition

    stack = _stack_with_scenes(tmp_path)
    index = Acquisition(pd.DatetimeIndex(["2024-01-01", "2024-01-13"]))
    out = stack.scenes[index]
    assert set(out) == {"20240101", "20240113"}
    for view in out.values():
        assert view._session is stack


def test_scenes_acquisitions_and_reference(tmp_path: Path) -> None:
    """The acquisitions index and reference property describe the view."""
    import pandas as pd

    stack = _stack_with_scenes(tmp_path)
    scenes = stack.scenes
    expected = pd.DatetimeIndex(["2024-01-01", "2024-01-13", "2024-01-25"])
    pd.testing.assert_index_equal(pd.DatetimeIndex(scenes.acquisitions), expected)
    assert scenes.reference == stack.reference
    assert scenes.dates == ["20240101", "20240113", "20240125"]


def test_scenes_sel_dates_list_and_slice(tmp_path: Path) -> None:
    """sel(dates=) narrows the view by list or date slice."""
    stack = _stack_with_scenes(tmp_path)
    scenes = stack.scenes

    view = scenes.sel(dates=["2024-01-13", "2024-01-25"])
    assert isinstance(view, StackSceneCollection)
    assert view.dates == ["20240113", "20240125"]
    assert scenes.dates == ["20240101", "20240113", "20240125"]

    sliced = scenes.sel(dates=slice("2024-01-10", "2024-12-31"))
    assert sliced.dates == ["20240113", "20240125"]
    open_ended = scenes.sel(dates=slice("2024-01-10", None))
    assert open_ended.dates == ["20240113", "20240125"]


def test_scenes_sel_scalar_demotes_to_layers(tmp_path: Path) -> None:
    """sel(dates=scalar) demotes to that date's SceneLayers."""
    stack = _stack_with_scenes(tmp_path)
    scenes = stack.scenes
    layers = scenes.sel(dates="20240113")
    assert isinstance(layers, SceneLayers)
    assert layers._session is stack
    assert layers._store.date_id == "20240113"


def test_scenes_sel_rejects_invalid_args(tmp_path: Path) -> None:
    """Sel() rejects empty, unknown, and malformed selectors."""
    stack = _stack_with_scenes(tmp_path)
    scenes = stack.scenes

    with pytest.raises(TypeError, match="dates"):
        scenes.sel()
    with pytest.raises(KeyError):
        scenes.sel(dates="20250101")
    with pytest.raises(KeyError, match="not committed"):
        scenes.sel(dates=["20250101"])
    with pytest.raises(ValueError, match="empty"):
        scenes.sel(dates=slice("2030-01-01", "2030-12-31"))
    with pytest.raises(TypeError, match="dates must be"):
        scenes.sel(dates=123)


def test_scenes_getitem_date_list(tmp_path: Path) -> None:
    """A list of date strings returns stores keyed on dates."""
    stack = _stack_with_scenes(tmp_path)
    stores = stack.scenes[["2024-01-13", "2024-01-25"]]
    assert set(stores) == {"20240113", "20240125"}
    for store in stores.values():
        assert isinstance(store, CoregisteredSceneStore)
        assert store.layers._session is stack


def test_scenes_getitem_acquisition_record(tmp_path: Path) -> None:
    """A record-shaped Acquisition returns the bound scene store."""
    import pandas as pd

    from faninsar.core.acquisition import Acquisition
    from faninsar.network.products import AcquisitionKey

    stack = _stack_with_scenes(tmp_path)
    key = AcquisitionKey(
        acquisition_id="20240113",
        frame="f0",
        swath="IW1",
        channel="slc",
        polarization="VV",
    )
    record = Acquisition(key, pd.Timestamp("2024-01-13"))
    store = stack.scenes[record]
    assert isinstance(store, CoregisteredSceneStore)
    assert store.date_id == "20240113"
    assert store.layers._session is stack


def test_scenes_repr_shows_table(tmp_path: Path) -> None:
    """Repr shows a Pairs-like table of committed scenes."""
    stack = _stack_with_scenes(tmp_path)
    text = repr(stack.scenes)
    assert "StackSceneCollection(3 scenes" in text
    assert "reference='20240101'" in text
    assert "secondary" in text
    assert "reference" in text
    assert str(stack.scenes) == ("StackSceneCollection(3 scenes, reference='20240101')")


def test_scenes_getitem_str_binds_geocode(tmp_path: Path) -> None:
    """A str date returns the store bound to the geocode session."""
    stack = _stack_with_scenes(tmp_path)
    store = stack.scenes["20240113"]
    # The bound store's layers reach geocode through the injected session.
    assert store.layers.slc._session is stack
    assert store.layers.slc.extent == (8, 16)  # union bbox of both bursts
    assert store.layers.slc.values.shape == (8, 16)


def test_unbound_store_layers_geocode_raises(tmp_path: Path) -> None:
    """An unbound store's layers raise on geocode()."""
    stack = _stack_with_scenes(tmp_path)
    store = CoregisteredSceneStore.open(stack.config.work_dir / "coreg" / "20240113")
    layer = store.layers.slc
    assert layer._session is None
    with pytest.raises(Exception, match="not bound to a Stack session"):
        layer.geocode()


def test_scene_layer_extent_and_origin(tmp_path: Path) -> None:
    """sel(bursts=) narrows the layer extent to the union bbox."""
    stack = _stack_with_scenes(tmp_path)
    sel = stack.scenes["20240113"].layers.sel(bursts=("f0_IW1_b0",))
    assert sel.slc.extent == (8, 8)
    assert sel.slc.origin == (0, 0)
    assert sel.slc.values.shape == (8, 8)
