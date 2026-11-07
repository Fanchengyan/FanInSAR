"""Public collection merge and layer crop/window tests (PROPOSAL-0074)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.io.storage.ifg_store import IFG_KIND, write_ifg_artifact
from faninsar.io.storage.product import open_product
from faninsar.network.layers import InterferogramLayers, ProductLayer

if TYPE_CHECKING:
    from pathlib import Path

_SOURCES = {"primary": "a" * 64, "secondary": "b" * 64}
_PHYSICAL_IDS = ("IW1-anx1000000", "IW2-anx2000000")


def _unique_complex(shape: tuple[int, int]) -> np.ndarray:
    """Return a complex field whose every sample is a distinct value."""
    rows, cols = shape
    real = np.arange(rows * cols, dtype=np.float32).reshape(shape)
    imag = np.arange(rows * cols, dtype=np.float32)[::-1].reshape(shape)
    return (real + 1j * imag).astype(np.complex64)


def _write_ifg(
    root: Path,
    *,
    shape: tuple[int, int] = (6, 8),
    complex_ifg: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    coherence: np.ndarray | None = None,
) -> InterferogramLayers:
    """Commit one IFG product and return its public layer view."""
    data = _unique_complex(shape) if complex_ifg is None else complex_ifg
    store = write_ifg_artifact(
        root,
        pair=("20240101", "20240113"),
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests=dict(_SOURCES),
        complex_ifg=data,
        coherence=(
            np.full(data.shape, 0.75, dtype=np.float32)
            if coherence is None
            else coherence
        ),
        valid_mask=valid_mask,
        replace_existing=True,
    )
    return InterferogramLayers(
        open_product(store.root, kind=IFG_KIND),
        store.shape,
        store.looks,
    )


def test_ifg_layer_read_window_slices_before_materializing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ProductLayer.read(window) returns the grid slice without a full load."""
    import zarr

    layers = _write_ifg(tmp_path / "ifg")
    layer = layers.complex
    full = layer.read()
    window = (2, 5, 1, 4)
    expected = full[2:5, 1:4]

    slices: list[object] = []
    real_open = zarr.open_array

    def open_array(*args: object, **kwargs: object) -> object:
        array = real_open(*args, **kwargs)

        class _Recorder:
            def __getitem__(self, key: object) -> np.ndarray:
                slices.append(key)
                return array[key]

            def __getattr__(self, name: str) -> object:
                return getattr(array, name)

        return _Recorder()

    monkeypatch.setattr(zarr, "open_array", open_array)
    got = layer.read(window=window)
    np.testing.assert_array_equal(got, expected)
    assert got.shape == (3, 3)

    def _elements(key: object) -> int | None:
        if not isinstance(key, tuple) or len(key) != 2:
            return None
        sizes: list[int] = []
        for dim, item in enumerate(key):
            if not isinstance(item, slice):
                return None
            start = 0 if item.start is None else int(item.start)
            stop = full.shape[dim] if item.stop is None else int(item.stop)
            sizes.append(stop - start)
        return sizes[0] * sizes[1]

    windowed = [count for count in (_elements(key) for key in slices) if count == 9]
    assert windowed, f"expected a 3x3 zarr slice before materializing; got {slices!r}"
    assert all(
        count is None or count <= 9 for count in (_elements(key) for key in slices)
    )


def _use_agg() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")


def test_ifg_layer_read_without_window_returns_full_array(tmp_path: Path) -> None:
    """Omitting window keeps the existing full-layer read."""
    layers = _write_ifg(tmp_path / "ifg")
    values = layers.complex.read()
    assert values.shape == (6, 8)
    np.testing.assert_array_equal(values, layers.complex.values)


def test_plot_crop_false_keeps_scientific_extent(tmp_path: Path) -> None:
    """crop=False plots the original scientific view, not a tightened bbox."""
    _use_agg()
    data = np.zeros((6, 8), dtype=np.complex64)
    data[2:4, 3:5] = 1.0 + 0.0j
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:4, 3:5] = True
    layers = _write_ifg(tmp_path / "ifg", complex_ifg=data, valid_mask=mask)
    _, _, im = layers.wrapped_phase.plot(crop=False, colorbar=False)
    plotted = np.asarray(im.get_array())
    assert plotted.shape == (6, 8)
    np.testing.assert_allclose(im.get_extent(), (-0.5, 7.5, 5.5, -0.5))


def test_plot_crop_true_preserves_hole_and_supported_zero(
    tmp_path: Path,
) -> None:
    """Interior holes stay holes; a valid 0j sample is not treated as empty."""
    _use_agg()
    data = np.ones((6, 8), dtype=np.complex64)
    data[1, 1] = 0.0 + 0.0j
    mask = np.ones((6, 8), dtype=bool)
    mask[3, 4] = False
    layers = _write_ifg(tmp_path / "ifg", complex_ifg=data, valid_mask=mask)
    _, _, im = layers.complex.plot(crop=True, colorbar=False)
    real_i, _real_q = np.asarray(im[0].get_array()), np.asarray(im[1].get_array())
    assert real_i.shape == (6, 8)
    assert real_i[1, 1] == 0.0
    assert np.isnan(real_i[3, 4])
    assert im[0].get_extent() == im[1].get_extent()


def test_plot_crop_true_preserves_disjoint_distance(tmp_path: Path) -> None:
    """Disconnected valid regions keep their original spatial separation."""
    _use_agg()
    data = np.zeros((6, 8), dtype=np.complex64)
    data[0, 0] = 1.0 + 0.0j
    data[5, 7] = 2.0 + 0.0j
    mask = np.zeros((6, 8), dtype=bool)
    mask[0, 0] = True
    mask[5, 7] = True
    layers = _write_ifg(tmp_path / "ifg", complex_ifg=data, valid_mask=mask)
    _, _, im = layers.wrapped_phase.plot(crop=True, colorbar=False)
    plotted = np.asarray(im.get_array())
    assert plotted.shape == (6, 8)
    assert not np.isnan(plotted[0, 0])
    assert not np.isnan(plotted[5, 7])
    assert np.isnan(plotted[2, 3])


def test_plot_empty_support_reports_no_data(tmp_path: Path) -> None:
    """Empty support draws a no-data panel instead of a zero image."""
    _use_agg()
    data = np.ones((4, 4), dtype=np.complex64)
    mask = np.zeros((4, 4), dtype=bool)
    layers = _write_ifg(
        tmp_path / "ifg", shape=(4, 4), complex_ifg=data, valid_mask=mask
    )
    _, ax, im = layers.wrapped_phase.plot(crop=True, colorbar=False)
    assert im is None
    texts = [child.get_text() for child in ax.texts]
    assert "no data" in texts


def test_selected_burst_crop_uses_full_footprint(tmp_path: Path) -> None:
    """A selected burst plots its ROI footprint, not an owner==burst trim."""
    from faninsar.io.storage.scene_store import write_scene

    _use_agg()
    left = np.full((8, 8), 1.0 + 0.0j, dtype=np.complex64)
    right = np.full((8, 8), 2.0 + 0.0j, dtype=np.complex64)
    store = write_scene(
        tmp_path / "scene",
        date_id="20240101",
        reference_id="20240101",
        domain="radar",
        units=[
            {"tag": "f0_IW1_b0", "payload": left, "row_origin": 0, "col_origin": 0},
            {"tag": "f0_IW2_b0", "payload": right, "row_origin": 0, "col_origin": 8},
        ],
        grid_shape=(8, 16),
        overwrite=True,
    )
    selected = store.layers.sel(bursts=("f0_IW1_b0",))
    assert selected.slc.extent == (8, 8)
    _, _, im = selected.slc.plot(crop=True, colorbar=False)
    real_i = np.asarray(im[0].get_array())
    assert real_i.shape == (8, 8)
    np.testing.assert_allclose(real_i, 1.0)


def test_scenes_merge_bursts_returns_lazy_collection(tmp_path: Path) -> None:
    """scenes.merge_bursts() returns the same collection type per date."""
    from faninsar.network.readers.stack import StackSceneCollection
    from tests.network.test_scene_collection import _stack_with_scenes

    stack = _stack_with_scenes(tmp_path)
    original = stack.scenes
    merged = original.merge_bursts()
    assert isinstance(merged, StackSceneCollection)
    assert merged is not original
    assert original.dates == merged.dates
    assert original["20240113"].layers.slc._merge_on_read is False
    assert merged["20240113"].layers.slc._merge_on_read is True


def test_interferograms_sel_representation_and_merge_bursts(
    tmp_path: Path,
) -> None:
    """sel(representation=) defaults to merged; merge_bursts stays a collection."""
    from tests.network.test_scene_collection import _stack_with_scenes

    stack = _stack_with_scenes(tmp_path)
    collection = stack.interferograms
    with pytest.raises(ValueError, match="representation"):
        collection.sel(representation="average")  # type: ignore[arg-type]
    burst_view = collection.sel(representation="burst", burst=0)
    assert burst_view._selection.representation == "burst"
    assert burst_view._selection.burst_ids == ("0",)
    merged = collection.merge_bursts()
    assert type(merged) is type(collection)
    assert merged._selection.merge_on_read is True
    assert collection._selection.merge_on_read is False


def test_plot_crop_true_reads_tight_interior_window(tmp_path: Path) -> None:
    """crop=True window-reads the tight valid bbox, not the full scientific frame."""
    _use_agg()
    data = np.zeros((6, 8), dtype=np.complex64)
    data[2:4, 3:5] = 1.0 + 0.0j
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:4, 3:5] = True
    layers = _write_ifg(tmp_path / "ifg", complex_ifg=data, valid_mask=mask)
    _, _, im = layers.wrapped_phase.plot(crop=True, colorbar=False)
    plotted = np.asarray(im.get_array())
    assert plotted.shape == (2, 2)
    np.testing.assert_allclose(im.get_extent(), (2.5, 4.5, 3.5, 1.5))


def _merge_scope(stack: object) -> object:
    """Return a radar MergeScope bound to the Stack Reference grid."""
    from faninsar.processing.mosaicking import MergeScope

    store = stack.scenes[stack.reference]
    return MergeScope(
        domain="radar",
        track_id="T000_A",
        channel="vv",
        reference_id=str(stack.reference),
        grid_identity=str(store.grid_identity),
        polygon_identity="poly-0",
    )


def _publish_stack_layout(
    stack: object,
    *,
    source_ids: tuple[str, str],
    window: tuple[int, int, int, int] | None = None,
) -> None:
    """Publish a unique-coverage radar layout for one Stack Reference."""
    from faninsar.io.storage.merge_layout import publish_merge_layout
    from faninsar.processing.mosaicking import LayoutSourceWindow, build_merge_layout

    layout = build_merge_layout(
        (
            LayoutSourceWindow(
                source_ids[0],
                (0, 8, 0, 8),
                np.ones((8, 8), dtype=bool),
                "IW1",
            ),
            LayoutSourceWindow(
                source_ids[1],
                (0, 8, 8, 16),
                np.ones((8, 8), dtype=bool),
                "IW2",
            ),
        ),
        scope=_merge_scope(stack),
        window=window,
    )
    publish_merge_layout(
        stack.config.work_dir / "layout" / str(stack.reference),
        layout,
        overwrite=True,
    )


def _publish_overlap_layout(stack: object) -> None:
    """Publish a range-overlap layout whose midpoint cut differs from first-valid."""
    from faninsar.io.storage.merge_layout import publish_merge_layout
    from faninsar.processing.mosaicking import (
        LayoutSourceWindow,
        MergeSeam,
        build_merge_layout,
    )

    layout = build_merge_layout(
        (
            LayoutSourceWindow(
                "f0_IW1_b0",
                (0, 8, 0, 10),
                np.ones((8, 10), dtype=bool),
                "IW1",
            ),
            LayoutSourceWindow(
                "f0_IW2_b0",
                (0, 8, 6, 16),
                np.ones((8, 10), dtype=bool),
                "IW2",
            ),
        ),
        scope=_merge_scope(stack),
        seams=(
            MergeSeam(
                axis="range",
                boundary=8,
                earlier_id="f0_IW1_b0",
                later_id="f0_IW2_b0",
            ),
        ),
    )
    publish_merge_layout(
        stack.config.work_dir / "layout" / str(stack.reference),
        layout,
        overwrite=True,
    )


def _write_date_units(
    stack: object,
    date_id: str,
    units: list[dict[str, object]],
) -> None:
    """Overwrite one committed scene with *units* on the Reference grid."""
    from faninsar.io.storage.scene_store import write_scene
    from tests.support.scene_stores import coreg_dir

    write_scene(
        coreg_dir(stack.config.work_dir, date_id),
        date_id=date_id,
        reference_id=stack.reference,
        domain="radar",
        units=units,
        grid_shape=(8, 16),
        inputs={"reference": stack.scenes["20240101"].write_id},
        overwrite=True,
    )


def test_scenes_merge_bursts_read_requires_committed_layout(tmp_path: Path) -> None:
    """merge_bursts() is lazy, but a read without a layout product is rejected."""
    from faninsar.processing.errors import InvalidProcessingStateError
    from tests.network.test_scene_collection import _stack_with_scenes

    stack = _stack_with_scenes(tmp_path)
    merged = stack.scenes.merge_bursts()
    with pytest.raises(InvalidProcessingStateError, match="merge_layout"):
        merged["20240113"].layers.slc.read()


def test_scenes_merge_bursts_reads_committed_layout(tmp_path: Path) -> None:
    """Per-date merge-on-read uses the published layout, not burst bounding boxes."""
    from faninsar.io.storage.scene_store import write_scene
    from tests.network.test_scene_collection import _stack_with_scenes
    from tests.support.scene_stores import coreg_dir

    stack = _stack_with_scenes(tmp_path)
    left = np.full((8, 8), 1.0 + 0.0j, dtype=np.complex64)
    right = np.full((8, 8), 2.0 + 0.0j, dtype=np.complex64)
    write_scene(
        coreg_dir(stack.config.work_dir, "20240113"),
        date_id="20240113",
        reference_id=stack.reference,
        domain="radar",
        units=[
            {
                "tag": "f0_IW1_b0",
                "payload": left,
                "row_origin": 0,
                "col_origin": 0,
                "burst_id": "f0_IW1_b0",
            },
            {
                "tag": "f0_IW2_b0",
                "payload": right,
                "row_origin": 0,
                "col_origin": 8,
                "burst_id": "f0_IW2_b0",
            },
        ],
        grid_shape=(8, 16),
        inputs={"reference": stack.scenes["20240101"].write_id},
        overwrite=True,
    )
    _publish_stack_layout(stack, source_ids=("f0_IW1_b0", "f0_IW2_b0"))
    values = stack.scenes.merge_bursts()["20240113"].layers.slc.read()
    assert values.shape == (8, 16)
    np.testing.assert_allclose(values[:, :8], 1.0)
    np.testing.assert_allclose(values[:, 8:], 2.0)


def test_interferograms_merge_bursts_reads_committed_layout(tmp_path: Path) -> None:
    """Per-pair merge-on-read selects burst IFGs through the committed layout."""
    from faninsar.io.storage.ifg_store import ifg_artifact_dir, write_ifg_artifact
    from tests.network.test_scene_collection import _stack_with_scenes

    stack = _stack_with_scenes(tmp_path)
    _publish_stack_layout(stack, source_ids=("f0_IW1_b0", "f0_IW2_b0"))
    pair = ("20240101", "20240113")
    primary = stack.scenes["20240101"].write_id
    secondary = stack.scenes["20240113"].write_id
    for burst_id, payload, col0 in (
        ("f0_IW1_b0", np.full((8, 8), 1.0 + 0.0j, dtype=np.complex64), 0),
        ("f0_IW2_b0", np.full((8, 8), 2.0 + 0.0j, dtype=np.complex64), 8),
    ):
        write_ifg_artifact(
            ifg_artifact_dir(
                stack.config.work_dir,
                (1, 1),
                pair,
                representation="burst",
                burst_id=burst_id,
            ),
            pair=pair,
            looks=(1, 1),
            filter_name="none",
            filter_parameters={},
            source_manifest_digests={"primary": primary, "secondary": secondary},
            complex_ifg=payload,
            valid_mask=np.ones((8, 8), dtype=bool),
            representation="burst",
            burst_id=burst_id,
            row_origin=0,
            col_origin=col0,
            grid_shape=(8, 16),
        )
    values = stack.interferograms.merge_bursts()["20240101_20240113"].complex.read()
    assert values.shape == (8, 16)
    np.testing.assert_allclose(values[:, :8], 1.0)
    np.testing.assert_allclose(values[:, 8:], 2.0)


def test_merge_to_xarray_uses_shared_midpoint_selector(tmp_path: Path) -> None:
    """Lazy xarray compute must match merge_bursts, not first-valid mosaicking."""
    from tests.network.test_scene_collection import _stack_with_scenes

    stack = _stack_with_scenes(tmp_path)
    _write_date_units(
        stack,
        "20240113",
        [
            {
                "tag": "f0_IW1_b0",
                "payload": np.full((8, 10), 1.0 + 0.0j, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 0,
                "burst_id": "f0_IW1_b0",
            },
            {
                "tag": "f0_IW2_b0",
                "payload": np.full((8, 10), 2.0 + 0.0j, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 6,
                "burst_id": "f0_IW2_b0",
            },
        ],
    )
    _publish_overlap_layout(stack)
    layer = stack.scenes.merge_bursts()["20240113"].layers.slc
    eager = layer.read()
    lazy = np.asarray(layer.to_xarray().compute())
    np.testing.assert_array_equal(lazy, eager)
    np.testing.assert_allclose(eager[:, :8], 1.0)
    np.testing.assert_allclose(eager[:, 8:], 2.0)


def test_merge_window_read_keeps_owner_labels_on_shifted_layout(
    tmp_path: Path,
) -> None:
    """Cropped owner labels stay 1-based and aligned to a non-zero layout origin."""
    from tests.network.test_scene_collection import _stack_with_scenes

    stack = _stack_with_scenes(tmp_path)
    _write_date_units(
        stack,
        "20240113",
        [
            {
                "tag": "f0_IW1_b0",
                "payload": np.full((8, 8), 1.0 + 0.0j, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 0,
                "burst_id": "f0_IW1_b0",
            },
            {
                "tag": "f0_IW2_b0",
                "payload": np.full((8, 8), 2.0 + 0.0j, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 8,
                "burst_id": "f0_IW2_b0",
            },
        ],
    )
    _publish_stack_layout(
        stack, source_ids=("f0_IW1_b0", "f0_IW2_b0"), window=(2, 6, 4, 12)
    )
    layer = stack.scenes.merge_bursts()["20240113"].layers.slc
    roi = layer.read(window=(2, 6, 4, 12))
    assert roi.shape == (4, 8)
    np.testing.assert_allclose(roi[:, :4], 1.0)
    np.testing.assert_allclose(roi[:, 4:], 2.0)
    inner = layer.read(window=(3, 5, 6, 10))
    assert inner.shape == (2, 4)
    np.testing.assert_allclose(inner[:, :2], 1.0)
    np.testing.assert_allclose(inner[:, 2:], 2.0)


def test_merge_rejects_mismatched_phase_conventions(tmp_path: Path) -> None:
    """Burst windows that disagree on phase convention fail closed at read."""
    from faninsar.processing.errors import InvalidProcessingStateError
    from tests.network.test_scene_collection import _stack_with_scenes

    stack = _stack_with_scenes(tmp_path)
    _write_date_units(
        stack,
        "20240113",
        [
            {
                "tag": "f0_IW1_b0",
                "payload": np.full((8, 8), 1.0 + 0.0j, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 0,
                "burst_id": "f0_IW1_b0",
                "phase_state": {"convention": "reramped"},
            },
            {
                "tag": "f0_IW2_b0",
                "payload": np.full((8, 8), 2.0 + 0.0j, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 8,
                "burst_id": "f0_IW2_b0",
                "phase_state": {"convention": "deramped"},
            },
        ],
    )
    _publish_stack_layout(stack, source_ids=("f0_IW1_b0", "f0_IW2_b0"))
    with pytest.raises(InvalidProcessingStateError, match="phase_convention"):
        stack.scenes.merge_bursts()["20240113"].layers.slc.read()


def test_merge_read_rejects_replaced_layout_product(tmp_path: Path) -> None:
    """A layer bound to a replaced merge_layout product must not keep reading."""
    from faninsar.processing.errors import InvalidProcessingStateError
    from tests.network.test_scene_collection import _stack_with_scenes

    stack = _stack_with_scenes(tmp_path)
    _write_date_units(
        stack,
        "20240113",
        [
            {
                "tag": "f0_IW1_b0",
                "payload": np.full((8, 8), 1.0 + 0.0j, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 0,
                "burst_id": "f0_IW1_b0",
            },
            {
                "tag": "f0_IW2_b0",
                "payload": np.full((8, 8), 2.0 + 0.0j, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 8,
                "burst_id": "f0_IW2_b0",
            },
        ],
    )
    _publish_stack_layout(stack, source_ids=("f0_IW1_b0", "f0_IW2_b0"))
    layer = stack.scenes.merge_bursts()["20240113"].layers.slc
    np.testing.assert_allclose(layer.read()[:, :8], 1.0)
    _publish_stack_layout(stack, source_ids=("f0_IW1_b0", "f0_IW2_b0"))
    with pytest.raises(InvalidProcessingStateError, match="replaced"):
        layer.read()


def test_scenes_merge_uses_physical_burst_id_not_unit_tag(tmp_path: Path) -> None:
    """Layout source_ids are physical ANX ids; local tags must not be selected."""
    from tests.network.test_scene_collection import _stack_with_scenes

    stack = _stack_with_scenes(tmp_path)
    _write_date_units(
        stack,
        "20240113",
        [
            {
                "tag": "f0_IW1_b0",
                "payload": np.full((8, 8), 1.0 + 0.0j, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 0,
                "burst_id": _PHYSICAL_IDS[0],
            },
            {
                "tag": "f0_IW2_b0",
                "payload": np.full((8, 8), 2.0 + 0.0j, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 8,
                "burst_id": _PHYSICAL_IDS[1],
            },
        ],
    )
    _publish_stack_layout(stack, source_ids=_PHYSICAL_IDS)
    values = stack.scenes.merge_bursts()["20240113"].layers.slc.read()
    np.testing.assert_allclose(values[:, :8], 1.0)
    np.testing.assert_allclose(values[:, 8:], 2.0)


def test_scenes_merge_rejects_tag_only_units_missing_burst_id(
    tmp_path: Path,
) -> None:
    """A local tag is not a substitute for the persisted physical burst id."""
    from faninsar.processing.errors import InvalidProcessingStateError
    from tests.network.test_scene_collection import _stack_with_scenes

    stack = _stack_with_scenes(tmp_path)
    _write_date_units(
        stack,
        "20240113",
        [
            {
                "tag": "f0_IW1_b0",
                "payload": np.full((8, 8), 1.0 + 0.0j, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 0,
            },
            {
                "tag": "f0_IW2_b0",
                "payload": np.full((8, 8), 2.0 + 0.0j, dtype=np.complex64),
                "row_origin": 0,
                "col_origin": 8,
            },
        ],
    )
    _publish_stack_layout(stack, source_ids=_PHYSICAL_IDS)
    with pytest.raises(InvalidProcessingStateError, match="burst_id"):
        stack.scenes.merge_bursts()["20240113"].layers.slc.read()


def test_interferograms_merge_uses_persisted_physical_burst_id(
    tmp_path: Path,
) -> None:
    """Burst IFG merge keys on faninsar:burst_id, not a local path basename guess."""
    from faninsar.io.storage.ifg_store import ifg_artifact_dir, write_ifg_artifact
    from tests.network.test_scene_collection import _stack_with_scenes

    stack = _stack_with_scenes(tmp_path)
    _publish_stack_layout(stack, source_ids=_PHYSICAL_IDS)
    pair = ("20240101", "20240113")
    primary = stack.scenes["20240101"].write_id
    secondary = stack.scenes["20240113"].write_id
    for burst_id, payload, col0 in (
        (_PHYSICAL_IDS[0], np.full((8, 8), 1.0 + 0.0j, dtype=np.complex64), 0),
        (_PHYSICAL_IDS[1], np.full((8, 8), 2.0 + 0.0j, dtype=np.complex64), 8),
    ):
        write_ifg_artifact(
            ifg_artifact_dir(
                stack.config.work_dir,
                (1, 1),
                pair,
                representation="burst",
                burst_id=burst_id,
            ),
            pair=pair,
            looks=(1, 1),
            filter_name="none",
            filter_parameters={},
            source_manifest_digests={"primary": primary, "secondary": secondary},
            complex_ifg=payload,
            valid_mask=np.ones((8, 8), dtype=bool),
            representation="burst",
            burst_id=burst_id,
            row_origin=0,
            col_origin=col0,
            grid_shape=(8, 16),
        )
    values = stack.interferograms.merge_bursts()["20240101_20240113"].complex.read()
    np.testing.assert_allclose(values[:, :8], 1.0)
    np.testing.assert_allclose(values[:, 8:], 2.0)
