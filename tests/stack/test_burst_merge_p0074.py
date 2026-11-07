"""P0074 Stack session seams: merge defaults, discovery, burst IFG paths."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tests.support.local_stack import LocalStack
from tests.support.scene_stores import write_test_scene

from faninsar.io.storage.ifg_store import InterferogramArtifactStore, ifg_artifact_dir
from faninsar.io.storage._zarr_stac import read_zarr_array
from faninsar.io.storage.merge_layout import (
    LAYOUT_KIND,
    open_merge_layout,
    publish_merge_layout,
)
from faninsar.io.storage.product import open_product
from faninsar.io.storage.scene_store import (
    MERGED_SCENE_UNIT_TAG,
    CoregisteredSceneStore,
    build_scene_merge_layout,
    write_scene,
)
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.mosaicking import MergeSeam
from faninsar.stack.session import Stack

if TYPE_CHECKING:
    from pathlib import Path


def _two_date_stack(tmp_path: Path) -> Stack:
    """Return a prepared two-date LocalStack with coregistered scenes."""
    dates = ("20160101", "20160113")
    safe_paths = []
    for date_id in dates:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000_{date_id}T000001.SAFE"
        path.mkdir()
        safe_paths.append(path)
    stack = LocalStack.from_paths(safe_paths, work_dir=tmp_path / "out")
    stack.prepare_data()
    reference = np.ones((2, 3), dtype=np.complex64)
    secondary = np.full((2, 3), 1.0 + 2.0j, dtype=np.complex64)
    for date_id in dates:
        write_test_scene(
            stack.config.work_dir,
            date_id=date_id,
            reference_id=dates[0],
            payload=reference if date_id == dates[0] else secondary,
            reference_payload=reference,
            tag=("IW1_b0", "IW1_b1"),
        )
    reference_store = CoregisteredSceneStore.open(
        stack.config.work_dir / "coreg" / dates[0]
    )
    layout = build_scene_merge_layout(
        reference_store,
        seams=(
            MergeSeam(
                axis="azimuth",
                boundary=1,
                earlier_id="IW1_b0",
                later_id="IW1_b1",
            ),
        ),
    )
    publish_merge_layout(stack.config.work_dir / "layout" / dates[0], layout)
    return stack


def test_coregister_and_formation_merge_defaults() -> None:
    """Coregistration keeps bursts; IFG formation merges by default."""
    assert inspect.signature(Stack.coregister_scenes).parameters["merge"].default is False
    assert inspect.signature(Stack.form_interferograms).parameters["merge"].default is True


def test_scene_discovery_skips_merged_container(tmp_path: Path) -> None:
    """``coreg/merged/<date>`` is not a burst scene date root."""
    stack = _two_date_stack(tmp_path)
    payload = np.ones((2, 3), dtype=np.complex64)
    write_scene(
        stack.config.work_dir / "coreg" / "merged" / "20160101",
        date_id="20160101",
        reference_id="20160101",
        domain="radar",
        units=[
            {
                "tag": "IW1_b0",
                "payload": payload,
                "row_origin": 0,
                "col_origin": 0,
            }
        ],
        properties={"faninsar:representation": "merged"},
    )
    stores = stack._scene_stores()
    assert set(stores) == {"20160101", "20160113"}
    assert all(store.representation != "merged" for store in stores.values())
    assert "merged" not in stack.scenes


def test_form_interferograms_merge_false_writes_burst_products(
    tmp_path: Path,
) -> None:
    """Independent burst IFGs persist beside the default merged product path."""
    stack = _two_date_stack(tmp_path)
    stack.form_interferograms(multilook=(1, 1), merge=False, phase_filter=None)
    burst_root = ifg_artifact_dir(
        stack.config.work_dir,
        (1, 1),
        ("20160101", "20160113"),
        representation="burst",
        burst_id="IW1_b0",
    )
    store = InterferogramArtifactStore.open(burst_root)
    assert store.representation == "burst"
    assert store.burst_id == "IW1_b0"
    np.testing.assert_array_equal(
        store.read().complex, np.full((2, 3), 1.0 - 2.0j, dtype=np.complex64)
    )
    merged_root = ifg_artifact_dir(
        stack.config.work_dir, (1, 1), ("20160101", "20160113")
    )
    assert not (merged_root / "item.json").is_file()


def test_form_interferograms_default_writes_merged_representation(
    tmp_path: Path,
) -> None:
    """Default formation still publishes the merged Pair product."""
    stack = _two_date_stack(tmp_path)
    stack.form_interferograms(multilook=(1, 1), phase_filter=None)
    root = ifg_artifact_dir(stack.config.work_dir, (1, 1), ("20160101", "20160113"))
    store = InterferogramArtifactStore.open(root)
    assert store.representation == "merged"
    assert store.layout_id
    layout_root = stack.config.work_dir / "layout" / "20160101"
    assert open_product(layout_root, kind=LAYOUT_KIND).write_id == store.layout_id
    open_merge_layout(layout_root)
    np.testing.assert_array_equal(
        store.read().complex, np.full((2, 3), 1.0 - 2.0j, dtype=np.complex64)
    )


def test_merged_scene_product_is_distinct_from_burst_date_root(
    tmp_path: Path,
) -> None:
    """Explicit merged SLCs live under ``coreg/merged/<date>``, not the burst root."""
    stack = _two_date_stack(tmp_path)
    burst_root = stack.config.work_dir / "coreg" / "20160113"
    payload = np.ones((2, 3), dtype=np.complex64)
    write_scene(
        stack.config.work_dir / "coreg" / "merged" / "20160113",
        date_id="20160113",
        reference_id="20160101",
        domain="radar",
        units=[
            {
                "tag": "IW1_b0",
                "payload": payload,
                "row_origin": 0,
                "col_origin": 0,
            }
        ],
        properties={"faninsar:representation": "merged"},
        inputs={"reference": CoregisteredSceneStore.open(burst_root).write_id},
    )
    assert (burst_root / "item.json").is_file()
    burst = CoregisteredSceneStore.open(burst_root)
    merged = CoregisteredSceneStore.open(
        stack.config.work_dir / "coreg" / "merged" / "20160113"
    )
    assert burst.representation == "burst"
    assert merged.representation == "merged"
    assert stack._scene_stores()["20160113"].root == burst_root


def test_publish_merged_scenes_consumes_layout_and_keeps_burst_root(
    tmp_path: Path,
) -> None:
    """Explicit scene merge consumes the committed layout and keeps burst SLCs."""
    stack = _two_date_stack(tmp_path)
    burst_root = stack.config.work_dir / "coreg" / "20160113"
    burst = CoregisteredSceneStore.open(burst_root)
    stack._publish_merged_scenes(["20160113"], [None], overwrite=True)
    merged = CoregisteredSceneStore.open(
        stack.config.work_dir / "coreg" / "merged" / "20160113"
    )
    layout = open_merge_layout(stack.config.work_dir / "layout" / "20160101")
    assert (burst_root / "item.json").is_file()
    assert CoregisteredSceneStore.open(burst_root).write_id == burst.write_id
    assert merged.representation == "merged"
    assert merged.layout_id
    assert merged.inputs["scene"] == burst.write_id
    assert merged.inputs["layout"] == merged.layout_id
    slc, unit = merged.read(MERGED_SCENE_UNIT_TAG)
    assert unit.tag == MERGED_SCENE_UNIT_TAG
    np.testing.assert_array_equal(slc, np.full((2, 3), 1.0 + 2.0j, dtype=np.complex64))
    mask_path = merged.unit_map()[MERGED_SCENE_UNIT_TAG].valid_mask_path
    assert mask_path is not None
    np.testing.assert_array_equal(
        read_zarr_array(mask_path), np.ones((2, 3), dtype=bool)
    )
    assert layout.source_ids == ("IW1_b0", "IW1_b1")
    assert "merged" not in stack._scene_stores()
    assert stack._scene_stores()["20160113"].root == burst_root


def test_form_interferograms_rejects_overlap_without_committed_layout(
    tmp_path: Path,
) -> None:
    """Merged formation fails closed when overlapping units have no layout product."""
    dates = ("20160101", "20160113")
    safe_paths = []
    for date_id in dates:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date_id}T000000_{date_id}T000001.SAFE"
        path.mkdir()
        safe_paths.append(path)
    stack = LocalStack.from_paths(safe_paths, work_dir=tmp_path / "out")
    stack.prepare_data()
    payload = np.ones((2, 3), dtype=np.complex64)
    for date_id in dates:
        write_test_scene(
            stack.config.work_dir,
            date_id=date_id,
            reference_id=dates[0],
            payload=payload,
            reference_payload=payload,
            tag=("IW1_b0", "IW1_b1"),
        )
    with pytest.raises(InvalidProcessingStateError, match="committed physical merge_layout"):
        stack.form_interferograms(multilook=(1, 1), phase_filter=None)
