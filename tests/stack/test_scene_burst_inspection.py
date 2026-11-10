"""Read-only scene burst inspection (PROPOSAL-0084)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.network.test_scene_collection import _stack_with_scenes

if TYPE_CHECKING:
    from pathlib import Path


def test_scenes_bursts_enumerates_one_observation_per_committed_unit(
    tmp_path: Path,
) -> None:
    """Each committed scene unit is one date-scoped observation handle."""
    stack = _stack_with_scenes(tmp_path)
    bursts = stack.scenes.bursts
    assert len(bursts) == 6
    tags = [(item.date, item.burst) for item in bursts]
    assert tags == [
        ("20240101", "f0_IW1_b0"),
        ("20240101", "f0_IW2_b0"),
        ("20240113", "f0_IW1_b0"),
        ("20240113", "f0_IW2_b0"),
        ("20240125", "f0_IW1_b0"),
        ("20240125", "f0_IW2_b0"),
    ]
    first = bursts[0]
    assert first.date == "20240101"
    assert first.burst == "f0_IW1_b0"


def test_scenes_bursts_discovery_does_not_open_annotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """len/iter/index must not parse SAFE annotation."""
    from faninsar.missions.s1 import annotation

    calls = {"n": 0}
    original = annotation.parse_annotation_xml

    def wrapped(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(annotation, "parse_annotation_xml", wrapped)
    stack = _stack_with_scenes(tmp_path)
    bursts = stack.scenes.bursts
    assert len(bursts) == 6
    list(bursts)
    _ = bursts[0]
    _ = bursts[-1]
    assert calls["n"] == 0


def test_scenes_bursts_respects_parent_date_selection(tmp_path: Path) -> None:
    """Parent date selection narrows the observation collection."""
    stack = _stack_with_scenes(tmp_path)
    selected = stack.scenes.sel(dates=["20240113"]).bursts
    assert len(selected) == 2
    assert {item.date for item in selected} == {"20240113"}
    assert [item.burst for item in selected] == ["f0_IW1_b0", "f0_IW2_b0"]
    assert len(stack.scenes.bursts) == 6


def test_scene_layers_bursts_remains_tuple_of_strings(tmp_path: Path) -> None:
    """SceneLayers.bursts stays the existing local-tag tuple."""
    stack = _stack_with_scenes(tmp_path)
    layers = stack.scenes["20240101"].layers
    assert isinstance(layers.bursts, tuple)
    assert layers.bursts == ("f0_IW1_b0", "f0_IW2_b0")
    selected = layers.sel(bursts=["f0_IW1_b0"])
    assert selected._selection == frozenset({"f0_IW1_b0"})
    assert layers.bursts == ("f0_IW1_b0", "f0_IW2_b0")


def test_handle_exposes_date_local_tag_and_nullable_official_id(
    tmp_path: Path,
) -> None:
    """Missing official IDs stay None; internal burst_id is not substituted."""
    stack = _stack_with_scenes(tmp_path)
    handle = stack.scenes.bursts[0]
    assert handle.date == "20240101"
    assert handle.burst == "f0_IW1_b0"
    assert handle.absolute_burst_id is None
    assert handle.absolute_burst_id != 0


def _enable_s1_burst_hook(stack: object) -> None:
    """LocalStack tests opt into the Sentinel-1 inspection hook."""
    stack._sensor_registry_key = "sentinel1"


def test_footprints_is_epsg4326_gdf_with_required_columns(tmp_path: Path) -> None:
    """Lazy footprints are EPSG:4326 with one row per observation."""
    stack = _stack_with_scenes(tmp_path)
    _enable_s1_burst_hook(stack)
    gdf = stack.scenes.bursts.footprints
    assert list(gdf.columns) == [
        "date",
        "track",
        "burst",
        "absolute_burst_id",
        "geometry",
    ]
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 4326
    assert len(gdf) == 6
    assert str(gdf["absolute_burst_id"].dtype) == "Int64"
    assert gdf["absolute_burst_id"].isna().all()
    assert gdf["burst"].tolist() == [
        "f0_IW1_b0",
        "f0_IW2_b0",
        "f0_IW1_b0",
        "f0_IW2_b0",
        "f0_IW1_b0",
        "f0_IW2_b0",
    ]


def test_empty_bursts_footprints_keeps_schema(tmp_path: Path) -> None:
    """An empty observation collection still exposes the GDF schema."""
    from faninsar import Pairs
    from tests.support.local_stack import LocalStack

    path = tmp_path / "S1A_IW_SLC__1SDV_20240101T000000.SAFE"
    path.mkdir()
    stack = LocalStack.from_paths(
        [path],
        work_dir=tmp_path / "out",
        pairs=Pairs.from_names(["20240101_20240113"]),
        multilook=(1, 1),
    )
    _enable_s1_burst_hook(stack)
    gdf = stack.scenes.bursts.footprints
    assert len(gdf) == 0
    assert list(gdf.columns) == [
        "date",
        "track",
        "burst",
        "absolute_burst_id",
        "geometry",
    ]
    assert gdf.crs.to_epsg() == 4326


def test_footprints_access_is_lazy_relative_to_collection_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collection construction does not resolve footprints."""
    from faninsar.missions.s1 import annotation

    calls = {"n": 0}
    original = annotation.parse_annotation_xml

    def wrapped(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(annotation, "parse_annotation_xml", wrapped)
    stack = _stack_with_scenes(tmp_path)
    _enable_s1_burst_hook(stack)
    bursts = stack.scenes.bursts
    assert calls["n"] == 0
    _ = bursts.footprints
    # Empty fixture SAFEs have no annotation documents to parse.
    assert calls["n"] == 0


def test_caller_gdf_mutation_does_not_affect_next_access(tmp_path: Path) -> None:
    """Returned GeoDataFrames are independent of later footprints access."""
    stack = _stack_with_scenes(tmp_path)
    _enable_s1_burst_hook(stack)
    bursts = stack.scenes.bursts
    first = bursts.footprints
    first.loc[:, "burst"] = "mutated"
    second = bursts.footprints
    assert (second["burst"] != "mutated").all()
    assert (first["burst"] == "mutated").all()


def test_missing_local_source_keeps_null_geometry_row(tmp_path: Path) -> None:
    """Missing local annotation yields a null-geometry row, not a download."""
    stack = _stack_with_scenes(tmp_path)
    _enable_s1_burst_hook(stack)
    gdf = stack.scenes.bursts.footprints
    assert len(gdf) == 6
    assert gdf.geometry.isna().all()


def test_generic_reader_does_not_import_missions_s1() -> None:
    """Network readers stay mission-neutral; S1 is reached through the Sensor hook."""
    import inspect

    from faninsar.network.readers import scene_bursts, stack

    assert "faninsar.missions.s1" not in inspect.getsource(scene_bursts)
    assert "faninsar.missions.s1" not in inspect.getsource(stack)


def test_unsupported_mission_burst_inspection_is_descriptive() -> None:
    """Missions without the optional hook fail closed with a burst-inspection error."""
    import faninsar.missions.nisar.adapter  # register nisar
    from faninsar.network.readers.scene_bursts import SceneBurstCollection

    stack = type("NisarStack", (), {"_sensor_registry_key": "nisar"})()
    collection = SceneBurstCollection(stack, ())
    with pytest.raises(NotImplementedError, match="burst inspection"):
        _ = collection.footprints


def test_existing_local_sel_bursts_and_swaths_unchanged(tmp_path: Path) -> None:
    """SceneLayers.sel(bursts=/swaths=) still selects local unit tags."""
    stack = _stack_with_scenes(tmp_path)
    layers = stack.scenes["20240101"].layers
    selected = layers.sel(bursts=["f0_IW1_b0"])
    assert selected._selection == frozenset({"f0_IW1_b0"})
    by_swath = layers.sel(swaths=["IW1"])
    assert by_swath._selection == frozenset({"f0_IW1_b0"})
    assert layers.bursts == ("f0_IW1_b0", "f0_IW2_b0")


def test_existing_scenes_sel_burst_integer_still_filters_dates(tmp_path: Path) -> None:
    """scenes.sel(burst=int) remains an absolute-ID date filter, not a local tag."""
    import inspect

    from faninsar.network.readers.stack import StackSceneCollection

    stack = _stack_with_scenes(tmp_path)
    assert "burst" in inspect.signature(StackSceneCollection.sel).parameters
    with pytest.raises((ValueError, KeyError, NotImplementedError)):
        stack.scenes.sel(burst=12345)
    assert len(stack.scenes.bursts) == 6
