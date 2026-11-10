"""Optional official absolute burst ID persistence (PROPOSAL-0084)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.io.storage.scene_store import CoregisteredSceneStore, write_scene
from faninsar.processing.errors import InvalidProcessingStateError

if TYPE_CHECKING:
    from pathlib import Path


def _unit(
    tag: str,
    *,
    burst_id: str | None = None,
    absolute_burst_id: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "tag": tag,
        "payload": np.ones((4, 4), np.complex64),
        "row_origin": 0,
        "col_origin": 0,
    }
    if burst_id is not None:
        payload["burst_id"] = burst_id
    if absolute_burst_id is not None:
        payload["absolute_burst_id"] = absolute_burst_id
    return payload


def test_write_scene_persists_optional_absolute_burst_id_and_reopens(
    tmp_path: Path,
) -> None:
    """Official IDs survive write_scene and CoregisteredSceneStore.open."""
    store = write_scene(
        tmp_path / "scene",
        date_id="20240101",
        reference_id="20240101",
        domain="radar",
        units=[
            _unit("f0_IW1_b0", burst_id="IW1-111", absolute_burst_id=111),
            _unit("f0_IW1_b1", burst_id="IW1-222", absolute_burst_id=222),
        ],
        grid_shape=(4, 8),
    )
    assert [unit.absolute_burst_id for unit in store.units] == [111, 222]
    assert [unit.burst_id for unit in store.units] == ["IW1-111", "IW1-222"]
    reopened = CoregisteredSceneStore.open(tmp_path / "scene")
    assert [unit.absolute_burst_id for unit in reopened.units] == [111, 222]
    assert [unit.burst_id for unit in reopened.units] == ["IW1-111", "IW1-222"]


def test_absent_absolute_burst_id_stays_none_on_reopen(tmp_path: Path) -> None:
    """Legacy products without the optional field reopen with None."""
    store = write_scene(
        tmp_path / "scene",
        date_id="20240101",
        reference_id="20240101",
        domain="radar",
        units=[_unit("f0_IW1_b0", burst_id="IW1-anx1")],
        grid_shape=(4, 4),
    )
    assert store.units[0].absolute_burst_id is None
    reopened = CoregisteredSceneStore.open(tmp_path / "scene")
    assert reopened.units[0].absolute_burst_id is None
    assert reopened.units[0].burst_id == "IW1-anx1"


def test_internal_burst_id_is_not_treated_as_official_id(tmp_path: Path) -> None:
    """faninsar:burst_id is not an official absolute burst ID."""
    from tests.network.test_scene_collection import _stack_with_scenes

    stack = _stack_with_scenes(tmp_path)
    from faninsar.io.storage.scene_store import write_scene as write
    from tests.support.scene_stores import coreg_dir

    write(
        coreg_dir(stack.config.work_dir, "20240101"),
        date_id="20240101",
        reference_id="20240101",
        domain="radar",
        units=[
            {
                "tag": "f0_IW1_b0",
                "payload": np.ones((8, 8), np.complex64),
                "row_origin": 0,
                "col_origin": 0,
                "burst_id": "IW1-anx123",
            },
            {
                "tag": "f0_IW2_b0",
                "payload": np.ones((8, 8), np.complex64),
                "row_origin": 0,
                "col_origin": 8,
                "burst_id": "IW2-anx456",
                "absolute_burst_id": 456,
            },
        ],
        grid_shape=(8, 16),
        overwrite=True,
    )
    handles = {
        item.burst: item for item in stack.scenes.bursts if item.date == "20240101"
    }
    assert handles["f0_IW1_b0"].absolute_burst_id is None
    assert handles["f0_IW2_b0"].absolute_burst_id == 456


def _persist_parsed_official_id(tmp_path: Path, official: int | None) -> int | None:
    """Write a one-unit scene from a parsed official ID and reopen it."""
    store = write_scene(
        tmp_path / "scene",
        date_id="20161207",
        reference_id="20161207",
        domain="radar",
        units=[
            _unit(
                "f0_IW1_b0",
                burst_id="IW1-internal",
                absolute_burst_id=official,
            )
        ],
        grid_shape=(4, 4),
    )
    reopened = CoregisteredSceneStore.open(tmp_path / "scene")
    assert store.units[0].burst_id == "IW1-internal"
    assert reopened.units[0].burst_id == "IW1-internal"
    return reopened.units[0].absolute_burst_id


def test_asf_absolute_id_survives_materialize_parse_scene_reopen(
    tmp_path: Path,
) -> None:
    """ASF catalog IDs survive path-SAFE write, parse, scene assets, and reopen."""
    from datetime import UTC, datetime

    from faninsar.missions.s1.acquisition import materialize_asf_burst_safes
    from faninsar.missions.s1.safe import open_safe_product
    from faninsar.missions.s1.scene_bursts import official_absolute_burst_id
    from tests.missions.s1.test_track_materialize import (
        _FakeCatalogItem,
        _write_burst_zip,
    )

    items = [
        _FakeCatalogItem(
            "S1A_BURST_1",
            absolute_burst_id=3001,
            sensing=datetime(2016, 12, 7, 11, 18, 53, tzinfo=UTC),
        )
    ]

    def fake_download(asset, destination):  # noqa: ARG001
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_burst_zip(destination, 0, 53)
        return destination

    paths = materialize_asf_burst_safes(
        items,
        work_dir=tmp_path / "asf",
        download=fake_download,
    )
    product = open_safe_product(paths[0])
    burst = product.swaths[0].bursts[0]
    official = official_absolute_burst_id(burst)
    assert official == 3001
    assert _persist_parsed_official_id(tmp_path / "asf-scene", official) == 3001


def test_cdse_catalog_id_survives_materialize_parse_scene_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CDSE search IDs stamp the materialized annotation and reach scene assets."""
    from datetime import UTC, datetime

    from faninsar.data.query import BoundingBox
    from faninsar.missions.s1 import acquisition as acq
    from faninsar.missions.s1.acquisition import materialize_cdse_burst_safes
    from faninsar.missions.s1.safe import open_safe_product
    from faninsar.missions.s1.scene_bursts import official_absolute_burst_id
    from tests.missions.s1.test_track_materialize import (
        _burst_raster,
        _parent_xml,
    )

    xml_text = _parent_xml()
    monkeypatch.setattr(
        acq, "_fetch_annotation_xml", lambda item, fetch_context=None: xml_text
    )

    def fake_bodies(
        url,  # noqa: ARG001
        *,
        file_size,  # noqa: ARG001
        line0,
        line1,
        col0,  # noqa: ARG001
        col1,  # noqa: ARG001
        samples,  # noqa: ARG001
        fetch_context=None,  # noqa: ARG001
        max_span_bytes=None,  # noqa: ARG001
    ):
        data = _burst_raster(2)
        pairs = np.stack(
            [data.real.astype("<i2"), data.imag.astype("<i2")], axis=-1
        )
        return {f"row-{line0}-{line1}": pairs.tobytes()}

    monkeypatch.setattr(acq, "_fetch_measurement_bodies", fake_bodies)
    roi = BoundingBox(99.9, 41.9, 100.1, 42.1, crs="EPSG:4326")
    items = [
        {
            "product_id": "S1A_IW_SLC__1SDV_TEST",
            "annotation_url": "cdse://annotation.xml",
            "measurement_url": "cdse://measurement.tiff",
            "measurement_size": 0,
            "annotation_size": 0,
            "polarization": "VV",
            "swath": "IW1",
            "mission_id": "A",
            "relative_orbit": 88,
            "orbit_direction": "ASCENDING",
        }
    ]
    paths = materialize_cdse_burst_safes(
        roi,
        items=items,
        work_dir=tmp_path / "cdse",
        absolute_burst_ids=[4242],
        catalog_sensing_times=[
            datetime(2016, 12, 7, 11, 18, 56, 543845, tzinfo=UTC)
        ],
    )
    product = open_safe_product(paths[0])
    burst = product.swaths[0].bursts[0]
    official = official_absolute_burst_id(burst)
    assert official == 4242
    assert _persist_parsed_official_id(tmp_path / "cdse-scene", official) == 4242
    written = next((paths[0] / "annotation").glob("*.xml")).read_text(encoding="utf-8")
    assert "4242" in written
    assert "absoluteBurstID" in written


def test_conflicting_qualified_official_ids_fail_closed() -> None:
    """Disagreement between two qualified official IDs is rejected."""
    from faninsar.missions.s1.scene_bursts import qualified_absolute_burst_id

    with pytest.raises(InvalidProcessingStateError, match="conflicting"):
        qualified_absolute_burst_id(111, 222, tag="f0_IW1_b0")
    assert qualified_absolute_burst_id(111, 111, tag="f0_IW1_b0") == 111
    assert qualified_absolute_burst_id(111, None, tag="f0_IW1_b0") == 111
    assert qualified_absolute_burst_id(None, 222, tag="f0_IW1_b0") is None
