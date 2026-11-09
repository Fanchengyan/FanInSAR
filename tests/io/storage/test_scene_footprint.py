"""Reference-scene burst footprint GeoJSON persistence."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from faninsar.io.storage.scene_store import (
    CoregisteredSceneStore,
    SceneWriter,
    write_scene,
)
from faninsar.missions.s1.coregister import _stage_reference_footprint

if TYPE_CHECKING:
    from pathlib import Path

_RING = (
    (101.0, 38.0),
    (101.5, 38.0),
    (102.0, 38.0),
    (101.8, 37.5),
    (101.3, 37.5),
    (100.8, 37.5),
)


def test_reference_scene_footprint_geojson_roundtrip(tmp_path: Path) -> None:
    """A GeoJSON asset on the Reference scene reopens as a GeoDataFrame."""
    gdf = gpd.GeoDataFrame(
        [
            {
                "tag": "f0_IW1_b0",
                "burst_id": "IW1-111",
                "absolute_burst_id": 111,
                "geometry": Polygon(_RING),
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    with SceneWriter(
        tmp_path / "scene",
        date_id="20240101",
        reference_id="20240101",
        domain="radar",
        grid_shape=(4, 4),
    ) as writer:
        writer.add_unit(
            tag="f0_IW1_b0",
            payload=np.ones((4, 4), np.complex64),
            row_origin=0,
            col_origin=0,
            burst_id="IW1-111",
            absolute_burst_id=111,
        )
        writer.add_footprint(gdf)
        store = writer.commit()

    assert (store.root / "footprint.geojson").is_file()
    loaded = store.footprint
    assert {"tag", "burst_id", "absolute_burst_id", "geometry"} <= set(loaded.columns)
    assert len(loaded) == 1
    row = loaded.iloc[0]
    assert row["tag"] == "f0_IW1_b0"
    assert row["burst_id"] == "IW1-111"
    assert int(row["absolute_burst_id"]) == 111
    assert loaded.crs is not None
    assert loaded.crs.to_epsg() == 4326
    assert row["geometry"].bounds == (100.8, 37.5, 102.0, 38.0)

    reopened = CoregisteredSceneStore.open(tmp_path / "scene")
    again = reopened.footprint
    assert len(again) == 1
    assert again.iloc[0]["tag"] == "f0_IW1_b0"


def test_scene_without_footprint_geojson_returns_empty(tmp_path: Path) -> None:
    """A scene without the GeoJSON asset yields an empty GeoDataFrame."""
    store = write_scene(
        tmp_path / "scene",
        date_id="20240101",
        reference_id="20240101",
        domain="radar",
        units=[
            {
                "tag": "f0_IW1_b0",
                "payload": np.ones((4, 4), np.complex64),
                "row_origin": 0,
                "col_origin": 0,
            }
        ],
        grid_shape=(4, 4),
    )
    gdf = store.footprint
    assert len(gdf) == 0
    assert "geometry" in gdf.columns
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 4326


def _writer(root: Path) -> SceneWriter:
    return SceneWriter(
        root,
        date_id="20240101",
        reference_id="20240101",
        domain="radar",
        grid_shape=(4, 4),
    )


def _unit(*, footprint: object, absolute_burst_id: int | None = 111) -> SimpleNamespace:
    burst = SimpleNamespace(
        footprint=footprint,
        absolute_burst_id=absolute_burst_id,
        azimuth_anx_time_s=0.0,
    )
    return SimpleNamespace(
        tag="f0_IW1_b0",
        swath="IW1",
        scene=SimpleNamespace(burst=burst),
    )


def test_stage_reference_footprint_writes_geojson(tmp_path: Path) -> None:
    """Reference publication copies XML burst rings into footprint.geojson."""
    with _writer(tmp_path / "scene") as writer:
        writer.add_unit(
            tag="f0_IW1_b0",
            payload=np.ones((4, 4), np.complex64),
            row_origin=0,
            col_origin=0,
            burst_id="IW1-111",
        )
        _stage_reference_footprint(writer, {"f0_IW1_b0": _unit(footprint=_RING)})
        store = writer.commit()

    gdf = store.footprint
    assert len(gdf) == 1
    row = gdf.iloc[0]
    assert row["tag"] == "f0_IW1_b0"
    assert row["burst_id"] == "IW1-111"
    assert int(row["absolute_burst_id"]) == 111
    assert row["geometry"].bounds == (100.8, 37.5, 102.0, 38.0)


def test_stage_reference_footprint_skips_invalid_rings(tmp_path: Path) -> None:
    """A non-ring burst.footprint does not publish a GeoJSON asset."""
    with _writer(tmp_path / "scene") as writer:
        writer.add_unit(
            tag="f0_IW1_b0",
            payload=np.ones((4, 4), np.complex64),
            row_origin=0,
            col_origin=0,
            burst_id="IW1-111",
        )
        _stage_reference_footprint(writer, {"f0_IW1_b0": _unit(footprint=object())})
        store = writer.commit()

    assert not (store.root / "footprint.geojson").exists()
    assert len(store.footprint) == 0
