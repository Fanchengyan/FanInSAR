"""Contract tests for the minimal remote boundary."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from faninsar import remote
from faninsar.query import Points, Polygons

if TYPE_CHECKING:
    from pathlib import Path


def _register(name: str = "contract") -> str:
    """Register a deterministic fixture catalog for one test."""
    payload = b"fixture bytes"
    remote._register_fixture(
        [
            {
                "id": "item-1",
                "collection": "dem",
                "geometry": Polygon(((-1, -1), (1, -1), (1, 1), (-1, 1))),
                "properties": {"token": "secret", "safe": "yes"},
                "assets": {
                    "data": {
                        "href": "https://fixture.invalid/assets/data",
                        "data": payload,
                        "size": len(payload),
                        "version": "immutable-1",
                        "checksum": f"sha256:{hashlib.sha256(payload).hexdigest()}",
                    }
                },
            }
        ],
        name=name,
    )
    return name


def test_points_are_exact_and_metadata_is_safe() -> None:
    """Points do not become a bounding box and secrets are removed."""
    name = _register("points-contract")
    items = remote.search(Points([(0, 0), (10, 10)], crs=4326), catalog=name)

    assert len(items) == 1
    assert items[0].matched_point_indices == (0,)
    assert "token" not in items[0].raw_metadata
    with pytest.raises(TypeError):
        items[0].assets["data"].properties["x"] = "y"  # type: ignore[index]


def test_polygons_subtract_undesired_region() -> None:
    """A candidate wholly inside an undesired region is excluded."""
    name = _register("polygon-contract")
    gdf = Polygons(
        gpd.GeoDataFrame(
            geometry=[
                Polygon(((-2, -2), (2, -2), (2, 2), (-2, 2))),
                Polygon(((-1.1, -1.1), (1.1, -1.1), (1.1, 1.1), (-1.1, 1.1))),
            ],
            crs=4326,
        ),
        types=["desired", "undesired"],
        crs=4326,
    )
    assert remote.search(gdf, catalog=name) == []


def test_download_is_atomic_and_reuses_verified_destination(tmp_path: Path) -> None:
    """A qualified manifest permits reuse while a different asset conflicts."""
    name = _register("download-contract")
    asset = remote.search(Points([(0, 0)], crs=4326), catalog=name)[0].assets["data"]
    destination = tmp_path / "asset.bin"

    assert remote.download(asset, destination) == destination
    assert remote.download(asset, destination) == destination
    destination.write_bytes(b"changed")
    with pytest.raises(remote.RemoteIntegrityError, match="destination_conflict"):
        remote.download(asset, destination)
