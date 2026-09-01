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


def test_unqualified_download_refreshes_existing_destination(tmp_path: Path) -> None:
    """An asset without a version or checksum is fetched again on each call."""
    name = "download-unqualified"
    remote._register_fixture(
        [
            {
                "id": "item-1",
                "geometry": Polygon(((-1, -1), (1, -1), (1, 1), (-1, 1))),
                "assets": {
                    "data": {
                        "href": "https://fixture.invalid/assets/data",
                        "data": b"fresh bytes",
                    }
                },
            }
        ],
        name=name,
    )
    asset = remote.search(Points([(0, 0)], crs=4326), catalog=name)[0].assets["data"]
    destination = tmp_path / "asset.bin"

    destination.write_bytes(b"stale bytes")
    assert remote.download(asset, destination).read_bytes() == b"fresh bytes"
    destination.write_bytes(b"changed bytes")
    assert remote.download(asset, destination).read_bytes() == b"fresh bytes"


def test_invalid_spatial_and_datetime_inputs_are_query_errors() -> None:
    """Malformed CRS and datetime values use the public query error type."""
    name = _register("invalid-inputs")
    points = Points([(0, 0)], crs=4326)
    points._crs = "not-a-crs"  # type: ignore[assignment]
    with pytest.raises(remote.RemoteQueryError) as error:
        remote.search(points, catalog=name)
    assert error.value.reason == "invalid_crs"
    with pytest.raises(remote.RemoteQueryError) as error:
        remote.search(
            Points([(0, 0)], crs=4326),
            catalog=name,
            datetime_range=("not-a-date", "2024-01-01T00:00:00+00:00"),  # type: ignore[arg-type]
        )
    assert error.value.reason == "invalid_datetime"
