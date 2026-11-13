"""Search filters, exact geometry, and generation-bound pagination (P0089)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from faninsar.io.stac.index import StacCollection, StacIndex
from faninsar.io.stac.record import StacRecord, TemporalCoverage
from faninsar.io.stac.search import (
    SearchRequest,
    StaleSearchTokenError,
    StacSearchError,
    search,
)

_START = datetime(2024, 1, 1, tzinfo=UTC)
_END = datetime(2024, 1, 2, 23, 59, 59, 999999, tzinfo=UTC)


def _record(
    public_id: str,
    *,
    geometry: object | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    start: datetime = _START,
    end: datetime = _END,
    collection_id: str | None = None,
    family: str = "scenes",
) -> StacRecord:
    """Build one synthetic normalized record for search tests."""
    item: dict[str, object] = {
        "stac_version": "1.0.0",
        "type": "Feature",
        "id": public_id,
        "geometry": geometry,
        "properties": {"datetime": start.isoformat()},
        "assets": {},
        "links": [],
    }
    if bbox is not None:
        item["bbox"] = list(bbox)
    return StacRecord(
        public_id=public_id,
        write_id=public_id,
        kind="scene",
        family=family,
        source_item_path=Path("/workspace") / public_id / "item.json",
        source_relative_path=public_id,
        collection_id=collection_id,
        temporal=TemporalCoverage(start=start, end=end, precision="timestamp"),
        bbox=bbox,
        item=item,
        asset_paths={},
    )


def _index(
    records: list[StacRecord],
    collections: tuple[StacCollection, ...] = (),
    *,
    generation: str = "generation-1",
) -> StacIndex:
    """Build one synthetic index ordered by public id."""
    ordered = tuple(sorted(records, key=lambda record: record.public_id))
    return StacIndex(
        generation=generation,
        created_at="2024-01-01T00:00:00.000000Z",
        work_dir=Path("/workspace"),
        records=ordered,
        collections=collections,
        exclusions=(),
    )


def _square() -> dict[str, object]:
    """Return one closed GeoJSON square."""
    return {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]],
    }


def test_ids_collections_and_public_id_order() -> None:
    """Filters match public ids and actual Collection membership only."""
    collection = StacCollection(
        collection_id="scenes-geographic",
        family="scenes",
        item_ids=("item-b",),
        bbox=(0.0, 0.0, 1.0, 1.0),
        interval=(_START, _END),
    )
    geo = _record(
        "item-b",
        geometry=_square(),
        bbox=(0.0, 0.0, 2.0, 2.0),
        collection_id="scenes-geographic",
    )
    radar = _record("item-a")
    index = _index([geo, radar], (collection,))

    assert [item.public_id for item in search(index, SearchRequest()).items] == [
        "item-a",
        "item-b",
    ]
    assert [
        item.public_id
        for item in search(index, SearchRequest(ids=("item-b",))).items
    ] == ["item-b"]
    assert search(index, SearchRequest(ids=("missing",))).items == ()
    assert [
        item.public_id
        for item in search(
            index,
            SearchRequest(collections=("scenes-geographic",)),
        ).items
    ] == ["item-b"]
    with pytest.raises(StacSearchError, match="unknown collections"):
        search(index, SearchRequest(collections=("missing-geographic",)))


def test_limit_pagination_and_number_matched() -> None:
    """A cursor pages the full ordered result set without repeating items."""
    index = _index([_record(f"item-{letter}") for letter in "abcde"])
    first = search(index, SearchRequest(limit=2))
    assert [item.public_id for item in first.items] == ["item-a", "item-b"]
    assert first.number_matched == 5
    assert first.next_token is not None

    second = search(index, SearchRequest(limit=2, token=first.next_token))
    assert [item.public_id for item in second.items] == ["item-c", "item-d"]
    assert second.next_token is not None

    third = search(index, SearchRequest(limit=2, token=second.next_token))
    assert [item.public_id for item in third.items] == ["item-e"]
    assert third.next_token is None


def test_limit_validation_and_clamping() -> None:
    """Non-integers and zero fail; large limits clamp to the advertised maximum."""
    index = _index([_record(f"item-{number:02d}") for number in range(11)])
    with pytest.raises(StacSearchError, match="at least 1"):
        search(index, SearchRequest(limit=0))
    with pytest.raises(StacSearchError, match="integer"):
        search(index, SearchRequest(limit=True))
    with pytest.raises(StacSearchError, match="integer"):
        search(index, SearchRequest(limit=2.5))  # type: ignore[arg-type]
    page = search(index, SearchRequest(limit=10))
    assert len(page.items) == 10
    assert page.next_token is not None
    assert page.number_matched == 11
    assert len(search(index, SearchRequest(limit=5000)).items) == 11


def test_datetime_instant_interval_and_open_bounds() -> None:
    """RFC 3339 instants and intervals intersect Item temporal coverage."""
    index = _index([_record("item-a")])
    matching = (
        "2024-01-01T12:00:00Z",
        "2024-01-01T13:00:00+01:00",
        "2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
        "2024-01-02T00:00:00Z/..",
        "../2024-01-02T00:00:00Z",
        "../..",
    )
    for value in matching:
        assert len(search(index, SearchRequest(datetime=value)).items) == 1, value
    nonmatching = (
        "2023-12-31T23:59:59Z",
        "2024-01-03T00:00:00Z",
        "../2023-12-31T23:59:59Z",
    )
    for value in nonmatching:
        assert search(index, SearchRequest(datetime=value)).items == (), value


@pytest.mark.parametrize(
    "value",
    [
        "2024-01-01",
        "2024-01-01T00:00:00",
        "2024-01-01T00:00:00Z/2024-01-02T00:00:00Z/2024-01-03T00:00:00Z",
        "2024-01-03T00:00:00Z/2024-01-02T00:00:00Z",
        "",
    ],
)
def test_invalid_datetime_is_rejected(value: str) -> None:
    """Date-only, offset-free, malformed, and inverted intervals fail."""
    index = _index([_record("item-a")])
    with pytest.raises(StacSearchError):
        search(index, SearchRequest(datetime=value))


def test_bbox_four_and_six_dimensions_and_antimeridian() -> None:
    """bbox ignores elevation and splits a westward box at the antimeridian."""
    index = _index(
        [
            _record("east", geometry=_square(), bbox=(0.0, 0.0, 2.0, 2.0)),
            _record(
                "dateline-east",
                geometry={
                    "type": "Polygon",
                    "coordinates": [
                        [[178.0, -1.0], [180.0, -1.0], [180.0, 1.0], [178.0, 1.0], [178.0, -1.0]]
                    ],
                },
                bbox=(178.0, -1.0, 180.0, 1.0),
            ),
            _record(
                "dateline-west",
                geometry={
                    "type": "Polygon",
                    "coordinates": [
                        [[-180.0, -1.0], [-178.0, -1.0], [-178.0, 1.0], [-180.0, 1.0], [-180.0, -1.0]]
                    ],
                },
                bbox=(-180.0, -1.0, -178.0, 1.0),
            ),
        ]
    )
    assert [
        item.public_id for item in search(index, SearchRequest(bbox=(0, 0, 3, 3))).items
    ] == ["east"]
    assert [
        item.public_id
        for item in search(index, SearchRequest(bbox=(0, 0, -10, 3, 3, 10))).items
    ] == ["east"]
    assert [
        item.public_id
        for item in search(index, SearchRequest(bbox=(170, -10, -170, 10))).items
    ] == ["dateline-east", "dateline-west"]


def test_degenerate_bbox_matches_boundary_and_invalid_bbox_fails() -> None:
    """Degenerate in-range boxes are accepted; domain and ordering errors fail."""
    index = _index([_record("item-a", geometry=_square(), bbox=(0, 0, 2, 2))])
    assert len(search(index, SearchRequest(bbox=(0, 0, 0, 1))).items) == 1
    for value in (
        (0, 2, 1, 1),
        (-181, 0, 1, 1),
        (0, -91, 1, 1),
        (0, 0, float("nan"), 1),
        (0, 0, 1),
    ):
        with pytest.raises(StacSearchError):
            search(index, SearchRequest(bbox=value))


def test_bbox_and_intersects_are_mutually_exclusive() -> None:
    """Supplying both spatial filters returns a request error."""
    index = _index([_record("item-a")])
    with pytest.raises(StacSearchError, match="mutually exclusive"):
        search(
            index,
            SearchRequest(bbox=(0, 0, 1, 1), intersects=_square()),
        )


@pytest.mark.parametrize(
    "geometry",
    [
        {"type": "Point", "coordinates": [1.0, 1.0]},
        {"type": "MultiPoint", "coordinates": [[1.0, 1.0], [9.0, 9.0]]},
        {
            "type": "LineString",
            "coordinates": [[-1.0, 1.0], [3.0, 1.0]],
        },
        {
            "type": "MultiLineString",
            "coordinates": [
                [[-1.0, 1.0], [3.0, 1.0]],
                [[1.0, -1.0], [1.0, 3.0]],
            ],
        },
        _square(),
        {
            "type": "MultiPolygon",
            "coordinates": [
                [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]
            ],
        },
        {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Point", "coordinates": [1.0, 1.0]},
                {
                    "type": "LineString",
                    "coordinates": [[-1.0, 1.0], [3.0, 1.0]],
                },
            ],
        },
    ],
)
def test_intersects_supports_every_geojson_geometry_type(
    geometry: dict[str, object],
) -> None:
    """All seven GeoJSON geometry types match a covering Item polygon."""
    index = _index([_record("item-a", geometry=_square(), bbox=(0, 0, 2, 2))])
    page = search(index, SearchRequest(intersects=geometry))
    assert [item.public_id for item in page.items] == ["item-a"]


def test_intersects_is_exact_geometry_not_bbox() -> None:
    """A query inside the Item bbox but outside its polygon does not match."""
    triangle = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]],
    }
    index = _index([_record("triangle", geometry=triangle, bbox=(0, 0, 1, 1))])
    assert (
        search(
            index,
            SearchRequest(intersects={"type": "Point", "coordinates": [0.9, 0.9]}),
        ).items
        == ()
    )
    assert search(index, SearchRequest(bbox=(0.8, 0.8, 0.9, 0.9))).items == ()
    assert (
        search(
            index,
            SearchRequest(intersects={"type": "Point", "coordinates": [0.1, 0.1]}),
        ).items
        == (index.records[0],)
    )


def test_null_geometry_never_matches_spatial_filters() -> None:
    """A geometry-less Item is discoverable but never spatially matches."""
    index = _index([_record("radar")])
    assert search(index, SearchRequest(bbox=(0, 0, 1, 1))).items == ()
    assert search(index, SearchRequest(intersects=_square())).items == ()
    assert len(search(index, SearchRequest(ids=("radar",))).items) == 1


@pytest.mark.parametrize(
    "geometry",
    [
        "not-an-object",
        {"type": "Feature", "geometry": _square()},
        {"type": "Point"},
        {"type": "Point", "coordinates": []},
        {"type": "LineString", "coordinates": [[0.0, 0.0]]},
        {"type": "Polygon", "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]},
        {
            "type": "Polygon",
            "coordinates": [[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]],
        },
        {"type": "GeometryCollection"},
        {"type": "GeometryCollection", "geometries": [{"type": "Unknown"}]},
        {"type": "Point", "coordinates": ["a", "b"]},
    ],
)
def test_invalid_intersects_geometry_is_rejected(geometry: object) -> None:
    """Malformed, unknown, and invalid GeoJSON geometries fail with 400."""
    index = _index([_record("item-a")])
    with pytest.raises(StacSearchError):
        search(index, SearchRequest(intersects=geometry))  # type: ignore[arg-type]


def test_cursor_is_bound_to_generation_and_query() -> None:
    """A cursor cannot cross generations, filters, or intersects geometry."""
    index = _index([_record("item-a"), _record("item-b")], generation="generation-1")
    first = search(index, SearchRequest(limit=1))
    token = first.next_token
    assert token is not None

    with pytest.raises(StaleSearchTokenError):
        search(index, SearchRequest(limit=1, ids=("item-b",), token=token))
    with pytest.raises(StaleSearchTokenError):
        search(
            index,
            SearchRequest(
                limit=1,
                intersects={"type": "Point", "coordinates": [0.0, 0.0]},
                token=token,
            ),
        )
    changed = _index(
        [_record("item-a"), _record("item-b")], generation="generation-2"
    )
    with pytest.raises(StaleSearchTokenError):
        search(changed, SearchRequest(limit=1, token=token))
    identical = _index(
        [_record("item-a"), _record("item-b")], generation="generation-1"
    )
    assert [
        item.public_id
        for item in search(identical, SearchRequest(limit=1, token=token)).items
    ] == ["item-b"]


@pytest.mark.parametrize("token", ["!!!", "not-base64", "e30", "eyJ2IjoxfQ"])
def test_malformed_or_incomplete_cursor_is_rejected(token: str) -> None:
    """A malformed or structurally incomplete cursor fails as a request error."""
    index = _index([_record("item-a"), _record("item-b")])
    with pytest.raises(StacSearchError):
        search(index, SearchRequest(limit=1, token=token))


def test_index_search_method_uses_shared_engine() -> None:
    """``StacIndex.search`` delegates to the same search implementation."""
    index = _index([_record("item-a")])
    page = index.search(SearchRequest(ids=("item-a",)))
    assert [item.public_id for item in page.items] == ["item-a"]
