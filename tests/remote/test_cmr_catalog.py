"""Fixture-first CMR JSON and UMM discovery tests."""

from __future__ import annotations

import urllib.parse
from datetime import UTC, datetime
from typing import Any

import pytest

from faninsar import remote
from faninsar.remote.cmr import CMRCollectionAdapter


class _ResponseFixture:
    """Small response object consumed by the adapter page seam."""

    def __init__(self, body: dict[str, Any], token: str | None = None) -> None:
        self.body = body
        self.headers = {"cmr-search-after": token} if token else {}


def test_cmr_items_builds_server_side_query_from_normalized_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CMR receives spatial, temporal, collection, and page-limit filters."""
    adapter = CMRCollectionAdapter(
        provider="FIXTURE",
        collection="S1_GRD",
        endpoint="https://cmr.invalid/search/granules.json",
        page_size=100,
    )
    seen_urls: list[str] = []

    def page(
        _self: CMRCollectionAdapter,
        url: str,
        _headers: dict[str, str],
        _ledger: Any,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        seen_urls.append(url)
        return {"feed": {"entry": []}}, {}

    monkeypatch.setattr(CMRCollectionAdapter, "_request_page", page)
    spatial = remote._query_geometry(remote.BoundingBox(-2, 3, 4, 5, crs=4326))[0]
    records = list(
        adapter.items(
            spatial=spatial,
            spatial_kind="bbox",
            datetime_range=(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
            ),
            collections=("S1_GRD",),
            limit=7,
            ledger=remote._CallLedger(remote.RemoteResourceBudget()),
        )
    )

    assert records == []
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(seen_urls[0]).query)
    assert params["bounding_box"] == ["-2,3,4,5"]
    assert params["temporal"] == ["2024-01-01T00:00:00Z,2024-01-02T00:00:00Z"]
    assert params["short_name"] == ["S1_GRD"]
    assert params["page_size"] == ["7"]


def test_cmr_json_pagination_uses_search_after_and_collection_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pages advance using the response token and retain registered identity."""
    adapter = CMRCollectionAdapter(
        provider="FIXTURE",
        collection="C123",
        endpoint="https://cmr.invalid/search/granules.json",
        page_size=1,
    )
    pages = [
        _ResponseFixture(
            {
                "feed": {
                    "entry": [
                        {
                            "id": "G1",
                            "collection_concept_id": "C123",
                            "polygons": [["0 0", "0 1", "1 1", "0 0"]],
                            "time_start": "2024-01-01T00:00:00Z",
                            "links": [
                                {
                                    "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                                    "href": "https://cmr.invalid/data/g1",
                                },
                                {
                                    "rel": "http://esipfed.org/ns/fedsearch/1.1/browse#",
                                    "href": "https://cmr.invalid/browse/g1",
                                },
                                {
                                    "rel": "http://esipfed.org/ns/fedsearch/1.1/service#",
                                    "href": "https://cmr.invalid/service/g1",
                                },
                            ],
                        }
                    ]
                }
            },
            token="token-1",
        ),
        _ResponseFixture(
            {
                "feed": {
                    "entry": [
                        {
                            "id": "G2",
                            "collection_concept_id": "C123",
                            "polygons": [["0 0", "0 1", "1 1", "0 0"]],
                            "time_start": "2024-01-02T00:00:00Z",
                            "links": [
                                {
                                    "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                                    "href": "https://cmr.invalid/data/g2",
                                }
                            ],
                        }
                    ]
                }
            }
        ),
    ]
    seen_headers: list[dict[str, str]] = []

    def page(
        _self: CMRCollectionAdapter,
        _url: str,
        headers: dict[str, str],
        _ledger: Any,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        seen_headers.append(headers)
        fixture = pages.pop(0)
        return fixture.body, fixture.headers

    monkeypatch.setattr(CMRCollectionAdapter, "_request_page", page)
    records = list(
        adapter.items(ledger=remote._CallLedger(remote.RemoteResourceBudget()))
    )

    assert [record["id"] for record in records] == ["G1", "G2"]
    assert all(record["collection"] == "C123" for record in records)
    assert seen_headers[1]["CMR-Search-After"] == "token-1"


def test_umm_requires_explicit_get_data_candidate() -> None:
    """Metadata and browse URLs never become downloadable assets."""
    adapter = CMRCollectionAdapter(
        provider="FIXTURE",
        collection="S1",
        endpoint="https://cmr.invalid/search/granules.json",
    )
    entry = {
        "GranuleUR": "G1",
        "CollectionReference": {"ShortName": "S1"},
        "SpatialExtent": {
            "HorizontalSpatialDomain": {
                "Geometry": {
                    "GPolygon": {
                        "Boundary": {
                            "Points": [
                                {"Longitude": 0, "Latitude": 0},
                                {"Longitude": 1, "Latitude": 0},
                                {"Longitude": 1, "Latitude": 1},
                            ]
                        }
                    }
                }
            }
        },
        "DataGranule": {
            "RelatedUrls": [
                {"Type": "GET RELATED URL", "URL": "https://cmr.invalid/metadata/g1"}
            ]
        },
    }
    with pytest.raises(remote.RemoteAccessError) as error:
        adapter._normalize(entry)
    assert error.value.reason == "missing_data_asset"


def test_official_umm_geometry_and_temporal_variants() -> None:
    """Decode UMM GPolygons, singular TemporalExtent, and concept metadata."""
    adapter = CMRCollectionAdapter(
        provider="ASF",
        collection="provider-short-name",
        collection_concept_id="C123",
        endpoint="https://cmr.invalid/search/granules.umm_json",
        data_origins=("https://cmr.invalid",),
    )
    entry = {
        "GranuleUR": "G1",
        "meta": {"collection-concept-id": "C123", "provider-id": "ASF"},
        "CollectionReference": {"ShortName": "different-provider-alias"},
        "SpatialExtent": {
            "HorizontalSpatialDomain": {
                "Geometry": {
                    "GPolygons": [
                        {
                            "Boundary": {
                                "Points": [
                                    {"Longitude": 0, "Latitude": 0},
                                    {"Longitude": 1, "Latitude": 0},
                                    {"Longitude": 1, "Latitude": 1},
                                ]
                            }
                        }
                    ]
                }
            }
        },
        "TemporalExtent": {
            "RangeDateTime": {
                "BeginningDateTime": "2024-01-01T00:00:00Z",
                "EndingDateTime": "2024-01-02T00:00:00Z",
            }
        },
        "DataGranule": {
            "RelatedUrls": [{"Type": "GET DATA", "URL": "https://cmr.invalid/data/g1"}]
        },
    }
    record = adapter._normalize(entry)
    assert record["geometry"]["coordinates"][0][-1] == [0.0, 0.0]
    assert record["acquisition"]["start"] == datetime(2024, 1, 1, tzinfo=UTC)
    assert record["acquisition"]["end"] == datetime(2024, 1, 2, tzinfo=UTC)


def test_compact_polygon_accepts_one_coordinate_string() -> None:
    """Decode compact CMR's one-string latitude/longitude polygon form."""
    adapter = CMRCollectionAdapter(
        provider="FIXTURE",
        collection="C123",
        endpoint="https://cmr.invalid/search/granules.json",
    )
    entry = {
        "id": "G1",
        "collection_concept_id": "C123",
        "polygons": ["0 0 0 1 1 1 0 0"],
        "links": [{"rel": "data#", "href": "https://cmr.invalid/data/g1"}],
    }
    record = adapter._normalize(entry)
    assert record["geometry"] == {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
    }


def test_auth_profile_resolver_is_operation_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject profile headers per operation without retaining credentials."""
    seen: list[dict[str, str]] = []

    def resolve(profile: str) -> dict[str, str]:
        assert profile == "earthdata-asf"
        return {"Authorization": "Bearer transient"}

    adapter = CMRCollectionAdapter(
        provider="FIXTURE",
        collection="C123",
        endpoint="https://cmr.invalid/search/granules.json",
        profiles=("anonymous", "earthdata-asf", "lpdaac"),
        auth_resolver=resolve,
    )

    def page(
        _self: CMRCollectionAdapter,
        _url: str,
        headers: dict[str, str],
        _ledger: Any,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        seen.append(headers)
        return {"feed": {"entry": []}}, {}

    monkeypatch.setattr(CMRCollectionAdapter, "_request_page", page)
    list(
        adapter.items(
            auth_profile="earthdata-asf",
            ledger=remote._CallLedger(remote.RemoteResourceBudget()),
        )
    )
    assert seen == [{"Authorization": "Bearer transient"}]
    assert not hasattr(adapter, "_auth_headers_value")
