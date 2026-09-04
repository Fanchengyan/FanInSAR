"""Fixture-first CMR JSON and UMM discovery tests."""

from __future__ import annotations

import urllib.parse
from datetime import UTC, datetime
from typing import Any

import pytest

from faninsar import remote
from faninsar.remote.cmr import CMRCollectionAdapter, _compact_size_bytes


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


def test_asf_cmr_profile_registers_provider_auth_redirects() -> None:
    """Direct CMR discovery and ASF delivery share the scoped auth policy."""
    adapter = CMRCollectionAdapter(
        provider="ASF",
        collection="sentinel-1",
        collection_concept_id="C1214470488-ASF",
        data_origins=("https://datapool.asf.alaska.edu",),
        profiles=("anonymous", "earthdata-asf"),
    )

    assert "https://sentinel1.asf.alaska.edu" in adapter.redirect_origins
    assert "https://urs.earthdata.nasa.gov" in adapter.redirect_origins
    assert "https://cumulus.asf.alaska.edu" in adapter.redirect_origins


def test_asf_cmr_auth_redirects_use_scoped_exact_paths() -> None:
    """Default ASF registration admits only its token, OAuth, and root routes."""
    adapter = CMRCollectionAdapter(
        provider="ASF",
        collection="sentinel-1",
        data_origins=("https://datapool.asf.alaska.edu",),
        profiles=("anonymous", "earthdata-asf"),
    )
    for url in (
        "https://urs.earthdata.nasa.gov/api/users/find_or_create_token",
        "https://urs.earthdata.nasa.gov/oauth/authorize?response_type=code",
        "https://cumulus.asf.alaska.edu/login",
        "https://cumulus.asf.alaska.edu/",
    ):
        assert remote._validate_url(url, adapter, redirect=True) == url
    for url in (
        "https://urs.earthdata.nasa.gov/api/users/other",
        "https://urs.earthdata.nasa.gov/api/account",
        "https://urs.earthdata.nasa.gov/",
        "https://cumulus.asf.alaska.edu/admin",
        "https://cumulus.asf.alaska.edu/login/other",
    ):
        with pytest.raises(remote.RemoteAccessError, match="unregistered_endpoint"):
            remote._validate_url(url, adapter, redirect=True)


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
            "ArchiveAndDistributionInformation": [
                {"Name": "Not provided", "Size": 1.5, "SizeUnit": "MB"}
            ],
            "RelatedUrls": [{"Type": "GET DATA", "URL": "https://cmr.invalid/data/g1"}],
        },
    }
    record = adapter._normalize(entry)
    assert record["geometry"]["coordinates"][0][-1] == [0.0, 0.0]
    assert record["acquisition"]["start"] == datetime(2024, 1, 1, tzinfo=UTC)
    assert record["acquisition"]["end"] == datetime(2024, 1, 2, tzinfo=UTC)
    assert "size" not in record["assets"]["data"]
    # Official CMR responses wrap UMM and ``meta`` as sibling fields.
    enveloped = adapter._normalize(
        {"meta": {"collection-concept-id": "C123"}, "umm": entry}
    )
    assert enveloped["collection"] == "provider-short-name"


def test_umm_size_uses_only_exact_size_in_bytes() -> None:
    """Human-readable archive size metadata is not an integrity claim."""
    adapter = CMRCollectionAdapter(
        provider="FIXTURE",
        collection="C123",
        endpoint="https://cmr.invalid/search/granules.json",
    )
    base = {
        "GranuleUR": "G1",
        "CollectionReference": {"ShortName": "C123"},
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
            "ArchiveAndDistributionInformation": [{"Size": 1.5, "SizeUnit": "MB"}],
            "RelatedUrls": [{"Type": "GET DATA", "URL": "https://cmr.invalid/data/g1"}],
        },
    }
    without_exact = adapter._normalize(base)
    assert "size" not in without_exact["assets"]["data"]
    base["DataGranule"]["SizeInBytes"] = 123
    with_exact = adapter._normalize(base)
    assert with_exact["assets"]["data"]["size"] == 123


def test_provider_candidate_filter_selects_valid_nisar_tiff() -> None:
    """A registered NISAR filter can skip a VRT and choose the native COG."""
    adapter = CMRCollectionAdapter(
        provider="ASF",
        collection="nisar-glo30",
        collection_concept_id="C3803703055-ASF",
        endpoint="https://cmr.invalid/search/granules.json",
        data_origins=("https://nisar.asf.earthdatacloud.nasa.gov",),
        candidate_filter=lambda candidate, href: (
            href.lower().endswith(".tif")
            and "epsg4326" in href.lower()
            and "vrt" not in str(candidate.get("Name", "")).lower()
        ),
    )
    entry = {
        "GranuleUR": "G3964549387-ASF",
        "CollectionReference": {"ShortName": "nisar-glo30"},
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
                {
                    "Type": "GET DATA",
                    "Name": "DEM_VRT",
                    "URL": "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/DEM/v1.2/EPSG4326/S90/S90_W180/DEM_VRT.tif",
                },
                {
                    "Type": "GET DATA",
                    "Name": "DEM_S90_00_W180_00_C01.tif",
                    "URL": "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/DEM/v1.2/EPSG4326/S90/S90_W180/DEM_S90_00_W180_00_C01.tif",
                },
            ]
        },
    }
    record = adapter._normalize(entry)
    assert record["assets"]["data"]["href"].endswith("DEM_S90_00_W180_00_C01.tif")


def test_provider_candidate_filter_skips_mismatch_and_continues_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filtered representation mismatches do not abort later CMR pages."""
    adapter = CMRCollectionAdapter(
        provider="ASF",
        collection="nisar-glo30",
        endpoint="https://cmr.invalid/search/granules.json",
        page_size=1,
        data_origins=("https://nisar.asf.earthdatacloud.nasa.gov",),
        candidate_filter=lambda _candidate, href: href.lower().endswith(".tif"),
    )
    base = {
        "collection_concept_id": "C3803703055-ASF",
        "polygons": ["0 0 0 1 1 1 0 0"],
    }
    pages = [
        (
            {
                **base,
                "id": "VRT",
                "links": [
                    {
                        "rel": "data#",
                        "href": "https://nisar.asf.earthdatacloud.nasa.gov/adjacent.vrt",
                    }
                ],
            },
            {"cmr-search-after": "next"},
        ),
        (
            {
                **base,
                "id": "TIFF",
                "links": [
                    {
                        "rel": "data#",
                        "href": "https://nisar.asf.earthdatacloud.nasa.gov/native.tif",
                    }
                ],
            },
            {},
        ),
    ]

    def page(
        _self: CMRCollectionAdapter,
        _url: str,
        _headers: dict[str, str],
        _ledger: Any,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        entry, headers = pages.pop(0)
        return {"feed": {"entry": [entry]}}, headers

    monkeypatch.setattr(CMRCollectionAdapter, "_request_page", page)
    records = list(
        adapter.items(ledger=remote._CallLedger(remote.RemoteResourceBudget()))
    )
    assert [record["id"] for record in records] == ["TIFF"]


def test_cmr_candidate_filter_and_unfiltered_adapters_fail_closed() -> None:
    """A matching unsafe URL, or any unsafe unfiltered URL, is rejected."""
    entry = {
        "id": "G1",
        "collection_concept_id": "C123",
        "polygons": ["0 0 0 1 1 1 0 0"],
        "links": [{"rel": "data#", "href": "https://evil.invalid/native.tif"}],
    }
    filtered = CMRCollectionAdapter(
        provider="FIXTURE",
        collection="C123",
        endpoint="https://cmr.invalid/search/granules.json",
        candidate_filter=lambda _candidate, href: href.endswith(".tif"),
    )
    unfiltered = CMRCollectionAdapter(
        provider="FIXTURE",
        collection="C123",
        endpoint="https://cmr.invalid/search/granules.json",
    )
    for adapter in (filtered, unfiltered):
        with pytest.raises(remote.RemoteAccessError, match="unregistered_endpoint"):
            adapter._normalize(entry)


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


def test_compact_granule_size_preserves_exact_binary_mib_value() -> None:
    """Preserve the exact byte count declared by a real compact CMR item."""
    adapter = CMRCollectionAdapter(
        provider="ASF",
        collection="sentinel-1",
        endpoint="https://cmr.invalid/search/granules.json",
    )
    entry = {
        "id": "G4297731264-ASF",
        "collection_concept_id": "C123",
        "granule_size": "149.3316593170166",
        "polygons": ["0 0 0 1 1 1 0 0"],
        "links": [{"rel": "data#", "href": "https://cmr.invalid/data/g1"}],
    }

    record = adapter._normalize(entry)

    assert record["assets"]["data"]["size"] == 156_585_594
    item = remote._normalize_record(record, "fixture", adapter, "anonymous")
    assert item.assets["data"].size_bytes == 156_585_594


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        "not-a-number",
        float("nan"),
        float("inf"),
        -1,
        "-1",
        "1.0000001",
    ],
)
def test_compact_granule_size_rejects_invalid_declarations(value: Any) -> None:
    """Ignore malformed, unsafe, and fractional-byte compact sizes."""
    assert _compact_size_bytes(value) is None


def test_compact_polygon_accepts_nested_single_coordinate_string() -> None:
    """Decode the nested single-string ring emitted by real compact CMR."""
    adapter = CMRCollectionAdapter(
        provider="FIXTURE",
        collection="C123",
        endpoint="https://cmr.invalid/search/granules.json",
    )
    entry = {
        "id": "G1",
        "collection_concept_id": "C123",
        "polygons": [["34.0 -117.0 34.0 -116.0 35.0 -116.0 34.0 -117.0"]],
        "links": [{"rel": "data#", "href": "https://cmr.invalid/data/g1"}],
    }
    record = adapter._normalize(entry)
    assert record["geometry"] == {
        "type": "Polygon",
        "coordinates": [
            [[-117.0, 34.0], [-116.0, 34.0], [-116.0, 35.0], [-117.0, 34.0]]
        ],
    }


def test_compact_polygon_rejects_ambiguous_nested_ring() -> None:
    """Do not silently combine a complete string with point strings."""
    adapter = CMRCollectionAdapter(
        provider="FIXTURE",
        collection="C123",
        endpoint="https://cmr.invalid/search/granules.json",
    )
    entry = {
        "id": "G1",
        "collection_concept_id": "C123",
        "polygons": [["0 0 0 1 1 1 0 0", "0 0"]],
        "links": [{"rel": "data#", "href": "https://cmr.invalid/data/g1"}],
    }
    with pytest.raises(remote.RemoteAccessError) as error:
        adapter._normalize(entry)
    assert error.value.reason == "missing_footprint"


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
