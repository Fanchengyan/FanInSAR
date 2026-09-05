"""Per-public-call accounting tests for the remote boundary."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from faninsar import remote
from faninsar.data.query import Points

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


class _MeteredAdapter:
    """Fixture adapter that charges both public operations independently."""

    provider = "metered"
    origins = ("https://metered.invalid",)
    path_prefixes = ("/",)
    redirect_origins = ()
    profiles = ("anonymous",)

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.ledgers: list[Any] = []

    def items(self, *, ledger: Any) -> Iterable[Mapping[str, Any]]:
        self.ledgers.append(ledger)
        ledger.request()
        ledger.response_bytes(54)
        return (
            {
                "id": "metered-item",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                },
                "assets": {
                    "data": {
                        "href": "https://metered.invalid/data.bin",
                        "size": len(self.payload),
                        "checksum": (
                            f"sha256:{hashlib.sha256(self.payload).hexdigest()}"
                        ),
                    }
                },
            },
        )

    def fetch(
        self,
        asset: remote.RemoteAsset,
        budget: remote.RemoteResourceBudget,
        *,
        ledger: Any,
    ) -> Iterable[bytes]:
        del asset, budget
        self.ledgers.append(ledger)
        ledger.request()
        ledger.response_bytes(len(self.payload))
        return (self.payload,)


class _QueryAdapter(_MeteredAdapter):
    """Fixture adapter recording the normalized public search contract."""

    def items(
        self,
        *,
        spatial: Any,
        spatial_kind: str,
        point_geometries: tuple[Any, ...],
        datetime_range: tuple[Any, Any] | None,
        collections: tuple[str, ...] | None,
        auth_profile: str,
        limit: int,
        budget: remote.RemoteResourceBudget,
        ledger: Any,
    ) -> Iterable[Mapping[str, Any]]:
        self.query = {
            "spatial": spatial,
            "spatial_kind": spatial_kind,
            "point_geometries": point_geometries,
            "datetime_range": datetime_range,
            "collections": collections,
            "auth_profile": auth_profile,
            "limit": limit,
            "budget": budget,
        }
        return super().items(ledger=ledger)


def test_search_and_download_receive_fresh_ledgers(
    tmp_path: Path,
) -> None:
    """Search and its later download do not share accounting state."""
    payload = b"metered payload"
    adapter = _MeteredAdapter(payload)
    remote._register_adapter("metered-ledger", adapter)

    items = remote.search(
        Points([(0.5, 0.5)], crs=4326),
        catalog="metered-ledger",
    )
    remote.download(items[0].assets["data"], tmp_path / "data.bin")

    assert len(adapter.ledgers) == 2
    assert adapter.ledgers[0] is not adapter.ledgers[1]
    assert adapter.ledgers[0].requests == 1
    assert adapter.ledgers[1].requests == 1
    assert adapter.ledgers[0].response_bytes_total == 54
    assert adapter.ledgers[1].response_bytes_total == len(payload)


def test_persisted_metadata_scrubs_the_complete_azure_sas_query() -> None:
    """Azure SAS fields and unrelated signed-query material never persist."""
    signed = (
        "https://blob.invalid/path/file.tif?sp=r&st=2024-01-01T00%3A00%3A00Z&"
        "se=2024-01-02T00%3A00%3A00Z&sv=2023-11-03&sr=b&spr=https&sip=127.0.0.1&"
        "si=policy&sig=secret&x-request-context=also-signed"
    )
    clean = remote._sanitize(
        {
            "href": signed,
            "sp": "r",
            "sv": "2023-11-03",
            "nested": {"si": "policy", "safe": "value"},
        }
    )

    assert clean == {
        "href": "https://blob.invalid/path/file.tif",
        "nested": {"safe": "value"},
    }


def test_redirect_preserves_signed_query_and_strips_cross_origin_headers() -> None:
    """An approved signed redirect keeps SAS query material request-local."""
    adapter = _MeteredAdapter(b"payload")
    adapter.origins = ("https://source.invalid",)
    adapter.redirect_origins = ("https://cdn.invalid",)
    handler = remote._RedirectHandler(
        adapter,
        remote.RemoteResourceBudget(max_redirects=1),
    )
    request = remote.urllib.request.Request(
        "https://source.invalid/data/file.tif",
        headers={"Authorization": "Bearer secret", "Cookie": "session=secret"},
    )
    location = (
        "https://cdn.invalid/data/file.tif?sv=2023-11-03&sig=secret&ss=b&srt=o&sdd=1"
    )
    redirected = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {"Location": location},
        location,
    )

    assert redirected is not None
    assert redirected.full_url == location
    assert "Authorization" not in redirected.headers
    assert "Cookie" not in redirected.headers


def test_search_propagates_normalized_query_to_capable_adapter() -> None:
    """Adapters receive the same normalized values used by public filtering."""
    adapter = _QueryAdapter(b"payload")
    remote._register_adapter("query-contract", adapter)
    start = "2024-01-01T00:00:00+00:00"
    end = "2024-01-02T00:00:00+00:00"

    remote.search(
        Points([(0.5, 0.5)], crs=4326),
        catalog="query-contract",
        collections=["C123"],
        datetime_range=(start, end),  # type: ignore[arg-type]
        auth_profile="anonymous",
        limit=3,
    )

    assert adapter.query["spatial_kind"] == "points"
    assert adapter.query["spatial"].x == 0.5
    assert adapter.query["spatial"].y == 0.5
    assert adapter.query["collections"] == ("C123",)
    assert adapter.query["datetime_range"][0].tzinfo is not None
    assert adapter.query["auth_profile"] == "anonymous"
    assert adapter.query["limit"] == 3
