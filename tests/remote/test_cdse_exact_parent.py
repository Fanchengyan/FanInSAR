"""CDSE exact-name OData fallback for parent SAFEs absent from STAC.

These are offline provider tests: a scripted opener serves the public
Products/Attributes and metadata Nodes JSON observed for
``S1A_IW_SLC__1SDV_20150428T171437_20150428T171510_005687_0074BD_87A1``.
No token or SAFE payload is fetched; every request stays on a registered
CDSE origin.
"""

from __future__ import annotations

import io
import json
import urllib.request
from datetime import UTC, datetime
from typing import Any

import pytest

from faninsar.remote.errors import RemoteAccessError, RemoteLimitError
from faninsar.remote.protocols import _CallLedger
from faninsar.remote.providers.cdse import (
    CDSE_CATALOGUE_ORIGIN,
    CDSE_DOWNLOAD_ORIGIN,
    lookup_cdse_safe_record,
)
from faninsar.remote.records import RemoteResourceBudget

PARENT = "S1A_IW_SLC__1SDV_20150428T171437_20150428T171510_005687_0074BD_87A1"
SAFE_NAME = f"{PARENT}.SAFE"
PRODUCT_UUID = "ab0e1fd5-6011-574f-ab49-08119ba6c91c"

_FOOTPRINT = {
    "type": "MultiPolygon",
    "coordinates": [
        [
            [
                [7.960505, 45.028915],
                [11.172586, 45.424347],
                [10.734909, 47.377434],
                [7.403312, 46.981205],
                [7.960505, 45.028915],
            ]
        ]
    ],
}


def _product_document(
    *, name: str = SAFE_NAME, product_id: Any = PRODUCT_UUID
) -> dict[str, Any]:
    return {
        "value": [
            {
                "Id": product_id,
                "Name": name,
                "S3Path": f"/eodata/Sentinel-1/SAR/SLC/2015/04/28/{name}",
                "GeoFootprint": _FOOTPRINT,
                "ContentDate": {
                    "Start": "2015-04-28T17:14:37.931000Z",
                    "End": "2015-04-28T17:15:10.518000Z",
                },
                "Attributes": [
                    {"Name": "relativeOrbitNumber", "Value": 15},
                    {"Name": "orbitDirection", "Value": "ASCENDING"},
                    {"Name": "platformSerialIdentifier", "Value": "A"},
                    {"Name": "polarisationChannels", "Value": "VV&VH"},
                    {"Name": "swathIdentifier", "Value": "IW1 IW2 IW3"},
                ],
            }
        ]
    }


def _stem(swath: str, polarization: str, sequence: int) -> str:
    return (
        f"s1a-{swath}-slc-{polarization}-"
        "20150428t171437-20150428t171508-005687-0074bd-"
        f"{sequence:03d}"
    )


def _node_document(role: str, *, untrusted_uri: bool = False) -> dict[str, Any]:
    result = []
    sizes = {"annotation": 100_000, "measurement": 1_000_000}
    for swath_index, swath in enumerate(("iw1", "iw2", "iw3"), start=1):
        for pol_index, polarization in enumerate(("vv", "vh"), start=1):
            sequence = swath_index if polarization == "vv" else swath_index + 3
            stem = _stem(swath, polarization, sequence)
            extension = "xml" if role == "annotation" else "tiff"
            name = f"{stem}.{extension}"
            node: dict[str, Any] = {
                "Id": name,
                "Name": name,
                "ContentLength": sizes[role] + swath_index + pol_index,
                "ChildrenNumber": 0,
            }
            if untrusted_uri:
                node["Nodes"] = {"uri": "https://evil.example/steal"}
            result.append(node)
    if role == "annotation":
        result.append(
            {
                "Id": "calibration",
                "Name": "calibration",
                "ContentLength": 0,
                "ChildrenNumber": 12,
            }
        )
    return {"result": result}


class _JSONResponse(io.BytesIO):
    """Minimal urllib response carrying one JSON document."""

    def __init__(self, document: Any) -> None:
        payload = (
            document if isinstance(document, bytes) else json.dumps(document).encode()
        )
        super().__init__(payload)
        self.headers = {"Content-Encoding": "identity"}


class _Opener:
    """Return or raise one scripted outcome for each request."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[urllib.request.Request] = []

    def open(self, request: urllib.request.Request, **kwargs: Any) -> Any:
        del kwargs
        self.calls.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return _JSONResponse(outcome)


def _run_lookup(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[Any],
    *,
    parent: str = PARENT,
    budget: RemoteResourceBudget | None = None,
) -> tuple[_Opener, _CallLedger, Any]:
    opener = _Opener(outcomes)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: opener)
    ledger = _CallLedger(budget or RemoteResourceBudget())
    item = lookup_cdse_safe_record(parent, ledger=ledger)
    return opener, ledger, item


def test_exact_parent_lookup_returns_every_iw_swath_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact parent binds IW1/IW2/IW3 measurement and annotation nodes."""
    opener, ledger, item = _run_lookup(
        monkeypatch,
        [
            _product_document(),
            _node_document("annotation", untrusted_uri=True),
            _node_document("measurement", untrusted_uri=True),
        ],
    )

    assert item is not None
    assert item.item_id == PARENT
    assert set(item.assets) == {
        f"{role}-{swath}-{polarization}"
        for role in ("annotation", "measurement")
        for swath in ("iw1", "iw2", "iw3")
        for polarization in ("vv", "vh")
    }
    assert item.assets["measurement-iw2-vv"].size_bytes == 1_000_003
    assert item.assets["annotation-iw3-vv"].size_bytes == 100_004
    assert item.raw_metadata["relative_orbit"] == 15
    assert item.raw_metadata["orbit_direction"] == "ASCENDING"
    assert item.acquisition.start == datetime(
        2015, 4, 28, 17, 14, 37, 931000, tzinfo=UTC
    )
    assert ledger.requests == 3
    assert ledger.response_bytes_total > 0
    assert "$expand=Attributes" in opener.calls[0].full_url
    assert f"Name%20eq%20%27{SAFE_NAME}%27" in opener.calls[0].full_url
    assert opener.calls[0].full_url.startswith(
        f"{CDSE_CATALOGUE_ORIGIN}/odata/v1/Products"
    )
    assert all(
        call.full_url.startswith(f"{CDSE_DOWNLOAD_ORIGIN}/odata/v1/Products")
        for call in opener.calls[1:]
    )
    assert all(call.get_header("Authorization") is None for call in opener.calls)


def test_exact_parent_lookup_returns_all_requested_polarizations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both VV and VH pairs are bound so the caller can filter later."""
    item = None
    opener = _Opener(
        [
            _product_document(),
            _node_document("annotation"),
            _node_document("measurement"),
        ]
    )
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: opener)
    item = lookup_cdse_safe_record(PARENT, ledger=_CallLedger(RemoteResourceBudget()))
    assert item is not None
    assert {key for key in item.assets if key.startswith("measurement-")} == {
        "measurement-iw1-vv",
        "measurement-iw1-vh",
        "measurement-iw2-vv",
        "measurement-iw2-vh",
        "measurement-iw3-vv",
        "measurement-iw3-vh",
    }


def test_exact_parent_lookup_rejects_a_different_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A neighboring parent or version is never substituted."""
    neighbor = SAFE_NAME.replace("87A1", "AAAA")
    opener = _Opener([_product_document(name=neighbor)])
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: opener)
    ledger = _CallLedger(RemoteResourceBudget())
    with pytest.raises(RemoteAccessError) as error:
        lookup_cdse_safe_record(PARENT, ledger=ledger)
    assert error.value.reason == "invalid_product"
    assert ledger.requests == 1
    assert len(opener.calls) == 1


def test_exact_parent_lookup_rejects_ambiguous_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two exact-name matches fail closed instead of picking the first."""
    document = _product_document()
    document["value"].append(document["value"][0])
    opener = _Opener([document])
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: opener)
    ledger = _CallLedger(RemoteResourceBudget())
    with pytest.raises(RemoteAccessError) as error:
        lookup_cdse_safe_record(PARENT, ledger=ledger)
    assert error.value.reason == "ambiguous_product"
    assert ledger.requests == 1


def test_exact_parent_lookup_rejects_invalid_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-UUID product id cannot become a request path."""
    opener = _Opener([_product_document(product_id="../../evil")])
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: opener)
    with pytest.raises(RemoteAccessError) as error:
        lookup_cdse_safe_record(PARENT, ledger=_CallLedger(RemoteResourceBudget()))
    assert error.value.reason == "invalid_product"


def test_exact_parent_lookup_returns_none_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No exact product is a missing parent, not a substitute."""
    opener = _Opener([{"value": []}])
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: opener)
    ledger = _CallLedger(RemoteResourceBudget())
    assert lookup_cdse_safe_record(PARENT, ledger=ledger) is None
    assert ledger.requests == 1


def test_exact_parent_lookup_fails_closed_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed OData JSON is an access error, never an empty result."""
    opener = _Opener([b"<html>not json</html>"])
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: opener)
    with pytest.raises(RemoteAccessError) as error:
        lookup_cdse_safe_record(PARENT, ledger=_CallLedger(RemoteResourceBudget()))
    assert error.value.reason == "malformed_results"


def test_exact_parent_lookup_honors_the_shared_request_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The metadata lookup cannot exceed the operation's request budget."""
    opener = _Opener(
        [
            _product_document(),
            _node_document("annotation"),
            _node_document("measurement"),
        ]
    )
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: opener)
    ledger = _CallLedger(RemoteResourceBudget(max_requests=2))
    with pytest.raises(RemoteLimitError) as error:
        lookup_cdse_safe_record(PARENT, ledger=ledger)
    assert error.value.reason == "max_requests"
    assert ledger.requests == 3
