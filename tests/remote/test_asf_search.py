"""Contract tests for the optional ``asf-search`` discovery engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from faninsar import remote
from faninsar.remote.providers.asf_search import (
    ASFSearchAdapter,
    EngineUnavailableError,
    UnsupportedASFSearchVersionError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class _Response:
    """Minimal response object exposing the bytes the ledger must charge."""

    def __init__(self, body: bytes, *, status_code: int = 200) -> None:
        self.content = body
        self.status_code = status_code
        self.history: list[Any] = []
        self.is_redirect = False

    def json(self) -> Mapping[str, Any]:
        """Return a harmless response payload."""
        return {}


class _Session:
    """Requests-like fixture session with an observable POST boundary."""

    def __init__(self, response: _Response) -> None:
        self.response = response
        self.posts: list[dict[str, Any]] = []
        self.hooks: dict[str, list[Any]] = {"response": []}
        self.closed = False

    def post(self, *, url: str, data: Any = None, timeout: float = 0.0) -> _Response:
        """Record one package request and return its fixture response."""
        self.posts.append({"url": url, "data": data, "timeout": timeout})
        return self.response

    def close(self) -> None:
        """Record operation cleanup."""
        self.closed = True


class _ASFOptions:
    """Minimal options class accepting the injected session."""

    def __init__(self, **kwargs: Any) -> None:
        self.values = kwargs


class _ASFModule:
    """Minimal asf-search module fixture."""

    __version__ = "13.0.0"
    ASFSearchOptions = _ASFOptions

    def __init__(self, product: Mapping[str, Any]) -> None:
        self.product = product
        self.options: Any = None

    def search_generator(
        self, *, opts: _ASFOptions
    ) -> Iterable[list[Mapping[str, Any]]]:
        """Yield one page while exercising the supplied operation session."""
        self.options = opts
        response = opts.values["session"].post(
            url="https://cmr.earthdata.nasa.gov/search/granules.umm_json",
            data=dict(opts.values),
            timeout=opts.values.get("timeout", 1.0),
        )
        del response
        yield [self.product]


def _product() -> Mapping[str, Any]:
    """Return a minimal UMM product fixture."""
    return {
        "GranuleUR": "S1_FIXTURE",
        "Provider": "ASF",
        "CollectionReference": {"ShortName": "S1-SLC"},
        "TemporalExtents": [
            {"SingleDateTime": "2024-01-01T00:00:00Z"},
        ],
        "SpatialExtent": {
            "HorizontalSpatialDomain": {
                "Geometry": {
                    "BoundingRect": {
                        "WestBoundingCoordinate": 0,
                        "EastBoundingCoordinate": 1,
                        "SouthBoundingCoordinate": 0,
                        "NorthBoundingCoordinate": 1,
                    }
                }
            }
        },
        "Platforms": [
            {"ShortName": "SENTINEL-1", "Instruments": [{"ShortName": "C-SAR"}]}
        ],
        "DataGranule": {
            "RelatedUrls": [
                {
                    "Type": "GET DATA",
                    "Name": "SAFE ZIP",
                    "URL": "https://datapool.asf.alaska.edu/SLC/S1_FIXTURE.zip",
                }
            ]
        },
    }


def test_unsupported_asf_search_fails_before_session_or_provider_io() -> None:
    """An unsupported package version is rejected before constructing a session."""
    created: list[object] = []
    module = _ASFModule(_product())
    module.__version__ = "12.2.3"
    adapter = ASFSearchAdapter(
        collection="S1-SLC",
        package=module,
        session_factory=lambda: created.append(object()),
    )

    with pytest.raises(UnsupportedASFSearchVersionError) as error:
        list(adapter.items(ledger=remote._CallLedger(remote.RemoteResourceBudget())))

    assert error.value.reason == "unsupported_asf_search_version"
    assert created == []


def test_asf_search_uses_supplied_operation_session_and_normalizes_product() -> None:
    """Package requests and response bytes are charged to the supplied ledger."""
    module = _ASFModule(_product())
    session = _Session(_Response(b'{"items":[1]}'))
    adapter = ASFSearchAdapter(
        collection="S1-SLC",
        package=module,
        session_factory=lambda: session,
    )
    ledger = remote._CallLedger(remote.RemoteResourceBudget())

    records = list(adapter.items(ledger=ledger))

    assert [record["id"] for record in records] == ["S1_FIXTURE"]
    assert records[0]["provider"] == "ASF"
    assert records[0]["assets"]["data"]["href"].endswith("S1_FIXTURE.zip")
    assert "ASFProduct" not in repr(records)
    assert ledger.requests == 1
    assert ledger.response_bytes_total == len(b'{"items":[1]}')
    assert session.closed


def test_asf_search_failure_does_not_fall_back_to_cmr() -> None:
    """A package failure is terminal and does not invoke a second engine."""

    class FailingModule(_ASFModule):
        def search_generator(self, *, opts: _ASFOptions) -> Iterable[list[Any]]:
            del opts
            raise RuntimeError
            yield []

    session = _Session(_Response(b"{}"))
    adapter = ASFSearchAdapter(
        collection="S1-SLC",
        package=FailingModule(_product()),
        session_factory=lambda: session,
    )

    with pytest.raises(EngineUnavailableError) as error:
        list(adapter.items(ledger=remote._CallLedger(remote.RemoteResourceBudget())))

    assert error.value.reason == "asf_search_failed"
    assert session.posts == []
