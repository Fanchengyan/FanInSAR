"""Contract tests for the optional ``asf-search`` discovery engine."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pytest
import requests

from faninsar import remote
from faninsar.remote.providers.asf_search import (
    ASFSearchAdapter,
    ASFSearchError,
    EngineUnavailableError,
    UnsupportedASFSearchVersionError,
    _probe_burst_safe_url,
    _transfer_burst_safe_url,
    _with_asset_link,
)


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


class _Page(list[Mapping[str, Any]]):
    """ASF page fixture carrying the package completion marker."""

    def __init__(self, products: Iterable[Mapping[str, Any]], complete: bool) -> None:
        super().__init__(products)
        self.searchComplete = complete


class _TwoPageModule(_ASFModule):
    """ASF module fixture exposing an intermediate and terminal page."""

    def search_generator(
        self, *, opts: _ASFOptions
    ) -> Iterable[list[Mapping[str, Any]]]:
        self.options = opts
        for complete in (False, True):
            opts.values["session"].post(
                url="https://cmr.earthdata.nasa.gov/search/granules.umm_json",
                data=dict(opts.values),
                timeout=opts.values.get("timeout", 1.0),
            )
            yield _Page([self.product], complete)


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


def _geojson_product() -> Mapping[str, Any]:
    """Return the compact shape emitted by real ``ASFProduct.geojson``."""
    return {
        "type": "Feature",
        "id": "S1_WV_FIXTURE",
        "geometry": {"type": "Point", "coordinates": [0.5, 0.5]},
        "properties": {
            "sceneName": "S1_WV_FIXTURE",
            "startTime": "2026-09-03T07:10:57Z",
            "stopTime": "2026-09-03T07:11:15Z",
            "platform": "Sentinel-1D",
            "processingLevel": "SLC",
            "url": "https://datapool.asf.alaska.edu/SLC/SD/S1_WV_FIXTURE.zip",
            "bytes": 156_585_594,
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


def test_asf_search_yields_intermediate_and_terminal_pages() -> None:
    """Intermediate ``searchComplete=False`` pages are valid ASF results."""
    module = _TwoPageModule(_product())
    session = _Session(_Response(b'{"items":[1]}'))
    adapter = ASFSearchAdapter(
        collection="S1-SLC",
        package=module,
        session_factory=lambda: session,
    )
    ledger = remote._CallLedger(remote.RemoteResourceBudget())

    records = list(adapter.items(ledger=ledger))

    assert len(records) == 2
    assert ledger.requests == 2
    assert session.closed


def test_remote_search_fetches_terminal_page_before_applying_public_limit() -> None:
    """A small public limit cannot truncate an incomplete ASF traversal."""
    module = _TwoPageModule(_product())
    session = _Session(_Response(b'{"items":[1]}'))
    adapter = ASFSearchAdapter(
        collection="S1-SLC",
        package=module,
        session_factory=lambda: session,
    )
    remote._register_adapter("asf-complete-before-limit", adapter)

    items = remote.search(
        remote.Points([(0.5, 0.5)], crs=4326),
        catalog="asf-complete-before-limit",
        limit=1,
    )

    assert len(items) == 1
    assert len(session.posts) == 2


def test_asf_search_registers_current_download_redirect_origins() -> None:
    """Current ASF Data Pool and OAuth hops are admitted internally."""
    adapter = ASFSearchAdapter(collection="sentinel-1", package=_ASFModule(_product()))

    assert "https://sentinel1.asf.alaska.edu" in adapter.redirect_origins
    assert "https://cumulus.asf.alaska.edu" in adapter.redirect_origins


def test_asf_scene_alias_uses_registered_cmr_concept_id() -> None:
    """The S1-SLC alias resolves to ASF's full-scene collection identity."""
    adapter = ASFSearchAdapter(collection="S1-SLC", package=_ASFModule(_product()))

    assert adapter.query_collection == "C1214470488-ASF"


def test_asf_burst_collection_uses_registered_cmr_concept_id() -> None:
    """The SLC-BURST alias resolves to ASF's burst collection identity."""
    adapter = ASFSearchAdapter(collection="SLC-BURST", package=_ASFModule(_product()))

    assert adapter.query_collection == "C2709161906-ASF"


def test_asf_search_preserves_declared_complete_safe_size() -> None:
    """ASF's exact byte count reaches the normalized remote asset."""
    module = _ASFModule(_geojson_product())
    session = _Session(_Response(b'{"items":[1]}'))
    adapter = ASFSearchAdapter(
        collection="sentinel-1",
        package=module,
        session_factory=lambda: session,
    )
    remote._register_adapter("asf-size", adapter)

    items = remote.search(
        remote.Points([(0.5, 0.5)], crs=4326),
        catalog="asf-size",
        auth_profile="earthdata-asf",
        limit=1,
    )

    assert items[0].assets["data"].size_bytes == 156_585_594


def test_asf_search_maps_md5sum_to_complete_asset_checksum() -> None:
    """ASF's published ``md5sum`` reaches the complete SAFE asset as ``md5:<hex>``."""
    md5 = "d41d8cd98f00b204e9800998ecf8427e"
    product = dict(_geojson_product())
    product["properties"] = {**product["properties"], "md5sum": md5}
    module = _ASFModule(product)
    session = _Session(_Response(b'{"items":[1]}'))
    adapter = ASFSearchAdapter(
        collection="sentinel-1",
        package=module,
        session_factory=lambda: session,
    )
    remote._register_adapter("asf-md5sum", adapter)

    items = remote.search(
        remote.Points([(0.5, 0.5)], crs=4326),
        catalog="asf-md5sum",
        auth_profile="earthdata-asf",
        limit=1,
    )

    assert items[0].assets["data"].checksum == f"md5:{md5}"


def test_asf_search_without_md5sum_leaves_checksum_none() -> None:
    """A burst record with no ``md5sum`` keeps the complete asset unverifiable."""
    product = _geojson_product()
    module = _ASFModule(product)
    session = _Session(_Response(b'{"items":[1]}'))
    adapter = ASFSearchAdapter(
        collection="sentinel-1",
        package=module,
        session_factory=lambda: session,
    )
    remote._register_adapter("asf-no-md5sum", adapter)

    items = remote.search(
        remote.Points([(0.5, 0.5)], crs=4326),
        catalog="asf-no-md5sum",
        auth_profile="earthdata-asf",
        limit=1,
    )

    assert items[0].assets["data"].checksum is None


def test_asf_search_translates_common_direction_and_burst_filters() -> None:
    """Public common fields reach ASF's provider vocabulary exactly once."""
    module = _ASFModule(_product())
    session = _Session(_Response(b'{"items":[1]}'))
    adapter = ASFSearchAdapter(
        collection="sentinel-1",
        package=module,
        session_factory=lambda: session,
    )
    remote._register_adapter("asf-query-fields", adapter)

    remote.search(
        remote.Points([(0.5, 0.5)], crs=4326),
        catalog="asf-query-fields",
        flight_direction="ASCENDING",
        filters={"dataset": "SLC-BURST", "fullBurstID": "burst-1"},
        limit=1,
    )

    assert module.options.values["flightDirection"] == "ASCENDING"
    assert module.options.values["dataset"] == "SLC-BURST"
    assert module.options.values["fullBurstID"] == "burst-1"
    assert module.options.values["collections"] == "C2709161906-ASF"


def test_asf_search_rejects_duplicate_common_filter_before_session_creation() -> None:
    """A provider spelling cannot silently shadow a first-class field."""
    created: list[object] = []
    adapter = ASFSearchAdapter(
        collection="sentinel-1",
        package=_ASFModule(_product()),
        session_factory=lambda: created.append(object()),
    )
    remote._register_adapter("asf-duplicate-fields", adapter)

    with pytest.raises(remote.RemoteQueryError) as error:
        remote.search(
            remote.Points([(0.5, 0.5)], crs=4326),
            catalog="asf-duplicate-fields",
            flight_direction="ASCENDING",
            filters={"flightDirection": "ASCENDING"},
        )

    assert error.value.reason == "duplicate_query_option"
    assert created == []


def test_asf_burst_asset_links_preserve_tiff_and_xml_representations() -> None:
    """Burst TIFF, XML, and SAFE ZIP links retain their actual formats."""
    product = _geojson_product()
    product["properties"] = {
        **product["properties"],
        "url": "https://sentinel1.asf.alaska.edu/burst.tiff",
        "additionalUrls": [
            "https://sentinel1.asf.alaska.edu/burst.xml",
            "https://datapool.asf.alaska.edu/burst.zip",
        ],
    }

    linked = _with_asset_link(dict(product), product)

    assert [link["title"] for link in linked["links"]] == [
        "SAFE ZIP",
        "BURST TIFF",
        "BURST XML",
    ]
    assert linked["links"][2]["rel"] == "related"


def test_compact_burst_record_exposes_complete_safe_asset() -> None:
    """A compact burst product promotes its additional SAFE ZIP URL."""
    from faninsar.missions.s1.search import complete_safe_asset

    product = dict(_geojson_product())
    product["properties"] = {
        **product["properties"],
        "url": "https://sentinel1.asf.alaska.edu/burst.tiff",
        "additionalUrls": [
            "https://sentinel1.asf.alaska.edu/burst.xml",
            "https://sentinel1-burst.asf.alaska.edu/jobs/burst.zip",
        ],
    }
    module = _ASFModule(product)
    session = _Session(_Response(b'{"items":[1]}'))
    adapter = ASFSearchAdapter(
        collection="sentinel-1",
        package=module,
        session_factory=lambda: session,
    )
    remote._register_adapter("asf-compact-burst", adapter)

    item = remote.search(
        remote.Points([(0.5, 0.5)], crs=4326),
        catalog="asf-compact-burst",
        limit=1,
    )[0]

    safe = complete_safe_asset(item)
    assert safe.href.endswith("/burst.zip")
    assert safe.representation == "complete"
    assert all(not asset.href.endswith(".tiff") for asset in (safe,))


def test_asf_umm_burst_service_safe_is_selected_as_complete_asset() -> None:
    """The Burst Extractor SAFE service URL wins over a TIFF data link."""
    product = dict(_product())
    product["DataGranule"] = {
        "RelatedUrls": [
            {
                "Type": "GET DATA",
                "Name": "BURST TIFF",
                "URL": "https://sentinel1.asf.alaska.edu/burst.tiff",
            },
            {
                "Type": "USE SERVICE API",
                "Format": "SAFE",
                "Name": "Burst SAFE",
                "URL": "https://sentinel1-burst.asf.alaska.edu/jobs/burst.zip",
            },
        ]
    }
    module = _ASFModule(product)
    session = _Session(_Response(b'{"items":[1]}'))
    adapter = ASFSearchAdapter(
        collection="sentinel-1",
        package=module,
        session_factory=lambda: session,
    )
    remote._register_adapter("asf-burst-safe", adapter)

    item = remote.search(
        remote.Points([(0.5, 0.5)], crs=4326),
        catalog="asf-burst-safe",
        limit=1,
    )[0]

    assert item.assets["data"].href == (
        "https://sentinel1-burst.asf.alaska.edu/jobs/burst.zip"
    )
    assert "https://sentinel1-burst.asf.alaska.edu" in adapter.origins


def test_asf_rejects_multiple_complete_safe_urls() -> None:
    """Multiple distinct SAFE ZIP representations fail at ASF normalization."""
    product = dict(_product())
    product["DataGranule"] = {
        "RelatedUrls": [
            {
                "Type": "GET DATA",
                "Name": "SAFE ZIP",
                "URL": "https://datapool.asf.alaska.edu/SLC/one.zip",
            },
            {
                "Type": "GET DATA",
                "Name": "SAFE ZIP",
                "URL": "https://datapool.asf.alaska.edu/SLC/two.zip",
            },
        ]
    }
    module = _ASFModule(product)
    session = _Session(_Response(b'{"items":[1]}'))
    adapter = ASFSearchAdapter(
        collection="sentinel-1",
        package=module,
        session_factory=lambda: session,
    )

    with pytest.raises(ASFSearchError) as raised:
        list(adapter.items(ledger=remote._CallLedger(remote.RemoteResourceBudget())))

    assert getattr(raised.value, "reason", None) == "ambiguous_representation"


class _PollingResponse:
    """Small HEAD response fixture for burst extractor polling."""

    def __init__(
        self,
        status: int,
        *,
        location: str | None = None,
        retry_after: str | None = None,
    ) -> None:
        self.status = status
        self.code = status
        self.headers = {"Location": location} if location else {}
        if retry_after is not None:
            self.headers["Retry-After"] = retry_after
        self.body = b"queued" if status == 202 else b""
        self.closed = False

    def read(self, _size: int = -1) -> bytes:
        """Return the bounded polling response body."""
        return self.body

    def geturl(self) -> str:
        """Return the generated object URL."""
        return "https://sentinel1-burst.asf.alaska.edu/jobs/burst.zip"

    def close(self) -> None:
        """Mark the response as closed."""
        self.closed = True


def test_asf_burst_transfer_performs_one_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Burst SAFE transfer setup performs one observation only."""
    responses = iter(
        [
            _PollingResponse(
                202,
                location="https://sentinel1-burst.asf.alaska.edu/jobs/burst.zip",
            )
        ]
    )

    class _Opener:
        def open(self, _request: Any, *, timeout: float) -> _PollingResponse:
            del timeout
            return next(responses)

    monkeypatch.setattr("urllib.request.build_opener", lambda *_handlers: _Opener())
    adapter = ASFSearchAdapter(collection="sentinel-1", package=_ASFModule(_product()))
    ledger = remote._CallLedger(remote.RemoteResourceBudget(max_retries=2))

    with pytest.raises(remote.BurstPendingError, match="burst_pending"):
        _transfer_burst_safe_url(
            "https://sentinel1-burst.asf.alaska.edu/jobs/burst.zip",
            adapter,
            ledger.budget,
            ledger,
        )

    assert ledger.requests == 1
    assert ledger.retries == 0
    assert ledger.response_bytes_total == len(b"queued")


def test_asf_burst_poll_budget_and_backoff_are_dedicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fifty observations are allowed without expanding ordinary retries."""
    responses = iter(
        [_PollingResponse(202) for _ in range(49)] + [_PollingResponse(200)]
    )
    delays: list[float] = []

    class _Opener:
        def open(self, _request: Any, *, timeout: float) -> _PollingResponse:
            del timeout
            return next(responses)

    monkeypatch.setattr("urllib.request.build_opener", lambda *_handlers: _Opener())
    adapter = ASFSearchAdapter(collection="sentinel-1", package=_ASFModule(_product()))
    ledger = remote._CallLedger(
        remote.RemoteResourceBudget(max_retries=1, max_poll_attempts=50)
    )

    url = "https://sentinel1-burst.asf.alaska.edu/jobs/burst.zip"
    for attempt in range(50):
        observation = _probe_burst_safe_url(
            url, adapter, ledger.budget, ledger, attempt
        )
        if observation.state == "ready":
            break
        delays.append(observation.delay_seconds)
        ledger.poll_retry()
        url = observation.url

    assert ledger.requests == 50
    assert ledger.retries == 49
    assert delays[:4] == [2.0, 4.0, 8.0, 10.0]
    assert all(delay <= 10.0 for delay in delays)


def test_asf_burst_poll_limit_rejects_fiftieth_pending_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fiftieth pending response fails without a fifty-first request."""
    responses = iter([_PollingResponse(202) for _ in range(50)])

    class _Opener:
        def open(self, _request: Any, *, timeout: float) -> _PollingResponse:
            del timeout
            return next(responses)

    monkeypatch.setattr("urllib.request.build_opener", lambda *_handlers: _Opener())
    adapter = ASFSearchAdapter(collection="sentinel-1", package=_ASFModule(_product()))
    ledger = remote._CallLedger(remote.RemoteResourceBudget(max_poll_attempts=50))

    url = "https://sentinel1-burst.asf.alaska.edu/jobs/burst.zip"
    for attempt in range(50):
        observation = _probe_burst_safe_url(url, adapter, ledger.budget, ledger)
        assert observation.state == "pending"
        if attempt == 49:
            with pytest.raises(remote.RemoteLimitError, match="max_poll_attempts"):
                remote._fail(remote.RemoteLimitError, "max_poll_attempts")
            break
        ledger.poll_retry()
        url = observation.url

    assert ledger.requests == 50


class _RedirectTransport(requests.adapters.BaseAdapter):
    """Requests transport returning one denied redirect."""

    def __init__(self) -> None:
        self.requests: list[requests.PreparedRequest] = []

    def send(
        self, request: requests.PreparedRequest, **kwargs: Any
    ) -> requests.Response:
        del kwargs
        self.requests.append(request)
        response = requests.Response()
        response.status_code = 302
        response.headers["Location"] = "https://evil.invalid/steal"
        response.url = request.url
        response.request = request
        response._content = b"redirect"
        return response

    def close(self) -> None:
        """Close the fixture transport."""


def test_asf_search_denies_redirect_before_following_credentials() -> None:
    """A denied Location is rejected before Requests sends the next hop."""

    class RedirectModule(_ASFModule):
        def search_generator(
            self, *, opts: _ASFOptions
        ) -> Iterable[list[Mapping[str, Any]]]:
            self.options = opts
            opts.values["session"].post(
                url="https://cmr.earthdata.nasa.gov/search/granules.umm_json",
                headers={
                    "Authorization": "Bearer test-secret",
                    "Cookie": "session=test-secret",
                },
            )
            yield []

    session = requests.Session()
    transport = _RedirectTransport()
    session.mount("https://", transport)
    adapter = ASFSearchAdapter(
        collection="S1-SLC",
        package=RedirectModule(_product()),
        session_factory=lambda: session,
    )

    with pytest.raises(remote.RemoteAccessError, match="unregistered_endpoint"):
        list(adapter.items(ledger=remote._CallLedger(remote.RemoteResourceBudget())))

    assert len(transport.requests) == 1
    assert transport.requests[0].url.startswith("https://cmr.earthdata.nasa.gov/")


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
