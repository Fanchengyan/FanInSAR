"""Fixed, opt-in P0047 live transfer verifier.

The transfer verifier is inert unless ``FANINSAR_P0047_LIVE=1`` is set.  Its
deterministic discovery checks remain available without network access.  Live
payloads and evidence are written only below the caller-selected temporary
root.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.parse
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest
import requests

from faninsar import remote
from faninsar.data.query import BoundingBox

_LIVE_ENABLED = os.environ.get("FANINSAR_P0047_LIVE") == "1"
_ASF_GRANULE = "G4297731264-ASF"
_ASF_ITEM = "S1D_WV_SLC__1SSV_20260903T071057_20260903T071115_004410_008296_C928"
_LPDAAC_GRANULE = "G2816843744-LPCLOUD"
_LPDAAC_ITEM = "NASADEM_HGT_n36w121"
_PC_ITEM = "Copernicus_DSM_COG_10_N39_00_W105_00_DEM"
_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
_ROOT = Path(
    os.environ.get("FANINSAR_P0047_LIVE_ROOT", f"/private/tmp/p0047-live-{_COMMIT}")
)
_REPORT = _ROOT / "report.json"
_REGISTERED_TERMINAL_ORIGINS = frozenset(
    {
        "https://datapool.asf.alaska.edu",
        "https://sentinel1.asf.alaska.edu",
        "https://nisar.asf.earthdatacloud.nasa.gov",
        "https://urs.earthdata.nasa.gov",
        "https://cumulus.asf.alaska.edu",
        "https://data.lpdaac.earthdatacloud.nasa.gov",
        "https://planetarycomputer.microsoft.com",
        "https://elevationeuwest.blob.core.windows.net",
    }
)


def _digest(path: Path) -> tuple[int, str]:
    """Return the size and SHA-256 of one completed file."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _representation(path: Path) -> str:
    """Classify the downloaded representation using the bounded prefix."""
    with path.open("rb") as stream:
        prefix = stream.read(512)
    suffix = path.suffix.lower()
    magic = {
        ".zip": prefix.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")),
        ".tif": prefix.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")),
        ".tiff": prefix.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")),
    }
    return "valid" if magic.get(suffix, True) else "invalid"


def _safe_url(value: str) -> dict[str, Any]:
    """Represent an outgoing URL without retaining its query capability."""
    parsed = urlsplit(value)
    return {
        "url": urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")),
        "query_present": bool(parsed.query),
    }


def _safe_headers(headers: Any) -> dict[str, Any]:
    """Record only credential-presence facts from prepared headers."""
    names = {str(name).lower() for name in getattr(headers, "keys", lambda: ())()}
    account_names = {
        name
        for name in names
        if name in {"authorization", "cookie", "proxy-authorization"}
        or any(token in name for token in ("token", "credential", "password"))
    }
    return {
        "account_headers": sorted(account_names),
        "has_account_headers": bool(account_names),
    }


class _LiveRecorder:
    """Capture prepared request facts and private ledger counters in memory."""

    def __init__(self) -> None:
        self.current: dict[str, Any] | None = None
        self.ledger_operations: dict[int, dict[str, Any]] = {}

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Install process-local hooks at prepared request boundaries."""
        original_init = remote._CallLedger.__post_init__
        original_request = remote._CallLedger.request
        original_retry = remote._CallLedger.retry
        original_redirect = remote._CallLedger.redirect
        original_response_bytes = remote._CallLedger.response_bytes
        original_send = requests.sessions.Session.send
        original_open = remote.urllib.request.OpenerDirector.open

        def ledger_init(ledger: remote._CallLedger) -> None:
            original_init(ledger)
            if self.current is not None:
                self.ledger_operations[id(ledger)] = self.current

        def ledger_request(ledger: remote._CallLedger) -> None:
            self._counter(ledger, "requests_count")
            original_request(ledger)

        def ledger_retry(ledger: remote._CallLedger) -> None:
            self._counter(ledger, "retries")
            original_retry(ledger)

        def ledger_redirect(ledger: remote._CallLedger) -> None:
            self._counter(ledger, "redirects")
            original_redirect(ledger)

        def ledger_response_bytes(ledger: remote._CallLedger, count: int) -> None:
            original_response_bytes(ledger, count)
            operation = self.ledger_operations.get(id(ledger))
            if operation is not None:
                operation["response_bytes"] = (
                    operation.get("response_bytes", 0) + count
                )

        def send(
            session: requests.Session,
            request: requests.PreparedRequest,
            *args: Any,
            **kwargs: Any,
        ) -> requests.Response:
            self._request(request.url, request.headers, "requests")
            response = original_send(session, request, *args, **kwargs)
            self._response(response)
            return response

        def open_request(
            opener: remote.urllib.request.OpenerDirector,
            request: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            url = getattr(request, "full_url", request)
            self._request(str(url), getattr(request, "headers", {}), "urllib")
            response = original_open(opener, request, *args, **kwargs)
            self._response(response)
            return response

        monkeypatch.setattr(remote._CallLedger, "__post_init__", ledger_init)
        monkeypatch.setattr(remote._CallLedger, "request", ledger_request)
        monkeypatch.setattr(remote._CallLedger, "retry", ledger_retry)
        monkeypatch.setattr(remote._CallLedger, "redirect", ledger_redirect)
        monkeypatch.setattr(remote._CallLedger, "response_bytes", ledger_response_bytes)
        monkeypatch.setattr(requests.sessions.Session, "send", send)
        monkeypatch.setattr(remote.urllib.request.OpenerDirector, "open", open_request)

    def operation(self, name: str, callback: Any) -> Any:
        """Run one public operation while associating all private evidence."""
        operation: dict[str, Any] = {
            "name": name,
            "started": time.monotonic(),
            "requests": [],
            "requests_count": 0,
            "retries": 0,
            "redirects": 0,
            "response_bytes": 0,
        }
        self.current = operation
        try:
            return callback()
        finally:
            operation["elapsed_seconds"] = time.monotonic() - operation["started"]
            self.current = None

    def _counter(self, ledger: remote._CallLedger, name: str) -> None:
        operation = self.ledger_operations.get(id(ledger))
        if operation is not None:
            operation[name] = operation.get(name, 0) + 1

    def _request(self, url: str, headers: Any, transport: str) -> None:
        if self.current is not None:
            self.current["requests"].append(
                {"transport": transport, **_safe_url(url), **_safe_headers(headers)}
            )

    def _response(self, response: Any) -> None:
        if self.current is None:
            return
        headers = getattr(response, "headers", {})
        status = getattr(response, "status_code", getattr(response, "status", None))
        raw_url = getattr(response, "url", None)
        if not isinstance(raw_url, str):
            geturl = getattr(response, "geturl", None)
            raw_url = geturl() if callable(geturl) else ""
        self.current.setdefault("responses", []).append(
            {
                **_safe_url(str(raw_url)),
                "status": status,
                "content_range": any(
                    str(name).lower() == "content-range" and value
                    for name, value in getattr(headers, "items", lambda: ())()
                ),
            }
        )

    def ledger(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Return only the stable ledger fields for persisted evidence."""
        return {
            "requests": operation.get("requests_count", 0),
            "retries": operation.get("retries", 0),
            "redirects": operation.get("redirects", 0),
            "response_bytes": operation.get("response_bytes", 0),
            "elapsed_seconds": operation.get("elapsed_seconds", 0.0),
        }

    def last_operation(self, name: str) -> dict[str, Any]:
        """Return the most recent operation with one stable name."""
        # ``ledger_operations`` retains each operation in insertion order;
        # no request or credential value is included in this lookup.
        for operation in reversed(list(self.ledger_operations.values())):
            if operation.get("name") == name:
                return operation
        return {
            "name": name,
            "requests": [],
            "responses": [],
            "requests_count": 0,
            "retries": 0,
            "redirects": 0,
            "response_bytes": 0,
            "elapsed_seconds": 0.0,
        }


def _lane_result(
    lane: str,
    identity: str,
    item: remote.CatalogItem,
    destination: Path,
    recorder: _LiveRecorder,
) -> dict[str, Any]:
    """Download one discovered asset and return the fixed sanitized schema."""
    asset = next(iter(item.assets.values()))
    download_operation = recorder.operation(
        "download", lambda: remote.download(asset, destination, overwrite=True)
    )
    del download_operation
    size, digest = _digest(destination)
    operation = recorder.last_operation("download")
    requests_seen = operation.get("requests", [])
    responses = operation.get("responses", [])
    terminal = responses[-1] if responses else {}
    terminal_request = requests_seen[-1] if requests_seen else {}
    search_operation = recorder.last_operation("search")
    evidence_path = destination.with_name("requests.json")
    evidence_path.write_text(
        json.dumps(
            {"search": search_operation.get("requests", []), "download": requests_seen},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if (
        terminal.get("status") != 200
        or terminal.get("content_range", False)
        or terminal_request.get("has_account_headers", True)
        or _representation(destination) != "valid"
    ):
        remote._fail(remote.RemoteAccessError, "live_transfer_invariant")
    return {
        "lane": lane,
        "status": "PASS",
        "identity": identity,
        "cmr_concept_id": (
            item.raw_metadata.get("properties", {}).get("cmr_concept_id")
            if isinstance(item.raw_metadata.get("properties", {}), Mapping)
            else None
        ),
        "provider": item.provider,
        "collection": item.collection,
        "item_id": item.item_id,
        "search_ledger": recorder.ledger(search_operation),
        "download_ledger": recorder.ledger(operation),
        "elapsed_seconds": operation.get("elapsed_seconds", 0.0),
        "bytes": size,
        "sha256": digest,
        "representation": _representation(destination),
        "final_origin": _origin(terminal.get("url", asset.href)),
        "final_status": terminal.get("status"),
        "content_range": terminal.get("content_range", False),
        "terminal_account_headers": terminal_request.get("has_account_headers", True),
        "signed_query_persisted": False,
        "request_evidence": str(evidence_path),
        "destination": str(destination),
    }


def _origin(value: str) -> str:
    """Return a URL origin for sanitized live evidence."""
    parsed = urlsplit(value)
    return f"{parsed.scheme}://{parsed.netloc}"


def _select_pinned_item(
    items: list[remote.CatalogItem], concept_id: str | None, item_id: str
) -> remote.CatalogItem | None:
    """Select an item only when its CMR and normalized identities both match.

    CMR compact JSON uses ``id`` for the granule concept identifier while the
    normalized remote contract uses ``producer_granule_id`` as ``item_id``.
    Keeping these checks separate prevents an accidental match on either
    identity alone.
    """
    for item in items:
        properties = item.raw_metadata.get("properties", {})
        returned_concept = (
            properties.get("cmr_concept_id")
            if isinstance(properties, Mapping)
            else None
        )
        if (
            concept_id is None or returned_concept == concept_id
        ) and item.item_id == item_id:
            return item
    return None


def _register_lanes() -> tuple[tuple[str, str | None, str, str], ...]:
    """Register the three fixed production lanes."""
    from faninsar.remote import register_cmr_catalog, register_pc_catalog

    register_cmr_catalog(
        "p0047-live-asf",
        provider="ASF",
        collection="sentinel-1",
        endpoint=(
            "https://cmr.earthdata.nasa.gov/search/granules.json"
            f"?concept_id={_ASF_GRANULE}"
        ),
        collection_concept_id="C4175278193-ASF",
        profiles=("earthdata-asf",),
        data_origins=("https://datapool.asf.alaska.edu",),
    )
    register_cmr_catalog(
        "p0047-live-lpdaac",
        provider="LPCLOUD",
        collection="NASADEM",
        endpoint=(
            "https://cmr.earthdata.nasa.gov/search/granules.json"
            f"?concept_id={_LPDAAC_GRANULE}"
        ),
        collection_concept_id="C2763264762-LPCLOUD",
        profiles=("earthdata-lpdaac",),
        data_origins=("https://data.lpdaac.earthdatacloud.nasa.gov",),
    )
    register_pc_catalog("p0047-live-pc")
    return (
        ("ASF SAFE", _ASF_GRANULE, _ASF_ITEM, "p0047-live-asf"),
        ("LP DAAC NASADEM", _LPDAAC_GRANULE, _LPDAAC_ITEM, "p0047-live-lpdaac"),
        ("Planetary Computer DEM", None, _PC_ITEM, "p0047-live-pc"),
    )


def test_pinned_item_selector_requires_both_identities() -> None:
    """CMR concept IDs and normalized product IDs are checked independently."""
    from faninsar.remote.cmr import CMRCollectionAdapter

    adapter = CMRCollectionAdapter(
        provider="ASF",
        collection="sentinel-1",
        endpoint="https://cmr.invalid/search/granules.json",
        data_origins=("https://datapool.asf.alaska.edu",),
    )
    record = adapter._normalize(
        {
            "id": _ASF_GRANULE,
            "producer_granule_id": _ASF_ITEM,
            "collection_concept_id": "C4175278193-ASF",
            "polygons": ["0 0 0 1 1 1 0 0"],
            "links": [
                {
                    "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                    "href": "https://datapool.asf.alaska.edu/data/item.zip",
                }
            ],
        }
    )
    item = remote._normalize_record(record, "fixture", adapter, "anonymous")

    assert _select_pinned_item([item], _ASF_GRANULE, "wrong-item") is None
    assert _select_pinned_item([item], _ASF_GRANULE, _ASF_ITEM) is not None
    assert _select_pinned_item([item], "G0000000000-ASF", item.item_id) is None


def test_cmr_granule_pin_is_merged_with_public_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pinned CMR concept remains a server-side search parameter."""
    from faninsar.remote.cmr import CMRCollectionAdapter

    endpoint = f"https://cmr.invalid/search/granules.json?concept_id={_ASF_GRANULE}"
    adapter = CMRCollectionAdapter(
        provider="ASF",
        collection="sentinel-1",
        endpoint=endpoint,
    )
    seen_urls: list[str] = []

    def page(
        _self: CMRCollectionAdapter,
        url: str,
        _headers: Mapping[str, str],
        _ledger: Any,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        seen_urls.append(url)
        return {"feed": {"entry": []}}, {}

    monkeypatch.setattr(CMRCollectionAdapter, "_request_page", page)
    list(adapter.items(ledger=remote._CallLedger(remote.RemoteResourceBudget())))
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(seen_urls[0]).query)
    assert params["concept_id"] == [_ASF_GRANULE]


def test_asf_cmr_selection_uses_authenticated_download_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CMR ASF records retain the selected profile for authenticated transfer."""
    from faninsar.remote.cmr import CMRCollectionAdapter

    def page(
        _self: CMRCollectionAdapter,
        _url: str,
        _headers: Mapping[str, str],
        _ledger: Any,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        return {
            "feed": {
                "entry": [
                    {
                        "id": _ASF_GRANULE,
                        "producer_granule_id": _ASF_ITEM,
                        "collection_concept_id": "C4175278193-ASF",
                        "polygons": ["0 0 0 1 1 1 0 0"],
                        "links": [
                            {
                                "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                                "href": (
                                    "https://datapool.asf.alaska.edu/"
                                    "data/item.zip"
                                ),
                            }
                        ],
                    }
                ]
            }
        }, {}

    monkeypatch.setattr(CMRCollectionAdapter, "_request_page", page)
    _register_lanes()
    items = remote.search(
        BoundingBox(-1, -1, 2, 2, crs=4326),
        catalog="p0047-live-asf",
        auth_profile="earthdata-asf",
        limit=1,
    )
    selected = _select_pinned_item(items, _ASF_GRANULE, _ASF_ITEM)
    assert selected is not None
    asset = selected.assets["data"]
    assert selected.provider == "ASF"
    assert asset.auth_profile == "earthdata-asf"

    payload = b"PK\x03\x04fixture-safe-zip"
    calls: list[str] = []

    class _Response:
        def __init__(self) -> None:
            self.status_code = 200
            self.headers: dict[str, str] = {}

        def iter_content(self, chunk_size: int) -> list[bytes]:
            del chunk_size
            return [payload]

        def close(self) -> None:
            pass

    class _Session:
        def close(self) -> None:
            pass

    def authenticated_session(
        selected_asset: remote.RemoteAsset,
        _adapter: Any,
        _budget: remote.RemoteResourceBudget,
        _ledger: remote._CallLedger,
    ) -> tuple[_Session, str]:
        assert selected_asset.auth_profile == "earthdata-asf"
        return _Session(), "fixture-token"

    def asf_request(
        _session: _Session,
        method: str,
        _url: str,
        **_kwargs: Any,
    ) -> _Response:
        calls.append(method)
        return _Response()

    monkeypatch.setattr(remote, "_asf_authenticated_session", authenticated_session)
    monkeypatch.setattr(remote, "_asf_request", asf_request)
    monkeypatch.setattr(
        remote.urllib.request.OpenerDirector,
        "open",
        lambda *_args, **_kwargs: pytest.fail("ASF download used urllib"),
    )

    destination = tmp_path / "asset.zip"
    remote.download(asset, destination, overwrite=True)
    assert destination.read_bytes() == payload
    assert calls == ["GET"]


@pytest.mark.skipif(
    not _LIVE_ENABLED, reason="set FANINSAR_P0047_LIVE=1 for the opt-in live verifier"
)
def test_live_p0047_fixed_full_transfers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify fresh ASF, LP DAAC, and Planetary Computer complete transfers."""
    if _ROOT.exists():
        pytest.fail(f"live verifier root must be fresh: {_ROOT}")
    _ROOT.mkdir(parents=True)
    recorder = _LiveRecorder()
    recorder.install(monkeypatch)
    started = datetime.now(UTC)
    lanes: list[dict[str, Any]] = []
    try:
        registrations = _register_lanes()
    except Exception as error:
        pytest.fail(f"live registration failed ({type(error).__name__})")
    for lane, concept_id, item_id, catalog in registrations:
        suffix = ".tif" if lane.startswith("Planetary") else ".zip"
        destination = _ROOT / lane.lower().replace(" ", "-") / f"asset{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            bounds = BoundingBox(-180, -90, 180, 90, crs=4326)
            if lane.startswith("Planetary"):
                bounds = BoundingBox(-105, 39, -104, 40, crs=4326)
            items = recorder.operation(
                "search",
                lambda bounds=bounds, catalog=catalog, lane=lane: remote.search(
                    bounds,
                    catalog=catalog,
                    auth_profile=(
                        "earthdata-asf"
                        if lane.startswith("ASF")
                        else "earthdata-lpdaac"
                        if lane.startswith("LP DAAC")
                        else "anonymous"
                    ),
                    limit=1,
                ),
            )
            selected = _select_pinned_item(items, concept_id, item_id)
            if selected is None:
                remote._fail(remote.RemoteAccessError, "identity_mismatch")
            lanes.append(
                _lane_result(lane, item_id, selected, destination, recorder)
            )
        except Exception as error:
            reason = getattr(error, "reason", "")
            status = "INCONCLUSIVE" if reason in {
                "missing_earthdata_credentials",
                "asf_search_unavailable",
                "missing_optional_dependency",
            } else "FAIL"
            lanes.append(
                {
                    "lane": lane,
                    "status": status,
                    "identity": item_id,
                    "cmr_concept_id": concept_id,
                    "error_class": type(error).__name__,
                    "reason": reason or "live_transfer_failed",
                }
            )
    report = {
        "proposal": "PROPOSAL-0047",
        "commit": _COMMIT,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "lanes": lanes,
    }
    _REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    # INCONCLUSIVE is useful in the report for diagnosing unavailable live
    # services, but it cannot satisfy the three-lane acceptance gate.
    assert len(lanes) == 3, report
    assert all(lane.get("status") == "PASS" for lane in lanes), report
    # At least one lane must prove the changed-origin object-delivery path;
    # a report containing only registered gateway origins is insufficient.
    assert any(
        lane.get("final_origin") not in _REGISTERED_TERMINAL_ORIGINS
        for lane in lanes
    ), report
