"""Planetary Computer adapter seams (PROPOSAL-0045 Task C)."""

# Test fixtures intentionally use mutable simple objects and runtime pytest.
# ruff: noqa: TC003, RUF012, PYI034

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from faninsar import remote
from faninsar.query import BoundingBox
from faninsar.remote.providers.planetary_computer import (
    COP_DEM_GLO30_COLLECTION,
    PlanetaryComputerAdapter,
    _instrument_session,
)
from faninsar.remote.standards import MalformedSTACItemError


@dataclass
class _Asset:
    href: str
    extra_fields: dict[str, object]


@dataclass
class _Item:
    id: str
    geometry: dict[str, object]
    assets: dict[str, _Asset]
    properties: dict[str, object]
    collection_id: str = COP_DEM_GLO30_COLLECTION
    stac_version: str = "1.1.0"
    stac_extensions: list[str] = field(default_factory=list)


class _Search:
    def __init__(self, item: _Item) -> None:
        self._item = item

    def items(self) -> tuple[_Item, ...]:
        return (self._item,)


class _Client:
    _faninsar_offline = True

    def __init__(self, item: _Item) -> None:
        self.item = item
        self.bboxes: list[list[float]] = []
        self.search_kwargs: dict[str, object] = {}

    def search(
        self, *, collections: list[str], bbox: list[float], **_kwargs: object
    ) -> _Search:
        assert collections == [COP_DEM_GLO30_COLLECTION]
        self.bboxes.append(bbox)
        self.search_kwargs = {"collections": collections, "bbox": bbox, **_kwargs}
        return _Search(self.item)


def _item() -> _Item:
    return _Item(
        id="tile-1",
        geometry={
            "type": "Polygon",
            "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, -1]]],
        },
        assets={
            "data": _Asset(
                "https://elevationeuwest.blob.core.windows.net/tile.tif",
                {"file:size": 4},
            )
        },
        properties={"datetime": "2024-01-02T03:04:05Z"},
        stac_extensions=[],
    )


def test_planetary_computer_search_stays_unsigned_and_preserves_safe_identity() -> (
    None
):
    """STAC discovery emits a safe unsigned record for later transfer signing."""
    item = _item()
    calls: list[object] = []

    def signer(value: object) -> object:
        calls.append(value)
        value.assets["data"].href += "?sig=secret"  # type: ignore[attr-defined]
        return value

    adapter = PlanetaryComputerAdapter(client=_Client(item), signer=signer)
    records = list(adapter.items())

    assert len(records) == 1
    assert calls == []
    assert records[0]["assets"]["data"]["href"].endswith("tile.tif")
    assert not hasattr(adapter, "_signed")


def test_planetary_computer_is_registered_with_remote_boundary() -> None:
    """The provider-neutral remote API can search the registered PC catalog."""
    item = _item()
    adapter = PlanetaryComputerAdapter(client=_Client(item), signer=lambda value: value)
    adapter.register("pc-task-c")
    found = remote.search(BoundingBox(-1, -1, 1, 1, crs=4326), catalog="pc-task-c")
    assert found[0].provider == "pc"
    assert found[0].assets["data"].href.endswith("tile.tif")


def test_planetary_computer_propagates_query_budget_to_stac_client() -> None:
    """The typed query limit and datetime interval reach STAC discovery."""
    client = _Client(_item())
    adapter = PlanetaryComputerAdapter(client=client, signer=lambda value: value)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 3, tzinfo=UTC)

    list(adapter.items(limit=3, datetime_range=(start, end)))

    assert client.search_kwargs["max_items"] == 3
    assert client.search_kwargs["datetime"] == (
        "2024-01-01T00:00:00+00:00/2024-01-03T00:00:00+00:00"
    )


class _StacResponse:
    headers = {"Content-Encoding": "identity"}

    def __init__(self, content: bytes, history: tuple[_StacResponse, ...] = ()) -> None:
        self.content = content
        self.history = history


class _StacSession:
    def __init__(self, response: _StacResponse) -> None:
        self.response = response

    def send(self, _request: object, **_kwargs: object) -> _StacResponse:
        return self.response


class _RetryStacSession(_StacSession):
    def __init__(self, response: _StacResponse) -> None:
        super().__init__(response)
        self.calls = 0

    def send(self, request: object, **kwargs: object) -> _StacResponse:
        self.calls += 1
        if self.calls == 1:
            message = "transient"
            raise OSError(message)
        return super().send(request, **kwargs)


def test_planetary_computer_stac_transport_meters_response_history() -> None:
    """Discovery transport charges responses and returned redirect history."""
    redirected = _StacResponse(b"redirect")
    response = _StacResponse(b"final", history=(redirected,))
    session = _StacSession(response)
    budget = remote.RemoteResourceBudget()
    ledger = remote._CallLedger(budget)
    adapter = PlanetaryComputerAdapter(signer=lambda value: value)

    restore = _instrument_session(session, ledger, adapter)
    try:
        request = type(
            "Request",
            (),
            {"url": "https://planetarycomputer.microsoft.com/api/stac/v1"},
        )()
        session.send(request)
    finally:
        restore()

    assert ledger.requests == 2
    assert ledger.redirects == 1
    assert ledger.response_bytes_total == len(b"redirectfinal")


def test_planetary_computer_stac_transport_charges_retries() -> None:
    """Transient transport failures consume the operation retry budget."""
    session = _RetryStacSession(_StacResponse(b"final"))
    ledger = remote._CallLedger(remote.RemoteResourceBudget(max_retries=1))
    adapter = PlanetaryComputerAdapter(signer=lambda value: value)

    restore = _instrument_session(session, ledger, adapter)
    try:
        request = type(
            "Request",
            (),
            {"url": "https://planetarycomputer.microsoft.com/api/stac/v1"},
        )()
        session.send(request)
    finally:
        restore()

    assert session.calls == 2
    assert ledger.requests == 2
    assert ledger.retries == 1


class _Response:
    headers = {"Content-Encoding": "identity", "Content-Length": "12"}
    status = 200

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int) -> bytes:
        return b"pc-dem-bytes" if not hasattr(self, "done") else b""


def test_planetary_computer_download_signs_inside_the_mediated_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A download resolves a fresh signed URL without adapter retention."""
    payload = b"II*\x00pc-dem-bytes"
    response = _Response()

    def read(_size: int) -> bytes:
        if getattr(response, "done", False):
            return b""
        response.done = True
        return payload

    response.read = read  # type: ignore[method-assign]
    adapter = PlanetaryComputerAdapter(signer=lambda value: value)
    class _Opener:
        def open(self, *_args: object, **_kwargs: object) -> _Response:
            return response

    monkeypatch.setattr(remote.urllib.request, "build_opener", lambda *_args: _Opener())
    asset = remote.RemoteAsset(
        provider="pc",
        catalog="pc-task-c-fetch",
        collection=COP_DEM_GLO30_COLLECTION,
        item_id="tile-1",
        key="data",
        href="https://elevationeuwest.blob.core.windows.net/tile.tif",
        checksum=f"sha256:{hashlib.sha256(payload).hexdigest()}",
    )
    adapter.register("pc-task-c-fetch")
    destination = remote.download(asset, tmp_path / "dem.tif")
    assert destination.read_bytes() == payload
    assert not hasattr(adapter, "_signed")


def test_planetary_computer_default_signer_uses_real_sdk_mapping_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The installed SDK signer accepts the operation-local STAC mapping."""
    import planetary_computer

    calls: list[str] = []

    def sign_url(url: str) -> str:
        calls.append(url)
        return f"{url}?sig=fixture"

    monkeypatch.setattr(planetary_computer.sas, "sign_url", sign_url)
    adapter = PlanetaryComputerAdapter()
    asset = remote.RemoteAsset(
        provider="pc",
        catalog="pc-sdk-shape",
        collection=COP_DEM_GLO30_COLLECTION,
        item_id="tile-1",
        key="data",
        href="https://elevationeuwest.blob.core.windows.net/tile.tif",
    )
    ledger = remote._CallLedger(remote.RemoteResourceBudget())

    signed_href = adapter._transfer_url(asset, budget=ledger.budget, ledger=ledger)

    assert signed_href.endswith("tile.tif?sig=fixture")
    assert calls == [asset.href]
    assert asset.href == "https://elevationeuwest.blob.core.windows.net/tile.tif"
    assert not hasattr(adapter, "_signed")


def test_planetary_computer_rejects_unobservable_injected_client() -> None:
    """A client without an inspectable transport fails before network I/O."""
    item = _item()
    adapter = PlanetaryComputerAdapter(client=_Client(item), signer=lambda value: value)
    adapter.client._faninsar_offline = False  # type: ignore[attr-defined]
    with pytest.raises(remote.RemoteAccessError) as error:
        list(adapter.items())
    assert error.value.reason == "unobservable_discovery"


def test_planetary_computer_normalizes_and_types_malformed_stac_items() -> None:
    """Required STAC identity and asset fields cannot be silently skipped."""
    item = _item()
    item.id = ""
    adapter = PlanetaryComputerAdapter(client=_Client(item), signer=lambda value: value)
    with pytest.raises(MalformedSTACItemError) as error:
        list(adapter.items())
    assert error.value.reason == "invalid_item_id"
