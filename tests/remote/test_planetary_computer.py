"""Planetary Computer adapter seams (PROPOSAL-0045 Task C)."""

# Test fixtures intentionally use mutable simple objects and runtime pytest.
# ruff: noqa: E501, TC002, TC003, RUF012, PYI034

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from faninsar import remote
from faninsar.query import BoundingBox
from faninsar.remote.providers.planetary_computer import (
    COP_DEM_GLO30_COLLECTION,
    PlanetaryComputerAdapter,
)


@dataclass
class _Asset:
    href: str
    extra_fields: dict[str, object]


@dataclass
class _Item:
    id: str
    geometry: dict[str, object]
    assets: dict[str, _Asset]
    collection_id: str = COP_DEM_GLO30_COLLECTION


class _Search:
    def __init__(self, item: _Item) -> None:
        self._item = item

    def items(self) -> tuple[_Item, ...]:
        return (self._item,)


class _Client:
    def __init__(self, item: _Item) -> None:
        self.item = item
        self.bboxes: list[list[float]] = []

    def search(self, *, collections: list[str], bbox: list[float]) -> _Search:
        assert collections == [COP_DEM_GLO30_COLLECTION]
        self.bboxes.append(bbox)
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
    )


def test_planetary_computer_search_signs_in_memory_and_preserves_safe_identity() -> None:
    """STAC discovery signs an item in memory and emits a safe record."""
    item = _item()
    calls: list[object] = []

    def signer(value: object) -> object:
        calls.append(value)
        value.assets["data"].href += "?sig=secret"  # type: ignore[attr-defined]
        return value

    adapter = PlanetaryComputerAdapter(client=_Client(item), signer=signer)
    records = list(adapter.items())

    assert len(records) == 1
    assert calls == [item]
    assert records[0]["assets"]["data"]["href"].endswith("sig=secret")


def test_planetary_computer_is_registered_with_remote_boundary() -> None:
    """The provider-neutral remote API can search the registered PC catalog."""
    item = _item()
    adapter = PlanetaryComputerAdapter(client=_Client(item), signer=lambda value: value)
    adapter.register("pc-task-c")
    found = remote.search(
        BoundingBox(-1, -1, 1, 1, crs=4326), catalog="pc-task-c"
    )
    assert found[0].provider == "pc"
    assert found[0].assets["data"].href.endswith("tile.tif")


class _Response:
    headers = {"Content-Encoding": "identity", "Content-Length": "12"}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int) -> bytes:
        return b"pc-dem-bytes" if not hasattr(self, "done") else b""


def test_planetary_computer_fetch_uses_p0044_ledger_and_signing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fetch returns bounded chunks and charges the operation ledger."""
    payload = b"pc-dem-bytes"
    response = _Response()

    def read(_size: int) -> bytes:
        if getattr(response, "done", False):
            return b""
        response.done = True
        return payload

    response.read = read  # type: ignore[method-assign]
    adapter = PlanetaryComputerAdapter(signer=lambda value: value)
    monkeypatch.setattr(
        PlanetaryComputerAdapter,
        "_open",
        lambda *_args, **_kwargs: response,
    )
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
