"""Complete-file streaming and publication tests."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pytest

from faninsar import remote
from faninsar.query import Points

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


class _ChunkAdapter:
    """Fixture adapter exposing a one-pass chunk stream."""

    provider = "chunks"
    origins = ("https://chunks.invalid",)
    path_prefixes = ("/",)
    redirect_origins = ()
    profiles = ("anonymous",)

    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self.chunks = chunks
        self.consumed = False

    def items(self) -> Iterable[Mapping[str, Any]]:
        payload = b"".join(self.chunks)
        return (
            {
                "id": "chunk-item",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                },
                "assets": {
                    "data": {
                        "href": "https://chunks.invalid/data.bin",
                        "size": len(payload),
                        "checksum": f"sha256:{hashlib.sha256(payload).hexdigest()}",
                    }
                },
            },
        )

    def fetch(
        self, asset: remote.RemoteAsset, budget: remote.RemoteResourceBudget
    ) -> Iterable[bytes]:
        del asset, budget
        assert not self.consumed
        self.consumed = True
        return iter(self.chunks)


def _asset(adapter_name: str) -> remote.RemoteAsset:
    return remote.search(Points([(0.5, 0.5)], crs=4326), catalog=adapter_name)[
        0
    ].assets["data"]


def test_complete_download_streams_chunks_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    """A chunk stream is written, checked, and atomically published."""
    adapter = _ChunkAdapter((b"first-", b"second-", b"third"))
    remote._register_adapter("chunks", adapter)
    destination = tmp_path / "nested" / "data.bin"

    path = remote.download(_asset("chunks"), destination)

    assert path == destination
    assert destination.read_bytes() == b"first-second-third"
    assert not list(destination.parent.glob(f".{destination.name}.*"))
    assert adapter.consumed


def test_stream_length_failure_leaves_destination_unchanged(tmp_path: Path) -> None:
    """Length validation happens before replacing an existing destination."""
    adapter = _ChunkAdapter((b"short",))
    remote._register_adapter("short", adapter)
    asset = _asset("short")
    destination = tmp_path / "data.bin"
    destination.write_bytes(b"old content")
    object.__setattr__(asset, "size_bytes", 99)

    with pytest.raises(remote.RemoteIntegrityError, match="content_length_mismatch"):
        remote.download(asset, destination, overwrite=True)

    assert destination.read_bytes() == b"old content"
    assert not list(tmp_path.glob(f".{destination.name}.*"))


def test_parallel_parts_assemble_ranges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """HTTP range workers reassemble a complete asset in source order."""
    payload = b"0123456789" * 3

    class _HTTPAdapter:
        provider = "http"
        origins = ("https://http.invalid",)
        path_prefixes = ("/",)
        redirect_origins = ()
        profiles = ("anonymous",)

        def items(self) -> Iterable[Mapping[str, Any]]:
            return ()

    adapter = _HTTPAdapter()
    asset = remote.RemoteAsset(
        "http",
        "http",
        None,
        "item",
        "data",
        "https://http.invalid/data",
        size_bytes=len(payload),
    )

    class _Response:
        headers: dict[str, str] = {}

        def __init__(self, data: bytes) -> None:
            self.data = data

        def read(self, size: int = -1) -> bytes:
            return self.data[:size]

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class _Opener:
        @contextmanager
        def open(self, request: Any, timeout: float) -> Iterable[_Response]:
            del timeout
            start, end = (
                int(value) for value in request.headers["Range"][6:].split("-")
            )
            yield _Response(payload[start : end + 1])

    monkeypatch.setattr(remote.urllib.request, "build_opener", lambda *args: _Opener())
    staging = tmp_path / "staging.bin"
    ledger = remote._CallLedger(remote.RemoteResourceBudget(max_workers=3))
    remote._stream_download_parts(asset, adapter, ledger.budget, ledger, staging, 7)
    assert staging.read_bytes() == payload


def test_download_many_parallel_files(tmp_path: Path) -> None:
    """The multi-file convenience API preserves input ordering."""
    adapter = _ChunkAdapter((b"a",))
    records = [
        {
            "id": f"item-{i}",
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "assets": {
                "data": {
                    "href": "https://chunks.invalid/data",
                    "data": b"a",
                }
            },
        }
        for i in range(2)
    ]
    adapter = remote._FixtureAdapter(records)
    remote._register_adapter("many", adapter)
    assets = [
        remote.RemoteAsset(
            "fixture",
            "many",
            None,
            f"item-{i}",
            "data",
            "https://fixture.invalid/data",
            version="1",
        )
        for i in range(2)
    ]
    paths = remote.download_many(
        assets, [tmp_path / "a", tmp_path / "b"], parallel_files=True
    )
    assert [path.read_bytes() for path in paths] == [b"a", b"a"]
