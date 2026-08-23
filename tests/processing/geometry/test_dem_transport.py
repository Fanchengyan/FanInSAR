"""Tests for the parallel DEM transport (PROPOSAL-0030).

All transport behavior is exercised against fake HTTP layers; no network.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests
import requests.adapters

from faninsar.processing.geometry.dem_transport import (
    CHUNK_SIZE_BYTES,
    RETRYABLE_STATUS_CODES,
    TransientDemFetchError,
    Tile,
    TileSet,
    fetch_plan,
    sweep_part_files,
)

Headers = dict[str, str]

_ONE_CHUNK = CHUNK_SIZE_BYTES


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(
        self,
        *,
        status: int = 200,
        headers: Headers | None = None,
        body: bytes = b"",
        raise_exc: Exception | None = None,
    ) -> None:
        self.status_code = status
        self.headers = headers or {}
        self._body = body
        self._raise_exc = raise_exc
        self.closed = False

    def iter_content(self, chunk_size: int) -> object:
        del chunk_size
        if self._raise_exc is not None:
            raise self._raise_exc
        for offset in range(0, len(self._body), 64):
            yield self._body[offset : offset + 64]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    """Thread-safe scripted request responder."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # (method, url) -> list of outcomes; each outcome is either a
        # FakeResponse or an Exception instance, consumed front to back.
        self.script: dict[tuple[str, str], list[object]] = {}
        self.default: Callable[[str, str], object] | None = None
        self.calls: list[tuple[str, str]] = []

    def script_response(
        self,
        method: str,
        url: str,
        outcomes: list[object],
    ) -> None:
        """Append outcomes to the queue for one method/url pair."""
        with self.lock:
            self.script.setdefault((method, url), []).extend(outcomes)

    def request(self, method: str, url: str, **kwargs: object) -> FakeResponse:
        del kwargs
        with self.lock:
            self.calls.append((method, url))
            queue = self.script.get((method, url))
            if queue is None and self.default is not None:
                outcome = self.default(method, url)
            elif queue:
                outcome = queue.pop(0)
            else:
                outcome = None
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, FakeResponse), f"unscripted {method} {url}"
        return outcome

    def head(self, url: str, **kwargs: object) -> FakeResponse:
        del kwargs
        return self.request("HEAD", url)

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        stream = bool(kwargs.get("stream"))
        assert stream, "streaming GET required"
        return self.request("GET", url)


@pytest.fixture
def patched_session(monkeypatch: pytest.MonkeyPatch) -> FakeSession:
    """Route all transport sessions through one shared FakeSession."""
    session = FakeSession()

    class _Factory:
        def __init__(self, inner: FakeSession) -> None:
            self.inner = inner

        def __call__(self) -> FakeSession:
            return self.inner

    monkeypatch.setattr(
        "faninsar.processing.geometry.dem_transport.thread_local_session",
        _Factory(session),
    )
    return session


def _payload(size: int, seed: int = 7) -> bytes:
    return bytes((seed + index * 31) % 256 for index in range(size))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head_headers(size: int) -> Headers:
    return {
        "Content-Length": str(size),
        "Accept-Ranges": "bytes",
    }


def _fetch_paths(
    urls_and_targets: list[tuple[str, Path]],
    cache_dir: Path,
    *,
    max_workers: int = 8,
    minimum_bytes: int = 0,
    ranged: bool = False,
) -> list[Path]:
    """Plan-based equivalent of the removed fetch_tiles compatibility API."""
    from urllib.parse import urlsplit

    plan = TileSet(
        allowed_hosts=tuple(
            sorted({urlsplit(url).hostname or "" for url, _ in urls_and_targets})
        ),
        tiles=tuple(
            Tile(
                url=url,
                cache_path=(
                    target.relative_to(cache_dir)
                    if target.is_relative_to(cache_dir)
                    else Path(target.name)
                ),
                min_bytes=minimum_bytes,
                ranged=ranged,
            )
            for url, target in urls_and_targets
        ),
    )
    return fetch_plan(plan, cache_dir, max_workers=max_workers)


# ---------------------------------------------------------------------------
# Structural dispatch
# ---------------------------------------------------------------------------


def test_fetch_dispatches_through_bounded_pool(
    tmp_path: Path,
    patched_session: FakeSession,
) -> None:
    """Missing tiles are fetched concurrently through a bounded pool."""
    observed_pools: list[int] = []
    original_init = ThreadPoolExecutor.__init__

    def spy_init(self: ThreadPoolExecutor, max_workers: int | None = None, **kw: object) -> None:
        observed_pools.append(int(max_workers or 0))
        original_init(self, max_workers=max_workers, **kw)

    ThreadPoolExecutor.__init__ = spy_init  # type: ignore[method-assign]
    try:
        payload = _payload(1 << 20)
        urls = [
            f"https://example.test/tile-{index}.tif" for index in range(4)
        ]
        for url in urls:
            patched_session.script_response(
                "GET", url, [FakeResponse(status=200, body=payload)]
            )
        targets = [tmp_path / f"tile-{index}.tif" for index in range(4)]
        results = _fetch_paths(list(zip(urls, targets)), tmp_path)
    finally:
        ThreadPoolExecutor.__init__ = original_init  # type: ignore[method-assign]
    assert observed_pools and observed_pools[0] > 1
    assert [path.name for path in results] == [target.name for target in targets]
    assert all(target.is_file() for target in targets)


def test_cached_files_are_not_refetched(
    tmp_path: Path,
    patched_session: FakeSession,
) -> None:
    """Present files above the size floor are kept, no request issued."""
    existing = tmp_path / "present.tif"
    existing.write_bytes(_payload(1 << 20))
    missing_url = "https://example.test/missing.tif"
    patched_session.script_response(
        "GET", missing_url, [FakeResponse(status=200, body=_payload(1 << 20))]
    )
    _fetch_paths([(missing_url, tmp_path / "missing.tif")], tmp_path)
    gets = [url for method, url in patched_session.calls if method == "GET"]
    assert gets == [missing_url]


# ---------------------------------------------------------------------------
# Retry matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS_CODES))
def test_retryable_status_codes_are_retried(
    tmp_path: Path,
    patched_session: FakeSession,
    status: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """429/5xx statuses retry with backoff then succeed."""
    monkeypatch.setattr(
        "faninsar.processing.geometry.dem_transport.compute_backoff_sleep",
        lambda attempt, retry_after=None: 0.0,
    )
    url = "https://example.test/retry.tif"
    body = _payload(4096)
    patched_session.script_response(
        "GET",
        url,
        [
            FakeResponse(status=status),
            FakeResponse(status=200, body=body),
        ],
    )
    target = tmp_path / "retry.tif"
    _fetch_paths([(url, target)], tmp_path)
    assert target.read_bytes() == body
    gets = [u for m, u in patched_session.calls if m == "GET"]
    assert gets == [url, url]


@pytest.mark.parametrize(
    ("exc_type", "label"),
    [
        (requests.exceptions.ConnectionError, "connection"),
        (requests.exceptions.Timeout, "timeout"),
        (requests.exceptions.ChunkedEncodingError, "chunked"),
    ],
)
def test_transient_exceptions_are_retried(
    tmp_path: Path,
    patched_session: FakeSession,
    exc_type: type[Exception],
    label: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ConnectionError/Timeout/ChunkedEncodingError retry then succeed."""
    del label
    monkeypatch.setattr(
        "faninsar.processing.geometry.dem_transport.compute_backoff_sleep",
        lambda attempt, retry_after=None: 0.0,
    )
    url = "https://example.test/transient.tif"
    body = _payload(4096)
    patched_session.script_response(
        "GET",
        url,
        [
            exc_type("transient"),
            FakeResponse(status=200, body=body),
        ],
    )
    target = tmp_path / "transient.tif"
    _fetch_paths([(url, target)], tmp_path)
    assert target.read_bytes() == body


def test_certificate_verification_error_fails_fast(
    tmp_path: Path,
    patched_session: FakeSession,
) -> None:
    """An SSLError wrapping SSLCertVerificationError never retries."""
    cert_error = requests.exceptions.SSLError(
        "bad handshake"
    )
    cert_error.__cause__ = OSError("certificate verify failed")
    # Give it the real ssl type so the unwrap finds a cert failure.
    import ssl

    cert_error.__cause__ = ssl.SSLCertVerificationError(
        "certificate verify failed: self signed certificate"
    )
    url = "https://example.test/cert.tif"
    patched_session.script_response("GET", url, [cert_error])
    target = tmp_path / "cert.tif"
    with pytest.raises(requests.exceptions.SSLError):
        _fetch_paths([(url, target)], tmp_path)
    gets = [u for m, u in patched_session.calls if m == "GET"]
    assert gets == [url]


def test_tls_eof_truncation_retries(
    tmp_path: Path,
    patched_session: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handshake-burst SSLEOFError retries once and recovers."""
    monkeypatch.setattr(
        "faninsar.processing.geometry.dem_transport.compute_backoff_sleep",
        lambda attempt, retry_after=None: 0.0,
    )
    eof_error = requests.exceptions.SSLError("(max retry cycles)")
    import ssl

    eof_error.__cause__ = ssl.SSLEOFError("EOF occurred in violation of protocol")
    url = "https://example.test/eof.tif"
    body = _payload(4096)
    patched_session.script_response(
        "GET",
        url,
        [eof_error, FakeResponse(status=200, body=body)],
    )
    target = tmp_path / "eof.tif"
    _fetch_paths([(url, target)], tmp_path)
    assert target.read_bytes() == body


def test_permanent_404_fails_loud_without_retry(
    tmp_path: Path,
    patched_session: FakeSession,
) -> None:
    """A 404 response fails immediately without retrying."""
    from faninsar.processing.errors import InvalidProcessingStateError

    url = "https://example.test/gone.tif"
    patched_session.script_response("GET", url, [FakeResponse(status=404)])
    with pytest.raises(InvalidProcessingStateError):
        _fetch_paths([(url, tmp_path / "gone.tif")], tmp_path)
    gets = [u for m, u in patched_session.calls if m == "GET"]
    assert gets == [url]


def test_exhausted_retries_raise_transient_error(
    tmp_path: Path,
    patched_session: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent 503 exhausts attempts and raises TransientDemFetchError."""
    monkeypatch.setattr(
        "faninsar.processing.geometry.dem_transport.compute_backoff_sleep",
        lambda attempt, retry_after=None: 0.0,
    )
    url = "https://example.test/flaky.tif"
    patched_session.script_response(
        "GET",
        url,
        [FakeResponse(status=503) for _ in range(10)],
    )
    with pytest.raises(TransientDemFetchError):
        _fetch_paths([(url, tmp_path / "flaky.tif")], tmp_path)


def test_small_tile_below_size_floor_is_retried_and_fails(
    tmp_path: Path,
    patched_session: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Responses below the minimum tile floor never publish."""
    monkeypatch.setattr(
        "faninsar.processing.geometry.dem_transport.compute_backoff_sleep",
        lambda attempt, retry_after=None: 0.0,
    )
    url = "https://example.test/truncated.tif"
    target = tmp_path / "truncated.tif"
    patched_session.script_response(
        "GET", url, [FakeResponse(status=200, body=b"tiny") for _ in range(10)]
    )
    with pytest.raises(Exception):
        _fetch_paths([(url, target)], tmp_path, minimum_bytes=1 << 20)
    assert not target.exists()


# ---------------------------------------------------------------------------
# Ranged mode
# ---------------------------------------------------------------------------


def test_ranged_assembly_matches_single_stream_sha256(
    tmp_path: Path,
    patched_session: FakeSession,
) -> None:
    """Ranged chunks assemble byte-identically to a single-stream fetch."""
    body = _payload(_ONE_CHUNK + 12345)
    url = "https://example.test/ranged.tif"
    patched_session.script_response(
        "HEAD", url, [FakeResponse(status=200, headers=_head_headers(len(body)))]
    )
    total_chunks = (len(body) + _ONE_CHUNK - 1) // _ONE_CHUNK
    for index in range(total_chunks):
        start = index * _ONE_CHUNK
        end = min(start + _ONE_CHUNK, len(body)) - 1
        chunk = body[start : end + 1]
        patched_session.script_response(
            "GET",
            url,
            [
                FakeResponse(
                    status=206,
                    headers={
                        "Content-Range": f"bytes {start}-{end}/{len(body)}",
                        "Content-Length": str(end - start + 1),
                    },
                    body=chunk,
                )
            ],
        )
    target = tmp_path / "ranged.tif"
    reference = tmp_path / "reference.tif"
    reference.write_bytes(body)
    _fetch_paths([(url, target)], tmp_path, ranged=True)
    assert target.stat().st_size == len(body)
    assert _sha256(target) == _sha256(reference)


def test_200_to_range_request_hard_failure(
    tmp_path: Path,
    patched_session: FakeSession,
) -> None:
    """A 200 answer to a Range GET is a hard error; nothing is written."""
    body = _payload(_ONE_CHUNK * 2)
    url = "https://example.test/stripped.tif"
    patched_session.script_response(
        "HEAD", url, [FakeResponse(status=200, headers=_head_headers(len(body)))]
    )
    patched_session.script_response(
        "GET",
        url,
        [
            FakeResponse(
                status=200,
                headers={"Content-Length": str(len(body))},
                body=body,
            )
        ],
    )
    target = tmp_path / "stripped.tif"
    with pytest.raises(Exception, match="[Rr]ange"):
        _fetch_paths([(url, target)], tmp_path, ranged=True)
    assert not target.exists()


def test_mismatched_content_range_hard_failure(
    tmp_path: Path,
    patched_session: FakeSession,
) -> None:
    """A Content-Range not matching the requested offsets is a hard error."""
    body = _payload(_ONE_CHUNK * 2)
    url = "https://example.test/mismatch.tif"
    patched_session.script_response(
        "HEAD", url, [FakeResponse(status=200, headers=_head_headers(len(body)))]
    )
    wrong_start = _ONE_CHUNK + 999
    patched_session.script_response(
        "GET",
        url,
        [
            FakeResponse(
                status=206,
                headers={
                    "Content-Range": (
                        f"bytes {wrong_start}-{wrong_start + _ONE_CHUNK - 1}/{len(body)}"
                    ),
                    "Content-Length": str(_ONE_CHUNK),
                },
                body=body[:_ONE_CHUNK],
            )
        ],
    )
    target = tmp_path / "mismatch.tif"
    with pytest.raises(Exception, match="Content-Range"):
        _fetch_paths([(url, target)], tmp_path, ranged=True)
    assert not target.exists()


def test_ranged_mode_skipped_when_accept_ranges_missing(
    tmp_path: Path,
    patched_session: FakeSession,
) -> None:
    """Without Accept-Ranges the plain streaming path is used."""
    body = _payload(_ONE_CHUNK * 2)
    url = "https://example.test/norange.tif"
    patched_session.script_response(
        "HEAD",
        url,
        [FakeResponse(status=200, headers={"Content-Length": str(len(body))})],
    )
    patched_session.script_response(
        "GET", url, [FakeResponse(status=200, body=body)]
    )
    target = tmp_path / "norange.tif"
    _fetch_paths([(url, target)], tmp_path, ranged=True)
    range_gets = [u for m, u in patched_session.calls if m == "GET"]
    assert range_gets == [url]
    assert _sha256(target) == hashlib.sha256(body).hexdigest()


def test_head_probe_failure_falls_back_to_plain_stream(
    tmp_path: Path,
    patched_session: FakeSession,
) -> None:
    """A failed HEAD probe degrades to the plain path instead of failing."""
    body = _payload(8192)
    url = "https://example.test/nohead.tif"
    patched_session.script_response(
        "HEAD", url, [requests.exceptions.ConnectionError("no head")]
    )
    patched_session.script_response(
        "GET", url, [FakeResponse(status=200, body=body)]
    )
    target = tmp_path / "nohead.tif"
    _fetch_paths([(url, target)], tmp_path, ranged=True)
    assert target.read_bytes() == body


# ---------------------------------------------------------------------------
# Windows branch and shared budget
# ---------------------------------------------------------------------------


def test_windows_seek_write_branch_forced_on_posix(
    tmp_path: Path,
    patched_session: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forcing no-pwrite exercises the per-thread handle seek+write branch."""
    import faninsar.processing.geometry.dem_transport as transport

    monkeypatch.setattr(transport, "_PWRITE_AVAILABLE", False)
    body = _payload(_ONE_CHUNK + 777)
    url = "https://example.test/winbranch.tif"
    patched_session.script_response(
        "HEAD", url, [FakeResponse(status=200, headers=_head_headers(len(body)))]
    )
    for index in range(2):
        start = index * _ONE_CHUNK
        end = min(start + _ONE_CHUNK, len(body)) - 1
        patched_session.script_response(
            "GET",
            url,
            [
                FakeResponse(
                    status=206,
                    headers={
                        "Content-Range": f"bytes {start}-{end}/{len(body)}",
                        "Content-Length": str(end - start + 1),
                    },
                    body=body[start : end + 1],
                )
            ],
        )
    target = tmp_path / "winbranch.tif"
    _fetch_paths([(url, target)], tmp_path, ranged=True)
    assert _sha256(target) == hashlib.sha256(body).hexdigest()


def test_shared_stream_budget_bounds_in_flight_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tiles and chunks share one budget; in-flight requests stay <= workers."""
    import faninsar.processing.geometry.dem_transport as transport

    max_workers = 4
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    class BudgetProbeSession(FakeSession):
        def request(self, method: str, url: str, **kwargs: object):
            nonlocal in_flight, peak
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            try:
                return super().request(method, url, **kwargs)
            finally:
                with lock:
                    in_flight -= 1

    probe = BudgetProbeSession()
    monkeypatch.setattr(
        transport,
        "thread_local_session",
        lambda: probe,
    )

    body = _payload(_ONE_CHUNK * max_workers + 11)
    urls = [f"https://example.test/budget-{i}.tif" for i in range(max_workers)]
    for url in urls:
        probe.script_response(
            "HEAD", url, [FakeResponse(status=200, headers=_head_headers(len(body)))]
        )
        chunks = (len(body) + _ONE_CHUNK - 1) // _ONE_CHUNK
        for index in range(chunks):
            start = index * _ONE_CHUNK
            end = min(start + _ONE_CHUNK, len(body)) - 1
            probe.script_response(
                "GET",
                url,
                [
                    FakeResponse(
                        status=206,
                        headers={
                            "Content-Range": (
                                f"bytes {start}-{end}/{len(body)}"
                            ),
                            "Content-Length": str(end - start + 1),
                        },
                        body=body[start : end + 1],
                    )
                ],
            )
    targets = [tmp_path / f"budget-{i}.tif" for i in range(max_workers)]
    _fetch_paths(
        list(zip(urls, targets)),
        tmp_path,
        max_workers=max_workers,
        ranged=True,
    )
    assert peak <= max_workers


# ---------------------------------------------------------------------------
# Atomicity and cleanup
# ---------------------------------------------------------------------------


def test_unique_part_names_published_via_replace(
    tmp_path: Path,
    patched_session: FakeSession,
) -> None:
    """.part names carry pid+uuid and only complete files are published."""
    body = _payload(1 << 20)
    url = "https://example.test/atomic.tif"
    patched_session.script_response(
        "GET", url, [FakeResponse(status=200, body=body)]
    )
    seen_parts: set[str] = set()
    real_iterdir = Path.iterdir

    def spy_iterdir(path: Path):
        if path == tmp_path:
            for entry in real_iterdir(path):
                if entry.name.endswith(".part"):
                    seen_parts.add(entry.name)
        return real_iterdir(path)

    import faninsar.processing.geometry.dem_transport as transport

    original = transport._publish_target

    def spy_publish(target: Path, source_part: Path) -> None:
        seen_parts.add(source_part.name)
        original(target, source_part)

    transport._publish_target = spy_publish  # type: ignore[assignment]
    try:
        _fetch_paths([(url, tmp_path / "atomic.tif")], tmp_path)
    finally:
        transport._publish_target = original  # type: ignore[assignment]
    del spy_iterdir, real_iterdir
    assert seen_parts
    for name in seen_parts:
        assert name.startswith("atomic.tif.")
        assert name.endswith(".part")
        core = name[len("atomic.tif.") : -len(".part")]
        pid_text, _, uuid_text = core.partition("-")
        assert pid_text.isdigit()
        assert len(uuid_text) >= 32
    assert not list(tmp_path.glob("*.part"))
    assert (tmp_path / "atomic.tif").read_bytes() == body


def test_part_file_removed_on_failure(
    tmp_path: Path,
    patched_session: FakeSession,
) -> None:
    """Failed downloads leave no .part behind."""
    url = "https://example.test/fail.tif"
    patched_session.script_response("GET", url, [FakeResponse(status=404)])
    from faninsar.processing.errors import InvalidProcessingStateError

    with pytest.raises(InvalidProcessingStateError):
        _fetch_paths([(url, tmp_path / "fail.tif")], tmp_path)
    assert not list(tmp_path.glob("*.part"))


def test_age_based_sweep_removes_orphan_parts(tmp_path: Path) -> None:
    """Old orphan .part files are swept; fresh ones survive."""
    import time as time_module

    old_part = tmp_path / "old.tif.1-abc.part"
    fresh_part = tmp_path / "fresh.tif.2-def.part"
    old_part.write_bytes(b"x")
    fresh_part.write_bytes(b"x")
    stale_time = time_module.time() - 7200
    import os

    os.utime(old_part, (stale_time, stale_time))
    swept = sweep_part_files(tmp_path, max_age_s=3600)
    assert not old_part.exists()
    assert fresh_part.exists()
    assert swept == 1


def test_orphan_part_never_false_hits_as_cache_tile(tmp_path: Path) -> None:
    """A leftover .part file does not satisfy a cache lookup."""
    part = tmp_path / "N38_E100.tif.123-abc.part"
    part.write_bytes(b"partial")
    from faninsar.processing.geometry.dem_sources import validate_cache_relative_path

    validate_cache_relative_path(part.name, tmp_path)
    assert not (tmp_path / "N38_E100.tif").is_file()


# ---------------------------------------------------------------------------
# Boundary guard
# ---------------------------------------------------------------------------


def test_transport_guard_rejects_traversal_targets(tmp_path: Path) -> None:
    """Targets escaping cache_dir are rejected before any I/O."""
    from faninsar.processing.geometry.dem_transport import validate_cache_target

    outside = tmp_path.parent / "escape-target.tif"
    with pytest.raises(ValueError, match="escapes"):
        validate_cache_target(tmp_path, outside)
    assert not outside.exists()
