"""PROPOSAL-0030 S1 v2 transport tests: FetchPlan grammar, host pinning,
credential hygiene, redirects, Earthdata ranged exclusion, FTP, and zip.
"""

from __future__ import annotations

import io
import logging
import threading
import zipfile
from pathlib import Path
from urllib.error import URLError

import pytest
import requests

from faninsar.processing.geometry import dem_transport as transport
from faninsar.processing.geometry.dem_transport import (
    Artifact,
    CredentialProvider,
    Tile,
    TileSet,
    fetch_plan,
    resolve_credentials,
    validate_plan_urls,
)

# ---------------------------------------------------------------------------
# Fake HTTP layer shared with the v1 scaffold conventions
# ---------------------------------------------------------------------------


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(
        self,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
        raise_exc: Exception | None = None,
        history: tuple[object, ...] = (),
    ) -> None:
        self.status_code = status
        self.headers = headers or {}
        self._body = body
        self._raise_exc = raise_exc
        self.history = history
        self.closed = False

    def iter_content(self, chunk_size: int) -> object:
        del chunk_size
        if self._raise_exc is not None:
            raise self._raise_exc
        yield self._body

    def close(self) -> None:
        self.closed = True


class RecordingSession:
    """Scripted responder recording auth headers per request."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.script: dict[tuple[str, str], list[object]] = {}
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    def script_response(
        self,
        method: str,
        url: str,
        outcomes: list[object],
    ) -> None:
        with self.lock:
            self.script.setdefault((method, url), []).extend(outcomes)

    def _next(self, method: str, url: str) -> object:
        with self.lock:
            queue = self.script.get((method, url))
            outcome = queue.pop(0) if queue else None
            self.calls.append((method, url, {}))
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, FakeResponse), f"unscripted {method} {url}"
        return outcome

    def head(self, url: str, **kwargs: object) -> FakeResponse:
        return self.request("HEAD", url, **kwargs)

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        return self.request("GET", stream=True, **kwargs)

    def request(self, method: str, url: str, **kwargs: object) -> FakeResponse:
        headers = {k.lower(): str(v) for k, v in kwargs.get("headers", {}).items()}
        with self.lock:
            self.calls.append((method, url, headers))
            queue = self.script.get((method, url))
            outcome = queue.pop(0) if queue else None
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, FakeResponse), f"unscripted {method} {url}"
        return outcome


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> RecordingSession:
    instance = RecordingSession()
    monkeypatch.setattr(
        "faninsar.processing.geometry.dem_transport.thread_local_session",
        lambda: instance,
    )
    return instance


def _payload(size: int, seed: int = 11) -> bytes:
    return bytes((seed + index * 7) % 251 for index in range(size))


# ---------------------------------------------------------------------------
# 1. FetchPlan union grammar
# ---------------------------------------------------------------------------


class TestFetchPlanGrammar:
    def test_tile_set_holds_allowed_hosts_and_tiles(self) -> None:
        plan = TileSet(
            allowed_hosts=("copernicus-dem-30m.s3.amazonaws.com",),
            tiles=(
                Tile(
                    url="https://copernicus-dem-30m.s3.amazonaws.com/a/b.tif",
                    cache_path=Path("a/b.tif"),
                    min_bytes=1024,
                ),
            ),
        )
        assert plan.allowed_hosts == ("copernicus-dem-30m.s3.amazonaws.com",)
        assert plan.tiles[0].min_bytes == 1024

    def test_artifact_holds_members_and_credential_ref(self) -> None:
        plan = Artifact(
            allowed_hosts=("data.lpdaac.earthdatacloud.nasa.gov",),
            scheme="https",
            url="https://data.lpdaac.earthdatacloud.nasa.gov/g.zip",
            members=("ALPSMLC30_N000E006_DSM.tif",),
            member_pattern=None,
            expand="zip",
            credential_ref="earthdata",
        )
        assert plan.expand == "zip"
        assert plan.credential_ref == "earthdata"

    def test_unknown_fields_rejected(self) -> None:
        with pytest.raises(TypeError):
            TileSet(allowed_hosts=("x.test",), bogus_field=1)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            Artifact(
                allowed_hosts=("x.test",),
                scheme="https",
                url="https://x.test/a.zip",
                unknown=True,  # type: ignore[call-arg]
            )

    def test_artifact_rejects_bad_scheme_and_conflicting_members(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            Artifact(
                allowed_hosts=("x.test",),
                scheme="gopher",  # type: ignore[arg-type]
                url="gopher://x.test/a",
            )
        with pytest.raises(ValueError, match="member"):
            Artifact(
                allowed_hosts=("x.test",),
                scheme="https",
                url="https://x.test/a.zip",
                members=("a.tif",),
                member_pattern="b*",
            )


# ---------------------------------------------------------------------------
# 2. URL host pinning
# ---------------------------------------------------------------------------


class TestHostPinning:
    def test_url_host_outside_allowlist_rejected_before_connect(
        self,
        tmp_path: Path,
    ) -> None:
        plan = TileSet(
            allowed_hosts=("good.example.test",),
            tiles=(
                Tile(
                    url="https://evil.example.test/steal.tif",
                    cache_path=Path("steal.tif"),
                    min_bytes=1,
                ),
            ),
        )
        with pytest.raises(ValueError, match="allowlist"):
            validate_plan_urls(plan)
        with pytest.raises(ValueError, match="allowlist"):
            fetch_plan(plan, tmp_path)

    def test_non_https_url_rejected(self, tmp_path: Path) -> None:
        plan = TileSet(
            allowed_hosts=("good.example.test",),
            tiles=(
                Tile(
                    url="http://good.example.test/plain.tif",
                    cache_path=Path("plain.tif"),
                    min_bytes=1,
                ),
            ),
        )
        with pytest.raises(ValueError, match="https"):
            validate_plan_urls(plan)

    def test_ftp_artifact_binds_only_to_declared_url(self, tmp_path: Path) -> None:
        artifact = Artifact(
            allowed_hosts=("ftp.eorc.jaxa.jp",),
            scheme="ftp",
            url="ftp://ftp.eorc.jaxa.jp/pub/AW3D30/x.zip",
            expand="zip",
        )
        validate_plan_urls(artifact)
        bad = Artifact(
            allowed_hosts=("other.example.test",),
            scheme="ftp",
            url="ftp://ftp.eorc.jaxa.jp/pub/AW3D30/x.zip",
        )
        with pytest.raises(ValueError, match="allowlist"):
            validate_plan_urls(bad)


# ---------------------------------------------------------------------------
# 3. Redirects and credentials
# ---------------------------------------------------------------------------

_TOKEN = "secret-earthdata-token-value-123"


class _NetrcProvider(CredentialProvider):
    """Credential provider yielding a fixed bearer token."""

    def __init__(self, token: str) -> None:
        self.token = token

    def headers_for(self, url: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


class TestCredentialsAndRedirects:
    def test_resolve_earthdata_netrc_then_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        del tmp_path
        monkeypatch.setenv("EARTHDATA_TOKEN", _TOKEN)
        creds = resolve_credentials("earthdata")
        assert creds is not None
        assert "Bearer " + _TOKEN in str(creds.headers_for("https://x.test"))

    def test_missing_earthdata_credentials_fail_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
        monkeypatch.setattr(transport, "_netrc_auth", lambda _host: None)
        with pytest.raises(RuntimeError, match="[Nn]etrc|EARTHDATA_TOKEN"):
            resolve_credentials("earthdata")

    def test_cross_host_redirect_followed_without_credentials(
        self,
        tmp_path: Path,
        session: RecordingSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(transport, "_netrc_auth", lambda _host: ("user", _TOKEN))
        body = b"ok-bytes"
        start = "https://auth.example.test/granule.tif"
        redirect = "https://data.example.test/granule.tif"
        session.script_response(
            "GET",
            start,
            [
                FakeResponse(
                    status=302,
                    headers={"Location": redirect},
                )
            ],
        )
        session.script_response(
            "GET", redirect, [FakeResponse(status=200, body=body)]
        )

        class _Creds(CredentialProvider):
            def headers_for(self, url: str) -> dict[str, str]:
                if url.startswith(start):
                    return {"Authorization": f"Bearer {_TOKEN}"}
                return {}

        plan = Artifact(
            allowed_hosts=("auth.example.test",),
            scheme="https",
            url=start,
            credential_ref=None,
            min_total_bytes=1,
            _credentials=_Creds(),
        )
        fetch_plan(plan, tmp_path)
        target = tmp_path / "granule.tif"
        assert target.read_bytes() == body
        redirected_calls = [c for c in session.calls if c[1] == redirect]
        assert redirected_calls
        for _, _, headers in redirected_calls:
            assert not any(
                k.startswith("authorization") or k == "cookie" for k in headers
            )
        start_calls = [c for c in session.calls if c[1] == start]
        assert any("authorization" in h for _, _, h in start_calls)

    def test_cross_host_redirect_with_auth_is_hard_error(
        self,
        tmp_path: Path,
        session: RecordingSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(transport, "_is_earthdata_head_safe", lambda _u: False)
        start = "https://auth.example.test/granule.tif"
        redirect = "https://evil.example.test/granule.tif"

        class _Creds(CredentialProvider):
            def headers_for(self, url: str) -> dict[str, str]:
                return {"Authorization": "Bearer x"}

        session.script_response(
            "GET", start, [FakeResponse(status=302, headers={"Location": redirect})]
        )
        plan = Artifact(
            allowed_hosts=("auth.example.test", "evil.example.test"),
            scheme="https",
            url=start,
            min_total_bytes=1,
            _credentials=_Creds(),
        )
        with pytest.raises(ValueError, match="redirect|credential"):
            fetch_plan(plan, tmp_path)


# ---------------------------------------------------------------------------
# 4. Retry matrix additions: 401/403 terminal, Retry-After honored
# ---------------------------------------------------------------------------


class TestTerminalAuthErrors:
    @pytest.mark.parametrize("status", [401, 403])
    def test_401_403_terminal_loud_never_retry(
        self,
        tmp_path: Path,
        session: RecordingSession,
        status: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The engine must not probe Earthdata hosts with a HEAD.
        monkeypatch.setattr(transport, "_is_earthdata_host", lambda _url: True)
        url = "https://example.test/denied.tif"
        session.script_response("GET", url, [FakeResponse(status=status)])
        plan = TileSet(
            allowed_hosts=("example.test",),
            tiles=(
                Tile(
                    url=url,
                    cache_path=Path("denied.tif"),
                    min_bytes=1024,
                    ranged=False,
                ),
            ),
        )
        with pytest.raises(transport.DemAuthProviderError) as excinfo:
            fetch_plan(plan, tmp_path)
        assert excinfo.value.status == status
        gets = [c for c in session.calls if c[0] == "GET"]
        assert len(gets) == 1
        assert not list(tmp_path.glob("*.part"))

    def test_retry_after_header_respected(
        self,
        tmp_path: Path,
        session: RecordingSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(transport.time, "sleep", sleeps.append)
        url = "https://example.test/slow.tif"
        session.script_response("HEAD", url, [FakeResponse(status=404)])
        payload = _payload(1 << 20)
        session.script_response(
            "GET",
            url,
            [
                FakeResponse(status=429, headers={"Retry-After": "3"}),
                FakeResponse(status=200, body=payload),
            ],
        )
        plan = TileSet(
            allowed_hosts=("example.test",),
            tiles=(Tile(url=url, cache_path=Path("slow.tif"), min_bytes=1024),),
        )
        fetch_plan(plan, tmp_path)
        assert any(s >= 3.0 for s in sleeps)


# ---------------------------------------------------------------------------
# 5. Ranged exclusion for Earthdata hosts
# ---------------------------------------------------------------------------


class TestEarthdataRangedExclusion:
    def test_ranged_excluded_for_earthdata_host_whole_file_stream(
        self,
        tmp_path: Path,
        session: RecordingSession,
    ) -> None:
        body = _payload(4096)
        url = "https://data.lpdaac.earthdatacloud.nasa.gov/N00E006.dem.zip"

        class _Creds(CredentialProvider):
            def headers_for(self, url: str) -> dict[str, str]:
                return {}

        session.script_response(
            "GET", url, [FakeResponse(status=200, body=body)]
        )
        # A HEAD that would advertise ranges must never be issued.
        plan = TileSet(
            allowed_hosts=("data.lpdaac.earthdatacloud.nasa.gov",),
            tiles=(
                Tile(
                    url=url,
                    cache_path=Path("n00e006.dem.zip"),
                    min_bytes=16,
                    ranged=False,
                ),
            ),
            _credentials=_Creds(),
        )
        fetch_plan(plan, tmp_path)
        heads = [c for c in session.calls if c[0] == "HEAD"]
        assert heads == []
        gets = [c for c in session.calls if c[0] == "GET"]
        assert len(gets) == 1
        assert (tmp_path / "n00e006.dem.zip").read_bytes() == body

    def test_ranged_mode_used_for_plain_hosts(self, tmp_path: Path, session: RecordingSession) -> None:
        chunk = transport.CHUNK_SIZE_BYTES
        body = _payload(chunk + 10)
        url = "https://example.test/ranged-on.tif"
        session.script_response(
            "HEAD",
            url,
            [FakeResponse(status=200, headers={"Content-Length": str(len(body)), "Accept-Ranges": "bytes"})],
        )
        total_chunks = (len(body) + chunk - 1) // chunk
        for index in range(total_chunks):
            start = index * chunk
            end = min(start + chunk, len(body)) - 1
            session.script_response(
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
        tile = Tile(
            url=url,
            cache_path=Path("ranged-on.tif"),
            min_bytes=16,
            ranged=True,
        )
        plan = TileSet(allowed_hosts=("example.test",), tiles=(tile,))
        fetch_plan(plan, tmp_path, max_workers=1)
        assert any(c[0] == "HEAD" for c in session.calls)
        get_headers = [c[2] for c in session.calls if c[0] == "GET"]
        assert any("range" in h for h in get_headers)


# ---------------------------------------------------------------------------
# 7. FTP branch
# ---------------------------------------------------------------------------


class TestFtpBranch:
    def test_ftp_fetch_via_urllib_writes_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        payload = _payload(2048)

        def fake_urlopen(url: str, timeout: float) -> object:
            assert url.startswith("ftp://")
            return io.BytesIO(payload)

        monkeypatch.setattr(transport.urllib.request, "urlopen", fake_urlopen)
        plan = Artifact(
            allowed_hosts=("ftp.eorc.jaxa.jp",),
            scheme="ftp",
            url="ftp://ftp.eorc.jaxa.jp/pub/AW3D30/x.zip",
            min_total_bytes=1024,
        )
        fetch_plan(plan, tmp_path)
        published = [p for p in tmp_path.iterdir() if p.name == "x.zip"]
        assert published
        assert published[0].read_bytes() == payload

    def test_ftp_without_content_length_enforces_floor_post_hoc(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        payload = b"too-small"

        def fake_urlopen(url: str, timeout: float) -> object:
            return io.BytesIO(payload)

        monkeypatch.setattr(transport.urllib.request, "urlopen", fake_urlopen)
        plan = Artifact(
            allowed_hosts=("ftp.eorc.jaxa.jp",),
            scheme="ftp",
            url="ftp://ftp.eorc.jaxa.jp/pub/AW3D30/small.zip",
            min_total_bytes=1024,
        )
        with pytest.raises(Exception, match="small|floor|bytes"):
            fetch_plan(plan, tmp_path)

    def test_ftp_failure_raises_loud(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(url: str, timeout: float) -> object:
            raise URLError("ftp down")

        monkeypatch.setattr(transport.urllib.request, "urlopen", fake_urlopen)
        plan = Artifact(
            allowed_hosts=("ftp.eorc.jaxa.jp",),
            scheme="ftp",
            url="ftp://ftp.eorc.jaxa.jp/pub/AW3D30/dead.zip",
        )
        with pytest.raises(Exception, match="ftp|FTP|dead"):
            fetch_plan(plan, tmp_path)


# ---------------------------------------------------------------------------
# 8. Zip expansion: containment + CRC + selective extraction
# ---------------------------------------------------------------------------


class TestZipExpansion:
    def _make_zip(self, path: Path, entries: dict[str, bytes]) -> None:
        with zipfile.ZipFile(path, "w") as zf:
            for name, data in entries.items():
                zf.writestr(name, data)

    def test_zip_slip_member_rejected_named_test(
        self,
        tmp_path: Path,
    ) -> None:
        staging = tmp_path / "staging"
        staging.mkdir(parents=True)
        archive = tmp_path / "hostile.zip"
        # Build a zip whose member name escapes the staging dir.
        hostile_name = "../escape.tif"
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w") as zf:
            zf.writestr("ok.tif", b"data")
        # Rewrite central directory entry names by hand via ZipInfo filename.
        with zipfile.ZipFile(raw, "a") as zf:
            info = zipfile.ZipInfo(hostile_name)
            zf.writestr(info, b"evil")
        archive.write_bytes(raw.getvalue())
        from faninsar.processing.geometry.dem_transport import (
            extract_zip_members,
            safe_member_target,
        )

        with pytest.raises(ValueError, match="traversal|contain|zip-slip|\\.\\.") as excinfo:
            safe_member_target(staging, hostile_name)
        with pytest.raises(ValueError):
            extract_zip_members(
                archive,
                staging,
                members=None,
                member_pattern="*.tif",
            )

    def test_crc_corruption_detected(
        self,
        tmp_path: Path,
    ) -> None:
        import binascii
        import struct
        import zipfile

        from faninsar.processing.geometry.dem_transport import extract_zip_members

        payload = b"original-bytes-here"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("ALPSMLC30_N000E006_DSM.tif", payload)
        blob = bytearray(buf.getvalue())
        good_crc = binascii.crc32(payload) & 0xFFFFFFFF
        bad_crc = (good_crc ^ 0xFFFFFFFF) & 0xFFFFFFFF
        needle = struct.pack("<I", good_crc)
        pos = blob.find(needle)
        while pos != -1:
            blob[pos : pos + 4] = struct.pack("<I", bad_crc)
            pos = blob.find(needle, pos + 4)
        corrupt = tmp_path / "corrupt2.zip"
        corrupt.write_bytes(bytes(blob))
        staging = tmp_path / "stage-corrupt"
        staging.mkdir(parents=True)
        with pytest.raises(Exception):  # BadZipFile CRC check failed
            extract_zip_members(
                corrupt, staging, members=None, member_pattern="*.tif"
            )

    def test_only_matching_members_extracted(
        self,
        tmp_path: Path,
    ) -> None:
        from faninsar.processing.geometry.dem_transport import extract_zip_members

        archive = tmp_path / "aw3d30.zip"
        self._make_zip(
            archive,
            {
                "ALPSMLC30_N000E006_DSM.tif": b"dsm-data",
                "ALPSMLC30_N000E006_MSK.tif": b"msk-data",
            },
        )
        staging = tmp_path / "stage"
        extracted = extract_zip_members(
            archive, staging, members=None, member_pattern="*_DSM.tif"
        )
        assert [p.name for p in extracted] == ["ALPSMLC30_N000E006_DSM.tif"]
        assert extracted[0].read_bytes() == b"dsm-data"


# ---------------------------------------------------------------------------
# 9. Atomic publish naming
# ---------------------------------------------------------------------------


class TestPartNaming:
    def test_part_name_contains_pid_and_uuid(self, tmp_path: Path) -> None:
        part = transport.part_path(tmp_path / "tile.tif")
        assert part.parent == tmp_path
        assert part.name.startswith("tile.tif.")
        assert part.name.endswith(".part")
        core = part.name[len("tile.tif.") : -len(".part")]
        pid_text, sep, uuid_text = core.partition("-")
        assert sep == "-"
        assert pid_text.isdigit()
        assert len(uuid_text) >= 32


# ---------------------------------------------------------------------------
# 10. Credential hygiene (caplog)
# ---------------------------------------------------------------------------


class TestCredentialHygiene:
    def test_token_absent_from_logs_errors_and_dumps(
        self,
        tmp_path: Path,
        session: RecordingSession,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(transport, "_netrc_auth", lambda _host: ("user", _TOKEN))
        url = "https://auth.example.test/granule.tif"

        class _Creds(CredentialProvider):
            def headers_for(self, url: str) -> dict[str, str]:
                return {"Authorization": f"Bearer {_TOKEN}"}

        session.script_response(
            "GET", url, [FakeResponse(status=503), FakeResponse(status=503)]
        )
        plan = TileSet(
            allowed_hosts=("auth.example.test",),
            tiles=(
                Tile(
                    url=url,
                    cache_path=Path("g.tif"),
                    min_bytes=1024,
                    ranged=False,
                ),
            ),
        )
        with caplog.at_level(logging.DEBUG, logger="faninsar.processing.geometry.dem_transport"):
            with pytest.raises(Exception) as excinfo:
                fetch_plan(plan, tmp_path)
        rendered_logs = caplog.text
        assert _TOKEN not in rendered_logs
        assert _TOKEN not in repr(excinfo.value)
        assert _TOKEN not in str(excinfo.value)
        for record in caplog.records:
            message = record.getMessage()
            assert _TOKEN not in message

    def test_sas_query_params_scrubbed_from_log_lines(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        signed_url = (
            "https://pc.blob.core.windows.net/tile.tif"
            "?sig=TOPSECRET&se=2026-01-01T00:00:00Z&sp=r"
        )
        scrubbed = transport.redact_url(signed_url)
        assert "TOPSECRET" not in scrubbed
        assert "sig=REDACTED" in scrubbed or "sig=[REDACTED]" in scrubbed


# ---------------------------------------------------------------------------
# 11. BLOCKER-0030-B2: multi-tile fan-out consumes every part
# ---------------------------------------------------------------------------


class TestMultiTileParts:
    def test_all_parts_executed_into_mosaic_inputs(
        self,
        tmp_path: Path,
        session: RecordingSession,
    ) -> None:
        """A Tile plan carrying a ``_MultiTileTile`` fetches every sub-tile."""
        from faninsar.processing.geometry.dem_sources import _MultiTileTile

        body = _payload(4096)
        urls = [f"https://example.test/07_40/{i}.tif" for i in (1, 2, 3, 4)]
        for url in urls:
            session.script_response("GET", url, [FakeResponse(status=200, body=body)])
        parts = tuple(
            Tile(
                url=url,
                cache_path=Path(f"quad/part-{index}.tif"),
                min_bytes=16,
                ranged=False,
            )
            for index, url in enumerate(urls, start=1)
        )
        plan = TileSet(
            allowed_hosts=("example.test",),
            tiles=(_MultiTileTile(*parts),),
        )
        executed = fetch_plan(plan, tmp_path)
        expected_names = {f"part-{index}.tif" for index in range(1, 5)}
        assert {path.name for path in executed} == expected_names
        for index in range(1, 5):
            assert (tmp_path / "quad" / f"part-{index}.tif").read_bytes() == body
        gets = [url for method, url, _headers in session.calls if method == "GET"]
        assert set(gets) == set(urls)
        assert len(gets) == 4


# ---------------------------------------------------------------------------
# 12. BLOCKER-0030-B1: multi-artifact plans execute every block
# ---------------------------------------------------------------------------


class TestMultiArtifactExecution:
    def test_multi_artifact_plan_executes_every_block(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """FtpZipSource multi-block plans fetch each zip sequentially."""
        import zipfile

        from faninsar.processing.geometry.dem_sources import parse_selection

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("ALPSMLC30_N000E006_DSM.tif", b"dsm-1")
            zf.writestr("padding.bin", b"\0" * (1 << 20))
        archive = buf.getvalue()
        assert len(archive) >= 1 << 20  # min_total_bytes floor

        calls: list[str] = []

        def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
            calls.append(url)
            return io.BytesIO(archive)

        monkeypatch.setattr(transport.urllib.request, "urlopen", fake_urlopen)
        source = parse_selection("alos-dem:jaxa-ftp")
        plan = source.plan((1.0, 3.0, 7.0, 7.0))
        assert len(plan.artifacts) >= 2
        executed = fetch_plan(plan, tmp_path)
        assert len(calls) == len(plan.artifacts)
        assert len(executed) == len(plan.artifacts)
        for path in executed:
            assert path.is_file()
            assert path.read_bytes() == archive


# ---------------------------------------------------------------------------
# 13. ADVISORY-0030-A2: expected decompressed size enforced by the engine
# ---------------------------------------------------------------------------


class TestExpectedDecompressedSize:
    def test_decompressed_size_enforced_after_download(
        self,
        tmp_path: Path,
        session: RecordingSession,
    ) -> None:
        import gzip

        raw = _payload(16 << 10)
        body = gzip.compress(raw)
        url = "https://example.test/skadi/N34/N34E094.hgt.gz"
        session.script_response("GET", url, [FakeResponse(status=200, body=body)])
        tile = Tile(
            url=url,
            cache_path=Path("skadi/N34/N34E094.hgt.gz"),
            min_bytes=16,
            ranged=False,
            expected_decompressed_bytes=len(raw),
        )
        plan = TileSet(allowed_hosts=("example.test",), tiles=(tile,))
        executed = fetch_plan(plan, tmp_path)
        target = tmp_path / "skadi" / "N34" / "N34E094.hgt.gz"
        assert executed == [target]
        assert target.read_bytes() == body

    def test_decompressed_size_mismatch_refuses_publish(
        self,
        tmp_path: Path,
        session: RecordingSession,
    ) -> None:
        import gzip

        raw = _payload(16 << 10)
        body = gzip.compress(raw)
        url = "https://example.test/skadi/N34/N34E094.hgt.gz"
        session.script_response("GET", url, [FakeResponse(status=200, body=body)])
        tile = Tile(
            url=url,
            cache_path=Path("skadi/N34/N34E094.hgt.gz"),
            min_bytes=16,
            ranged=False,
            expected_decompressed_bytes=len(raw) + 1,
        )
        plan = TileSet(allowed_hosts=("example.test",), tiles=(tile,))
        with pytest.raises(
            transport.InvalidProcessingStateError, match="decompressed"
        ):
            fetch_plan(plan, tmp_path)
        assert not (tmp_path / "skadi" / "N34" / "N34E094.hgt.gz").is_file()
        assert not list(tmp_path.rglob("*.part"))


# ---------------------------------------------------------------------------
# 14. ADVISORY-0030-A3: ocean_404_skip honored by the engine
# ---------------------------------------------------------------------------


class TestOcean404Skip:
    def test_ocean_404_skip_becomes_skip_with_log(
        self,
        tmp_path: Path,
        session: RecordingSession,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        url = "https://example.test/skadi/N34/N34E094.hgt.gz"
        session.script_response("GET", url, [FakeResponse(status=404)])
        tile = Tile(
            url=url,
            cache_path=Path("skadi/N34/N34E094.hgt.gz"),
            min_bytes=1024,
            ranged=False,
            ocean_404_skip=True,
        )
        plan = TileSet(allowed_hosts=("example.test",), tiles=(tile,))
        with caplog.at_level(
            logging.INFO, logger="faninsar.processing.geometry.dem_transport"
        ):
            executed = fetch_plan(plan, tmp_path)
        assert executed == []
        assert not (tmp_path / "skadi" / "N34" / "N34E094.hgt.gz").is_file()
        assert any("ocean" in record.message.lower() for record in caplog.records)

    def test_ocean_404_skip_false_still_hard_fails(
        self,
        tmp_path: Path,
        session: RecordingSession,
    ) -> None:
        url = "https://example.test/skadi/N34/N34E094.hgt.gz"
        session.script_response("GET", url, [FakeResponse(status=404)])
        tile = Tile(
            url=url,
            cache_path=Path("skadi/N34/N34E094.hgt.gz"),
            min_bytes=1024,
            ranged=False,
            ocean_404_skip=False,
        )
        plan = TileSet(allowed_hosts=("example.test",), tiles=(tile,))
        with pytest.raises(transport.InvalidProcessingStateError, match="404"):
            fetch_plan(plan, tmp_path)
